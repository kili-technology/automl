# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from kiliautoml.utils.constants import AUTOML_REPO_ROOT
from kiliautoml.utils.helpers import kili_print
from kiliautoml.utils.label_errors.yolo_metrics import ap_per_class, box_iou

YOLO_DIR = str(Path(str(AUTOML_REPO_ROOT), "kiliautoml", "utils", "ultralytics", "yolov5"))
if YOLO_DIR not in sys.path:
    sys.path.append(YOLO_DIR)

print(sys.path)

from kiliautoml.utils.ultralytics.yolov5.models.common import DetectMultiBackend  # noqa
from kiliautoml.utils.ultralytics.yolov5.utils.dataloaders import (  # noqa
    create_dataloader,
)
from kiliautoml.utils.ultralytics.yolov5.utils.general import (  # noqa
    LOGGER,
    check_dataset,
    check_img_size,
    colorstr,
    non_max_suppression,
    scale_coords,
    xywh2xyxy,
)
from kiliautoml.utils.ultralytics.yolov5.utils.torch_utils import (  # noqa
    select_device,
    time_sync,
)

os.environ["OMP_NUM_THREADS"] = "1"


def process_batch(detections, labels_gt, iou_thresholds):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels_gt (Array[M, 5]), class, x1, y1, x2, y2
            "ground truth"
    Returns:
        correct bool (Array[N, 10]), for 10 IoU levels (10=iou_thresholds.shape[0])
    """
    correct = np.zeros((detections.shape[0], iou_thresholds.shape[0])).astype(bool)
    iou = box_iou(labels_gt[:, 1:], detections[:, :4])
    correct_class = labels_gt[:, 0:1] == detections[:, 5]
    for i in range(len(iou_thresholds)):
        x = torch.where(
            (iou >= iou_thresholds[i]) & correct_class
        )  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = (
                torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            )  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iou_thresholds.device)


@torch.no_grad()
def run(
    data,
    weights=None,  # model.pt path(s)
    batch_size=32,  # batch size
    imgsz=640,  # inference size (pixels)
    conf_thres=0.001,  # confidence threshold
    iou_thres=0.65,  # NMS IoU threshold
    task="test",  # train, val, test, speed or study
    workers=8,  # max dataloader workers (per RANK in DDP mode)
    single_cls=False,  # treat as single-class dataset
    augment=False,  # augmented inference
    save_hybrid=False,  # save label+prediction hybrid results to *.txt
    half=True,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    model=None,
    dataloader=None,
    compute_loss=None,
):

    device = select_device("", batch_size=batch_size)

    # Load model
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # type:ignore # check image size
    half = model.fp16  # FP16 supported on limited backends with CUDA
    if engine:
        batch_size = model.batch_size
    else:
        device = model.device
        if not (pt or jit):
            batch_size = 1  # export.py models default to batch-size 1
            LOGGER.info("Forcing --batch-size 1 for non-PyTorch models")

    # Data
    data = check_dataset(data)  # check

    # Configure
    model.eval()
    cuda = device.type != "cpu"
    iou_thresholds = torch.linspace(0.5, 0.95, 10).to(str(device))  # iou vector for mAP@0.5:0.95
    niou = iou_thresholds.numel()

    image_names = []

    model.warmup(imgsz=(1, 3, imgsz, imgsz))  # warmup

    pad = 0.5
    task = task if task in ("train", "val", "test") else "val"  # path to train/val/test images
    dataloader = create_dataloader(
        data[task],
        imgsz,
        batch_size,
        stride,
        single_cls,
        pad=pad,
        rect=pt,  # type:ignore
        workers=workers,
        prefix=colorstr(f"{task}: "),
    )[0]

    image_with_labels_counter = 0
    names = {
        k: v
        for k, v in enumerate(
            model.names if hasattr(model, "names") else model.module.names  # type:ignore
        )
    }
    dt, map = [0.0, 0.0, 0.0], 0.0
    loss = torch.zeros(3, device=str(device))
    stats, ap = [], []

    # Predict by batch of images
    pbar = tqdm(dataloader, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")  # progress bar
    for _, (imgs_batche, targets, paths, shapes) in enumerate(pbar):
        t1 = time_sync()
        if cuda:
            imgs_batche = imgs_batche.to(device, non_blocking=True)
            targets = targets.to(device)
        imgs_batche = imgs_batche.half() if half else imgs_batche.float()  # uint8 to fp16/32
        imgs_batche /= 255  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = imgs_batche.shape  # batch size, channels, height, width
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        out, train_out = model(imgs_batche, augment=augment, val=True)  # inference, loss outputs
        dt[1] += time_sync() - t2

        # Loss
        if compute_loss:
            loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls

        # NMS
        targets[:, 2:] *= torch.tensor(
            (width, height, width, height), device=str(device)
        )  # to pixels
        lb = (
            [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []
        )  # for autolabelling
        t3 = time_sync()
        out = non_max_suppression(
            out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls
        )
        dt[2] += time_sync() - t3

        # Metrics for each image
        compute_metrics_by_images(
            single_cls,
            iou_thresholds,
            niou,
            image_names,
            image_with_labels_counter,
            stats,
            imgs_batche,
            targets,
            paths,
            shapes,
            out,
            device,
        )

    # Compute metrics
    metrics_per_images = np.array([])

    if len(stats):
        for stat in stats:
            stat = [np.array(x) for x in stat]
            if len(stat) and stat[0].any():
                ap = ap_per_class(*stat, plot=False, names=names)
                ap = ap.mean(1)  # AP@0.5:0.95
                map = ap.mean()
                metrics_per_images = np.append(metrics_per_images, map)
            else:
                metrics_per_images = np.append(metrics_per_images, 0)

    dataset_mean_map = metrics_per_images.mean()
    print("dataset mAP: ", dataset_mean_map)

    metrics_per_image_dict = {}

    for idx, map_value in enumerate(metrics_per_images):
        metrics_per_image_dict[image_names[idx]] = map_value

    return metrics_per_image_dict


def compute_metrics_by_images(
    single_cls,
    iou_thresholds,
    niou,
    image_names,
    image_with_labels_counter,
    stats,
    imgs_batche,
    targets,
    paths,
    shapes,
    out,
    device,
):
    for index_image, pred in tqdm(enumerate(out)):
        labels = targets[targets[:, 0] == index_image, 1:]
        n_labels, n_pred = labels.shape[0], pred.shape[0]  # number of labels, predictions
        target_class = labels[:, 0]
        image_path, image_shape = Path(paths[index_image]), shapes[index_image][0]
        image_with_labels_counter = image_with_labels_counter + 1
        correct = torch.zeros(n_pred, niou, dtype=torch.bool, device=device)  # init

        if n_pred == 0:
            if n_labels:
                image_names.append(image_path.stem)
                stats.append((correct, *torch.zeros((3, 0), device=device)))
            continue

        # Predictions
        if single_cls:
            pred[:, 5] = 0
        predn = pred.clone()
        scale_coords(
            imgs_batche[index_image].shape[1:],
            predn[:, :4],
            image_shape,
            shapes[index_image][1],
        )  # native-space pred

        # Evaluate
        if n_labels:
            tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
            scale_coords(
                imgs_batche[index_image].shape[1:], tbox, image_shape, shapes[index_image][1]
            )  # native-space labels
            labelsn = torch.cat((labels[:, 0:1], tbox), dim=1)  # type:ignore # native-space labels
            correct = process_batch(predn, labelsn, iou_thresholds)
            # check that we can take the output of the var correct to lower the granularity
            # down to the bbox level

        image_names.append(image_path.stem)
        stats.append((correct, pred[:, 4], pred[:, 5], target_class))  # (correct, conf, pcls, tcls)


def compute_per_image_map(
    json_path: str,
    data_yaml_path: str,
    weights_path: str,
    cv_fold: int,
):
    metric_per_image = run(data=data_yaml_path, weights=weights_path)

    found_errors_json = json.dumps(metric_per_image, sort_keys=True, indent=4)
    if found_errors_json is not None:
        json_file_path = os.path.join(json_path, f"error_labels_{cv_fold}.json")
        with open(json_file_path, "wb") as output_file:
            output_file.write(found_errors_json.encode("utf-8"))
            kili_print(f"Per-image metric for fold {cv_fold} written to: ", json_file_path)
    return metric_per_image
