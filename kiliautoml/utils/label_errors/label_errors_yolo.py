# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python path/to/val.py --weights yolov5s.pt --data coco128.yaml --img 640

Usage - formats:
    $ python path/to/val.py --weights yolov5s.pt                 # PyTorch
                                      yolov5s.torchscript        # TorchScript
                                      yolov5s.onnx           # ONNX Runtime or OpenCV DNN with --dnn
                                      yolov5s.xml                # OpenVINO
                                      yolov5s.engine             # TensorRT
                                      yolov5s.mlmodel            # CoreML (MacOS-only)
                                      yolov5s_saved_model        # TensorFlow SavedModel
                                      yolov5s.pb                 # TensorFlow GraphDef
                                      yolov5s.tflite             # TensorFlow Lite
                                      yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from kiliautoml.utils.helpers import kili_print
from kiliautoml.utils.label_errors.yolo_metrics import ap_per_class, box_iou

sys.path.append("kiliautoml/utils/ultralytics/yolov5/")

from kiliautoml.utils.ultralytics.yolov5.models.common import DetectMultiBackend  # noqa
from kiliautoml.utils.ultralytics.yolov5.utils.datasets import create_dataloader  # noqa
from kiliautoml.utils.ultralytics.yolov5.utils.general import (  # noqa
    LOGGER,
    check_dataset,
    check_img_size,
    check_yaml,
    colorstr,
    non_max_suppression,
    print_args,
    scale_coords,
    xywh2xyxy,
)
from kiliautoml.utils.ultralytics.yolov5.utils.torch_utils import (  # noqa
    select_device,
    time_sync,
)

FILE = Path(__file__).resolve()

os.environ["OMP_NUM_THREADS"] = "1"

default_path = Path("")


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
    correct = torch.zeros(
        detections.shape[0], iou_thresholds.shape[0], dtype=torch.bool, device=iou_thresholds.device
    )
    iou = box_iou(labels_gt[:, 1:], detections[:, :4])
    x = torch.where(
        (iou >= iou_thresholds[0]) & (labels_gt[:, 0:1] == detections[:, 5])
    )  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = (
            torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
        )  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iou_thresholds.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iou_thresholds
    return correct


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
    plots=False,
    compute_loss=None,
):

    device = select_device("", batch_size=batch_size)

    # Load model
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, pt, jit, onnx, engine = model.stride, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # type:ignore # check image size
    half &= (
        pt or jit or onnx or engine
    ) and device.type != "cpu"  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()
    elif engine:
        batch_size = model.batch_size
    else:
        half = False
        batch_size = 1  # export.py models default to batch-size 1
        device = torch.device("cpu")
        LOGGER.info(
            f"Forcing --batch-size 1 square inference shape(1,3,{imgsz},{imgsz}) for            "
            " non-PyTorch backends"
        )

    # Data
    data = check_dataset(data)  # check

    # Configure
    model.eval()
    iou_thresholds = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iou_thresholds.numel()

    image_names = []

    model.warmup(imgsz=(1, 3, imgsz, imgsz), half=bool(half))  # warmup

    pad = 0.0 if task == "speed" else 0.5
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
    loss = torch.zeros(3, device=device)
    stats, ap = [], []

    # Predict by batch of images
    pbar = tqdm(dataloader, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")  # progress bar
    for _, (imgs_batche, targets, paths, shapes) in enumerate(pbar):
        t1 = time_sync()
        if pt or jit or engine:
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
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
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
        )  # (correct, conf, pcls, tcls)

    # Compute metrics
    metrics_per_images = np.array([])

    if len(stats):
        for stat in stats:
            stat = [np.array(x) for x in stat]
            if len(stat) and stat[0].any():
                ap = ap_per_class(*stat, plot=plots, names=names)
                ap = ap.mean(1)  # AP@0.5:0.95
                map = ap.mean()
                metrics_per_images = np.append(metrics_per_images, map)
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
):
    for index_image, pred in tqdm(enumerate(out)):
        labels = targets[targets[:, 0] == index_image, 1:]
        n_labels = len(labels)
        target_class = labels[:, 0].tolist() if n_labels else []
        image_path, image_shape = Path(paths[index_image]), shapes[index_image][0]
        image_with_labels_counter = image_with_labels_counter + 1

        if len(pred) == 0:
            if n_labels:
                stats.append(
                    (
                        torch.zeros(0, niou, dtype=torch.bool),
                        torch.Tensor(),
                        torch.Tensor(),
                        target_class,
                    )
                )
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
            labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
            correct = process_batch(predn, labelsn, iou_thresholds)

        else:
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
        image_names.append(image_path.stem)
        stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), target_class))


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./data/coco128.yaml", help="dataset.yaml path")
    parser.add_argument(
        "--weights", nargs="+", type=str, default="./yolov5s.pt", help="model.pt path(s)"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument(
        "--imgsz", "--img", "--img-size", type=int, default=640, help="inference size (pixels)"
    )
    parser.add_argument("--conf-thres", type=float, default=0.001, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--task", default="val", help="train, val, test, speed or study")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument(
        "--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)"
    )
    parser.add_argument("--single-cls", action="store_true", help="treat as single-class dataset")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--verbose", action="store_true", help="report mAP by class")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument(
        "--save-hybrid", action="store_true", help="save label+prediction hybrid results to *.txt"
    )
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument(
        "--exist-ok", action="store_true", help="existing project/name ok, do not increment"
    )
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--json-path")
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_txt |= opt.save_hybrid
    print_args(FILE.stem, opt)
    return opt


def compute_per_image_map(
    json_path: str,
    data_yaml_path: str,
    weights_path: str,
):
    metric_per_image = run(data=data_yaml_path, weights=weights_path)

    found_errors_json = json.dumps(metric_per_image, sort_keys=True, indent=4)
    if found_errors_json is not None:
        json_file_path = os.path.join(json_path, "error_labels.json")
        with open(json_file_path, "wb") as output_file:
            output_file.write(found_errors_json.encode("utf-8"))
            kili_print("Per-image metric written to: ", json_file_path)
    return metric_per_image


def main():
    """
    if opt.task in ("train", "val", "test"):
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(
                f"WARNING: confidence threshold {opt.conf_thres} >> 0.001 will produce \
                invalid mAP values."
            )
        metric_per_image = run(**vars(opt))

        found_errors_json = json.dumps(metric_per_image, sort_keys=True, indent=4)
        if found_errors_json is not None:
            json_path = os.path.join(opt.json_path, "error_labels.json")
            with open(json_path, "wb") as output_file:
                output_file.write(found_errors_json.encode("utf-8"))
                kili_print("Per-image metric written to: ", json_path)
    """

    path_result_train = "/Volumes/GoogleDrive-109043737286691549671/My Drive/kili/automl/ckzdzhh260ec00mub7gqjfetz/JOB_0/ultralytics/model/pytorch/2022-05-17_13_57_12/Plastic detection in river/"  # noqa

    compute_per_image_map(
        "./",
        path_result_train + "kili.yaml",
        path_result_train + "exp/weights/best.pt",
    )


if __name__ == "__main__":
    # opt = parse_opt()
    main()
