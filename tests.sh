#!/bin/bash
git submodule update --init && pip install -r requirements.txt --quiet && pip install -r utils/ultralytics/yolov5/requirements.txt --quiet

echo "TRAINING"
python train.py        --project-id cl0wihlop3rwc0mtj9np28ti2  --max-assets 50 --json-args '{"epochs": 5}'

echo "PREDICT"
python predict.py      --project-id cl0wihlop3rwc0mtj9np28ti2  --max-assets 5 --label-types DEFAULT,REVIEW --verbose 1

echo "PRIORITIZE"
python prioritize.py   --project-id cl0wihlop3rwc0mtj9np28ti2  --max-assets 20
