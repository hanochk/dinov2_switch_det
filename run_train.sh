#!/bin/bash
echo "start runbatch.sh" >> ./train_dinov2.log
python -u main.py --epoch 20 --batch-size 64 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier --lr 0.005
python -u main.py --epoch 20 --batch-size 64 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier --lr 0.001
python -u main.py --epoch 20 --batch-size 64 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier --lr 0.005 --linear-lr
python -u main.py --epoch 20 --batch-size 64 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier --lr 0.001 --linear-lr
python -u main.py --epoch 20 --batch-size 64 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier --lr 0.005 --dropout 0.2
python -u main.py --epoch 20 --batch-size 64 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier --lr 0.001 --dropout 0.2
python -u main.py --epoch 20 --batch-size 64 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier --lr 0.005 --linear-lr --dropout 0.2
python -u main.py --epoch 20 --batch-size 64 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier --lr 0.001 --linear-lr --dropout 0.2
python -u main.py --epoch 20 --batch-size 64 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier --lr 0.01
python -u main.py --epoch 20 --batch-size 64 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier --lr 0.01 --linear-lr
python -u main.py --epoch 20 --batch-size 64 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier --lr 0.005
python -u main.py --epoch 20 --batch-size 64 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier --lr 0.001
python -u main.py --epoch 20 --batch-size 64 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier --lr 0.005 --linear-lr
python -u main.py --epoch 20 --batch-size 64 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier --lr 0.001 --linear-lr
python -u main.py --epoch 20 --batch-size 64 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier --lr 0.005 --dropout 0.2
python -u main.py --epoch 20 --batch-size 64 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier --lr 0.001 --dropout 0.2
python -u main.py --epoch 20 --batch-size 64 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier --lr 0.005 --linear-lr --dropout 0.2
python -u main.py --epoch 20 --batch-size 64 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier --lr 0.001 --linear-lr --dropout 0.2
python -u main.py --epoch 20 --batch-size 64 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier --lr 0.01
python -u main.py --epoch 20 --batch-size 64 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier --lr 0.01 --linear-lr

#python -u ./yolov7/train.py --workers 8 --device 0 --batch-size 16 --data data/tir_od.yaml --img 640 640 --weights ./yolov7/yolov7-tiny.pt --cfg cfg/training/yolov7-tiny.yaml --name yolov7 --hyp data/hyp.tir_od.tiny.yaml --adam --norm-type single_image_percentile_0_1 --input-channels 1 --linear-lr --nosave
