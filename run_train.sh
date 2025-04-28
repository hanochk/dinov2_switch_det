#!/bin/bash
echo "start runbatch.sh" >> ./train_dinov2.log

python -u main.py --epoch 50 --batch-size 64 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier --lr 0.0005 --linear-lr --global-crops-size 448 --dropout 0.2 --var-hidden-dim
python -u main.py --epoch 50 --batch-size 64 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier --lr 0.0005 --linear-lr --global-crops-size 448 --var-hidden-dim
python -u main.py --epoch 50 --batch-size 64 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier --lr 0.0005 --linear-lr --global-crops-size 518
python -u main.py --epoch 50 --batch-size 64 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier --lr 0.0005 --linear-lr --global-crops-size 518 --dropout 0.2
python -u main.py --epoch 50 --batch-size 64 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier --lr 0.0005 --linear-lr --global-crops-size 518 --var-hidden-dim

#python -u main.py --epoch 20 --batch-size 64 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier --lr 0.0005 --linear-lr --global-crops-size 448
#python -u main.py --epoch 50 --batch-size 64 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier --lr 0.0005 --linear-lr --global-crops-size 448 --dropout 0.2
#python -u main.py --epoch 50 --batch-size 64 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier --lr 0.0005 --linear-lr
#python -u main.py --epoch 50 --batch-size 64 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier --lr 0.0005 --linear-lr --dropout 0.2
#python -u main.py --epoch 50 --batch-size 64 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier --lr 0.0005 --linear-lr --global-crops-size 448 --dropout 0.2 --var-hidden-dim
#python -u main.py --epoch 50 --batch-size 64 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier --lr 0.0005 --linear-lr --var-hidden-dim
#python -u main.py --epoch 50 --batch-size 64 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier --lr 0.0005 --linear-lr --dropout 0.2 --var-hidden-dim

#python -u main.py --epoch 50 --batch-size 64 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier --lr 0.0005
#python -u main.py --epoch 50 --batch-size 64 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier --lr 0.0005 --dropout 0.2
#python -u main.py --epoch 20 --batch-size 64 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier --lr 0.0005 --linear-lr
#python -u main.py --epoch 20 --batch-size 64 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier --lr 0.0005
#python -u main.py --epoch 20 --batch-size 64 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier --lr 0.0005 --dropout 0.2
#python -u main.py --epoch 20 --batch-size 64 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier --lr 0.0005 --linear-lr --dropout 0.2
#python -u main.py --epoch 20 --batch-size 64 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier --lr 0.0005
#python -u main.py --epoch 20 --batch-size 64 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier --lr 0.0005 --linear-lr
#python -u main.py --epoch 20 --batch-size 64 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier --lr 0.0005
#python -u main.py --epoch 20 --batch-size 64 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier --lr 0.0005
#python -u main.py --epoch 20 --batch-size 64 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier --lr 0.0005 --linear-lr
#python -u main.py --epoch 20 --batch-size 64 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier --lr 0.0005 --linear-lr
#python -u main.py --epoch 20 --batch-size 64 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier --lr 0.0005 --dropout 0.2
#python -u main.py --epoch 20 --batch-size 64 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier --lr 0.0005 --dropout 0.2
#python -u main.py --epoch 20 --batch-size 64 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier --lr 0.0005 --linear-lr --dropout 0.2
#python -u main.py --epoch 20 --batch-size 64 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier --lr 0.0005 --linear-lr --dropout 0.2

#python -u ./yolov7/train.py --workers 8 --device 0 --batch-size 16 --data data/tir_od.yaml --img 640 640 --weights ./yolov7/yolov7-tiny.pt --cfg cfg/training/yolov7-tiny.yaml --name yolov7 --hyp data/hyp.tir_od.tiny.yaml --adam --norm-type single_image_percentile_0_1 --input-channels 1 --linear-lr --nosave
