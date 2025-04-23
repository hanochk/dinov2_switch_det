import os.path as osp

import torch
import torch.nn as nn
import torch.optim as optim
import os
from data_loader_pin_space import create_dataloader
from lib.evaluation import accuracy_calc
from lib.models import load_backbone, load_classifier
from lib.datasets import get_dataloader
from tqdm import tqdm
from lib.utils import AverageMeter, cal_acc, increment_path
from config import get_args
import cv2
import numpy as np
from pathlib import Path
import math
import torch.optim.lr_scheduler as lr_scheduler
import yaml

def main(args):
    # base_save_path =
    # HK TODO add Wdecay, dropout, cosine anealing, or lr decay
    debug = True
    flavour = 'base'

    # train_set_path = '/mnt/Data/hanoch/back_switch/sw_v0.7_bsw_v2.8/train'
    # val_set_path = '/mnt/Data/hanoch/back_switch/backswitch_detection/validation'
    # test_set_path = '/mnt/Data/hanoch/back_switch/sw_v0.7_bsw_v2.8/test'
    train_set_path = '/mnt/Data/hanoch/back_switch/backswitch_detection/train'
    val_set_path = '/mnt/Data/hanoch/back_switch/backswitch_detection/validation'
    test_set_path = '/mnt/Data/hanoch/back_switch/backswitch_detection/test'

    backbone = load_backbone(model=args.backbone_model, flavour=flavour)
    if flavour == 'small':
        fc_dim = 384
        global_crops_size = 224
    elif flavour == 'base':
        fc_dim = 768
        global_crops_size = 224
    elif flavour == 'large':
        fc_dim = 1024
        global_crops_size = 518
    else:
        raise ValueError('Unknown backbone model')


    ce_loss = nn.CrossEntropyLoss()
    if args.task !='test':
        if args.output_folder == '':
            args.save_dir = increment_path(Path(args.project) / args.name,
                                           exist_ok=args.exist_ok | args.evolve)  # increment run
        else:
            args.save_dir = increment_path(os.path.join(args.output_folder, Path(args.project), args.name),
                                           exist_ok=False | False)

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        results_file = os.path.join(args.save_dir , 'results.txt')

        with open(os.path.join(args.save_dir ,  'opt.yaml'), 'w') as f:
            yaml.dump(vars(args), f, sort_keys=False)

        model_save_root = os.path.join(args.save_dir,
                                       args.cls_model_name)  # osp.join(args.output_folder, args.cls_model_name)
        debug_save_path = os.path.join(args.save_dir, 'images')
        if not os.path.exists(debug_save_path):
            os.makedirs(debug_save_path)

        classifier = load_classifier(fc_dim, n_cls=2, dropout=args.dropout) # Flower is 17
        optimizer = optim.AdamW(classifier.parameters(), lr=args.lr)

        scheduler = lr_scheduler_setup(optimizer, linear_lr=args.linear_lr, start_epoch = 0)

          # do not move

            # lf = one_cycle(1, lrf, args.epoch)  # cosine 1->hyp['lrf']


        if 1:
             # 224
            train_dataloader = create_dataloader(train_set_path, crop_size=global_crops_size,
                                                 num_workers=args.num_workers,
                                                 batch_size=64, shuffle=True)
            train_dataloader.dataset.df.to_csv(os.path.join(args.save_dir, 'training_set.csv'))
            print("Train Class 0, 1 distribution",
                   np.unique([x[1] for x in train_dataloader.dataset.samples], return_counts=True))

            valid_dataloader = create_dataloader(val_set_path, crop_size=global_crops_size, num_workers=args.num_workers, batch_size=64, shuffle=False)
            valid_dataloader.dataset.df.to_csv(os.path.join(args.save_dir, 'val_set.csv'))
            print("Val Class 0, 1 distribution",
               np.unique([x[1] for x in valid_dataloader.dataset.samples], return_counts=True))

        # train_dataloader = CustomImageDataset(train_set_path, transform=train_transform, crop_size=global_crops_size)
            # train_dataloader = DataLoader(train_dataloader, batch_size=batch_size, shuffle=shuffle, num_workers=args.num_workers)
        else:
            train_dataloader = get_dataloader(batch_size=64, shuffle=True, num_workers=4, mode="train", split=args.split, model=args.backbone_model)
            valid_dataloader = get_dataloader(batch_size=64, shuffle=False, num_workers=4, mode="valid", split=args.split, model=args.backbone_model)

        best_acc = 0
        for epoch in range(args.epoch):
            train_losses = AverageMeter()
            train_acces = AverageMeter()
            for ix , (imgs, labels, unnorm_img) in enumerate(tqdm(train_dataloader)):
                # if debug:
                #     for ii, img in enumerate(imgs):
                #         cv2.imwrite(os.path.join(debug_save_path, str(ii+ix*args.batch_size)+ '.png'), img.detach().cpu().permute(1, 2, 0).numpy()*255)

                imgs = imgs.cuda()
                labels = labels.cuda()
                bs = imgs.shape[0]

                with torch.no_grad():
                    embeddings = backbone(imgs)
                outputs = classifier(embeddings)
                loss = ce_loss(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.update(loss.item(), bs)
                acc, err_dict = cal_acc(outputs, labels)
                train_acces.update(acc, bs)
                last_lr = scheduler.get_last_lr()

            scheduler.step() # lr step
            print(last_lr)

            valid_losses = AverageMeter()
            valid_acces = AverageMeter()
            labels_acm = list()
            predictions_acm = list()

            with torch.no_grad():
                for imgs, labels, unnorm_img in valid_dataloader:
                    imgs = imgs.cuda()
                    labels = labels.cuda()
                    bs = imgs.shape[0]
                    embeddings = backbone(imgs)
                    outputs = classifier(embeddings)

                    predictions_acm.append(outputs.detach().cpu().numpy())
                    labels_acm.append(labels.detach().cpu().numpy())

                    loss = ce_loss(outputs, labels)
                    valid_losses.update(loss.item(), bs)
                    acc, err_dict = cal_acc(outputs, labels)
                    valid_acces.update(acc, bs)


            acc = accuracy_calc(args.save_dir, labels_acm, predictions_acm, roc_plot_en=False)

            s = ('%10s' * 1 + '%10.4g' * 3) % (
                '%g/%g' % (epoch, args.epoch - 1), acc, labels.shape[0], imgs.shape[-1])
            with open(results_file, 'a') as f:
                f.write(s + '\n')  # append metrics, val_loss

            print(f"Epoch: {epoch+1} | train loss: {train_losses.avg:.4f} | train acc: {acc*100:.1f} | valid loss: {valid_losses.avg:.4f} | valid acc: {valid_acces.avg*100:.1f} ")

            if valid_acces.avg > best_acc:
                best_acc = valid_acces.avg
                torch.save(classifier.state_dict(), model_save_root)
                print(f"Save best model at epoch {epoch+1}.")
    else: # load pretrained model for test set only
        model_save_root = os.path.join(args.output_folder,
                                       args.cls_model_name)  # osp.join(args.output_folder, args.cls_model_name)

        args.save_dir = args.output_folder
        results_file = os.path.join(args.save_dir , 'results.txt')


    # Test set
    plot_err_ex = False
    classifier = load_classifier(fc_dim, n_cls=2, checkpoint=model_save_root)
    if 1:
        test_dataloader = create_dataloader(test_set_path, crop_size=global_crops_size, num_workers=args.num_workers, batch_size=64, shuffle=False)
        test_dataloader.dataset.df.to_csv(os.path.join(args.save_dir, 'test_set.csv'))
        print("Class 0, 1 distribution", np.unique([x[1] for x in test_dataloader.dataset.samples], return_counts=True))
    else:
        test_dataloader = get_dataloader(batch_size=64, shuffle=False, num_workers=4, mode="train", split=args.split, model=args.backbone_model)

    test_losses = AverageMeter()
    test_acces = AverageMeter()
    labels_acm = list()
    predictions_acm = list()
    with torch.no_grad():
        for ix, (imgs, labels, unnorm_img) in enumerate(tqdm(test_dataloader)):
            imgs = imgs.cuda()
            labels = labels.cuda()
            bs = imgs.shape[0]

            embeddings = backbone(imgs)
            outputs = classifier(embeddings)
            predictions_acm.append(outputs.detach().cpu().numpy())
            labels_acm.append(labels.detach().cpu().numpy())
            loss = ce_loss(outputs, labels)
            test_losses.update(loss.item(), bs)
            acc, err_dict = cal_acc(outputs, labels, threshold=0.6)
            if plot_err_ex:
                if any(err_dict['err_ind']):
                    for ii, conf in zip(err_dict['err_ind'], err_dict['conf_acm']):
                        cv2.imwrite(os.path.join(debug_save_path, str(ix*args.batch_size+ii)+ '_'+str(conf)+ '.png'), unnorm_img[ii,...].detach().cpu().permute(1, 2, 0).numpy()*255)

            test_acces.update(acc, bs)

    acc = accuracy_calc(args.save_dir, labels_acm, predictions_acm, roc_plot_en=True)

    test_results_file = os.path.join(args.save_dir, 'test_results.txt')

    s = ('%10s' * 1 + '%10.4g' * 3) % (
        '%g/%g' % (1, 1), acc, labels.shape[0], imgs.shape[-1])

    with open(test_results_file, 'a') as f:
        f.write(s + '\n')  # append metrics, val_loss

    print(f"Test loss: {test_losses.avg:.4f} | test acc: {acc*100:.1f}")


def lr_scheduler_setup(optimizer, linear_lr=True, start_epoch=0):

    # cosine_anneal = False
    lrf = 0.01  # final OneCycleLR learning rate (lr0 * lrf)
    warmup_epochs = 3

    def one_cycle(y1=0.0, y2=1.0, steps=100):
        # lambda function for sinusoidal ramp from y1 to y2
        return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1

    if linear_lr:
        lf = lambda x: (1 - x / (args.epoch - 1)) * (1.0 - lrf) + lrf  # linear
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    else:
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,
                                                             # T_0 period of 1st wamup Number of iterations for the first restart  ;T_mult=1 increase T_0 each period
                                                             T_0=int(2 * warmup_epochs), T_mult=2,
                                                             eta_min=0,  # hyp['lr0'] / 10,
                                                             last_epoch=-1)  # lr range test take max warmup/4 for CLR https://arxiv.org/abs/1803.09820s
    scheduler.last_epoch = start_epoch - 1
    return scheduler


if __name__ == "__main__":
    args = get_args()
    main(args)

    """
    --epoch 20 --batch_size 32 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier
training
    --epoch 20 --batch-size 32 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier --dropout 0.2
    
    test set only
    --epoch 20 --batch-size 32 --output-folder /mnt/Data/hanoch/runs/dinov2_classifier --dropout 0.2 --task test --output-folder /mnt/Data/hanoch/runs/dinov2_classifier/runs/train/dinov26
    """