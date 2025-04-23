import argparse

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epoch", default=5, type=int)
    
    parser.add_argument("--backbone_model", default="dinov2", choices=["resnet50", "dinov2"], type=str)
    parser.add_argument("--split", default="1", type=str)
    parser.add_argument("--output-folder", default="outputs", type=str)
    parser.add_argument("--task", default="train_val", choices=["train_val", "test"], type=str)
    parser.add_argument("--dropout", default=0, type=float)
    parser.add_argument("--name", default="dinov2", type=str)
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--num-workers', type=int, default=8, help='maximum number of dataloader workers')





    args = parser.parse_args()

    if args.backbone_model == "dinov2":
        args.cls_model_name = "cls_dinov2.pth"
        # args.fc_dim = 384
    elif args.backbone_model == "resnet50":
        args.cls_model_name = "cls_resnet50.pth"
        args.fc_dim = 2048

    return args