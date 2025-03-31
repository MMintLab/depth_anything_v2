import argparse
import logging
import os
import pprint
import random

import warnings
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
import wandb
import torchvision.utils as vutils

from dataset.hypersim import Hypersim
from dataset.kitti import KITTI
from dataset.vkitti2 import VKITTI2
from dataset.tactile import BUBBLES, GELSLIMS
from depth_anything_v2.dpt import DepthAnythingV2
from util.loss import SiLogLoss
from util.metric import eval_depth
from util.utils import init_log

parser = argparse.ArgumentParser(description='Depth Anything V2 for Metric Depth Estimation')

parser.add_argument('--encoder', default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
parser.add_argument('--dataset', default='hypersim', choices=['hypersim', 'vkitti', 'bubbles', 'gelslims'])
parser.add_argument('--img-size', default=518, type=int)
parser.add_argument('--min-depth', default=-0.001, type=float)
parser.add_argument('--max-depth', default=20, type=float)
parser.add_argument('--epochs', default=40, type=int)
parser.add_argument('--bs', default=2, type=int)
parser.add_argument('--lr', default=0.000005, type=float)
parser.add_argument('--pretrained-from', type=str)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--masked', action='store_true')
parser.add_argument('--scale', default=1, type=float)

def main():
    args = parser.parse_args()

    if not os.path.exists(args.save_path):\
        os.makedirs(args.save_path)

    run_name = os.path.basename(args.save_path)
    
    warnings.simplefilter('ignore', np.RankWarning)
    
    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:1")
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    logger.info(f'Using device: {device}')
    
    wandb.init(project="depth-anything-v2", config=vars(args), name=run_name)
    
    cudnn.enabled = True
    cudnn.benchmark = True
    
    train_tools = ['pattern_02_2_lines_angle_2',
             'pattern_33',
             'pattern_03_2_lines_angle_3',
             'pattern_37',
             'pattern_01_2_lines_angle_1',
             'pattern_32',
             'pattern_06_5_lines_angle_1',
             'pattern_04_3_lines_angle_1',
             'pattern_31_rod']

    test_tools = ['pattern_05_3_lines_angle_2', 'pattern_35', 'pattern_36']

    size = (args.img_size, args.img_size)
    if args.dataset == 'hypersim':
        trainset = Hypersim('dataset/splits/hypersim/train.txt', 'train', size=size)
    elif args.dataset == 'vkitti':
        trainset = VKITTI2('dataset/splits/vkitti2/train.txt', 'train', size=size)
    elif args.dataset == 'bubbles':
        trainset = BUBBLES('/home/samanta/T2D2/data/train_evaluation/bubbles', 'train', train_tools, size=size, masked=args.masked, scale=args.scale)
    elif args.dataset == 'gelslims':
        trainset = GELSLIMS('/home/samanta/T2D2/data/train_evaluation/gelslims_undistorted', 'train', train_tools, size=size, masked=args.masked, scale=args.scale)
    else:
        raise NotImplementedError
    
    trainloader = DataLoader(trainset, batch_size=args.bs, pin_memory=True, num_workers=4, drop_last=True, shuffle=True)
    
    # if args.dataset == 'hypersim':
    #     valset = Hypersim('dataset/splits/hypersim/val.txt', 'val', size=size)
    # elif args.dataset == 'vkitti':
    #     valset = KITTI('dataset/splits/kitti/val.txt', 'val', size=size)
    # elif args.dataset == 'bubbles':
    #     valset = BUBBLES('/home/samanta/T2D2T/data/test_only/bubbles', 'val', train_tools, size=size)
    # elif args.dataset == 'gelslims':
    #     valset = GELSLIMS('/home/samanta/T2D2T/data/test_only/gelslims_undistorted', 'val', train_tools, size=size)
    # else:
    #     raise NotImplementedError
    
    # valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=4, drop_last=True, shuffle=False)
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    
    if args.pretrained_from:
        model.load_state_dict(torch.load(args.pretrained_from, map_location='cpu'), strict=False)
    
    model.to(device)
    
    criterion = SiLogLoss().to(device)
    
    # optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)

    optimizer = AdamW([{'params': [param for name, param in model.named_parameters() if 'pretrained' in name], 'lr': args.lr},
                       {'params': [param for name, param in model.named_parameters() if 'pretrained' not in name], 'lr': args.lr * 10.0}],
                      lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    
    import pdb; pdb.set_trace()
    
    total_iters = args.epochs * len(trainloader)

    if args.masked:
        args.min_depth = 0.001
    
    for epoch in range(args.epochs):
        logger.info(f'Epoch {epoch+1}/{args.epochs}')
        
        model.train()
        total_loss = 0
        
        for i, (sample_r, sample_l) in enumerate(trainloader):
            optimizer.zero_grad()
            
            img_r, depth_r, valid_mask_r = sample_r['image'].to(device), sample_r['depth'].to(device), sample_r['valid_mask'].to(device)
            img_l, depth_l, valid_mask_l = sample_l['image'].to(device), sample_l['depth'].to(device), sample_l['valid_mask'].to(device)
            img = torch.cat([img_r, img_l], dim=0)
            depth = torch.cat([depth_r, depth_l], dim=0)
            valid_mask = torch.cat([valid_mask_r, valid_mask_l], dim=0)

            if random.random() < 0.5:
                img = img.flip(-1)
                depth = depth.flip(-1)
                valid_mask = valid_mask.flip(-1)
            
            pred = model(img)
            loss = criterion(pred, depth, (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            # TODO: Add lr scheduler if needed
            
            if i % 100 == 0:
                logger.info(f'Iter {i}/{len(trainloader)}, Loss: {loss.item():.4f}')
        
        with torch.no_grad():
            img_grid = vutils.make_grid(img.cpu(), normalize=True, scale_each=True)
            pred_grid = vutils.make_grid(pred.unsqueeze(1).cpu(), normalize=True, scale_each=True)
            depth_grid = vutils.make_grid(depth.unsqueeze(1).cpu(), normalize=True, scale_each=True)
            wandb.log(data = {"Input Image": wandb.Image(img_grid, caption="Input")})
            wandb.log(data = {"Predicted Depth": wandb.Image(pred_grid, caption="Predicted")})
            wandb.log(data = {"Ground Truth Depth": wandb.Image(depth_grid, caption="Ground Truth")})
            rmse_error = eval_depth(pred, depth)['rmse']
            wandb.log({"train_loss": total_loss / len(trainloader), "RMSE Error": rmse_error, "epoch": epoch})

        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}, os.path.join(args.save_path, 'latest.pth'))
        wandb.save(os.path.join(args.save_path, 'latest.pth'))

        tools = {'train_tools': train_tools, 'test_tools': test_tools}
        torch.save(tools, os.path.join(args.save_path, 'tools.pt'))

if __name__ == '__main__':
    main()