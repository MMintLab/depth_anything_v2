import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import os
import numpy as np

from dataset.transform import Resize, NormalizeImage, PrepareForNet


class BUBBLES(Dataset):
    def __init__(self, path, mode, tools, size=(518, 518), masked = False, scale=1):
        self.mode = mode
        self.size = size
        self.masked = masked
        self.scale = scale
        
        self.filelist = os.listdir(path)
        self.filelist = [f for f in self.filelist if any(tool in f for tool in tools)]
        self.filelist = [os.path.join(path, f) for f in self.filelist]
        
        net_w, net_h = size
        self.transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=True if mode == 'train' else False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
    
    def __getitem__(self, item):
        img_path = self.filelist[item]
        depth_path = img_path.replace('bubbles', 'bubbles_depth')
        
        bubble_imprint = torch.load(img_path)['bubble_imprint'].permute(0, 2, 3, 1).repeat(1, 1, 1, 3).numpy()
        bubble_ref = torch.load(img_path)['bubble_depth_ref'].permute(0, 2, 3, 1).repeat(1, 1, 1, 3).numpy()
        bubble_image = bubble_ref - bubble_imprint
        # bubble_image = bubble_imprint
        
        image = (bubble_image - np.min(bubble_image)) / (np.max(bubble_image) - np.min(bubble_image))
        
        depth = torch.load(depth_path)['depth'].squeeze(1).numpy() * self.scale

        if not self.masked:
            depth[depth <= 0] = 1e-9
        
        sample_r = self.transform({'image': image[0], 'depth': depth[0]})
        sample_r['image'] = torch.from_numpy(sample_r['image'])
        sample_r['depth'] = torch.from_numpy(sample_r['depth'])
        sample_r['depth'] = sample_r['depth']  # convert in meters

        if self.masked:
            sample_r['valid_mask'] = sample_r['depth'] > 0.0
        else:
            sample_r['valid_mask'] = sample_r['depth'] > -0.01
        sample_r['image_path'] = img_path

        sample_l = self.transform({'image': image[1], 'depth': depth[1]})
        sample_l['image'] = torch.from_numpy(sample_l['image'])
        sample_l['depth'] = torch.from_numpy(sample_l['depth'])
        sample_l['depth'] = sample_l['depth']

        if self.masked:
            sample_l['valid_mask'] = sample_l['depth'] > 0.0
        else:
            sample_l['valid_mask'] = sample_l['depth'] > -0.01
        sample_l['image_path'] = img_path
        
        return sample_r, sample_l

    def __len__(self):
        return len(self.filelist)

class GELSLIMS(Dataset):
    def __init__(self, path, mode, tools, size=(518, 518), masked = False, scale=1):
        self.mode = mode
        self.size = size
        self.masked = masked
        self.scale = scale
        
        self.filelist = os.listdir(path)
        self.filelist = [f for f in self.filelist if any(tool in f for tool in tools)]
        self.filelist = [os.path.join(path, f) for f in self.filelist]
        
        net_w, net_h = size
        self.transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=True if mode == 'train' else False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
    
    def __getitem__(self, item):
        img_path = self.filelist[item]
        depth_path = img_path.replace('gelslims_undistorted', 'gelslims_undistorted_depth')
        
        gelslim_image = torch.load(img_path)['gelslim'].permute(0, 2, 3, 1).numpy()
        gelslim_image_ref = torch.load(img_path)['gelslim_ref'].permute(0, 2, 3, 1).numpy()
        image  = gelslim_image # - gelslim_image_ref
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        
        depth = torch.load(depth_path)['depth'].squeeze(1).numpy() * self.scale

        if not self.masked:
            depth[depth <= 0] = 1e-9
        
        sample_r = self.transform({'image': image[0], 'depth': depth[0]})
        sample_r['image'] = torch.from_numpy(sample_r['image'])
        sample_r['depth'] = torch.from_numpy(sample_r['depth'])
        sample_r['depth'] = sample_r['depth']  # convert in meters

        if self.masked:
            sample_r['valid_mask'] = sample_r['depth'] > 0.0
        else:
            sample_r['valid_mask'] = sample_r['depth'] > -0.01
        sample_r['image_path'] = img_path

        sample_l = self.transform({'image': image[1], 'depth': depth[1]})
        sample_l['image'] = torch.from_numpy(sample_l['image'])
        sample_l['depth'] = torch.from_numpy(sample_l['depth'])
        sample_l['depth'] = sample_l['depth']

        if self.masked:
            sample_l['valid_mask'] = sample_l['depth'] > 0.0
        else:
            sample_l['valid_mask'] = sample_l['depth'] > -0.01
        sample_l['image_path'] = img_path
        
        return sample_r, sample_l

    def __len__(self):
        return len(self.filelist)