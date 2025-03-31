import os
import torch
from torchvision import transforms
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2
from mmint_tools.camera_tools.pointcloud_utils import pack_o3d_pcd
from mmint_tools.camera_tools.img_utils import project_depth_image
from dataset.transform import Resize, NormalizeImage, PrepareForNet, ResizeTensor, PrepareForNetTensor
from torchvision.transforms import Compose
import torch.nn.functional as F
import cv2
import open3d as o3d
import copy
import imageio
import time
from scipy.spatial import cKDTree

def extract_tool_names(train_path):
    train_tools = os.listdir(train_path)

    for i, tool in enumerate(train_tools):
        remove_idx = tool.find('_data_')
        train_tools[i] = tool[:remove_idx]

    train_tools = list(set(train_tools))
    train_tools.sort()
    return train_tools

def logging_image_grid(images, captions, path, ncol=7, normalize = True, save = True):
    if not normalize:
        norm_text = "_not_normalized"
    else:
        norm_text = ""

    grids = [make_grid(img, nrow=ncol,padding=1, normalize=normalize, scale_each=True) for img in images]
    for grid, caption in zip(grids, captions):
        if save:
            save_image(grid, path +  '/' + caption + norm_text + '.png')
        else:
            plt.imshow(np.asarray(grid.permute((1,2,0)).cpu()))
            plt.title(caption)
            plt.axis('off')
            plt.show()
    return

class unnormalize(object):
    """Zeros out part of image with close values to zero"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, normalized_image):
        self.mean = self.mean.to(normalized_image.device)
        self.std = self.std.to(normalized_image.device)
        image = normalized_image*self.std + self.mean
        return image

class depth_testing(Dataset):
    def __init__(self, tool_name, model, model_path, bubbles_path, depth_path, device, masked = False, scale = 1.0):
        self.tool_name = tool_name
        model_loaded = torch.load(model_path, map_location='cpu')['model']
        self.model = model
        self.model.load_state_dict(model_loaded)
        self.model.eval()

        self.bubbles_path = bubbles_path
        self.depth_path = depth_path
        self.bubbles_data_paths = [path for path in os.listdir(bubbles_path) if tool_name in path]
        self.depth_data_paths = [path for path in os.listdir(depth_path) if tool_name in path]

        self.transform_1 = Compose([
            PrepareForNetTensor(),
            ResizeTensor(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method="bicubic",
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.transform = Compose([
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        self.device = device
        self.masked = masked
        self.scale = scale
    def __len__(self):
        return len(self.bubbles_data_paths)

    def __getitem__(self, idx):
        bubbles_data = torch.load(os.path.join(self.bubbles_path, self.bubbles_data_paths[idx]))
        bubbles_imprint = bubbles_data['bubble_imprint'].permute(0, 2, 3, 1).repeat(1, 1, 1, 3)
        bubbles_ref = bubbles_data['bubble_depth_ref'].permute(0, 2, 3, 1).repeat(1, 1, 1, 3)
        bubbles_img_original = bubbles_ref - bubbles_imprint
        bubbles_img_in = (bubbles_img_original - bubbles_img_original.min()) / (bubbles_img_original.max() - bubbles_img_original.min())
        bubbles_img = bubbles_img_in.numpy()
        h, w = bubbles_img.shape[1:-1]

        bubbles_img_r = self.transform({'image': bubbles_img[0]})['image']
        bubbles_img_l = self.transform({'image': bubbles_img[1]})['image']
        bubbles_img = np.concatenate((np.expand_dims(bubbles_img_r, axis=0), np.expand_dims(bubbles_img_l, axis=0)), axis=0)
        bubbles_img = torch.from_numpy(bubbles_img)
        bubbles_img = bubbles_img.to(self.device)
        depth_pred = self.model(bubbles_img) # HxW raw depth map in numpy
        depth_pred = F.interpolate(depth_pred[:, None], (h, w), mode="bilinear", align_corners=True)[:, 0]
        depth_pred = depth_pred / self.scale

        depth_data = torch.load(os.path.join(self.depth_path, self.depth_data_paths[idx]))
        depth_gt = depth_data['depth'].squeeze(0)

        if not self.masked:
            depth_gt[depth_gt <= 0] = 1e-9

        return bubbles_img_original.permute(0, 3, 1, 2), depth_gt, depth_pred.unsqueeze(1)
    
def get_data(tool_name, bubbles_path, depth_path, max_depth, device, model_results_path = '', masked = False, scale = 1.0):
    model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }
    encoder = 'vits' # or 'vits', 'vitb'
    dataset = 'bubbles' # 'hypersim' for indoor model, 'vkitti' for outdoor model
    model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})

    dataset = depth_testing(tool_name, model, model_results_path, bubbles_path, depth_path, device, masked=masked, scale=scale)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    return next(iter(dataloader))

def depth_qualitative(bubbles_img, depth_gt, depth_pred):
    idxs = [2,6,10,14]
    bubbles_img = bubbles_img[idxs]
    bubbles_img_viz_single = bubbles_img[:,1]
    bubbles_img_viz = torch.cat((bubbles_img[:,1], bubbles_img[:,0]), dim=2)

    depth_pred = depth_pred[idxs]
    depth_pred_viz_single = depth_pred[:,1]
    depth_pred_viz = torch.cat((depth_pred[:,1], depth_pred[:,0]), dim=2)

    depth_gt = depth_gt[idxs]
    depth_gt_viz_single = depth_gt[:,1]
    depth_gt_viz = torch.cat((depth_gt[:,1], depth_gt[:,0]), dim=2)

    depth_qualitative_results = {
                                'bubbles_img_viz': bubbles_img_viz,
                                'bubbles_img_viz_single': bubbles_img_viz_single,
                                'depth_pred_viz': depth_pred_viz,
                                'depth_pred_viz_single': depth_pred_viz_single,
                                'depth_gt_viz': depth_gt_viz,
                                'depth_gt_viz_single': depth_gt_viz_single
                                }
    
    return depth_qualitative_results

def chamfer_distance(points1, points2):
    """
    Compute the chamfer distance between two point clouds (points1 and points2).
    For each point in points1, we find the nearest neighbor in points2 and vice versa.
    The chamfer distance is the sum of the average squared distances from each cloud.
    """
    tree1 = cKDTree(points1)
    tree2 = cKDTree(points2)
    
    distances1, _ = tree2.query(points1, k=1)
    distances2, _ = tree1.query(points2, k=1)
    
    return np.mean(distances1**2) + np.mean(distances2**2)

def depth_quantitative(depth_gt, depth_pred, camera_info_path, camera_name):
    """
    Computes qualitative depth evaluation metrics including MSE, AbsRel, RMSE, LogRMSE, and SiLog.

    Args:
    - bubbles_input (torch.Tensor): Input image tensor (not used in metric calculation but retained for compatibility).
    - depth_gt (torch.Tensor): Ground truth depth map (B, C, H, W).
    - depth_pred (torch.Tensor): Predicted depth map (B, C, H, W).

    Returns:
    - visual_qualitative_results (dict): Dictionary containing computed depth metrics.
    """

    # Concatenate channels to match your format
    depth_gt_flat = torch.cat([depth_gt[:, 1], depth_gt[:, 0]], dim=0)
    depth_pred_flat = torch.cat([depth_pred[:, 1], depth_pred[:, 0]], dim=0)

    # Avoid division/log(0) issues by setting minimum depth value
    epsilon = 1e-8
    depth_gt_flat = torch.clamp(depth_gt_flat, min=epsilon)
    depth_pred_flat = torch.clamp(depth_pred_flat, min=epsilon)

    # Create mask for valid depth values (avoid zero or invalid pixels)
    valid_mask = (depth_gt_flat > 1e-8).float()

    # Count valid pixels
    valid_pixels = valid_mask.sum()

    # Mean Squared Error (MSE)
    mse_error = F.mse_loss(depth_pred_flat, depth_gt_flat, reduction='sum') / torch.numel(depth_pred_flat)

    # Absolute Relative Error (AbsRel)
    abs_rel = ((torch.abs(depth_pred_flat - depth_gt_flat) / depth_gt_flat) * valid_mask).sum() / valid_pixels

    # Root Mean Squared Error (RMSE)
    rmse = torch.sqrt(((depth_pred_flat - depth_gt_flat) ** 2 * valid_mask).sum() / valid_pixels)

    # Log RMSE
    log_rmse = torch.sqrt(((torch.log(depth_pred_flat) - torch.log(depth_gt_flat)) ** 2 * valid_mask).sum() / valid_pixels)

    # Scale-Invariant Logarithmic (SiLog) Loss
    log_diff = torch.log(depth_pred_flat) - torch.log(depth_gt_flat)
    silog_loss = torch.sqrt((log_diff ** 2).mean() - (log_diff.mean() ** 2)) + 0.15 * torch.abs(log_diff).mean()

    # Chamfer Loss betweeb PCD GT and PCD Pred
    chamfer_loss = 0.0
    len_samples = depth_gt.shape[0]
    
    for idx in range(len_samples):
        pcd_gt_r, pcd_gt_l = get_pcd(depth_gt[idx], camera_info_path, camera_name)
        pcd_pred_r, pcd_pred_l = get_pcd(depth_pred[idx].detach(), camera_info_path, camera_name)

        nb_neighbors = 100
        std_ratio = 20.0
        pcd_gt_r = clean_point_cloud(pcd_gt_r, method='statistical', nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        pcd_gt_l = clean_point_cloud(pcd_gt_l, method='statistical', nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        pcd_pred_r = clean_point_cloud(pcd_pred_r, method='statistical', nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        pcd_pred_l = clean_point_cloud(pcd_pred_l, method='statistical', nb_neighbors=nb_neighbors, std_ratio=std_ratio)

        # Convert the point clouds to NumPy arrays.
        points_gt_r = np.asarray(pcd_gt_r.points)
        points_gt_l = np.asarray(pcd_gt_l.points)
        points_pred_r = np.asarray(pcd_pred_r.points)
        points_pred_l = np.asarray(pcd_pred_l.points)

        # Compute chamfer distance for right and left views.
        chamfer_right = chamfer_distance(points_gt_r, points_pred_r)
        chamfer_left  = chamfer_distance(points_gt_l, points_pred_l)

        # Average the chamfer loss over both views.
        frame_loss = (chamfer_right + chamfer_left) / 2.0
        chamfer_loss += frame_loss

    # Average over all samples.
    chamfer_loss /= len_samples

    # Store results in a dictionary
    depth_qualitative_results = {
        'mse_error': mse_error.item(),
        'abs_rel': abs_rel.item(),
        'rmse': rmse.item(),
        'log_rmse': log_rmse.item(),
        'silog': silog_loss.item(),
        'chamfer_loss': chamfer_loss.item()
    }

    return depth_qualitative_results

def get_pcd(depth, camera_info_path, camera_name):
    camera_info_r_path = os.path.join(camera_info_path, camera_name + '_right.npy')
    camera_info_l_path = os.path.join(camera_info_path, camera_name + '_left.npy')
    intrinsics_r = np.load(camera_info_r_path, allow_pickle=True).item()['K']
    intrinsics_l = np.load(camera_info_l_path, allow_pickle=True).item()['K']

    depth[depth <= 1e-6] = 0
    
    pts_r_projected = project_depth_image(depth[0], intrinsics_r)
    pcd_r = pack_o3d_pcd(pts_r_projected.reshape(-1, 3))
    pcd_r.estimate_normals()

    pts_l_projected = project_depth_image(depth[1], intrinsics_l)
    pcd_l = pack_o3d_pcd(pts_l_projected.reshape(-1, 3))
    pcd_l.estimate_normals()
    
    return pcd_r, pcd_l

def compute_extrinsic_matrix(eye, center, up):
    forward = (center - eye)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    
    rot = np.eye(4)
    rot[0:3, 0] = right
    rot[0:3, 1] = up
    rot[0:3, 2] = -forward
    rot[0:3, 3] = eye
    extrinsic = np.linalg.inv(rot)
    return extrinsic

def get_rotation_matrix(view):
    # Rotates scene into position as if viewed from that direction
    rotations = {
        'front':   np.eye(3),
        'back':    o3d.geometry.get_rotation_matrix_from_axis_angle([0, np.pi, 0]),
        'left':    o3d.geometry.get_rotation_matrix_from_axis_angle([0, -np.pi/2, 0]),
        'right':   o3d.geometry.get_rotation_matrix_from_axis_angle([0, np.pi/2, 0]),
        'top':     o3d.geometry.get_rotation_matrix_from_axis_angle([np.pi/2, 0, 0]),
        'bottom':  o3d.geometry.get_rotation_matrix_from_axis_angle([-np.pi/2, 0, 0]),
    }
    return rotations.get(view, np.eye(3))  # Default to front

def clean_point_cloud(pcd, method='statistical', **kwargs):
    if method == 'statistical':
        nb_neighbors = kwargs.get('nb_neighbors', 20)
        std_ratio = kwargs.get('std_ratio', 2.0)
        pcd_clean, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    elif method == 'radius':
        nb_points = kwargs.get('nb_points', 16)
        radius = kwargs.get('radius', 0.01)
        pcd_clean, _ = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    else:
        raise ValueError(f"Unknown outlier removal method: {method}")
    return pcd_clean

def visualize_pcd_results(pcd_gt, pcd_pred, side='left', show=False):
    # Translate both point clouds so that the center of pcd_gt_r is at the origin.
    print("Frame pcd gt center before translation:", pcd_gt.get_center())
    print("Frame pcd pred center before translation:", pcd_pred.get_center())
    pcd_gt_r = copy.deepcopy(pcd_gt)
    pcd_pred_r = copy.deepcopy(pcd_pred)

    center_gt = pcd_gt_r.get_center()
    pcd_gt_r.translate(-center_gt)
    pcd_pred_r.translate(-center_gt)

    R = get_rotation_matrix(side)
    pcd_gt.rotate(R, center=(0, 0, 0))
    pcd_pred.rotate(R, center=(0, 0, 0))
    
    frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.005, 
                                                                     origin=np.array([0.0, 0.0, 0.0]))
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=show)

    # Color point clouds
    pcd_gt_r.paint_uniform_color([1, 0, 1])  # Magenta
    pcd_pred_r.paint_uniform_color([1, 1, 0])  # Yellow

    # Add geometries
    vis.add_geometry(pcd_gt_r)
    vis.add_geometry(pcd_pred_r)
    # vis.add_geometry(frame_mesh)

    print("Frame pcd gt center after translation:", pcd_gt_r.get_center())
    print("Frame pcd pred center after translation:", pcd_pred_r.get_center())
    print("Frame mesh center:", frame_mesh.get_center())


    view_ctl = vis.get_view_control()
    camera_params = view_ctl.convert_to_pinhole_camera_parameters()

    # Now that pcd_gt_r is centered at the origin, use the origin as the center.
    center = np.array([0, 0, 0])
    # Adjust the distance value according to your scene scale.
    distance = 0.0005  # (Use a positive value; adjust as needed)

    # Define view offsets and up directions for various views.
    view_offsets = {
        'front':   np.array([0, 0, distance]),
        'back':    np.array([0, 0, -distance]),
        'left':    np.array([-distance, 0, 0]),
        'right':   np.array([distance, 0, 0]),
        'top':     np.array([0, distance, 0]),
        'bottom':  np.array([0, -distance, 0]),
    }

    up_vectors = {
        'front':   [0, -1, 0],
        'back':    [0, -1, 0],
        'left':    [0, -1, 0],
        'right':   [0, -1, 0],
        'top':     [0, 0, -1],
        'bottom':  [0, 0, 1],
    }

    if side not in view_offsets:
        print(f"Warning: '{side}' is not recognized. Defaulting to 'left'.")
        side = 'left'

    eye = center + view_offsets['front']
    up = up_vectors['front']

    # Compute and set the extrinsic matrix for the view.
    extrinsic = compute_extrinsic_matrix(eye, center, up)
    camera_params.extrinsic = np.linalg.inv(extrinsic)
    view_ctl.convert_from_pinhole_camera_parameters(camera_params)
    view_ctl.scale(50)

    if show:
        vis.run()

    image = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()
    return image

def create_rotation_frames(pcd_gt, pcd_pred, axis='z', num_frames=60, 
                              rotation_degrees=360, image_size=(512,512)):
    """
    Create an animated GIF by rotating the point clouds (ground-truth and prediction)
    around the specified axis about the GT center. A fixed camera view is maintained
    by setting the extrinsic matrix via pinhole camera parameters.
    """
    # Deep-copy and center point clouds based on the GT center.
    pcd_gt_centered = copy.deepcopy(pcd_gt)
    pcd_pred_centered = copy.deepcopy(pcd_pred)
    center_gt = pcd_gt_centered.get_center()
    pcd_gt_centered.translate(-center_gt)
    pcd_pred_centered.translate(-center_gt)
    
    # Fixed camera parameters: choose an eye point along +Z.
    eye = np.array([0, 0, 0.01])
    center = np.array([0, 0, 0])
    up = np.array([0, -1, 0])
    extrinsic = compute_extrinsic_matrix(eye, center, up)
    
    images = []
    
    # Map axis letter to axis vector.
    axis_map = {'x': np.array([1, 0, 0]),
                'y': np.array([0, 1, 0]),
                'z': np.array([0, 0, 1])}
    if axis not in axis_map:
        raise ValueError("Axis must be one of 'x', 'y', or 'z'")
    rot_axis = axis_map[axis]
    
    # Precompute the fixed initial rotation: 180° about y-axis.
    R_y = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([0, np.pi, 0]))

    # Create one Visualizer window (invisible) for all frames.
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=image_size[0], height=image_size[1])
    view_ctl = vis.get_view_control()
    camera_params = view_ctl.convert_to_pinhole_camera_parameters()
    # Invert the computed extrinsic to mimic the interactive camera setup.
    camera_params.extrinsic = np.linalg.inv(extrinsic)
    view_ctl.convert_from_pinhole_camera_parameters(camera_params)
    view_ctl.scale(50)
    
    for i in range(num_frames):
        # Compute the additional rotation angle for the current frame.
        angle = np.deg2rad(rotation_degrees * i / num_frames)
        R_frame = o3d.geometry.get_rotation_matrix_from_axis_angle(rot_axis * angle)
        # Total rotation: always start with 180° about y, then apply additional rotation.
        R_total = R_frame @ R_y
        
        # Rotate the centered point clouds around the origin.
        gt_rotated = copy.deepcopy(pcd_gt_centered).rotate(R_total, center=(0, 0, 0))
        pred_rotated = copy.deepcopy(pcd_pred_centered).rotate(R_total, center=(0, 0, 0))
        gt_rotated.paint_uniform_color([1, 0, 1])   # Magenta for GT.
        pred_rotated.paint_uniform_color([1, 1, 0])   # Yellow for prediction.
        
        # Clear previous geometries and add the updated ones.
        vis.clear_geometries()
        vis.add_geometry(gt_rotated)
        vis.add_geometry(pred_rotated)
        
        # Update the renderer.
        vis.poll_events()
        vis.update_renderer()
        # Optional: wait a short time to ensure rendering is complete.
        time.sleep(0.05)
        image = vis.capture_screen_float_buffer(do_render=True)
        image_np = (np.asarray(image) * 255).astype(np.uint8)
        images.append(image_np)
    
    vis.destroy_window()
    return images

def create_rotation_animation(pcd_gt, pcd_pred, num_frames=60, 
                              rotation_degrees=360, image_size=(512,512), 
                              output_path="rotation.gif"):
    
    """
    Create an animated GIF by rotating the point clouds (ground-truth and prediction)"
    "around the specified axis about the GT center. A fixed camera view is maintained
    by setting the extrinsic matrix via pinhole camera parameters."
    """

    images_y = create_rotation_frames(pcd_gt, pcd_pred, axis='y', num_frames=num_frames,
                                      rotation_degrees=rotation_degrees, image_size=image_size)
    images_x = create_rotation_frames(pcd_gt, pcd_pred, axis='x', num_frames=num_frames,
                                      rotation_degrees=rotation_degrees, image_size=image_size)
    
    images_y = np.array(images_y)
    images_x = np.array(images_x)
    images = np.concatenate((images_y, images_x), axis=2)

    # Save all frames as an animated GIF.
    imageio.mimsave(output_path, images, duration=(1000 * 1/15))
    print(f"Saved animation to {output_path}")

    return


def create_rotation_animation_offscreen(pcd_gt, pcd_pred, axis='z', num_frames=60,
                                        rotation_degrees=360, width=512, height=512,
                                        output_path="rotation.gif"):
    # Center the point clouds based on the ground-truth center.
    pcd_gt_centered = copy.deepcopy(pcd_gt)
    pcd_pred_centered = copy.deepcopy(pcd_pred)
    center_gt = pcd_gt_centered.get_center()
    pcd_gt_centered.translate(-center_gt)
    pcd_pred_centered.translate(-center_gt)

    # Setup the offscreen renderer.
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([1, 1, 1, 1])  # white background

    # Create materials for the point clouds.
    material_gt = o3d.visualization.rendering.MaterialRecord()
    material_gt.shader = "defaultUnlit"
    material_pred = o3d.visualization.rendering.MaterialRecord()
    material_pred.shader = "defaultUnlit"

    # Setup a fixed camera (looking from +Z) using a computed extrinsic.
    eye = np.array([0, 0, 0.01])
    center = np.array([0, 0, 0])
    up = np.array([0, -1, 0])
    extrinsic = compute_extrinsic_matrix(eye, center, up)
    # As suggested, invert the extrinsic so it matches what the interactive code would do.
    camera_model_matrix = np.linalg.inv(extrinsic)

    # Setup the perspective camera.
    fov = 60.0
    aspect = width / height
    znear = 0.001
    zfar = 1000.0
    # Use a vertical field-of-view type.
    fov_type = o3d.visualization.rendering.Camera.FovType.Vertical
    renderer.scene.camera.set_projection(fov, aspect, znear, zfar, fov_type)
    renderer.scene.camera.set_model_matrix(camera_model_matrix)

    # Define the rotation axis.
    axis_map = {
        'x': np.array([1, 0, 0]),
        'y': np.array([0, 1, 0]),
        'z': np.array([0, 0, 1])
    }
    if axis not in axis_map:
        raise ValueError("Axis must be 'x', 'y', or 'z'")
    rot_axis = axis_map[axis]

    images = []
    for i in range(num_frames):
        angle = np.deg2rad(rotation_degrees * i / num_frames)
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(rot_axis * angle)

        # Rotate the centered point clouds around the origin (which is the GT center).
        gt_rotated = copy.deepcopy(pcd_gt_centered).rotate(R, center=(0, 0, 0))
        pred_rotated = copy.deepcopy(pcd_pred_centered).rotate(R, center=(0, 0, 0))
        gt_rotated.paint_uniform_color([1, 0, 1])   # Magenta for GT.
        pred_rotated.paint_uniform_color([1, 1, 0])   # Yellow for prediction.

        # Remove any existing geometries from the scene.
        if "gt" in renderer.scene.get_geometry_names():
            renderer.scene.remove_geometry("gt")
        if "pred" in renderer.scene.get_geometry_names():
            renderer.scene.remove_geometry("pred")
        renderer.scene.add_geometry("gt", gt_rotated, material_gt)
        renderer.scene.add_geometry("pred", pred_rotated, material_pred)

        # Render the frame.
        img_o3d = renderer.render_to_image()
        img_np = np.asarray(img_o3d)
        images.append(img_np)

    # Save all frames as an animated GIF.
    imageio.mimsave(output_path, images, fps=30)
    print(f"Saved animation to {output_path}")
    return

def depth_pcd_quantitative(depth_gt, depth_pred, camera_info_path, camera_name, results_path='results'):
    right_path = os.path.join(results_path, 'pcd_right.gif')
    left_path = os.path.join(results_path, 'pcd_left.gif')
    
    if (not os.path.exists(right_path)) or (not os.path.exists(left_path)):
        idx = 2
        pcd_gt_r, pcd_gt_l = get_pcd(depth_gt[idx], camera_info_path, camera_name)
        pcd_pred_r, pcd_pred_l = get_pcd(depth_pred[idx].detach(), camera_info_path, camera_name)

        nb_neighbors = 100
        std_ratio = 20.0
        pcd_gt_r = clean_point_cloud(pcd_gt_r, method='statistical', nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        pcd_gt_l = clean_point_cloud(pcd_gt_l, method='statistical', nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        pcd_pred_r = clean_point_cloud(pcd_pred_r, method='statistical', nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        pcd_pred_l = clean_point_cloud(pcd_pred_l, method='statistical', nb_neighbors=nb_neighbors, std_ratio=std_ratio)

        create_rotation_animation(pcd_gt_r, pcd_pred_r, output_path=os.path.join(results_path, 'pcd_right.gif'))
        create_rotation_animation(pcd_gt_l, pcd_pred_l, output_path=os.path.join(results_path, 'pcd_left.gif'))

    return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='TST Testing')
    parser.add_argument('--model', type=str, default='depth_anything_metric', help='Model options: depth_anything_metric')
    parser.add_argument('--name', type=str, default='fine_tuning_bubbles_max0.12_no_mask_not_zero_train_tools', help='Model name')
    parser.add_argument('--max_depth', type=float, default=0.12, help='Max depth for the model')
    parser.add_argument('--scale', type=float, default=1.0, help='Scale Depth Map')
    parser.add_argument('--masked', action='store_true')
    parser.add_argument('--depth_qual' , action='store_true')
    parser.add_argument('--depth_quant' , action='store_true')
    parser.add_argument('--depth_pcd_qual', action='store_true')
    parser.add_argument('--only_unseen_tools', action='store_true')
    parser.add_argument('--all_metrics', action='store_true', default=True)
    args = parser.parse_args()

    # Model details
    method = args.model
    name = args.name
    project_path = '/home/samanta/depth_anything_v2/metric_depth'
    output_folder_name = 'results'
    print('Project Path:', project_path)
    print('Method:', method)
    print('Model:', name)

    # Get data paths
    datasets_path = '/home/samanta/T2D2/data'
    train_path = os.path.join(datasets_path, "train_evaluation")
    test_path = os.path.join(datasets_path, "test_evaluation")
    test_unseen_path = test_path

    model_results_path = os.path.join(project_path, "exp", name)
    model_path = os.path.join(model_results_path, "latest.pth")
    results_paths = [model_path, model_path, model_path]

    # Get tools
    train_tools = torch.load(os.path.join(model_results_path, "tools.pt"))['train_tools']
    train_tools.sort()
    test_tools = torch.load(os.path.join(model_results_path, "tools.pt"))['train_tools']
    test_tools.sort()
    test_unseen_tools = torch.load(os.path.join(model_results_path, "tools.pt"))['test_tools']
    test_unseen_tools.sort()

    print('Train tools:', len(train_tools))
    for tool in train_tools:
        print(tool)
    print('-----------------------------')
    print('Test tools:', len(test_tools))
    for tool in test_tools:
        print(tool)
    print('-----------------------------')
    print('Test unseen tools:', len(test_unseen_tools))
    for tool in test_unseen_tools:
        print(tool)
    print('-----------------------------')

    # Define datasets
    if args.only_unseen_tools:
        dataset_paths = [test_unseen_path]
        datasets = ["test_unseen"]
        datasets_tools = [test_unseen_tools]
    else:
        dataset_paths = [train_path, test_path, test_unseen_path]
        datasets = ["train", "test", "test_unseen"]
        datasets_tools = [train_tools, test_tools, test_unseen_tools]

    for j in range(len(dataset_paths)):
        dataset_path = dataset_paths[j]
        model_results_path = results_paths[j]
        dataset = datasets[j]
        dataset_tools = datasets_tools[j]
        print('Dataset:', dataset)

        for i in range(len(dataset_tools)):
            print(dataset_tools[i])
            tool_name = dataset_tools[i]
            output_path = os.path.join(project_path, output_folder_name, name, dataset, tool_name)
            
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            bubbles_input, depth_gt, depth_pred = get_data(tool_name, os.path.join(dataset_path, 'bubbles'), os.path.join(dataset_path, 'bubbles_depth'), 
                                                           args.max_depth, 'cpu', model_results_path, masked = args.masked, scale = args.scale)

            # Visual Qualitative Results
            if args.depth_qual or args.all_metrics:
                if not os.path.exists(output_path + '/depth_qualitative_results.pt'):
                    depth_qualitative_results = depth_qualitative(bubbles_input, depth_gt, depth_pred)
                    torch.save(depth_qualitative_results, output_path + '/depth_qualitative_results.pt')

            # Visual Quantitative Results
            if args.depth_quant or args.all_metrics:
                if not os.path.exists(output_path + '/depth_quantitative_results.pt'):
                    depth_quantitative_results = depth_quantitative(depth_gt, depth_pred, datasets_path, 'bubbles_camera_info') 
                    torch.save(depth_quantitative_results, output_path + '/depth_quantitative_results.pt')

            # Visual PCD Qualitative Results
            if args.depth_pcd_qual or args.all_metrics:
                depth_pcd_quantitative(depth_gt, depth_pred, datasets_path, 'bubbles_camera_info', results_path=output_path)
