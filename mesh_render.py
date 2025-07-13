import torch
import os
import addict
import argparse
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.gaussian_model import GaussianModel2D
from models.exposure_model import ExposureModel
from utils.render import render
from functools import partial
from utils.mesh_utils import GaussianExtractor, post_process_mesh
from diff_gaussian_rasterization.scene.cameras import PerspectiveCamera
import open3d as o3d

def get_configs():
    parser = argparse.ArgumentParser(description='Mesh Render config')
    parser.add_argument("--model_path", required=True, help='Path to the model directory')
    parser.add_argument('--config_file', default="local_nusc_mini.yaml", help='config yaml file name')
    args = parser.parse_args()
    config_path = os.path.join(args.model_path, args.config_file)
    with open(config_path) as file:
        configs = yaml.safe_load(file)
    configs["model_path"] = args.model_path
    return configs

if __name__ == "__main__":
    configs = get_configs()
    configs = addict.Dict(configs)
    dataset_cfg = configs.dataset
    model_cfg = configs.model
    pipeline_cfg = configs.pipeline
    opt_cfg = configs.optimization
    mesh_cfg = configs.mesh_render

    print("Rendering " + configs.model_path)
    
    device = torch.device(configs["device"] if torch.cuda.is_available() else "cpu")
    if dataset_cfg["dataset"] == "NuscDataset":
        from datasets.nusc import NuscDataset as Dataset
    elif dataset_cfg["dataset"] == "KittiDataset":
        from datasets.kitti import KittiDataset as Dataset
    else:
        raise NotImplementedError("Dataset not implemented")
    dataset = Dataset(dataset_cfg, use_label=opt_cfg.seg_loss_weight > 0, use_depth=opt_cfg.depth_loss_weight > 0)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=True, drop_last=True)
    trainCameras = []
    for sample in tqdm(dataloader, desc="Find cameras"):
        for key, value in sample.items():
            if key != "image_name":
                sample[key] = value[0].to(device)
            else:
                sample[key] = value[0]
        image_name = sample["image_name"]
        gt_image = sample["image"]
        image_idx = sample["idx"].item()
        cam_idx = sample["cam_idx"].item()
        R, T = sample["R"], sample["T"]

        NEAR, FAR = 1, 20
        viewpoint_cam = PerspectiveCamera(R, T, sample["K"], sample["W"], sample["H"], NEAR, FAR, device, cam_idx)
        trainCameras.append(viewpoint_cam)
    print("Find cameras: ", len(trainCameras))
    
    gaussians = GaussianModel2D(model_cfg)
    model_params = torch.load(os.path.join(configs.model_path, 'final.pth'), weights_only=False)
    gaussians.restore(model_params, opt_cfg)
    
    if model_cfg.use_exposure:
        exposure_model = ExposureModel(num_camera=len(dataset.camera_names)).to(device)
        exposure_model.load_state_dict(torch.load(os.path.join(configs.model_path, "exposure.pth")))
    
    bg_color = [1, 1, 1] if model_cfg.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)
    gaussian_render = partial(render, pipe = pipeline_cfg, bg_color=background)
    gaussExtractor = GaussianExtractor(gaussians, exposure_model, gaussian_render)    

    print("export mesh ...")
    mesh_output_dir = os.path.join(configs.model_path, 'mesh')
    os.makedirs(mesh_output_dir, exist_ok=True)
    # set the active_sh to 0 to export only diffuse texture
    gaussExtractor.gaussians.active_sh_degree = 0
    gaussExtractor.reconstruction(trainCameras)
    # extract the mesh and save
    name = 'mesh.ply'
    depth_trunc = (gaussExtractor.radius * 2.0) if mesh_cfg.depth_trunc < 0  else mesh_cfg.depth_trunc
    voxel_size = (depth_trunc / mesh_cfg.mesh_res) if mesh_cfg.voxel_size < 0 else mesh_cfg.voxel_size
    sdf_trunc = 5.0 * voxel_size if mesh_cfg.sdf_trunc < 0 else mesh_cfg.sdf_trunc
    mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
    
    #o3d.io.write_triangle_mesh(os.path.join(mesh_output_dir, name), mesh)
    #print("mesh saved at {}".format(os.path.join(mesh_output_dir, name)))
    
    # post-process the mesh and save, saving the largest N clusters
    mesh_post = post_process_mesh(mesh, cluster_to_keep=mesh_cfg.num_cluster)
    o3d.io.write_triangle_mesh(os.path.join(mesh_output_dir, name.replace('.ply', '_post.ply')), mesh_post)
    print("mesh post processed saved at {}".format(os.path.join(mesh_output_dir, name.replace('.ply', '_post.ply'))))