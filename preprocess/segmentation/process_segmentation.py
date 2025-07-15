# Copyright (c) Facebook, Inc. and its affiliates.
from kitti_dataset import CrawlKittiDataPath
from nuscenes_scenes import crawl_scenes_paths
from nuscenes_dataset import CrawlNuScenesDataPath
from predictor import VisualizationDemo
from detectron2.utils.logger import setup_logger
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.detection_utils import read_image
from detectron2.config import get_cfg
from os.path import join
from pathlib import Path
import tqdm
import numpy as np
import cv2
import warnings
import time
import tempfile
import argparse
import glob
import multiprocessing as mp
import os
from poplib import CR

# fmt: off
import sys
from turtle import pd
sys.path.insert(1, os.path.join(sys.path[0], '..'))

mask2former_path = os.path.join(Path.home(), "src/Mask2Former")
sys.path.append(mask2former_path)
from mask2former import add_maskformer2_config
# fmt: on


# constants
WINDOW_NAME = "mask2former demo"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default=os.path.join(mask2former_path, "configs/mapillary-vistas/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_300k.yaml"),
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--scene_names", type=str, nargs="+", 
        required=True, 
        help="scene names to process, e.g. scene-0655")
    parser.add_argument(
        "--base_dir",
        default="~/data/nuscenes-mini/",
        help="nuScenes base dir",
    )

    parser.add_argument(
        "--save_dir",
        default="~/data/nuscenes-mini/nuScenes_clip/",
        help="nuScenes base dir for segmentation results",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=["MODEL.WEIGHTS", os.path.join(mask2former_path, "mask2former_mapillary_vistas_swin_L.pkl")],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    
    # Nuscenes
    #file_paths = CrawlNuScenesDataPath(args.base_dir, camera_names)
    print(f"scene_names: {args.scene_names}")
    scenes = args.scene_names
    root_dir = os.path.expanduser(args.base_dir)
    save_dir = os.path.expanduser(args.save_dir)
    version = "mini"
    camera_names = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
                    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
    file_paths = crawl_scenes_paths(root_dir, version, scenes, camera_names)

    save_paths = [file_path.replace(root_dir, save_dir)  for file_path in file_paths]

    label_save_paths = [save_path.replace("/CAM", "/seg_CAM")  for save_path in save_paths]
    label_save_paths = [label_save_path.replace(".jpg", ".png")  for label_save_path in label_save_paths]
    
    
    for i in tqdm.tqdm(range(len(file_paths))):
        # use PIL, to be consistent with evaluation
        source_name = file_paths[i]
        label_name = label_save_paths[i]
        img = read_image(source_name, format="BGR")
        predictions, visualized_output = demo.run_on_image(img)
        # visualized_output.save(vis_label_name)
        save_img = predictions["sem_seg"].argmax(dim=0).cpu().numpy()  # (H, W)
        Path(os.path.dirname(label_name)).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(label_name, save_img)
