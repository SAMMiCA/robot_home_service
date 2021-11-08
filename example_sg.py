from collections import OrderedDict, defaultdict
import os

import numpy as np
from env.tasks import HomeServiceTaskSampler, HomeServiceTaskType
from experiments.home_service_base import HomeServiceBaseExperimentConfig
from PIL import Image
import argparse
import torch
import torch.backends.cudnn as cudnn
import pickle

import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

# from scene_graph.tsdf_scan_whole import Scan
# from scene_graph.visualize import ScannetVis
from scene_graph.utils.functions import SavePath
from scene_graph.data import cfg, set_cfg, set_dataset
from scene_graph.yolact import Yolact
from scene_graph.fusion_whole import rigid_transform
from scene_graph.utils.util import str2bool, calc_map, get_parser, get_view_frustum, get_camera_pose
from scene_graph.utils.augmentations import FastBaseTransform
from scene_graph.test_scan import Scan
from scene_graph.test_visualize import ScannetVis
from allenact.embodiedai.mapping.mapping_utils.point_cloud_utils import depth_frame_to_camera_space_xyz

"""
bash command
python example_sg.py --trained_model=scene_graph/weights/yolact_plus_resnet50_54_800000.pth --score_threshold=0.85 --top_k=15
"""
use_yolact = True
args = get_parser(use_yolact=use_yolact)

data_folder = '/home/yhkim/python_projects/robot_home_service/scene_graph_data/' + args.scene
intrinsic_folder = 'intrinsic'
intrinsic_file = 'intrinsic_depth.txt'
root_path = '/home/yhkim/python_projects/robot_home_service/robot_home_service/scene_graph'

if args.config is not None:
    set_cfg(args.config)
else:
    model_path = SavePath.from_str(args.trained_model)
    args.config = model_path.model_name + '_config'
    print('Config not specified. Parsed %s from the file name.\n' % args.config)
    set_cfg(args.config)

if args.detect:
    cfg.eval_mask_branch = False

if args.dataset is not None:
    set_dataset(args.dataset)

with torch.no_grad():
    if not os.path.exists('results'):
        os.makedirs('results')

    if args.cuda:
        cudnn.fastest = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    if args.resume and not args.display:
        with open(args.ap_data_file, 'rb') as f:
            ap_data = pickle.load(f)
        calc_map(ap_data)
        exit()

    dataset = None

    print('Loading model...', end='')
    net = Yolact()
    net.load_weights(args.trained_model)
    net.eval()
    print(' Done.')

    if args.cuda:
        net = net.cuda()

    net.detect.use_fast_nms = args.fast_nms
    net.detect.use_cross_class_nms = args.cross_class_nms
    cfg.mask_proto_debug = args.mask_proto_debug

mesh_plot = True
use_gpu = True
scannet_data = args.is_scannet
skip_imgs = 1

task_sampler_params = HomeServiceBaseExperimentConfig.stagewise_task_sampler_args(
    stage="train", process_ind=0, total_processes=1, headless=False
)
task_sampler: HomeServiceTaskSampler = HomeServiceBaseExperimentConfig.make_sampler_fn(
    **task_sampler_params,
    task_type=HomeServiceTaskType.SIMPLE_PICK_AND_PLACE,
    force_cache_reset=True,
    epochs=1,
)

num_tasks = 500
success = 0
app = QApplication(sys.argv)

for i in range(num_tasks):
    print(f'{i}-th task')
    task = task_sampler.next_task()
    print(f'  task: {task.env.current_task_spec.task_type}')
    # print(f'  pickup_target: {task.env.current_task_spec.pickup_target}')
    # print(f'  place_target: {task.env.current_task_spec.place_target}')

    # rgb = task.env.last_event.frame
    # depth = task.env.last_event.depth_frame

    scan = Scan(cam_intr=np.eye(3), mesh_plot=mesh_plot, scannet_data=scannet_data, mask_net=net, args=args, root_path=root_path, use_gpu=use_gpu)
    vis = ScannetVis(scan=scan, task=task, offset=0, skip_im=skip_imgs, mesh_plot=mesh_plot, parent=None)

    while not task.is_done():
        obs = task.get_observations()
        # depth = obs['depth'].copy()
        # depth[depth > 1.5] = 0

        # max_depth = np.max(depth)
        # multiplier = np.tan(np.deg2rad(90 / 2.0))
        # band = max_depth * multiplier
        # c_vertices = np.array(
        #     [
        #         [0., -band, -band, band, band],
        #         [0., -band, band, -band, band],
        #         [0., max_depth, max_depth, max_depth, max_depth]
        #     ]
        # )

        # agent_y = task.env.last_event.metadata['agent']['position']['y']
        # camera_locs = np.array(
        #     [
        #         obs['rel_position_change']['agent_locs'][0], 
        #         agent_y + 0.675 * obs['rel_position_change']['agent_locs'][-2], 
        #         obs['rel_position_change']['agent_locs'][1],
        #         obs['rel_position_change']['agent_locs'][2],
        #         obs['rel_position_change']['agent_locs'][-1]]
        # )
        # view_frust_pts = get_view_frustum(depth, camera_locs, fov=90)
        # cam_pose = get_camera_pose(camera_locs[:3], camera_locs[-2], camera_locs[-1])
        # import pdb; pdb.set_trace()

        # action_ind = task.action_space.sample()
        # action_ind, _ = task.query_expert()
        # import pdb; pdb.set_trace()
        vis.update_scan()
        # import pdb; pdb.set_trace()
        action_ind = int(input(f"action_ind="))

        # if task.num_steps_taken() % 10 == 0:
        #     print(
        #         f'step: {task.num_steps_taken()}:'
        #         f' taking action {task.action_names()[action_ind]}'
        #     )
        step_result = task.step(action=action_ind)
        # task.greedy_expert.update(
        #     action_taken=action_ind,
        #     action_success=step_result.info['action_success']
        # )

        if step_result.info['action_name'] == "done":
            success += 1
        
        # if task.current_subtask[0] == "Done":
        #     print(f"All subtasks DONE")
        #     import pdb; pdb.set_trace()
            
    print(f'{i}-th task done')

task_sampler.close()

print(f'finishied {num_tasks} tasks')
print(f'Success {success} out of {num_tasks}')

