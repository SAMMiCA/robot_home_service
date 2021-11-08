import argparse
import numpy as np
from typing_extensions import OrderedDict
from allenact.embodiedai.mapping.mapping_utils.point_cloud_utils import camera_space_xyz_to_world_xyz
import torch

from scene_graph.fusion_whole import rigid_transform
from ..data import cfg

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def calc_map(ap_data, iou_thresholds):
    print('Calculating mAP...')
    aps = [{'box': [], 'mask': []} for _ in iou_thresholds]

    for _class in range(len(cfg.dataset.class_names)):
        for iou_idx in range(len(iou_thresholds)):
            for iou_type in ('box', 'mask'):
                ap_obj = ap_data[iou_type][iou_idx][_class]

                if not ap_obj.is_empty():
                    aps[iou_idx][iou_type].append(ap_obj.get_ap())

    all_maps = {'box': OrderedDict(), 'mask': OrderedDict()}

    # Looking back at it, this code is really hard to read :/
    for iou_type in ('box', 'mask'):
        all_maps[iou_type]['all'] = 0  # Make this first in the ordereddict
        for i, threshold in enumerate(iou_thresholds):
            mAP = sum(aps[i][iou_type]) / len(aps[i][iou_type]) * 100 if len(aps[i][iou_type]) > 0 else 0
            all_maps[iou_type][int(threshold * 100)] = mAP
        all_maps[iou_type]['all'] = (sum(all_maps[iou_type].values()) / (len(all_maps[iou_type].values()) - 1))

    print_maps(all_maps)

    # Put in a prettier format so we can serialize it to json during training
    all_maps = {k: {j: round(u, 2) for j, u in v.items()} for k, v in all_maps.items()}
    return all_maps

def print_maps(all_maps):
    # Warning: hacky
    make_row = lambda vals: (' %5s |' * len(vals)) % tuple(vals)
    make_sep = lambda n:  ('-------+' * n)

    print()
    print(make_row([''] + [('.%d ' % x if isinstance(x, int) else x + ' ') for x in all_maps['box'].keys()]))
    print(make_sep(len(all_maps['box']) + 1))
    for iou_type in ('box', 'mask'):
        print(make_row([iou_type] + ['%.2f' % x if x < 100 else '%.1f' % x for x in all_maps[iou_type].values()]))
    print(make_sep(len(all_maps['box']) + 1))
    print()

# def get_view_frustum(depth_img: np.ndarray, camera_locs: np.ndarray, fov: float = 90):
#     """
#     depth_img: (H, W)
#     camera_locs: (cam_x, cam_y, cam_z, cam_rot, cam_horizon) in the world coordinate.
#     """
#     max_depth = np.max(depth_img)
#     multiplier = np.tan(np.deg2rad(fov / 2.0))
#     band = max_depth * multiplier
#     c_vertices = np.array(
#         [
#             [-band, -band, -band, band, band],
#             [-band, -band, band, -band, band],
#             [0., max_depth, max_depth, max_depth, max_depth]
#         ]
#     )

#     camera_xyz = -camera_locs[:3]
#     camera_rotation = camera_locs[3]
#     camera_horizon = camera_locs[4]
    
#     return camera_space_xyz_to_world_xyz(
#         camera_space_xyzs=torch.from_numpy(c_vertices),
#         camera_world_xyz=torch.from_numpy(camera_xyz),
#         rotation=camera_rotation,
#         horizon=camera_horizon,
#     ).cpu().numpy()
def get_view_frustum(depth_img: np.ndarray, camera_locs: np.ndarray, fov: float = 90):
    """
    depth_img: (H, W)
    camera_locs: (cam_x, cam_y, cam_z, cam_rot, cam_horizon) in the world coordinate.
    """
    max_depth = np.max(depth_img)
    multiplier = np.tan(np.deg2rad(fov / 2.0))
    band = max_depth * multiplier
    c_vertices = np.array(
        [
            [0, -band, -band, band, band],
            [0, -band, band, -band, band],
            [0., max_depth, max_depth, max_depth, max_depth]
        ]
    )
    cam_pose = get_camera_pose(camera_locs[:3], camera_locs[-2], camera_locs[-1])
    return rigid_transform(c_vertices.T, cam_pose).T

def get_camera_pose(
    camera_world_xyz: np.ndarray,
    camera_rotation: float,
    camera_horizon: float,
) -> torch.Tensor:
    psi = np.deg2rad(camera_horizon)
    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)
    horizon_transform = np.array(
        [
            [1, 0, 0],
            [0, cos_psi, sin_psi],
            [0, -sin_psi, cos_psi],
        ],
    )

    phi = np.deg2rad(-camera_rotation)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    rotation_transform = np.array(
        [
            [cos_phi, 0, -sin_phi],
            [0, 1, 0],
            [sin_phi, 0, cos_phi],
        ],
    )
    camera_world_xyz[1] *= -1
    return np.vstack(
        (
            np.column_stack(
                (
                    rotation_transform @ horizon_transform,
                    camera_world_xyz,
                )
            ),
            np.array(
                [0, 0, 0, 1]
            )
        )
    )



def get_parser(use_yolact: bool = True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, help='ScanNet Scene Number (scene0000, 0005, 0007, 0010, 0017)',
                        default='0000')
    parser.add_argument('--is_scannet', dest='is_scannet', default=False, action='store_true', help='True if it is scannet dataset')
    if use_yolact:
        parser.add_argument('--trained_model',
                            default='weights/ssd300_mAP_77.43_v2.pth', type=str,
                            help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
        parser.add_argument('--top_k', default=5, type=int,
                            help='Further restrict the number of predictions to parse')
        parser.add_argument('--cuda', default=True, type=str2bool,
                            help='Use cuda to evaulate model')
        parser.add_argument('--fast_nms', default=True, type=str2bool,
                            help='Whether to use a faster, but not entirely correct version of NMS.')
        parser.add_argument('--cross_class_nms', default=False, type=str2bool,
                            help='Whether compute NMS cross-class or per-class.')
        parser.add_argument('--display_masks', default=True, type=str2bool,
                            help='Whether or not to display masks over bounding boxes')
        parser.add_argument('--display_bboxes', default=True, type=str2bool,
                            help='Whether or not to display bboxes around masks')
        parser.add_argument('--display_text', default=True, type=str2bool,
                            help='Whether or not to display text (class [score])')
        parser.add_argument('--display_scores', default=True, type=str2bool,
                            help='Whether or not to display scores in addition to classes')
        parser.add_argument('--display', dest='display', action='store_true',
                            help='Display qualitative results instead of quantitative ones.')
        parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                            help='Shuffles the images when displaying them. Doesn\'t have much of an effect when display is off though.')
        parser.add_argument('--ap_data_file', default='results/ap_data.pkl', type=str,
                            help='In quantitative mode, the file to save detections before calculating mAP.')
        parser.add_argument('--resume', dest='resume', action='store_true',
                            help='If display not set, this resumes mAP calculations from the ap_data_file.')
        parser.add_argument('--max_images', default=-1, type=int,
                            help='The maximum number of images from the dataset to consider. Use -1 for all.')
        parser.add_argument('--output_coco_json', dest='output_coco_json', action='store_true',
                            help='If display is not set, instead of processing IoU values, this just dumps detections into the coco json file.')
        parser.add_argument('--bbox_det_file', default='results/bbox_detections.json', type=str,
                            help='The output file for coco bbox results if --coco_results is set.')
        parser.add_argument('--mask_det_file', default='results/mask_detections.json', type=str,
                            help='The output file for coco mask results if --coco_results is set.')
        parser.add_argument('--config', default=None,
                            help='The config object to use.')
        parser.add_argument('--output_web_json', dest='output_web_json', action='store_true',
                            help='If display is not set, instead of processing IoU values, this dumps detections for usage with the detections viewer web thingy.')
        parser.add_argument('--web_det_path', default='web/dets/', type=str,
                            help='If output_web_json is set, this is the path to dump detections into.')
        parser.add_argument('--no_bar', dest='no_bar', action='store_true',
                            help='Do not output the status bar. This is useful for when piping to a file.')
        parser.add_argument('--display_lincomb', default=False, type=str2bool,
                            help='If the config uses lincomb masks, output a visualization of how those masks are created.')
        parser.add_argument('--benchmark', default=False, dest='benchmark', action='store_true',
                            help='Equivalent to running display mode but without displaying an image.')
        parser.add_argument('--no_sort', default=False, dest='no_sort', action='store_true',
                            help='Do not sort images by hashed image ID.')
        parser.add_argument('--seed', default=None, type=int,
                            help='The seed to pass into random.seed. Note: this is only really for the shuffle and does not (I think) affect cuda stuff.')
        parser.add_argument('--mask_proto_debug', default=False, dest='mask_proto_debug', action='store_true',
                            help='Outputs stuff for scripts/compute_mask.py.')
        parser.add_argument('--no_crop', default=False, dest='crop', action='store_false',
                            help='Do not crop output masks with the predicted bounding box.')
        parser.add_argument('--image', default=None, type=str,
                            help='A path to an image to use for display.')
        parser.add_argument('--images', default=None, type=str,
                            help='An input folder of images and output folder to save detected images. Should be in the format input->output.')
        parser.add_argument('--video', default=None, type=str,
                            help='A path to a video to evaluate on. Passing in a number will use that index webcam.')
        parser.add_argument('--video_multiframe', default=1, type=int,
                            help='The number of frames to evaluate in parallel to make videos play at higher fps.')
        parser.add_argument('--score_threshold', default=0, type=float,
                            help='Detections with a score under this threshold will not be considered. This currently only works in display mode.')
        parser.add_argument('--dataset', default=None, type=str,
                            help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
        parser.add_argument('--detect', default=False, dest='detect', action='store_true',
                            help='Don\'t evauluate the mask branch at all and only do object detection. This only works for --display and --benchmark.')
        parser.add_argument('--display_fps', default=False, dest='display_fps', action='store_true',
                            help='When displaying / saving video, draw the FPS on the frame')
        parser.add_argument('--emulate_playback', default=False, dest='emulate_playback', action='store_true',
                            help='When saving a video, emulate the framerate that you\'d get running in real-time mode.')
        parser.set_defaults(no_bar=False, display=False, resume=False, output_coco_json=False, output_web_json=False, shuffle=False,
                            benchmark=False, no_sort=False, no_hash=False, mask_proto_debug=False, crop=True, detect=False, display_fps=False,
                            emulate_playback=False)

    args = parser.parse_args()
    return args