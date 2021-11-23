import enum
from typing import Any, Optional, Sequence, Union
import torch

import gym.spaces
import numpy as np
from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import Task
from scene_graph.layers.output_utils import postprocess
from scene_graph.utils.augmentations import FastBaseTransform

from scene_graph.utils.functions import SavePath
from scene_graph.yolact import Yolact
from scene_graph.data import Config, cfg, set_cfg, set_dataset

try:
    from allenact.embodiedai.sensors.vision_sensors import RGBSensor
except ImportError:
    raise ImportError("Please update to allenact>=0.4.0.")

from allenact.embodiedai.sensors.vision_sensors import DepthSensor
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact_plugins.ithor_plugin.ithor_util import include_object_data, round_to_factor

from env.constants import SCENE_TO_SCENE_TYPE, SCENE_TYPE_TO_LABEL, STEP_SIZE
from env.environment import HomeServiceTHOREnvironment
from env.tasks import (
    HomeServiceBaseTask,
    AbstractHomeServiceTask,
)



class RGBHomeServiceSensor(
    RGBSensor[HomeServiceTHOREnvironment, HomeServiceBaseTask]
):
    def frame_from_env(
        self, env: HomeServiceTHOREnvironment, task: HomeServiceBaseTask
    ) -> np.ndarray:
        if isinstance(task, HomeServiceBaseTask):
            return task.env.last_event.frame.copy()
        else:
            raise NotImplementedError(
                f"Unknown task type {type(task)}, must be an `HomeServiceBaseTask`."
            )


class DepthHomeServiceSensor(
    DepthSensor[HomeServiceTHOREnvironment, HomeServiceBaseTask]
):
    def frame_from_env(
        self, env: HomeServiceTHOREnvironment, task: HomeServiceBaseTask
    ) -> np.ndarray:
        if isinstance(task, HomeServiceBaseTask):
            return task.env.controller.last_event.depth_frame.copy()
        else:
            raise NotImplementedError(
                f"Unknown task type {type(task)}, must be an `HomeServiceBaseTask`."
            )


class SubtaskType(enum.Enum):
        Done = 0
        Goto = 1
        Scan = 2
        Navigate = 3
        Pickup = 4
        Put = 5
        Open = 6
        Close = 7       


class SubtaskHomeServiceSensor(
    Sensor[HomeServiceTHOREnvironment, HomeServiceBaseTask]
):

    def __init__(
        self, 
        object_types: Sequence[str],
        uuid: str = "subtask",
         **kwargs: Any
    ):
        self.ordered_object_types = list(object_types)
        assert self.ordered_object_types == sorted(
            self.ordered_object_types
        ), "object types input to goal object type sensor must be ordered"
        self.object_type_to_ind = {
            ot: i for i, ot in enumerate(self.ordered_object_types)
        }

        observation_space = gym.spaces.Dict(
            {
                "task_type": gym.spaces.Discrete(6),
                "target_type": gym.spaces.Discrete(1 + 4 + len(self.ordered_object_types)),
                "target_position": gym.spaces.Box(
                    low=np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32),
                    high=np.array([np.inf, np.inf, np.inf], dtype=np.float32),
                    shape=(3,),
                    dtype=np.float32,
                ),
                # "target_visible": gym.spaces.Box(low=0, high=1, shape=(1, ), dtype=np.bool),
                "task_type": gym.spaces.Discrete(3),
                "place_type": gym.spaces.Discrete(1 + len(self.ordered_object_types)),
                "place_position": gym.spaces.Box(
                    low=np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32),
                    high=np.array([np.inf, np.inf, np.inf], dtype=np.float32),
                    shape=(3,),
                    dtype=np.float32,
                ),
                # "place_visible": gym.spaces.Box(low=0, high=1, shape=(1, ), dtype=np.bool),
                "place_visible": gym.spaces.Discrete(3),
            }
        )

        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self,
        env: HomeServiceTHOREnvironment,
        task: Optional[Task[HomeServiceTHOREnvironment]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        # pickup_target = env.current_task_spec.pickup_target
        # place_target = env.current_task_spec.place_target
        current_subtask_action, current_subtask_target, current_subtask_place = task.current_subtask
        
        subtask_type = SubtaskType[current_subtask_action].value
        # if not isinstance(current_subtask_target, str) and current_subtask_target is not None:
        #     current_subtask_target = current_subtask_target["objectType"]
        # if not isinstance(current_subtask_place, str) and current_subtask_place is not None:
        #     current_subtask_place = current_subtask_place["objectType"]

        if current_subtask_action in ("Done", "Scan"):
            subtask_target_type = 0
            subtask_target_position = np.zeros(3, dtype=np.float32)
            subtask_target_visible = 0
            subtask_place_type = 0
            subtask_place_position = np.zeros(3, dtype=np.float32)
            subtask_place_visible = 0
        elif current_subtask_action == "Goto":
            subtask_target_type = SCENE_TYPE_TO_LABEL[current_subtask_target]
            subtask_target_position = np.array(list(env.current_task_spec.agent_positions[current_subtask_target].values())).astype(np.float32)
            subtask_target_visible = 0
            subtask_place_type = 0
            subtask_place_position = np.zeros(3, dtype=np.float32)
            subtask_place_visible = 0
        elif current_subtask_action in ("Navigate", "Pickup", "Open", "Close"):
            assert env.scene == env.current_task_spec.target_scene

            # from metadata
            subtask_target = next(
                (o for o in env.last_event.metadata['objects'] if o['objectType'] == current_subtask_target), None
            )
            # assert subtask_target is not None, f"subtask: {current_subtask_action}, subtask_target: {current_subtask_target}"
            # subtask_target_type = self.object_type_to_ind[subtask_target['objectType']] + 1 + 4
            # subtask_target_position = np.array(list(subtask_target["position"].values()), dtype=np.float32)
            # subtask_target_visible = subtask_target["visible"] + 1
            subtask_target_type = self.object_type_to_ind[current_subtask_target] + 1 + 4
            subtask_target_position = task.target_positions[current_subtask_target]
            subtask_target_visible = subtask_target["visible"] + 1 # Should be modified

            subtask_place_type = 0
            subtask_place_position = np.zeros(3, dtype=np.float32)
            subtask_place_visible = 0

        elif current_subtask_action == "Put":
            assert env.scene == env.current_task_spec.target_scene
            assert env.held_object is not None
            assert current_subtask_target == env.held_object["objectType"]
            subtask_target_type = self.object_type_to_ind[current_subtask_target] + 1 + 4
            subtask_target_position = np.zeros(3, dtype=np.float32)
            subtask_target_visible = 0

            # from metadata
            subtask_place = next(
                (o for o in env.last_event.metadata['objects'] if o['objectType'] == current_subtask_place), None
            )
            # assert subtask_place is not None
            # subtask_place_type = self.object_type_to_ind[subtask_place['objectType']] + 1
            # subtask_place_position = np.array(list(subtask_place["position"].values()), dtype=np.float32)
            subtask_place_type = self.object_type_to_ind[current_subtask_place] + 1 + 4
            subtask_place_position = task.target_positions[current_subtask_place]
            subtask_place_visible = subtask_place["visible"] + 1 # Should be modified
        else:
            raise RuntimeError()

        return {
            "type": subtask_type,
            "target_type": subtask_target_type,
            "target_position": subtask_target_position,
            "target_visible": subtask_target_visible,
            "place_type": subtask_place_type,
            "place_position": subtask_place_position,
            "place_visible": subtask_place_visible,
        }


class RelativePositionChangeSensor(
    Sensor[HomeServiceTHOREnvironment, HomeServiceBaseTask]
):
    def __init__(self, uuid: str = "rel_position_change", base: int = 90, **kwargs: Any):
        observation_space = gym.spaces.Dict(
            {
                "last_allocentric_position": gym.spaces.Box(
                    low=np.array([-np.inf, -np.inf, 0], dtype=np.float32),
                    high=np.array([np.inf, np.inf, 360], dtype=np.float32),
                    shape=(3,),
                    dtype=np.float32,
                ),
                "dx_dz_dr": gym.spaces.Box(
                    low=np.array([-np.inf, -np.inf, -360], dtype=np.float32),
                    high=np.array([-np.inf, -np.inf, 360], dtype=np.float32),
                    shape=(3,),
                    dtype=np.float32,
                ),
            }
        )
        super().__init__(**prepare_locals_for_super(locals()))

        self.last_xzr: Optional[np.ndarray] = None
        self.init_xzr: Optional[np.ndarray] = None
        self.should_init = True
        self.base = base

    @staticmethod
    def get_relative_position_change(from_xzr: np.ndarray, to_xzr: np.ndarray, base: int):
        dx_dz_dr = to_xzr - from_xzr

        # Transform dx, dz (in global coordinates) into the relative coordinates
        # given by rotation r0=from_xzr[-2]. This requires rotating everything so that
        # r0 is facing in the positive z direction. Since thor rotations are negative
        # the usual rotation direction this means we want to rotate by r0 degrees.
        theta = np.pi * from_xzr[-1] / 180
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        dx_dz_dr = (
            np.array(
                [
                    [cos_theta, -sin_theta, 0],
                    [sin_theta, cos_theta, 0],
                    [0, 0, 1],  # Don't change dr
                ]
            )
            @ dx_dz_dr.reshape(-1, 1)
        ).reshape(-1)

        dx_dz_dr[:2] = np.round(dx_dz_dr[:2], 2)
        dx_dz_dr[-1] = round_to_factor(dx_dz_dr[-1] % 360, base)
        return dx_dz_dr

    def get_observation(
        self,
        env: HomeServiceTHOREnvironment,
        task: Optional[Task[HomeServiceTHOREnvironment]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        p = env.controller.last_event.metadata["agent"]["position"]
        r = env.controller.last_event.metadata["agent"]["rotation"]["y"]

        if task.is_done():
            self.should_init = True

        if task.require_init_position_sensor:
            self.should_init = True
            task.require_init_position_sensor = False

        if task.num_steps_taken() == 0 or self.should_init:
            self.last_xzr = np.array([p["x"], p["z"], r % 360])
            self.last_xzr[:2] = np.round(self.last_xzr[:2], 2)
            self.last_xzr[-1] = round_to_factor(self.last_xzr[-1] % 360, self.base)
            self.init_xzr = self.last_xzr.copy()
            self.should_init = False

        current_xzr = np.array([p["x"], p["z"], r % 360])
        current_xzr[:2] = np.round(current_xzr[:2], 2)
        current_xzr[-1] = round_to_factor(current_xzr[-1] % 360, self.base)

        last_dx_dz_dr = self.get_relative_position_change(
            from_xzr=self.init_xzr, to_xzr=self.last_xzr, base=self.base,
        )
        dx_dz_dr = self.get_relative_position_change(
            from_xzr=self.last_xzr, to_xzr=current_xzr, base=self.base,
        )
        cum_dx_dz_dr = self.get_relative_position_change(to_xzr=current_xzr, from_xzr=self.init_xzr, base=self.base)
        std_hz = np.array([env.get_agent_location()['standing'], env.get_agent_location()['horizon']])
        agent_locs = np.append(cum_dx_dz_dr, std_hz)

        to_return = {"last_allocentric_position": self.last_xzr.astype(np.float32),
                     "allocentric_position": current_xzr.astype(np.float32),
                     "last_dx_dz_dr":last_dx_dz_dr.astype(np.float32),
                     "cum_dx_dz_dr": cum_dx_dz_dr.astype(np.float32),
                     "dx_dz_dr": dx_dz_dr.astype(np.float32),
                     "agent_locs": agent_locs.astype(np.float32)}

        self.last_xzr = current_xzr

        return to_return


class InstanceSegmentationSensor(
    Sensor[HomeServiceTHOREnvironment, HomeServiceBaseTask]
):
    def __init__(
        self,
        ordered_object_types: Sequence[str] = None,
        uuid: str = "instance_segmentation",
        **kwargs: Any
    ):
        self.ordered_object_types = list(ordered_object_types)
        assert self.ordered_object_types == sorted(self.ordered_object_types)

        self.object_type_to_idx = {ot: i for i, ot in enumerate(self.ordered_object_types)}
        self.idn_to_object_type = {i: ot for i, ot in enumerate(self.ordered_object_types)}

        observation_space = gym.spaces.Space()
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self,
        env: HomeServiceTHOREnvironment,
        task: Optional[Task[HomeServiceTHOREnvironment]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        e = env.last_event
        rgb = e.frame.copy()
        inst_seg_frame = e.instance_segmentation_frame
        inst_mask = []
        inst_bbox = []
        inst_label = []
        inst_detected = np.zeros(len(self.ordered_object_types), dtype=np.int32)

        det_objs = [obj for obj in e.instance_masks if obj.split('|')[0] in self.ordered_object_types]
        for obj in det_objs:
            obj_type = obj.split("|")[0]
            obj_type_idx = self.object_type_to_idx[obj_type]
            rgb[e.instance_masks[obj]] = inst_seg_frame[e.instance_masks[obj]]
            inst_label.append(obj_type_idx)
            inst_mask.append(e.instance_masks[obj])
            inst_bbox.append(e.instance_detections2D[obj])
            inst_detected[obj_type_idx] += 1
            if obj_type in ("Sink", "Bathtub"):
                add_ = True
                with include_object_data(env.controller):
                    objs = env.last_event.metadata["objects"]
                    basin_obj = next(
                        (
                            obj for obj in objs
                            if obj["objectType"] == f"{obj_type}Basin"
                        ), None
                    )
                    if basin_obj is None:
                        add_ = False
                if add_:
                    obj_type_idx = self.object_type_to_idx[obj_type + "Basin"]
                    inst_label.append(obj_type_idx)
                    inst_mask.append(e.instance_masks[obj])
                    inst_bbox.append(e.instance_detections2D[obj])
                    inst_detected[obj_type_idx] += 1

        return {
            'inst_seg_image': inst_seg_frame,
            'inst_seg_on_rgb': rgb,
            'inst_label': np.array(inst_label),
            'inst_mask': np.array(inst_mask),
            'inst_bbox': np.array(inst_bbox),
            'inst_detected': inst_detected,
        }

# class YolactObjectDetectionSensor(
#     Sensor[HomeServiceTHOREnvironment, HomeServiceBaseTask]
# ):
#     def __init__(
#         self,
#         ordered_object_types: Sequence[str] = None,
#         uuid: str = "yolact",
#         *args: Any,
#         **kwargs: Any,
#     ):

#         self.ordered_object_types = list(ordered_object_types)
#         assert self.ordered_object_types == sorted(self.ordered_object_types)
        
#         model_path = SavePath.from_str(kwargs["trained_model"])
#         config = model_path.model_name + "_config"
#         set_cfg(config)
#         set_dataset(kwargs["dataset"])
#         cfg.num_classes = len(cfg.dataset.class_names) + 1
#         assert len(self.ordered_object_types) == cfg.num_classes - 1
#         net = Yolact()
#         net.load_weights(kwargs["trained_model"])
#         net.eval()
#         self.cuda = False
#         if kwargs['cuda']:
#             net = net.cuda()
#             self.cuda = True

#         net.detect.use_fast_nms = kwargs['fast_nms']
#         net.detect.use_cross_class_nms = kwargs['cross_class_nms']
#         # cfg.mask_proto_debug = kwargs['mask_proto_debug']
#         self.net = net
        
#         self.object_type_to_idn = {ot: i for i, ot in enumerate(self.ordered_object_types)}
#         self.idn_to_object_type = {i: ot for i, ot in enumerate(self.ordered_object_types)}

#         observation_space = gym.spaces.Space()
#         super().__init__(**prepare_locals_for_super(locals()))

#     def get_observation(
#         self,
#         env: HomeServiceTHOREnvironment,
#         task: Optional[Task[HomeServiceTHOREnvironment]],
#         *args: Any,
#         **kwargs: Any,
#     ) -> Any:
#         e = env.last_event
#         rgb = e.frame.copy()
#         h, w, _ = rgb.shape
#         depth = e.depth_frame.copy()

#         rgb = torch.from_numpy(rgb).float()
#         if self.cuda:
#             rgb = rgb.cuda()
#         batch = FastBaseTransform()(rgb.unsqueeze(0))

#         with torch.no_grad():
#             preds = self.net(batch)
        
#         save = cfg.rescore_bbox
#         cfg.rescore_bbox = True
#         t = postprocess(preds, w, h, )

#         inst_seg_frame = e.instance_segmentation_frame
#         inst_mask = []
#         inst_bbox = []
#         inst_label = []
#         inst_detected = np.zeros(len(self.ordered_object_types), dtype=np.int32)

#         det_objs = [obj for obj in e.instance_masks if obj.split('|')[0] in self.ordered_object_types]
#         for obj in det_objs:
#             obj_type = obj.split("|")[0]
#             obj_type_idx = self.object_type_to_idn[obj_type]
#             rgb[e.instance_masks[obj]] = inst_seg_frame[e.instance_masks[obj]]
#             inst_label.append(obj_type_idx)
#             inst_mask.append(e.instance_masks[obj])
#             inst_bbox.append(e.instance_detections2D[obj])
#             inst_detected[obj_type_idx] += 1

#         return {
#             'inst_seg_image': inst_seg_frame,
#             'inst_seg_on_rgb': rgb,
#             'inst_label': np.array(inst_label),
#             'inst_mask': np.array(inst_mask),
#             'inst_bbox': np.array(inst_bbox),
#             'inst_detected': inst_detected,
#         }
