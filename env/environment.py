from collections import OrderedDict
import enum
import random
import math
import numpy as np
import pprint
import traceback
from typing import Dict, Any, Tuple, Optional, Callable, List, Union, Sequence
from packaging import version
from torch._C import set_autocast_enabled
from torch.distributions.utils import lazy_property

import ai2thor
import ai2thor.controller
import ai2thor.fifo_server
import ai2thor.server
import ai2thor.wsgi_server

from allenact.utils.system import get_logger
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment
from allenact_plugins.ithor_plugin.ithor_util import (
    round_to_factor,
    include_object_data,
)
from datagen.datagen_constants import OBJECT_TYPES_TO_NOT_MOVE
from datagen.datagen_utils import (
    open_objs,
    get_object_ids_to_not_move_from_object_types,
    remove_objects_until_all_have_identical_meshes,
)
from env.constants import (
    REQUIRED_THOR_VERSION,
    MAX_HAND_METERS,
    PICKUPABLE_OBJECTS,
    OPENABLE_OBJECTS,
    RECEPTACLE_OBJECTS
)
from env.utils import (
    BoundedFloat,
    HomeServiceActionSpace,
    ObjectInteractablePostionsCache,
    # PoseMismatchError,
    # ObjectInteractablePostionsCache,
    execute_action,
    # get_pose_info,
    # iou_box_3d,
)


class HomeServiceMode(enum.Enum):
    """Mode"""

    MANIPULATE = "MANIPULATE"
    SNAP = "SNAP"


class HomeServiceTaskSpec:
    """
    
    """

    def __init__(
        self,
        scene: str,
        stage: Optional[str] = None,
        agent_position: Optional[Dict[str, float]] = None,
        agent_rotation: Optional[float] = None,
        openable_data: Optional[Sequence[Dict[str, Any]]] = None,
        starting_poses: Optional[Sequence[Dict[str, Any]]] = None,
        target_poses: Optional[Sequence[Dict[str, Any]]] = None,
        runtime_sample: bool = False,
        runtime_data: Optional[Dict[str, Any]] = None,
        **metrics,
    ):
        """HomeServiceTaskSpec"""
        self.scene = scene
        self.stage = stage
        self.agent_position = agent_position
        self.agent_rotation = agent_rotation
        self.openable_data = openable_data
        self.starting_poses = starting_poses
        self.target_poses = target_poses
        self.runtime_sample = runtime_sample
        self.runtime_data: Dict[str, Any] = (
            runtime_data if runtime_data is not None else {}
        )
        
        self.metrics = metrics

    def __str__(self):
        """"""
        return pprint.pformat(self.__dict__)

    @property
    def unique_id(self):
        if self.runtime_sample:
            raise NotImplementedError("Cannot create a unique id for a runtime sample.")

        return f"{self.scene}__{self.stage}__{self.metrics['index']}"


class HomeServiceSimplePickAndPlaceTaskSpec(HomeServiceTaskSpec):
    def __init__(
        self,
        task_type: Optional[str] = None,
        **spec_kwargs,
    ):
        super().__init__(**spec_kwargs)
        self._pickup_target = None
        self._place_target = None

    @property
    def unique_id(self):
        return f"{super().unique_id}__{self.task_type}" if self.task_type is not None else super().unique_id

    @property
    def pickup_target(self):
        return self._pickup_target

    @pickup_target.setter
    def pickup_target(self, obj):
        self._pickup_target = obj

    @property
    def place_target(self):
        return self._place_target

    @place_target.setter
    def place_target(self, obj):
        self._place_target = obj

    @property
    def task_type(self):
        if self.pickup_target is not None and self.place_target is not None:
            return f"Pick_{self.pickup_target['objectType']}_And_Place_{self.place_target['objectType']}"
        return None


class HomeServiceTHOREnvironment:

    def __init__(
        self,
        mode: HomeServiceMode = HomeServiceMode.SNAP,
        force_cache_reset: Optional[bool] = None,
        controller_kwargs: Optional[Dict[str, Any]] = None,
        enhanced_physics_determinism: bool = True,
    ):

        if ai2thor.__version__ is not None:
            if ai2thor.__version__ not in ["0.0.1", None] and version.parse(
                ai2thor.__version__
            ) < version.parse(REQUIRED_THOR_VERSION):
                raise ImportError(
                    f"To run the rearrangment baseline experiments you must use"
                    f" ai2thor version {REQUIRED_THOR_VERSION} or higher."
                )

        # Saving attributes
        if mode == HomeServiceMode.SNAP:
            assert (
                force_cache_reset is not None
            ), "When in HomeServiceMode.SNAP mode you must specify a value for 'force_cache_reset'"
        else:
            force_cache_reset = force_cache_reset
        self.force_cache_reset = force_cache_reset
        self.mode = mode
        self._controller_kwargs = {} if controller_kwargs is None else controller_kwargs
        self._enhanced_physics_determinism = enhanced_physics_determinism

        self.physics_step_kwargs = {}
        if self._enhanced_physics_determinism:
            self.physics_step_kwargs = {
                "actionSimulationSeconds" : 0.26,
                "fixedDeltaTime": 0.02,
            }

        # Cache of where objects can be interacted with
        self._interactable_positions_cache = ObjectInteractablePostionsCache()

        # Object poses at start
        # Reset after every call to reset
        self.object_name_to_start_pose: Optional[Dict[str, Dict]] = None
        self._cached_poses: Optional[Tuple[List, List, List]] = None

        # Current task specification
        self.current_task_spec: Optional[HomeServiceTaskSpec] = None

        # Caches of starting object poses and other information
        # Reset on every call to reset
        self._sorted_and_extracted_start_poses: Optional[List] = None
        self._have_warned_about_mismatch = False
        self._agent_signals_done = False

        # instance masks now not supported. But an Exception would be thrown if
        # `mode == RearrangeMode.MANIPULATE` and render_instance_masks is True, since masks are
        # only available on RearrangeMode.SNAP mode.
        self._render_instance_masks: bool = False
        if self.mode == HomeServiceMode.MANIPULATE and self._render_instance_masks:
            raise Exception(
                "render_instance_masks is only available on HomeServiceMode.SNAP mode."
            )

        self.rotate_step_degrees = self._controller_kwargs.get("rotateStepDegrees", 90)
        self.horizon_step_degrees = self._controller_kwargs.get("horizonStepDegrees", 30)
        if "horizonStepDegrees" in self._controller_kwargs:
            del self._controller_kwargs["horizonStepDegrees"]
        self.grid_size = self._controller_kwargs.get("gridSize", 0.25)

        # local THOR controller to execute all the actions
        self.controller = self.create_controller()

    def create_controller(self):
        """Create the ai2thor controller."""

        assert ("width" in self._controller_kwargs) == (
            "height" in self._controller_kwargs
        ), "Either controller_kwargs must contain either both of width/height or neither."
        self._controller_kwargs["width"] = self._controller_kwargs.get("width", 300)
        self._controller_kwargs["height"] = self._controller_kwargs.get("height", 300)

        smoothing_factor = self._controller_kwargs.get("smoothingFactor", 1)

        self._controller_kwargs["rotateStepDegrees"] = self._controller_kwargs.get("rotateStepDegrees", 90) / smoothing_factor
        self.horizon_step_degrees = self.horizon_step_degrees / smoothing_factor
        self._controller_kwargs["gridSize"] = self._controller_kwargs.get("gridSize", 0.25) / smoothing_factor
        if "smoothingFactor" in self._controller_kwargs:
            del self._controller_kwargs["smoothingFactor"]

        controller = ai2thor.controller.Controller(
            **{
                "scene": "FloorPlan17_physics",
                "server_class": ai2thor.fifo_server.FifoServer,
                # "server_class": ai2thor.wsgi_server.WsgiServer,  # Possibly useful in debugging
                **self._controller_kwargs,
            },
        )
        return controller

    @property
    def held_object(self) -> Optional[Dict[str, Any]]:
        """Return the data corresponding to the object held by the agent (if any)."""
        with include_object_data(self.controller):
            metadata = self.controller.last_event.metadata

            if len(metadata["inventoryObjects"]) == 0:
                return None

            assert len(metadata["inventoryObjects"]) <= 1

            held_obj_id = metadata["inventoryObjects"][0]["objectId"]
            return next(o for o in metadata["objects"] if o["objectId"] == held_obj_id)

    @property
    def last_event(self) -> ai2thor.server.Event:
        return self.controller.last_event

    @property
    def scene(self) -> str:

        return self.controller.last_event.metadata["sceneName"].replace("_physics", "")
    
    def get_agent_location(self) -> Dict[str, Union[float, int, bool]]:

        metadata = self.controller.last_event.metadata
        return {
            "x": metadata["agent"]["position"]["x"],
            "y": metadata["agent"]["position"]["y"],
            "z": metadata["agent"]["position"]["z"],
            "rotation": metadata["agent"]["rotation"]["y"],
            "horizon": metadata["agent"]["cameraHorizon"],
            "standing": metadata.get("isStanding", metadata["agent"].get("isStanding")),
        }

    @property
    def observation(self) -> Tuple[np.array, Optional[np.array]]:

        rgb = self.last_event.frame
        depth = (
            self.last_event.depth_frame
            if hasattr(self.last_event, "depth_frame")
            else None
        )
        return rgb, depth

    @lazy_property
    def action_space(self) -> HomeServiceActionSpace:

        actions: Dict[Callable, Dict[str, BoundedFloat]] = {
            self.move_ahead: {},
            self.move_right: {},
            self.move_left: {},
            self.move_back: {},
            self.rotate_right: {},
            self.rotate_left: {},
            self.stand: {},
            self.crouch: {},
            self.look_up: {
                "angle": BoundedFloat(low=0, high=1),
            },
            self.look_down: {
                "angle": BoundedFloat(low=0, high=1),
            },
            self.done: {},
        }

        actions.update(
            {
                self.open_object: {
                    "x": BoundedFloat(low=0, high=1),
                    "y": BoundedFloat(low=0, high=1),
                    "openness": BoundedFloat(low=0, high=1),
                },
                self.pickup_object: {
                    "x": BoundedFloat(low=0, high=1),
                    "y": BoundedFloat(low=0, high=1),
                },
                self.push_object: {
                    "x": BoundedFloat(low=0, high=1),
                    "y": BoundedFloat(low=0, high=1),
                    "rel_x_force": BoundedFloat(low=-0.5, high=0.5),
                    "rel_y_force": BoundedFloat(low=-0.5, high=0.5),
                    "rel_z_force": BoundedFloat(low=-0.5, high=0.5),
                    "force_magnitude": BoundedFloat(low=0, high=1),
                },
                self.put_object: {
                    "x": BoundedFloat(low=0, high=1),
                    "y": BoundedFloat(low=0, high=1),
                },
                self.move_held_object: {
                    "x_meters": BoundedFloat(low=-0.5, high=0.5),
                    "y_meters": BoundedFloat(low=-0.5, high=0.5),
                    "z_meters": BoundedFloat(low=-0.5, high=0.5),
                },
                self.rotate_held_object: {
                    "x": BoundedFloat(low=-0.5, high=0.5),
                    "y": BoundedFloat(low=-0.5, high=0.5),
                    "z": BoundedFloat(low=-0.5, high=0.5),
                },
                self.drop_held_object: {},
            }
        )

        # if self.mode == HomeServiceMode.SNAP:
        #     actions.update(
        #         {
        #             self.drop_held_object_with_snap: {},
        #         }
        #     )

        return HomeServiceActionSpace(actions)

    def move_ahead(self) -> bool:
        
        return execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.move_ahead,
            thor_action="MoveAhead",
            default_thor_kwargs=self.physics_step_kwargs,
        )

    def move_back(self) -> bool:
        
        return execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.move_back,
            thor_action="MoveBack",
            default_thor_kwargs=self.physics_step_kwargs,
        )

    def move_right(self) -> bool:
        return execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.move_right,
            thor_action="MoveRight",
            default_thor_kwargs=self.physics_step_kwargs,
        )

    def move_left(self) -> bool:
        return execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.move_left,
            thor_action="MoveLeft",
            default_thor_kwargs=self.physics_step_kwargs,
        )

    def rotate_right(self) -> bool:
        return execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.rotate_right,
            thor_action="RotateRight",
            default_thor_kwargs=self.physics_step_kwargs,
        )

    def rotate_left(self) -> bool:
        return execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.rotate_left,
            thor_action="RotateLeft",
            default_thor_kwargs=self.physics_step_kwargs,
        )

    def stand(self) -> bool:
        return execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.stand,
            thor_action="Stand",
            default_thor_kwargs=self.physics_step_kwargs,
        )

    def crouch(self) -> bool:
        return execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.crouch,
            thor_action="Crouch",
            default_thor_kwargs=self.physics_step_kwargs,
        )

    def look_up(self, angle: float = 1) -> bool:
        def rescale_angle(kwargs: Dict[str, Any]):
            kwargs["degrees"] = self.horizon_step_degrees * kwargs["angle"]
            del kwargs["angle"]

        return execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.look_up,
            thor_action="LookUp",
            preprocess_kwargs_inplace=rescale_angle,
            angle=angle,
            default_thor_kwargs=self.physics_step_kwargs,
        )

    def look_down(self, angle: float = 1) -> bool:
        def rescale_angle(kwargs: Dict[str, Any]):
            kwargs["degrees"] = self.horizon_step_degrees * kwargs["angle"]
            del kwargs["angle"]

        return execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.look_down,
            thor_action="LookDown",
            preprocess_kwargs_inplace=rescale_angle,
            angle=angle,
            default_thor_kwargs=self.physics_step_kwargs,
        )
    
    def done(self) -> bool:
        self._agent_signals_done = True
        return execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.done,
            thor_action="Done",
        )

    def open_object(self, x: float, y: float, openness: float) -> bool:
        """Open the object corresponding to x/y to openness.

        The action will not be successful if the specified openness would
        cause a collision or if the object at x/y is not openable.

        # Parameters
        x : (float, min=0.0, max=1.0) horizontal percentage from the last frame
           that the target object is located.
        y : (float, min=0.0, max=1.0) vertical percentage from the last frame
           that the target object is located.

        # Returns
        `True` if the action was successful, otherwise `False`.
        """
        # If an object is already open, THOR doesn't support changing
        # it's openness without first closing it. So we simply try to first
        # close the object before reopening it.
        return execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.open_object,
            thor_action="OpenObject",
            error_message=(
                "x/y/openness must be in [0:1]."
            ),
            x=x,
            y=y,
            openness=openness,
            default_thor_kwargs=self.physics_step_kwargs,
        )

    def pickup_object(self, x: float, y: float) -> bool:
        if len(self.last_event.metadata["inventoryObjects"]) != 0:
            return False
        return execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.pickup_object,
            thor_action="PickupObject",
            error_message=(
                "x/y must be in [0:1]."
            ),
            x=x,
            y=y,
            default_thor_kwargs=self.physics_step_kwargs,
        )

    def push_object(
        self,
        x: float,
        y: float,
        rel_x_force: float,
        rel_y_force: float,
        rel_z_force: float,
        force_magnitude: float,
    ) -> bool:

        def preprocess_kwargs(kwargs: Dict[str, Any]):
            direction = {}
            for k in ["x", "y", "z"]:
                force_key = f"rel_{k}_force"
                direction[k] = kwargs[force_key]
                del kwargs[force_key]
            kwargs["direction"] = direction
            kwargs["force_magnitude"] = 50 * kwargs["force_magnitude"]

        return execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.push_object,
            thor_action="TouchThenApplyForce",
            error_message="Error in call to push object."
            " x, y, force_magnitude must be in [0:1],"
            " and rel_(x/y/z)_force must be in [-0.5:0.5]",
            default_thor_kwargs=self.physics_step_kwargs,
            preprocess_kwargs_inplace=preprocess_kwargs,
            x=x,
            y=y,
            rel_x_force=rel_x_force,
            rel_y_force=rel_y_force,
            rel_z_force=rel_z_force,
            moveMagnitude=force_magnitude,
        )

    def put_object(self, x: float, y: float) -> bool:
        if len(self.last_event.metadata["inventoryObjects"]) == 0:
            return False

        return execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.put_object,
            thor_action="PutObject",
            error_message=(
                "x/y must be in [0:1]."
            ),
            x=x,
            y=y,
            default_thor_kwargs=self.physics_step_kwargs,
        )

    def move_held_object(self, x_meters: float, y_meters: float, z_meters: float) -> bool:

        mag = math.sqrt(x_meters ** 2 + y_meters ** 2 + z_meters ** 2)

        if MAX_HAND_METERS > mag:
            scale = MAX_HAND_METERS / mag
            x_meters *= scale
            y_meters *= scale
            z_meters *= scale

        return execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.move_held_object,
            thor_action="MoveHandDelta",
            updated_kwarg_names={"x_meters": "x", "y_meters": "y", "z_meters": "z'"},
            x_meters=x_meters,
            y_meters=y_meters,
            z_meters=z_meters,
            default_thor_kwargs=self.physics_step_kwargs,
        )

    def rotate_held_object(self, x: float, y: float, z: float) -> bool:

        def rescale_xyz(kwargs: Dict[str, Any]):
            for k in ["x", "y", "z"]:
                kwargs[k] = 180 * kwargs[k]

        return execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.rotate_held_object,
            thor_action="RotateHand",
            preprocess_kwargs_inplace=rescale_xyz,
            x=x,
            y=y,
            z=z,
            default_thor_kwargs=self.physics_step_kwargs,
        )

    def drop_held_object(self) -> bool:

        return execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.drop_held_object,
            thor_action="DropHandObject",
            default_thor_kwargs={
                "autoSimulation": False,
                "randomMagnitude": 0.0,
                **self.physics_step_kwargs,
            },
        )

    # def drop_held_object_with_snap(self) -> bool:

    #     if not self.mode == HomeServiceMode.SNAP:
    #         raise Exception("Must be in HomeServiceMode.SNAP mode")

    #     DEC = 2

    #     with include_object_data(self.controller):
    #         event = self.controller.last_event
    #         held_obj = self.held_object

    #         if held_obj is None:
    #             return False

    #         self.controller.step(
    #             "MakeObjectBreakable", objectId=self.held_object["objectId"]
    #         )
    #     return execute_action(
    #         controller=self.controller,
    #         action_space=self.action_space,
    #         action_fn=self.,
    #         thor_action="",
    #         default_thor_kwargs=self.physics_step_kwargs,
    #     )

    # @staticmethod
    # def compare_poses(
    #     goal_pose: Union[Dict[str, Any], Sequence[Dict[str, Any]]],
    #     cur_pose: Union[Dict[str, Any], Sequence[Dict[str, Any]]],
    # ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:

    #     if isinstance(goal_pose, Sequence):
    #         assert isinstance(cur_pose, Sequence)
    #         return [
    #             HomeServiceTHOREnvironment.compare_poses(goal_pose=gp, cur_pose=cp)
    #             for gp, cp in zip(goal_pose, cur_pose)
    #         ]

    #     assert goal_pose["type"] == cur_pose["type"]
    #     assert not goal_pose["broken"]

    #     if cur_pose["broken"]:
    #         return {
    #             "broken": True,
    #             "iou": None,
    #             "openness_diff": None,
    #             "position_dist": None,
    #             "rotation_dist": None,
    #         }

    #     if goal_pose["bounding_box"] is None and cur_pose["bounding_box"] is None:
    #         iou = None
    #         position_dist = None
    #         rotation_dist = None
    #     else:
    #         position_dist = IThorEnvironment.position_dist(
    #             goal_pose["position"], cur_pose["position"]
    #         )
    #         rotation_dist = IThorEnvironment.angle_between_rotations(
    #             goal_pose["rotation"], cur_pose["rotation"]
    #         )
    #         if position_dist < 1e-2 and rotation_dist < 10.0:
    #             iou = 1.0
    #         else:
    #             try:
    #                 iou = iou_box_3d(
    #                     goal_pose["bounding_box"], cur_pose["bounding_box"]
    #                 )
    #             except Exception as _:
    #                 get_logger().warning(
    #                     "Could not compute IOU, will assume it was 0. Error during IOU computation:"
    #                     f"\n{traceback.format_exc()}"
    #                 )
    #                 iou = 0

    #     if goal_pose["openness"] is None and cur_pose["openness"] is None:
    #         openness_diff = None
    #     else:
    #         openness_diff = abs(goal_pose["openness"] - cur_pose["openness"])

    #     return {
    #         "broken": False,
    #         "iou": iou,
    #         "openness_diff": openness_diff,
    #         "position_dist": position_dist,
    #         "rotation_dist": rotation_dist,
    #     }
    
    # @classmethod
    # def pose_difference_energy(
    #     cls,
    #     goal_pose: Union[Dict[str, Any], Sequence[Dict[str, Any]]],
    #     cur_pose: Union[Dict[str, Any], Sequence[Dict[str, Any]]],
    #     min_iou: float = 0.5,
    #     open_tol: float = 0.2,
    #     pos_barrier: float = 2.0,
    # ) -> Union[float, np.ndarray]:
    #     if isinstance(goal_pose, Sequence):
    #         assert isinstance(cur_pose, Sequence)
    #         return np.array(
    #             [
    #                 cls.pose_difference_energy(
    #                     goal_pose=p0,
    #                     cur_pose=p1,
    #                     min_iou=min_iou,
    #                     open_tol=open_tol,
    #                     pos_barrier=pos_barrier,
    #                 )
    #                 for p0, p1 in zip(goal_pose, cur_pose)
    #             ]
    #         )
    #     assert not goal_pose["broken"]

    #     pose_diff = cls.compare_poses(goal_pose=goal_pose, cur_pose=cur_pose)
    #     if pose_diff["broken"]:
    #         return 1.0

    #     if pose_diff["openness_diff"] is None:
    #         gbb = np.array(goal_pose["bounding_box"])
    #         cbb = np.array(cur_pose["bounding_box"])

    #         iou = pose_diff["iou"]
    #         iou_energy = max(1 - iou / min_iou, 0)

    #         if iou > 0:
    #             position_dist_energy = 0.0
    #         else:
    #             min_pairwise_dist_between_corners = np.sqrt(
    #                 (
    #                     (
    #                         np.tile(gbb, (1, 8)).reshape(-1, 3)
    #                         - np.tile(cbb, (8, 1)).reshape(-1, 3)
    #                     )
    #                     ** 2
    #                 ).sum(1)
    #             ).min()
    #             position_dist_energy = min(
    #                 min_pairwise_dist_between_corners / pos_barrier, 1.0
    #             )

    #         return 0.5 * iou_energy + 0.5 * position_dist_energy

    #     else:
    #         return 1.0 * (pose_diff["openness_diff"] > open_tol)

    # @classmethod
    # def are_poses_equal(
    #     cls,
    #     goal_pose: Union[Dict[str, Any], Sequence[Dict[str, Any]]],
    #     cur_pose: Union[Dict[str, Any], Sequence[Dict[str, Any]]],
    #     min_iou: float = 0.5,
    #     open_tol: float = 0.2,
    #     treat_broken_as_unequal: bool = False,
    # ) -> Union[bool, np.ndarray]:
    #     if isinstance(goal_pose, Sequence):
    #         assert isinstance(cur_pose, Sequence)
    #         return np.array(
    #             [
    #                 cls.are_poses_equal(
    #                     goal_pose=p0,
    #                     cur_pose=p1,
    #                     min_iou=min_iou,
    #                     open_tol=open_tol,
    #                     treat_broken_as_unequal=treat_broken_as_unequal,
    #                 )
    #                 for p0, p1 in zip(goal_pose, cur_pose)
    #             ]
    #         )
    #     assert not goal_pose["broken"]

    #     if cur_pose["broken"]:
    #         if treat_broken_as_unequal:
    #             return False
    #         else:
    #             raise RuntimeError(
    #                 f"Cannot determine if poses of two objects are"
    #                 f" equal if one is broken object ({goal_pose} v.s. {cur_pose})."
    #             )

    #     pose_diff = cls.compare_poses(goal_pose=goal_pose, cur_pose=cur_pose)

    #     return (pose_diff["iou"] is None or pose_diff["iou"] > min_iou) and (
    #         pose_diff["openness_diff"] is None or pose_diff["openness_diff"] <= open_tol
    #     )

    @property
    def all_rearranged(self) -> bool:
        pass

    @property
    def poses(
        self,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        pass

    def _runtime_reset(
        self, task_spec: HomeServiceTaskSpec, force_axis_aligned_start: bool
    ):
        assert (
            task_spec.runtime_sample
        ), "Attempted to use a runtime reset with a task spec which has a `False` `runtime_sample` property."

        if (
            task_spec.scene != self.scene
            or self.current_task_spec.runtime_data["count"] >= 25
        ):
            count = 1
            self.controller.reset(task_spec.scene)

            if self._enhanced_physics_determinism:
                self.controller.step("PausePhysicsAutoSim")

            remove_objects_until_all_have_identical_meshes(self.controller)
            self.controller.step(
                "InitialRandomSpawn", forceVisible=True, placeStationary=True,
            )
            md = self.controller.step("GetReachablePositions").metadata
            assert md["lastActionSuccess"]
            reachable_positions = md["actionReturn"]
        else:
            count = 1 + self.current_task_spec.runtime_data["count"]
            reachable_positions = self.current_task_spec.runtime_data["reachable_positions"]
        
        self.current_task_spec = task_spec
        self.current_task_spec.stage = "Unknown"
        self.current_task_spec.runtime_data = {
            "count": count,
            "reachable_positions": reachable_positions,
        }
        with include_object_data(self.controller):
            random.shuffle(reachable_positions)

            max_teleports = min(10, len(reachable_positions))
            for teleport_count, pos in enumerate(reachable_positions):
                pos = {k: round(v, 3) for k, v in pos.items()}
                rot = 30 * random.randint(0, 11)
                if force_axis_aligned_start:
                    rot = round_to_factor(30 * random.randint(0, 11), 90)
                md = self.controller.step(
                    "TeleportFull",
                    **pos,
                    rotation={"x": 0, "y": rot, "z": 0},
                    horizon=0.0,
                    standing=True,
                    forceAction=teleport_count == max_teleports - 1,
                ).metadata
                if md["lastActionSuccess"]:
                    break
                else:
                    raise RuntimeError("No reachable positions?")
            
            assert md["lastActionSuccess"]
            self.current_task_spec.agent_position = pos
            self.current_task_spec.agent_rotation = rot
            self.current_task_spec.runtime_data["starting_objects"] = md["objects"]

    def _task_spec_reset(
        self, task_spec: HomeServiceTaskSpec, force_axis_aligned_start: bool
    ):
        assert (
            not task_spec.runtime_sample
        ), "`_task_spec_reset` requires that `task_spec.runtime_sample` is `False`."

        self.current_task_spec = task_spec
        self.controller.reset(self.current_task_spec.scene)
        if self._enhanced_physics_determinism:
            self.controller.step("PausePhysicsAutoSim")

        if force_axis_aligned_start:
            self.current_task_spec.agent_rotation = round_to_factor(
                self.current_task_spec.agent_rotation, 90
            )

        pos = self.current_task_spec.agent_position
        rot = {"x": 0, "y": self.current_task_spec.agent_rotation, "z": 0}
        self.controller.step(
            "TeleportFull",
            **pos,
            rotation=rot,
            horizon=0.0,
            standing=True,
            forceAction=True,
        )

        with include_object_data(self.controller):
            for obj in self.current_task_spec.openable_data:
                current_obj_info = next(
                    l_obj
                    for l_obj in self.last_event.metadata["objects"]
                    if l_obj["name"] == obj["name"]
                )
                self.controller.step(
                    action="OpenObject",
                    objectId=current_obj_info["objectId"],
                    openness=obj["target_openness"],
                    forceAction=True,
                    **self.physics_step_kwargs,
                )
            
            self.controller.step(
                "SetObjectPoses",
                objectPoses=self.current_task_spec.target_poses,
                forceKinematic=False,
                enablePhysicsJitter=True,
                forceRigidbodySleep=True,
            )
            assert self.controller.last_event.metadata["lastActionSuccess"]

    def reset(
        self, task_spec: HomeServiceTaskSpec, force_axis_aligned_start: bool = False,
    ) -> None:
        if task_spec.runtime_sample:
            self._runtime_reset(
                task_spec=task_spec, force_axis_aligned_start=force_axis_aligned_start
            )
        else:
            self._task_spec_reset(
                task_spec=task_spec, force_axis_aligned_start=force_axis_aligned_start
            )
        
        self.object_name_to_start_pose = self._obj_list_to_obj_name_to_pose_dict(
            self.last_event.metadata["objects"]
        )

        self._have_warned_about_mismatch = False
        self._sorted_and_extracted_start_poses = None
        self._agent_signals_done = False

    @staticmethod
    def _obj_list_to_obj_name_to_pose_dict(
        objects: List[Dict[str, Any]]
    ) -> OrderedDict:
        """Helper function to transform a list of object data dicts into a
        dictionary."""
        objects = [
            o
            for o in objects
            if o["openable"] or o.get("objectOrientedBoundingBox") is not None
        ]
        d = OrderedDict(
            (o["name"], o) for o in sorted(objects, key=lambda x: x["name"])
        )
        assert len(d) == len(objects)
        return d

    def stop(self):
        """Terminate the current AI2-THOR session."""
        try:
            self.controller.stop()
        except Exception as _:
            pass

    def __del__(self):
        self.stop()
