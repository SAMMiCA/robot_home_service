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
# from datagen.datagen_constants import OBJECT_TYPES_TO_NOT_MOVE
from datagen.datagen_utils import (
    open_objs,
    get_object_ids_to_not_move_from_object_types,
    # remove_objects_until_all_have_identical_meshes,
    scene_from_type_idx,
)
from env.constants import (
    REQUIRED_THOR_VERSION,
    MAX_HAND_METERS,
    PICKUPABLE_OBJECTS,
    OPENABLE_OBJECTS,
    RECEPTACLE_OBJECTS,
    SCENE_TO_SCENE_TYPE
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
        scene_index: int,
        start_scene: str,
        target_scene: str,
        stage: Optional[str] = None,
        agent_positions: Optional[Dict[str, Dict[str, float]]] = None,
        agent_rotations: Optional[Dict[str, float]] = None,
        starting_poses: Optional[Sequence[Dict[str, Any]]] = None,
        objs_to_open: Optional[Sequence[Dict[str, Any]]] = None,
        runtime_sample: bool = False,
        runtime_data: Optional[Dict[str, Any]] = None,
        **metrics,
    ):
        """HomeServiceTaskSpec"""
        self.scene_index = scene_index
        self.start_scene = start_scene
        self.target_scene = target_scene
        self.stage = stage
        self.agent_positions = agent_positions
        self.agent_rotations = agent_rotations
        self.starting_poses = starting_poses
        self.objs_to_open = objs_to_open
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

        return f"{self.stage}__{self.start_scene}__{self.target_scene}"


class HomeServiceSimpleTaskOrderTaskSpec(HomeServiceTaskSpec):
    def __init__(
        self,
        pickup_object: str,
        start_receptacle: str,
        place_receptacle: str,
        **spec_kwargs,
    ):
        super().__init__(**spec_kwargs)
        self.pickup_object = pickup_object
        self.start_receptacle = start_receptacle
        self.place_receptacle = place_receptacle

    @property
    def unique_id(self):
        return f"{super().unique_id}__{self.pickup_object}__{self.start_receptacle}__{self.place_receptacle}"

    @property
    def task_type(self):
        return (
            f"Pick_{self.pickup_object}_And_Place_{self.place_receptacle}"
            if self.place_receptacle != "User" else
            f"Bring_Me_{self.pickup_object}"
        )


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
                self.goto_kitchen: {},
                self.goto_living_room: {},
                self.goto_bedroom: {},
                self.goto_bathroom: {},
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

    def goto_kitchen(self) -> bool:
        pass

    def goto_living_room(self) -> bool:
        pass

    def goto_bedroom(self) -> bool:
        pass

    def goto_bathroom(self) -> bool:
        pass

    @property
    def all_rearranged(self) -> bool:
        pass

    @property
    def poses(
        self,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        # with include_object_data(self.controller):
        #     obj_name_to_current_obj = self._obj_list_to_obj_name_to_pose_dict(
        #         self.controller.last_event.metadata["objects"]
        #     )
        # ordered_obj_names = list(self.object_name_to_start_pose.keys())

        # current_objs_list = []
        # for obj_name in ordered_obj_names:
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

            # remove_objects_until_all_have_identical_meshes(self.controller)
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
        self, task_spec: HomeServiceTaskSpec, force_axis_aligned_start: bool, scene_type: Optional[str] = None,
    ):
        assert (
            not task_spec.runtime_sample
        ), "`_task_spec_reset` requires that `task_spec.runtime_sample` is `False`."

        self.current_task_spec = task_spec
        if scene_type is None:
            reset_scene = self.current_task_spec.start_scene
            scene_type = SCENE_TO_SCENE_TYPE[reset_scene]
        else:
            reset_scene = scene_from_type_idx(scene_type, self.current_task_spec.scene_index)

        self.controller.reset(reset_scene)
        if self._enhanced_physics_determinism:
            self.controller.step("PausePhysicsAutoSim")

        if force_axis_aligned_start:
            self.current_task_spec.agent_rotations[scene_type] = round_to_factor(
                self.current_task_spec.agent_rotations[scene_type], 90
            )

        pos = self.current_task_spec.agent_positions[scene_type]
        rot = {
            "x": 0, 
            "y": round_to_factor(
                self.current_task_spec.agent_rotations[scene_type], 90
            ) if force_axis_aligned_start else self.current_task_spec.agent_rotations[scene_type],
            "z": 0,
        }
        self.controller.step(
            "TeleportFull",
            **pos,
            rotation=rot,
            horizon=0.0,
            standing=True,
            forceAction=True,
        )
        assert self.controller.last_event.metadata["lastActionSuccess"]

        if reset_scene == self.current_task_spec.target_scene:
            with include_object_data(self.controller):
                for obj in self.current_task_spec.objs_to_open:
                    current_obj_info = next(
                        l_obj
                        for l_obj in self.last_event.metadata["objects"]
                        if l_obj["name"] == obj["name"]
                    )
                    self.controller.step(
                        action="OpenObject",
                        objectId=current_obj_info["objectId"],
                        openness=1,
                        forceAction=True,
                        **self.physics_step_kwargs,
                    )
                
                self.controller.step(
                    "SetObjectPoses",
                    objectPoses=self.current_task_spec.starting_poses,
                    forceKinematic=False,
                    enablePhysicsJitter=True,
                    forceRigidbodySleep=True,
                )
                assert self.controller.last_event.metadata["lastActionSuccess"]

    def reset(
        self, task_spec: HomeServiceTaskSpec, force_axis_aligned_start: bool = False, scene_type: str = None,
    ) -> None:
        if task_spec.runtime_sample:
            self._runtime_reset(
                task_spec=task_spec, force_axis_aligned_start=force_axis_aligned_start
            )
        else:
            self._task_spec_reset(
                task_spec=task_spec, force_axis_aligned_start=force_axis_aligned_start, scene_type=scene_type,
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
