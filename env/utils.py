import copy
import compress_pickle
import prior
import logging
import random
from typing import Dict, Callable, Tuple, Union, List, Any, Optional, Sequence

import ai2thor.controller
import lru
import numpy as np

from allenact.utils.system import get_logger
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment
from allenact_plugins.ithor_plugin.ithor_util import include_object_data
from datagen.datagen_utils import scene_from_type_idx

from env.constants import DEFAULT_COMPATIBLE_RECEPTACLES, MAX_HAND_METERS, ROTATION_ANGLE, STEP_SIZE, NOT_PROPER_RECEPTACLES

_UNIFORM_BOX_CACHE = {}


class BoundedFloat(object):
    
    def __init__(self, low: float, high: float):
        self.types = {float, int, np.float64}
        if type(low) not in self.types or type(high) not in self.types:
            raise ValueError("Bound must both be floats.")
        if low > high:
            raise ValueError("low must be less than high.")

        self.low = low
        self.high = high
    
    def sample(self) -> float:
        return random.random() * (self.high - self.low) + self.low

    def __contains__(self, n: float):
        if type(n) not in self.types:
            raise ValueError("n must be a float (or an int).")
        
        return n >= self.low and n <= self.high


class HomeServiceActionSpace(object):

    def __init__(self, actions: Dict[Callable, Dict[str, BoundedFloat]]):
        self.keys = list(actions.keys())
        self.actions = actions

    def execute_random_action(self, log_choice: bool = True) -> None:

        action = random.choice(self.keys)
        kwargs = {
            name: bounds.sample() for name, bounds in self.actions[action].items()
        }

        # logging
        if log_choice:
            kwargs_str = str("".join(f"  {k}: {v},\n" for k, v in kwargs.items()))
            kwargs_str = "\n" + kwargs[:-2] if kwargs_str else ""
            logging.info(f"Executing {action.__name__}(" + kwargs_str + ")")

        action(**kwargs)

    def __contains__(
        self, action_fn_and_kwargs: Tuple[Callable, Dict[str, float]]
    ) -> bool:

        action_fn, variables = action_fn_and_kwargs

        if action_fn not in self.actions:
            return False

        for name, x in variables.items():
            if x not in self.actions[action_fn][name]:
                return False

        return True

    def __str__(self) -> str:

        return self.__repr__()

    def __repr__(self) -> str:
        
        s = ""
        tab = " " * 2
        for action_fn, vars in self.actions.items():
            fn_name = action_fn.__name__
            vstr = ""
            for i, (var_name, bound) in enumerate(vars.items()):
                low = bound.low
                high = bound.high
                vstr += f"{tab * 2}{var_name}: float(low={low}, high={high}"
                vstr += "\n" if i + 1 == len(vars) else ",\n"
            vstr = "\n" + vstr[:-1] if vstr else ""
            s += f"{tab}{fn_name}({vstr}),\n"
        s = s[:-2] if s else ""
        return "ActionSpace(\n" + s + "\n)"


class ObjectInteractablePostionsCache:
    def __init__(
        self, 
        grid_size: float = STEP_SIZE, 
        rotate_step_degrees: float = ROTATION_ANGLE, 
        max_size: int = 20000, 
        ndigits=2
    ):
        self._key_to_positions = lru.LRU(size=max_size)

        self.grid_size = grid_size
        self.rotate_step_degrees = rotate_step_degrees

        self.ndigits = ndigits
        self.max_size = max_size

    def reset_cache(self):
        self._key_to_positions.clear()

    def _get_key(
        self,
        scene_name: str,
        obj: Dict[str, Any],
        hor: Optional[float],
        stand: Optional[bool],
        include_options: bool = False,
    ):
        p = obj["position"]
        return (
            scene_name,
            obj["type"] if "type" in obj else obj["objectType"],
            round(p["x"], self.ndigits),
            round(p["y"], self.ndigits),
            round(p["z"], self.ndigits),
        ) + ((hor, stand) if include_options else ())

    def get(
        self,
        scene_name: str,
        obj: Dict[str, Any],
        controller: ai2thor.controller.Controller,
        reachable_positions: Optional[Sequence[Dict[str, float]]] = None,
        force_cache_refresh: bool = False,
        force_horizon: Optional[int] = None,
        force_standing: Optional[bool] = None,
        avoid_teleport: bool = False,
        max_distance: float = None,
    ) -> List[Dict[str, Union[float, int, bool]]]:
        scene_name = scene_name.replace("_physics", "")

        env = None
        include_options_in_key = False
        if hasattr(controller, "controller"):
            env = controller
            controller = env.controller
            include_options_in_key = True
        
        obj_key = self._get_key(
            scene_name=scene_name,
            obj=obj,
            hor=force_horizon,
            stand=force_standing,
            include_options=include_options_in_key,
        )

        if force_cache_refresh or obj_key not in self._key_to_positions:
            with include_object_data(controller):
                init_metadata = controller.last_event.metadata

            if env is None:
                cur_scene_name = init_metadata["sceneName"].replace("_physics", "")
                key = "name"
            else:
                cur_scene_name = env.scene
                key = "objectId"

            assert (
                scene_name == cur_scene_name
            ), f"Scene names must match when filling a cache miss ({scene_name} != {cur_scene_name})."

            obj_in_scene = next(
                (o for o in init_metadata["objects"] if o["name"] == obj["name"]), None,
            )
            if obj_in_scene is None:
                raise RuntimeError(
                    f"Object with name {obj['name']} must be in the scene when filling a cache miss"
                )

            desired_pos = obj["position"]
            desired_rot = obj["rotation"]

            cur_pos = obj_in_scene["position"]
            cur_rot = obj_in_scene["rotation"]

            should_teleport = (
                IThorEnvironment.position_dist(desired_pos, cur_pos) >= 1e-3
                or IThorEnvironment.rotation_dist(desired_rot, cur_rot) >= 1
            ) and not avoid_teleport

            object_held = obj_in_scene["isPickedUp"]
            physics_was_unpaused = controller.last_event.metadata.get(
                "physicsAutoSimulation", True
            )
            if should_teleport:
                if object_held:
                    if not hand_in_initial_position(
                        controller=controller, ignore_rotation=True,
                    ):
                        raise NotImplementedError

                    if physics_was_unpaused:
                        controller.step("PausePhysicsAutoSim")
                        assert controller.last_event.metadata["lastActionSuccess"]

                event = controller.step(
                    "TeleportObject",
                    objectId=obj_in_scene["objectId"],
                    rotation=desired_rot,
                    **desired_pos,
                    forceAction=True,
                    allowTeleportOutOfHand=True,
                    forceKinematic=True,
                )
                assert event.metadata["lastActionSuccess"]
            
            options = {}
            if force_standing is not None:
                options["standings"] = [force_standing]
            if force_horizon is not None:
                options["horizons"] = [force_horizon]

            metadata = controller.step(
                action="GetInteractablePoses",
                objectId=obj["objectId"],
                maxDistance=max_distance,
                positions=reachable_positions,
                **options,
            ).metadata
            assert metadata["lastActionSuccess"]
            self._key_to_positions[obj_key] = metadata["actionReturn"]

            if should_teleport:
                if object_held:
                    if hand_in_initial_position(
                        controller=controller, ignore_rotation=True,
                    ):
                        controller.step(
                            "PickupObject",
                            objectId=obj_in_scene["objectId"],
                            forceAction=True,
                        )
                        assert controller.last_event.metadata["lastActionSuccess"]

                        if physics_was_unpaused:
                            controller.step("UnpausePhysicsAutoSim")
                            assert controller.last_event.metadata["lastActionSuccess"]
                    else:
                        raise NotImplementedError
                else:
                    event = controller.step(
                        "TeleportObject",
                        objectId=obj_in_scene["objectId"],
                        rotation=cur_rot,
                        **cur_pos,
                        forceAction=True,
                    )
                    assert event.metadata["lastActionSuccess"]

        return self._key_to_positions[obj_key]


def hand_in_initial_position(
    controller: ai2thor.controller.Controller,
    ignore_rotation: bool = False,
):
    metadata = controller.last_event.metadata
    return (
        IThorEnvironment.position_dist(
            metadata["heldObjectPose"]["localPosition"], {"x": 0, "y": -0.16, "z": 0.38},
        ) < 1e-4
        and (
            ignore_rotation
            or IThorEnvironment.angle_between_rotations(
                metadata["heldObjectPose"]["localRotation"],
                {"x": -metadata["agent"]["cameraHorizon"], "y": 0, "z": 0},
            ) < 1e-2
        )
    )

def sample_pick_and_place_target(
    env,
    randomizer: random.Random,
    pickup_target: str = None,
    place_target: str = None,
):
    metadata = env.controller.last_event.metadata
    objects = metadata["objects"]
    pick = None
    if pickup_target is not None:
        pick_sample = [
            o for o in objects if o['objectType'] == pickup_target
        ]
        assert len(pick_sample) > 0, f"{pickup_target} is not in the scene"
    else:
        pick_sample = [
            o for o in objects 
            if o["pickupable"]
            and len(env._interactable_positions_cache.get(
                scene_name=env.scene,
                obj=o,
                controller=env.controller,
                max_distance=2 * MAX_HAND_METERS,
            )) > 0
        ]

    while pick is None and len(pick_sample) > 0:
        pick = randomizer.choice(pick_sample)
        event = env.controller.step(
            "PickupObject", 
            objectId=pick["objectId"], 
            forceAction=True, 
            **env.physics_step_kwargs
        )
        assert event.metadata["lastActionSuccess"]

        if place_target is not None:
            place_sample = [
                o for o in objects if o['objectType'] == place_target
            ]
            assert len(place_sample) > 0, f"{place_target} is not in the scene"
        else:
            place_sample = [
                o for o in objects
                if (
                    o['objectType'] in DEFAULT_COMPATIBLE_RECEPTACLES[pick['objectType']]
                    if pick['objectType'] in DEFAULT_COMPATIBLE_RECEPTACLES
                    else True                    
                ) and o["receptacle"] and not o["pickupable"]
                and len(env._interactable_positions_cache.get(
                    scene_name=env.scene,
                    obj=o,
                    controller=env.controller,
                    max_distance=2 * MAX_HAND_METERS,
                )) > 0
            ]
        
        if pick["parentReceptacles"] is not None:
            recep_types = [
                o["objectType"] for o in objects
                for recep in pick["parentReceptacles"]
                if o["objectId"] == recep
            ]
            place_sample = [
                o for o in place_sample
                if o["objectType"] not in recep_types
            ]
        
        place_sample = [
            o for o in place_sample
            if o['objectType'] not in NOT_PROPER_RECEPTACLES
        ]

        place = None
        while place is None and len(place_sample) > 0:
            place = randomizer.choice(place_sample)
            event = env.controller.step(
                "PutObject", 
                objectId=place["objectId"], 
                forceAction=True, 
                **env.physics_step_kwargs
            )
            if event.metadata["lastActionSuccess"]:
                event = env.controller.step("PausePhysicsAutoSim")
                assert event.metadata["lastActionSuccess"]

                event = env.controller.step(
                    "TeleportObject", 
                    objectId=pick["objectId"], 
                    rotation=pick["rotation"],
                    **pick["position"],
                    forceAction=True,
                    allowTeleportOutOfHand=True,
                    forceKinematic=True,
                )
                assert event.metadata["lastActionSuccess"]
            else:
                place_sample.remove(place)
                place = None
        
        if len(place_sample) == 0:
            assert place is None
            event = env.controller.step("PausePhysicsAutoSim")
            assert event.metadata["lastActionSuccess"]
            event = env.controller.step(
                "TeleportObject", 
                objectId=pick["objectId"], 
                rotation=pick["rotation"],
                **pick["position"],
                forceAction=True,
                allowTeleportOutOfHand=True,
                forceKinematic=True,
            )
            assert event.metadata["lastActionSuccess"]
            pick_sample.remove(pick)
            pick = None
    
    if len(pick_sample) == 0:
        print(f"task_spec {env.current_task_spec.unique_id} is somewhat weird...")
        raise RuntimeError()

    return pick, place

def extract_obj_data(obj):
    """Return object evaluation metrics based on the env state."""
    if "type" in obj:
        return {
            "type": obj["type"],
            "position": obj["position"],
            "rotation": obj["rotation"],
            "openness": obj["openness"],
            "pickupable": obj["pickupable"],
            "broken": obj["broken"],
            "bounding_box": obj["bounding_box"],
            "objectId": obj["objectId"],
            "name": obj["name"],
            "parentReceptacles": obj.get("parentReceptacles", []),
        }
    return {
        "type": obj["objectType"],
        "position": obj["position"],
        "rotation": obj["rotation"],
        "openness": obj["openness"] if obj["openable"] else None,
        "pickupable": obj["pickupable"],
        "broken": obj["isBroken"],
        "objectId": obj["objectId"],
        "name": obj["name"],
        "parentReceptacles": obj.get("parentReceptacles", []),
        "bounding_box": obj["objectOrientedBoundingBox"]["cornerPoints"]
        if obj["objectOrientedBoundingBox"]
        else None,
    }

def get_pose_info(
    objs: Union[Sequence[Dict[str, Any]], Dict[str, Any]]
) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """Return data about each specified object.

    For each object, the return consists of its type, position,
    rotation, openness, and bounding box.
    """
    # list of objects
    if isinstance(objs, Sequence):
        return [extract_obj_data(obj) for obj in objs]
    # single object
    return extract_obj_data(objs)

def get_basis_for_3d_box(corners: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert corners[0].sum() == 0.0

    without_first = corners[1:]
    magnitudes1 = np.sqrt((without_first * without_first).sum(1))
    v0_ind = np.argmin(magnitudes1)
    v0_mag = magnitudes1[v0_ind]

    if v0_mag < 1e-8:
        raise RuntimeError(f"Could not find basis for {corners}")

    v0 = without_first[np.argmin(magnitudes1)] / v0_mag

    orth_to_v0 = (v0.reshape(1, -1) * without_first).sum(-1) < v0_mag / 2.0
    inds_orth_to_v0 = np.where(orth_to_v0)[0]
    v1_ind = inds_orth_to_v0[np.argmin(magnitudes1[inds_orth_to_v0])]
    v1_mag = magnitudes1[v1_ind]
    v1 = without_first[v1_ind, :] / magnitudes1[v1_ind]

    orth_to_v1 = (v1.reshape(1, -1) * without_first).sum(-1) < v1_mag / 2.0
    inds_orth_to_v0_and_v1 = np.where(orth_to_v0 & orth_to_v1)[0]

    if len(inds_orth_to_v0_and_v1) != 1:
        raise RuntimeError(f"Could not find basis for {corners}")

    v2_ind = inds_orth_to_v0_and_v1[0]
    v2 = without_first[v2_ind, :] / magnitudes1[v2_ind]

    orth_mat = np.stack((v0, v1, v2), axis=1)  # Orthonormal matrix

    return orth_mat, magnitudes1[[v0_ind, v1_ind, v2_ind]]

def uniform_box_points(n):
    if n not in _UNIFORM_BOX_CACHE:
        start = 1.0 / (2 * n)
        lin_space = np.linspace(start, 1 - start, num=n).reshape(n, 1)
        mat = lin_space
        for i in range(2):
            mat = np.concatenate(
                (np.repeat(lin_space, mat.shape[0], 0), np.tile(mat, (n, 1))), axis=1,
            )
        _UNIFORM_BOX_CACHE[n] = mat

    return _UNIFORM_BOX_CACHE[n]


def iou_box_3d(b1: Sequence[Sequence[float]], b2: Sequence[Sequence[float]]) -> float:
    """Calculate the IoU between 3d bounding boxes b1 and b2."""
    b1 = np.array(b1)
    b2 = np.array(b2)

    assert b1.shape == b2.shape == (8, 3)

    b1_center = b1[:1, :]
    b1 = b1 - b1_center
    b1_orth_basis, b1_mags = get_basis_for_3d_box(corners=b1)

    b2 = (b2 - b1_center) @ b1_orth_basis
    b2_center = b2[:1, :]
    b2 = b2 - b2_center

    b2_orth_basis, b2_mags = get_basis_for_3d_box(corners=b2)

    sampled_points = b2_center.reshape(1, 3) + (
        uniform_box_points(13) @ (b2_mags.reshape(-1, 1) * np.transpose(b2_orth_basis))
    )

    prop_intersection = (
        np.logical_and(
            sampled_points > -1e-3, sampled_points <= 1e-3 + b1_mags.reshape(1, 3)
        )
        .all(-1)
        .mean()
    )

    b1_vol = np.prod(b1_mags)
    b2_vol = np.prod(b2_mags)
    intersect_vol = b2_vol * prop_intersection

    return intersect_vol / (b1_vol + b2_vol - intersect_vol)

def execute_action(
    controller: ai2thor.controller.Controller,
    action_space: HomeServiceActionSpace,
    action_fn: Callable,
    thor_action: str,
    error_message: str = "",
    updated_kwarg_names: Optional[Dict[str, str]] = None,
    default_thor_kwargs: Optional[Dict[str, Any]] = None,
    preprocess_kwargs_inplace: Optional[Callable] = None,
    **kwargs: float,
) -> bool:

    if updated_kwarg_names is None:
        updated_kwarg_names = {}
    if default_thor_kwargs is None:
        default_thor_kwargs = {}

    if (action_fn, kwargs) not in action_space:
        raise ValueError(
            error_message
            + f" action_fn=={action_fn}, kwargs=={kwargs}, action_space=={action_space}."
        )

    if preprocess_kwargs_inplace is not None:
        if len(updated_kwarg_names) != 0:
            raise NotImplementedError(
                "Cannot have non-empty `updated_kwarg_names` and a non-None `preprocess_kwargs_inplace` argument."
            )
        preprocess_kwargs_inplace(kwargs)

    for better_kwarg, thor_kwarg in updated_kwarg_names.items():
        kwargs[thor_kwarg] = kwargs[better_kwarg]
        del kwargs[thor_kwarg]

    for name, value in default_thor_kwargs.items():
        kwargs[name] = value

    event = controller.step(thor_action, **kwargs)
    return event.metadata["lastActionSuccess"]

def filter_positions(
    grid_size: float, 
    rotation_angle: float, 
    positions: Sequence[Dict[str, Any]], 
    compare_keys: List[str]
):
    n_positions = []
    for position in positions:
        assert all(key in position for key in compare_keys)
        skip = False
        for key in compare_keys:
            if key in ['x', 'z']:
                position[key] = round(position[key], 3)
                if (position[key] % grid_size > 1e-2):
                    skip = True
                    break
            if key == 'rotation':
                if position[key] % rotation_angle > 1e-2:
                    skip = True
                    break
        if not skip:
            for n_position in n_positions:
                if all([position[key] == n_position[key] for key in compare_keys]):
                    skip = True
                    break
        if not skip:
            n_positions.append(position)

    return n_positions


def get_top_down_frame(controller):
    import copy
    from PIL import Image
    
    # Setup the top-down camera
    event = controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
    pose = copy.deepcopy(event.metadata["actionReturn"])

    bounds = event.metadata["sceneBounds"]["size"]
    max_bound = max(bounds["x"], bounds["z"])

    pose["fieldOfView"] = 50
    pose["position"]["y"] += 1.1 * max_bound
    pose["orthographic"] = False
    pose["farClippingPlane"] = 50
    del pose["orthographicSize"]

    # add the camera to the scene
    event = controller.step(
        action="AddThirdPartyCamera",
        **pose,
        skyboxColor="white",
        raise_for_failure=True,
    )
    top_down_frame = event.third_party_camera_frames[-1]
    return Image.fromarray(top_down_frame)


def save_frames_to_mp4(frames: Sequence[np.ndarray], file_name: str, fps=3):
    import matplotlib.pyplot as plt
    from matplotlib import animation
    import pylab

    h, w, _ = frames[0].shape
    aspect_ratio = w / h
    fig = plt.figure(figsize=(5 * aspect_ratio, 5))
    ax = fig.add_subplot(111)
    ax.set_frame_on(False)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax.set_aspect("equal")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    im = ax.imshow(frames[0], cmap="gray", interpolation="nearest")
    im.set_clim([0, 255])

    pylab.tight_layout()

    def update_img(n):
        if n >= len(frames):
            im.set_data(frames[-1])
        else:
            im.set_data(frames[n])
        return im

    ani = animation.FuncAnimation(fig, update_img, len(frames) - 1, interval=200)
    writer = animation.writers["ffmpeg"](fps=fps)

    ani.save(file_name, writer=writer, dpi=300)


class Houses:
    def __init__(
        self, revision="main", valid_houses_file=None,
    ):
        if valid_houses_file is None:
            self._data = prior.load_dataset("procthor-10k", revision=revision, offline=True)
            self._mode = "train"
        else:
            get_logger().info(f"Using valid_houses_file {valid_houses_file}")
            self._data = {"val": compress_pickle.load(valid_houses_file)}
            self._mode = "val"

    def mode(self, mode: str):
        if mode in ["val", "valid", "validation"]:
            mode = "val"
        assert mode in [
            "train",
            "val",
            "test",
        ], f"missing {mode} (available 'train', 'val', 'test')"
        self._mode = mode

    @property
    def current_mode(self):
        return self._mode

    def __getitem__(self, pos: int):
        return copy.deepcopy(self._data[self._mode][pos])

    def __len__(self):
        return len(self._data[self._mode])
