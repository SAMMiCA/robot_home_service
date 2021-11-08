import copy
import logging
import random
from typing import Dict, Callable, Tuple, Union, List, Any, Optional, Sequence

import ai2thor.controller
import lru
import numpy as np

from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment
from allenact_plugins.ithor_plugin.ithor_util import include_object_data

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

    def _get_key(self, scene_name: str, obj: Dict[str, Any]):
        p = obj["position"]
        return (
            scene_name,
            obj["type"] if "type" in obj else obj["objectType"],
            round(p["x"], self.ndigits),
            round(p["y"], self.ndigits),
            round(p["z"], self.ndigits),
        )

    def get(
        self,
        scene_name: str,
        obj: Dict[str, Any],
        controller: ai2thor.controller.Controller,
        reachable_positions: Optional[Sequence[Dict[str, float]]] = None,
        force_cache_refresh: bool = False,
        max_distance: float = None,
    ) -> List[Dict[str, Union[float, int, bool]]]:
        scene_name = scene_name.replace("_physics", "")
        obj_key = self._get_key(scene_name=scene_name, obj=obj)

        if force_cache_refresh or obj_key not in self._key_to_positions:
            with include_object_data(controller):
                init_metadata = controller.last_event.metadata

            cur_scene_name = init_metadata["sceneName"].replace("_physics", "")
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
            )
            physics_was_unpaused = controller.last_event.metadata.get(
                "physicsAutoSimulation", True
            )

            initialization_parameters = copy.deepcopy(controller.initialization_parameters)
            init_grid_size = initialization_parameters["gridSize"]
            init_rotate_step_degrees = initialization_parameters["rotateStepDegrees"]
            should_init = not (
                initialization_parameters["gridSize"] == self.grid_size
                and initialization_parameters["rotateStepDegrees"] == self.rotate_step_degrees
            )

            # print(f'befor init, agent_location: {controller.last_event.metadata["agent_position"]}')

            # if should_init:
            #     initialization_parameters["gridSize"] = self.grid_size
            #     initialization_parameters["rotateStepDegrees"] = self.rotate_step_degrees

            #     if physics_was_unpaused:
            #         controller.step("PausePhysicsAutoSim")
            #         assert controller.last_event.metadata["lastActionSuccess"]
            #     # controller.initialization_parameters.update(initialization_parameters)

            #     event = controller.step(
            #         "Initialize",
            #         raise_for_failure=True,
            #         **initialization_parameters,
            #         # gridSize=init_grid_size,
            #         # rotateStepDegrees=init_rotate_step_degrees,
            #     )

            # print(f'after init, agent_location: {controller.last_event.metadata["agent_position"]}')

            object_held = obj_in_scene["isPickedUp"]
            if should_teleport:
                if object_held:
                    if not hand_in_initial_position(controller=controller):
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

            metadata = controller.step(
                action="GetInteractablePoses",
                objectId=obj["objectId"],
                maxDistance=max_distance,
                positions=reachable_positions,
            ).metadata
            assert metadata["lastActionSuccess"]
            # self._key_to_positions[obj_key] = filter_positions(
            #     grid_size=STEP_SIZE, 
            #     rotation_angle=ROTATION_ANGLE, 
            #     positions=metadata["actionReturn"],
            #     compare_keys=["x", "z", "rotation", "horizon", "standing"]
            # )
            self._key_to_positions[obj_key] = metadata["actionReturn"]

            if should_teleport:
                if object_held:
                    if hand_in_initial_position(controller=controller):
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

            # print(f'after get, agent_location: {controller.last_event.metadata["agent_position"]}')

            # if should_init:
            #     initialization_parameters["gridSize"] = init_grid_size
            #     initialization_parameters["rotateStepDegrees"] = init_rotate_step_degrees

            #     if physics_was_unpaused:
            #         controller.step("PausePhysicsAutoSim")
            #         assert controller.last_event.metadata["lastActionSuccess"]
            #     # controller.initialization_parameters.update(initialization_parameters)

            #     event = controller.step(
            #         "Initialize",
            #         raise_for_failure=True,
            #         **initialization_parameters,
            #         # gridSize=init_grid_size,
            #         # rotateStepDegrees=init_rotate_step_degrees,
            #     )
            #     assert event.metadata["lastActionSuccess"]

            #     if physics_was_unpaused:
            #         controller.step("PausePhysicsAutoSim")
            #         assert controller.last_event.metadata["lastActionSuccess"]

            #     event = controller.step(
            #         "TeleportFull",
            #         position=init_metadata["agent"]["position"],
            #         rotation=init_metadata["agent"]["rotation"],
            #         horizon=init_metadata["agent"]["cameraHorizon"],
            #         standing=init_metadata["agent"]["isStanding"],
            #         forceAction=True,
            #     )
            #     assert event.metadata["lastActionSuccess"]
            
            # print(f'after get & init, agent_location: {controller.last_event.metadata["agent_position"]}')

        return self._key_to_positions[obj_key]


def hand_in_initial_position(controller: ai2thor.controller.Controller):
    metadata = controller.last_event.metadata
    return (
        IThorEnvironment.position_dist(
            metadata["hand"]["localPosition"], {"x": 0, "y": -0.16, "z": 0.38},
        )
        < 1e-4
        and IThorEnvironment.angle_between_rotations(
            metadata["hand"]["localRotation"], {"x": 0, "y": 0, "z": 0}
        )
        < 1e-2
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


# def sample_pick_target(
#     env: IThorEnvironment,
#     random: random.Random,
#     pick: Union[Optional[Dict[str, Any]], str] = None,
# ) -> Dict[str, Any]:
#     with include_object_data(env.controller):
#         objects = env.last_event.metadata["objects"]
#     pickupable_object_types = [
#         obj['objectType'] for obj in objects 
#         if obj["pickupable"]
#     ]

#     if pick is not None:
#         if not isinstance(pick, str):
#             pick = pick["objectType"]
        
#         assert pick in pickupable_object_types, f"{pick} is not in the scene."
#         pick = next(
#             obj for obj in objects
#             if obj['objectType'] == pick                        
#         )
#     else:
#         sample = [
#             obj for obj in objects
#             if obj["objectType"] in pickupable_object_types
#         ]
#         pick = random.choice(sample) if len(sample) > 0 else None
    
#     return pick

# def sample_place_target(
#     env: IThorEnvironment,
#     random: random.Random,
#     pick_type: str,
#     place: Union[Optional[Dict[str, Any]], str] = None,
# ) -> Dict[str, Any]:
#     with include_object_data(env.controller):
#         objects = env.last_event.metadata["objects"]
#     receptacle_object_types = [
#         obj['objectType'] for obj in objects 
#         if obj["receptacle"] and not obj["pickupable"] and not obj['openable']
#     ]

#     assert pick_type is not None, f"pick_type should not be None"
#     if place is not None:
#         if not isinstance(place, str):
#             place = place["objectType"]
        
#         assert place in receptacle_object_types, f"{place} is not in the scene."
#         if pick_type in DEFAULT_COMPATIBLE_RECEPTACLES:
#             assert place in DEFAULT_COMPATIBLE_RECEPTACLES[pick_type], f"{place} is not compatible with {pick_type}"
                
#         place = next(
#             obj for obj in objects
#             if obj['objectType'] == place                        
#         )

#     else:
#         # randomly pick one of the receptacle objects which is compatible to the pick
#         if pick_type in DEFAULT_COMPATIBLE_RECEPTACLES:
#             sample = [
#                 obj for obj in objects
#                 if obj["objectType"] in receptacle_object_types
#                 and obj["objectType"]  in DEFAULT_COMPATIBLE_RECEPTACLES[pick_type]
#             ]
#         else:
#             sample = [
#                 obj for obj in objects
#                 if obj["objectType"] in receptacle_object_types
#             ]
#         place = random.choice(sample) if len(sample) > 0 else None
#         if place is None:
#             print(
#                 f"Compatible receptacles for {pick_type} is not in the scene.\n"
#                 f"Please change the pickup_target instead of {pick_type}"
#             )

#     return place
        

# def sample_pick_place_target(
#     env: IThorEnvironment,
#     random: random.Random,
#     pick: Optional[Dict[str, Any]] = None,
#     place: Optional[Dict[str, Any]] = None,
# ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
#     pick = sample_pick_target(env, random, pick)
#     place = (
#         sample_place_target(env, random, pick["objectType"], place) 
#         if pick is not None and "objectType" in pick
#         else None
#     )

#     return pick, place

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