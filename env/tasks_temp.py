import copy
import os
import enum
import random
import traceback
from abc import ABC
from typing import Any, Tuple, Optional, Dict, Sequence, List, Union, cast, Set

import compress_pickle
import gym.spaces
import numpy as np
import stringcase

from allenact.base_abstractions.misc import RLStepResult
from allenact.base_abstractions.sensor import SensorSuite
from allenact.base_abstractions.task import Task, TaskSampler
from allenact.utils.system import get_logger
from allenact_plugins.ithor_plugin.ithor_util import round_to_factor
from env.constants import STARTER_HOME_SERVICE_DATA_DIR
from env.environment import (
    HomeServiceEnvironment,
    HomeServiceTaskSpec,
)
from env.expert import (
    # GreedySimplePickAndPlaceExpert,
    ShortestPathNavigatorTHOR,
    SubTaskExpert,
)
from env.utils import (
    HomeServiceActionSpace,
    include_object_data,
    get_pose_info
)


class AbstractHomeServiceTask(Task[HomeServiceEnvironment], ABC):
    @staticmethod
    def agent_location_to_tuple(
        agent_loc: Dict[str, Union[Dict[str, float], bool, float, int]],
        base_rotation: int = 90,
        base_horizon: int = 30,
    ) -> Tuple[float, float, int, int, int]:
        if "position" in agent_loc:
            agent_loc = {
                "x": agent_loc["position"]["x"],
                "y": agent_loc["position"]["y"],
                "z": agent_loc["position"]["z"],
                "rotation": agent_loc["rotation"]["y"],
                "horizon": agent_loc["cameraHorizon"],
                "standing": agent_loc.get("isStanding"),
            }
        return (
            round(agent_loc["x"], 3),
            round(agent_loc["z"], 3),
            round_to_factor(agent_loc["rotation"], base_rotation) % 360,
            1 * agent_loc["standing"],
            round_to_factor(agent_loc["horizon"], base_horizon) % 360,
        )

    @property
    def agent_location_tuple(self) -> Tuple[float, float, int, int, int]:
        return self.agent_location_to_tuple(
            agent_loc=self.env.get_agent_location(),
            base_rotation=self.env.rotate_step_degrees,
            base_horizon=self.env.horizon_step_degrees,
        )


class HomeServiceTask(AbstractHomeServiceTask):
    NAME_KEY = "objectId"

    def __init__(
        self,
        sensors: SensorSuite,
        env: HomeServiceEnvironment,
        max_steps: int,
        discrete_actions: Tuple[str, ...],
        smooth_nav: bool = False,
        smoothing_factor: int = 1,
        force_axis_aligned_start: bool = False,
        require_done_action: bool = False,
        task_spec_in_metrics: bool = False,
    ) -> None:
        """
        Create a new HomeService task.
        """

        super().__init__(
            env=env, sensors=sensors, task_info=dict(), max_steps=max_steps,
        )
        self.discrete_actions = discrete_actions
        self.smooth_nav = smooth_nav
        self.smoothing_factor = smoothing_factor if self.smooth_nav else 1
        self.force_axis_aligned_start = force_axis_aligned_start
        self.require_done_action = require_done_action
        self.task_spec_in_metrics = task_spec_in_metrics

        self._took_end_action: bool = False
        
        self.greedy_expert = None
        self.actions_taken = []
        self.actions_taken_success = []
        self.agent_locs = [self.env.get_agent_location()]

    @property
    def action_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(len(self.action_names()))

    def close(self) -> None:
        try:
            self.env.stop()
        except Exception as _:
            pass

    def metrics(self) -> Dict[str, Any]:
        if not self.is_done():
            return {}

        env = self.env
        ips, gps, cps = env.poses

        metrics = super().metrics()
        
        task_info = metrics["task_info"]
        task_info["scene"] = self.env.scene
        

        return metrics

    def action_names(self, **kwargs) -> Tuple[str, ...]:

        return self.discrete_actions
    
    def render(self, *args, **kwargs) -> Dict[str, np.array]:

        obs = self.env.observation
        return {
            "rgb": obs[0], "depth": obs[1],
        }

    def reached_terminal_state(self) -> bool:

        return (self.require_done_action and self._took_end_action) or (
            (not self.require_done_action)
        )

    # def _judge(self, obs, action, next_obs, action_success, current_subtask, subtask_done) -> float:
    #     """Return the reward from a new (s, a, s')."""
    #     action_name = self.action_names()[action]
    #     reward = -0.05
    #     if not action_success:
    #         reward += -0.05

    #     if subtask_done:
    #         reward += 1

    #     if current_subtask[0] == "Done":
    #         if action_name == "done":
    #             reward += 10
    #         else:
    #             # should take "done" when all the task is done
    #             reward += -10
    #     else:
    #         # If "done" action taken when it is not "Done" subtask
    #         if action_name == "done":
    #             reward += -10

    #         if current_subtask[0] != "Goto":
    #             if action_name.startswith("goto"):
    #                 # Wrongly moved to other room type
    #                 reward += -10
        
    #     if self._took_subtask_rollback:
    #         reward += -1 * self._rollback_count
    #         self._took_subtask_rollback = False
    #         self._rollback_count = 0

    #     return reward

    def _judge(self, obs, action, next_obs, action_success,) -> float:
        """Return the reward from a new (s, a, s')."""
        # TODO: Log reward scenarios.

        reward = 0.

        return reward

    def find_object_by_type(self, object_type: str, visible: bool = True,):
        with include_object_data(self.env.controller):
            metadata = self.env.last_event.metadata

            possible_objects = [
                o
                for o in metadata["objects"]
                if (
                    (o["objectType"] == object_type)
                    and (o["visible"] or not visible)
                )
            ]
            
            possible_objects = sorted(
                possible_objects, key=lambda po: (po["distance"], po["name"])
            )
            
            return possible_objects

    def _step(self, action: int) -> RLStepResult:
        """
        action: is the index of the action from self.action_names()
        """
        action_name = self.action_names()[action]

        if action_name.startswith("pickup"):
            with include_object_data(self.env.controller):
                md = self.env.last_event.metadata

                if len(md["inventoryObjects"]) != 0:
                    action_success = False
                else:
                    object_type = stringcase.pascalcase(
                        action_name.replace("pickup_", "")
                    )
                    possible_objs = self.find_object_by_type(object_type)
                    obj_before = None
                    if len(possible_objs) > 0:
                        obj_before = possible_objs[0]
                        obj_id = obj_before["objectId"]

                    if obj_before is not None:
                        self.env.controller.step(
                            "PickupObject",
                            objecId=obj_id,
                            **self.env.physics_step_kwargs,
                        )
                        action_success = self.env.last_event.metadata["lastActionSuccess"]
                    else:
                        action_success = False
                    
                    if action_success and self.env.held_object is None:
                        get_logger().warning(
                            f"`PickupObject` was successful in picking up {obj_id} but we're not holding"
                            f" any object! Current task spec: \n {self.env.current_task_spec}"
                        )
                        action_success = False

        elif action_name.startswith("open_by_type"):
            object_type = stringcase.pascalcase(
                action_name.replace("open_by_type_", "")
            )
            with include_object_data(self.env.controller):

                possible_objs = self.find_object_by_type(object_type)

                obj_before = None
                if len(possible_objs) > 0:
                    obj_before = possible_objs[0]
                    obj_id = obj_before["objectId"]

                if obj_before is not None:
                    action_success = True
                    if obj_before["openness"] < 1.0:
                        self.env.controller.step(
                            "OpenObject",
                            objecId=obj_id,
                            openness=1.0,
                            **self.env.physics_step_kwargs,
                        )
                        action_success = self.env.last_event.metadata["lastActionSuccess"]
                else:
                    action_success = False

        elif action_name.startswith("close_by_type"):
            object_type = stringcase.pascalcase(
                action_name.replace("close_by_type_", "")
            )
            with include_object_data(self.env.controller):

                possible_objs = self.find_object_by_type(object_type, openable=True)

                obj_before = None
                if len(possible_objs) > 0:
                    obj_before = possible_objs[0]
                    obj_id = obj_before["objectId"]

                if obj_before is not None:
                    action_success = True
                    if obj_before["openness"] > 0.0:
                        self.env.controller.step(
                            "CloseObject",
                            objecId=obj_id,
                            openness=1.0,
                            **self.env.physics_step_kwargs,
                        )
                        action_success = self.env.last_event.metadata["lastActionSuccess"]
                else:
                    action_success = False

        elif action_name.startswith("put"):
            # USE This instead of "drop_held_object_with_snap"
            DEC = 2

            with include_object_data(self.env.controller):
                metadata = self.env.last_event.metadata

                if len(metadata["inventoryObjects"]) == 0:
                    # The agent is not holding an object.
                    action_success = False
                else:
                    # When dropping up an object, make it breakable.
                    self.env.controller.step(
                        "MakeObjectBreakable", objectId=self.held_object["objectId"]
                    )

                    object_type = stringcase.pascalcase(
                        action_name.replace("put_", "")
                    )
                    # In case that the object_type is target receptacle
                    if object_type in self.env.current_task_spec.place_receptacle:
                        object_type = self.env.current_task_spec.place_receptacle

                        # Trying to put object by snapping
                        # Determine whether the agent is located valid position
                        agent = metadata["agent"]
                        held_obj = self.env.held_object
                        goal_pose = self.env.object_name_to_target_pose[held_obj["name"]]
                        goal_pos = goal_pose["position"]
                        goal_rot = goal_pose["rotation"]
                        good_positions_to_put_obj = self.env._interactable_positions_cache.get(
                            scene_name=metadata["sceneName"],
                            obj={**held_obj, **{"position": goal_pos, "rotation": goal_rot},},
                            controller=self.env.controller,
                            force_cache_refresh=self.env.force_cache_reset,
                        )

                        def position_to_tuple(position: Dict[str, float]):
                            return tuple(round(position[k], DEC) for k in ["x", "y", "z"])

                        agent_xyz = position_to_tuple(agent["position"])
                        agent_rot = (round(agent["rotation"]["y"] / self.env.rotate_step_degrees) * self.env.rotate_step_degrees) % 360
                        agent_standing = int(agent["isStanding"])
                        agent_horizon = round(agent["cameraHorizon"])

                        for valid_agent_pos in good_positions_to_put_obj:
                            # Checks if the agent is close enough to the target
                            # for the object to be snapped to the target location.
                            valid_xyz = position_to_tuple(valid_agent_pos)
                            valid_rot = (round(valid_agent_pos["rotation"] / self.env.rotate_step_degrees) * self.env.rotate_step_degrees) % 360
                            valid_standing = int(valid_agent_pos["standing"])
                            valid_horizon = round(valid_agent_pos["horizon"])
                            if (
                                valid_xyz == agent_xyz  # Position
                                and valid_rot == agent_rot  # Rotation
                                and valid_standing == agent_standing  # Standing
                                and round(valid_horizon) == agent_horizon  # Horizon
                            ):
                                # Try a few locations near the target for robustness' sake
                                positions = [
                                    {
                                        "x": goal_pos["x"] + 0.001 * xoff,
                                        "y": goal_pos["y"] + 0.001 * yoff,
                                        "z": goal_pos["z"] + 0.001 * zoff,
                                    }
                                    for xoff in [0, -1, 1]
                                    for zoff in [0, -1, 1]
                                    for yoff in [0, 1, 2]
                                ]
                                self.env.controller.step(
                                    action="TeleportObject",
                                    objectId=held_obj["objectId"],
                                    rotation=goal_rot,
                                    positions=positions,
                                    forceKinematic=True,
                                    allowTeleportOutOfHand=True,
                                    makeUnbreakable=True,
                                )
                                break
                        
                        if self.env.held_object is None:
                            put_obj = next(
                                get_pose_info(o)
                                for o in self.env.last_event.metadata["objects"]
                                if o["name"] == goal_pose["name"]
                            )
                            if len(put_obj["parentReceptacles"]) > 0:
                                recep = None
                                for recep_id in put_obj["parentReceptacles"]:
                                    recep = next(
                                        (
                                            o
                                            for o in self.env.last_event.metadata["objects"]
                                            if o["objectType"] == object_type and o["objectId"] == recep_id
                                        ), None
                                    )
                                    if recep is not None:
                                        action_success = True
                                        break
                                
                                if recep is None:
                                    action_success = False
                            else:
                                action_success = False
                    
                    assert self.env.held_object is not None
                    possible_objs = self.find_object_by_type(object_type)
                    
                    obj_before = None
                    if len(possible_objs) > 0:
                        obj_before = possible_objs[0]
                        obj_id = obj_before["objectId"]

                    if obj_before is not None:
                        self.env.controller.step(
                            "PutObject",
                            objectId=obj_id,
                            **self.env.physics_step_kwargs,
                        )
                        action_success = self.env.last_event.metadata["lastActionSuccess"]
                    else:
                        action_success = False

        elif action_name.startswith(("move", "rotate", "look", "stand", "crouch")):
            action_success = getattr(self.env, action_name)()
        elif action_name == "done":
            self._took_end_action = True
            action_success = True
        elif action_name == "pass":
            event = self.env.controller.step("Pass")
            action_success = event.metadata["lastActionSuccess"]
        else:
            raise RuntimeError(
                f"Action '{action_name}' is not in the action space {HomeServiceActionSpace}"
            )

        self.actions_taken.append(action_name)
        self.actions_taken_success.append(action_success)
        if self.task_spec_in_metrics:
            self.agent_locs.append(self.env.get_agent_location())

        reward = self._judge(
            obs=self._obs,      # s0
            action=action,      # a0
            next_obs=self.get_observations(),      # s1
            action_success=action_success,
        )
        # self.rewards.append(reward)

        # self._obs is updated
        return RLStepResult(
            observation=self._obs,
            reward=reward,
            done=self.is_done(),
            info={"action_name": action_name, "action_success": action_success},
        )

    def get_observations(self, **kwargs) -> Any:
        self._obs = super().get_observations(**kwargs)
        return self._obs

    def step(self, action: int) -> RLStepResult:
        step_result = super().step(action=action)
        if self.greedy_expert is not None:
            self.greedy_expert.update(
                action_taken=action, action_success=step_result.info["action_success"]
            )
        step_result = RLStepResult(
            observation=step_result.observation,
            reward=step_result.reward,
            done=step_result.done,
            info=step_result.info,
        )
        return step_result
    
    def pickupable_objects(self, visible_only: bool = True):
        with include_object_data(self.env.controller):
            return [
                o
                for o in self.env.last_event.metadata["objects"]
                if ((o["visible"] or not visible_only) and o["pickupable"])
            ]

    def openable_not_pickupable_objects(self, visible_only: bool = True):
        with include_object_data(self.env.controller):
            return [
                o
                for o in self.env.last_event.metadata["objects"]
                if (
                    (o["visible"] or not visible_only)
                    and (o["openable"] and not o["pickupable"])
                )
            ]

    def receptacle_not_pickupable_objects(self, visible_only: bool = True):
        with include_object_data(self.env.controller):
            return [
                o
                for o in self.env.last_event.metadata["objects"]
                if (
                    (o["visible"] or not visible_only)
                    and (o["receptacle"] and not o["pickupable"])
                )
            ]

    def pickupable_or_openable_objects(self, visible_only: bool = True):
        with include_object_data(self.env.controller):
            return [
                o
                for o in self.env.last_event.metadata["objects"]
                if (
                    (o["visible"] or not visible_only)
                    and (o["pickupable"] or (o["openable"] and not o["pickupable"]))
                )
            ]


class HomeServiceTaskSpecIterable:

    def __init__(
        self,
        task_keys_to_task_spec_dicts: Dict[str, List[Dict]],
        seed: int,
        epochs: Union[int, float],
        shuffle: bool = True,
    ):
        assert epochs >= 1

        self.task_keys_to_task_spec_dicts = {
            k: [*v] for k, v in task_keys_to_task_spec_dicts.items()
        }
        assert len(self.task_keys_to_task_spec_dicts) != 0

        self._seed = seed
        self.random = random.Random(self.seed)
        self.start_epochs = epochs
        self.remaining_epochs = epochs
        self.shuffle = shuffle

        self.remaining_task_keys: List[str] = []
        self.task_spec_dicts_for_current_task_key: List[Dict[str, Any]] = []
        self.current_task_key: Optional[str] = None

        self.reset()

    @property
    def seed(self) -> int:
        return self._seed

    @seed.setter
    def seed(self, seed: int):
        self._seed = seed
        self.random.seed(seed)

    @property
    def length(self):
        if self.remaining_epochs == float("inf"):
            return float("inf")

        return (
            len(self.task_spec_dicts_for_current_task_key)
            + sum(
                len(self.task_keys_to_task_spec_dicts[task_key])
                for task_key in self.remaining_task_keys
            )
            + self.remaining_epochs
            * (sum(len(v) for v in self.task_keys_to_task_spec_dicts.values()))
        )
    
    @property
    def total_unique(self):
        return sum(len(v) for v in self.task_keys_to_task_spec_dicts.values())

    def reset(self):
        self.random.seed(self.seed)
        self.remaining_epochs = self.start_epochs
        self.remaining_task_keys.clear()
        self.task_spec_dicts_for_current_task_key.clear()
        self.current_task_key = None

    def refresh_remaining_scenes(self):
        if self.remaining_epochs <= 0:
            raise StopIteration
        self.remaining_epochs -= 1

        self.remaining_task_keys = list(sorted(self.task_keys_to_task_spec_dicts.keys()))
        if self.shuffle:
            self.random.shuffle(self.remaining_task_keys)
        return self.remaining_task_keys

    def __next__(self) -> HomeServiceTaskSpec:
        while len(self.task_spec_dicts_for_current_task_key) == 0:
            if len(self.remaining_task_keys) == 0:
                self.refresh_remaining_scenes()
            self.current_task_key = self.remaining_task_keys.pop()

            self.task_spec_dicts_for_current_task_key = [
                *self.task_keys_to_task_spec_dicts[self.current_task_key]
            ]
            if self.shuffle:
                self.random.shuffle(self.task_spec_dicts_for_current_task_key)

        new_task_spec_dict = self.task_spec_dicts_for_current_task_key.pop()
        
        if "task_key" not in new_task_spec_dict:
            new_task_spec_dict["task_key"] = self.current_task_key
        else:
            assert self.current_task_key == new_task_spec_dict["task_key"]
        
        return HomeServiceTaskSpec(**new_task_spec_dict)


class HomeServiceTaskSampler(TaskSampler):

    def __init__(
        self,
        stage: str,
        task_keys_to_task_spec_dicts: Dict[str, List[Dict[str, Any]]],
        home_service_env_kwargs: Optional[Dict[str, Any]],
        sensors: SensorSuite,
        max_steps: int,
        discrete_actions: Tuple[str, ...],
        smooth_nav: bool,
        require_done_action: bool,
        force_axis_aligned_start: bool,
        epochs: Union[int, float, str] = "default",
        smoothing_factor: int = 1,
        seed: Optional[int] = None,
        task_spec_in_metrics: bool = False,
    ) -> None:
        
        self.sensors = sensors
        self.stage = stage
        self.main_seed = seed if seed is not None else random.randint(0, 2 * 30 - 1)

        self.task_spec_in_metrics = task_spec_in_metrics
        self.task_keys_to_task_spec_dicts = copy.deepcopy(task_keys_to_task_spec_dicts)

        if isinstance(epochs, str):
            if epochs.lower().strip() != "default":
                raise NotImplementedError(f"Unknown value for `epochs` (=={epochs})")
            epochs = float("inf") if stage == "train" else 1

        self.task_spec_iterator = HomeServiceTaskSpecIterable(
            task_keys_to_task_spec_dicts=self.task_keys_to_task_spec_dicts,
            seed=self.main_seed,
            epochs=epochs,
            shuffle=epochs == float("inf"),
        )
        
        self.env = HomeServiceEnvironment(**home_service_env_kwargs)

        self.task_keys = list(self.task_keys_to_task_spec_dicts.keys())

        self.max_steps = max_steps
        self.discrete_actions = discrete_actions
        self.smooth_nav = smooth_nav
        self.smoothing_factor = smoothing_factor
        self.require_done_action = require_done_action
        self.force_axis_aligned_start = force_axis_aligned_start

        self._last_sampled_task: Optional[HomeServiceTask] = None

    @classmethod
    def from_fixed_simple_pick_and_place_data(
        cls,
        stage: str,
        allowed_task_keys: Optional[Sequence[str]] = None,
        allowed_pickup_objs: Optional[Sequence[str]] = None,
        allowed_start_receps: Optional[Sequence[str]] = None,
        allowed_target_receps: Optional[Sequence[str]] = None,
        allowed_scene_inds: Optional[Sequence[int]] = None,
        randomize_start_rotation: bool = False,
        **init_kwargs,
    ):
        task_keys_to_task_spec_dicts = cls._filter_task_keys_to_task_spec_dicts(
            task_keys_to_task_spec_dicts=cls.load_simple_pick_and_place_data_from_path(
                stage=stage, base_dir=STARTER_HOME_SERVICE_DATA_DIR
            ),
            allowed_task_keys=allowed_task_keys,
            allowed_pickup_objs=allowed_pickup_objs,
            allowed_start_receps=allowed_start_receps,
            allowed_target_receps=allowed_target_receps,
            allowed_scene_inds=allowed_scene_inds,
        )
        if randomize_start_rotation:
            random_gen = random.Random(1)
            for task_key in sorted(task_keys_to_task_spec_dicts):
                for task_spec_dict in task_keys_to_task_spec_dicts[task_key]:
                    for room in task_spec_dict["agent_rotations"]:
                        task_spec_dict["agent_rotations"][room] = 360.0 * random_gen.random()
        
        return cls(
            stage=stage,
            task_keys_to_task_spec_dicts=task_keys_to_task_spec_dicts,
            **init_kwargs
        )

    @classmethod
    def _filter_task_keys_to_task_spec_dicts(
        cls,
        task_keys_to_task_spec_dicts: Dict[str, List[Dict[str, Any]]],
        allowed_task_keys: Optional[Sequence[str]],
        allowed_pickup_objs: Optional[Sequence[str]],
        allowed_start_receps: Optional[Sequence[str]],
        allowed_target_receps: Optional[Sequence[str]],
        allowed_scene_inds: Optional[Sequence[int]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        if allowed_task_keys is not None:
            task_keys_to_task_spec_dicts = {
                task_key: task_keys_to_task_spec_dicts[task_key]
                for task_key in allowed_task_keys
            }

        filtered_keys = []
        if (
            allowed_pickup_objs is not None
            or allowed_start_receps is not None
            or allowed_target_receps is not None
        ):
            for task_key in task_keys_to_task_spec_dicts:
                splits = task_key.split("_")
                pickup_allowed = False
                start_recep_allowed = False
                target_recep_allowed = False

                if splits[0] == "Pick":
                    pickup_obj = splits[1]
                    start_recep = splits[3]
                    target_recep = splits[6]

                else:
                    pickup_obj = splits[2]
                    start_recep = splits[4]
                    target_recep = None
            
                if allowed_pickup_objs is not None:
                    if pickup_obj in allowed_pickup_objs:
                        pickup_allowed = True
                else:
                    pickup_allowed = True
                
                if allowed_start_receps is not None:
                    if start_recep in allowed_start_receps:
                        start_recep_allowed = True
                else: 
                    start_recep_allowed = True
                
                if allowed_target_receps is not None:
                    if "User" in allowed_target_receps and splits[0] == "Bring":
                        target_recep_allowed = True
                    elif target_recep is not None and target_recep in allowed_target_receps:
                        target_recep_allowed = True
                else:
                    target_recep_allowed = True

                if pickup_allowed and start_recep_allowed and target_recep_allowed:
                    filtered_keys.append(task_key)
        else:
            filtered_keys = [task_key for task_key in task_keys_to_task_spec_dicts]

        task_keys_to_task_spec_dicts = {
            task_key: task_keys_to_task_spec_dicts[task_key]
            for task_key in filtered_keys
        }

        if allowed_scene_inds is not None:
            for task_key, task_spec_dicts in task_keys_to_task_spec_dicts.items():
                task_keys_to_task_spec_dicts[task_key] = [
                    task_spec_dict
                    for task_spec_dict in task_spec_dicts
                    if task_spec_dict["scene_index"] in allowed_scene_inds
                ]
            
        return task_keys_to_task_spec_dicts

    @classmethod
    def load_simple_pick_and_place_data_from_path(
        cls,
        stage: str,
        base_dir: Optional[str] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        stage = stage.lower()

        if stage == "valid":
            stage = "val"

        data_path = os.path.abspath(os.path.join(base_dir, f"{stage}.pkl.gz"))
        if not os.path.exists(data_path):
            raise RuntimeError(f"No data at path {data_path}")

        data = compress_pickle.load(path=data_path)
        for task_key in data:
            for ind, task_spec_dict in enumerate(data[task_key]):
                task_spec_dict["task_key"] = task_key

                if "index" not in task_spec_dict:
                    task_spec_dict["index"] = ind
                
                if "stage" not in task_spec_dict:
                    task_spec_dict["stage"] = stage

        return data

    @property
    def length(self) -> float:
        return self.task_spec_iterator.length

    @property
    def total_unique(self):
        return self.task_spec_iterator.total_unique

    @property
    def last_sampled_task(self) -> Optional[HomeServiceTask]:
        return self._last_sampled_task

    @property
    def all_observation_spaces_equal(self) -> bool:
        return True

    def close(self) -> None:
        try:
            self.env.stop()
        except Exception as _:
            pass
    
    def reset(self) -> None:
        self.task_spec_iterator.reset()
        self._last_sampled_task = None
    
    def set_seed(self, seed: int) -> None:
        self.task_spec_iterator.seed = seed
        self.main_seed = seed

    @property
    def current_task_spec(self) -> HomeServiceTaskSpec:
        return self.env.current_task_spec

    def next_task(
        self, 
        forced_task_spec: Optional[HomeServiceTaskSpec] = None,
        forced_start_scene_type: Optional[str] = None,
        **kwargs
    ) -> Optional[HomeServiceTask]:

        try:
            if forced_task_spec is None:
                task_spec: HomeServiceTaskSpec = next(self.task_spec_iterator)
            else:
                task_spec = forced_task_spec
        except StopIteration:
            self._last_sampled_task = None
            return self._last_sampled_task

        runtime_sample = task_spec.runtime_sample

        try:
            self.env.reset(
                task_spec=task_spec,
                force_axis_aligned_start=self.force_axis_aligned_start,
                scene_type=forced_start_scene_type,
            )

            self._last_sampled_task = HomeServiceTask(
                sensors=self.sensors,
                env=self.env,
                max_steps=self.max_steps,
                discrete_actions=self.discrete_actions,
                smooth_nav=self.smooth_nav,
                smoothing_factor=self.smoothing_factor,
                force_axis_aligned_start=self.force_axis_aligned_start,
                require_done_action=self.require_done_action,
                task_spec_in_metrics=self.task_spec_in_metrics,
            )
            
        except Exception as e:
            if runtime_sample:
                get_logger().error(
                    "Encountered exception while sampling a next task."
                    " As this next task was a 'runtime sample' we are"
                    " simply returning the next task."
                )
                get_logger().error(traceback.format_exc())
                return self.next_task()
            else:
                raise e

        return self._last_sampled_task
