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
    HomeServiceGreedyActionExpert,
    HomeServiceSubtaskActionExpert,
    ShortestPathNavigator,
)
from env.utils import (
    HomeServiceActionSpace,
    include_object_data,
    get_pose_info
)
from env.subtasks import HomeServiceSimpleSubtaskPlanner

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
        expert_exploration_enabled: bool = True,
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
        self.expert_exploration_enabled = expert_exploration_enabled

        self._took_end_action: bool = False
        self._obs = None
        
        self.action_expert = None
        self.actions_taken = []
        self.actions_taken_success = []
        self.agent_locs = [self.env.get_agent_location()]

    def create_navigator(self):
        return ShortestPathNavigator(
            env=self.env,
            grid_size=self.env.grid_size,
            include_move_left_right=all(
                f"move_{k}" in self.action_names() for k in ["left", "right"]
            )
        )

    def create_greedy_expert(self):
        return HomeServiceGreedyActionExpert(
            task=self,
            shortest_path_navigator=self.env.shortest_path_navigator,
            exploration_enabled=self.expert_exploration_enabled,
        )

    def create_subtask_expert(self):
        return HomeServiceSubtaskActionExpert(
            task=self,
            shortest_path_navigator=self.env.shortest_path_navigator,
        )
    
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
        start_energies = env.pose_difference_energy(gps, ips)
        end_energies = env.pose_difference_energy(gps, cps)
        start_energy = start_energies.sum()
        end_energy = end_energies.sum()
        change_energies = env.pose_difference_energy(ips, cps)
        change_energy = change_energies.sum()

        metrics = super().metrics()
        
        task_info = metrics["task_info"]
        task_info["scene"] = self.env.scene
        task_info["target_object"] = self.env.target_object_id
        task_info["target_receptacle"] = self.env.target_recep_id
        if self.task_spec_in_metrics:
            task_info["task_spec"] = {**self.env.current_task_spec.__dict__}
            task_info["poses"] = self.env.poses
            task_info["gps_vs_cps"] = self.env.compare_poses(gps, cps)
            task_info["ips_vs_cps"] = self.env.compare_poses(ips, cps)
            task_info["gps_vs_ips"] = self.env.compare_poses(gps, ips)

        task_info["actions"] = self.actions_taken
        task_info["action_successes"] = self.actions_taken_success
        task_info["unique_id"] = self.env.current_task_spec.unique_id
        
        # TODO: Implement metrics
        put_success = False
        target_object = next(
            cp for cp in cps if cp["objectId"] == self.env.target_object_id
        )

        try:
            metrics["success"] = (
                float(end_energy == 0)
                or (
                    target_object["parentReceptacles"] is not None
                    and self.env.target_recep_id in target_object["parentReceptacles"]
                )
            )
        except:
            import pdb; pdb.set_trace()
        metrics["start_energy"] = start_energy
        metrics["end_energy"] = end_energy
        metrics["change_energy"] = change_energy

        if self.action_expert is not None:
            metrics["expert_actions"] = copy.deepcopy(
                [
                    self.action_names()[expert_action]
                    for expert_action in self.action_expert.expert_action_list[:-1]
                ]
            )

        if (
            self.action_expert is not None
            and hasattr(self.action_expert, "planner")
            and self.action_expert.planner is not None
        ):
            metrics["expert_subtasks"] = copy.deepcopy(
                [
                    self.action_expert.planner.subtask_str(expert_subtask)
                    for expert_subtask in self.action_expert.expert_subtask_list
                ]
            )

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

    def _judge(self, obs, action, next_obs, action_success,) -> float:
        """Return the reward from a new (s, a, s')."""
        # TODO: Implement reward function
        reward = 0.

        return reward

    def find_object_by_type(self, object_type: str, visible: bool = True, openable: bool = False):
        with include_object_data(self.env.controller):
            metadata = self.env.last_event.metadata

            possible_objects = [
                o
                for o in metadata["objects"]
                if (
                    (o["objectType"] == object_type)
                    and (o["visible"] or not visible)
                    and (o["openable"] or not openable)
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
                            objectId=obj_id,
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
                            objectId=obj_id,
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
                            objectId=obj_id,
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
                        "MakeObjectBreakable", objectId=self.env.held_object["objectId"]
                    )

                    object_type = stringcase.pascalcase(
                        action_name.replace("put_", "")
                    )

                    # Trying to put object by snapping
                    # Determine whether the agent is located valid position
                    agent = metadata["agent"]
                    held_obj = self.env.held_object
                    goal_pose = self.env.object_id_to_target_pose[held_obj["objectId"]]
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
                    
                    else:
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
        elif action_name == "drop_held_object_with_snap":
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

        if self.action_expert is not None:
            self.action_expert.update(
                action_taken=action,
                action_success=action_success,
            )

        reward = self._judge(
            obs=self._obs,      # s0
            action=action,      # a0
            next_obs=self.get_observations(),      # s1
            action_success=action_success,
        )

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
        # if self.expert is not None:
        #     self.expert.update(
        #         action_taken=action, action_success=step_result.info["action_success"]
        #     )
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
        tasks_to_houses_to_task_spec_dicts: Dict[str, Dict[str, List[Dict]]],
        seed: int,
        epochs: Union[int, float],
        shuffle: bool = True,
    ):
        assert epochs >= 1

        self.tasks_to_houses_to_task_spec_dicts = {
            k: {**v} for k, v in tasks_to_houses_to_task_spec_dicts.items()
        }
        assert (
            len(self.tasks_to_houses_to_task_spec_dicts) != 0
            and all(
                len(self.tasks_to_houses_to_task_spec_dicts[task]) != 0
                for task in self.tasks_to_houses_to_task_spec_dicts
            )
            and all(
                len(self.tasks_to_houses_to_task_spec_dicts[task][house]) != 0
                for task, houses in self.tasks_to_houses_to_task_spec_dicts.items()
                for house in houses
            )
        )

        self._seed = seed
        self.random = random.Random(self.seed)
        self.start_epochs = epochs
        self.remaining_epochs = epochs
        self.shuffle = shuffle

        self.remaining_tasks: List[str] = []
        self.remaining_houses: List[str] = []
        self.houses_to_task_spec_dicts_for_current_task: Dict[str, List[Dict[str, Any]]] = {}
        self.task_spec_dicts_for_current_task_house: List[Dict[str, Any]] = []
        self.current_task: Optional[str] = None
        self.current_house: Optional[str] = None

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
            len(self.task_spec_dicts_for_current_task_house)
            + sum(
                len(self.houses_to_task_spec_dicts_for_current_task[house])
                for house in self.remaining_houses
            )
            + sum(
                sum(
                    len(self.tasks_to_houses_to_task_spec_dicts[task][house])
                    for house in self.tasks_to_houses_to_task_spec_dicts[task]
                )
                for task in self.remaining_tasks
            )
            + self.remaining_epochs
            * sum(
                sum(
                    len(v) for v in houses.values()
                )
                for houses in self.tasks_to_houses_to_task_spec_dicts.values()
            )
        )
    
    @property
    def total_unique(self):
        return sum(
            sum(
                len(v) for v in houses.values()
            )
            for houses in self.tasks_to_houses_to_task_spec_dicts.values()
        )

    def reset(self):
        self.random.seed(self.seed)
        self.remaining_epochs = self.start_epochs
        self.remaining_tasks.clear()
        self.remaining_houses.clear()
        self.houses_to_task_spec_dicts_for_current_task.clear()
        self.task_spec_dicts_for_current_task_house.clear()
        self.current_task = None
        self.current_house = None

    def refresh_remaining_houses(self):
        assert self.current_task is not None
        self.remaining_houses = list(
            sorted(
                self.houses_to_task_spec_dicts_for_current_task.keys(),
                key=lambda s: int(
                    s.replace("train_", "")
                    .replace("val_", "")
                    .replace("test_", "")
                ),
            )
        )
        if self.shuffle:
            self.random.shuffle(self.remaining_houses)
        return self.remaining_houses

    def refresh_remaining_tasks(self):
        if self.remaining_epochs <= 0:
            raise StopIteration
        self.remaining_epochs -= 1

        self.remaining_tasks = list(sorted(self.tasks_to_houses_to_task_spec_dicts.keys()))
        if self.shuffle:
            self.random.shuffle(self.remaining_tasks)
        return self.remaining_tasks

    def __next__(self) -> HomeServiceTaskSpec:
        if len(self.task_spec_dicts_for_current_task_house) == 0:
            if len(self.houses_to_task_spec_dicts_for_current_task) == 0:
                if len(self.remaining_tasks) == 0:
                    self.refresh_remaining_tasks()
                self.current_task = self.remaining_tasks.pop()
                self.houses_to_task_spec_dicts_for_current_task = copy.deepcopy(
                    self.tasks_to_houses_to_task_spec_dicts[self.current_task]
                )

            if len(self.remaining_houses) == 0:
                self.refresh_remaining_houses()
            self.current_house = self.remaining_houses.pop()
            
            self.task_spec_dicts_for_current_task_house = [
                *self.houses_to_task_spec_dicts_for_current_task.pop(self.current_house)
            ]
            if self.shuffle:
                self.random.shuffle(self.task_spec_dicts_for_current_task_house)
        
        new_task_spec_dict = self.preprocess_spec_dict(
            self.task_spec_dicts_for_current_task_house.pop()
        )

        if "task" not in new_task_spec_dict:
            new_task_spec_dict["task"] = self.current_task
        else:
            assert self.current_task == new_task_spec_dict["task"]
        
        if "scene" not in new_task_spec_dict:
            new_task_spec_dict["scene"] = self.current_house
        else:
            assert self.current_house == new_task_spec_dict["scene"]
        
        return HomeServiceTaskSpec(**new_task_spec_dict)

    def preprocess_spec_dict(self, spec_dict):
        return compress_pickle.loads(spec_dict, compression="gzip")


class HomeServiceTaskSampler(TaskSampler):

    def __init__(
        self,
        stage: str,
        tasks_to_houses_to_task_spec_dicts: Dict[str, Dict[str, List[Dict[str, Any]]]],
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
        repeats_per_task: Optional[int] = None,
        task_spec_in_metrics: bool = False,
        expert_exploration_enabled: bool = True,
    ) -> None:
        
        assert (
            repeats_per_task is None
            or repeats_per_task >= 1
        ), f"`repeats_per_task` (=={repeats_per_task}) must be >= 1"
        self.sensors = sensors
        self.stage = stage
        self.main_seed = seed if seed is not None else random.randint(0, 2 * 30 - 1)
        self.repeats_per_task = (
            1
            if repeats_per_task is None
            else repeats_per_task
        )
        self.cur_repeat_count = 0

        self.task_spec_in_metrics = task_spec_in_metrics
        self.tasks_to_houses_to_task_spec_dicts = copy.deepcopy(tasks_to_houses_to_task_spec_dicts)
        if isinstance(epochs, str):
            if epochs.lower().strip() != "default":
                raise NotImplementedError(f"Unknown value for `epochs` (=={epochs})")
            epochs = float("inf") if stage == "train" else 1

        self.task_spec_iterator: HomeServiceTaskSpecIterable = self.make_task_spec_iterable(epochs)
        self.env: HomeServiceEnvironment = self.create_env(**home_service_env_kwargs)
        self.tasks = list(self.tasks_to_houses_to_task_spec_dicts.keys())

        self.max_steps = max_steps
        self.discrete_actions = discrete_actions
        self.smooth_nav = smooth_nav
        self.smoothing_factor = smoothing_factor
        self.require_done_action = require_done_action
        self.force_axis_aligned_start = force_axis_aligned_start
        self.expert_exploration_enabled = expert_exploration_enabled

        self._last_sampled_task: Optional[HomeServiceTask] = None

    def create_env(self, **kwargs) -> HomeServiceEnvironment:
        return HomeServiceEnvironment(**kwargs)

    def make_task_spec_iterable(self, epochs) -> HomeServiceTaskSpecIterable:
        return HomeServiceTaskSpecIterable(
            tasks_to_houses_to_task_spec_dicts=self.tasks_to_houses_to_task_spec_dicts,
            seed=self.main_seed,
            epochs=epochs,
            shuffle=epochs == float("inf"),
        )

    @classmethod
    def get_base_dir(cls) -> str:
        return STARTER_HOME_SERVICE_DATA_DIR

    @classmethod
    def from_fixed_dataset(
        cls,
        stage: str,
        allowed_tasks: Optional[Sequence[str]] = None,
        allowed_pickup_objs: Optional[Sequence[str]] = None,
        allowed_target_receps: Optional[Sequence[str]] = None,
        randomize_start_rotation: bool = False,
        **init_kwargs,
    ):
        tasks_to_houses_to_task_spec_dicts=cls.load_home_service_data_from_path(
            stage=stage, base_dir=cls.get_base_dir()
        )
        tasks_to_houses_to_task_spec_dicts = cls._filter_tasks_to_houses_to_task_spec_dicts(
            tasks_to_houses_to_task_spec_dicts=tasks_to_houses_to_task_spec_dicts,
            allowed_tasks=allowed_tasks,
            allowed_pickup_objs=allowed_pickup_objs,
            allowed_target_receps=allowed_target_receps,
        )
        if randomize_start_rotation:
            random_gen = random.Random(1)
            for task_key in sorted(tasks_to_houses_to_task_spec_dicts):
                for task_spec_dict in tasks_to_houses_to_task_spec_dicts[task_key]:
                    for room in task_spec_dict["agent_rotations"]:
                        task_spec_dict["agent_rotations"][room] = 360.0 * random_gen.random()
        
        return cls(
            stage=stage,
            tasks_to_houses_to_task_spec_dicts=tasks_to_houses_to_task_spec_dicts,
            **init_kwargs,
        )

    @classmethod
    def _filter_tasks_to_houses_to_task_spec_dicts(
        cls,
        tasks_to_houses_to_task_spec_dicts: Dict[str, List[Dict[str, Any]]],
        allowed_tasks: Optional[Sequence[str]],
        allowed_pickup_objs: Optional[Sequence[str]],
        allowed_target_receps: Optional[Sequence[str]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        if allowed_tasks is not None:
            tasks_to_houses_to_task_spec_dicts = {
                task: tasks_to_houses_to_task_spec_dicts[task]
                for task in allowed_tasks
                if task in tasks_to_houses_to_task_spec_dicts
            }

        filtered_tasks = []
        if (
            allowed_pickup_objs is not None
            or allowed_target_receps is not None
        ):
            for task in tasks_to_houses_to_task_spec_dicts:
                pickup_allowed = False
                target_recep_allowed = False
                if task.startswith("pick_and_place"):
                    pickup_obj = task.split("_")[-2]
                    target_recep = task.split("_")[-1]
                elif task.startswith("bring_me"):
                    pickup_obj = task.split("_")[-1]
                    target_recep = "User"
            
                if allowed_pickup_objs is not None:
                    if pickup_obj in allowed_pickup_objs:
                        pickup_allowed = True
                else:
                    pickup_allowed = True
                                
                if allowed_target_receps is not None:
                    if target_recep in allowed_target_receps:
                        target_recep_allowed = True
                else:
                    target_recep_allowed = True

                if pickup_allowed and target_recep_allowed:
                    filtered_tasks.append(task)
        else:
            filtered_tasks = [task for task in tasks_to_houses_to_task_spec_dicts]

        tasks_to_houses_to_task_spec_dicts = {
            task: tasks_to_houses_to_task_spec_dicts[task]
            for task in filtered_tasks
        }
            
        return tasks_to_houses_to_task_spec_dicts

    @classmethod
    def load_home_service_data_from_path(
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

        new_data = {}
        for task in data:
            task_key = task[len(stage)+1:]
            if task_key not in new_data:
                new_data[task_key] = {}
            for house_ind, house in enumerate(
                sorted(
                    data[task], 
                    key=lambda s: int(
                        s.replace("train_", "").
                        replace("val_", "").
                        replace("test_", "")
                    )
                )
            ):
                if house not in new_data[task_key]:
                    new_data[task_key][house] = []
                for scene_ind, task_spec_dict in enumerate(data[task][house]):
                    new_data[task_key][house].append(
                        compress_pickle.dumps(
                            task_spec_dict, compression="gzip"
                        )
                    )

        return new_data

    @property
    def length(self) -> float:
        count = self.task_spec_iterator.length * self.repeats_per_task
        if (
            self.last_sampled_task is not None
            and self.cur_repeat_count < self.repeats_per_task
        ):
            count += self.repeats_per_task - self.cur_repeat_count

        return count

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
        self.cur_repeat_count = 0
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
        **kwargs
    ) -> Optional[HomeServiceTask]:

        should_repeat = (
            forced_task_spec is None
            and self.cur_repeat_count < self.repeats_per_task
        )

        if (
            self.last_sampled_task is None
            or not should_repeat
        ):
            self.cur_repeat_count = 0
        
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
                if runtime_sample:
                    raise NotImplementedError
                
                self.env.reset(
                    task_spec=task_spec,
                    force_axis_aligned_start=self.force_axis_aligned_start,
                )
                
                self.cur_repeat_count += 1
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
                    expert_exploration_enabled=self.expert_exploration_enabled,
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
        
        else:
            self.cur_repeat_count += 1
            self.env.reset(
                task_spec=self.env.current_task_spec,
                force_axis_aligned_start=self.force_axis_aligned_start,
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
                expert_exploration_enabled=self.expert_exploration_enabled,
            )

        return self._last_sampled_task
