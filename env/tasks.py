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
from env.constants import DEFAULT_COMPATIBLE_RECEPTACLES, OBJECT_TYPES_WITH_PROPERTIES, SCENE_TO_SCENE_TYPE, STARTER_HOME_SERVICE_DATA_DIR, STARTER_HOME_SERVICE_SIMPLE_PICK_AND_PLACE_DATA_DIR, STARTER_REARRANGE_DATA_DIR, STEP_SIZE
from env.environment import (
    HomeServiceSimpleTaskOrderTaskSpec,
    HomeServiceTHOREnvironment,
    HomeServiceTaskSpec,
)
from env.expert import (
    GreedySimplePickAndPlaceExpert,
    ShortestPathNavigatorTHOR,
    SubTaskExpert,
)
from env.utils import (
    HomeServiceActionSpace,
    include_object_data,
)

from ..sEDM.test_edm import sEDM_model

class HomeServiceTaskType(enum.Enum):

    SIMPLE_PICK_AND_PLACE = "SimplePickAndPlace"
    REARRANGE = "Rearrange"


class AbstractHomeServiceTask(Task, ABC):
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


class HomeServiceBaseTask(AbstractHomeServiceTask):
    def __init__(
        self,
        sensors: SensorSuite,
        env: HomeServiceTHOREnvironment,
        max_steps: int,
        discrete_actions: Tuple[str, ...],
        smooth_nav: bool = False,
        smoothing_factor: int = 1,
        require_done_action: bool = False,
        task_spec_in_metrics: bool = False,
    ) -> None:

        super().__init__(
            env=env, sensors=sensors, task_info=dict(), max_steps=max_steps,
        )
        self.env = env
        self.discrete_actions = discrete_actions
        self.smooth_nav = smooth_nav
        self.smoothing_factor = smoothing_factor if self.smooth_nav else 1
        self.require_done_action = require_done_action
        self.task_spec_in_metrics = task_spec_in_metrics

        self._took_end_action: bool = False

        self.task_planner = None
        self.greedy_expert = None
        self._subtask_step = 0

        self.actions_taken = []
        self.actions_taken_success = []
        self.agent_locs = [self.env.get_agent_location()]

    @property
    def action_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(len(self.action_names()))

    @property
    def num_subtasks(self) -> int:
        return len(self.query_planner())

    @property
    def planned_task(self):
        return self.query_planner()

    @property
    def current_subtask(self):
        return (
            self.planned_task[self.subtask_step()] 
            if self.subtask_step() < self.num_subtasks
            else ("Done", None, None)
        )
    
    def subtask_step(self) -> int:
        return self._subtask_step

    def rollback_subtask(self):
        self._subtask_step -= 1

    def is_current_subtask_done(self):
        subtask_action, subtask_target, subtask_place = self.current_subtask
        metadata = self.env.last_event.metadata
        if subtask_action == "Done":
            return True

        elif subtask_action == "Navigate":
            assert subtask_place is None
            cur_subtask_target = next(
                (o for o in metadata["objects"] if o["objectType"] == subtask_target), None
            )
            assert cur_subtask_target is not None
            if cur_subtask_target["visible"]:
                self._subtask_step += 1
                return True

        elif subtask_action == "Pickup":
            assert subtask_place is None
            if metadata["lastActionSuccess"] and (
                metadata["lastAction"] == f"{subtask_action}Object"
            ) and (
                self.env.held_object["objectType"] == subtask_target
            ):
                self._subtask_step += 1
                return True

        elif subtask_action in ["Open", "Close"]:
            assert subtask_place is None
            if metadata["lastActionSuccess"] and (
                metadata["lastAction"] == f"{subtask_action}Object"
            ):
                self._subtask_step += 1
                return True

        elif subtask_action == "Put":
            assert subtask_place is not None
            cur_subtask_target = next(
                (o for o in metadata["objects"] if o["objectType"] == subtask_target), None
            )
            assert cur_subtask_target is not None
            cur_subtask_place = next(
                (o for o in metadata["objects"] if o["objectType"] == subtask_place), None
            )
            assert cur_subtask_place is not None

            if metadata["lastActionSuccess"] and (
                metadata["lastAction"] == f"{subtask_action}Object"
            ) and (
                self.env.held_object is None
            ):
                if cur_subtask_place["objectId"] in cur_subtask_target["parentReceptacles"]:
                    self._subtask_step += 1
                    return True

        elif subtask_action == "Goto":
            if metadata["lastActionSuccess"] and (
                SCENE_TO_SCENE_TYPE[self.env.scene] == subtask_target
            ):
                self._subtask_step += 1
                return True

        else:
            raise NotImplementedError(
                f"Action {subtask_action} for the subtasks is not implemented"
            )

        return False

    def close(self) -> None:
        try:
            self.env.step()
        except Exception as _:
            pass

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
            and self.current_subtask[0] == "Done"
        )

    def query_planner(self):
        return []

    def _judge(self, obs, action, next_obs, action_success) -> float:
        """Return the reward from a new (s, a, s')."""
        reward = -0.05
        if self.is_current_subtask_done():
            reward += 1

        if self.current_subtask[0] == "Done":
            reward += 5

        return reward

    def _step(self, action: int) -> RLStepResult:
        """
        action: is the index of the action from self.action_names()
        """
        obs = [self.get_observations()]
        action_name = self.action_names()[action]

        if action_name.startswith("pickup"):
            with include_object_data(self.env.controller):
                metadata = self.env.last_event.metadata

                if len(metadata["inventoryObjects"]) != 0:
                    action_success = False
                else:
                    object_type = stringcase.pascalcase(
                        action_name.replace("pickup_", "")
                    )
                    possible_objects = [
                        o
                        for o in metadata["objects"]
                        if o["visible"] and o["objectType"] == object_type
                    ]

                    possible_objects = sorted(
                        possible_objects, key=lambda po: (po["distance"], po["name"])
                    )

                    object_before = None
                    if len(possible_objects) > 0:
                        object_before = possible_objects[0]
                        object_id = object_before["objectId"]

                    if object_before is not None:
                        self.env.controller.step(
                            "PickupObject",
                            objectId=object_id,
                            **self.env.physics_step_kwargs,
                        )
                        action_success = self.env.last_event.metadata["lastActionSuccess"]
                    else:
                        action_success = False

                    if action_success and self.env.held_object is None:
                        get_logger().warning(
                            f"`PickupObject` was successful in picking up {object_id} but we're not holding"
                            f" any object! Current task spec: \n {self.env.current_task_spec}"
                        )
                        action_success = False

        elif action_name.startswith("open_by_type"):
            object_type = stringcase.pascalcase(
                action_name.replace("open_by_type_", "")
            )
            with include_object_data(self.env.controller):
                metadata = self.env.last_event.metadata
                pickup_target = self.env.current_task_spec.pickup_target
                place_target = self.env.current_task_spec.place_target

                pickup_target_openable_receptacle = None
                if pickup_target["parentReceptacles"] is not None:
                    for obj in metadata["objects"]:
                        if (
                            obj["openable"]
                            and obj["objectId"] in pickup_target["parentReceptacles"]
                        ):
                            pickup_target_openable_receptacle = obj
                            break

                object_before = None
                pickup_target_openable_receptacle_name = (
                    pickup_target_openable_receptacle["name"]
                    if pickup_target_openable_receptacle is not None and "name" in pickup_target_openable_receptacle
                    else None
                )

                for obj in metadata["objects"]:
                    if (
                        obj["visible"]
                        and obj["openable"]
                        and obj["objectType"] == object_type
                        and (
                            obj["name"] == place_target["name"]
                            or obj["name"] == pickup_target_openable_receptacle_name
                        )
                    ):
                        object_before = obj
                        break

                if object_before is not None:
                    if object_before["openness"] > 0.0:
                        self.env.controller.step(
                            "CloseObject",
                            objectId=object_before["objectId"],
                            **self.env.physics_step_kwargs,
                        )
                    
                    self.env.controller.step(
                        "OpenObject",
                        objectId=object_before["objectId"],
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
                metadata = self.env.last_event.metadata
                pickup_target = self.env.current_task_spec.pickup_target
                place_target = self.env.current_task_spec.place_target

                pickup_target_openable_receptacle = None
                if pickup_target["parentReceptacles"] is not None:
                    for obj in metadata["objects"]:
                        if (
                            obj["openable"]
                            and obj["objectId"] in pickup_target["parentReceptacles"]
                        ):
                            pickup_target_openable_receptacle = obj
                            break
                        
                object_before = None
                pickup_target_openable_receptacle_name = (
                    pickup_target_openable_receptacle["name"]
                    if pickup_target_openable_receptacle is not None and "name" in pickup_target_openable_receptacle
                    else None
                )

                for obj in metadata["objects"]:
                    if (
                        obj["visible"]
                        and obj["openable"]
                        and obj["objectType"] == object_type
                        and (
                            obj["name"] == place_target["name"]
                            or obj["name"] == pickup_target_openable_receptacle_name
                        )
                    ):
                        object_before = obj
                        break

                if object_before is not None:
                    if object_before["openness"] > 0.0:
                        self.env.controller.step(
                            "CloseObject",
                            objectId=object_before["objectId"],
                            **self.env.physics_step_kwargs,
                        )
                    
                    action_success = self.env.last_event.metadata["lastActionSuccess"]
                else:
                    action_success = False

        elif action_name.startswith("put_by_type"):
            object_type = stringcase.pascalcase(
                action_name.replace("put_by_type_", "")
            )
            with include_object_data(self.env.controller):
                metadata = self.env.last_event.metadata
                pickup_target = self.env.current_task_spec.pickup_target
                place_target = self.env.current_task_spec.place_target

                if len(metadata["inventoryObjects"]) == 0:
                    action_success = False
                else:
                    object_before = None
                    for obj in metadata["objects"]:
                        if (
                            obj["visible"]
                            and obj["receptacle"]
                            and obj["objectType"] == object_type
                            and obj["name"] == place_target["name"]
                        ):
                            object_before = obj
                            break
                    
                    if object_before is not None:
                        self.env.controller.step(
                            "PutObject",
                            objectId=object_before["objectId"],
                            **self.env.physics_step_kwargs,
                        )
                        action_success = self.env.last_event.metadata["lastActionSuccess"]
                    else:
                        action_success = False

        elif action_name.startswith(("move", "rotate")):
            opposites = {
                "ahead": "back",
                "back": "ahead",
                "right": "left",
                "left": "right",
            }
            direction = action_name.split("_")[-1]
            opposite_direction = opposites[direction]
            for i in range(self.smoothing_factor):
                action_success = getattr(self.env, action_name)()
                # obs.append(self.get_observations())

                if not action_success:
                    # obs.pop()
                    for j in range(i):
                        getattr(self.env, "_".join([action_name.split("_")[0], opposite_direction]))()
                        # obs.pop()
                    break
        
        elif action_name.startswith("look"):
            opposites = {
                "up": "down",
                "down": "up",
            }
            direction = action_name.split("_")[-1]
            opposite_direction = opposites[direction]
            for i in range(self.smoothing_factor):
                action_success = getattr(self.env, action_name)(1.0 / self.smoothing_factor)
                # obs.append(self.get_observations())

                if not action_success:
                    # obs.pop()
                    for j in range(i):
                        getattr(self.env, "_".join([action_name.split("_")[0], opposite_direction]))(1.0 / self.smoothing_factor)
                        # obs.pop()
                    break

        elif action_name.startswith(("stand", "crouch")):
            action_success = getattr(self.env, action_name)()
        elif action_name == "done":
            self._took_end_action = True
            # action_success = getattr(self.env, action_name)()
            action_success = True
        elif action_name == "pass":
            event = self.env.controller.step("Pass")
            action_success = event.metadata["lastActionSuccess"]
        elif action_name.startswith("goto"):
            scene_type = "_".join(action_name.split("_")[1:])
            assert scene_type in ("kitchen", "living_room", "bedroom", "bathroom")
            self.env.reset(
                task_spec=self.env.current_task_spec,
                scene_type=stringcase.pascalcase(scene_type)
            )
            action_success = self.env.last_event.metadata['lastActionSuccess']
        else:
            raise RuntimeError(
                f"Action '{action_name}' is not in the action space {HomeServiceActionSpace}"
            )

        self.actions_taken.append(action_name)
        self.actions_taken_success.append(action_success)
        if self.task_spec_in_metrics:
            self.agent_locs.appned(self.env.get_agent_location())
        
        return RLStepResult(
            observation=None,
            reward=self._judge(
                obs=obs[0],
                action=action,
                next_obs=self.get_observations(),
                action_success=action_success,
            ),
            done=self.is_done(),
            info={"action_name": action_name, "action_success": action_success},
        )

    def step(self, action: int) -> RLStepResult:
        step_result = super().step(action=action)
        if self.greedy_expert is not None:
            self.greedy_expert.update(
                action_taken=action, action_success=step_result.info["action_success"]
            )
        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=step_result.reward,
            done=step_result.done,
            info=step_result.info,
        )
        return step_result


class HomeServiceSimplePickAndPlaceTask(HomeServiceBaseTask):
    def __init__(
        self,
        **init_kwargs,
    ):
        super().__init__(**init_kwargs)
        self.greedy_expert: Optional[GreedySimplePickAndPlaceExpert] = None

    def query_planner(self, **kwargs) -> Sequence[Tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]]:
        """
        Query task planning result from task planner
        self.task_planner = TaskPlanner(
            task=self,
        )
        """
        target_object = self.env.current_task_spec.pickup_target['name'].split("_")[0]
        target_place = self.env.current_task_spec.place_target['name'].split("_")[0]
        self.task_planner = sEDM_model()
        task_plan = self.task_planner.inference(target_object=target_object, target_place=target_place)
        
        if self.task_planner is None:
            task_plan = []
            pickup_object = self.env.current_task_spec.pickup_object
            start_receptacle = self.env.current_task_spec.start_receptacle
            place_receptacle = self.env.current_task_spec.place_receptacle
            
            start_scene = self.env.current_task_spec.start_scene
            start_scene_type = SCENE_TO_SCENE_TYPE[start_scene]
            target_scene = self.env.current_task_spec.target_scene
            target_scene_type = SCENE_TO_SCENE_TYPE[target_scene]
            task_plan.append(("Goto", target_scene_type, None))
            task_plan.append(("Scan", None, None))
            task_plan.append(("Navigate", pickup_object, None))    
            task_plan.append(("Pickup", pickup_object, None))
            if place_receptacle != "User":
                task_plan.append(("Navigate", place_receptacle, None))
                task_plan.append(("Put", pickup_object, place_receptacle))
            else:
                task_plan.append(("Goto", start_scene_type, None))

        return task_plan
        
    def query_expert(self, **kwargs) -> Tuple[Any, bool]:
        if self.greedy_expert is None:
            if not hasattr(self.env, "shortest_path_navigator"):
                self.env.shortest_path_navigator = ShortestPathNavigatorTHOR(
                    controller = self.env.controller,
                    grid_size=STEP_SIZE,
                    include_move_left_right=all(
                        f"move_{k}" in self.action_names() for k in ["left", "right"]
                    ),
                )
            
            # self.greedy_expert = GreedySimplePickAndPlaceExpert(
            #     task=self,
            #     shortest_path_navigator=self.env.shortest_path_navigator,
            # )
            self.greedy_expert = SubTaskExpert(
                task=self,
                shortest_path_navigator=self.env.shortest_path_navigator,
            )
            
        action = self.greedy_expert.expert_action
        if action is None:
            return 0, False
        else:
            return action, True

    def metrics(self) -> Dict[str, Any]:
        if not self.is_done():
            return {}
        
        env = self.env
        pickup_target = env.current_task_spec.pickup_target
        place_target = env.current_task_spec.place_target

        target_object = next(
            (
                obj
                for obj in env.last_event.metadata["objects"]
                if obj["name"] == pickup_target["name"]
            ), None
        )
        assert target_object is not None

        possible_place_objects = [
            obj for obj in env.last_event.metadata["objects"]
            if obj["objectType"] == place_target["objectType"]
        ]
        assert len(possible_place_objects) > 0

        receptacle = None
        if target_object["parentReceptacles"] is not None:
            receptacle = next(
                (
                    o for o in possible_place_objects
                    if o['objectId'] in target_object["parentReceptacles"]
                ), None
            )
        

        metrics = {
            **super().metrics(),
            **{
                "success": float(True if receptacle is not None else False),
            }
        }

        task_info = metrics["task_info"]
        task_info["scene"] = env.scene
        task_info["index"] = env.current_task_spec.metrics.get("index")
        task_info["stage"] = env.current_task_spec.stage
        del metrics["task_info"]

        if self.task_spec_in_metrics:
            task_info["task_spec"] = {**env.current_task_spec.__dict__}

        task_info["actions_taken"] = self.actions_taken
        task_info["actions_taken_success"] = self.actions_taken_success
        task_info["unique_id"] = env.current_task_spec.unique_id if not env.current_task_spec.runtime_sample else None

        metrics = {
            "task_info": task_info,
            **metrics,
        }

        return metrics

    
class HomeServiceTaskSpecIterable:

    def __init__(
        self,
        # scenes_to_task_spec_dicts: Dict[str, List[Dict]],
        task_keys_to_task_spec_dicts: Dict[str, List[Dict]],
        seed: int,
        epochs: Union[int, float],
        shuffle: bool = True,
        task_type: HomeServiceTaskType = HomeServiceTaskType.SIMPLE_PICK_AND_PLACE,
        # scenes_to_task_dicts: Dict[str, List[Dict]] = None,
    ):
        assert epochs >= 1

        self.task_keys_to_task_spec_dicts = {
            k: [*v] for k, v in task_keys_to_task_spec_dicts.items()
        }
        # assert len(self.task_keys_to_task_spec_dicts) != 0 and all(
        #     len(self.task_keys_to_task_spec_dicts[task_key]) != 0
        #     for task_key in self.task_keys_to_task_spec_dicts
        # )
        assert len(self.task_keys_to_task_spec_dicts) != 0
        # self.scenes_to_task_spec_dicts = {
        #     k: [*v] for k, v in scenes_to_task_spec_dicts.items()
        # }
        # assert len(self.scenes_to_task_spec_dicts) != 0 and all(
        #     len(self.scenes_to_task_spec_dicts[scene]) != 0
        #     for scene in self.scenes_to_task_spec_dicts
        # )
        # self.scenes_to_task_dicts = None
        # if scenes_to_task_dicts is not None:
        #     self.scenes_to_task_dicts = {
        #         k: [*v] for k, v in scenes_to_task_dicts.items()
        #     }        
        self._seed = seed
        self.random = random.Random(self.seed)
        self.start_epochs = epochs
        self.remaining_epochs = epochs
        self.shuffle = shuffle
        self.task_type = task_type

        # self.remaining_scenes: List[str] = []
        self.remaining_task_keys: List[str] = []
        # self.task_spec_dicts_for_current_scene: List[Dict[str, Any]] = []
        self.task_spec_dicts_for_current_task_key: List[Dict[str, Any]] = []
        # self.current_scene: Optional[str] = None
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
        if len(self.task_spec_dicts_for_current_task_key) == 0:
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
        
        if self.task_type == HomeServiceTaskType.SIMPLE_PICK_AND_PLACE:
            return HomeServiceSimpleTaskOrderTaskSpec(**new_task_spec_dict)
        else:
            return HomeServiceTaskSpec(**new_task_spec_dict)


class HomeServiceTaskSampler(TaskSampler):

    def __init__(
        self,
        stage: str,
        # scenes_to_task_spec_dicts: Dict[str, List[Dict[str, Any]]],
        task_keys_to_task_spec_dicts: Dict[str, List[Dict[str, Any]]],
        home_service_env_kwargs: Optional[Dict[str, Any]],
        sensors: SensorSuite,
        max_steps: int,
        discrete_actions: Tuple[str, ...],
        smooth_nav: bool,
        require_done_action: bool,
        force_axis_aligned_start: bool,
        task_type: HomeServiceTaskType = HomeServiceTaskType.SIMPLE_PICK_AND_PLACE,
        # scenes_to_task_dicts: Optional[Dict[str, List[Dict[str,Any]]]] = None,
        epochs: Union[int, float, str] = "default",
        smoothing_factor: int = 1,
        seed: Optional[int] = None,
        task_spec_in_metrics: bool = False,
    ) -> None:
        
        self.sensors = sensors
        self.stage = stage
        self.main_seed = seed if seed is not None else random.randint(0, 2 * 30 - 1)

        self.task_spec_in_metrics = task_spec_in_metrics

        # self.scenes_to_task_spec_dicts = copy.deepcopy(scenes_to_task_spec_dicts)
        self.task_keys_to_task_spec_dicts = copy.deepcopy(task_keys_to_task_spec_dicts)
        # self.scenes_to_task_dicts = None
        # if scenes_to_task_dicts is not None:
        #     self.scenes_to_task_dicts = copy.deepcopy(scenes_to_task_dicts)

        if isinstance(epochs, str):
            if epochs.lower().strip() != "default":
                raise NotImplementedError(f"Unknown value for `epochs` (=={epochs})")
            epochs = float("inf") if stage == "train" else 1

        self.task_spec_iterator = HomeServiceTaskSpecIterable(
            task_keys_to_task_spec_dicts=self.task_keys_to_task_spec_dicts,
            seed=self.main_seed,
            epochs=epochs,
            shuffle=epochs == float("inf"),
            task_type=task_type,
            # scenes_to_task_dicts=self.scenes_to_task_dicts,
        )
        
        self.env = HomeServiceTHOREnvironment(**home_service_env_kwargs)

        self.task_keys = list(self.task_keys_to_task_spec_dicts.keys())

        self.max_steps = max_steps
        self.discrete_actions = discrete_actions
        self.smooth_nav = smooth_nav
        self.smoothing_factor = smoothing_factor
        self.require_done_action = require_done_action
        self.force_axis_aligned_start = force_axis_aligned_start
        self.task_type = task_type

        self._last_sampled_task: Optional[HomeServiceBaseTask] = None

    # FOR REARRANGE DATA
    # @classmethod
    # def from_fixed_dataset(
    #     cls,
    #     stage: str,
    #     task_type: HomeServiceTaskType,
    #     allowed_scenes: Optional[Sequence[str]] = None,
    #     scene_to_allowed_inds: Optional[Dict[str, Sequence[int]]] = None,
    #     randomize_start_rotation: bool = False,
    #     **init_kwargs,
    # ):
    #     scenes_to_task_spec_dicts = cls._filter_scenes_to_task_spec_dicts(
    #         scenes_to_task_spec_dicts=cls.load_rearrange_data_from_path(
    #             stage=stage, base_dir=STARTER_REARRANGE_DATA_DIR
    #         ),
    #         allowed_scenes=allowed_scenes,
    #         scene_to_allowed_inds=scene_to_allowed_inds,
    #     )
    #     if randomize_start_rotation:
    #         random_gen = random.Random(1)
    #         for scene in sorted(scenes_to_task_spec_dicts.keys()):
    #             for task_spec_dict in scenes_to_task_spec_dicts[scene]:
    #                 task_spec_dict["agent_rotation"] = 360.0 * random_gen.random()

    #     return cls(
    #         stage=stage,
    #         task_type=task_type,
    #         scenes_to_task_spec_dicts=scenes_to_task_spec_dicts,
    #         **init_kwargs
    #     )

    # FOR REARRANGE DATA
    # @classmethod
    # def _filter_scenes_to_task_spec_dicts(
    #     cls,
    #     scenes_to_task_spec_dicts: Dict[str, List[Dict[str, Any]]],
    #     allowed_scenes: Optional[Sequence[str]],
    #     scene_to_allowed_inds: Optional[Dict[str, Sequence[int]]],
    # ) -> Dict[str, List[Dict[str, Any]]]:
    #     if allowed_scenes is not None:
    #         scenes_to_task_spec_dicts = {
    #             scene: scenes_to_task_spec_dicts[scene] for scene in allowed_scenes
    #         }

    #     if scene_to_allowed_inds is not None:
    #         scenes_to_task_spec_dicts = {
    #             scene: [
    #                 scenes_to_task_spec_dicts[scene][ind]
    #                 for ind in sorted(scene_to_allowed_inds[scene])
    #             ]
    #             for scene in scene_to_allowed_inds
    #             if scene in scenes_to_task_spec_dicts
    #         }
    #     return scenes_to_task_spec_dicts

    # FOR REARRANGE DATA
    # @classmethod
    # def load_rearrange_data_from_path(
    #     cls, stage: str, base_dir: Optional[str] = None,
    # ) -> Dict[str, List[Dict[str, Any]]]:
    #     stage = stage.lower()

    #     if stage == "valid":
    #         stage = "val"

    #     data_path = os.path.abspath(os.path.join(base_dir, f"{stage}.pkl.gz"))
    #     if not os.path.exists(data_path):
    #         raise RuntimeError(f"No data at path {data_path}")

    #     data = compress_pickle.load(path=data_path)
    #     for scene in data:
    #         for ind, task_spec_dict in enumerate(data[scene]):
    #             task_spec_dict["scene"] = scene

    #             if "index" not in task_spec_dict:
    #                 task_spec_dict["index"] = ind

    #             if "stage" not in task_spec_dict:
    #                 task_spec_dict["stage"] = stage
    #     return data

    @classmethod
    def from_fixed_simple_pick_and_place_data(
        cls,
        stage: str,
        task_type: HomeServiceTaskType,
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
            task_type=task_type,
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

                    target_recep_allowed = True
            
                if allowed_pickup_objs is not None:
                    if pickup_obj in allowed_pickup_objs:
                        pickup_allowed = True
                
                if allowed_start_receps is not None:
                    if start_recep in allowed_start_receps:
                        start_recep_allowed = True
                
                if allowed_target_receps is not None:
                    if target_recep is not None and target_recep in allowed_target_receps:
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

    # @classmethod
    # def from_scenes_at_runtime(
    #     cls,
    #     stage: str,
    #     allowed_scenes: Sequence[str],
    #     repeats_before_scene_change: int,
    #     **init_kwargs,
    # ):
    #     assert "scene_to_allowed_inds" not in init_kwargs
    #     assert repeats_before_scene_change >= 1
    #     return cls(
    #         stage=stage,
    #         scenes_to_task_spec_dicts={
    #             scene: tuple(
    #                 {scene: scene, "runtime_sample": True}
    #                 for _ in range(repeats_before_scene_change)
    #             )
    #             for scene in allowed_scenes
    #         },
    #         **init_kwargs,
    #     )

    @property
    def length(self) -> float:
        return self.task_spec_iterator.length

    @property
    def total_unique(self):
        return self.task_spec_iterator.total_unique

    @property
    def last_sampled_task(self) -> Optional[HomeServiceBaseTask]:
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
        # pickup_target: str = None,
        # place_target: str = None,
        **kwargs
    ) -> Optional[HomeServiceBaseTask]:

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
            )

            if self.task_type == HomeServiceTaskType.SIMPLE_PICK_AND_PLACE:

                # pick, place = sample_pick_and_place_target(
                #     env=self.env,
                #     randomizer=self.task_spec_iterator.random,
                #     pickup_target=pickup_target,
                #     place_target=place_target
                # )
                # self.env.current_task_spec.pickup_target = pick
                # self.env.current_task_spec.place_target = place

                self._last_sampled_task = HomeServiceSimplePickAndPlaceTask(
                    sensors=self.sensors,
                    env = self.env,
                    max_steps=self.max_steps,
                    discrete_actions=self.discrete_actions,
                    smooth_nav=self.smooth_nav,
                    smoothing_factor=self.smoothing_factor,
                    require_done_action=self.require_done_action,
                    task_spec_in_metrics=self.task_spec_in_metrics,
                )
            else:
                self._last_sampled_task = HomeServiceBaseTask(
                    sensors=self.sensors,
                    env = self.env,
                    max_steps=self.max_steps,
                    discrete_actions=self.discrete_actions,
                    smooth_nav=self.smooth_nav,
                    smoothing_factor=self.smoothing_factor,
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
