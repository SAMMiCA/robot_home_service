from typing import Dict, Union, Optional, Any, TYPE_CHECKING
import stringcase

from allenact.utils.system import get_logger
from allenact_plugins.ithor_plugin.ithor_util import include_object_data
from env.environment import HomeServiceEnvironment
if TYPE_CHECKING:
    from env.tasks import HomeServiceTask

"""
Simple Subtasks (13)
Stop (1): To stop an episode and call next task.
Explore(1): To explore to find out target object & receptacle.
MoveTo (4): To move to the designated room type.
Goto (2): To move to the target object/receptacle.
PickupObject (1): To pickup a target object in the house.
PutObject (1): To put down a holding object on the target place. 
OpenReceptacle (1): To open a receptacle.
CloseReceptacle (1): To close a receptacle.
"""

SUBTASKS = [
    "Stop",
    "Explore",
    "MoveToKitchen",
    "MoveToLivingRoom",
    "MoveToBedroom",
    "MoveToBathroom",
    "GotoObject",
    "GotoReceptacle",
    "Pickup",
    "Put",
    "Open",
    "Close",
]

NUM_SUBTASKS = len(SUBTASKS)
NAV_SUBTASKS = [
    subtask for subtask in SUBTASKS if (
        subtask.startswith("Explore")
        or subtask.startswith("Move")
        or subtask.startswith("Goto")
    )
]
INTERACT_SUBTASKS = [
    "Pickup",
    "Put",
    "Open",
    "Close",
]


class HomeServiceSimpleSubtaskPlanner:
    def __init__(
        self,
        task: "HomeServiceTask",
        max_steps_for_subtask: int = 50,
    ):
        get_logger().debug(
            f"Subtask Planner CREATED."
        )
        self._subtask: int = -1  # Initial
        self.task = task
        self.subtask_count: int = 0
        self.max_steps = max_steps_for_subtask


    @property
    def env(self) -> HomeServiceEnvironment:
        return self.task.env

    @property
    def current_subtask(self):
        return self._subtask

    def set_subtask(self, subtask: Union[str, int]):
        if isinstance(subtask, str):
            subtask = SUBTASKS.index(subtask)
        elif not isinstance(subtask, int):
            raise RuntimeError
        
        assert 0 <= subtask < len(SUBTASKS)
        if subtask != self._subtask:
            self.subtask_count = 0
            get_logger().info(
                f"Subtask CHANGED ' O'//"
                f" from {self.subtask_str(self._subtask)} to {self.subtask_str(subtask)}"
            )
        self._subtask = subtask

    def subtask_str(self, subtask: int) -> str:
        assert isinstance(subtask, int)
        if subtask == -1:
            return "Start"
        return SUBTASKS[subtask]

    def next_subtask(self, recursive: bool = True) -> Optional[str]:
        if self.subtask_str(self.current_subtask) == "Start":
            # Initial
            # Move to the Receptacle room
            next_subtask = f"MoveTo{self.env.get_room_type(self.env.target_recep_room_id)}"
        elif self.subtask_str(self.current_subtask) == "Stop":
            # Stop
            return None
        elif self.subtask_str(self.current_subtask) == "Explore":
            # Explore
            found_object = (
                self.env.target_object_id in self.task.action_expert.scanned_objects
                if hasattr(self.task.action_expert, "scanned_objects") else False
            )
            found_receptacle = (
                self.env.target_recep_id in self.task.action_expert.scanned_receps
                if hasattr(self.task.action_expert, "scanned_receps") else False
            )
            # Keep current subtask
            next_subtask = self.subtask_str(self.current_subtask)
            # assert found_object or found_receptacle, f"At least receptacle has been found"
            if found_receptacle:
                if self.env.current_room != self.env.target_object_room_id:
                    next_subtask = f"MoveTo{self.env.get_room_type(self.env.target_object_room_id)}"
                elif found_object:
                    next_subtask = f"GotoObject"
                # next_subtask = f"MoveTo{self.env.get_room_type(self.env.target_object_room_id)}"

            #     else:
            #         # Keep current subtask
            #         next_subtask = self.subtask_str(self.current_subtask)

            #     if not found_object:
            #         if self.env.current_room != self.env.target_object_room_id:
            #             next_subtask = f"MoveTo{self.env.get_room_type(self.env.target_object_room_id)}"
            #         else:
            #             # Keep current subtask
            #             next_subtask = self.subtask_str(self.current_subtask)
            #     else:
            #         if self.env.current_room != self.env.target_object_room_id:
            #             next_subtask = f"MoveTo{self.env.get_room_type(self.env.target_object_room_id)}"
            #         else:
            #             next_subtask = f"GotoObject"
            # else:
            #     # Keep current subtask
            #     next_subtask = self.subtask_str(self.current_subtask)
        elif self.subtask_str(self.current_subtask).startswith("MoveTo"):
            # MoveTo
            target_room_type = self.subtask_str(self.current_subtask).replace("MoveTo", "")
            if self.env.current_room_type != target_room_type:
                # Keep current subtask
                next_subtask = self.subtask_str(self.current_subtask)
            else:
                found_object = (
                    self.env.target_object_id in self.task.action_expert.scanned_objects
                    if hasattr(self.task.action_expert, "scanned_objects") else False
                )
                found_receptacle = (
                    self.env.target_recep_id in self.task.action_expert.scanned_receps
                    if hasattr(self.task.action_expert, "scanned_receps") else False
                )

                if found_object and found_receptacle:
                    with include_object_data(self.env.controller):
                        visible_objects = {
                            o["objectId"]
                            for o in self.env.last_event.metadata["objects"]
                            if o["visible"]
                        }
                    if (
                        self.env.current_room == self.env.target_object_room_id
                        and self.env.held_object is None
                    ):
                        target_object = next(
                            o for o in self.env.objects()
                            if o['objectId'] == self.env.target_object_id
                        )
                        interactable_positions = self.env._interactable_positions_cache.get(
                            scene_name=self.env.scene,
                            obj=target_object,
                            controller=self.env
                        )
                        ip_keys = [
                            self.env.shortest_path_navigator.get_full_key(ip)
                            for ip in interactable_positions
                        ]
                        if (
                            self.env.target_object_id in visible_objects
                            and self.env.shortest_path_navigator.get_full_key(
                                self.env.get_agent_location()
                            ) in ip_keys
                        ):
                            # target object is visible and interactable right now
                            next_subtask = f"Pickup"
                        else:
                            next_subtask = f"GotoObject"
                    elif self.env.current_room == self.env.target_recep_room_id:
                        assert self.env.held_object is not None
                        target_recep = next(
                            o for o in self.env.objects()
                            if o['objectId'] == self.env.target_recep_id
                        )
                        interactable_positions = self.env._interactable_positions_cache.get(
                            scene_name=self.env.scene,
                            obj=target_recep,
                            controller=self.env
                        )
                        ip_keys = [
                            self.env.shortest_path_navigator.get_full_key(ip)
                            for ip in interactable_positions
                        ]
                        if (
                            self.env.target_recep_id in visible_objects
                            and self.env.shortest_path_navigator.get_full_key(
                                self.env.get_agent_location()
                            ) in ip_keys
                        ):
                            next_subtask = f"Put"
                        else:
                            next_subtask = f"GotoReceptacle"
                    else:
                        # Keep current subtask
                        next_subtask = self.subtask_str(self.current_subtask)
                else:
                    # The agent has moved to object room
                    next_subtask = f"Explore"
        elif self.subtask_str(self.current_subtask) == "GotoObject":
            assert self.env.current_room == self.env.target_object_room_id
            target_object = next(
                o for o in self.env.objects()
                if o['objectId'] == self.env.target_object_id
            )
            with include_object_data(self.env.controller):
                visible_objects = {
                    o["objectId"]
                    for o in self.env.last_event.metadata["objects"]
                    if o["visible"]
                }
            interactable_positions = self.env._interactable_positions_cache.get(
                scene_name=self.env.scene,
                obj=target_object,
                controller=self.env
            )
            ip_keys = [
                self.env.shortest_path_navigator.get_full_key(ip)
                for ip in interactable_positions
            ]
            if (
                self.env.target_object_id in visible_objects
                and self.env.shortest_path_navigator.get_full_key(
                    self.env.get_agent_location()
                ) in ip_keys
            ):
                next_subtask = f"Pickup"
            else:
                # Keep current subtask
                next_subtask = self.subtask_str(self.current_subtask)
        elif self.subtask_str(self.current_subtask) == "GotoReceptacle":
            if self.env.current_room != self.env.target_recep_room_id:
                # import pdb; pdb.set_trace()
                next_subtask = f"MoveTo{self.env.get_room_type(self.env.target_recep_room_id)}"
            else:

                # assert self.env.current_room == self.env.target_recep_room_id
                with include_object_data(self.env.controller):
                    visible_objects = {
                        o["objectId"]
                        for o in self.env.last_event.metadata["objects"]
                        if o["visible"]
                    }
                target_recep = next(
                    o for o in self.env.objects()
                    if o['objectId'] == self.env.target_recep_id
                )
                interactable_positions = self.env._interactable_positions_cache.get(
                    scene_name=self.env.scene,
                    obj=target_recep,
                    controller=self.env
                )
                ip_keys = [
                    self.env.shortest_path_navigator.get_full_key(ip)
                    for ip in interactable_positions
                ]
                if (
                    self.env.target_recep_id in visible_objects
                    and self.env.shortest_path_navigator.get_full_key(
                        self.env.get_agent_location()
                    ) in ip_keys
                ):
                    if target_recep["openable"] and target_recep["openness"] < 1.0:
                        if not hasattr(self, "initial_openness") or self.initial_openness is None:
                            self.initial_openness = target_recep["openness"]
                            next_subtask = f"Open"
                        else:
                            next_subtask = f"Close"
                    else:
                        next_subtask = f"Put"
                else:
                    # Keep current subtask
                    next_subtask = self.subtask_str(self.current_subtask)
        elif self.subtask_str(self.current_subtask) == "Pickup":
            if self.env.held_object is None:
                # Keep current subtask
                next_subtask = self.subtask_str(self.current_subtask)
                interactable_positions = self.env._interactable_positions_cache.get(
                    scene_name=self.env.scene,
                    obj=next(
                        o for o in self.env.objects()
                        if o['objectId'] == self.env.target_object_id
                    ),
                    controller=self.env
                )
                ip_keys = [
                    self.env.shortest_path_navigator.get_full_key(ip)
                    for ip in interactable_positions
                ]
                if (
                    self.env.shortest_path_navigator.get_full_key(
                        self.env.get_agent_location()
                    ) not in ip_keys
                ):
                    next_subtask = f"GotoObject"
            elif self.env.held_object["objectId"] == self.env.target_object_id:
                next_subtask = f"MoveTo{self.env.get_room_type(self.env.target_recep_room_id)}"
            else:
                next_subtask = f"Put"
        elif self.subtask_str(self.current_subtask) == "Put":
            if self.env.held_object is None:
                target_object = next(
                    o for o in self.env.objects()
                    if o['objectId'] == self.env.target_object_id
                )
                if (
                    target_object is not None
                    and target_object["parentReceptacles"] is not None
                    and self.env.target_recep_id in target_object["parentReceptacles"]
                ):
                    target_recep = next(
                        o for o in self.env.objects()
                        if o['objectId'] == self.env.target_recep_id
                    )
                    if target_recep["openable"] and hasattr(self, "initial_openness"):
                        assert self.initial_openness is not None
                        if self.initial_openness < 1.0:
                            next_subtask = f"Close"
                    
                    else:
                        next_subtask = f"Stop"
                else:
                    next_subtask = f"GotoObject"
            else:
                next_subtask = self.subtask_str(self.current_subtask)
                interactable_positions = self.env._interactable_positions_cache.get(
                    scene_name=self.env.scene,
                    obj=next(
                        o for o in self.env.objects()
                        if o['objectId'] == self.env.target_recep_id
                    ),
                    controller=self.env
                )
                ip_keys = [
                    self.env.shortest_path_navigator.get_full_key(ip)
                    for ip in interactable_positions
                ]
                if (
                    self.env.shortest_path_navigator.get_full_key(self.env.get_agent_location()) not in ip_keys
                ):
                    next_subtask = f"GotoReceptacle"
            # elif (
            #     self.task.actions_taken[-1].startswith('put')
            #     and not self.task.actions_taken_success[-1]
            # ):
            #     next_subtask = f"GotoReceptacle"
        elif self.subtask_str(self.current_subtask) == "Open":
            assert self.env.held_object is not None
            target_recep = next(
                o for o in self.env.objects()
                if o['objectId'] == self.env.target_recep_id
            )
            assert target_recep["openable"]
            if target_recep["openness"] < 1.0:
                # Open failed
                next_subtask = "GotoReceptacle"
            else:
                next_subtask = f"Put"
        elif self.subtask_str(self.current_subtask) == "Close":
            assert self.env.held_object is None
            target_recep = next(
                o for o in self.env.objects()
                if o['objectId'] == self.env.target_recep_id
            )
            assert target_recep["openable"]
            if target_recep["openness"] > 0.0:
                # Open failed
                next_subtask = "GotoReceptacle"
            else:
                next_subtask = f"Stop"

        if self.subtask_str(self.current_subtask) != next_subtask:
            get_logger().info(
                f" Current Subtask[{self.subtask_str(self.current_subtask)}({self.current_subtask})] DONE!"
            )
            self.set_subtask(next_subtask)
            if recursive:
                self.next_subtask()
        # if not recursive:
        #     self.set_subtask(next_subtask)
        return SUBTASKS[self.current_subtask]

    def check_whether_successfully_done(self):
        target_object = next(
            o for o in self.env.objects()
            if o['objectId'] == self.env.target_object_id
        )
        return self.env.target_recep_id in target_object["parentReceptacles"]