from typing import (
    Generic,
    Dict,
    Any,
    Optional,
    TYPE_CHECKING,
    TypeVar,
    Sequence,
    Union,
    Tuple,
    cast,
)
if TYPE_CHECKING:
    from allenact.base_abstractions.task import SubTaskType
else:
    SubTaskType = TypeVar("SubTaskType", bound="Task")
from allenact.base_abstractions.sensor import AbstractExpertActionSensor
from allenact.utils.misc_utils import prepare_locals_for_super

from env.expert import ShortestPathNavigator, HomeServiceGreedyActionExpert
from env.environment import HomeServiceEnvironment
from env.tasks import HomeServiceTask
from env.constants import (
    DEFAULT_COMPATIBLE_RECEPTACLES,
    OBJECT_TYPES_WITH_PROPERTIES,
    STARTER_HOME_SERVICE_DATA_DIR,
    STEP_SIZE
)


class HomeServiceGreedyActionExpertSensor(AbstractExpertActionSensor):
    def query_expert(
        self,
        task: HomeServiceTask,
        expert_sensor_group_name: Optional[str]
    ) -> Tuple[Any, bool]:
        env: HomeServiceEnvironment = task.env
        if (
            not hasattr(task, "action_expert")
            or task.action_expert is None
        ):
            if (
                not hasattr(env, "shortest_path_navigator")
                or env.shortest_path_navigator is None
            ):
                env.shortest_path_navigator = task.create_navigator()
            
            task.action_expert = task.create_greedy_expert()

        action = task.action_expert.expert_action
        if action is None:
            return 0, False
        else:
            return action, True


class HomeServiceSubtaskActionExpertSensor(AbstractExpertActionSensor):
    def query_expert(
        self,
        task: HomeServiceTask,
        expert_sensor_group_name: Optional[str]
    ) -> Tuple[Any, bool]:
        env: HomeServiceEnvironment = task.env
        if (
            not hasattr(task, "action_expert")
            or task.action_expert is None
        ):
            if (
                not hasattr(env, "shortest_path_navigator")
                or env.shortest_path_navigator is None
            ):
                env.shortest_path_navigator = task.create_navigator()
            
            task.action_expert = task.create_subtask_expert()

        action = task.action_expert.expert_action
        if action is None:
            return 0, False
        else:
            return action, True

# class SubTaskExpertSensor(AbstractExpertActionSensor):
#     def query_expert(
#         self,
#         task: SubTaskType,
#         expert_sensor_group_name: Optional[str]
#     ) -> Tuple[Any, bool]:
#         self.task = task
#         self.env = self.task.env
#         if self.task.greedy_expert is None:
#             if not hasattr(self.env, "shortest_path_navigator"):
#                 self.env.shortest_path_navigator = ShortestPathNavigatorTHOR(
#                     controller = self.env.controller,
#                     grid_size=STEP_SIZE,
#                     include_move_left_right=all(
#                         f"move_{k}" in self.task.action_names() for k in ["left", "right"]
#                     ),
#                 )
            
#             self.task.greedy_expert = SubTaskExpert(
#                 task=self.task,
#                 shortest_path_navigator=self.env.shortest_path_navigator
#             )
        
#         action = self.task.greedy_expert.expert_action
#         if action is None:
#             return 0, False
#         else:
#             return action, True        
