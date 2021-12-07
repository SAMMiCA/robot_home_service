"""Definitions for a greedy expert for the `Unshuffle` task."""

import copy
import random
import math
from networkx.algorithms.shortest_paths.generic import shortest_path
import numpy as np
from collections import defaultdict
from typing import (
    Dict,
    Tuple,
    Any,
    Optional,
    Union,
    List,
    Sequence,
    TYPE_CHECKING,
)

import ai2thor.controller
import ai2thor.server
import networkx as nx
import stringcase
from torch.distributions.utils import lazy_property

from allenact.utils.system import get_logger
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment
from allenact_plugins.ithor_plugin.ithor_util import (
    round_to_factor,
    include_object_data,
)
from env.constants import NOT_PROPER_RECEPTACLES, SCENE_TO_SCENE_TYPE, STEP_SIZE
from env.environment import (
    HomeServiceTHOREnvironment,
    HomeServiceMode,
    HomeServiceTaskSpec,
)
from env.utils import filter_positions

if TYPE_CHECKING:
    from env.tasks import HomeServiceSimplePickAndPlaceTask, HomeServiceBaseTask

AgentLocKeyType = Tuple[float, float, int, int]
PositionKeyType = Tuple[float, float, float]


# class ExplorerTHOR:
#     def __init__(
#         self,
#         task_spec: HomeServiceTaskSpec,
#     ):
#         self.current_task_spec = task_spec

#     @property
#     def goto_actions(self) -> Sequence[str]:
#         goto_actions = []
#         target_scene_type = SCENE_TO_SCENE_TYPE[self.current_task_spec.target_scene]
#         goto_actions.append(f"Goto{target_scene_type}")
#         for _ in range(4):
#             goto_actions.append(f"RotateRight")
        
#         return goto_actions

    


class ShortestPathNavigatorTHOR:
    """Tracks shortest paths in AI2-THOR environments.

    Assumes 90 degree rotations and fixed step sizes.

    # Attributes
    controller : The AI2-THOR controller in which shortest paths are computed.
    """

    def __init__(
        self,
        controller: ai2thor.controller.Controller,
        grid_size: float,
        include_move_left_right: bool = False,
    ):
        """Create a `ShortestPathNavigatorTHOR` instance.

        # Parameters
        controller : An AI2-THOR controller which represents the environment in which shortest paths should be
            computed.
        grid_size : The distance traveled by an AI2-THOR agent when taking a single navigational step.
        include_move_left_right : If `True` the navigational actions will include `MoveLeft` and `MoveRight`, otherwise
            they wil not.
        """
        self._cached_graphs: Dict[str, nx.DiGraph] = {}

        self._current_scene: Optional[nx.DiGraph] = None
        self._current_graph: Optional[nx.DiGraph] = None

        self._grid_size = grid_size
        self.controller = controller

        self._include_move_left_right = include_move_left_right
        self._position_to_object_id: Dict[PositionKeyType, str] = {}

    @lazy_property
    def nav_actions_set(self) -> frozenset:
        """Navigation actions considered when computing shortest paths."""
        nav_actions = [
            "LookUp",
            "LookDown",
            "RotateLeft",
            "RotateRight",
            "MoveAhead",
        ]
        if self._include_move_left_right:
            nav_actions.extend(["MoveLeft", "MoveRight"])
        return frozenset(nav_actions)

    @property
    def scene_name(self) -> str:
        """Current ai2thor scene."""
        return self.controller.last_event.metadata["sceneName"]

    @property
    def last_action_success(self) -> bool:
        """Was the last action taken by the agent a success?"""
        return self.controller.last_event.metadata["lastActionSuccess"]

    @property
    def last_event(self) -> ai2thor.server.Event:
        """Last event returned by the controller."""
        return self.controller.last_event

    def on_reset(self):
        """Function that must be called whenever the AI2-THOR controller is
        reset."""
        self._current_scene = None
        self._position_to_object_id = {}

    @property
    def graph(self) -> nx.DiGraph:
        """A directed graph representing the navigation graph of the current
        scene."""
        if self._current_scene == self.scene_name:
            return self._current_graph

        if self.scene_name not in self._cached_graphs:
            g = nx.DiGraph()
            points = self.reachable_points_with_rotations_and_horizons()
            for p in points:
                self._add_node_to_graph(g, self.get_key(p))

            self._cached_graphs[self.scene_name] = g

        self._current_scene = self.scene_name
        self._current_graph = self._cached_graphs[self.scene_name].copy()
        return self._current_graph

    def reachable_points_with_rotations_and_horizons(
        self,
    ) -> List[Dict[str, Union[float, int]]]:
        """Get all the reaachable positions in the scene along with possible
        rotation/horizons."""

        self.controller.step(action="GetReachablePositions")
        assert self.last_action_success

        points_slim = filter_positions(self._grid_size, 90.0, self.last_event.metadata["actionReturn"], ["x", "z"])

        points = []
        for r in [0, 90, 180, 270]:
            for horizon in [-30, 0, 30, 60]:
                for p in points_slim:
                    p = copy.copy(p)
                    p["rotation"] = r
                    p["horizon"] = horizon
                    points.append(p)

        return points

    @staticmethod
    def location_for_key(key, y_value=0.0) -> Dict[str, Union[float, int]]:
        """Return a agent location dictionary given a graph node key."""
        x, z, rot, hor = key
        loc = dict(x=x, y=y_value, z=z, rotation=rot, horizon=hor)
        return loc

    @staticmethod
    def get_key(input_dict: Dict[str, Any], ndigits: int = 2) -> AgentLocKeyType:
        """Return a graph node key given an input agent location dictionary."""
        if "x" in input_dict:
            x = input_dict["x"]
            z = input_dict["z"]
            rot = input_dict["rotation"]
            hor = input_dict["horizon"]
        else:
            x = input_dict["position"]["x"]
            z = input_dict["position"]["z"]
            rot = input_dict["rotation"]["y"]
            hor = input_dict["cameraHorizon"]

        return (
            round(x, ndigits),
            round(z, ndigits),
            round_to_factor(rot, 90) % 360,
            round_to_factor(hor, 30) % 360,
        )

    def update_graph_with_failed_action(self, failed_action: str):
        """If an action failed, update the graph to let it know this happened
        so it won't try again."""
        if (
            self.scene_name not in self._cached_graphs
            or failed_action not in self.nav_actions_set
        ):
            return

        source_key = self.get_key(self.last_event.metadata["agent"])
        self._check_contains_key(source_key)

        edge_dict = self.graph[source_key]
        to_remove_key = None
        for target_key in self.graph[source_key]:
            if edge_dict[target_key]["action"] == failed_action:
                to_remove_key = target_key
                break
        if to_remove_key is not None:
            self.graph.remove_edge(source_key, to_remove_key)

    def _add_from_to_edge(
        self, g: nx.DiGraph, s: AgentLocKeyType, t: AgentLocKeyType,
    ):
        """Add an edge to the graph."""

        def ae(x, y):
            return abs(x - y) < 0.001

        s_x, s_z, s_rot, s_hor = s
        t_x, t_z, t_rot, t_hor = t

        l1_dist = round(abs(s_x - t_x) + abs(s_z - t_z), 2)
        angle_dist = (round_to_factor(t_rot - s_rot, 90) % 360) // 90
        horz_dist = (round_to_factor(t_hor - s_hor, 30) % 360) // 30

        # If source and target differ by more than one action, continue
        if sum(x != 0 for x in [l1_dist, angle_dist, horz_dist]) != 1:
            return

        grid_size = self._grid_size
        action = None
        if angle_dist != 0:
            if angle_dist == 1:
                action = "RotateRight"
            elif angle_dist == 3:
                action = "RotateLeft"

        elif horz_dist != 0:
            if horz_dist == 11:
                action = "LookUp"
            elif horz_dist == 1:
                action = "LookDown"
        elif ae(l1_dist, grid_size):

            if s_rot == 0:
                forward = round((t_z - s_z) / grid_size)
                right = round((t_x - s_x) / grid_size)
            elif s_rot == 90:
                forward = round((t_x - s_x) / grid_size)
                right = -round((t_z - s_z) / grid_size)
            elif s_rot == 180:
                forward = -round((t_z - s_z) / grid_size)
                right = -round((t_x - s_x) / grid_size)
            elif s_rot == 270:
                forward = -round((t_x - s_x) / grid_size)
                right = round((t_z - s_z) / grid_size)
            else:
                raise NotImplementedError(f"source rotation == {s_rot} unsupported.")

            if forward > 0:
                g.add_edge(s, t, action="MoveAhead")
            elif self._include_move_left_right:
                if forward < 0:
                    # Allowing MoveBack results in some really unintuitive
                    # expert trajectories (i.e. moving backwards to the goal and the
                    # rotating, for now it's disabled.
                    pass  # g.add_edge(s, t, action="MoveBack")
                elif right > 0:
                    g.add_edge(s, t, action="MoveRight")
                elif right < 0:
                    g.add_edge(s, t, action="MoveLeft")

        if action is not None:
            g.add_edge(s, t, action=action)

    @lazy_property
    def possible_neighbor_offsets(self) -> Tuple[AgentLocKeyType, ...]:
        """Offsets used to generate potential neighbors of a node."""
        grid_size = round(self._grid_size, 2)
        offsets = []
        for rot_diff in [-90, 0, 90]:
            for horz_diff in [-30, 0, 30, 60]:
                for x_diff in [-grid_size, 0, grid_size]:
                    for z_diff in [-grid_size, 0, grid_size]:
                        if (rot_diff != 0) + (horz_diff != 0) + (x_diff != 0) + (
                            z_diff != 0
                        ) == 1:
                            offsets.append((x_diff, z_diff, rot_diff, horz_diff))
        return tuple(offsets)

    def _add_node_to_graph(self, graph: nx.DiGraph, s: AgentLocKeyType):
        """Add a node to the graph along with any adjacent edges."""
        if s in graph:
            return

        existing_nodes = set(graph.nodes())
        graph.add_node(s)

        for x_diff, z_diff, rot_diff, horz_diff in self.possible_neighbor_offsets:
            t = (
                s[0] + x_diff,
                s[1] + z_diff,
                (s[2] + rot_diff) % 360,
                (s[3] + horz_diff) % 360,
            )
            if t in existing_nodes:
                self._add_from_to_edge(graph, s, t)
                self._add_from_to_edge(graph, t, s)

    def _check_contains_key(self, key: AgentLocKeyType, add_if_not=True) -> bool:
        """Check if a node key is in the graph.

        # Parameters
        key : The key to check.
        add_if_not : If the key doesn't exist and this is `True`, the key will be added along with
            edges to any adjacent nodes.
        """
        key_in_graph = key in self.graph
        if not key_in_graph:
            get_logger().debug(
                "{} was not in the graph for scene {}.".format(key, self.scene_name)
            )
            if add_if_not:
                self._add_node_to_graph(self.graph, key)
                if key not in self._cached_graphs[self.scene_name]:
                    self._add_node_to_graph(self._cached_graphs[self.scene_name], key)
        return key_in_graph

    def shortest_state_path(
        self, source_state_key: AgentLocKeyType, goal_state_key: AgentLocKeyType
    ) -> Optional[Sequence[AgentLocKeyType]]:
        """Get the shortest path between node keys."""
        self._check_contains_key(source_state_key)
        self._check_contains_key(goal_state_key)
        # noinspection PyBroadException
        path = nx.shortest_path(
            G=self.graph, source=source_state_key, target=goal_state_key
        )
        return path

    def action_transitioning_between_keys(self, s: AgentLocKeyType, t: AgentLocKeyType):
        """Get the action that takes the agent from node s to node t."""
        self._check_contains_key(s)
        self._check_contains_key(t)
        if self.graph.has_edge(s, t):
            return self.graph.get_edge_data(s, t)["action"]
        else:
            return None

    def shortest_path_next_state(
        self, source_state_key: AgentLocKeyType, goal_state_key: AgentLocKeyType
    ):
        """Get the next node key on the shortest path from the source to the
        goal."""
        if source_state_key == goal_state_key:
            raise RuntimeError("called next state on the same source and goal state")
        state_path = self.shortest_state_path(source_state_key, goal_state_key)
        return state_path[1]

    def shortest_path_next_action(
        self, source_state_key: AgentLocKeyType, goal_state_key: AgentLocKeyType
    ):
        """Get the next action along the shortest path from the source to the
        goal."""
        next_state_key = self.shortest_path_next_state(source_state_key, goal_state_key)
        return self.graph.get_edge_data(source_state_key, next_state_key)["action"]

    def shortest_path_next_action_multi_target(
        self,
        source_state_key: AgentLocKeyType,
        goal_state_keys: Sequence[AgentLocKeyType],
    ):
        """Get the next action along the shortest path from the source to the
        closest goal."""
        self._check_contains_key(source_state_key)

        terminal_node = (-1.0, -1.0, -1, -1)
        self.graph.add_node(terminal_node)
        for gsk in goal_state_keys:
            self._check_contains_key(gsk)
            self.graph.add_edge(gsk, terminal_node, action=None)

        next_state_key = self.shortest_path_next_state(source_state_key, terminal_node)
        action = self.graph.get_edge_data(source_state_key, next_state_key)["action"]

        self.graph.remove_node(terminal_node)
        return action

    def shortest_path_length(
        self, source_state_key: AgentLocKeyType, goal_state_key: AgentLocKeyType
    ):
        """Get the path shorest path length between the source and the goal."""
        self._check_contains_key(source_state_key)
        self._check_contains_key(goal_state_key)
        try:
            return nx.shortest_path_length(self.graph, source_state_key, goal_state_key)
        except nx.NetworkXNoPath as _:
            return float("inf")


def _are_agent_locations_equal(
    ap0: Dict[str, Union[float, int, bool]],
    ap1: Dict[str, Union[float, int, bool]],
    ignore_standing: bool,
    tol=1e-2,
    ignore_y: bool = True,
):
    """Determines if two agent locations are equal up to some tolerance."""

    def rot_dist(r0: float, r1: float):
        diff = abs(r0 - r1) % 360
        return min(diff, 360 - diff)

    return (
        all(
            abs(ap0[k] - ap1[k]) <= tol
            for k in (["x", "z"] if ignore_y else ["x", "y", "z"])
        )
        and rot_dist(ap0["rotation"], ap1["rotation"]) <= tol
        and rot_dist(ap0["horizon"], ap1["horizon"]) <= tol
        and (ignore_standing or (ap0["standing"] == ap1["standing"]))
    )


class GreedySimplePickAndPlaceExpert:
    """An agent which greedily attempts to complete a given unshuffle task."""

    def __init__(
        self,
        task: "HomeServiceSimplePickAndPlaceTask",
        shortest_path_navigator: ShortestPathNavigatorTHOR,
        max_priority_per_object: int = 3,
    ):
        """Initializes a `GreedyUnshuffleExpert` object.

        # Parameters
        task : An `UnshuffleTask` that the greedy expert should attempt to complete.
        shortest_path_navigator : A `ShortestPathNavigatorTHOR` object defined on the same
            AI2-THOR controller used by the `task`.
        max_priority_per_object : The maximum number of times we should try to unshuffle an object
            before giving up.
        """
        self.task = task
        self.shortest_path_navigator = shortest_path_navigator
        self.max_priority_per_object = max_priority_per_object

        assert self.task.num_steps_taken() == 0

        self.expert_action_list: List[int] = []

        self._last_held_object_name: Optional[str] = None
        self._last_to_interact_object_pose: Optional[Dict[str, Any]] = None
        # self._name_of_object_we_wanted_to_pickup: Optional[str] = None
        # self._name_of_object_we_wanted_to_open_close: Optional[str] = None
        # self.object_name_to_priority: defaultdict = defaultdict(lambda: 0)

        # Added parameter
        self._target_object_to_pickup: Optional[Dict[str, Any]] = None
        self._receptacle_object_to_open: Optional[Dict[str, Any]] = None
        self._receptacle_object_to_close: Optional[Dict[str, Any]] = None

        self.shortest_path_navigator.on_reset()
        self.update(action_taken=None, action_success=None)


    @property
    def expert_action(self) -> int:
        """Get the current greedy expert action.

        # Returns An integer specifying the expert action in the current
        state. This corresponds to the order of actions in
        `self.task.action_names()`. For this action to be available the
        `update` function must be called after every step.
        """
        assert self.task.num_steps_taken() == len(self.expert_action_list) - 1, (
            f"self.task.num_steps_taken(): {self.task.num_steps_taken()} is not equal to \
                len(self.expert_action_list) - 1: {len(self.expert_action_list) - 1}"
        )
        return self.expert_action_list[-1]

    def update(self, action_taken: Optional[int], action_success: Optional[bool]):
        """Update the expert with the last action taken and whether or not that
        action succeeded."""
        if action_taken is not None:
            assert action_success is not None

            action_names = self.task.action_names()
            last_expert_action = self.expert_action_list[-1]
            agent_took_expert_action = action_taken == last_expert_action
            action_str = action_names[action_taken]

            was_nav_action = any(k in action_str for k in ["move", "rotate", "look"])

            if "open_by_type" in action_str and agent_took_expert_action:
                self._receptacle_object_to_open = self._last_to_interact_object_pose

            if not action_success:  ## not succeeded
                if was_nav_action:
                    self.shortest_path_navigator.update_graph_with_failed_action(
                        stringcase.pascalcase(action_str)
                    )
                elif (
                    "pickup_" in action_str 
                    or "open_by_type" in action_str 
                    or "close_by_type" in action_str 
                    or "put_by_type" in action_str
                ) and action_taken == last_expert_action:
                    assert self._last_to_interact_object_pose is not None
                    self._invalidate_interactable_loc_for_pose(
                        location=self.task.env.get_agent_location(),
                        obj_pose=self._last_to_interact_object_pose,
                    )
                elif (
                    ("crouch" in action_str or "stand" in action_str) 
                    and action_taken == last_expert_action
                ):
                    agent_loc = self.task.env.get_agent_location()
                    agent_loc["standing"] = not agent_loc["standing"]
                    self._invalidate_interactable_loc_for_pose(
                        location=agent_loc,
                        obj_pose=self._last_to_interact_object_pose,
                    )

            else:
                # If the action succeeded and was not a move action then let's force an update
                # of our currently targeted object
                if not was_nav_action:
                    if "open_by_type" in action_str:
                        self._receptacle_object_to_close = self._receptacle_object_to_open
                        self._receptacle_object_to_open = None
                    elif "close_by_type" in action_str:
                        self._receptacle_object_to_close = None
                    elif "put_by_type" in action_str:
                        assert self.task.env.held_object is None
                        self._target_object_to_pickup = None
                    self._last_to_interact_object_pose = None

        held_object = self.task.env.held_object
        if self.task.env.held_object is not None:
            self._last_held_object_name = held_object["name"]

        self._generate_and_record_expert_action()

    def _expert_nav_action_to_obj(self, obj: Dict[str, Any]) -> Optional[str]:
        """Get the shortest path navigational action towards the object obj.

        The navigational action takes us to a position from which the
        object is interactable.
        """
        env: HomeServiceTHOREnvironment = self.task.env
        agent_loc = env.get_agent_location()
        shortest_path_navigator = self.shortest_path_navigator

        interactable_positions = env._interactable_positions_cache.get(
            scene_name=env.scene, obj=obj, controller=env.controller,
        )

        target_keys = [
            shortest_path_navigator.get_key(loc) for loc in interactable_positions
        ]
        if len(target_keys) == 0:
            return None

        source_state_key = shortest_path_navigator.get_key(env.get_agent_location())

        action = "Pass"
        if source_state_key not in target_keys:
            try:
                action = shortest_path_navigator.shortest_path_next_action_multi_target(
                    source_state_key=source_state_key, goal_state_keys=target_keys,
                )
            except nx.NetworkXNoPath as _:
                # Could not find the expert actions
                return None

        if action != "Pass":
            return action
        else:
            agent_x = agent_loc["x"]
            agent_z = agent_loc["z"]
            for gdl in interactable_positions:
                d = round(abs(agent_x - gdl["x"]) + abs(agent_z - gdl["z"]), 2)
                if d <= 1e-2:
                    if _are_agent_locations_equal(agent_loc, gdl, ignore_standing=True):
                        if agent_loc["standing"] != gdl["standing"]:
                            return "Crouch" if agent_loc["standing"] else "Stand"
                        else:
                            # We are already at an interactable position
                            return "Pass"
            return None

    def _invalidate_interactable_loc_for_pose(
        self, location: Dict[str, Any], obj_pose: Dict[str, Any]
    ) -> bool:
        """Invalidate a given location in the `interactable_positions_cache` as
        we tried to interact but couldn't."""
        env = self.task.env

        interactable_positions = env._interactable_positions_cache.get(
            scene_name=env.scene, obj=obj_pose, controller=env.controller
        )
        for i, loc in enumerate([*interactable_positions]):
            if (
                self.shortest_path_navigator.get_key(loc)
                == self.shortest_path_navigator.get_key(location)
                and loc["standing"] == location["standing"]
            ):
                interactable_positions.pop(i)
                return True
        return False

    def _generate_expert_action_dict(self) -> Dict[str, Any]:
        """Generate a dictionary describing the next greedy expert action."""
        env: HomeServiceTHOREnvironment = self.task.env

        if env.mode != HomeServiceMode.SNAP:
            raise NotImplementedError(
                f"Expert only defined for 'easy' mode (current mode: {env.mode}"
            )

        held_object = env.held_object
        agent_loc = env.get_agent_location()
        pickup_target = env.current_task_spec.pickup_target
        place_target = env.current_task_spec.place_target

        expert_nav_action = self._expert_nav_action_to_obj(
            obj=env.current_task_spec.pickup_target
        )
        with include_object_data(env.controller):
            visible_object_ids = [
                obj['objectId'] for obj in env.last_event.metadata["objects"]
                if obj["visible"]
            ]

        if held_object is None:
            if self._last_to_interact_object_pose is None:
                if self._receptacle_object_to_close is None:
                    try:
                        current_pickup_target = next(
                            obj for obj in env.last_event.metadata["objects"]
                            if obj["objectId"] == pickup_target["objectId"]
                        )
                        if current_pickup_target["parentReceptacles"] is not None:
                            if place_target["objectId"] in current_pickup_target["parentReceptacles"]:
                                # SimplePickAndPlaceTask Done. SUCCESS :)
                                # print(f"{env.current_task_spec.task_type} DONE!!!!")
                                return dict(action="Done")
                        else:
                            # Already put pickup target object in wrong place.
                            # should pickup again
                            self._last_to_interact_object_pose = current_pickup_target

                    except:
                        current_pickup_target = None

                    if self._receptacle_object_to_open is None:
                        try:
                            self._receptacle_object_to_open = next(
                                obj for obj in env.last_event.metadata["objects"]
                                if obj["openable"] and obj["objectId"] in pickup_target["parentReceptacles"]
                                and obj["objectType"] not in NOT_PROPER_RECEPTACLES
                            )
                            self._last_to_interact_object_pose = self._receptacle_object_to_open
                        except:
                            self._receptacle_object_to_open = None
                            self._target_object_to_pickup = pickup_target
                            self._last_to_interact_object_pose = pickup_target
                    else:
                        self._last_to_interact_object_pose = self._receptacle_object_to_open
                else:
                    self._last_to_interact_object_pose = self._receptacle_object_to_close
            
            expert_nav_action = self._expert_nav_action_to_obj(
                obj=self._last_to_interact_object_pose
            )

            if expert_nav_action is None:
                interactable_positions = env._interactable_positions_cache.get(
                    scene_name=env.scene,
                    obj=self._last_to_interact_object_pose,
                    controller=env.controller,
                )
                if len(interactable_positions) != 0:
                    # Could not find a path to the target.
                    # Please increase the place count of the object and try generating a new action.
                    get_logger().debug(
                        f"Could not find a path to the object {self._last_to_interact_object_pose['objectId']}"
                        f" in scene {env.scene}"
                        f" when at position {agent_loc}."
                    )
                else:
                    get_logger().debug(
                        f"Object {self._last_to_interact_object_pose['objectId']} in scene {env.scene}"
                        f" has no interactable positions."
                    )
                # return dict(action="Done")

            elif expert_nav_action == "Pass":
                if self._last_to_interact_object_pose["objectId"] not in visible_object_ids:
                    if self._invalidate_interactable_loc_for_pose(
                        location=agent_loc, obj_pose=self._last_to_interact_object_pose
                    ):
                        return self._generate_expert_action_dict()
                    raise RuntimeError("This should not be possible.")

                if self._last_to_interact_object_pose["objectId"] == pickup_target["objectId"]:
                    # Trying to PickupObject
                    return dict(action="Pickup", objectType=self._last_to_interact_object_pose["objectType"])
                
                elif self._last_to_interact_object_pose["objectId"] in pickup_target["parentReceptacles"]:
                    # Pickup target object is located in the openable receptacle
                    # Try to Open Receptacle
                    return dict(action="OpenByType", objectType=self._last_to_interact_object_pose["objectType"])

                elif self._last_to_interact_object_pose["objectType"] == place_target["objectType"]:
                    return dict(action="CloseByType", objectType=self._last_to_interact_object_pose["objectType"])
                
                else:
                    raise RuntimeError(" REALLY?? ")
                
            return dict(action=expert_nav_action)
        
        else:
            if self._last_to_interact_object_pose is None:
                if self._receptacle_object_to_close is None:
                    if self._receptacle_object_to_open is None:
                        try:
                            current_place_target = next(
                                obj for obj in env.last_event.metadata["objects"]
                                if obj["name"] == place_target["name"]
                            )
                            if current_place_target["openable"] and current_place_target["openness"] < 0.8:
                                self._receptacle_object_to_open = current_place_target
                                self._last_to_interact_object_pose = self._receptacle_object_to_open
                            else:
                                self._receptacle_object_to_open = None
                                self._last_to_interact_object_pose = place_target
                        
                        except:
                            current_place_target = None
                            # Cannot find place target in the current scene?
                            # Something Weird!!
                            raise RuntimeError(" NO ")                        
                    else:
                        self._last_to_interact_object_pose = self._receptacle_object_to_open
                else:
                    self._last_to_interact_object_pose = self._receptacle_object_to_close

            expert_nav_action = self._expert_nav_action_to_obj(
                obj=self._last_to_interact_object_pose
            )

            if expert_nav_action is None:
                interactable_positions = env._interactable_positions_cache.get(
                    scene_name=env.scene,
                    obj=self._last_to_interact_object_pose,
                    controller=env.controller,
                )
                if len(interactable_positions) != 0:
                    # Could not find a path to the target.
                    # Please increase the place count of the object and try generating a new action.
                    get_logger().debug(
                        f"Could not find a path to the object {self._last_to_interact_object_pose['objectId']}"
                        f" in scene {env.scene}"
                        f" when at position {agent_loc}."
                    )
                else:
                    get_logger().debug(
                        f"Object {self._last_to_interact_object_pose['objectId']} in scene {env.scene}"
                        f" has no interactable positions."
                    )

            elif expert_nav_action == "Pass":
                if self._last_to_interact_object_pose["objectId"] not in visible_object_ids:
                    if self._invalidate_interactable_loc_for_pose(
                        location=agent_loc, obj_pose=self._last_to_interact_object_pose
                    ):
                        return self._generate_expert_action_dict()
                    raise RuntimeError("This should not be possible.")

                if self._receptacle_object_to_open is not None:
                    # Should open the place_target
                    return dict(action="OpenByType", objectType=self._last_to_interact_object_pose["objectType"])
                
                elif self._receptacle_object_to_close is not None:
                    # Should close the receptacle of the pickup target
                    if self._last_to_interact_object_pose["objectId"] == place_target["objectId"]:
                        # Put the held object (pickup_target) on the place_target
                        return dict(action="PutByType", objectType=self._last_to_interact_object_pose["objectType"])
                    
                    return dict(action="CloseByType", objectType=self._last_to_interact_object_pose["objectType"])
                
                elif self._last_to_interact_object_pose["objectId"] == place_target["objectId"]:
                        # Put the held object (pickup_target) on the place_target
                        return dict(action="PutByType", objectType=self._last_to_interact_object_pose["objectType"])

                else:
                    raise RuntimeError(" HOW??? ")
            
            return dict(action=expert_nav_action)

    def _generate_and_record_expert_action(self):
        """Generate the next greedy expert action and save it to the
        `expert_action_list`."""
        if self.task.num_steps_taken() == len(self.expert_action_list) + 1:
            get_logger().warning(
                f"Already generated the expert action at step {self.task.num_steps_taken()}"
            )
            return

        assert self.task.num_steps_taken() == len(
            self.expert_action_list
        ), f"{self.task.num_steps_taken()} != {len(self.expert_action_list)}"
        expert_action_dict = self._generate_expert_action_dict()

        action_str = stringcase.snakecase(expert_action_dict["action"])
        if action_str not in self.task.action_names():
            if "objectType" in expert_action_dict:
                obj_type = stringcase.snakecase(expert_action_dict["objectType"])
                action_str = f"{action_str}_{obj_type}"

        try:
            self.expert_action_list.append(self.task.action_names().index(action_str))
        except ValueError:
            get_logger().error(
                f"{action_str} is not a valid action for the given task."
            )
            self.expert_action_list.append(None)


class SubTaskExpert:
    def __init__(
        self,
        task: "HomeServiceBaseTask",
        shortest_path_navigator: ShortestPathNavigatorTHOR,
    ):
        self.task = task
        self.shortest_path_navigator = shortest_path_navigator
        assert self.task.num_steps_taken() == 0

        self.expert_action_list: List[int] = []
        self.goto_action_list: List[str] = ["RotateRight" for _ in range(4)]
        self.check_room_type_done: bool = False
        self.require_check_room_type: bool = True

        self._last_to_interact_object_pose: Optional[Dict[str, Any]] = None
        self.map_oracle = True

        self.shortest_path_navigator.on_reset()
        self.update(action_taken=None, action_success=None)

    @property
    def expert_action(self) -> int:
        assert self.task.num_steps_taken() == len(self.expert_action_list) - 1, (
            f"self.task.num_steps_taken(): {self.task.num_steps_taken()} is not equal to \
                len(self.expert_action_list) - 1: {len(self.expert_action_list) - 1}"
        )
        return self.expert_action_list[-1]

    @property
    def goto_action(self) -> str:        
        if len(self.goto_action_list) > 0:
            return self.goto_action_list.pop()

        else:
            return None

    def update(
        self,
        action_taken: Optional[int],
        action_success: Optional[bool],
    ):
        if action_taken is not None:
            assert action_success is not None

            action_names = self.task.action_names()
            last_expert_action = self.expert_action_list[-1]
            agent_took_expert_action = action_taken == last_expert_action
            action_str = action_names[action_taken]

            was_nav_action = any(k in action_str for k in ['move', 'rotate', 'look'])
            was_goto_action = 'goto' in action_str

            if not action_success:
                if was_nav_action:
                    self.shortest_path_navigator.update_graph_with_failed_action(
                        stringcase.pascalcase(action_str)
                    )
                # elif (
                #     "pickup" in action_str
                #     or "open_by_type" in action_str
                #     or "close_by_type" in action_str
                #     or "put_by_type" in action_str
                # ) and action_taken == last_expert_action:
                #     assert self._last_to_interact_object_pose is not None
                #     self._invalidate_interactable_loc_for_pose(
                #         location=self.task.env.get_agent_location(),
                #         obj_pose=self._last_to_interact_object_pose,
                #     )
                #     self.task.rollback_subtask()
                elif was_goto_action:
                    # Reset Fail?
                    raise RuntimeError

                elif (
                    action_str == "pickup"
                ) and action_taken == last_expert_action:
                    if self.task.env.scene == self.task.env.current_task_spec.target_scene:
                        assert self._last_to_interact_object_pose is not None
                    self._invalidate_interactable_loc_for_pose(
                        location=self.task.env.get_agent_location(),
                        obj_pose=self._last_to_interact_object_pose,
                    )

                elif (
                    action_str == "put"
                ) and action_taken == last_expert_action:
                    if self.task.env.scene == self.task.env.current_task_spec.target_scene:
                        assert self._last_to_interact_object_pose is not None
                    self._invalidate_interactable_loc_for_pose(
                        location=self.task.env.get_agent_location(),
                        obj_pose=self._last_to_interact_object_pose,
                    )

                elif (
                    ("crouch" in action_str or "stand" in action_str) 
                    and action_taken == last_expert_action
                ):
                    if self.task.env.scene == self.task.env.current_task_spec.target_scene:
                        assert self._last_to_interact_object_pose is not None
                    agent_loc = self.task.env.get_agent_location()
                    agent_loc["standing"] = not agent_loc["standing"]
                    self._invalidate_interactable_loc_for_pose(
                        location=agent_loc,
                        obj_pose=self._last_to_interact_object_pose,
                    )
            else:
                if not was_nav_action:
                    # if (
                    #     "pickup" in action_str
                    #     or "open_by_type" in action_str
                    #     or "close_by_type" in action_str
                    #     or "put_by_type" in action_str
                    # ): 
                    #     self._last_to_interact_object_pose = None
                    if action_str == "pickup":
                        held_object = self.task.env.held_object
                        if held_object is None:
                            raise RuntimeError(
                                f"Impossible..."
                            )
                        elif held_object["objectType"] != self._last_to_interact_object_pose["objectType"]:
                            raise RuntimeError(
                                f"Impossible......"
                            )
                        else:
                            self._last_to_interact_object_pose = None
                    elif action_str == "put":
                        assert self.task.env.held_object is None
                        self._last_to_interact_object_pose = None
                    
                    elif was_goto_action:
                        self.require_check_room_type = True
                        self.goto_action_list = ["RotateRight" for _ in range(4)]
                        # If current subtask is not GOTO, rollback subtasks to GOTO
                        if self.task.current_subtask[0] != "Goto":
                            while self.task.current_subtask[0] == "Goto":
                                self.task.rollback_subtask()

        self._generate_and_record_expert_action()

    def _get_interactable_positions(
        self,
        obj: Dict[str, Any]
    ):
        if self.map_oracle:
            if obj is not None:
                return self.task.env._interactable_positions_cache.get(
                    scene_name=self.task.env.scene, obj=obj, controller=self.task.env.controller,
                    # max_distance=1.0
                )
            else:
                return []
        else:
            #TODO
            pass

    def _expert_goto_action_to_scene_type(
        self,
        scene_type: str
    ) -> Optional[str]:
        # if len(self.goto_action_list) == 0:
        #     if not self.task._1st_check:
        #         for _ in range(4):
        #             self.goto_action_list.append("RotateRight")
        #         self.task._1st_check = True
        #     else:
        #         if not self.task._took_goto_action:
        #             self.goto_action_list.append(f"Goto{scene_type}")
        #         elif not self.task._2nd_check:
        #             for _ in range(4):
        #                 self.goto_action_list.append("RotateRight")
        #             self.task._2nd_check = True

        # goto_action = self.goto_action
        # if len(self.goto_action_list) == 0:
        #     if self.task._2nd_check:
        #         self.task._check_goto_done = True
        #     elif self.task._1st_check and (
        #         SCENE_TO_SCENE_TYPE[self.task.env.scene] == scene_type
        #     ):
        #         self.task._check_goto_done = True
        if len(self.goto_action_list) > 0:
            self.check_room_type_done = False
            goto_action = self.goto_action
            if len(self.goto_action_list) == 0:
                self.check_room_type_done = True
                self.require_check_room_type = False

        else:
            goto_action = f"Goto{scene_type}"
            self.require_check_room_type = True

        return goto_action

    def _expert_nav_action_to_obj(
        self,
        obj: Dict[str, Any]
    ) -> Optional[str]:
        env: HomeServiceTHOREnvironment = self.task.env
        agent_loc = env.get_agent_location()
        shortest_path_navigator = self.shortest_path_navigator

        interactable_positions = self._get_interactable_positions(obj)

        target_keys = [
            shortest_path_navigator.get_key(loc) for loc in interactable_positions
        ]

        if len(target_keys) == 0:
            # print(f'No target keys')
            return "Fail"

        source_state_key = shortest_path_navigator.get_key(env.get_agent_location())

        action = "Pass"
        if source_state_key not in target_keys:
            try:
                action = shortest_path_navigator.shortest_path_next_action_multi_target(
                    source_state_key=source_state_key, goal_state_keys=target_keys,
                )
            except nx.NetworkXNoPath as _:
                # print(f'No path exists from {source_state_key} to {target_keys}')
                rand_nav_action = random.choice(["MoveAhead", "RotateRight", "RotateLeft", "LookUp", "LookDown"])
                # print(f'take random nav action... {rand_nav_action}')
                # import pdb; pdb.set_trace()
                return rand_nav_action

        if action != "Pass":
            return action
        else:
            agent_x = agent_loc["x"]
            agent_z = agent_loc["z"]
            for gdl in interactable_positions:
                d = round(abs(agent_x - gdl["x"]) + abs(agent_z - gdl["z"]), 2)
                if d <= 1e-2:
                    if _are_agent_locations_equal(agent_loc, gdl, ignore_standing=True):
                        if agent_loc["standing"] != gdl["standing"]:
                            return "Crouch" if agent_loc["standing"] else "Stand"
                        else:
                            return "Pass"
        
        return None

    def _expert_nav_action_to_position(self, position) -> Optional[str]:
        """Get the shortest path navigational action towards the certain position

        """
        env: HomeServiceTHOREnvironment = self.task.env
        shortest_path_navigator = self.shortest_path_navigator

        if isinstance(position, np.ndarray):
            position_key = (round(position[0], 2), round(position[1], 2), round(position[2], 2))
            position = dict(
                x=position[0],
                y=position[1],
                z=position[2],
            )
        elif isinstance(position, dict):
            position_key = (round(position['x'], 2), round(position['y'], 2), round(position['z'], 2))
        
        if position_key not in shortest_path_navigator._position_to_object_id:
            # Spawn the TargetCircle and place it on the position
            event = env.controller.step("SpawnTargetCircle", anywhere=True)
            assert event.metadata["lastActionSuccess"]
            id_target_circle = event.metadata["actionReturn"]


            event = env.controller.step(
                "TeleportObject", 
                objectId=id_target_circle, 
                position=position, 
                rotation=0, 
                forceAction=True
            )
            assert event.metadata["lastActionSuccess"]
            
            # To change objectId for former target circle
            event = env.controller.step("SpawnTargetCircle", anywhere=True)
            assert event.metadata["lastActionSuccess"]
            id_target_circle = event.metadata["actionReturn"]

            event = env.controller.step("RemoveFromScene", objectId=id_target_circle)
            assert event.metadata["lastActionSuccess"]

            def distance(p1, p2):
                d = 0
                for c in ("x", "y", "z"):
                    d += (p1[c] - p2[c]) ** 2
                return round(math.sqrt(d), 2)
                
            # check
            target_circle_after_teleport = next(
                (
                    obj for obj in env.last_event.metadata['objects']
                    if obj['objectType'] == "TargetCircle" and distance(obj["position"], position) < 0.05
                ), None
            )
            assert target_circle_after_teleport is not None
            shortest_path_navigator._position_to_object_id[position_key] = target_circle_after_teleport['objectId']

        object_id = shortest_path_navigator._position_to_object_id[position_key]
        obj = next(
            obj for obj in env.last_event.metadata['objects']
            if obj['objectId'] == object_id
        )
        return self._expert_nav_action_to_obj(obj=obj)

    def _invalidate_interactable_loc_for_pose(
        self, 
        location: Dict[str, Any], 
        obj_pose: Dict[str, Any]
    ) -> bool:
        """Invalidate a given location in the `interactable_positions_cache` as
        we tried to interact but couldn't."""
        env: HomeServiceTHOREnvironment = self.task.env

        if obj_pose is None:
            return False
            
        interactable_positions = env._interactable_positions_cache.get(
            scene_name=env.scene, obj=obj_pose, controller=env.controller
        )
        for i, loc in enumerate([*interactable_positions]):
            if (
                self.shortest_path_navigator.get_key(loc)
                == self.shortest_path_navigator.get_key(location)
                and loc["standing"] == location["standing"]
            ):
                interactable_positions.pop(i)
                return True
        return False

    def _generate_and_record_expert_action(self):
        if self.task.num_steps_taken() == len(self.expert_action_list) + 1:
            get_logger().warning(
                f"Already generated the expert action at step {self.task.num_steps_taken()}"
            )
            return
        assert self.task.num_steps_taken() == len(
            self.expert_action_list
        ), f"{self.task.num_steps_taken()} != {len(self.expert_action_list)}"
        expert_action_dict = self._generate_expert_action_dict()

        if expert_action_dict is None or expert_action_dict["action"] is None:
            self.expert_action_list.append(None)
            return          

        action_str = stringcase.snakecase(expert_action_dict["action"])
        if action_str not in self.task.action_names():
            if "objectType" in expert_action_dict:
                obj_type = stringcase.snakecase(expert_action_dict["objectType"])
                action_str = f"{action_str}_{obj_type}"

        try:
            self.expert_action_list.append(self.task.action_names().index(action_str))
        except ValueError:
            get_logger().error(
                f"{action_str} is not a valid action for the given task."
            )
            self.expert_action_list.append(None)

    def _generate_expert_action_dict(self) -> Dict[str, Any]:
        env: HomeServiceTHOREnvironment = self.task.env

        if env.mode != HomeServiceMode.SNAP:
            raise NotImplementedError(
                f"Expert only defined for 'easy' mode (current mode: {env.mode}"
            )

        # Check current subtask
        subtask_action, subtask_target, subtask_place = self.task.current_subtask
        agent_loc = env.get_agent_location()
        held_object = env.held_object

        # subtask action
        #   - "Done" : target = None, place = None
        #   - "Goto" : place = None
        #   - "Scan" : target = None, place = None
        #   - "Navigate" : place = None
        #   - "Pickup" : place = None
        #   - "Put"
        #   - "Open" : place = None
        #   - "Close" : place = None

        if subtask_action == "Done":
            return dict(action="Done")

        elif subtask_action == "Goto":
            expert_goto_action = self._expert_goto_action_to_scene_type(
                scene_type=subtask_target
            )
            if expert_goto_action is None:
                raise RuntimeError
            
            return dict(action=expert_goto_action)

        elif subtask_action == "Scan":
            with include_object_data(env.controller):
                current_objects = env.last_event.metadata["objects"]
                target_obj = next(
                    (
                        obj for obj in current_objects
                        if obj['objectType'] == env.current_task_spec.pickup_object
                    ), None
                )
                place_receps = None
                if env.current_task_spec.place_receptacle != "User":
                    place_receps = [
                        obj for obj in current_objects
                        if obj['objectType'] == env.current_task_spec.place_receptacle
                    ]
                    if len(place_receps) == 0:
                        place_receps = None
                
                if target_obj is not None:
                    pos = target_obj["axisAlignedBoundingBox"]["center"]
                    self.task.target_positions = {
                        env.current_task_spec.pickup_object: np.array([pos[k] for k in ("x", "y", "z")], dtype=np.float32),
                    }
                if place_receps is not None:
                    pos = place_receps[0]["axisAlignedBoundingBox"]["center"]
                    self.task.target_positions[env.current_task_spec.place_receptacle] = np.array([pos[k] for k in ("x", "y", "z")], dtype=np.float32)
            
            return dict(action="Pass")

        elif subtask_action == "Navigate":
            # from metadata
            # if env.scene != env.current_task_spec.target_scene:
            #     # goto action performed
            #     while self.task.current_subtask[0] == "Goto":
            #         self.task.rollback_subtask()
                    
            #     return dict(action="Pass")
            with include_object_data(env.controller):
                current_objects = env.last_event.metadata["objects"]

                if subtask_target is not None:
                    if not isinstance(subtask_target, str):
                        subtask_target = subtask_target['objectType']
                    target_obj = next(
                        (o for o in current_objects if o['objectType'] == subtask_target), None
                    )

            assert target_obj is not None
            self._last_to_interact_object_pose = target_obj
            expert_nav_action = self._expert_nav_action_to_obj(
                obj=target_obj
            )
            
            # if position is given
            # expert_nav_action = self._expert_nav_action_to_position(
            #     position=self.task.target_positions[subtask_target]
            # )

            if expert_nav_action is None:
                interactable_positions = self._get_interactable_positions(obj=target_obj)
                if len(interactable_positions) != 0:
                    get_logger().debug(
                        f"Could not find a path to the object {target_obj['objectId']}"
                        f" in scene {env.scene}"
                        f" when at position {agent_loc}."
                    )
                else:
                    get_logger().debug(
                        f"Object {target_obj['objectId']} in scene {env.scene}"
                        f" has no interactable positions."
                    )
                return dict(action=expert_nav_action)
            elif expert_nav_action == "Pass":
                # retry = 0
                if not target_obj["visible"]:
                    if self._invalidate_interactable_loc_for_pose(
                        location=agent_loc, obj_pose=target_obj
                    ):
                        return self._generate_expert_action_dict()
                    # import pdb; pdb.set_trace()
                    # raise RuntimeError(" IMPOSSIBLE. ")
                    
                    else:
                        return dict(action="Fail")
                
            # elif expert_nav_action == "Fail":
            #     return dict(action=expert_nav_action)
            
            return dict(action=expert_nav_action)

        elif subtask_action == "Pickup":
            # if env.scene != env.current_task_spec.target_scene:
            #     # goto action performed
            #     while self.task.current_subtask[0] == "Goto":
            #         self.task.rollback_subtask()
                    
            #     return dict(action="Pass")
            with include_object_data(env.controller):
                current_objects = env.last_event.metadata["objects"]

                if subtask_target is not None:
                    if not isinstance(subtask_target, str):
                        subtask_target = subtask_target['objectType']
                    target_obj = next(
                        (o for o in current_objects if o['objectType'] == subtask_target), None
                    )
            assert target_obj is not None and target_obj['visible']
            self._last_to_interact_object_pose = target_obj
            # return dict(action="Pickup", objectType=target_obj["objectType"])
            return dict(action="Pickup")

        elif subtask_action == "Put":
            # if env.scene != env.current_task_spec.target_scene:
            #     # goto action performed
            #     while self.task.current_subtask[0] == "Goto":
            #         self.task.rollback_subtask()
                    
            #     return dict(action="Pass")
            with include_object_data(env.controller):
                current_objects = env.last_event.metadata["objects"]

                if subtask_target is not None:
                    if not isinstance(subtask_target, str):
                        subtask_target = subtask_target['objectType']
                    target_obj = next(
                        (o for o in current_objects if o['objectType'] == subtask_target), None
                    )
                if subtask_place is not None:
                    if not isinstance(subtask_place, str):
                        subtask_place = subtask_place['objectType']
                    place_obj = next(
                        (o for o in current_objects if o['objectType'] == subtask_place), None
                    )
            # if held_object is None:
            #     # Pickup has failed....
            #     for _ in range(3):
            #         self.task.rollback_subtask()
            #     return dict(action="Pass")
            
            assert target_obj is not None and held_object["objectId"] == target_obj["objectId"]
            assert place_obj is not None and place_obj["visible"]
            self._last_to_interact_object_pose = place_obj
            # return dict(action="PutByType", objectType=place_obj["objectType"])
            return dict(action="Put")
            
        elif subtask_action in ["Open", "Close"]:
            # if env.scene != env.current_task_spec.target_scene:
            #     # goto action performed
            #     while self.task.current_subtask[0] == "Goto":
            #         self.task.rollback_subtask()
                    
            #     return dict(action="Pass")
            with include_object_data(env.controller):
                current_objects = env.last_event.metadata["objects"]

                if subtask_target is not None:
                    if not isinstance(subtask_target, str):
                        subtask_target = subtask_target['objectType']
                    target_obj = next(
                        (o for o in current_objects if o['objectType'] == subtask_target), None
                    )
            assert target_obj is not None and target_obj['visible']
            self._last_to_interact_object_pose = target_obj
            return dict(action=f"{subtask_action}ByType", objectType=target_obj["objectType"])

        else:
            raise NotImplementedError(
                f"Subtask {subtask_action} is not implemented."
            )       

def __test():
    from experiments.home_service_base import (
        HomeServiceBaseExperimentConfig,
    )
    from env.tasks import HomeServiceTaskSampler, HomeServiceTaskType
    task_sampler_params = HomeServiceBaseExperimentConfig.stagewise_task_sampler_args(
        stage="train", process_ind=0, total_processes=1, headless=False,
    )

    # from env.utils import save_frames_to_mp4
    task_sampler: HomeServiceTaskSampler = HomeServiceBaseExperimentConfig.make_sampler_fn(
        **task_sampler_params, force_cache_reset=True, epochs=1, 
        task_type=HomeServiceTaskType.SIMPLE_PICK_AND_PLACE,
    )
    random_action_prob = 0.0

    shortest_path_navigator = ShortestPathNavigatorTHOR(
        controller=task_sampler.env.controller, grid_size=STEP_SIZE
    )
    k = 0

    while task_sampler.length > 0:
        print(k)
        random.seed(k)
        k += 1
        task = task_sampler.next_task()
        assert task is not None

        print(f'Pick{stringcase.pascalcase(task.pickup_target["objectType"])}Place{stringcase.pascalcase(task.place_target["objectType"])}')

        greedy_expert = GreedySimplePickAndPlaceExpert(
            task=task, shortest_path_navigator=shortest_path_navigator
        )
        controller = task_sampler.env.controller
        frames = [controller.last_event.frame]
        while not task.is_done():
            if random.random() < random_action_prob:
                assert task.action_names()[0] == "done"
                action_to_take = random.randint(1, len(task.action_names()) - 1)
            else:
                action_to_take = greedy_expert.expert_action

            step_result = task.step(action_to_take)
            # task.env.controller.step("Pass")
            # task.env.controller.step("Pass")
            
            greedy_expert.update(
                action_taken=action_to_take,
                action_success=step_result.info["action_success"]
            )

            frames.append(controller.last_event.frame)


if __name__ == "__main__":
    __test()
