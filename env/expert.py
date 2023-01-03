"""Definitions for a greedy expert for the `Unshuffle` task."""

import copy
import random
import math
import numpy as np
from collections import defaultdict
from typing import (
    Set,
    cast,
    Any,
    List,
    Dict,
    Tuple,
    Optional,
    Union,
    Sequence,
    TYPE_CHECKING,
)

import ai2thor.controller
import ai2thor.server
import networkx as nx
import stringcase
from torch.distributions.utils import lazy_property

from allenact.utils.system import get_logger
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment
from allenact_plugins.ithor_plugin.ithor_util import (
    round_to_factor,
    include_object_data,
)
from env.constants import NOT_PROPER_RECEPTACLES, STEP_SIZE, VISIBILITY_DISTANCE
from env.environment import (
    HomeServiceEnvironment,
    HomeServiceMode,
)
from env.subtasks import SUBTASKS, HomeServiceSimpleSubtaskPlanner

if TYPE_CHECKING:
    from env.tasks import HomeServiceTask

AgentLocKeyType = Tuple[float, float, int, int]
AgentFullLocKeyType = Tuple[
    float, float, int, int, float, bool
]  # x, z, rot, hor, y, standing
PositionKeyType = Tuple[float, float, float]


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


class ShortestPathNavigator:
    """
    # Attributes
    controller : The AI2-THOR controller in which shortest paths are computed.
    """

    def __init__(
        self,
        env: HomeServiceEnvironment,
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
        self.env = env

        self._include_move_left_right = include_move_left_right
        self._position_to_object_id: Dict[PositionKeyType, str] = {}

    '''
    ########################### Properties ###########################
    '''
    @property
    def scene_name(self) -> str:
        """Current ai2thor scene."""
        return self.env.scene

    @property
    def last_action_success(self) -> bool:
        """Was the last action taken by the agent a success?"""
        return self.env.controller.last_event.metadata["lastActionSuccess"]

    @property
    def last_event(self) -> ai2thor.server.Event:
        """Last event returned by the controller."""
        return self.env.controller.last_event

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
    '''
    ########################### Methods ###########################
    '''
    def on_reset(self):
        """Function that must be called whenever the AI2-THOR controller is
        reset."""
        self._current_scene = None
        self._position_to_object_id = {}

    def reachable_points_with_rotations_and_horizons(
        self,
    ) -> List[Dict[str, Union[float, int]]]:
        """Get all the reaachable positions in the scene along with possible
        rotation/horizons."""

        self.env.controller.step(action="GetReachablePositions")
        assert self.last_action_success

        points_slim = self.last_event.metadata["actionReturn"]

        points = []
        for r in [0, 90, 180, 270]:
            for horizon in [-30, 0, 30, 60]:
                for p in points_slim:
                    p = copy.copy(p)
                    p["rotation"] = r
                    p["horizon"] = horizon
                    points.append(p)

        return points

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
    '''
    ########################### Static Methods ###########################
    '''
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

    @staticmethod
    def get_full_key(input_dict: Dict[str, Any], ndigits: int = 2) -> AgentFullLocKeyType:
        key = ShortestPathNavigator.get_key(input_dict=input_dict, ndigits=ndigits)

        assert "standing" in input_dict

        if "y" in input_dict:
            return key + (
                cast(float, input_dict["y"]),
                cast(bool, input_dict["standing"]),
            )
        else:
            return key + (
                cast(float, input_dict["position"]["y"]),
                cast(bool, input_dict["standing"]),
            )

    @staticmethod
    def get_key_from_full(input_key: AgentFullLocKeyType) -> AgentLocKeyType:
        return input_key[:4]

    @staticmethod
    def location_for_full_key(
        key: AgentFullLocKeyType
    ) -> Dict[str, Union[float, int, bool]]:
        x, z, rot, hor, y, standing = key
        return dict(x=x, y=y, z=z, rotation=rot, horizon=hor, standing=standing)


class HomeServiceGreedyActionExpert:
    """An agent which greedily attempts to complete a given task."""

    def __init__(
        self,
        task: "HomeServiceTask",
        shortest_path_navigator: ShortestPathNavigator,
        max_priority: int = 3,
        steps_for_time_pressure: int = 250,
        exploration_enabled: bool = True,
        **kwargs,
    ):
        get_logger().debug(
            f"Expert started for {task.env.scene} (exploration: {exploration_enabled})"
        )
        self.exploration_enabled = exploration_enabled

        self.task = task
        assert self.task.num_steps_taken() == 0

        self.shortest_path_navigator = shortest_path_navigator

        self._last_to_target_recep_id: Optional[str] = None
        self.scanned_receps = set()
        self._current_object_target_keys: Optional[Set[AgentLocKeyType]] = None
        self.recep_id_loc_per_room = dict()
        self.cached_locs_for_recep = dict()
        self.scanned_objects = set()
        self.object_id_loc_per_recep_id = dict()
        self.cached_locs_for_objects = dict()

        self.max_priority = max_priority
        self.obj_id_to_priority: defaultdict = defaultdict(lambda: 0)
        self.visited_recep_ids_per_room = {
            room: set() for room in self.env.room_to_poly
        }
        self.unvisited_recep_ids_per_room = self.env.room_to_static_receptacle_ids()
        self.visited_object_ids_per_recep = {
            recep: set()
            for room in self.env.room_to_poly
            for recep in self.unvisited_recep_ids_per_room[room]
        }
        self.unvisited_object_ids_per_recep = {
            recep: set(obj["receptacleObjectIds"])
            for recep in self.visited_object_ids_per_recep
            for obj in self.env.objects()
            if obj["objectId"] == recep
        }

        self.steps_for_time_pressure = steps_for_time_pressure

        self.last_expert_mode: Optional[str] = None

        self.expert_action_list: List[int] = []

        self._last_held_object_id: Optional[str] = None
        self._last_to_interact_object_pose: Optional[Dict[str, Any]] = None
        self._id_of_object_we_wanted_to_pickup: Optional[str] = None

        self.shortest_path_navigator.on_reset()
        self.update(action_taken=None, action_success=None)

    '''
    ########################### Properties ###########################
    '''
    @property
    def env(self) -> HomeServiceEnvironment:
        return self.task.env

    @property
    def expert_action(self) -> int:
        """Get the current greedy expert action.

        # Returns An integer specifying the expert action in the current
        state. This corresponds to the order of actions in
        `self.task.action_names()`. For this action to be available the
        `update` function must be called after every step.
        """
        # assert self.task.num_steps_taken() == len(self.expert_action_list) - 1
        return self.expert_action_list[-1]

    '''
    ########################### Methods ###########################
    '''
    def _expert_nav_action_to_room(
        self,
        room: str,
        xz_tol: float = 0.75,
        horizon=30,
        future_agent_loc: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Get the shortest path navigational action towards the current room_id's centroid.
        """
        env = self.env
        shortest_path_navigator = self.shortest_path_navigator

        if future_agent_loc is None:
            agent_loc = env.get_agent_location()
        else:
            agent_loc = future_agent_loc
        source_state_key = shortest_path_navigator.get_key(agent_loc)

        goal_xz = env.room_reachable_centroid(room)
        if goal_xz is None:
            get_logger().debug(
                f"ERROR: Unable to find reachable location near {room}'s centroid."
            )
            return None
        
        goal_loc = dict(x=goal_xz[0], z=goal_xz[1], rotation=0, horizon=horizon,)
        target_key = shortest_path_navigator.get_key(goal_loc)

        action = "Pass"
        if (
            abs(source_state_key[0] - target_key[0]) > xz_tol
            or abs(source_state_key[1] - target_key[1]) > xz_tol
        ):
            try:
                action = shortest_path_navigator.shortest_path_next_action(
                    source_state_key=source_state_key, goal_state_key=target_key,
                )
            except nx.NetworkXNoPath as _:
                action = None

        return action

    def _expert_nav_action_to_obj(
        self,
        obj: Dict[str, Any],
        force_standing: Optional[int] = None,
        force_horizon: Optional[bool] = None,
        objs_on_recep: Optional[Set[str]] = None,
        future_agent_loc: Dict[str, Any] = None,
        recep_target_keys: Optional[Set[AgentLocKeyType]] = None,
    ) -> Optional[str]:
        """
        Get the shortest path navigational action towards the object obj.
        The navigational action takes us to a position from which the object is interactable.
        """
        env: HomeServiceEnvironment = self.env
        if future_agent_loc is None:
            agent_loc = env.get_agent_location()
        else:
            agent_loc = future_agent_loc
        shortest_path_navigator = self.shortest_path_navigator

        interactable_positions = None
        if recep_target_keys is None:
            reachable_positions = env.controller.step(
                "GetReachablePositions",
            ).metadata["actionReturn"]

            interactable_positions = env._interactable_positions_cache.get(
                scene_name=env.scene,
                obj=obj,
                controller=env,
                reachable_positions=reachable_positions,
                force_horizon=force_horizon,
                force_standing=force_standing,
                avoid_teleport=objs_on_recep is not None,
            )

            if len(interactable_positions) == 0:
                self._current_object_target_keys = set()
                return None

            if objs_on_recep is not None:
                interactable_positions = self._search_locs_to_interact_with_objs_on_recep(
                    obj,
                    interactable_positions,
                    objs_on_recep,
                    force_horizon,
                    force_standing,
                )
            
            full_target_keys = [
                shortest_path_navigator.get_full_key(loc)
                for loc in interactable_positions
            ]
        else:
            full_target_keys = list(recep_target_keys)

        if future_agent_loc is None:
            self._current_object_target_keys = set(full_target_keys)

        if len(full_target_keys) == 0:
            return None

        source_state_key = shortest_path_navigator.get_key(agent_loc)
        target_keys = [
            shortest_path_navigator.get_key_from_full(key)
            for key in full_target_keys
        ]

        action = "Pass"
        if source_state_key not in target_keys:
            try:
                action = shortest_path_navigator.shortest_path_next_action_multi_target(
                    source_state_key=source_state_key,
                    goal_state_keys=target_keys,
                )
            except nx.NetworkXNoPath as _:
                return None
        
        if action != "Pass":
            return action
        
        else:
            tol = 1e-2
            if interactable_positions is None:
                interactable_positions = [
                    shortest_path_navigator.location_for_full_key(key)
                    for key in full_target_keys
                ]
                tol = 2e-2
            return self.crouch_stand_if_needed(
                interactable_positions, agent_loc, tol=tol
            )

    def _search_locs_to_interact_with_objs_on_recep(
        self,
        obj: Dict[str, Any],
        interactable_positions: List[Dict[str, Union[float, int, bool]]],
        objs_on_recep: Set[str],
        force_horizon: int,
        force_standing: bool,
    ) -> List[Dict[str, Union[float, int, bool]]]:
        # Try to find an interactable positions for all objects on objs_on_recep
        interactable_positions = self._try_to_interact_with_objs_on_recep(
            obj["objectId"],
            interactable_positions,
            objs_on_recep,      # modified in-place
            force_horizon,
            force_standing
        )

        # Try to get close to the target
        obj_loc = tuple(obj["position"][x] for x in "xyz")
        radius = 0.7        # emprically, it seems unlikely to find a valid location closer than 0.7
        new_positions = []
        unused_positions = set(
            tuple(p[x] for x in ["x", "y", "z", "rotation", "standing", "horizon"])
            for p in interactable_positions
        )

        while len(new_positions) == 0:
            available_locs = list(unused_positions)
            for loc in available_locs:
                if sum((loc[x] - obj_loc[x]) ** 2 for x in [0, 2]) <= radius * radius:
                    new_positions.append(loc)
                    unused_positions.remove(loc)
            radius += 0.2

        return [
            {
                x: p[ix]
                for ix, x in enumerate(
                    ["x", "y", "z", "rotation", "standing", "horizon"]
                )
            }
            for p in new_positions
        ]

    def exploration_pose(self, horizon=30):
        if not self.env.last_event.metadata["agent"]["isStanding"]:
            return dict(action="Stand")
        if round(self.env.last_event.metadata["agent"]["cameraHorizon"]) > horizon + 15:
            return dict(action="LookUp")
        if round(self.env.last_event.metadata["agent"]["cameraHorizon"]) < horizon - 15:
            return dict(action="LookDown")
        return None

    def uncovered_ids_on_recep(self, recep_obj, max_objs_to_check=6):
        return set(recep_obj["receptacleObjectIds"][:max_objs_to_check])
    
    def update_visited_receps(self, horizon: int = 30):
        if self.env.current_room is None:
            if self.env._last_room_id is not None:
                rooms_to_check = [self.env._last_room_id]
            else:
                return
        else:
            rooms_to_check = [self.env.current_room]

        get_logger().debug(
            f"rooms to check during updating visited receps: {rooms_to_check} "
        )
        for room in rooms_to_check:
            if room not in self.recep_id_loc_per_room:
                recep_ids = self.unvisited_recep_ids_per_room[room]
                self.recep_id_loc_per_room[room] = self.env.object_ids_with_locs(
                    list(recep_ids)
                )
        
        self.get_unscanned_receps(
            rooms_to_check=rooms_to_check,
            horizon=horizon,
            standing=True,
        )

    def get_unscanned_receps(self, rooms_to_check, standing=True, horizon=30):
        # agent_key = self.shortest_path_navigator.get_key(self.env.get_agent_location())
        agent_key = self.shortest_path_navigator.get_full_key(self.env.get_agent_location())
        all_objects = self.env.objects()

        for room in rooms_to_check:
            recep_ids_to_check = list(self.unvisited_recep_ids_per_room[room])
            for current_recep_id in recep_ids_to_check:
                if current_recep_id in self.scanned_receps:
                    get_logger().debug(
                        f"ERROR: {current_recep_id} already in `self.scanned_receps`."
                    )
                    self.unvisited_recep_ids_per_room[room].remove(current_recep_id)
                    self.visited_recep_ids_per_room[room].add(current_recep_id)
                    continue

                current_recep = next(
                    o for o in all_objects if o["objectId"] == current_recep_id
                )
                
                uncovered = self.uncovered_ids_on_recep(current_recep)

                if current_recep["objectId"] not in self.cached_locs_for_recep:
                    needs_action = self._expert_nav_action_to_obj(
                        current_recep,
                        force_standing=standing,
                        force_horizon=horizon,
                        objs_on_recep=uncovered.copy(),
                    )
                    if needs_action is None:
                        self._expert_nav_action_to_obj(
                            current_recep,
                            objs_on_recep=uncovered,
                        )
                        if len(self._current_object_target_keys):
                            get_logger().debug(
                                f"Access {current_recep_id} by underconstraining the agent pose"
                            )
                    
                    get_logger().debug(
                        f"Update cached_locs_for_recep[{current_recep['objectId']}]"
                    )
                    self.cached_locs_for_recep[current_recep["objectId"]] = self._current_object_target_keys

                if agent_key in self.cached_locs_for_recep[current_recep["objectId"]]:
                    get_logger().debug(
                        f"Agent is located nearby [{current_recep['objectId']}]"
                    )
                    self.visited_recep_ids_per_room[room].add(current_recep_id)
                    self.scanned_receps.add(current_recep_id)
                    self.unvisited_recep_ids_per_room[room].remove(current_recep_id)
                    if current_recep_id == self._last_to_target_recep_id:
                        self._last_to_target_recep_id = None

                    for obj in uncovered:
                        if (
                            obj in self.cached_locs_for_objects
                            and agent_key in self.cached_locs_for_objects[obj]
                        ):
                            get_logger().debug(
                                f"Agent is located nearby [{obj}]"
                            )
                            if obj not in self.visited_object_ids_per_recep[current_recep['objectId']]:
                                self.visited_object_ids_per_recep[current_recep['objectId']].add(obj)
                            if obj not in self.scanned_objects:
                                self.scanned_objects.add(obj)
                            if obj in self.unvisited_object_ids_per_recep[current_recep['objectId']]:
                                self.unvisited_object_ids_per_recep[current_recep['objectId']].remove(obj)

    def current_direction(self):
        agent_rot = self.env.last_event.metadata["agent"]["rotation"]["y"] % 360
        if 225 <= agent_rot < 315:  # 270
            direction = (-1, 0)
        elif 315 <= agent_rot or agent_rot < 45:  # 0 (360)
            direction = (0, 1)
        elif 45 <= agent_rot <= 135:  # 90
            direction = (1, 0)
        else:  # if 135 <= agent_rot < 225:  # 180
            direction = (0, -1)
        return direction

    def time_pressure(self):
        if self.task.num_steps_taken() == self.steps_for_time_pressure:
            get_logger().debug(
                f"Expert rushing due to time pressure ({self.steps_for_time_pressure} steps)"
            )
        return self.task.num_steps_taken() >= self.steps_for_time_pressure

    def _log_output_mode(self, expert_mode: str, action_dict: Dict[str, Any]) -> Dict[str, Any]:
        if self.last_expert_mode != expert_mode:
            get_logger().debug(f"Change mode from [{self.last_expert_mode}] to [{expert_mode}]")
            self.last_expert_mode = expert_mode
        return action_dict

    def _try_to_interact_with_objs_on_recep(
        self,
        recep_obj_id: str,
        interactable_positions: List[Dict[str, Union[float, int, bool]]],
        objs_on_recep: Set[str],
        horizon: int,
        standing: bool,
    ) -> List[Dict[str, Union[float, int, bool]]]:
        env = self.env
        # Try to (greedily) find an interactable positions for all/most objects on objs_on_recep
        last_interactable = interactable_positions
        missing_objs_on_recep = copy.copy(objs_on_recep)
        if recep_obj_id not in self.object_id_loc_per_recep_id:
            self.object_id_loc_per_recep_id[recep_obj_id] = self.env.object_ids_with_locs(missing_objs_on_recep)
        for obj_id in missing_objs_on_recep:
            obj = next((o for o in env.objects() if o["objectId"] == obj_id), None)
            if obj is None:
                continue
            new_interactable = env._interactable_positions_cache.get(
                scene_name=env.scene,
                obj=obj,
                controller=env,
                reachable_positions=HomeServiceGreedyActionExpert.extract_xyz(
                    last_interactable
                ),
                force_horizon=horizon,
                force_standing=standing,
                avoid_teleport=True,
            )
            if len(new_interactable) > 0:
                objs_on_recep.remove(obj_id)
                get_logger().debug(
                    f"Update cached_locs_for_objects[{obj_id}]"
                )
                self.cached_locs_for_objects[obj_id] = set(
                    [
                        self.shortest_path_navigator.get_full_key(loc)
                        for loc in new_interactable
                    ]
                )
                last_interactable = new_interactable

        return last_interactable

    def _invalidate_interactable_loc_for_pose(
        self, location: Dict[str, Any], obj_pose: Dict[str, Any]
    ) -> bool:
        """Invalidate a given location in the `interactable_positions_cache` as
        we tried to interact but couldn't."""
        env = self.task.env

        interactable_positions = env._interactable_positions_cache.get(
            scene_name=env.scene, obj=obj_pose, controller=env
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

    def manage_held_object(self) -> Optional[Dict[str, Any]]:
        if self.env.held_object is None:
            return None

        get_logger().debug(
            f"In `manage_held_object` method..."
        )
        # self._last_to_interact_object_pose = None

        # Should navigate to a position where the held object can be placed
        self._last_to_interact_object_pose = {
            **self.env.held_object,
            **{
                k: self.env.object_id_to_target_pose[
                    self.env.held_object["objectId"]
                ][k]
                for k in ["position", "rotation"]
            },
        }
        expert_nav_action = self._expert_nav_action_to_obj(
            obj=self._last_to_interact_object_pose
        )
        if self.env.held_object["objectId"] == self.env.target_object_id:
            if (
                self.exploration_enabled
                and self.env.target_recep_id not in self.cached_locs_for_recep
            ):
                get_logger().debug(
                    f"WARNING: Target receptacle is not observed. Scan it first!"
                    f"Move to the start pose of the held object!"
                )
                # move held object to initial position
                expert_nav_action = self._expert_nav_action_to_obj(
                    obj={
                        **self.env.held_object,
                        **{
                            k: self.env.object_id_to_start_pose[
                                self.env.held_object["objectId"]
                            ][k]
                            for k in ["position", "rotation"]
                        },
                    },
                )
            elif self.env.current_room != self.env.target_recep_room_id:
                get_logger().debug(
                    f"The agent is not in target recep room."
                )
                return self.move_to_target_recep_room()
            # elif self.exploration_enabled:
            #     # move to target recep
            #     target_recep = next(
            #         o for o in self.env.objects() if o["objectId"] == self.env.target_recep_id
            #     )
            #     self._last_to_interact_object_pose = target_recep
            #     expert_nav_action = self._expert_nav_action_to_obj(
            #         self._last_to_interact_object_pose,
            #         # recep_target_keys=self.cached_locs_for_recep[self.env.target_recep_id]
            #     )

        if expert_nav_action is None:
            # Could not find a path to the target,
            get_logger().debug(
                f"Drop held object"
            )
            return dict(action="DropHeldObjectWithSnap")
        elif expert_nav_action == "Pass":
            # We're in a position where we can put the object
            parent_ids = self.env.object_id_to_target_pose[self.env.held_object["objectId"]]["parentReceptacles"]
            if parent_ids is not None and len(parent_ids) == 1:
                parent_type = stringcase.pascalcase(
                    self.env.object_id_to_target_pose[parent_ids[0]]["objectType"]
                )
                get_logger().debug(
                    f"Returning dict(action=Put{parent_type}"
                )
                return dict(action=f"Put{parent_type}")
            else:
                return dict(action="DropHeldObjectWithSnap")
        else:
            return dict(action=expert_nav_action)
        

    def update(self, action_taken: Optional[int], action_success: Optional[bool]):
        """Update the expert with the last action taken and whether or not that
        action succeeded."""
        if action_taken is not None:
            assert action_success is not None

            get_logger().debug(
                f"STEP-{self.task.num_steps_taken()}: action[{self.task.action_names()[action_taken]}({action_taken})] action_success[{action_success}]"
            )
            action_names = self.task.action_names()
            last_expert_action = self.expert_action_list[-1]
            agent_took_expert_action = action_taken == last_expert_action
            action_str = action_names[action_taken]

            was_nav_action = any(k in action_str for k in ["move", "rotate", "look"])

            if "pickup_" in action_str and agent_took_expert_action and action_success:
                self._id_of_object_we_wanted_to_pickup = self._last_to_interact_object_pose["objectId"]

            if ("put_" in action_str or "drop_held" in action_str) and agent_took_expert_action and action_success:
                if self._id_of_object_we_wanted_to_pickup is not None:
                    self.obj_id_to_priority[
                        self._id_of_object_we_wanted_to_pickup
                    ] += 1
                else:
                    self.obj_id_to_priority[
                        self._last_held_object_id
                    ] += 1

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
                    or "put_" in action_str
                ) and action_taken == last_expert_action:
                    assert self._last_to_interact_object_pose is not None
                    get_logger().debug(
                        f"Action {action_str} FAILED. "
                        f"Invalidate current location [{self.task.env.get_agent_location()}] "
                        f"for object pose [{self._last_to_interact_object_pose}]."
                    )
                    self._invalidate_interactable_loc_for_pose(
                        location=self.task.env.get_agent_location(),
                        obj_pose=self._last_to_interact_object_pose,
                    )
                elif (
                    ("crouch" in action_str or "stand" in action_str)
                    and self.task.env.held_object is not None
                    and action_taken == last_expert_action
                ):
                    held_object_id = self.task.env.held_object["objectId"]
                    agent_loc = self.task.env.get_agent_location()
                    agent_loc["standing"] = not agent_loc["standing"]
                    self._invalidate_interactable_loc_for_pose(
                        location=agent_loc,
                        # obj_pose=self.task.env.object_id_to_target_pose[held_object_id],
                        obj_pose=self._last_to_interact_object_pose,
                    )

            else:
                # If the action succeeded and was not a move action then let's force an update
                # of our currently targeted object
                if not was_nav_action and not (
                    "crouch" in action_str or "stand" in action_str
                ):
                    self._last_to_interact_object_pose = None

        held_object = self.task.env.held_object
        if self.task.env.held_object is not None:
            self._last_held_object_id = held_object["objectId"]

        self._generate_and_record_expert_action()

    def _generate_and_record_expert_action(self):
        """Generate the next greedy expert action and save it to the
        `expert_action_list`."""
        # if self.task.num_steps_taken() == len(self.expert_action_list) + 1:
        #     get_logger().warning(
        #         f"Already generated the expert action at step {self.task.num_steps_taken()}"
        #     )
        #     return

        # assert self.task.num_steps_taken() == len(
        #     self.expert_action_list
        # ), f"{self.task.num_steps_taken()} != {len(self.expert_action_list)}"
        expert_action_dict = self._generate_expert_action_dict()
        if expert_action_dict is None:
            self.expert_action_list.append(None)
            return

        action_str = stringcase.snakecase(expert_action_dict["action"])
        if action_str not in self.task.action_names():
            current_objectId = expert_action_dict["objectId"]
            current_obj = next(
                o for o in self.task.env.objects() if o["objectId"] == current_objectId
            )
            obj_type = stringcase.snakecase(current_obj["objectType"])
            action_str = f"{action_str}_{obj_type}"
            
        try:
            self.expert_action_list.append(self.task.action_names().index(action_str))
        except ValueError:
            get_logger().error(
                f"{action_str} is not a valid action for the given task."
            )
            self.expert_action_list.append(None)

    def _generate_expert_action_dict(self, horizon=30) -> Optional[Dict[str, Any]]:
        """Generate a dictionary describing the next greedy expert action."""
        if self.env.mode != HomeServiceMode.SNAP:
            raise NotImplementedError(
                f"Expert only defined for 'easy' mode (current mode: {self.env.mode}"
            )

        try:
            # Try to transport or put/drop the current object
            attempt = self.manage_held_object()
            if attempt is not None:
                return self._log_output_mode("held object", attempt)
            
            if self.exploration_enabled:
                self.update_visited_receps(horizon=horizon)

                if not self.time_pressure():
                    # Try to pickup the target object
                    attempt = self.home_service_pickup(mode="causal")
                    if attempt is not None:
                        return self._log_output_mode("home_service_pickup (causal)", attempt)
                    
                    # Try to scan for recep
                    attempt = self.scan_for_target_object(horizon=horizon)
                    if attempt is not None:
                        return self._log_output_mode("scan object", attempt)

                    # Try to scan for recep
                    attempt = self.scan_for_target_recep(horizon=horizon)
                    if attempt is not None:
                        return self._log_output_mode("scan recep", attempt)

            attempt = self.home_service_pickup(mode="whole")
            if attempt is not None:
                return self._log_output_mode("home_service_pickup (whole)", attempt)

            if self.exploration_enabled:
                if self.env.target_recep_room_id != self.env.current_room:
                    get_logger().debug(
                        f"WARNING: We cannot generate more actions despite being in {self.env.current_room},"
                        f" away from the target {self.env.target_recep_room_id}. Terminating"
                    )
            
            return dict(action="Done")

        except:
            import traceback

            get_logger().debug(f"EXCEPTION: Expert failure: {traceback.format_exc()}")
            return None

    def pose_action_if_compatible(
        self,
        nav_action: str,
        pose_action: Optional[Dict[str, Any]],
        horizon: int,
        current_recep: Optional[Dict[str, Any]] = None,
        standing: bool = None,
        room_id: Optional[str] = None,
    ):
        assert (current_recep is None) != (room_id is None)
        get_logger().debug(
            f"In `pose_action_if_compatible` method..."
            f"nav_action: {nav_action}, pose_action: {pose_action}"
        )

        if pose_action is None:
            get_logger().debug(
                f"Returning nav_action: {nav_action} as pose_action is None"
            )
            return nav_action
        
        future_agent_loc = self.env.get_agent_location()
        
        if pose_action["action"] == "Stand":
            future_agent_loc["standing"] = True
            incompatible_action = "Crouch"
        elif pose_action["action"] == "LookUp":
            future_agent_loc["horizon"] -= 30
            incompatible_action = "LookDown"
        elif pose_action["action"] == "LookDown":
            future_agent_loc["horizon"] += 30
            incompatible_action = "LookUp"

        get_logger().debug(
            f"future_agent_loc: {future_agent_loc}"
        )
        
        if current_recep is not None:
            future_nav_action = self._expert_nav_action_to_obj(
                current_recep,
                force_standing=standing,
                force_horizon=horizon,
                future_agent_loc=future_agent_loc,
                recep_target_keys=self.cached_locs_for_recep[current_recep["objectId"]],
            )
        else:
            future_nav_action = self._expert_nav_action_to_room(
                cast(str, room_id),
                horizon=horizon,
                future_agent_loc=future_agent_loc,
            )
        
        get_logger().debug(
            f"future_nav_action: {future_nav_action}"
        )
        if future_nav_action == incompatible_action:
            get_logger().debug(
                f"Returning nav_action: {nav_action}"
            )
            return nav_action

        if (
            len(self.expert_action_list) > 0
            and self.expert_action is not None
            and stringcase.pascalcase(self.task.action_names()[self.expert_action]) == pose_action['action']
            and not self.task.actions_taken_success[-1]
        ):
            get_logger().debug(
                f"Returning nav_action: {nav_action}"
            )
            return nav_action
        get_logger().debug(
            f"Returning pose_action: {pose_action}"
        )
        return pose_action

    def move_to_target_room(self, target_room: str, standing: bool = True, horizon: int = 30):
        env = self.env
        if env.current_room is None:
            get_logger().debug(
                f"In `env.current_room` is None..."
            )
            return None

        if target_room is None:
            get_logger().debug(
                f"In `target_room` is None..."
            )
            return None

        pose_act = self.exploration_pose(horizon=horizon)
        if env.current_room == target_room:
            get_logger().debug(
                "ERROR: We're already in the target room."
            )
            return None

        nav_act = self._expert_nav_action_to_room(target_room, horizon=horizon)
        if nav_act == "Pass":
            get_logger().debug(
                f"ERROR: Received 'Pass' when navigating to {target_room} but agent is located in {env.current_room}."
            )
            return None
        elif nav_act is None:
            get_logger().debug(
                f"ERROR: Failed to navigate to {target_room}; Received 'None'."
            )
            return None

        return self.pose_action_if_compatible(
            nav_action=dict(action=nav_act),
            pose_action=pose_act,
            horizon=horizon,
            room_id=target_room,
        )

    def move_to_target_recep_room(self, standing: bool = True, horizon: int = 30):
        get_logger().debug(
            f"In `move_to_target_recep_room` method..."
            f"Generate navigatorial action to move towards target recep room[{self.env.target_recep_room_id}]"
            f" from current room[{self.env.current_room}]"
        )
        return self.move_to_target_room(
            target_room=self.env.target_recep_room_id,
            standing=standing,
            horizon=horizon
        )

    def scan_for_target_recep(self, standing: bool = True, horizon: int = 30):
        get_logger().debug(
            f"In `scan_for_target_recep` method..."
            f"Generate navigatorial action to find out the target recep."
        )
        env = self.env

        if self.env.current_room is None:
            return None
        
        target_recep_id = env.target_recep_id
        self._last_to_target_recep_id = target_recep_id

        if target_recep_id in self.scanned_receps:
            get_logger().debug(
                f'Target recep {target_recep_id} is already found.'
            )
            return None

        if self.obj_id_to_priority[target_recep_id] > self.max_priority:
            get_logger().debug(
                f"Priority for object[{target_recep_id}] ({self.obj_id_to_priority[target_recep_id]})"
                f" exceeded the max priority[{self.max_priority}]"
            )
            return None

        pose_act = self.exploration_pose(horizon=horizon)
        target_recep_room = env.target_recep_room_id
        target_recep = next(
            o for o in env.objects() if o["objectId"] == target_recep_id
        )
        
        if self.env.current_room != target_recep_room:
            get_logger().debug(
                f"Agent[{self.env.current_room}] is not located on the target recep room[{target_recep_room}]..."
                f"Let's move to the target recep room"
            )
            return self.move_to_target_recep_room(standing=standing, horizon=horizon)

        get_logger().debug(
            f"Agent[{self.env.current_room}] is arrived to the target recep room[{target_recep_room}]..."
            f"Let's find out the target recep!"
        )
        
        assert target_recep_id in self.cached_locs_for_recep, f"it should be updated!"
        nav_needed = self._expert_nav_action_to_obj(
            target_recep,
            force_standing=standing,
            force_horizon=horizon,
            recep_target_keys=self.cached_locs_for_recep[target_recep_id],
        )
        get_logger().debug(
            f"nav_needed: {nav_needed} & pose_act: {pose_act}"
        )
        
        if nav_needed is None:
            if pose_act is not None:
                return pose_act
            get_logger().debug(
                f"Failed to navigate to {target_recep_id} in {env.current_room} during scan"
                f" re-try navigation for {target_recep_id}."
            )
            self.obj_id_to_priority[target_recep_id] += 1
            return self.scan_for_target_recep(standing, horizon)
        
        if nav_needed != "Pass":
            return self.pose_action_if_compatible(
                nav_action=dict(action=nav_needed),
                pose_action=pose_act,
                current_recep=target_recep,
                standing=standing,
                horizon=horizon,
            )
            
        if len(self._current_object_target_keys) > 0:
            self.scanned_receps.add(target_recep_id)
            self.unvisited_recep_ids_per_room[self.env.current_room].remove(
                target_recep_id
            )
            self.visited_recep_ids_per_room[self.env.current_room].add(target_recep_id)
            get_logger().debug(
                f"Agent Found the Target Recep 9' 3')9"
            )
            return self.scan_for_target_object(standing=standing, horizon=horizon)
        
        return None

    def move_to_target_object_room(self, standing: bool = True, horizon: int = 30):
        get_logger().debug(
            f"In `move_to_target_object_room` method..."
        )
        return self.move_to_target_room(
            target_room=self.env.target_object_room_id,
            standing=standing,
            horizon=horizon
        )

    def scan_for_target_object(self, standing=True, horizon=30):
        get_logger().debug(
            f"In `scan_for_target_object` method..."
        )
        env = self.env

        if self.env.current_room is None:
            return None

        target_recep_id = env.target_recep_id
        self._last_to_target_recep_id = target_recep_id
        if self.obj_id_to_priority[target_recep_id] > self.max_priority:
            get_logger().debug(
                f"Priority for object[{target_recep_id}] ({self.obj_id_to_priority[target_recep_id]})"
                f" exceeded the max priority[{self.max_priority}]"
            )
            return None
        
        if target_recep_id not in self.scanned_receps:
            get_logger().debug(
                f"Target recep should be found before trying to find out the target object"
            )
            return None

        target_object_id = env.target_object_id
        if self.obj_id_to_priority[target_object_id] > self.max_priority:
            get_logger().debug(
                f"Priority for object[{target_object_id}] ({self.obj_id_to_priority[target_object_id]})"
                f" exceeded the max priority[{self.max_priority}]"
            )
            return None

        target_object_room = env.target_object_room_id
        target_object_receptacle = None
        if target_object_id in self.scanned_objects:
            for k, v in self.visited_object_ids_per_recep.items():
                if target_object_id in v:
                    target_object_receptacle = k
                    break
            get_logger().debug(
                f"Target object is already found on {target_object_receptacle} in {target_object_room}"
            )
            return None

        if env.current_room != target_object_room:
            get_logger().debug(
                f"Agent[{self.env.current_room}] is not located in the target object room[{target_object_room}]..."
                f"Let's move to the target object room"
            )
            return self.move_to_target_object_room(standing=standing, horizon=horizon)

        get_logger().debug(
            f"Agent[{self.env.current_room}] arrived to the target object room[{target_object_room}]..."
            f"Let's find out the target object!"
        )

        target_object = next(
            o for o in env.objects() if o["objectId"] == target_object_id
        )
        if target_object_receptacle is None:
            for recep, objs in self.unvisited_object_ids_per_recep.items():
                if target_object_id in objs:
                    target_object_receptacle = recep
                    break
        
        nav_needed = self._expert_nav_action_to_obj(
            target_object,
            force_standing=standing,
            force_horizon=horizon,
            # recep_target_keys=self.cached_locs_for_recep[target_object_receptacle]
        )
        get_logger().debug(
            f"generate expert nav action to obj [{nav_needed}]"
            f" with standing {standing} and horizon {horizon}"
            # f" and recep target keys {self.cached_locs_for_recep[target_object_receptacle]}"
        )
        
        if nav_needed is None:
            get_logger().debug(
                f"Failed to navigate to {target_object_id} in {env.current_room} during scan"
                f" re-try navigation for {target_object_id}."
            )

            nav_needed = self._expert_nav_action_to_obj(
                target_object,
            )
            if nav_needed is None:
                interactable_positions = env._interactable_positions_cache.get(
                    scene_name=self.env.scene,
                    obj=target_object,
                    controller=self.env,
                )
                if len(interactable_positions) != 0:
                    get_logger().debug(
                        f"Could not find a path to {target_object['objectId']}"
                        f" in scene {self.task.env.scene}"
                        f" when at position {self.task.env.get_agent_location()}."
                    )
                else:
                    get_logger().debug(
                        f"Object {target_object['objectId']} in scene {self.task.env.scene}"
                        f" has no interactable positions."
                    )
                self.obj_id_to_priority[target_object_id] += 1
                return self.scan_for_target_object(standing, horizon)
            
        if nav_needed == "Pass":
            with include_object_data(self.env.controller):
                visible_objects = {
                    o["objectId"]
                    for o in self.env.last_event.metadata["objects"]
                    if o["visible"]
                }
            if target_object["objectId"] not in visible_objects:
                get_logger().debug(
                    f"target object {target_object['objectId']} is not visible"
                )
                get_logger().debug(
                    f"Try to invalidate the interactable loc {env.get_agent_location()}"
                    f" for obj_pose [{target_object}]"
                )
                if self._invalidate_interactable_loc_for_pose(
                    location=env.get_agent_location(),
                    obj_pose=target_object,
                ):
                    get_logger().debug(
                        f"Invalidated the interactable loc {env.get_agent_location()}"
                        f" for obj_pose [{target_object}]"
                    )
                    return self.scan_for_target_object(standing=standing, horizon=horizon)

                get_logger().debug(
                    f"Failed to invalidate the loc."
                )

                if self.shortest_path_navigator.get_full_key(
                    env.get_agent_location()
                ) in self.cached_locs_for_recep[target_object_receptacle]:
                    get_logger().debug(
                        f"Remove the current location from the cached locs for recep"
                    )
                    self.cached_locs_for_recep[target_object_receptacle].remove(
                        self.shortest_path_navigator.get_full_key(env.get_agent_location())
                    )
                    get_logger().debug(
                        f"Scan the target object with removed cached locs"
                    )
                    return self.scan_for_target_object(standing, horizon)

                interactable_positions = env._interactable_positions_cache.get(
                    scene_name=self.env.scene,
                    obj=target_object,
                    controller=self.env,
                )
                for ip in interactable_positions:
                    if (
                        ip["x"] == env.get_agent_location()["x"]
                        and ip["z"] == env.get_agent_location()["z"]
                        and ip["rotation"] == env.get_agent_location()["rotation"]
                    ):
                        get_logger().debug(
                            f"Change the standing and horizon constraints at current location"
                            f" then find out the target object [{ip}]"
                        )
                        return self.scan_for_target_object(standing=ip["standing"], horizon=ip["horizon"])

                get_logger().debug(
                    f"No interactable positions available at the current location"
                )
                
                get_logger().debug(
                    f"ERROR: IMPOSSIBLE RESULT..."
                )
                return None
            
            get_logger().debug(
                f"Agent Found the Target object {target_object['objectId']} 9' 3')9"
                f" and arrived to the interactable position"
            )
            self.unvisited_object_ids_per_recep[target_object_receptacle].remove(target_object['objectId'])
            self.scanned_objects.add(target_object['objectId'])
            self.visited_object_ids_per_recep[target_object_receptacle].add(target_object['objectId'])
            
            return self.home_service_pickup(mode="causal")
        
        get_logger().debug(
            f"Returning NAV_ACTION [{nav_needed}]"
        )
        return dict(action=nav_needed)

    def home_service_pickup(self, mode):
        get_logger().debug(
            f"In `home_service_pickup` method..."
        )
        if self.exploration_enabled and mode == "causal":
            if not (
                self.env.target_recep_id in self.scanned_receps
                and self.env.target_object_id in self.scanned_objects
            ):
                if self.env.target_object_id not in self.scanned_objects:
                    get_logger().debug(
                        f"Target object {self.env.target_object_id} not found yet."
                    )
                if self.env.target_recep_id not in self.scanned_receps:
                    get_logger().debug(
                        f"Target receptacle {self.env.target_recep_id} not found yet."
                    )
                return None

        target_object = next(
            o for o in self.env.objects() if o["objectId"] == self.env.target_object_id
        )
        target_recep = next(
            o for o in self.env.objects() if o["objectId"] == self.env.target_recep_id
        )
        if (
            target_object["objectId"] in target_recep["receptacleObjectIds"]
        ):
            get_logger().debug(
                f"target object is located on the target recep "
                f"Task DONE!!!!"
            )
            return None

        if self.obj_id_to_priority[self.env.target_object_id] > self.max_priority:
            get_logger().debug(
                f"Priority for object[{self.env.target_object_id}] ({self.obj_id_to_priority[self.env.target_object_id]})"
                f" exceeded the max priority[{self.max_priority}]"
            )
            return None

        if self.obj_id_to_priority[self.env.target_recep_id] > self.max_priority:
            get_logger().debug(
                f"Priority for object[{self.env.target_recep_id}] ({self.obj_id_to_priority[self.env.target_recep_id]})"
                f" exceeded the max priority[{self.max_priority}]"
            )
            return None

        if self.env.current_room != self.env.target_object_room_id:
            get_logger().debug(
                f"Move to the target object room {self.env.target_object_room_id}"
            )
            return self.move_to_target_object_room()

        self._last_to_interact_object_pose = target_object
        if self.exploration_enabled:
            target_object_receptacle = None
            for recep, objs in self.unvisited_object_ids_per_recep.items():
                if self.env.target_object_id in objs:
                    target_object_receptacle = recep
                    break

            expert_nav_action = self._expert_nav_action_to_obj(
                obj=target_object,
                recep_target_keys=self.cached_locs_for_recep[
                    target_object_receptacle
                ] if target_object_receptacle is not None else None,
            )

        else:
            expert_nav_action = self._expert_nav_action_to_obj(obj=target_object)

        if expert_nav_action is None:
            interactable_positions = self.env._interactable_positions_cache.get(
                scene_name=self.env.scene,
                obj=target_object,
                controller=self.env
            )
            if len(interactable_positions) != 0:
                get_logger().debug(
                    f"Could not find a path to {target_object['objectId']}"
                    f" in scene {self.task.env.scene}"
                    f" when at position {self.task.env.get_agent_location()}."
                )
            else:
                get_logger().debug(
                    f"Object {target_object['objectId']} in scene {self.task.env.scene}"
                    f" has no interactable positions."
                )
            self.obj_id_to_priority[target_object["objectId"]] += 1
            return self.home_service_pickup(mode=mode)
        
        if expert_nav_action == "Pass":
            with include_object_data(self.env.controller):
                visible_objects = {
                    o["objectId"]
                    for o in self.env.last_event.metadata["objects"]
                    if o["visible"]
                }

            if target_object["objectId"] not in visible_objects:
                if self._invalidate_interactable_loc_for_pose(
                    location=self.env.get_agent_location(),
                    obj_pose=target_object,
                ):
                    return self.home_service_pickup(mode=mode)

                get_logger().debug(
                    "ERROR: This should not be possible. Failed to invalidate interactable loc for obj pose"
                )
                return None
            
            return dict(action="Pickup", objectId=target_object["objectId"],)

        return dict(action=expert_nav_action)

    '''
    ########################### Static Methods ###########################
    '''
    @staticmethod
    def angle_to_recep(agent_pos, agent_dir, recep_pos):
        agent_to_recep = np.array(recep_pos) - np.array(agent_pos)
        agent_to_recep_dir = agent_to_recep / (np.linalg.norm(agent_to_recep) + 1e-6)
        ang_dist = np.degrees(np.arccos(np.dot(agent_to_recep_dir, agent_dir)))

        if ang_dist > 45:
            if agent_dir[0] == 0:  # z or -z
                if agent_dir[1] == 1:  # z, so next is x
                    if recep_pos[0] < agent_pos[0]:
                        ang_dist = 360 - ang_dist
                else:  # agent_dir[1] == -1:  # -z, so next is -x
                    if recep_pos[0] > agent_pos[0]:
                        ang_dist = 360 - ang_dist
            else:  # agent_dir[1] == 0, so x or -x
                if agent_dir[0] == 1:  # x, so next is -z
                    if recep_pos[1] > agent_pos[1]:
                        ang_dist = 360 - ang_dist
                else:  # agent_dir[0] == -1:  # -x, so next is z
                    if recep_pos[1] < agent_pos[1]:
                        ang_dist = 360 - ang_dist
        return ang_dist

    @staticmethod
    def extract_xyz(full_poses: List[Dict[str, Union[float, int, bool]]]):
        known_xyz = set()
        res_xyz = []
        for pose in full_poses:
            xyz = tuple(pose[x] for x in "xyz")
            if xyz in known_xyz:
                continue
            known_xyz.add(xyz)
            res_xyz.append({x: pose[x] for x in "xyz"})
        
        return res_xyz

    @staticmethod
    def crouch_stand_if_needed(
        interactable_positions: List[Dict[str, Union[float, int, bool]]],
        agent_loc: Dict[str, Union[float, int, bool]],
        tol: float = 1e-2,
    ) -> Optional[str]:
        for gdl in sorted(
            interactable_positions,
            key=lambda ap: ap["standing"] != agent_loc["standing"],
        ):
            if (
                round(
                    abs(agent_loc["x"] - gdl["x"]) + abs(agent_loc["z"] - gdl["z"]), 2
                )
                <= tol
            ):
                if _are_agent_locations_equal(
                    agent_loc, gdl, ignore_standing=True, tol=tol
                ):
                    if agent_loc["standing"] != gdl["standing"]:
                        return "Crouch" if agent_loc["standing"] else "Stand"
                    else:
                        return "Pass"

        return None


class HomeServiceSubtaskActionExpert(HomeServiceGreedyActionExpert):
    def __init__(
        self,
        task: "HomeServiceTask",
        shortest_path_navigator: ShortestPathNavigator,
        max_priority: int = 3,
        steps_for_time_pressure: int = 250,
        exploration_enabled: bool = True,
        use_planner: bool = True,
        subtasks: List[str] = SUBTASKS,
        **kwargs,
    ):
        get_logger().debug(
            f"SubtaskExpert started for {task.env.scene}"
        )
        self.exploration_enabled = exploration_enabled
        self.task = task
        assert self.task.num_steps_taken() == 0

        self.shortest_path_navigator = shortest_path_navigator

        self._last_to_target_recep_id: Optional[str] = None
        self.scanned_receps = set()
        self._current_object_target_keys: Optional[Set[AgentLocKeyType]] = None
        self.recep_id_loc_per_room = dict()
        self.cached_locs_for_recep = dict()
        self.scanned_objects = set()
        self.object_id_loc_per_recep_id = dict()
        self.cached_locs_for_objects = dict()

        self.max_priority = max_priority
        self.obj_id_to_priority: defaultdict = defaultdict(lambda: 0)
        self.visited_recep_ids_per_room = {
            room: set() for room in self.env.room_to_poly
        }
        self.unvisited_recep_ids_per_room = self.env.room_to_static_receptacle_ids()
        self.visited_object_ids_per_recep = {
            recep: set()
            for room in self.env.room_to_poly
            for recep in self.unvisited_recep_ids_per_room[room]
        }
        self.unvisited_object_ids_per_recep = {
            recep: set(obj["receptacleObjectIds"])
            for recep in self.visited_object_ids_per_recep
            for obj in self.env.objects()
            if obj["objectId"] == recep
        }

        self.steps_for_time_pressure = steps_for_time_pressure

        self.last_expert_mode: Optional[str] = None

        self.expert_action_list: List[int] = []

        self._last_held_object_id: Optional[str] = None
        self._last_to_interact_object_pose: Optional[Dict[str, Any]] = None
        self._id_of_object_we_wanted_to_pickup: Optional[str] = None

        self.subtasks_names = subtasks
        
        self.planner = None
        if use_planner:
            self.planner = HomeServiceSimpleSubtaskPlanner(
                task=task,
            )
            self.planner.next_subtask()
            self.expert_subtask_list: List[int] = []
            
        self.shortest_path_navigator.on_reset()
        self.update(
            action_taken=None,
            action_success=None,
        )

    @property
    def expert_subtask(self) -> int:
        return self.expert_subtask_list[-1]

    @property
    def current_subtask(self) -> Optional[int]:
        if self.planner is None:
            return None
        return self.planner.current_subtask

    @property
    def current_subtask_str(self) -> Optional[str]:
        if self.planner is None:
            return None
        return self.planner.subtask_str(self.current_subtask)

    @property
    def visible_objects(self) -> Sequence[str]:
        with include_object_data(self.env.controller):
            visible_objects = {
                o["objectId"]
                for o in self.env.last_event.metadata["objects"]
                if o["visible"]
            }

        return visible_objects

    '''
    ########################### Methods ###########################
    '''
    def update_visible_objects(self):
        for obj in self.visible_objects:
            obj_recep = None
            for recep, objs in self.unvisited_object_ids_per_recep.items():
                if obj in objs:
                    obj_recep = recep
                    break
            if obj_recep is None:
                continue
            if obj in self.cached_locs_for_objects:
                if self.shortest_path_navigator.get_full_key(
                    self.env.get_agent_location()
                ) in self.cached_locs_for_objects[obj]:
                    if obj not in self.scanned_objects:
                        self.scanned_objects.add(obj)
                    if obj in self.unvisited_object_ids_per_recep[obj_recep]:
                        self.unvisited_object_ids_per_recep[obj_recep].remove(obj)
                    if obj not in self.visited_object_ids_per_recep[obj_recep]:
                        self.visited_object_ids_per_recep[obj_recep].add(obj)
                continue
            interactable_positions = self.env._interactable_positions_cache.get(
                scene_name=self.env.scene,
                obj=next(
                    o for o in self.env.objects()
                    if o['objectId'] == obj
                ),
                controller=self.env,
            )
            if self.shortest_path_navigator.get_full_key(
                self.env.get_agent_location()
            ) in [
                self.shortest_path_navigator.get_full_key(ip)
                for ip in interactable_positions
            ]:
                self.cached_locs_for_objects[obj] = set(
                    [
                        self.shortest_path_navigator.get_full_key(loc)
                        for loc in interactable_positions
                    ]
                )
                self.unvisited_object_ids_per_recep[obj_recep].remove(obj)
                self.scanned_objects.add(obj)
                self.visited_object_ids_per_recep[obj_recep].add(obj)

    def update(
        self,
        action_taken: Optional[int],
        action_success: Optional[bool],
    ):
        self.update_visited_receps()
        self.update_visible_objects()
        if action_taken is not None:
            assert action_success is not None
            get_logger().debug(
                f"STEP-{self.task.num_steps_taken()}: action[{self.task.action_names()[action_taken]}({action_taken})] action_success[{action_success}]"
            )

            action_names = self.task.action_names()
            last_expert_action = self.expert_action_list[-1]
            agent_took_expert_action = action_taken == last_expert_action
            action_str = action_names[action_taken]

            if self.planner is not None:
                self.planner.next_subtask()
                self.expert_subtask_list.append(self.planner.current_subtask)
                last_expert_subtask = self.expert_subtask_list[-1]
                get_logger().debug(
                    f"STEP-{self.task.num_steps_taken()}: planned current subtask[{self.subtasks_names[self.planner.current_subtask]}({self.planner.current_subtask})]"
                )

            was_nav_action = any(k in action_str for k in ["move", "rotate", "look"])

            if "pickup_" in action_str and action_success:
                if agent_took_expert_action:
                    self._id_of_object_we_wanted_to_pickup = self._last_to_interact_object_pose["objectId"]
                else:
                    get_logger().debug(
                        f"The agent picked up the wrong object [{self._last_to_interact_object_pose['objectId']}]..."
                        f" thus change the subtask as [GotoObject]"
                    )
                    self.planner.set_subtask("GotoObject")
                

            if ("put_" in action_str or "drop_held" in action_str) and agent_took_expert_action and action_success:
                if self._id_of_object_we_wanted_to_pickup is not None:
                    self.obj_id_to_priority[
                        self._id_of_object_we_wanted_to_pickup
                    ] += 1
                else:
                    self.obj_id_to_priority[
                        self._last_held_object_id
                    ] += 1

            if "open_by_type" in action_str and agent_took_expert_action:
                self._receptacle_object_to_open = self._last_to_interact_object_pose

            if not action_success:
                if was_nav_action:
                    self.shortest_path_navigator.update_graph_with_failed_action(
                        stringcase.pascalcase(action_str)
                    )
                elif (
                    "pickup_" in action_str 
                    or "open_by_type" in action_str 
                    or "close_by_type" in action_str
                    or "put_" in action_str
                ) and action_taken == last_expert_action:
                    assert self._last_to_interact_object_pose is not None
                    get_logger().debug(
                        f"Action {action_str} FAILED. "
                        f"Invalidate current location [{self.task.env.get_agent_location()}] "
                        f"for object pose [{self._last_to_interact_object_pose}]."
                    )
                    self._invalidate_interactable_loc_for_pose(
                        location=self.task.env.get_agent_location(),
                        obj_pose=self._last_to_interact_object_pose,
                    )
                elif (
                    ("crouch" in action_str or "stand" in action_str)
                    and self.task.env.held_object is not None
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
                if not was_nav_action and not (
                    "crouch" in action_str or "stand" in action_str
                ):
                    self._last_to_interact_object_pose = None
            
        held_object = self.task.env.held_object
        if self.task.env.held_object is not None:
            self._last_held_object_id = held_object["objectId"]

        self._generate_and_record_expert_action()

    def _generate_and_record_expert_action(self):
        expert_action_dict = self._generate_expert_action_dict()
        # self.expert_subtask_list.append(self.planner.current_subtask)
        if expert_action_dict is None:
            self.expert_action_list.append(None)
            return
        
        action_str = stringcase.snakecase(expert_action_dict["action"])
        if action_str not in self.task.action_names():
            current_objectId = expert_action_dict["objectId"]
            current_obj = next(
                o for o in self.task.env.objects() if o["objectId"] == current_objectId
            )
            obj_type = stringcase.snakecase(current_obj["objectType"])
            action_str = f"{action_str}_{obj_type}"
            
        try:
            self.expert_action_list.append(self.task.action_names().index(action_str))
        except ValueError:
            get_logger().error(
                f"{action_str} is not a valid action for the given task."
            )
            self.expert_action_list.append(None)

    def _generate_expert_action_dict(self, horizon=30) -> Optional[Dict[str, Any]]:
        if self.env.mode != HomeServiceMode.SNAP:
            raise NotImplementedError(
                f"Expert only defined for 'easy' mode (current mode: {self.env.mode}"
            )

        try:
            attempt = None
            k = 0
            while attempt is None and k < 5:
                attempt = self.stop_episode()
                if attempt is not None:
                    return self._log_output_mode("Stop episode", attempt)
                
                attempt = self.put_object()
                if attempt is not None:
                    return self._log_output_mode("Put held object", attempt)

                attempt = self.pickup_object()
                if attempt is not None:
                    return self._log_output_mode("Pickup object", attempt)
                
                attempt = self.close_receptacle()
                if attempt is not None:
                    return self._log_output_mode("Close receptacle", attempt)
                
                attempt = self.open_receptacle()
                if attempt is not None:
                    return self._log_output_mode("Open receptacle", attempt)
                
                attempt = self.goto_target()
                if attempt is not None:
                    return self._log_output_mode("Goto target", attempt)
                
                attempt = self.move_room()
                if attempt is not None:
                    return self._log_output_mode("Move room", attempt)
                
                attempt = self.explore()
                if attempt is not None:
                    return self._log_output_mode("Explore", attempt)

                k += 1

            get_logger().debug(
                f"trial number exceeded 5"
            )
            # import pdb; pdb.set_trace()
            return dict(action="Done")

        except:
            import traceback

            get_logger().debug(f"EXCEPTION: Expert failure: {traceback.format_exc()}")
            return None

    def stop_episode(self) -> Optional[Dict[str, Any]]:
        get_logger().debug(
            f"In `stop_episode` method"
        )
        if self.current_subtask_str != "Stop":
            get_logger().debug(
                f"Current subtask is not [Stop]. Returning None."
            )
            return None
        get_logger().debug(
            f"Current subtask is [Stop]. Returning dict(action='Done')."
        )
        return dict(action="Done")

    def put_object(self) -> Optional[Dict[str, Any]]:
        get_logger().debug(
            f"In `put_object` method"
        )
        if self.current_subtask_str != "Put":
            get_logger().debug(
                f"Current subtask is not [Put]. Returning None."
            )
            return None
        
        if self.env.held_object is None:
            get_logger().debug(
                f"Currently trying to [Put] but the agent is holding nothing."
                f" Returning None"
            )
            return None
        
        if self.env.held_object['objectId'] != self.env.target_object_id:
            get_logger().debug(
                f"Currently trying to [Put] and the agent is NOT holding the target object."
                f" Returning dict(action=DropHeldObejctWithSnap)"
            )
            return dict(action="DropHeldObjectWithSnap")
        
        self._last_to_interact_object_pose = next(
            o for o in self.env.objects() if o["objectId"] == self.env.target_recep_id
        )
        if self._last_to_interact_object_pose['objectId'] in self.visible_objects:
            recep_type = stringcase.pascalcase(self._last_to_interact_object_pose['objectType'])
            get_logger().debug(
                f"Currently trying to [Put] and the agent is holding the target object "
                f"and the target recep is visible."
                f" Returning dict(action=Put{recep_type})"
            )
            return dict(action=f"Put{recep_type}")
        
        get_logger().debug(
            f"Currently trying to [Put] and the agent is holding the target object "
            f"but the target recep is NOT visible."
            f" Returning dict(action=DropHeldObejctWithSnap)"
        )
        return dict("DropHeldObjectWithSnap")

    def pickup_object(self) -> Optional[Dict[str, Any]]:
        get_logger().debug(
            f"In `pickup_object` method"
        )
        if self.current_subtask_str != "Pickup":
            get_logger().debug(
                f"Current subtask is not [Pickup]. Returning None."
            )
            return None
        
        if self._last_to_interact_object_pose is None:
            get_logger().debug(
                f"Currently trying to [Pickup] but self._last_to_interact_object_pose is None"
                f" Returning None"
            )
            return None

        if self._last_to_interact_object_pose['objectId'] not in self.visible_objects:
            get_logger().debug(
                f"Currently trying to [Pickup] but {self._last_to_interact_object_pose['objectId']} is NOT visible."
                f" Returning None"
            )
            return None
        
        get_logger().debug(
            f"Currently trying to [Pickup] and {self._last_to_interact_object_pose['objectId']} is visible."
            f" Returning dict(action='Pickup', objectId={self._last_to_interact_object_pose['objectId']})"
        )
        return dict(action=f"Pickup", objectId=self._last_to_interact_object_pose['objectId'],)
            
    def close_receptacle(self) -> Optional[Dict[str, Any]]:
        get_logger().debug(
            f"In `close_receptacle` method"
        )
        if self.current_subtask_str != "Close":
            get_logger().debug(
                f"Current subtask is not [Close]. Returning None."
            )
            return None

        if self._last_to_target_recep_id is None:
            get_logger().debug(
                f"Currently trying to [Close] but self._last_to_target_recep_id is None"
                f" Returning None"
            )
            return None

        if self._last_to_target_recep_id not in self.visible_objects:
            get_logger().debug(
                f"Currently trying to [Close] but {self._last_to_target_recep_id} is NOT visible."
                f" Returning None"
            )
            return None

        get_logger().debug(
            f"Currently trying to [Close] and {self._last_to_target_recep_id} is visible."
            f" Returning dict(action='CloseByType', objectId={self._last_to_target_recep_id})"
        )
        return dict(action="CloseByType", objectId=self._last_to_target_recep_id,)

    
    def open_receptacle(self) -> Optional[Dict[str, Any]]:
        get_logger().debug(
            f"In `open_receptacle` method"
        )
        if self.current_subtask_str != "Open":
            get_logger().debug(
                f"Current subtask is not [Open]. Returning None."
            )
            return None
        
        if self._last_to_target_recep_id is None:
            get_logger().debug(
                f"Currently trying to [Open] but self._last_to_target_recep_id is None"
                f" Returning None"
            )
            return None

        if self._last_to_target_recep_id not in self.visible_objects:
            get_logger().debug(
                f"Currently trying to [Open] but {self._last_to_target_recep_id} is NOT visible."
                f" Returning None"
            )
            return None

        get_logger().debug(
            f"Currently trying to [Open] and {self._last_to_target_recep_id} is visible."
            f" Returning dict(action='OpenByType', objectId={self._last_to_target_recep_id}, openness=1.0)"
        )
        return dict(action="OpenByType", objectId=self._last_to_target_recep_id, openness=1.0)
    
    def goto_target(self) -> Optional[Dict[str, Any]]:
        get_logger().debug(
            f"In `goto_target` method"
        )
        if not self.current_subtask_str.startswith("Goto"):
            get_logger().debug(
                f"Current subtask does not start with [Goto]. Returning None."
            )
            return None

        target = self.current_subtask_str.replace("Goto", "")
        if target == "Object":
            if self.env.held_object is not None:
                get_logger().debug(
                    f"Currently trying to [{self.current_subtask_str}] but the agent is holding object. "
                    f"Returning None."
                )
                return None

            # self._last_to_interact_object_pose = self.env.object_id_to_target_pose[
            #     self.env.target_object_id
            # ]
            self._last_to_interact_object_pose = next(
                o for o in self.env.objects()
                if o['objectId'] == self.env.target_object_id
            )
            expert_nav_action = self._expert_nav_action_to_obj(
                obj=self._last_to_interact_object_pose
            )

            get_logger().debug(
                f"Currently trying to [{self.current_subtask_str}] with "
                f"self._last_to_interact_object_pose['objectId']: {self._last_to_interact_object_pose['objectId']}."
            )
            if expert_nav_action is None:
                get_logger().debug(
                    f"The agent failed to go to the target object."
                    f" Returning None."
                )
                return None
            elif expert_nav_action == "Pass":
                get_logger().debug(
                    f"The agent arrived to the target object."
                    f" Returning None."
                )
                return None
            else:
                get_logger().debug(
                    f"The agent has not arrived to the target object."
                    f" Returning dict(action='{expert_nav_action}')."
                )
                return dict(action=expert_nav_action)

        elif target == "Receptacle":
            self._last_to_target_recep_id = self.env.target_recep_id
            recep = None
            if self._last_to_target_recep_id not in self.env.object_id_to_target_pose:
                recep = next(
                    o for o in self.env.objects()
                    if o['objectId'] == self._last_to_target_recep_id
                )
            else:
                recep = self.env.object_id_to_target_pose[self._last_to_target_recep_id]

            # if recep is None:
            #     import pdb; pdb.set_trace()
            
            expert_nav_action = self._expert_nav_action_to_obj(
                obj=recep
            )

            get_logger().debug(
                f"Currently trying to [{self.current_subtask_str}] with "
                f"self._last_to_target_recep_id: {self._last_to_target_recep_id}."
            )
            if expert_nav_action is None:
                get_logger().debug(
                    f"The agent failed to go to the target receptacle."
                    f" Returning None."
                )
                return None
            elif expert_nav_action == "Pass":
                get_logger().debug(
                    f"The agent arrived to the target receptacle."
                    f" Returning None."
                )
                return None
            else:
                get_logger().debug(
                    f"The agent has not arrived to the target receptacle."
                    f" Returning dict(action='{expert_nav_action}')."
                )
                return dict(action=expert_nav_action)

        else:
            get_logger().debug(
                f"target: {target}"
            )
            raise RuntimeError("Invalid subtask...")

    def move_room(self) -> Optional[Dict[str, Any]]:
        get_logger().debug(
            f"In `move_room` method"
        )
        if not self.current_subtask_str.startswith("MoveTo"):
            get_logger().debug(
                f"Current subtask does not start with [MoveTo]. Returning None."
            )
            return None
        target_room_type = self.current_subtask_str.replace("MoveTo", "")
        target_room_id = self.env.get_room_id_by_type(target_room_type)
        return self.move_to_target_room(target_room=target_room_id,)
    
    def explore(self, standing: bool = True, horizon: int = 30) -> Optional[Dict[str, Any]]:
        get_logger().debug(
            f"In `explore` method"
        )
        if self.current_subtask_str != "Explore":
            get_logger().debug(
                f"Current subtask is not [Explore]. Returning None."
            )
            return None
        
        found_object = self.env.target_object_id in self.scanned_objects
        found_receptacle = self.env.target_recep_id in self.scanned_receps

        if not found_receptacle:
            if self.env.current_room != self.env.target_recep_room_id:
                get_logger().debug(
                    f"The agent should explore to find target receptacle "
                    f"but the current room[{self.env.current_room}] is not the target receptacle room[{self.env.target_recep_room_id}]."
                    f" Returning None."
                )
                return None

            pose_act = self.exploration_pose(horizon=horizon)
            self._last_to_interact_object_pose = next(
                o for o in self.env.objects() if o['objectId'] == self.env.target_recep_id
            )

            assert self.env.target_recep_id in self.cached_locs_for_recep, f"It should be updated!"
            expert_nav_action = self._expert_nav_action_to_obj(
                self._last_to_interact_object_pose,
                force_standing=standing,
                force_horizon=horizon,
                recep_target_keys=self.cached_locs_for_recep[self.env.target_recep_id]
            )

            if expert_nav_action is None:
                if pose_act is not None:
                    get_logger().debug(
                        f"Change to the pose with pose_action[{pose_act}]"
                    )
                    return pose_act
                get_logger().debug(
                    f"Failed to navigate to {self.env.target_recep_id} in {self.env.current_room}"
                    f" Returning None."
                )
                return None
            
            if expert_nav_action != "Pass":
                return self.pose_action_if_compatible(
                    nav_action=dict(action=expert_nav_action),
                    pose_action=pose_act,
                    current_recep=self._last_to_interact_object_pose,
                    standing=standing,
                    horizon=horizon,
                )
            
            if len(self._current_object_target_keys) > 0:
                self.scanned_receps.add(self.env.target_recep_id)
                self.unvisited_recep_ids_per_room[self.env.current_room].remove(
                    self.env.target_recep_id
                )
                self.visited_recep_ids_per_room[self.env.current_room].add(self.env.target_recep_id)
                get_logger().debug(
                    f"Agent Found the Target Recep 9' 3')9"
                )
            
            return None

        elif not found_object:
            if self.env.current_room != self.env.target_object_room_id:
                get_logger().debug(
                    f"The agent should explore to find target object "
                    f"but the current room[{self.env.current_room}] is not the target object room[{self.env.target_object_room_id}]."
                    f" Returning None."
                )
                return None

            self._last_to_interact_object_pose = next(
                o for o in self.env.objects() if o['objectId'] == self.env.target_object_id
            )
            target_object_receptacle = None
            for recep, objs in self.unvisited_object_ids_per_recep.items():
                if self.env.target_object_id in objs:
                    target_object_receptacle = recep
                    break
            
            expert_nav_action = self._expert_nav_action_to_obj(
                self._last_to_interact_object_pose,
                force_standing=standing,
                force_horizon=horizon,
            )
            
            if expert_nav_action is None:
                get_logger().debug(
                    f"Failed to navigate to {self.env.target_object_id} in {self.env.current_room}."
                    f" Re-trying navigation."
                )
                expert_nav_action = self._expert_nav_action_to_obj(
                    self._last_to_interact_object_pose,
                )
                if expert_nav_action is None:
                    interactable_positions = self.env._interactable_positions_cache.get(
                        scene_name=self.env.scene,
                        obj=self._last_to_interact_object_pose,
                        controller=self.env,
                    )
                    if len(interactable_positions) != 0:
                        get_logger().debug(
                            f"Could not find a path to {self._last_to_interact_object_pose['objectId']}"
                            f" in scene {self.task.env.scene}"
                            f" when at position {self.task.env.get_agent_location()}."
                        )
                    else:
                        get_logger().debug(
                            f"Object {self._last_to_interact_object_pose['objectId']} in scene {self.task.env.scene}"
                            f" has no interactable positions."
                        )
                    get_logger().debug(
                        f"Returning None."
                    )
                    return None
            
            if expert_nav_action == "Pass":
                if self.env.target_object_id not in self.visible_objects:
                    get_logger().debug(
                        f"target object {self.env.target_object_id} is not visible."
                        f" Try to invalidate the interactable loc {self.env.get_agent_location()}"
                        f" for obj_pose [{self._last_to_interact_object_pose}]"
                    )
                    if self._invalidate_interactable_loc_for_pose(
                        location=self.env.get_agent_location(),
                        obj_pose=self._last_to_interact_object_pose,
                    ):
                        get_logger().debug(
                            f"Invalidated the interactable loc {self.env.get_agent_location()}"
                            f" for obj_pose [{self._last_to_interact_object_pose}]"
                        )
                        return self.explore(standing=standing, horizon=horizon)
                    
                    get_logger().debug(
                        f"Failed to invalidate the loc."
                    )
                    
                    if (
                        target_object_receptacle is not None
                        and self.shortest_path_navigator.get_full_key(
                            self.env.get_agent_location()
                        ) in self.cached_locs_for_recep[target_object_receptacle]
                    ):
                        get_logger().debug(
                            f"Remove the current location from the cached locs for recep."
                        )
                        self.cached_locs_for_recep[target_object_receptacle].remove(
                            self.shortest_path_navigator.get_full_key(self.env.get_agent_location())
                        )
                        get_logger().debug(
                            f"Explore the target object with removed cached locs."
                        )
                        return self.explore(standing=standing, horizon=horizon)
                    
                    interactable_positions = self.env._interactable_positions_cache.get(
                        scene_name=self.env.scene,
                        obj=self._last_to_interact_object_pose,
                        controller=self.env,
                    )
                    for ip in interactable_positions:
                        if (
                            ip["x"] == self.env.get_agent_location()["x"]
                            and ip["z"] == self.env.get_agent_location()["z"]
                            and ip["rotation"] == self.env.get_agent_location()["rotation"]
                        ):
                            get_logger().debug(
                                f"Change the standing and horizon constraints at current location"
                                f" then find out the target object [{ip}]"
                            )
                            return self.explore(standing=ip["standing"], horizon=ip["horizon"])
                    
                    get_logger().debug(
                        f"No interactable positions available at the current location... IMPOSSIBLE?!"
                        f" Returning None."
                    )
                    return None
                
                get_logger().debug(
                    f"Agent found the target object {self.env.target_object_id} 9' 3')9"
                    f" and arrived at the interactable position."
                )
                self.unvisited_object_ids_per_recep[target_object_receptacle].remove(self.env.target_object_id)
                self.scanned_objects.add(self.env.target_object_id)
                self.visited_object_ids_per_recep[target_object_receptacle].add(self.env.target_object_id)

                get_logger().debug(
                    f"Returning None."
                )
                return None
            
            get_logger().debug(
                f"Returning expert_nav_action: dict(action={expert_nav_action})."
            )
            return dict(action=expert_nav_action)

        else:
            get_logger().debug(
                f"Target object and target receptacle are both found already."
                f" No need to explore anymore... Returning None."
            )
            return None