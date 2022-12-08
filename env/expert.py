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


class GreedySimpleHomeServiceExpert:
    """An agent which greedily attempts to complete a given unshuffle task."""

    def __init__(
        self,
        task: "HomeServiceTask",
        shortest_path_navigator: ShortestPathNavigator,
        max_priority_per_object: int = 3,
        max_priority_per_receptacle: int = 3,
        steps_for_time_pressure: int = 200,
        exploration_enabled: bool = True,
        scan_before_move: bool = True,
    ):
        get_logger().debug(
            f"Expert started for {task.env.scene} (exploration: {exploration_enabled})"
        )
        self.exploration_enabled = exploration_enabled
        self.scan_before_move = scan_before_move

        self.task = task
        assert self.task.num_steps_taken() == 0

        self.shortest_path_navigator = shortest_path_navigator
        self.max_priority_per_object = max_priority_per_object

        self._last_to_target_recep_id: Optional[str] = None
        self.scanned_receps = set()
        self._current_object_target_keys: Optional[Set[AgentLocKeyType]] = None
        self.recep_id_loc_per_room = dict()
        self.cached_locs_for_recep = dict()

        self.max_priority_per_receptacle = max_priority_per_receptacle
        self.recep_id_to_priority: defaultdict = defaultdict(lambda: 0)
        self.visited_recep_ids_per_room = {
            room: set() for room in self.env.room_to_poly
        }
        self.unvisited_recep_ids_per_room = self.env.room_to_static_receptacle_ids()

        self.steps_for_time_pressure = steps_for_time_pressure

        self.last_expert_mode: Optional[str] = None

        self.expert_action_list: List[int] = []

        self._last_held_object_id: Optional[str] = None
        self._last_to_interact_object_pose: Optional[Dict[str, Any]] = None
        self._id_of_object_we_wanted_to_pickup: Optional[str] = None
        self.object_id_to_priority: defaultdict = defaultdict(lambda: 0)

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
        assert self.task.num_steps_taken() == len(self.expert_action_list) - 1
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
                    source_state_key=source_state_key, target_key=target_key,
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
            self.env,
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

    def get_unscanned_receps(self, rooms_to_check, standing=True, horizon=30):
        agent_key = self.shortest_path_navigator.get_key(self.env.get_agent_location())
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

    def _log_output_mode(self, expert_mode: str, action_dict: Dict[str, Any]) -> Dict[str, Any]:
        if self.last_expert_mode != expert_mode:
            get_logger().debug(f"{expert_mode} mode")
            self.last_expert_mode = expert_mode
        return action_dict

    def prioritize_receps(self):
        agent_loc = self.env.get_agent_location()

        assert self.env.current_room is not None
        recep_ids = self.unvisited_recep_ids_per_room[self.env.current_room]

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

    def manage_held_object(self) -> Optional[Dict[str, Any]]:
        if self.env.held_object is None:
            return None

        self._last_to_interact_object_pose = None

        # Should navigate to a position where the held object can be placed
        expert_nav_action = self._expert_nav_action_to_obj(
            obj={
                **self.env.held_object,
                **{
                    k: self.env.object_id_to_target_pose[
                        self.env.held_object["objectId"]
                    ][k]
                    for k in ["position", "rotation"]
                },
            },
        )

        if expert_nav_action is None:
            # Could not find a path to the target,
            return dict(action="DropHeldObjectWithSnap")
        elif expert_nav_action == "Pass":
            # We're in a position where we can put the object
            parent_ids = self.env.object_id_to_target_pose[self.env.held_object["objectId"]]["parentReceptacles"]
            if parent_ids is not None and len(parent_ids) == 1:
                parent_type = stringcase.pascalcase(
                    self.env.object_id_to_target_pose[parent_ids[0]]["objectType"]
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

            action_names = self.task.action_names()
            last_expert_action = self.expert_action_list[-1]
            agent_took_expert_action = action_taken == last_expert_action
            action_str = action_names[action_taken]

            was_nav_action = any(k in action_str for k in ["move", "rotate", "look"])

            if "pickup_" in action_str and agent_took_expert_action and action_success:
                self._id_of_object_we_wanted_to_pickup = self._last_to_interact_object_pose["objectId"]

            if "put_" in action_str and agent_took_expert_action and action_success:
                if self._id_of_object_we_wanted_to_pickup is not None:
                    self.object_id_to_priority[
                        self._id_of_object_we_wanted_to_pickup
                    ] += 1
                else:
                    self.object_id_to_priority[
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
                    or "put_by_type" in action_str
                ) and action_taken == last_expert_action:
                    assert self._last_to_interact_object_pose is not None
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
                        obj_pose=self.task.env.object_id_to_target_pose[held_object_id],
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
        if self.task.num_steps_taken() == len(self.expert_action_list) + 1:
            get_logger().warning(
                f"Already generated the expert action at step {self.task.num_steps_taken()}"
            )
            return

        assert self.task.num_steps_taken() == len(
            self.expert_action_list
        ), f"{self.task.num_steps_taken()} != {len(self.expert_action_list)}"
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
                # ppap
                if not self.time_pressure():
                    attempt = self.home_service_pickup(mode="")
                    if attempt is not None:
                        return self._log_output_mode("", attempt)
                    
                    # 
            attempt = self.home_service_pickup(mode="")
            if attempt is not None:
                return self._log_output_mode("", attempt)

            if self.exploration_enabled:
                if self.env.target_room_id != self.env.current_house:
                    get_logger().debug(
                        f"WARNING: We cannot generate more actions despite being in {self.env.current_house},"
                        f" away from the target {self.env.target_room_id}. Terminating"
                    )
            
            return dict(action="Done")

        except:
            import traceback

            get_logger().debug(f"EXCEPTION: Expert failure: {traceback.format_exc()}")
            return None

    def scan_for_target_recep(self, standing=True, horizon=30):
        env = self.env

        if self.env.current_room is None:
            return None
        
        self._last_to_target_recep_id = next(
            o["objectId"] for o in env.objects()
            if o["object"]
        )
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
    def _try_to_interact_with_objs_on_recep(
        env: HomeServiceEnvironment,
        interactable_positions: List[Dict[str, Union[float, int, bool]]],
        objs_on_recep: Set[str],
        horizon: int,
        standing: bool,
    ) -> List[Dict[str, Union[float, int, bool]]]:
        # Try to (greedily) find an interactable positions for all/most objects on objs_on_recep
        last_interactable = interactable_positions
        missing_objs_on_recep = copy.copy(objs_on_recep)
        for obj_id in missing_objs_on_recep:
            new_interactable = env._interactable_positions_cache.get(
                scene_name=env.scene,
                obj=next(o for o in env.objects() if o["objectId"] == obj_id),
                controller=env,
                reachable_positions=GreedySimpleHomeServiceExpert.extract_xyz(
                    last_interactable
                ),
                force_horizon=horizon,
                force_standing=standing,
                avoid_teleport=True,
            )
            if len(new_interactable) > 0:
                objs_on_recep.remove(obj_id)
                last_interactable = new_interactable

        return last_interactable

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
                        return "Crouch" if agent_loc["stadning"] else "Stand"
                    else:
                        return "Pass"

        return None


'''
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
            if self.task.current_subtask[0] == "Done":
                return self._generate_and_record_expert_action()
            # if self.task.current_subtask[0] == "Pickup":
            #     if self.task.env.scene == self.task.env.current_task_spec.target_scene:
            #         with include_object_data(self.task.env.controller):
            #             md = self.task.env.last_event.metadata
            #             cur_subtask_target = next(
            #                 (o for o in md["objects"] if o["objectType"] == self.task.current_subtask[1]), None
            #             )

            #             print(f'cur_subtask_target in AFTER navigate success in update()')
            #             print(f'visible: {cur_subtask_target["visible"]} | distance: {cur_subtask_target["distance"]}')
            # elif self.task.current_subtask[0] == "Put":
            #     if self.task.env.scene == self.task.env.current_task_spec.target_scene:
            #         with include_object_data(self.task.env.controller):
            #             md = self.task.env.last_event.metadata
            #             cur_subtask_place = next(
            #                 (o for o in md["objects"] if o["objectType"] == self.task.current_subtask[2]), None
            #             )

            #             print(f'cur_subtask_place in AFTER navigate success in update()')
            #             print(f'visible: {cur_subtask_place["visible"]} | distance: {cur_subtask_place["distance"]}')

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
                    # After invalidate current agent location, re-navigate to the object
                    if self.task.current_subtask[0] != "Navigate":
                        while self.task.current_subtask[0] != "Navigate":
                            self.task.rollback_subtask()

                elif (
                    action_str == "put"
                ) and action_taken == last_expert_action:
                    if self.task.env.scene == self.task.env.current_task_spec.target_scene:
                        assert self._last_to_interact_object_pose is not None
                    self._invalidate_interactable_loc_for_pose(
                        location=self.task.env.get_agent_location(),
                        obj_pose=self._last_to_interact_object_pose,
                    )
                    # After invalidate current agent location, re-navigate to the object
                    if self.task.current_subtask[0] != "Navigate":
                        # print("  > rollback subtask to navigate")
                        while self.task.current_subtask[0] != "Navigate":
                            self.task.rollback_subtask()
                    
                    # print(f'updated subtask {self.task.current_subtask}')

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
                        # if (
                        #     agent_took_expert_action or 
                        #     self.task.planned_task[self.task._subtask_step - 1][0] == "Pickup"
                        # ):
                        if agent_took_expert_action:
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
                        elif self.task.current_subtask[0] != "Goto":
                            # unintended pickup action succeeded
                            while self.task.current_subtask[0] != "Goto":
                                self.task.rollback_subtask()

                    elif action_str == "put":
                        # if (
                        #     agent_took_expert_action or 
                        #     self.task.planned_task[self.task._subtask_step - 1][0] == "Put"
                        # ):
                        if agent_took_expert_action:
                            assert self.task.env.held_object is None
                            self._last_to_interact_object_pose = None
                        elif self.task.current_subtask[0] != "Goto":
                            while self.task.current_subtask[0] != "Goto":
                                self.task.rollback_subtask()
                    
                    elif was_goto_action:
                        if self.task.current_subtask[0] != "Goto":
                            # If current subtask is not GOTO, rollback subtasks to GOTO
                            while self.task.current_subtask[0] != "Goto":
                                self.task.rollback_subtask()
                        self.require_check_room_type = True
                        self.goto_action_list = ["RotateRight" for _ in range(4)]

                    elif "crouch" in action_str or "stand" in action_str:
                        # took crouch/stand action in pickup/put subtask
                        # should navigate to target again
                        if self.task.current_subtask[0] in ("Pickup", "Put"):
                            while self.task.current_subtask[0] != "Navigate":
                                self.task.rollback_subtask()
                else:
                    if self.task.current_subtask[0] in ("Pickup", "Put"):
                        # took nav action in pickup/put subtask
                        # should navigate to target again
                        while self.task.current_subtask[0] != "Navigate":
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
                        # print("????")
                        # print(f'agent_loc: {agent_loc} | gdl: {gdl}')
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
            elif action_str == "fail":
                # print("fail?")
                action_str = "pass"

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
            
            if target_obj is None:
                print('target_obj is None!!')
                print(f'current scene: {self.task.env.scene} | target scene: {self.task.env.current_task_spec.target_scene}')
                print(f"target_obj: {target_obj} \n action_taken: {self.task.actions_taken} \n action_taken_success: {self.task.actions_taken_success}\n subtasks: {[subtask[0] for subtask in self.task.subtask_info]} \n current_subtask: {self.task.current_subtask}")
                print(f"planned_tasks: {self.task.planned_task} | num_subtasks: {self.task.num_subtasks} | subtask_step: {self.task._subtask_step}")
                print(f"unique_id: {self.task.env.current_task_spec.unique_id}")
                print(f'rewards: {self.task.rewards}')
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
                # print('aaaa')
                # print(f'target_obj: {target_obj["objectType"]} | visible: {target_obj["visible"]} | distance: {target_obj["distance"]}')
                if not target_obj["visible"] or target_obj["distance"] > VISIBILITY_DISTANCE:
                    if self._invalidate_interactable_loc_for_pose(
                        location=agent_loc, obj_pose=target_obj
                    ):
                        return self._generate_expert_action_dict()
                    # import pdb; pdb.set_trace()
                    # raise RuntimeError(" IMPOSSIBLE. ")
                    
                    else:
                        # print(f'invalidate failed')
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
            if target_obj is None:
                print('target_obj is None!!')
                print(f"target_obj: {target_obj} \n action_taken: {self.task.actions_taken} \n action_taken_success: {self.task.actions_taken_success}\n subtasks: {[subtask[0] for subtask in self.task.subtask_info]} \n current_subtask: {self.task.current_subtask}")
                print(f"planned_tasks: {self.task.planned_task} | num_subtasks: {self.task.num_subtasks} | subtask_step: {self.task._subtask_step}")
                print(f"unique_id: {self.task.env.current_task_spec.unique_id}")
                print(f'rewards: {self.task.rewards}')
            elif not target_obj['visible']:
                print('target_obj["visible"] is False!!!')
                print(f"target_obj: {target_obj} \n action_taken: {self.task.actions_taken} \n action_taken_success: {self.task.actions_taken_success}\n subtasks: {[subtask[0] for subtask in self.task.subtask_info]} \n current_subtask: {self.task.current_subtask}")
                print(f"planned_tasks: {self.task.planned_task} | num_subtasks: {self.task.num_subtasks} | subtask_step: {self.task._subtask_step}")
                print(f"unique_id: {self.task.env.current_task_spec.unique_id}")
                print(f'rewards: {self.task.rewards}')

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
                        (
                            o for o in current_objects 
                            if (
                                o['objectType'] == subtask_place
                                and o['visible']
                            )
                        ), 
                        None
                    )
            # if held_object is None:
            #     # Pickup has failed....
            #     for _ in range(3):
            #         self.task.rollback_subtask()
            #     return dict(action="Pass")
            
            if target_obj is None or held_object is None:
                if target_obj is None:
                    print('target_obj is None!!')
                if held_object is None:
                    print('held_object is None!!')
                print(f"target_obj: {target_obj} | held_object: {held_object}\n action_taken: {self.task.actions_taken} \n action_taken_success: {self.task.actions_taken_success}\n subtasks: {[subtask[0] for subtask in self.task.subtask_info]} \n current_subtask: {self.task.current_subtask}")
                print(f"planned_tasks: {self.task.planned_task} | num_subtasks: {self.task.num_subtasks} | subtask_step: {self.task._subtask_step}")
                print(f"unique_id: {self.task.env.current_task_spec.unique_id}")
                print(f'rewards: {self.task.rewards}')

            elif held_object["objectId"] != target_obj["objectId"]:
                print(f"held_object_id: {held_object['objectId']} | target_obj_id: {target_obj['objectId']}")
                print(f"target_obj: {target_obj} | held_object: {held_object}\n action_taken: {self.task.actions_taken} \n action_taken_success: {self.task.actions_taken_success}\n subtasks: {[subtask[0] for subtask in self.task.subtask_info]} \n current_subtask: {self.task.current_subtask}")
                print(f"planned_tasks: {self.task.planned_task} | num_subtasks: {self.task.num_subtasks} | subtask_step: {self.task._subtask_step}")
                print(f"unique_id: {self.task.env.current_task_spec.unique_id}")
                print(f'rewards: {self.task.rewards}')

            assert target_obj is not None and held_object["objectId"] == target_obj["objectId"]

            if place_obj is None:
                print('place_obj is None!!')
                print(f"place_obj: {place_obj}\n action_taken: {self.task.actions_taken} \n action_taken_success: {self.task.actions_taken_success}\n subtasks: {[subtask[0] for subtask in self.task.subtask_info]}")
                print(f"planned_tasks: {self.task.planned_task} | num_subtasks: {self.task.num_subtasks} | subtask_step: {self.task._subtask_step}")
                print(f"unique_id: {self.task.env.current_task_spec.unique_id}")
                print(f'rewards: {self.task.rewards}')
            elif not place_obj['visible']:
                print('place_obj["visible"] is False!!!!')
                print(f"place_obj: {place_obj}\n action_taken: {self.task.actions_taken} \n action_taken_success: {self.task.actions_taken_success}\n subtasks: {[subtask[0] for subtask in self.task.subtask_info]}")
                print(f"planned_tasks: {self.task.planned_task} | num_subtasks: {self.task.num_subtasks} | subtask_step: {self.task._subtask_step}")
                print(f"unique_id: {self.task.env.current_task_spec.unique_id}")
                print(f'rewards: {self.task.rewards}')

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
'''
