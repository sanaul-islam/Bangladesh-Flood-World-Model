from __future__ import annotations

import heapq
import math
import sqlite3
from pathlib import Path
from typing import Any

from pyproj import Transformer


DEFAULT_MAX_SEARCH_DISTANCE_KM = 30.0
DEFAULT_MAX_EXPANDED_NODES = 120000

SQL_CHUNK_SIZE = 500

FLOOD_WEIGHT = 2.0
BRIDGE_WEIGHT = 2.0
UNCERTAINTY_WEIGHT = 1.0


class RouteNotFoundError(RuntimeError):
    pass


class SQLiteRoadRouter:

    def __init__(
        self,
        database_path: str | Path,
    ) -> None:

        self.database_path = Path(
            database_path
        )

        if not self.database_path.exists():
            raise FileNotFoundError(
                f"Road database not found: {self.database_path}"
            )

        self.connection = sqlite3.connect(
            self.database_path,
            check_same_thread=False,
        )

        self.connection.execute(
            "PRAGMA query_only = ON"
        )

        self.connection.execute(
            "PRAGMA cache_size = -65536"
        )

        self.to_projected = Transformer.from_crs(
            "EPSG:4326",
            "EPSG:32646",
            always_xy=True,
        )

        self.to_wgs84 = Transformer.from_crs(
            "EPSG:32646",
            "EPSG:4326",
            always_xy=True,
        )

    def close(
        self,
    ) -> None:

        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def __enter__(
        self,
    ) -> "SQLiteRoadRouter":
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:

        self.close()

    def latlon_to_xy(
        self,
        latitude: float,
        longitude: float,
    ) -> tuple[float, float]:

        x, y = self.to_projected.transform(
            longitude,
            latitude,
        )

        return (
            float(x),
            float(y),
        )

    def xy_to_latlon(
        self,
        x: float,
        y: float,
    ) -> tuple[float, float]:

        longitude, latitude = (
            self.to_wgs84.transform(
                x,
                y,
            )
        )

        return (
            float(latitude),
            float(longitude),
        )

    def nearest_node(
        self,
        latitude: float,
        longitude: float,
        max_distance_m: float = 5000.0,
    ) -> tuple[int, float, float, float]:

        x, y = self.latlon_to_xy(
            latitude,
            longitude,
        )

        radius = 250.0

        while radius <= max_distance_m:

            min_x = x - radius
            max_x = x + radius
            min_y = y - radius
            max_y = y + radius

            row = self.connection.execute(
                """
                SELECT
                    node_id,
                    x,
                    y
                FROM nodes
                WHERE
                    x BETWEEN ? AND ?
                    AND y BETWEEN ? AND ?
                ORDER BY
                    (
                        (x - ?) * (x - ?)
                        +
                        (y - ?) * (y - ?)
                    )
                LIMIT 1
                """,
                (
                    min_x,
                    max_x,
                    min_y,
                    max_y,
                    x,
                    x,
                    y,
                    y,
                ),
            ).fetchone()

            if row is not None:

                node_id = int(
                    row[0]
                )

                node_x = float(
                    row[1]
                )

                node_y = float(
                    row[2]
                )

                distance = math.hypot(
                    node_x - x,
                    node_y - y,
                )

                if distance <= max_distance_m:

                    return (
                        node_id,
                        node_x,
                        node_y,
                        distance,
                    )

            radius *= 2.0

        raise RouteNotFoundError(
            "No road node found near requested coordinates."
        )

    def get_node(
        self,
        node_id: int,
    ) -> tuple[float, float]:

        row = self.connection.execute(
            """
            SELECT
                x,
                y
            FROM nodes
            WHERE node_id = ?
            """,
            (
                int(node_id),
            ),
        ).fetchone()

        if row is None:
            raise RouteNotFoundError(
                f"Road node {node_id} does not exist."
            )

        return (
            float(row[0]),
            float(row[1]),
        )

    def neighbors(
        self,
        node_id: int,
    ) -> list[tuple]:

        return self.connection.execute(
            """
            SELECT
                e.edge_id,
                e.u,
                e.v,
                e.length_m,
                e.travel_time_s,
                nu.x,
                nu.y,
                nv.x,
                nv.y,
                COALESCE(
                    r.flood_risk,
                    0.0
                ),
                COALESCE(
                    r.uncertainty_risk,
                    0.0
                ),
                COALESCE(
                    r.bridge_exposure,
                    0.0
                ),
                COALESCE(
                    r.bridge_risk,
                    0.0
                ),
                COALESCE(
                    r.total_risk,
                    0.0
                ),
                COALESCE(
                    r.risk_cost,
                    e.travel_time_s
                )
            FROM edges AS e
            JOIN nodes AS nu
                ON nu.node_id = e.u
            JOIN nodes AS nv
                ON nv.node_id = e.v
            LEFT JOIN road_edge_state AS r
                ON r.edge_id = e.edge_id
            WHERE
                e.u = ?
                OR e.v = ?
            """,
            (
                int(node_id),
                int(node_id),
            ),
        ).fetchall()

    def astar(
        self,
        start_node: int,
        goal_node: int,
        max_distance_km: float = DEFAULT_MAX_SEARCH_DISTANCE_KM,
        max_expanded_nodes: int = DEFAULT_MAX_EXPANDED_NODES,
    ) -> dict:
        """
        Risk-aware A* using a cost function expressed in seconds.

        Every edge cost is based on physical travel time multiplied by a
        dimensionless flood/bridge/uncertainty penalty. The heuristic is a
        lower bound on travel time derived from the fastest observed road
        speed in the database, so distance limits are enforced on accumulated
        road length rather than straight-line distance from the origin.
        """
        start_x, start_y = self.get_node(start_node)
        goal_x, goal_y = self.get_node(goal_node)

        max_distance_m = max(
            float(max_distance_km) * 1000.0,
            1.0,
        )

        # Cache the fastest observed road speed. This gives A* a consistent
        # time-based lower-bound heuristic without mixing metres and seconds.
        max_speed_mps = getattr(self, "_max_speed_mps", None)
        if max_speed_mps is None:
            row = self.connection.execute(
                """
                SELECT MAX(
                    CASE
                        WHEN travel_time_s > 0
                        THEN length_m / travel_time_s
                        ELSE 0.0
                    END
                )
                FROM edges
                """
            ).fetchone()

            max_speed_mps = (
                float(row[0])
                if row is not None and row[0] is not None
                else 13.8888888889
            )

            # Prevent a malformed/outlier database speed from making the
            # heuristic excessively aggressive.
            max_speed_mps = max(
                min(max_speed_mps, 55.5555555556),  # 200 km/h ceiling
                1.0,
            )
            self._max_speed_mps = max_speed_mps

        def heuristic(node_x: float, node_y: float) -> float:
            straight_line_m = math.hypot(
                goal_x - node_x,
                goal_y - node_y,
            )
            return straight_line_m / max_speed_mps

        queue = [
            (
                heuristic(start_x, start_y),
                0.0,
                int(start_node),
            )
        ]

        best_cost: dict[int, float] = {
            int(start_node): 0.0,
        }

        distance_from_start: dict[int, float] = {
            int(start_node): 0.0,
        }

        came_from: dict[int, int] = {}
        edge_for_node: dict[int, int] = {}

        expanded: set[int] = set()
        expanded_count = 0

        while queue:
            (
                _priority,
                current_cost,
                current_node,
            ) = heapq.heappop(queue)

            if current_node in expanded:
                continue

            expanded.add(current_node)
            expanded_count += 1

            if expanded_count > max_expanded_nodes:
                raise RouteNotFoundError(
                    "A* expansion limit reached."
                )

            if current_node == goal_node:
                break

            current_distance = distance_from_start[current_node]

            for row in self.neighbors(current_node):
                edge_id = int(row[0])
                u = int(row[1])
                v = int(row[2])

                if u == current_node:
                    neighbor = v
                    neighbor_x = float(row[7])
                    neighbor_y = float(row[8])
                else:
                    neighbor = u
                    neighbor_x = float(row[5])
                    neighbor_y = float(row[6])

                if neighbor in expanded:
                    continue

                length_m = max(float(row[3]), 0.0)
                base_travel_time_s = max(float(row[4]), 0.0)

                if base_travel_time_s <= 0.0:
                    # Defensive fallback for malformed edge records.
                    base_travel_time_s = length_m / 13.8888888889

                flood_risk = min(max(float(row[9]), 0.0), 1.0)
                uncertainty_risk = min(max(float(row[10]), 0.0), 1.0)
                bridge_risk = min(max(float(row[12]), 0.0), 1.0)

                # Keep routing physically grounded in travel time, then
                # increase that time by a dimensionless hazard penalty.
                risk_multiplier = (
                    1.0
                    + FLOOD_WEIGHT * flood_risk
                    + BRIDGE_WEIGHT * bridge_risk
                    + UNCERTAINTY_WEIGHT * uncertainty_risk
                )

                edge_cost = base_travel_time_s * risk_multiplier
                candidate_cost = current_cost + edge_cost

                candidate_distance = (
                    current_distance + length_m
                )

                # IMPORTANT: enforce the distance limit using accumulated
                # road distance, not Euclidean distance from the origin.
                if candidate_distance > max_distance_m:
                    continue

                previous = best_cost.get(neighbor)

                if (
                    previous is not None
                    and candidate_cost >= previous
                ):
                    continue

                best_cost[neighbor] = candidate_cost
                distance_from_start[neighbor] = candidate_distance
                came_from[neighbor] = current_node
                edge_for_node[neighbor] = edge_id

                h = heuristic(
                    neighbor_x,
                    neighbor_y,
                )

                heapq.heappush(
                    queue,
                    (
                        candidate_cost + h,
                        candidate_cost,
                        neighbor,
                    ),
                )

        if goal_node not in best_cost:
            raise RouteNotFoundError(
                "No risk-aware route found within the "
                "road-distance and search limits."
            )

        path = [int(goal_node)]
        current = int(goal_node)

        while current != int(start_node):
            if current not in came_from:
                raise RouteNotFoundError(
                    "Route reconstruction failed."
                )

            current = came_from[current]
            path.append(current)

        path.reverse()

        edge_ids = [
            int(edge_for_node[node])
            for node in path[1:]
        ]

        return {
            "path": path,
            "edge_ids": edge_ids,
            "risk_cost": float(best_cost[goal_node]),
            "road_distance_m": float(
                distance_from_start[goal_node]
            ),
            "expanded_nodes": int(expanded_count),
        }

    def _load_edge_statistics(
        self,
        edge_ids: list[int],
    ) -> dict[int, tuple]:

        result: dict[int, tuple] = {}

        for start in range(
            0,
            len(edge_ids),
            SQL_CHUNK_SIZE,
        ):

            chunk = edge_ids[
                start:start
                + SQL_CHUNK_SIZE
            ]

            placeholders = ",".join(
                "?"
                for _ in chunk
            )

            rows = self.connection.execute(
                f"""
                SELECT
                    e.edge_id,
                    e.length_m,
                    e.travel_time_s,
                    COALESCE(
                        r.flood_risk,
                        0.0
                    ),
                    COALESCE(
                        r.uncertainty_risk,
                        0.0
                    ),
                    COALESCE(
                        r.bridge_exposure,
                        0.0
                    ),
                    COALESCE(
                        r.bridge_risk,
                        0.0
                    ),
                    COALESCE(
                        r.total_risk,
                        0.0
                    )
                FROM edges AS e
                LEFT JOIN road_edge_state AS r
                    ON r.edge_id = e.edge_id
                WHERE e.edge_id IN (
                    {placeholders}
                )
                """,
                tuple(chunk),
            ).fetchall()

            for row in rows:
                result[
                    int(
                        row[0]
                    )
                ] = row

        return result

    def route(
        self,
        start_latitude: float,
        start_longitude: float,
        goal_latitude: float,
        goal_longitude: float,
        max_distance_km: float = DEFAULT_MAX_SEARCH_DISTANCE_KM,
        max_expanded_nodes: int = DEFAULT_MAX_EXPANDED_NODES,
    ) -> dict:

        (
            start_node,
            _,
            _,
            start_snap_distance,
        ) = self.nearest_node(
            start_latitude,
            start_longitude,
        )

        (
            goal_node,
            _,
            _,
            goal_snap_distance,
        ) = self.nearest_node(
            goal_latitude,
            goal_longitude,
        )

        result = self.astar(
            start_node=start_node,
            goal_node=goal_node,
            max_distance_km=max_distance_km,
            max_expanded_nodes=max_expanded_nodes,
        )

        edge_ids = result[
            "edge_ids"
        ]

        lookup = self._load_edge_statistics(
            edge_ids
        )

        total_length = 0.0
        total_time = 0.0

        flood_sum = 0.0
        uncertainty_sum = 0.0
        total_risk_sum = 0.0

        max_flood = 0.0
        max_uncertainty = 0.0
        max_bridge = 0.0

        bridge_edges = 0
        observed_edges = 0

        for edge_id in edge_ids:

            row = lookup.get(
                edge_id
            )

            if row is None:
                continue

            observed_edges += 1

            total_length += float(
                row[1]
            )

            total_time += float(
                row[2]
            )

            flood = float(
                row[3]
            )

            uncertainty = float(
                row[4]
            )

            bridge_exposure = float(
                row[5]
            )

            bridge_risk = float(
                row[6]
            )

            total_risk = float(
                row[7]
            )

            flood_sum += flood
            uncertainty_sum += uncertainty
            total_risk_sum += total_risk

            max_flood = max(
                max_flood,
                flood,
            )

            max_uncertainty = max(
                max_uncertainty,
                uncertainty,
            )

            max_bridge = max(
                max_bridge,
                bridge_risk,
            )

            if bridge_exposure > 0.5:
                bridge_edges += 1

        if observed_edges == 0:
            raise RouteNotFoundError(
                "Route contains no valid edge statistics."
            )

        mean_flood = (
            flood_sum
            / observed_edges
        )

        mean_uncertainty = (
            uncertainty_sum
            / observed_edges
        )

        mean_total_risk = (
            total_risk_sum
            / observed_edges
        )

        coordinates = []

        for node_id in result[
            "path"
        ]:

            x, y = self.get_node(
                node_id
            )

            latitude, longitude = (
                self.xy_to_latlon(
                    x,
                    y,
                )
            )

            coordinates.append(
                [
                    float(longitude),
                    float(latitude),
                ]
            )

        return {
            "start": {
                "latitude": float(
                    start_latitude
                ),
                "longitude": float(
                    start_longitude
                ),
                "nearest_node": int(
                    start_node
                ),
                "snap_distance_m": float(
                    start_snap_distance
                ),
            },
            "goal": {
                "latitude": float(
                    goal_latitude
                ),
                "longitude": float(
                    goal_longitude
                ),
                "nearest_node": int(
                    goal_node
                ),
                "snap_distance_m": float(
                    goal_snap_distance
                ),
            },
            "routing": {
                "algorithm": "SQLite-backed A*",
                "expanded_nodes": int(
                    result[
                        "expanded_nodes"
                    ]
                ),
                "max_search_distance_km": float(
                    max_distance_km
                ),
            },
            "route": {
                "nodes": result[
                    "path"
                ],
                "edge_ids": edge_ids,
                "coordinates": coordinates,
            },
            "statistics": {
                "road_edges": int(
                    observed_edges
                ),
                "road_distance_km": (
                    total_length
                    / 1000.0
                ),
                "estimated_travel_time_min": (
                    total_time
                    / 60.0
                ),
                "risk_cost": float(
                    result[
                        "risk_cost"
                    ]
                ),
                "mean_flood_risk": float(
                    mean_flood
                ),
                "maximum_flood_risk": float(
                    max_flood
                ),
                "mean_uncertainty_risk": float(
                    mean_uncertainty
                ),
                "maximum_uncertainty_risk": float(
                    max_uncertainty
                ),
                "mean_total_risk": float(
                    mean_total_risk
                ),
                "maximum_bridge_risk": float(
                    max_bridge
                ),
                "bridge_edges": int(
                    bridge_edges
                ),
            },
        }


RoadRouter = SQLiteRoadRouter
