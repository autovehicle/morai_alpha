import os
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any
import yaml

from scenario_runner_external import (
    RUNNER_CONFIG_DIR,
    RUNNER_ROOT,
    ensure_scenario_runner_on_path,
    load_global_cfg,
)

ensure_scenario_runner_on_path()

from scenario_runner.utils.map_loader import MGeoMapLoader
from scenario_runner.utils.route_utils import build_route_between
from scenario_runner.utils.transform_utils import (
    interpolate_on_polyline,
    polyline_length,
    nearest_point_index,
)


@dataclass
class EpisodeParams:
    zone: str
    scenario: str
    episode_id: int

    # Ego start
    start_x: float
    start_y: float
    start_yaw: float
    start_z: float = 0.0

    # Goal
    goal_x: float = 0.0
    goal_y: float = 0.0
    goal_z: float = 0.0
    goal_yaw: float = 0.0

    # Scenario
    npc_count: int = 0
    pedestrian_count: int = 0

    # Data collection
    is_longtail: bool = False
    longtail_triggers: List[str] = field(default_factory=list)
    max_steps: int = 1000

    # Link-based route info
    start_link: str = ""
    end_link: str = ""
    route_links: List[str] = field(default_factory=list)
    route_waypoint_indices: Dict[str, int] = field(default_factory=dict)
    route_points: List[List[float]] = field(default_factory=list)

    # MORAI route/cruise options
    decision_range_m: float = 30.0
    cruise_type: str = "link"
    link_speed_ratio: int = 40
    constant_velocity: float = 20.0

    run_id: int = 0

    def to_dict(self) -> dict:
        return {
            "zone": self.zone,
            "scenario": self.scenario,
            "episode_id": self.episode_id,
            "run_id": self.run_id,
            "start_x": self.start_x,
            "start_y": self.start_y,
            "start_z": self.start_z,
            "start_yaw": self.start_yaw,
            "goal_x": self.goal_x,
            "goal_y": self.goal_y,
            "goal_z": self.goal_z,
            "goal_yaw": self.goal_yaw,
            "npc_count": self.npc_count,
            "pedestrian_count": self.pedestrian_count,
            "is_longtail": self.is_longtail,
            "longtail_triggers": self.longtail_triggers,
            "max_steps": self.max_steps,
            "start_link": self.start_link,
            "end_link": self.end_link,
            "route_links": self.route_links,
            "route_waypoint_indices": self.route_waypoint_indices,
            "route_points": self.route_points,
            "decision_range_m": self.decision_range_m,
            "cruise_type": self.cruise_type,
            "link_speed_ratio": self.link_speed_ratio,
            "constant_velocity": self.constant_velocity,
        }


def deep_update(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


class ScenarioParamGenerator:
    """
    1순위: scenario_runner/config/{zone}.yaml의 link 기반 시나리오 사용
    2순위: 기존 collection_config.yaml의 spawn_ranges 방식 fallback
    """

    def __init__(self, config: dict):
        self.config = config
        self.runner_root = RUNNER_ROOT
        self.rng = random.Random()

    def _load_yaml(self, path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _load_global_cfg(self) -> dict:
        return load_global_cfg()

    def _load_runner_scenario_cfg(self, zone: str, scenario: str):
        zone_path = RUNNER_CONFIG_DIR / f"{zone}.yaml"
        if not zone_path.exists():
            return None

        zone_cfg = self._load_yaml(zone_path)
        scenarios = zone_cfg.get("scenarios", {})
        if scenario not in scenarios:
            return None

        return scenarios[scenario]

    def _load_zone_route_links(self, scenario_cfg: dict, zone: str) -> List[str]:
        zone_links_path = scenario_cfg.get("zone_links_path", "")
        if not zone_links_path:
            return []

        path = Path(zone_links_path)
        if not path.is_absolute():
            path = self.runner_root / path

        data = self._load_yaml(path)
        zone_data = data.get(zone, {})
        route_links = zone_data.get("route_links", [])
        exclude_links = set(zone_data.get("exclude_links", []))

        allow_connector_links = scenario_cfg.get("allow_connector_links", False)
        candidates = []
        for link_id in route_links:
            if link_id in exclude_links:
                continue
            if not allow_connector_links and "-" in link_id:
                continue
            candidates.append(link_id)

        return candidates

    def _build_route_points(self, map_loader: MGeoMapLoader, route_links: List[str]) -> List[List[float]]:
        points = []
        for link_id in route_links:
            link_points = map_loader.get_link_points(link_id)
            if points and link_points:
                points.extend(link_points[1:])
            else:
                points.extend(link_points)
        return points

    def _build_route_waypoint_indices(
        self,
        map_loader: MGeoMapLoader,
        route_links: List[str],
        goal_x: float,
        goal_y: float,
    ) -> Dict[str, int]:
        result = {}
        for link_id in route_links:
            pts = map_loader.get_link_points(link_id)
            result[link_id] = max(0, len(pts) - 1)

        if route_links:
            end_link = route_links[-1]
            end_pts = map_loader.get_link_points(end_link)
            result[end_link] = nearest_point_index(end_pts, goal_x, goal_y)

        return result

    def _select_link_route(self, map_loader: MGeoMapLoader, scenario_cfg: dict, zone: str):
        randomize = bool(scenario_cfg.get("randomize_links", False))

        if not randomize:
            start_link = scenario_cfg["start_link"]
            end_link = scenario_cfg["end_link"]
            route_links, route_len = build_route_between(map_loader, start_link, end_link)
            return start_link, end_link, route_links, route_len

        candidates = self._load_zone_route_links(scenario_cfg, zone)
        if len(candidates) < 2:
            raise RuntimeError(f"Not enough route link candidates for zone={zone}")

        attempts = int(scenario_cfg.get("random_route_attempts", 200))
        min_len = float(scenario_cfg.get("random_min_route_length_m", 20.0))
        max_len = float(scenario_cfg.get("random_max_route_length_m", 0.0))

        last_error = None
        for _ in range(attempts):
            start_link, end_link = self.rng.sample(candidates, 2)
            try:
                route_links, route_len = build_route_between(map_loader, start_link, end_link)
            except Exception as e:
                last_error = e
                continue

            if route_len < min_len:
                continue
            if max_len > 0.0 and route_len > max_len:
                continue

            return start_link, end_link, route_links, route_len

        raise RuntimeError(f"Failed to select valid random route. last_error={last_error}")

    def _generate_from_runner_config(self, zone: str, scenario: str, episode_id: int) -> EpisodeParams:
        scenario_cfg = self._load_runner_scenario_cfg(zone, scenario)
        if scenario_cfg is None:
            raise KeyError(f"No scenario_runner config for {zone}/{scenario}")

        global_cfg = self._load_global_cfg()
        mgeo_root = global_cfg["paths"]["mgeo_root"]
        map_loader = MGeoMapLoader(mgeo_root)

        start_link, end_link, route_links, route_len = self._select_link_route(
            map_loader, scenario_cfg, zone
        )

        # 시작점: start_link 위 offset 지점
        if "ego_spawn_min_offset_m" in scenario_cfg and "ego_spawn_max_offset_m" in scenario_cfg:
            start_offset = self.rng.uniform(
                float(scenario_cfg["ego_spawn_min_offset_m"]),
                float(scenario_cfg["ego_spawn_max_offset_m"]),
            )
        else:
            start_offset = float(scenario_cfg.get("ego_spawn_offset_m", 5.0))

        start_points = map_loader.get_link_points(start_link)
        sx, sy, sz, syaw = interpolate_on_polyline(start_points, start_offset)

        # 목표점: end_link 끝에서 goal_offset_from_end_m만큼 안쪽
        end_points = map_loader.get_link_points(end_link)
        goal_offset_from_end = float(scenario_cfg.get("goal_offset_from_end_m", 3.0))
        goal_offset = max(0.0, polyline_length(end_points) - goal_offset_from_end)
        gx, gy, gz, gyaw = interpolate_on_polyline(end_points, goal_offset)

        route_points = self._build_route_points(map_loader, route_links)
        route_waypoint_indices = self._build_route_waypoint_indices(
            map_loader, route_links, gx, gy
        )

        # collection_config에 같은 scenario가 있으면 longtail/npc 설정 일부 사용
        coll_scenario_cfg = self.config.get("scenarios", {}).get(scenario, {})
        triggers = coll_scenario_cfg.get("longtail_triggers", [])
        is_longtail = len(triggers) > 0

        npc_min = scenario_cfg.get("npc_count_min", coll_scenario_cfg.get("npc_count", [0, 0])[0] if "npc_count" in coll_scenario_cfg else 0)
        npc_max = scenario_cfg.get("npc_count_max", coll_scenario_cfg.get("npc_count", [0, 0])[1] if "npc_count" in coll_scenario_cfg else 0)
        ped_min, ped_max = coll_scenario_cfg.get("pedestrian_count", [0, 0])

        max_steps = self.config.get("zones", {}).get(zone, {}).get(
            "max_steps",
            int(float(scenario_cfg.get("run_duration_sec", 10.0)) * 10),
        )

        print(f"[ScenarioParamGenerator] link route selected")
        print(f"  start_link={start_link}")
        print(f"  end_link={end_link}")
        print(f"  start=({sx:.3f}, {sy:.3f}, {sz:.3f}, yaw={syaw:.3f})")
        print(f"  goal=({gx:.3f}, {gy:.3f}, {gz:.3f}, yaw={gyaw:.3f})")
        print(f"  route_links={len(route_links)}, length={route_len:.1f}m")

        return EpisodeParams(
            zone=zone,
            scenario=scenario,
            episode_id=episode_id,
            start_x=sx,
            start_y=sy,
            start_z=sz,
            start_yaw=syaw,
            goal_x=gx,
            goal_y=gy,
            goal_z=gz,
            goal_yaw=gyaw,
            npc_count=random.randint(int(npc_min), int(npc_max)),
            pedestrian_count=random.randint(int(ped_min), int(ped_max)),
            is_longtail=is_longtail,
            longtail_triggers=triggers,
            max_steps=int(max_steps),
            start_link=start_link,
            end_link=end_link,
            route_links=route_links,
            route_waypoint_indices=route_waypoint_indices,
            route_points=route_points,
            decision_range_m=float(scenario_cfg.get("decision_range_m", 30.0)),
            cruise_type=str(scenario_cfg.get("cruise_type", "link")),
            link_speed_ratio=int(scenario_cfg.get("link_speed_ratio", 40)),
            constant_velocity=float(scenario_cfg.get("constant_velocity", 20.0)),
        )

    def _generate_from_collection_config(self, zone: str, scenario: str, episode_id: int) -> EpisodeParams:
        zone_cfg = self.config["zones"][zone]
        scenario_cfg = self.config["scenarios"][scenario]

        start_x = random.uniform(*zone_cfg["spawn_ranges"]["x"])
        start_y = random.uniform(*zone_cfg["spawn_ranges"]["y"])
        start_yaw = random.uniform(*zone_cfg["spawn_ranges"]["yaw"])

        goal_x = random.uniform(*zone_cfg["goal_ranges"]["x"])
        goal_y = random.uniform(*zone_cfg["goal_ranges"]["y"])

        npc_count = random.randint(*scenario_cfg["npc_count"])
        pedestrian_count = random.randint(*scenario_cfg["pedestrian_count"])

        triggers = scenario_cfg.get("longtail_triggers", [])
        is_longtail = len(triggers) > 0

        print("[ScenarioParamGenerator] fallback: collection_config spawn_ranges 사용")

        return EpisodeParams(
            zone=zone,
            scenario=scenario,
            episode_id=episode_id,
            start_x=start_x,
            start_y=start_y,
            start_yaw=start_yaw,
            goal_x=goal_x,
            goal_y=goal_y,
            npc_count=npc_count,
            pedestrian_count=pedestrian_count,
            is_longtail=is_longtail,
            longtail_triggers=triggers,
            max_steps=zone_cfg.get("max_steps", 1000),
        )

    def generate(self, zone: str, scenario: str, episode_id: int) -> EpisodeParams:
        try:
            return self._generate_from_runner_config(zone, scenario, episode_id)
        except Exception as e:
            print(f"[ScenarioParamGenerator] runner config 사용 실패: {e}")
            return self._generate_from_collection_config(zone, scenario, episode_id)
