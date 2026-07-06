import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scenario_runner_external import (
    apply_collection_morai_overrides,
    ensure_scenario_runner_on_path,
    load_global_cfg,
)

ensure_scenario_runner_on_path()

from scenario_runner.utils.grpc_client import MoraiGrpcClient


class MoraiClient:
    def __init__(self, config):
        self.config = config
        self.grpc_cfg = self._load_grpc_config()
        self.grpc = MoraiGrpcClient(self.grpc_cfg)

        self.connected = False
        self.world_started = False
        self.npc_actors = []

    def _load_grpc_config(self):
        return apply_collection_morai_overrides(load_global_cfg(), self.config)

    def connect(self):
        try:
            self.grpc.connect()
            self.connected = True
            return True
        except Exception as e:
            print(f"[MoraiClient] gRPC connect failed: {e}")
            return False

    def disconnect(self):
        try:
            self.grpc.stop()
        except Exception:
            pass
        self.connected = False

    def _to_dict(self, params):
        """
        EpisodeParams 객체 / dict 둘 다 dict로 변환
        """
        if isinstance(params, dict):
            return params

        if hasattr(params, "to_dict"):
            return params.to_dict()

        return vars(params)

    def start_episode(self, params):
        """
        EpisodeParams 기준 episode 시작.
        link 기반 route_links가 있으면 MORAI route/cruise까지 설정한다.
        """
        p = self._to_dict(params)

        ego_tf = self.grpc.make_transform(
            p.get("start_x", 0.0),
            p.get("start_y", 0.0),
            p.get("start_z", 0.0),
            p.get("start_yaw", 0.0),
        )

        if not self.world_started:
            self.grpc.start_world(ego_tf)
            self.world_started = True
        else:
            self.grpc.restart_world(ego_tf)

        route_links = p.get("route_links", []) or []
        waypoint_indices = p.get("route_waypoint_indices", {}) or {}
        decision_range = float(p.get("decision_range_m", 30.0))

        if route_links:
            print(f"[MoraiClient] setting route_links={len(route_links)}")
            print(f"[MoraiClient] start_link={p.get('start_link')} end_link={p.get('end_link')}")

            # world restart 후 ego가 완전히 올라올 시간을 조금 준다.
            import time
            time.sleep(0.3)

            try:
                self.grpc.set_ego_transform(ego_tf)
            except Exception as e:
                print(f"[MoraiClient] set_ego_transform warning: {e}")

            route_ok = False
            try:
                route_ok = self.grpc.set_ego_route(
                    route_links,
                    decision_range=decision_range,
                    waypoint_indices=waypoint_indices,
                )
            except Exception as e:
                print(f"[MoraiClient] waypoint route failed: {e}")

            if not route_ok:
                try:
                    route_ok = self.grpc.set_ego_route(
                        route_links,
                        decision_range=decision_range,
                    )
                except Exception as e:
                    print(f"[MoraiClient] plain route failed: {e}")

            if route_ok:
                try:
                    self.grpc.set_ego_control_mode_cruise()
                    self.grpc.set_ego_cruise(
                        enable=True,
                        link_speed_ratio=int(p.get("link_speed_ratio", 40)),
                        constant_velocity=float(p.get("constant_velocity", 20.0)),
                        cruise_type=p.get("cruise_type", "link"),
                    )
                    print("[MoraiClient] route/cruise configured")
                except Exception as e:
                    print(f"[MoraiClient] cruise setting failed: {e}")
            else:
                print("[MoraiClient] route setup failed; ego may not drive")

        else:
            print("[MoraiClient] route_links 없음 → start/goal 좌표만 사용")
            goal_x = p.get("goal_x", None)
            goal_y = p.get("goal_y", None)
            if goal_x is not None and goal_y is not None:
                try:
                    self.grpc.set_ego_destination(
                        goal_x,
                        goal_y,
                        p.get("goal_z", 0.0),
                        decision_range=decision_range,
                    )
                    self.grpc.set_ego_control_mode_cruise()
                    self.grpc.set_ego_cruise()
                except Exception as e:
                    print(f"[MoraiClient] goal/cruise 설정 스킵: {e}")

        self.npc_actors = []
        return True

    def send_control(self, steer=0.0, throttle=0.0, brake=0.0):
        # MORAI built-in cruise가 route를 따라가므로 여기서는 제어를 건드리지 않는다.
        return True

    def setup_episode(self, params, map_name=None):
        """
        EpisodeManager가 호출하는 함수.
        link 기반 route가 있으면 전체 route_points와 route_links를 반환한다.
        """
        p = self._to_dict(params)
        self.start_episode(p)

        route_points = p.get("route_points", []) or []
        route_links = p.get("route_links", []) or []

        if route_points:
            waypoints = [[float(pt[0]), float(pt[1])] for pt in route_points]
        else:
            waypoints = [
                [p.get("start_x", 0.0), p.get("start_y", 0.0)],
                [p.get("goal_x", 0.0), p.get("goal_y", 0.0)],
            ]

        return waypoints, route_links
