"""
run_collect.py
--------------
시나리오 실행 + 데이터 수집 통합 스크립트.
lap이 끝날 때마다 자동으로 새 episode 폴더를 생성하고 데이터를 저장한다.

사용법:
    python3 run_collect.py --zone urban --scenario sudden_brake
"""

import argparse
import sys
import threading
import time
import yaml
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data_collector"))
sys.path.insert(0, str(ROOT / "scenario_runner"))


def _nav_waypoints(route_points, ego, n_ahead: int = 50):
    """현재 ego 위치에서 가장 가까운 경로 포인트부터 n_ahead개 반환."""
    import numpy as np
    if not route_points or ego is None:
        return np.zeros((0, 2), dtype=np.float32)
    pts = np.array([[p[0], p[1]] for p in route_points], dtype=np.float32)
    ego_xy = np.array([ego.x, ego.y], dtype=np.float32)
    closest = int(np.argmin(np.linalg.norm(pts - ego_xy, axis=1)))
    return pts[closest: closest + n_ahead]


def _deep_update(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_global_cfg() -> dict:
    cfg = _load_yaml(ROOT / "scenario_runner/config/global.yaml")
    local = ROOT / "scenario_runner/config/local.yaml"
    if local.exists():
        _deep_update(cfg, _load_yaml(local))

    for key, val in cfg.get("paths", {}).items():
        if isinstance(val, str) and not Path(val).is_absolute():
            cfg["paths"][key] = str(ROOT / val)

    return cfg


def main():
    parser = argparse.ArgumentParser(description="시나리오 실행 + 데이터 수집 통합")
    parser.add_argument("--zone",     default="urban")
    parser.add_argument("--scenario", default="sudden_brake")
    args = parser.parse_args()

    # ── 설정 로드 ──────────────────────────────────────────────
    coll_cfg   = _load_yaml(ROOT / "data_collector/config/collection_config.yaml")
    global_cfg = _load_global_cfg()
    zone_cfg   = _load_yaml(ROOT / f"scenario_runner/config/{args.zone}.yaml")
    scenario_cfg = zone_cfg["scenarios"][args.scenario]

    # ── 컴포넌트 초기화 ────────────────────────────────────────
    from scenario_runner.utils.grpc_client import MoraiGrpcClient
    from scenario_runner.utils.map_loader import MGeoMapLoader
    from data_collector.ros.ros_manager import ROSManager
    from data_collector.core.data_writer import DataWriter
    from data_collector.core.scenario_params import ScenarioParamGenerator

    coll_cfg["map_dir"] = str(ROOT / "morai_gym" / "lib" / "core" / "birdiview" / "map")

    print("[run_collect] ROS 구독 시작...")
    ros_mgr = ROSManager(coll_cfg)
    ros_mgr.start()

    print("[run_collect] gRPC 연결 중...")
    grpc_client = MoraiGrpcClient(global_cfg)
    grpc_client.connect()

    map_loader = MGeoMapLoader(global_cfg["paths"]["mgeo_root"])
    writer     = DataWriter(coll_cfg)
    param_gen  = ScenarioParamGenerator(coll_cfg)

    # ── 시나리오 생성 ──────────────────────────────────────────
    if args.zone == "urban" and args.scenario == "sudden_brake":
        from scenario_runner.zones.urban_scenarios import UrbanSuddenBrakeExpertScenario
        scenario = UrbanSuddenBrakeExpertScenario(
            grpc_client  = grpc_client,
            map_loader   = map_loader,
            global_cfg   = global_cfg,
            scenario_cfg = scenario_cfg,
        )
    elif args.zone == "urban" and args.scenario == "basic_drive":
        from scenario_runner.zones.urban_scenarios import UrbanBasicDriveScenario
        scenario = UrbanBasicDriveScenario(
            grpc_client  = grpc_client,
            map_loader   = map_loader,
            global_cfg   = global_cfg,
            scenario_cfg = scenario_cfg,
        )
    else:
        raise ValueError(f"지원하지 않는 시나리오: {args.zone}/{args.scenario}")

    # ── 에피소드 전환 이벤트 ───────────────────────────────────
    lap_end_event = threading.Event()
    done_event    = threading.Event()

    def on_lap_end(lap_num: int):
        print(f"[run_collect] lap {lap_num} 완료 → episode 전환")
        lap_end_event.set()

    scenario.on_lap_end = on_lap_end

    # ── 시나리오 백그라운드 스레드 ─────────────────────────────
    setup_done_event = threading.Event()

    def run_scenario():
        try:
            scenario.setup()
            print("[run_collect] 시나리오 setup 완료 → 데이터 수집 시작")
            setup_done_event.set()
            scenario.run_timeline()
        except Exception as e:
            print(f"[run_collect] 시나리오 오류: {e}")
        finally:
            done_event.set()
            lap_end_event.set()
            setup_done_event.set()

    scenario_thread = threading.Thread(target=run_scenario, daemon=True)
    scenario_thread.start()

    print("[run_collect] 시나리오 setup 대기 중...")
    setup_done_event.wait()

    # ── 데이터 수집 메인 루프 ──────────────────────────────────
    SAVE_HZ       = coll_cfg.get("collection", {}).get("longtail_hz", 10)
    save_period   = 1.0 / SAVE_HZ
    episode_id    = 0

    # 기존 episode 폴더 확인 후 이어서 번호 매기기
    from pathlib import Path as _Path
    import re as _re
    zone_str = f"zone_{args.zone}"
    scen_str = f"scenario_{args.scenario}"
    ep_root  = _Path(coll_cfg["dataset"]["root_dir"]) / zone_str / scen_str
    existing = [
        int(m.group(1))
        for d in (ep_root.glob("episode_*") if ep_root.exists() else [])
        if (m := _re.match(r"episode_(\d+)", d.name))
    ]
    episode_id = max(existing, default=0)
    print(f"[run_collect] 기존 episode {episode_id}개 확인 → episode_{episode_id+1:03d}부터 시작")
    print(f"[run_collect] 수집 시작 (zone={args.zone}, scenario={args.scenario}, {SAVE_HZ}Hz)")

    try:
        while not done_event.is_set():
            episode_id += 1
            params  = param_gen.generate(args.zone, args.scenario, episode_id)
            ep_dir  = writer.begin_episode(params)
            print(f"\n[run_collect] Episode {episode_id:03d} 시작 → {ep_dir}")

            lap_end_event.clear()
            frame_id  = 0
            last_save = 0.0

            # 첫 센서 데이터 대기
            waited = False
            while not done_event.is_set():
                if ros_mgr.wait_for_data(timeout_sec=0.5):
                    waited = True
                    break

            if not waited:
                writer.end_episode(False, "sensor_timeout")
                continue

            # 수집 루프 (lap 끝나거나 시나리오 종료까지)
            while not lap_end_event.is_set() and not done_event.is_set():
                snap = ros_mgr.get_snapshot(frame_id, params.is_longtail)
                if snap and snap.ego:
                    now = time.time()
                    if now - last_save >= save_period:
                        frame_id           += 1
                        snap.frame_id       = frame_id
                        snap.nav_waypoints  = _nav_waypoints(params.route_points, snap.ego)
                        writer.write_frame(snap)
                        last_save = now
                time.sleep(0.01)

            summary = writer.end_episode(True, "lap_complete")
            print(f"[run_collect] Episode {episode_id:03d} 저장 완료 "
                  f"frames={frame_id}  "
                  f"elapsed={summary.get('duration_sec', 0):.1f}s")

    except KeyboardInterrupt:
        print("\n[run_collect] 중단됨 (Ctrl+C)")
    finally:
        grpc_client.stop()
        ros_mgr.stop()
        writer.print_summary()


if __name__ == "__main__":
    main()
