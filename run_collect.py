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
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data_collector"))

from scenario_runner_external import (
    apply_collection_morai_overrides,
    create_scenario,
    ensure_scenario_runner_on_path,
    load_global_cfg,
    load_yaml,
    load_zone_cfg,
)

ensure_scenario_runner_on_path()


def _nav_waypoints(route_points, ego, n_ahead: int = 50):
    """현재 ego 위치에서 가장 가까운 경로 포인트부터 n_ahead개 반환."""
    import numpy as np
    if not route_points or ego is None:
        return np.zeros((0, 2), dtype=np.float32)
    pts = np.array([[p[0], p[1]] for p in route_points], dtype=np.float32)
    ego_xy = np.array([ego.x, ego.y], dtype=np.float32)
    closest = int(np.argmin(np.linalg.norm(pts - ego_xy, axis=1)))
    return pts[closest: closest + n_ahead]


# 0:background 1:center_yellow 2:solid 3:dashed 4:stopline 5:dynamic 6:drivable 7:crosswalk
_BEV_COLORS_BGR = {
    6: (60, 60, 60), 0: (30, 30, 30), 1: (0, 215, 255), 2: (255, 255, 255),
    3: (255, 220, 120), 4: (60, 60, 255), 7: (255, 120, 200), 5: (0, 140, 255),
}
_BEV_DRAW_ORDER = [6, 0, 1, 2, 3, 4, 7, 5]


def _bev_to_bgr(bev_map):
    """bev_map(H,W,8) → cv2 표시용 BGR 컬러 이미지."""
    import numpy as np
    rgb = np.zeros((*bev_map.shape[:2], 3), dtype="uint8")
    for ch in _BEV_DRAW_ORDER:
        rgb[bev_map[:, :, ch] > 0.5] = _BEV_COLORS_BGR[ch]
    return rgb


def main():
    parser = argparse.ArgumentParser(description="시나리오 실행 + 데이터 수집 통합")
    parser.add_argument("--zone",     default="urban")
    parser.add_argument("--scenario", default="sudden_brake")
    parser.add_argument("--show-bev", action="store_true",
                         help="수집 중인 BEV맵을 실시간 cv2 창으로 표시")
    args = parser.parse_args()

    # ── 설정 로드 ──────────────────────────────────────────────
    coll_cfg   = load_yaml(ROOT / "data_collector/config/collection_config.yaml")
    global_cfg = apply_collection_morai_overrides(load_global_cfg(), coll_cfg)
    zone_cfg   = load_zone_cfg(args.zone)
    scenario_cfg = zone_cfg["scenarios"][args.scenario]

    # ── 컴포넌트 초기화 ────────────────────────────────────────
    from scenario_runner.utils.grpc_client import MoraiGrpcClient
    from scenario_runner.utils.map_loader import MGeoMapLoader
    from data_collector.ros.ros_manager import ROSManager
    from data_collector.core.data_writer import DataWriter
    from data_collector.core.scenario_params import ScenarioParamGenerator

    coll_cfg["map_dir"] = global_cfg["paths"]["mgeo_root"]

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
    scenario = create_scenario(
        zone=args.zone,
        scenario=args.scenario,
        grpc_client=grpc_client,
        map_loader=map_loader,
        global_cfg=global_cfg,
        scenario_cfg=scenario_cfg,
    )

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
    # 저장 Hz는 이제 wall-clock이 아니라 gRPC sync_mode.tick_period(grpc_client.py)로 결정된다.
    # tick_period=100ms(=10Hz)로 설정돼 있으면 매 tick마다 저장 = 10Hz.
    SAVE_HZ       = coll_cfg.get("collection", {}).get("longtail_hz", 10)
    episode_id    = 0

    # 기존 scenario_run 폴더 확인 후 새 run 번호 부여 (이번 실행 = 새 scenario_run)
    import re as _re
    zone_str = f"zone_{args.zone}"
    scen_str = f"scenario_{args.scenario}"
    scen_root = Path(coll_cfg["dataset"]["root_dir"]) / zone_str / scen_str
    existing_runs = [
        int(m.group(1))
        for d in (scen_root.glob("scenario_run_*") if scen_root.exists() else [])
        if (m := _re.match(r"scenario_run_(\d+)", d.name))
    ]
    run_id = max(existing_runs, default=0) + 1
    print(f"[run_collect] scenario_run_{run_id:03d} 시작")
    print(f"[run_collect] 수집 시작 (zone={args.zone}, scenario={args.scenario}, {SAVE_HZ}Hz)")

    try:
        while not done_event.is_set():
            episode_id += 1
            params  = param_gen.generate(args.zone, args.scenario, episode_id)
            params.run_id = run_id
            ep_dir  = writer.begin_episode(params)
            print(f"\n[run_collect] Episode {episode_id:03d} 시작 → {ep_dir}")

            lap_end_event.clear()
            frame_id  = 0

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
            # gRPC Synchronous Mode: tick()을 명시적으로 호출해야 시뮬레이터가 진행됨.
            # wall-clock(time.time()) 대신 tick 리턴값(SyncTimestamp)을 프레임에 그대로 붙인다.
            while not lap_end_event.is_set() and not done_event.is_set():
                sync_ts = grpc_client.world.tick(1)
                if not sync_ts:
                    continue

                snap = ros_mgr.get_snapshot(
                    frame_id, params.is_longtail,
                    tick_count=sync_ts.frame_count,
                    sim_elapsed_ns=sync_ts.elapsed_time,
                )
                if snap and snap.ego:
                    frame_id           += 1
                    snap.frame_id        = frame_id
                    snap.nav_waypoints   = _nav_waypoints(scenario.route_points, snap.ego)
                    snap.nav_link_ids    = scenario.route_links
                    writer.write_frame(snap)

                    if args.show_bev and snap.bev_map is not None:
                        import cv2
                        cv2.imshow("BEV (live)", _bev_to_bgr(snap.bev_map))
                        cv2.waitKey(1)

            summary = writer.end_episode(True, "lap_complete")
            print(f"[run_collect] Episode {episode_id:03d} 저장 완료 "
                  f"frames={frame_id}  "
                  f"elapsed={summary.get('duration_sec', 0):.1f}s")

    except KeyboardInterrupt:
        print("\n[run_collect] 중단됨 (Ctrl+C)")
    finally:
        if args.show_bev:
            import cv2
            cv2.destroyAllWindows()
        try:
            scenario.cleanup()
        except Exception as e:
            print(f"[run_collect] scenario cleanup 오류: {e}")
        grpc_client.stop()
        ros_mgr.stop()
        writer.print_summary()


if __name__ == "__main__":
    main()
