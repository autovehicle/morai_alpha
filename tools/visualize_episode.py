"""
tools/visualize_episode.py
--------------------------
수집된 episode 데이터를 시각화하는 뷰어.

[재생 모드] 저장된 .npz를 프레임 순서대로 재생
    python3 tools/visualize_episode.py --episode <경로>
    python3 tools/visualize_episode.py --episode zone_urban/scenario_sudden_brake/episode_001
    python3 tools/visualize_episode.py --episode zone_urban/scenario_sudden_brake/episode_001 --speed 2.0

[실시간 모드] run_collect.py가 쓰는 폴더를 감시하며 실시간 표시
    python3 tools/visualize_episode.py --live
    python3 tools/visualize_episode.py --live --zone urban --scenario sudden_brake
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# NAS 기본 경로
NAS_ROOT = Path("/mnt/z/AIM_2026/대회/2026 대학생 AI SW 자율주행 경진대회/Team A.I.M")

# ─────────────────────────────────────────────
#  공통 유틸
# ─────────────────────────────────────────────

def find_episode_dir(episode_arg: str) -> Path:
    p = Path(episode_arg)
    if p.is_absolute() and p.exists():
        return p
    candidates = [
        ROOT / episode_arg,
        NAS_ROOT / episode_arg,
    ]
    for c in candidates:
        if c.exists():
            return c
    print(f"[ERROR] 경로를 찾을 수 없습니다: {episode_arg}")
    sys.exit(1)


def find_latest_episode(zone: str, scenario: str) -> Path | None:
    """현재 수집 중인 (가장 최신) episode 폴더를 반환."""
    base = NAS_ROOT / f"zone_{zone}" / f"scenario_{scenario}"
    if not base.exists():
        return None
    eps = sorted(base.glob("episode_*"))
    return eps[-1] if eps else None


def build_layout():
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.36)

    ax_cam   = fig.add_subplot(gs[:2, :2])
    ax_traj  = fig.add_subplot(gs[:2, 2])
    ax_speed = fig.add_subplot(gs[2, :2])
    ax_steer = fig.add_subplot(gs[2, 2])

    ax_cam.axis("off")
    ax_cam.set_title("Front Camera", fontsize=10)

    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    img_handle = ax_cam.imshow(dummy)
    frame_text = ax_cam.text(
        0.01, 0.97, "", transform=ax_cam.transAxes,
        color="white", fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55)
    )

    ax_traj.set_title("Ego Trajectory", fontsize=10)
    ax_traj.set_xlabel("x (m)", fontsize=8)
    ax_traj.set_ylabel("y (m)", fontsize=8)
    ax_traj.set_aspect("equal", "box")
    traj_line, = ax_traj.plot([], [], "b-", alpha=0.4, linewidth=1)
    traj_dot,  = ax_traj.plot([], [], "ro", markersize=6)

    ax_speed.set_title("Speed (m/s)", fontsize=10)
    ax_speed.set_xlabel("time (s)", fontsize=8)
    ax_speed.set_ylabel("m/s", fontsize=8)
    speed_line_bg, = ax_speed.plot([], [], "g-", alpha=0.3, linewidth=1)
    speed_dot, = ax_speed.plot([], [], "go", markersize=5)
    speed_cur = ax_speed.axvline(x=0, color="g", linestyle="--", alpha=0.7)

    ax_steer.set_title("Steer (rad)", fontsize=10)
    ax_steer.set_xlabel("time (s)", fontsize=8)
    ax_steer.set_ylabel("rad", fontsize=8)
    steer_line_bg, = ax_steer.plot([], [], "m-", alpha=0.3, linewidth=1)
    steer_dot, = ax_steer.plot([], [], "mo", markersize=5)
    steer_cur = ax_steer.axvline(x=0, color="m", linestyle="--", alpha=0.7)

    handles = dict(
        fig=fig,
        img=img_handle,
        frame_text=frame_text,
        traj_line=traj_line,
        traj_dot=traj_dot,
        speed_line_bg=speed_line_bg,
        speed_dot=speed_dot,
        speed_cur=speed_cur,
        steer_line_bg=steer_line_bg,
        steer_dot=steer_dot,
        steer_cur=steer_cur,
        ax_traj=ax_traj,
        ax_speed=ax_speed,
        ax_steer=ax_steer,
    )
    return handles


def render_frame(h: dict, frames: list, i: int, xs, ys, ts, speeds, steers):
    f = frames[i]
    n = len(frames)

    # 카메라
    cam = f.get("cam_front", None)
    if cam is not None and cam.ndim == 3 and cam.size > 0:
        h["img"].set_data(cam)
        h["img"].set_extent([0, cam.shape[1], cam.shape[0], 0])
        h["ax_traj"].figure.axes[0].set_xlim(0, cam.shape[1])

    h["frame_text"].set_text(
        f"frame {i+1}/{n}  t={ts[i]:.1f}s  "
        f"spd={speeds[i]:.1f}m/s  steer={steers[i]:.3f}rad"
    )

    # 궤적
    h["traj_line"].set_data(xs[:i+1], ys[:i+1])
    h["traj_dot"].set_data([xs[i]], [ys[i]])
    if i == 0:
        # 최초 렌더에서 축 범위 설정
        margin = 5
        h["ax_traj"].set_xlim(min(xs) - margin, max(xs) + margin)
        h["ax_traj"].set_ylim(min(ys) - margin, max(ys) + margin)

    # 속도
    h["speed_line_bg"].set_data(ts, speeds)
    h["speed_dot"].set_data([ts[i]], [speeds[i]])
    h["speed_cur"].set_xdata([ts[i], ts[i]])
    if i == 0:
        h["ax_speed"].set_xlim(0, max(ts) + 1 if ts else 1)
        h["ax_speed"].set_ylim(min(speeds) - 1, max(speeds) + 1)

    # 조향
    h["steer_line_bg"].set_data(ts, steers)
    h["steer_dot"].set_data([ts[i]], [steers[i]])
    h["steer_cur"].set_xdata([ts[i], ts[i]])
    if i == 0:
        h["ax_steer"].set_xlim(0, max(ts) + 1 if ts else 1)
        sr = max(abs(min(steers)), abs(max(steers))) + 0.05
        h["ax_steer"].set_ylim(-sr, sr)

    h["fig"].canvas.draw()
    h["fig"].canvas.flush_events()


# ─────────────────────────────────────────────
#  재생 모드
# ─────────────────────────────────────────────

def mode_playback(ep_dir: Path, speed: float):
    frame_files = sorted((ep_dir / "frames").glob("*.npz"))
    if not frame_files:
        print(f"[ERROR] frames 폴더에 .npz 없음: {ep_dir}/frames")
        sys.exit(1)

    print(f"[visualize] {len(frame_files)}개 프레임 로드 중...")
    frames = [np.load(str(f)) for f in frame_files]
    n = len(frames)

    xs     = [float(f["ego"][0]) for f in frames]
    ys     = [float(f["ego"][1]) for f in frames]
    speeds = [float(f["ego"][4]) for f in frames]
    steers = [float(f["ego"][5]) for f in frames]
    ts_raw = [float(f["timestamp_ns"][0]) / 1e9 for f in frames]
    t0     = ts_raw[0]
    ts     = [t - t0 for t in ts_raw]

    h = build_layout()
    h["fig"].suptitle(f"Replay: {ep_dir.name}", fontsize=11)
    plt.ion()
    plt.show()

    print(f"[visualize] 재생 시작 ({speed}배속). 창 닫으면 종료.")
    for i in range(n):
        t_start = time.time()
        render_frame(h, frames, i, xs, ys, ts, speeds, steers)
        if not plt.fignum_exists(h["fig"].number):
            break
        if i < n - 1:
            dt = (ts[i + 1] - ts[i]) / speed
            elapsed = time.time() - t_start
            wait = dt - elapsed
            if wait > 0:
                time.sleep(wait)

    print("[visualize] 재생 완료.")
    plt.ioff()
    plt.show()


# ─────────────────────────────────────────────
#  실시간 모드
# ─────────────────────────────────────────────

def mode_live(zone: str, scenario: str):
    """run_collect.py 실행 중 가장 최신 episode 폴더를 감시하며 실시간 표시."""
    print(f"[live] zone={zone} scenario={scenario} 감시 시작...")
    print(f"[live] 경로: {NAS_ROOT}/zone_{zone}/scenario_{scenario}/")

    h = build_layout()
    h["fig"].suptitle(f"Live: {zone}/{scenario}", fontsize=11)
    plt.ion()
    plt.show()

    seen_files: set = set()
    frames: list = []
    xs, ys, ts, speeds, steers = [], [], [], [], []
    current_ep = None
    t0 = None

    while plt.fignum_exists(h["fig"].number):
        # 최신 episode 폴더 감지
        ep = find_latest_episode(zone, scenario)

        if ep != current_ep:
            if ep is not None:
                print(f"[live] 새 episode 감지: {ep.name}")
            current_ep = ep
            seen_files.clear()
            frames.clear()
            xs.clear(); ys.clear(); ts.clear()
            speeds.clear(); steers.clear()
            t0 = None
            h["traj_line"].set_data([], [])
            h["speed_line_bg"].set_data([], [])
            h["steer_line_bg"].set_data([], [])

        if current_ep is None:
            h["frame_text"].set_text("수집 대기 중... (run_collect.py 실행 필요)")
            h["fig"].canvas.draw()
            h["fig"].canvas.flush_events()
            time.sleep(1.0)
            continue

        frames_dir = current_ep / "frames"
        if not frames_dir.exists():
            time.sleep(0.5)
            continue

        new_files = sorted(
            f for f in frames_dir.glob("*.npz")
            if f not in seen_files
        )

        for fpath in new_files:
            seen_files.add(fpath)
            try:
                f = np.load(str(fpath))
            except Exception:
                continue  # 아직 쓰는 중인 파일 스킵

            frames.append(f)
            xs.append(float(f["ego"][0]))
            ys.append(float(f["ego"][1]))
            speeds.append(float(f["ego"][4]))
            steers.append(float(f["ego"][5]))

            t_ns = float(f["timestamp_ns"][0]) / 1e9
            if t0 is None:
                t0 = t_ns
            ts.append(t_ns - t0)

            i = len(frames) - 1

            # 동적 축 범위 갱신
            if len(xs) > 1:
                margin = 5
                h["ax_traj"].set_xlim(min(xs) - margin, max(xs) + margin)
                h["ax_traj"].set_ylim(min(ys) - margin, max(ys) + margin)
                h["ax_speed"].set_xlim(0, max(ts) + 1)
                h["ax_speed"].set_ylim(min(speeds) - 0.5, max(speeds) + 0.5)
                h["ax_steer"].set_xlim(0, max(ts) + 1)
                sr = max(abs(min(steers)), abs(max(steers))) + 0.05
                h["ax_steer"].set_ylim(-sr, sr)

            render_frame(h, frames, i, xs, ys, ts, speeds, steers)

        if not new_files:
            # 새 파일 없을 때 50ms 대기
            h["fig"].canvas.flush_events()
            time.sleep(0.05)

    print("[live] 창이 닫혔습니다.")
    plt.ioff()


# ─────────────────────────────────────────────
#  진입점
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MORAI 수집 데이터 뷰어")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--episode", help="재생할 episode 폴더 경로")
    group.add_argument("--live",    action="store_true", help="실시간 감시 모드")

    parser.add_argument("--zone",     default="urban",       help="[live 모드] zone 이름")
    parser.add_argument("--scenario", default="sudden_brake",help="[live 모드] scenario 이름")
    parser.add_argument("--speed",    type=float, default=1.0, help="[재생 모드] 배속")

    args = parser.parse_args()

    if args.live:
        mode_live(args.zone, args.scenario)
    else:
        ep_dir = find_episode_dir(args.episode)
        print(f"[visualize] 에피소드: {ep_dir}")
        mode_playback(ep_dir, args.speed)


if __name__ == "__main__":
    main()
