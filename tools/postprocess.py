"""
tools/postprocess.py
--------------------
수집 완료된 에피소드에 미래 데이터(future_*)를 추가한다.

추가 필드:
  future_steering  (K=15,) float32  — 미래 15프레임 조향각  (10Hz, 1.5초)
  future_throttle  (K=15,) float32
  future_brake     (K=15,) float32
  future_traj_x    (T=6,)  float32  — 미래 궤적 Δx 오프셋 (2Hz 서브샘플, 3초)
  future_traj_y    (T=6,)  float32  — 미래 궤적 Δy 오프셋
  future_traj_vx   (T=6,)  float32  — 미래 속도 x 성분
  future_traj_vy   (T=6,)  float32  — 미래 속도 y 성분

sampling 기준:
  - 수집 주파수: 10Hz
  - future_action: t+1 ~ t+15 (10Hz, 1.5초)
  - future_traj  : t+5, t+10, ..., t+30 (2Hz 서브샘플, 3초)
  - 마지막 30프레임(= max(K, T*stride))은 미래 데이터 부족으로 제외

사용법:
  python tools/postprocess.py --dataset_root /path/to/dataset
  python tools/postprocess.py --dataset_root /path/to/dataset --out_root /path/to/output
"""

import argparse
import shutil
from pathlib import Path

import numpy as np

# ── 하이퍼파라미터 ────────────────────────────────────────────
K_ACTION     = 15  # future action 스텝 수
K_TRAJ       = 6   # future trajectory 스텝 수
TRAJ_STRIDE  = 5   # 10Hz → 2Hz 서브샘플 간격 (프레임 단위)
K_TRAJ_FRAMES = K_TRAJ * TRAJ_STRIDE  # 미래 궤적이 필요한 최대 프레임 오프셋 = 30
MIN_FUTURE   = max(K_ACTION, K_TRAJ_FRAMES)  # 한 프레임에 필요한 미래 프레임 수 = 30


# ── 에피소드 처리 ─────────────────────────────────────────────

def process_episode(ep_dir: Path, out_dir: Path) -> int:
    """
    에피소드 하나를 처리한다.
    반환: 저장된 유효 프레임 수 (0이면 스킵)
    """
    frames_dir = ep_dir / "frames"
    frame_files = sorted(frames_dir.glob("*.npz"))
    N = len(frame_files)

    if N <= MIN_FUTURE:
        print(f"  [SKIP] 프레임 수 부족: {N} <= {MIN_FUTURE}")
        return 0

    # ── 전체 에피소드에서 필요한 필드만 미리 로드 ──────────────
    # expert: (N, 3)  [steer, throttle, brake]
    # ego   : (N, 6)  [x, y, z, yaw, speed, steer]
    experts = np.empty((N, 3), dtype=np.float32)
    egos    = np.empty((N, 6), dtype=np.float32)

    for i, f in enumerate(frame_files):
        d = np.load(f)
        experts[i] = d["expert"]
        egos[i]    = d["ego"]

    # ── 유효 프레임: 0 ~ N-MIN_FUTURE-1 ──────────────────────
    valid_n = N - MIN_FUTURE
    out_frames = out_dir / "frames"
    out_frames.mkdir(parents=True, exist_ok=True)

    # 에피소드 메타 파일 복사
    for meta in ep_dir.glob("*.json"):
        shutil.copy2(meta, out_dir / meta.name)

    # ── 프레임별 future 필드 계산 및 저장 ─────────────────────
    for t in range(valid_n):
        data = dict(np.load(frame_files[t]))

        # future action — t+1 ~ t+K_ACTION
        fut_idx = np.arange(t + 1, t + 1 + K_ACTION)
        data["future_steering"] = experts[fut_idx, 0]
        data["future_throttle"] = experts[fut_idx, 1]
        data["future_brake"]    = experts[fut_idx, 2]

        # future trajectory — t+STRIDE, t+2*STRIDE, ..., t+K_TRAJ*STRIDE
        traj_idx = t + np.arange(1, K_TRAJ + 1) * TRAJ_STRIDE  # [t+5, t+10, ..., t+30]

        cur_x = egos[t, 0]
        cur_y = egos[t, 1]

        data["future_traj_x"]  = egos[traj_idx, 0] - cur_x
        data["future_traj_y"]  = egos[traj_idx, 1] - cur_y

        speeds = egos[traj_idx, 4]   # m/s
        yaws   = egos[traj_idx, 3]   # rad
        data["future_traj_vx"] = speeds * np.cos(yaws)
        data["future_traj_vy"] = speeds * np.sin(yaws)

        out_path = out_frames / frame_files[t].name
        np.savez_compressed(str(out_path), **data)

    return valid_n


# ── 전체 데이터셋 처리 ────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="에피소드 후처리: future_* 필드 추가")
    parser.add_argument("--dataset_root", required=True,
                        help="수집 데이터 루트 (zone_*/scenario_*/episode_* 구조)")
    parser.add_argument("--out_root", default=None,
                        help="출력 경로 (기본값: dataset_root/../processed/)")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    out_root = (
        Path(args.out_root)
        if args.out_root
        else dataset_root.parent / (dataset_root.name + "_processed")
    )

    episode_dirs = sorted(p for p in dataset_root.rglob("episode_*")
                          if (p / "frames").exists())

    if not episode_dirs:
        print(f"에피소드를 찾지 못했습니다: {dataset_root}")
        return

    print(f"에피소드 {len(episode_dirs)}개 발견")
    print(f"출력 경로: {out_root}\n")

    total_valid  = 0
    total_frames = 0

    for ep_dir in episode_dirs:
        rel      = ep_dir.relative_to(dataset_root)
        out_ep   = out_root / rel
        raw_n    = len(list((ep_dir / "frames").glob("*.npz")))

        print(f"[{rel}]  원본 {raw_n}프레임")
        valid_n = process_episode(ep_dir, out_ep)

        total_frames += raw_n
        total_valid  += valid_n
        if valid_n > 0:
            dropped = raw_n - valid_n
            print(f"  → 저장 {valid_n}프레임  (마지막 {dropped}프레임 제외)")

    print(f"\n완료: {total_valid}/{total_frames}프레임 저장 ({out_root})")


if __name__ == "__main__":
    main()
