# %% [markdown]
# # NPZ 데이터 검수 스크립트 (VSCode Jupyter Interactive Window)
#
# 사용법:
#   1. VSCode에 Python + Jupyter 확장 설치 확인
#   2. 아래 EPISODE_DIR / FRAME_ID 를 확인하려는 episode 폴더/프레임 번호로 수정
#   3. 셀마다 "▶ Run Cell" 클릭 (또는 Shift+Enter) 순서대로 실행
#
# 이 프로젝트의 실제 npz 키 (data_writer.py 기준):
#   timestamp_ns, frame_id, is_longtail,
#   cam_front, cam_left, cam_right, cam_back,
#   ego(6), gnss(6), imu(6),
#   gt_objects(N,7), tl_states(M,2),
#   nav_waypoints(N,2), nav_link_ids(K,),
#   gt_lane_geometry(M,4), gt_stopline_geometry(K,3),
#   expert(3)=[steer,throttle,brake], bev_map(200,200,8)

# %%
import glob
import numpy as np
import matplotlib.pyplot as plt

EPISODE_DIR = "/mnt/z/AIM_2026/대회/2026 대학생 AI SW 자율주행 경진대회/Team A.I.M/zone_urban/scenario_sudden_brake/scenario_run_018/episode_001"
FRAME_ID = 18  # 8번 셀에서 자세히 볼 프레임 번호

frame_files = sorted(glob.glob(f"{EPISODE_DIR}/frames/*.npz"))
print(f"episode: {EPISODE_DIR}")
print(f"frame 개수: {len(frame_files)}")

# %% [markdown]
# ## 1. 키 / shape / dtype 목록

# %%
sample = np.load(frame_files[0], allow_pickle=True)
for k in sample.files:
    arr = sample[k]
    print(f"{k:20s} shape={str(arr.shape):15s} dtype={arr.dtype}")

# %% [markdown]
# ## 2. NaN/Inf 체크 + 프레임 길이 일관성 확인
# (전체 프레임을 순회하며 수치 배열에 NaN/Inf 있는지, gt_objects/tl_states 개수 범위 확인)

# %%
NUMERIC_KEYS = ["ego", "gnss", "imu", "expert", "nav_waypoints",
                "gt_lane_geometry", "gt_stopline_geometry"]

bad_frames = []
corrupt_frames = []
gt_obj_counts = []
valid_frame_files = []
for f in frame_files:
    try:
        d = np.load(f, allow_pickle=True)
        if "ego" not in d.files:
            raise KeyError("ego missing")
        for k in NUMERIC_KEYS:
            arr = d[k]
            if arr.size > 0 and (np.isnan(arr).any() or np.isinf(arr).any()):
                bad_frames.append((f.split("/")[-1], k))
        gt_obj_counts.append(d["gt_objects"].shape[0])
        valid_frame_files.append(f)
    except Exception as e:
        corrupt_frames.append((f.split("/")[-1], str(e)))

print(f"손상된 파일(키 누락 등): {corrupt_frames if corrupt_frames else '없음'}")
print(f"NaN/Inf 발견된 (파일, 키): {bad_frames if bad_frames else '없음'}")
print(f"gt_objects 개수 범위: min={min(gt_obj_counts)} max={max(gt_obj_counts)}")

# 이후 셀은 손상된 파일을 제외한 목록으로 진행
frame_files = valid_frame_files

# %% [markdown]
# ## 3. 카메라 멀티뷰 grid

# %%
d = np.load(frame_files[FRAME_ID - 1], allow_pickle=True)
cam_keys = [k for k in d.files if k.startswith("cam_")]

fig, axes = plt.subplots(1, len(cam_keys), figsize=(4 * len(cam_keys), 4))
for ax, key in zip(axes, cam_keys):
    ax.imshow(d[key])
    ax.set_title(key)
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. expert 행동 (steer/throttle/brake) 시계열
# ego[5]는 차량의 "현재" 조향각(상태)이고, expert는 "명령"값이라 다른 데이터입니다.

# %%
steers, throttles, brakes = [], [], []
for f in frame_files:
    e = np.load(f)["expert"]
    steers.append(e[0]); throttles.append(e[1]); brakes.append(e[2])

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(steers, label="steer")
ax.plot(throttles, label="throttle")
ax.plot(brakes, label="brake")
ax.set_xlabel("frame index")
ax.legend()
ax.set_title("expert action over episode")
plt.show()

# %% [markdown]
# ## 5. ego 궤적 vs nav_waypoints (XY 평면)

# %%
ego_xy = np.array([np.load(f)["ego"][:2] for f in frame_files])
last_nav_wp = np.load(frame_files[-1], allow_pickle=True)["nav_waypoints"]

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(ego_xy[:, 0], ego_xy[:, 1], "g.-", label="ego trajectory (실제 주행)")
if last_nav_wp.shape[0] > 0:
    ax.plot(last_nav_wp[:, 0], last_nav_wp[:, 1], "b.--", label="nav_waypoints (마지막 프레임 기준)")
ax.set_aspect("equal")
ax.legend()
ax.set_title("ego trajectory vs nav_waypoints")
plt.show()

# %% [markdown]
# ## 6. BEV map 클래스별 컬러 시각화 + 픽셀 비율
# 0:background 1:center_yellow 2:solid 3:dashed 4:stopline 5:dynamic 6:drivable 7:crosswalk

# %%
BEV_COLORS = {
    6: (60, 60, 60), 0: (30, 30, 30), 1: (255, 215, 0), 2: (255, 255, 255),
    3: (120, 220, 255), 4: (255, 60, 60), 7: (200, 120, 255), 5: (255, 140, 0),
}
DRAW_ORDER = [6, 0, 1, 2, 3, 4, 7, 5]
CLASS_NAMES = {0: "background", 1: "center_yellow", 2: "solid", 3: "dashed",
               4: "stopline", 5: "dynamic", 6: "drivable", 7: "crosswalk"}

bev = np.load(frame_files[FRAME_ID - 1])["bev_map"]
bev_rgb = np.zeros((*bev.shape[:2], 3), dtype=np.uint8)
for ch in DRAW_ORDER:
    bev_rgb[bev[:, :, ch] > 0.5] = BEV_COLORS[ch]

plt.figure(figsize=(5, 5))
plt.imshow(bev_rgb)
plt.title(f"bev_map (frame {FRAME_ID})")
plt.axis("off")
plt.show()

total_px = bev.shape[0] * bev.shape[1]
for ch in range(8):
    ratio = (bev[:, :, ch] > 0.5).sum() / total_px * 100
    print(f"class {ch} ({CLASS_NAMES[ch]:14s}): {ratio:5.2f}%")

# %% [markdown]
# ## 7. 신호등 상태 (10클래스) 시계열
# 0:OFF 1:RED 2:YELLOW 3:RED+YELLOW 4:GREEN 5:YELLOW+GREEN
# 6:GREENLEFT 7:RED+GREENLEFT 8:YELLOW+GREENLEFT 9:GREEN+GREENLEFT

# %%
tl_over_time = []
for f in frame_files:
    tl = np.load(f)["tl_states"]
    tl_over_time.append(int(tl[0, 1]) if tl.shape[0] > 0 else -1)  # -1 = 신호등 미감지

fig, ax = plt.subplots(figsize=(10, 3))
ax.step(range(len(tl_over_time)), tl_over_time, where="post")
ax.set_yticks(range(-1, 10))
ax.set_xlabel("frame index")
ax.set_ylabel("traffic light state (-1=미감지)")
ax.set_title("traffic light state over episode")
plt.show()

# %% [markdown]
# ## 8. 프레임 하나 종합 확인 (카메라 + BEV + action)

# %%
d = np.load(frame_files[FRAME_ID - 1], allow_pickle=True)
print(f"frame_id={int(d['frame_id'][0])}  is_longtail={bool(d['is_longtail'][0])}")
print(f"ego(x,y,z,yaw,speed,steer) = {d['ego']}")
print(f"gnss = {d['gnss']}")
print(f"expert(steer,throttle,brake) = {d['expert']}")
print(f"nav_link_ids = {list(d['nav_link_ids'])}")
print(f"gt_objects count = {d['gt_objects'].shape[0]}")

fig, axes = plt.subplots(1, len(cam_keys) + 1, figsize=(4 * (len(cam_keys) + 1), 4))
for ax, key in zip(axes[:-1], cam_keys):
    ax.imshow(d[key])
    ax.set_title(key)
    ax.axis("off")
axes[-1].imshow(bev_rgb)
axes[-1].set_title("bev_map")
axes[-1].axis("off")
plt.tight_layout()
plt.show()
