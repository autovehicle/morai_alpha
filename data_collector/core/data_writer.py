"""
core/data_writer.py
-------------------
센서 스냅샷을 타임스탬프 동기화 후 .npz 프레임 파일로 저장.

폴더 구조:
  {root}/{zone}/{scenario}/episode_{NNN}/
      frames/
          000001.npz
          000002.npz
          ...
      episode_params.json
      episode_summary.json
"""

"""
1. 데이터 컨테이너 정의
CameraFrame     → 카메라 이미지 1장
GNSSData        → GPS 위치/속도
IMUData         → 가속도/각속도
EgoState        → 자차 위치/속도/조향
GTObject        → NPC 차량/보행자/장애물
GTLane          → 차선 정보
TrafficLightState → 신호등 상태
SensorSnapshot  → 위 전부를 한 타임스탬프에 묶은 것

2. BEV 맵 생성기
자차 중심 기준으로 30m x 20m 영역을 0.2m
해상도로 8채널 BEV 맵 생성.

3. DataWriter 클래스
- begin_episode(params) → 에피소드 폴더 생성 및 episode_params.json 저장
- write_frame(snapshot) → SensorSnapshot을 .npz 파일로 저장
- end_episode(success, reason) → episode_summary.json 저장 및 수집 현황 업데이트
"""

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from core.scenario_params import EpisodeParams


# ─────────────────────────────────────────────
# 데이터 컨테이너
# ─────────────────────────────────────────────

@dataclass
class CameraFrame:
    """카메라 1개 뷰 RGB 이미지 (H, W, 3) uint8"""
    timestamp_ns: int
    image: np.ndarray          # shape: (H, W, 3)


@dataclass
class GNSSData:
    timestamp_ns: int
    latitude: float
    longitude: float
    altitude: float
    velocity_x: float          # m/s, 차량 좌표계
    velocity_y: float
    velocity_z: float


@dataclass
class IMUData:
    timestamp_ns: int
    accel_x: float             # m/s²
    accel_y: float
    accel_z: float
    gyro_x: float              # rad/s
    gyro_y: float
    gyro_z: float


@dataclass
class EgoState:
    timestamp_ns: int
    x: float
    y: float
    z: float
    yaw: float                 # rad
    speed: float               # m/s
    steer: float               # rad


@dataclass
class LidarFrame:
    """VLP16 LiDAR 1스캔. points: (N, 4) — x, y, z(m, 센서 좌표계), intensity"""
    timestamp_ns: int
    points: np.ndarray         # shape: (N, 4)


@dataclass
class GTObject:
    obj_id: int
    obj_type: str              # "vehicle" | "pedestrian" | "static"
    x: float
    y: float
    z: float
    vel_x: float
    vel_y: float
    heading: float


@dataclass
class GTLane:
    lane_id: str
    boundary_left: np.ndarray  # shape: (N, 2)  x,y 좌표 배열
    boundary_right: np.ndarray
    lane_type: str             # "solid" | "dashed" | "center_yellow"


@dataclass
class TrafficLightState:
    tl_id: str
    state: int                 # 0~9 클래스


@dataclass
class SensorSnapshot:
    """매 타임스텝에 동기화된 전체 센서 묶음"""
    frame_id: int
    timestamp_ns: int

    cameras: Dict[str, CameraFrame]     # key: "front", "front_left", ...
    gnss: Optional[GNSSData]
    imu: Optional[IMUData]
    ego: Optional[EgoState]

    lidar: Optional[LidarFrame] = None
    gt_objects: List[GTObject] = field(default_factory=list)
    gt_lanes: List[GTLane]     = field(default_factory=list)
    tl_states: List[TrafficLightState] = field(default_factory=list)

    expert_steer: float    = 0.0
    expert_throttle: float = 0.0
    expert_brake: float    = 0.0

    nav_waypoints: Optional[np.ndarray] = None   # shape: (N, 2)
    nav_link_ids:  Optional[List[str]]  = None

    bev_map: Optional[np.ndarray] = None         # shape: (H, W, 8)

    is_longtail: bool = False
    tick_count: int = 0          # gRPC SyncTimestamp.frame_count
    sim_elapsed_ns: int = 0      # gRPC SyncTimestamp.elapsed_time


# ─────────────────────────────────────────────
# BEV 맵 생성기
# ─────────────────────────────────────────────

class BEVMapGenerator:
    """
    자차 중심 기준으로 8채널 BEV 맵을 생성.

    영역: 전방 30m / 후방 10m / 좌우 20m
    해상도: 0.2m/px → 200px(세로) × 200px(가로)

    채널:
        0: background
        1: center yellow line
        2: solid line
        3: dashed line
        4: stop line
        5: dynamic objects (vehicles, pedestrians)
        6: drivable area
        7: crosswalk
    """

    FRONT_M  = 30.0
    REAR_M   = 10.0
    SIDE_M   = 20.0
    RES_M    = 0.2     # 픽셀당 m
    LANE_RANGE_M = 50.0  # 차선 검색 반경

    def __init__(self, map_dir: str = None):
        self.h = int((self.FRONT_M + self.REAR_M) / self.RES_M)   # 200
        self.w = int((self.SIDE_M * 2) / self.RES_M)              # 200

        # MGeo 차선/정지선 데이터 (map_dir 지정 시 로드)
        self._lane_data: List[dict] = []      # {points, channel}
        self._stopline_data: List[dict] = []  # {points}
        if map_dir:
            self._load_lane_data(map_dir)

    # ── MGeo 차선 데이터 로드 ─────────────────────────────────

    def _load_lane_data(self, map_dir: str):
        map_path = Path(map_dir)

        # lane_marking_set.json → 채널 1(황색실선) / 2(실선) / 3(점선)
        lane_json = map_path / "lane_marking_set.json"
        if lane_json.exists():
            with open(lane_json, "r", encoding="utf-8") as f:
                items = json.load(f)
            for item in items:
                pts = np.asarray(item.get("points", []), dtype=np.float32)
                if pts.ndim != 2 or pts.shape[1] < 2 or pts.shape[0] < 2:
                    continue
                if item.get("lane_type") == 530:  # 경계선 제외
                    continue
                ch = self._lane_channel(item)
                self._lane_data.append({"points": pts, "channel": ch})
            print(f"[BEVMapGenerator] lane_marking_set 로드: {len(self._lane_data)}개")
        else:
            print(f"[BEVMapGenerator] WARNING: {lane_json} 없음, 차선 채널 비어있음")

        # stoplane_marking_set.json → 채널 4(정지선)
        stop_json = map_path / "stoplane_marking_set.json"
        if stop_json.exists():
            with open(stop_json, "r", encoding="utf-8") as f:
                items = json.load(f)
            for item in items:
                pts = np.asarray(item.get("points", []), dtype=np.float32)
                if pts.ndim != 2 or pts.shape[1] < 2 or pts.shape[0] < 2:
                    continue
                self._stopline_data.append({"points": pts})
            print(f"[BEVMapGenerator] stoplane_marking_set 로드: {len(self._stopline_data)}개")
        else:
            print(f"[BEVMapGenerator] WARNING: {stop_json} 없음, 정지선 채널 비어있음")

    @staticmethod
    def _lane_channel(item: dict) -> int:
        """lane_color + lane_shape → BEV 채널 번호"""
        color = item.get("lane_color", "").lower()
        shape_list = item.get("lane_shape", [])
        shape = shape_list[0].lower() if shape_list else "solid"
        if "yellow" in color:
            return 1  # 중앙 황색선
        if shape in ("broken", "dashed", "dash"):
            return 3  # 점선
        return 2      # 실선

    # ── BEV 생성 ─────────────────────────────────────────────

    def generate(self, ego: EgoState,
                 gt_objects: List[GTObject]) -> np.ndarray:
        """반환: (H, W, 8) float32 배열 (각 채널 0 또는 1)"""
        bev = np.zeros((self.h, self.w, 8), dtype=np.float32)

        # 채널 6: 전체 주행 가능 영역
        bev[:, :, 6] = 1.0

        # 채널 1~3: MGeo 차선 마킹
        for lane in self._lane_data:
            self._draw_mgeo_polyline(bev, ego, lane["points"], lane["channel"])

        # 채널 4: MGeo 정지선
        for sl in self._stopline_data:
            self._draw_mgeo_polyline(bev, ego, sl["points"], channel=4)

        # 채널 5: 동적 객체
        for obj in gt_objects:
            if obj.obj_type in ("vehicle", "pedestrian"):
                self._draw_object(bev, ego, obj)

        # 채널 0: background
        occupied = bev[:, :, 1:].sum(axis=2) > 0
        bev[~occupied, 0] = 1.0

        return bev

    def get_nearby_lane_geometry(self, ego: EgoState, radius: float = None):
        """
        ego 주변 차선/정지선 원본 좌표(월드 좌표계)를 벡터 형태로 반환.
        각 폴리라인의 점마다 한 행씩 뽑아서 (M, 4)/(K, 3) 고정 shape 배열로 만든다
        (npz는 ragged 배열을 그대로 저장할 수 없어서 flat 표현 사용).

        반환:
            lane_geometry:     (M, 4) float32  [lane_idx, channel, x, y]
            stopline_geometry: (K, 3) float32  [stopline_idx, x, y]
        """
        radius = radius if radius is not None else self.LANE_RANGE_M

        lane_rows = []
        for idx, lane in enumerate(self._lane_data):
            pts = lane["points"]
            dist = np.sqrt((pts[:, 0] - ego.x) ** 2 + (pts[:, 1] - ego.y) ** 2)
            near = pts[dist <= radius]
            if near.shape[0] < 2:
                continue
            ch = lane["channel"]
            for x, y in near[:, :2]:
                lane_rows.append((idx, ch, x, y))

        stop_rows = []
        for idx, sl in enumerate(self._stopline_data):
            pts = sl["points"]
            dist = np.sqrt((pts[:, 0] - ego.x) ** 2 + (pts[:, 1] - ego.y) ** 2)
            near = pts[dist <= radius]
            if near.shape[0] < 2:
                continue
            for x, y in near[:, :2]:
                stop_rows.append((idx, x, y))

        lane_geometry = np.array(lane_rows, dtype=np.float32) if lane_rows else np.zeros((0, 4), dtype=np.float32)
        stopline_geometry = np.array(stop_rows, dtype=np.float32) if stop_rows else np.zeros((0, 3), dtype=np.float32)
        return lane_geometry, stopline_geometry

    def _world_to_bev(self, ego: EgoState, wx: float, wy: float):
        """월드 좌표 1개 → BEV 픽셀 좌표 (행, 열)"""
        dx = wx - ego.x
        dy = wy - ego.y
        cos_y = np.cos(-ego.yaw)
        sin_y = np.sin(-ego.yaw)
        local_x =  cos_y * dx - sin_y * dy
        local_y =  sin_y * dx + cos_y * dy
        row = int((self.FRONT_M - local_x) / self.RES_M)
        col = int((self.SIDE_M  - local_y) / self.RES_M)
        return row, col

    def _world_pts_to_bev(self, ego: EgoState,
                          pts: np.ndarray) -> np.ndarray:
        """월드 좌표 배열 (N, 2+) → cv2용 픽셀 배열 (N, 1, 2) [col, row]"""
        dx = pts[:, 0] - ego.x
        dy = pts[:, 1] - ego.y
        cos_y = np.cos(-ego.yaw)
        sin_y = np.sin(-ego.yaw)
        local_x = cos_y * dx - sin_y * dy
        local_y = sin_y * dx + cos_y * dy
        rows = (self.FRONT_M - local_x) / self.RES_M
        cols = (self.SIDE_M  - local_y) / self.RES_M
        # cv2.polylines 는 (N, 1, 2) int32, 순서는 (x=col, y=row)
        return np.stack([cols, rows], axis=1).reshape(-1, 1, 2).astype(np.int32)

    def _draw_mgeo_polyline(self, bev: np.ndarray, ego: EgoState,
                            pts: np.ndarray, channel: int):
        """MGeo 포인트 배열을 ego 반경 필터 후 BEV 채널에 그린다."""
        dist = np.sqrt((pts[:, 0] - ego.x) ** 2 + (pts[:, 1] - ego.y) ** 2)
        pts_near = pts[dist <= self.LANE_RANGE_M]
        if pts_near.shape[0] < 2:
            return
        pts_bev = self._world_pts_to_bev(ego, pts_near)
        ch_img = (bev[:, :, channel] * 255).astype(np.uint8)
        cv2.polylines(ch_img, [pts_bev], isClosed=False, color=255, thickness=1)
        bev[:, :, channel] = ch_img.astype(np.float32) / 255.0

    def _draw_lane(self, bev: np.ndarray, ego: EgoState, lane: GTLane):
        lane_type_to_ch = {
            "center_yellow": 1,
            "solid":         2,
            "dashed":        3,
            "stop_line":     4,
            "crosswalk":     7,
        }
        ch = lane_type_to_ch.get(lane.lane_type, 2)

        for pts in [lane.boundary_left, lane.boundary_right]:
            if pts is None or len(pts) == 0:
                continue
            for pt in pts:
                r, c = self._world_to_bev(ego, pt[0], pt[1])
                if 0 <= r < self.h and 0 <= c < self.w:
                    bev[r, c, ch] = 1.0

    def _draw_object(self, bev: np.ndarray, ego: EgoState, obj: GTObject):
        r, c = self._world_to_bev(ego, obj.x, obj.y)
        # 객체 중심 ±2픽셀 박스
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                rr, cc = r + dr, c + dc
                if 0 <= rr < self.h and 0 <= cc < self.w:
                    bev[rr, cc, 5] = 1.0


# ─────────────────────────────────────────────
# DataWriter
# ─────────────────────────────────────────────

class DataWriter:
    """
    에피소드 시작/종료 및 프레임 단위 .npz 저장을 담당.

    사용 예시:
        writer = DataWriter(config)
        writer.begin_episode(params)
        writer.write_frame(snapshot)
        ...
        summary = writer.end_episode(success=True, reason="goal_reached")
    """

    CAMERA_KEYS = ["front", "left", "right", "back"]

    def __init__(self, config: dict):
        self.root      = Path(config["dataset"]["root_dir"])
        self.bev_gen   = BEVMapGenerator(map_dir=config.get("map_dir"))

        self._ep_dir: Optional[Path] = None
        self._frame_dir: Optional[Path] = None
        self._frame_id: int = 0
        self._ep_params: Optional[EpisodeParams] = None
        self._ep_start_time: float = 0.0

        # 수집 현황 (zone/scenario별 카운트)
        self._summary: Dict[str, Dict[str, int]] = {}

    # ── 에피소드 관리 ───────────────────────────────────────

    def begin_episode(self, params: EpisodeParams) -> Path:
        ep_str   = f"episode_{params.episode_id:03d}"
        zone_str = f"zone_{params.zone}"
        scen_str = f"scenario_{params.scenario}"

        scen_dir = self.root / zone_str / scen_str
        if params.run_id:
            scen_dir = scen_dir / f"scenario_run_{params.run_id:03d}"

        self._ep_dir    = scen_dir / ep_str
        self._frame_dir = self._ep_dir / "frames"
        self._frame_dir.mkdir(parents=True, exist_ok=True)

        self._frame_id      = 0
        self._ep_params     = params
        self._ep_start_time = time.time()

        # episode_params.json 저장
        params_path = self._ep_dir / "episode_params.json"
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(params.to_dict(), f, indent=2, ensure_ascii=False)

        return self._ep_dir

    def end_episode(self, success: bool, reason: str) -> dict:
        elapsed = time.time() - self._ep_start_time
        summary = {
            "success":     success,
            "reason":      reason,
            "total_frames": self._frame_id,
            "duration_sec": round(elapsed, 2),
            "zone":        self._ep_params.zone,
            "scenario":    self._ep_params.scenario,
            "episode_id":  self._ep_params.episode_id,
        }

        summary_path = self._ep_dir / "episode_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # 전체 현황 업데이트
        key = f"{self._ep_params.zone}/{self._ep_params.scenario}"
        self._summary.setdefault(key, {"total": 0, "success": 0, "frames": 0})
        self._summary[key]["total"]   += 1
        self._summary[key]["success"] += int(success)
        self._summary[key]["frames"]  += self._frame_id

        return summary

    def episode_exists(self, zone: str, scenario: str, episode_id: int) -> bool:
        """--resume 옵션 시 이미 완료된 에피소드 스킵용"""
        ep_str   = f"episode_{episode_id:03d}"
        zone_str = f"zone_{zone}"
        scen_str = f"scenario_{scenario}"
        summary  = self.root / zone_str / scen_str / ep_str / "episode_summary.json"
        return summary.exists()

    # ── 프레임 저장 ─────────────────────────────────────────

    def write_frame(self, snap: SensorSnapshot):
        """SensorSnapshot 하나를 .npz 1파일로 저장"""
        self._frame_id += 1
        fname = self._frame_dir / f"{self._frame_id:06d}.npz"

        # ── BEV 맵 생성 (ego + GT 정보 필요) ──────────────────
        if snap.ego is not None:
            snap.bev_map = self.bev_gen.generate(snap.ego, snap.gt_objects)
            lane_geometry, stopline_geometry = self.bev_gen.get_nearby_lane_geometry(snap.ego)
        else:
            lane_geometry = np.zeros((0, 4), dtype=np.float32)
            stopline_geometry = np.zeros((0, 3), dtype=np.float32)

        # ── 카메라 이미지 묶기 ─────────────────────────────────
        cameras = {}
        for key in self.CAMERA_KEYS:
            cam = snap.cameras.get(key)
            cameras[key] = cam.image if cam is not None else np.zeros((480, 640, 3), dtype=np.uint8)

        # ── GT 객체 배열 변환 (N, 7) ──────────────────────────────
        # [id, type_id, x, y, z, vel_x, vel_y]
        # type_id: 0=vehicle, 1=pedestrian, 2=static
        _TYPE_ID = {"vehicle": 0, "pedestrian": 1, "static": 2}

        gt_objects_arr = np.array([
            [o.obj_id, _TYPE_ID.get(o.obj_type, 2), o.x, o.y, o.z, o.vel_x, o.vel_y]
            for o in snap.gt_objects
        ], dtype=np.float32) if snap.gt_objects else np.zeros((0, 7), dtype=np.float32)

        # ── 신호등 상태 배열 ───────────────────────────────────
        # shape: (M, 2)  [tl_id_hash, state]
        if snap.tl_states:
            tl_arr = np.array([[
                hash(tl.tl_id) & 0xFFFF,
                tl.state
            ] for tl in snap.tl_states], dtype=np.int32)
        else:
            tl_arr = np.zeros((0, 2), dtype=np.int32)

        # ── ego state 배열 ─────────────────────────────────────
        ego = snap.ego
        ego_arr = np.array([
            ego.x, ego.y, ego.z, ego.yaw, ego.speed, ego.steer
        ], dtype=np.float32) if ego else np.zeros(6, dtype=np.float32)

        # ── GNSS 배열 ──────────────────────────────────────────
        gnss = snap.gnss
        gnss_arr = np.array([
            gnss.latitude, gnss.longitude, gnss.altitude,
            gnss.velocity_x, gnss.velocity_y, gnss.velocity_z
        ], dtype=np.float32) if gnss else np.zeros(6, dtype=np.float32)

        # ── IMU 배열 ───────────────────────────────────────────
        imu = snap.imu
        imu_arr = np.array([
            imu.accel_x, imu.accel_y, imu.accel_z,
            imu.gyro_x,  imu.gyro_y,  imu.gyro_z
        ], dtype=np.float32) if imu else np.zeros(6, dtype=np.float32)

        # ── LiDAR 포인트 배열 (N, 4): x, y, z, intensity ─────────
        lidar_arr = snap.lidar.points.astype(np.float32) if snap.lidar is not None \
                    else np.zeros((0, 4), dtype=np.float32)

        # ── nav waypoints / 도로 링크 ID ────────────────────────
        nav_wp = snap.nav_waypoints if snap.nav_waypoints is not None \
                 else np.zeros((0, 2), dtype=np.float32)
        nav_link_ids_arr = np.array(snap.nav_link_ids, dtype="<U32") if snap.nav_link_ids \
                            else np.zeros((0,), dtype="<U32")

        # ── expert 제어 ────────────────────────────────────────
        expert_arr = np.array([
            snap.expert_steer,
            snap.expert_throttle,
            snap.expert_brake
        ], dtype=np.float32)

        # ── .npz 저장 ──────────────────────────────────────────
        save_dict = {
            "timestamp_ns":   np.array([snap.timestamp_ns], dtype=np.int64),
            "frame_id":       np.array([snap.frame_id],     dtype=np.int32),
            "is_longtail":    np.array([snap.is_longtail],  dtype=bool),
            "tick_count":     np.array([snap.tick_count],     dtype=np.int64),
            "sim_elapsed_ns": np.array([snap.sim_elapsed_ns], dtype=np.int64),
            # 카메라
            "cam_front": cameras["front"],
            "cam_left":  cameras["left"],
            "cam_right": cameras["right"],
            "cam_back":  cameras["back"],
            # 센서
            "ego":     ego_arr,
            "gnss":    gnss_arr,
            "imu":     imu_arr,
            "lidar":   lidar_arr,       # (N,4): x, y, z, intensity
            # GT
            "gt_objects": gt_objects_arr,
            "tl_states":   tl_arr,
            # 경로
            "nav_waypoints": nav_wp,
            "nav_link_ids":  nav_link_ids_arr,
            # 차선/정지선 원본 지오메트리 (world 좌표, ego 반경 필터)
            "gt_lane_geometry":     lane_geometry,      # (M,4): [lane_idx, channel, x, y]
            "gt_stopline_geometry": stopline_geometry,  # (K,3): [stopline_idx, x, y]
            # expert 제어
            "expert": expert_arr,
            # BEV
            "bev_map": snap.bev_map if snap.bev_map is not None
                       else np.zeros((200, 200, 8), dtype=np.float32),
        }

        # 저장 도중 프로세스가 죽어도 fname 자체는 손상되지 않도록 임시 파일에 먼저 쓰고 원자적으로 교체
        # (파일 객체로 넘겨야 numpy가 파일명에 .npz를 추가로 덧붙이지 않음)
        tmp_fname = fname.with_suffix(".npz.tmp")
        with open(tmp_fname, "wb") as fh:
            np.savez_compressed(fh, **save_dict)
        os.replace(tmp_fname, fname)

    # ── 수집 현황 출력 ──────────────────────────────────────

    def print_summary(self):
        print("\n" + "=" * 55)
        print(f"{'Zone/Scenario':<35} {'에피소드':>6} {'성공':>5} {'프레임':>8}")
        print("-" * 55)
        for key, val in self._summary.items():
            print(f"{key:<35} {val['total']:>6} {val['success']:>5} {val['frames']:>8}")
        print("=" * 55)
