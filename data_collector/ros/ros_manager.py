"""
ros/ros_manager.py
------------------
Windows ROS1 Noetic 환경에서 MORAI 센서 토픽을 구독.

구독 토픽:
    - 카메라 5개 뷰 (sensor_msgs/Image)
    - GNSS (morai_msgs/GPSMessage 또는 sensor_msgs/NavSatFix)
    - IMU  (sensor_msgs/Imu)
    - GT 객체 (morai_msgs/ObjectStatusList)
    - GT 차선 (morai_msgs/LaneInfo 또는 커스텀)
    - 신호등 (morai_msgs/TrafficLightStatus)
    - Ego 상태 (morai_msgs/EgoVehicleStatus)

※ 실제 morai_msgs 필드명은 설치된 패키지에 따라 다를 수 있음.
  _cb_gnss, _cb_imu, _cb_objects, _cb_ego 콜백에서 필드 이름을 확인/수정 필요.
"""

import threading
import time
from collections import deque
from typing import Dict, Optional

import numpy as np

# Windows ROS1 환경에서는 python 경로에 ROS site-packages가 잡혀 있어야 함.
# collect.py에서 setup.bat 실행 후 환경변수를 설정한 상태로 import.
try:
    import rospy
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image, CompressedImage, Imu, NavSatFix, PointCloud2
    from sensor_msgs import point_cloud2
    # MORAI 전용 메시지 (설치된 morai_msgs 패키지 기준)
    from morai_msgs.msg import (
        EgoVehicleStatus,
        ObjectStatusList,
        GPSMessage,
        GetTrafficLightStatus,
        CtrlCmd,
    )
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    rospy = None

from core.data_writer import (
    CameraFrame, GNSSData, IMUData, EgoState, LidarFrame,
    GTObject, GTLane, TrafficLightState, SensorSnapshot
)


class ROSManager:
    """
    ROS 토픽 구독 및 최신 데이터 버퍼 관리.

    get_snapshot()을 호출하면 현재 버퍼에 있는 최신 데이터를
    타임스탬프 기준으로 동기화해 SensorSnapshot으로 반환.
    """

    CAMERA_KEYS = ["front", "left", "right", "back"]
    SYNC_WINDOW_NS = 20 * 1_000_000   # 20ms → ns

    def __init__(self, config: dict):
        self.cfg    = config
        self.topics = config["ros"]["topics"]

        self._bridge = CvBridge() if ROS_AVAILABLE else None
        self._lock   = threading.Lock()

        # 최신 데이터 버퍼
        self._cameras: Dict[str, CameraFrame] = {}
        self._gnss:    Optional[GNSSData]     = None
        self._imu:     Optional[IMUData]      = None
        self._ego:     Optional[EgoState]     = None
        self._lidar:   Optional[LidarFrame]   = None
        self._gt_objects = []
        self._gt_lanes   = []
        self._tl_states  = []

        # 카메라 최근 타임스탬프 (동기화 기준)
        self._cam_ts: Dict[str, int] = {}

        self._expert_steer    = 0.0
        self._expert_throttle = 0.0
        self._expert_brake    = 0.0

        self._initialized = False

    # ── 초기화 / 종료 ────────────────────────────────────────

    def start(self, node_name: str = "morai_data_collector"):
        if not ROS_AVAILABLE:
            print("[ROSManager] WARNING: rospy not available. Mock mode.")
            self._initialized = True
            return

        rospy.init_node(node_name, anonymous=True, disable_signals=True)

        # 카메라 4개 구독
        cam_topic_keys = {
            "front": "camera_front",
            "left":  "camera_left",
            "right": "camera_right",
            "back":  "camera_back",
        }
        for key, topic_key in cam_topic_keys.items():
            topic = self.topics[topic_key]
            rospy.Subscriber(topic, CompressedImage,
                             lambda msg, k=key: self._cb_camera(msg, k),
                             queue_size=2)

        # GNSS (morai_msgs/GPSMessage 우선, 없으면 NavSatFix)
        try:
            rospy.Subscriber(self.topics["gnss"], GPSMessage,
                             self._cb_gnss_morai, queue_size=5)
        except Exception:
            rospy.Subscriber(self.topics["gnss"], NavSatFix,
                             self._cb_gnss_navsatfix, queue_size=5)

        # IMU
        rospy.Subscriber(self.topics["imu"], Imu,
                         self._cb_imu, queue_size=5)

        # LiDAR (VLP16, sensor_msgs/PointCloud2)
        lidar_topic = self.topics.get("lidar")
        if lidar_topic:
            rospy.Subscriber(lidar_topic, PointCloud2,
                             self._cb_lidar, queue_size=1)
        else:
            print("[ROSManager] WARNING: config에 ros.topics.lidar 미설정 — LiDAR 구독 안 함")

        # GT 객체
        rospy.Subscriber(self.topics["gt_objects"], ObjectStatusList,
                         self._cb_objects, queue_size=2)

        # 신호등
        rospy.Subscriber(self.topics["traffic_light"], GetTrafficLightStatus,
                         self._cb_traffic_light, queue_size=2)

        # Ego 상태
        rospy.Subscriber(self.topics["ego_state"], EgoVehicleStatus,
                         self._cb_ego, queue_size=5)

        # Expert 제어 출력 (GT_BEV가 퍼블리시하는 제어 명령)
        expert_topic = self.topics.get("expert_ctrl", "/ctrl_cmd")
        try:
            rospy.Subscriber(expert_topic, CtrlCmd,
                             self._cb_expert_ctrl, queue_size=5)
        except Exception as e:
            print(f"[ROSManager] expert_ctrl 구독 실패 (topic={expert_topic}): {e}")

        self._initialized = True
        print("[ROSManager] 구독 시작 완료")

        # 콜백 실행을 위한 spin 스레드
        self._spin_thread = threading.Thread(target=rospy.spin, daemon=True)
        self._spin_thread.start()

    def stop(self):
        if ROS_AVAILABLE and rospy.is_shutdown() is False:
            rospy.signal_shutdown("DataCollector 종료")

    # ── 스냅샷 반환 ──────────────────────────────────────────

    def get_snapshot(self, frame_id: int, is_longtail: bool,
                      tick_count: int = 0, sim_elapsed_ns: int = 0) -> Optional[SensorSnapshot]:
        """
        현재 버퍼에서 카메라 타임스탬프를 기준으로 동기화된 스냅샷 반환.
        카메라 데이터가 없으면 None 반환.
        """
        with self._lock:
            if not self._cameras:
                return None

            # 기준 타임스탬프: front 카메라
            ref_ts = self._cam_ts.get("front")
            if ref_ts is None:
                return None

            # GNSS/IMU는 ±20ms 이내 데이터만 사용
            gnss = self._gnss
            imu  = self._imu
            if gnss and abs(gnss.timestamp_ns - ref_ts) > self.SYNC_WINDOW_NS:
                gnss = None
            if imu and abs(imu.timestamp_ns - ref_ts) > self.SYNC_WINDOW_NS:
                imu = None

            # LiDAR는 최대 15Hz(권장 10Hz)로 카메라(10Hz)보다 느리거나 비슷해
            # GNSS/IMU용 ±20ms 윈도우를 그대로 적용하면 대부분 탈락한다.
            # ego/gt_objects와 동일하게 별도 staleness 체크 없이 최신 버퍼값을 사용.
            snap = SensorSnapshot(
                frame_id      = frame_id,
                timestamp_ns  = ref_ts,
                cameras       = dict(self._cameras),
                gnss          = gnss,
                imu           = imu,
                ego           = self._ego,
                lidar         = self._lidar,
                gt_objects    = list(self._gt_objects),
                gt_lanes      = list(self._gt_lanes),
                tl_states     = list(self._tl_states),
                is_longtail   = is_longtail,
                expert_steer    = self._expert_steer,
                expert_throttle = self._expert_throttle,
                expert_brake    = self._expert_brake,
                tick_count     = tick_count,
                sim_elapsed_ns = sim_elapsed_ns,
            )
        return snap

    def wait_for_data(self, timeout_sec: float = 5.0) -> bool:
        """최초 센서 데이터가 들어올 때까지 대기"""
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if self._cameras and self._ego:
                return True
            time.sleep(0.05)
        return False

    # ── ROS 콜백 ─────────────────────────────────────────────

    def _cb_camera(self, msg: "CompressedImage", key: str):
        try:
            import cv2
            np_arr = np.frombuffer(msg.data, np.uint8)
            img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            ts  = msg.header.stamp.secs * 1_000_000_000 + msg.header.stamp.nsecs
            with self._lock:
                self._cameras[key]  = CameraFrame(timestamp_ns=ts, image=img)
                self._cam_ts[key]   = ts
        except Exception as e:
            print(f"[ROSManager] 카메라({key}) 변환 오류: {e}")

    def _cb_gnss_morai(self, msg: "GPSMessage"):
        """
        morai_msgs/GPSMessage 기준.
        실제 필드명: msg.latitude, msg.longitude, msg.altitude
        속도 필드는 msg.velocity (float) 또는 msg.linear_velocity 등 확인 필요.
        """
        try:
            ts = msg.header.stamp.secs * 1_000_000_000 + msg.header.stamp.nsecs
            with self._lock:
                self._gnss = GNSSData(
                    timestamp_ns = ts,
                    latitude     = msg.latitude,
                    longitude    = msg.longitude,
                    altitude     = msg.altitude,
                    velocity_x   = getattr(msg, "linear_velocity", 0.0),
                    velocity_y   = 0.0,
                    velocity_z   = 0.0,
                )
        except Exception as e:
            print(f"[ROSManager] GNSS 콜백 오류: {e}")

    def _cb_gnss_navsatfix(self, msg: "NavSatFix"):
        try:
            ts = msg.header.stamp.secs * 1_000_000_000 + msg.header.stamp.nsecs
            with self._lock:
                self._gnss = GNSSData(
                    timestamp_ns = ts,
                    latitude     = msg.latitude,
                    longitude    = msg.longitude,
                    altitude     = msg.altitude,
                    velocity_x   = 0.0,
                    velocity_y   = 0.0,
                    velocity_z   = 0.0,
                )
        except Exception as e:
            print(f"[ROSManager] NavSatFix 콜백 오류: {e}")

    def _cb_imu(self, msg: "Imu"):
        try:
            ts = msg.header.stamp.secs * 1_000_000_000 + msg.header.stamp.nsecs
            with self._lock:
                self._imu = IMUData(
                    timestamp_ns = ts,
                    accel_x = msg.linear_acceleration.x,
                    accel_y = msg.linear_acceleration.y,
                    accel_z = msg.linear_acceleration.z,
                    gyro_x  = msg.angular_velocity.x,
                    gyro_y  = msg.angular_velocity.y,
                    gyro_z  = msg.angular_velocity.z,
                )
        except Exception as e:
            print(f"[ROSManager] IMU 콜백 오류: {e}")

    def _cb_lidar(self, msg: "PointCloud2"):
        """
        sensor_msgs/PointCloud2 기준 (VLP16).
        필드: x, y, z, intensity (센서 좌표계, m / 반사강도).
        """
        try:
            ts = msg.header.stamp.secs * 1_000_000_000 + msg.header.stamp.nsecs
            pts = np.array(list(point_cloud2.read_points(
                msg, field_names=("x", "y", "z", "intensity"), skip_nans=True
            )), dtype=np.float32)
            if pts.size == 0:
                pts = np.zeros((0, 4), dtype=np.float32)
            with self._lock:
                self._lidar = LidarFrame(timestamp_ns=ts, points=pts)
        except Exception as e:
            print(f"[ROSManager] LiDAR 콜백 오류: {e}")

    def _cb_objects(self, msg: "ObjectStatusList"):
        """
        morai_msgs/ObjectStatusList 기준.
        msg.npc_list: ObjectStatus[] (차량)
        msg.pedestrian_list: ObjectStatus[] (보행자)
        msg.obstacle_list: ObjectStatus[] (정적 장애물)
        각 리스트가 이미 종류별로 분리돼서 오므로, 리스트 소속으로 분류한다.
        """
        try:
            objects = []
            # NPC 차량
            for npc in getattr(msg, "npc_list", []):
                objects.append(GTObject(
                    obj_id   = npc.unique_id,
                    obj_type = "vehicle",
                    x        = npc.position.x,
                    y        = npc.position.y,
                    z        = npc.position.z,
                    vel_x    = npc.velocity.x,
                    vel_y    = npc.velocity.y,
                    heading  = getattr(npc, "heading", 0.0),
                ))
            # 보행자
            for ped in getattr(msg, "pedestrian_list", []):
                objects.append(GTObject(
                    obj_id   = ped.unique_id,
                    obj_type = "pedestrian",
                    x        = ped.position.x,
                    y        = ped.position.y,
                    z        = ped.position.z,
                    vel_x    = ped.velocity.x,
                    vel_y    = ped.velocity.y,
                    heading  = getattr(ped, "heading", 0.0),
                ))
            # 정적 장애물
            for obs in getattr(msg, "obstacle_list", []):
                objects.append(GTObject(
                    obj_id   = getattr(obs, "unique_id", -1),
                    obj_type = "static",
                    x        = obs.position.x,
                    y        = obs.position.y,
                    z        = obs.position.z,
                    vel_x    = 0.0,
                    vel_y    = 0.0,
                    heading  = 0.0,
                ))
            with self._lock:
                self._gt_objects = objects
        except Exception as e:
            print(f"[ROSManager] GT Objects 콜백 오류: {e}")

    def _cb_traffic_light(self, msg: "GetTrafficLightStatus"):
        """
        morai_msgs/TrafficLightStatus 기준.
        msg.trafficLightIndex: 신호등 ID
        msg.trafficLightStatus: 상태값 (0~9)
        실제 필드명 확인 필요.
        """
        try:
            with self._lock:
                self._tl_states = [TrafficLightState(
                    tl_id = str(getattr(msg, "trafficLightIndex", 0)),
                    state = int(getattr(msg, "trafficLightStatus", 0)),
                )]
        except Exception as e:
            print(f"[ROSManager] 신호등 콜백 오류: {e}")

    def _cb_expert_ctrl(self, msg: "CtrlCmd"):
        try:
            with self._lock:
                self._expert_steer    = float(getattr(msg, "steering", 0.0))
                self._expert_throttle = float(getattr(msg, "accel",       0.0))
                self._expert_brake    = float(getattr(msg, "brake",       0.0))
        except Exception as e:
            print(f"[ROSManager] expert_ctrl 콜백 오류: {e}")

    def _cb_ego(self, msg: "EgoVehicleStatus"):
        """
        morai_msgs/EgoVehicleStatus 기준.
        msg.position: geometry_msgs/Point
        msg.heading:  float (도 단위, rad 변환 필요)
        msg.velocity: float (km/h → m/s 변환 필요)
        msg.accel:    float
        실제 필드명 확인 필요.
        """
        try:
            ts = int(time.time() * 1e9)   # EgoVehicleStatus에 header 없는 경우 대비
            if hasattr(msg, "header"):
                ts = msg.header.stamp.secs * 1_000_000_000 + msg.header.stamp.nsecs

            heading_rad = msg.heading * (np.pi / 180.0)  # 도 → rad
            speed_mps   = float(np.sqrt(msg.velocity.x**2 + msg.velocity.y**2))  # 벡터 크기

            with self._lock:
                self._ego = EgoState(
                    timestamp_ns = ts,
                    x     = msg.position.x,
                    y     = msg.position.y,
                    z     = msg.position.z,
                    yaw   = heading_rad,
                    speed = speed_mps,
                    steer = getattr(msg, "wheel_angle", 0.0),
                )
        except Exception as e:
            print(f"[ROSManager] Ego 콜백 오류: {e}")