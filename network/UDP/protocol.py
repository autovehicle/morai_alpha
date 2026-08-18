"""
MORAI UDP 프로토콜 — 데이터 클래스 및 상수 정의.

MORAI SIM:Drive 24.R2 UDP 프로토콜 기반.
문서: https://help-morai-sim.scrollhelp.site/ko/morai-sim-drive/24.R2/udp-1
"""
import math
from dataclasses import dataclass
from typing import Iterable

# ═══════════════════════════════════════════════════════════════════
# 프로토콜 헤더 (바이트 문자열)
# ═══════════════════════════════════════════════════════════════════
HEADER_EGO    = b'#MoraiInfo$'       # 11 bytes  — Ego Vehicle Status
HEADER_OBJ    = b'#MoraiObjInfo$'    # 14 bytes  — Object Info
HEADER_TL     = b'#TrafficLight$'    # 14 bytes  — Traffic Light (Get/Set 공용)
HEADER_CTRL   = b'#MoraiCtrlCmd$'    # 14 bytes  — Ego Ctrl Cmd

# ═══════════════════════════════════════════════════════════════════
# 객체 타입 코드  (ObjectData.obj_type)
# ═══════════════════════════════════════════════════════════════════
OBJ_TYPE_EGO        = -1   # Ego 차량 자신
OBJ_TYPE_PEDESTRIAN =  0   # 보행자
OBJ_TYPE_VEHICLE    =  1   # 차량 (NPC)
OBJ_TYPE_OBSTACLE   =  2   # 정적 장애물

# ═══════════════════════════════════════════════════════════════════
# 신호등 상태 코드  (TrafficLightData.status)
# ═══════════════════════════════════════════════════════════════════
TL_RED              =  1
TL_YELLOW           =  4
TL_RED_YELLOW       =  5
TL_GREEN            = 16
TL_YELLOW_GREEN     = 20
TL_GREEN_LEFT       = 32
TL_YELLOW_GREEN_LEFT = 36
TL_GREEN_GREEN_LEFT = 48
TL_DEFAULT          = -1

# ═══════════════════════════════════════════════════════════════════
# 제어 명령 상수
# ═══════════════════════════════════════════════════════════════════
CTRL_MODE_KEYBOARD = 1
CTRL_MODE_AUTO     = 2
GEAR_PARK    = 1
GEAR_REVERSE = 2
GEAR_NEUTRAL = 3
GEAR_DRIVE   = 4
CMD_TYPE_THROTTLE     = 1
CMD_TYPE_VELOCITY     = 2
CMD_TYPE_ACCELERATION = 3


# ═══════════════════════════════════════════════════════════════════
# Timestamp helper
# ═══════════════════════════════════════════════════════════════════

def timestamp_to_ns(timestamp_sec: float, timestamp_nsec: float) -> int:
    """Convert MORAI timestamp fields into a validated int64-compatible value.

    The source fields are float32 on the wire, so this conversion cannot create
    precision that is absent from the packet. ``round`` avoids a systematic
    downward truncation when a representable value lands just below an integer.
    """
    timestamp_sec = float(timestamp_sec)
    timestamp_nsec = float(timestamp_nsec)
    if not math.isfinite(timestamp_sec) or not math.isfinite(timestamp_nsec):
        raise ValueError('MORAI timestamp fields must be finite')
    value = int(round(timestamp_sec * 1e9 + timestamp_nsec))
    if value < -(2 ** 63) or value > (2 ** 63 - 1):
        raise OverflowError('MORAI timestamp does not fit int64')
    return value


# ═══════════════════════════════════════════════════════════════════
# 데이터 클래스
# ═══════════════════════════════════════════════════════════════════

@dataclass
class EgoState:
    """Ego Vehicle Status (SIM → User)

    좌표계: MORAI 전역 좌표 (UTM-like, 미터)
    각도  : degrees
    속도  : km/h (vel_x/y/z), signed_vel
    """
    timestamp_ns: int = 0       # ns, simulation timestamp
    # ── 위치 ──
    pos_x: float = 0.0          # m
    pos_y: float = 0.0          # m
    pos_z: float = 0.0          # m
    # ── 자세 ──
    roll:  float = 0.0          # deg
    pitch: float = 0.0          # deg
    yaw:   float = 0.0          # deg  (heading)
    # ── 속도 ──
    vel_x: float = 0.0          # km/h  (전방 속도 성분)
    vel_y: float = 0.0          # km/h
    vel_z: float = 0.0          # km/h
    signed_vel: float = 0.0     # km/h  (부호 포함 전방 속도)
    # ── 차량 치수 ──
    size_x: float = 0.0         # m  (전체 길이)
    size_y: float = 0.0         # m  (전체 너비)
    size_z: float = 0.0         # m  (전체 높이)
    overhang: float = 0.0       # m  (프론트 오버행)
    wheelbase: float = 0.0      # m
    rear_overhang: float = 0.0  # m
    # ── 제어 ──
    accel: float = 0.0          # 0 ~ 1
    brake: float = 0.0          # 0 ~ 1
    front_steer: float = 0.0   # deg  (전륜 조향각)
    ctrl_mode: int = 0          # 1=Keyboard, 2=AutoMode
    gear: int = 0               # 0=M, 1=P, 2=R, 3=N, 4=D, 5=L


@dataclass
class ObjectData:
    """Object Info — 주변 객체 1개 (SIM → User)
    MORAI는 ego 기준 가까운 순으로 최대 20개 객체를 전송한다.
    obj_id == 0 인 슬롯은 빈 슬롯이므로 무시해야 한다.

    BEV 렌더링에 필요한 핵심 필드:
      pos_x, pos_y    — 세계 좌표 위치 (m)
      heading          — 차량 heading (deg)
      size_x, size_y   — 전체 길이/너비 (m) → half-extent = size / 2
      obj_type          — 0=보행자, 1=차량, 2=장애물
    """
    timestamp_ns: int = 0       # ns, simulation timestamp
    obj_id:   int   = 0
    obj_type: int   = 0         # -1=Ego, 0=Ped, 1=Vehicle, 2=Obstacle
    pos_x:    float = 0.0       # m
    pos_y:    float = 0.0       # m
    pos_z:    float = 0.0       # m
    heading:  float = 0.0       # deg
    size_x:   float = 0.0       # m  (전체 길이)
    size_y:   float = 0.0       # m  (전체 너비)
    size_z:   float = 0.0       # m  (전체 높이)
    overhang: float = 0.0       # m
    wheelbase: float = 0.0      # m
    rear_overhang: float = 0.0  # m
    vel_x:    float = 0.0       # km/h
    vel_y:    float = 0.0       # km/h
    vel_z:    float = 0.0       # km/h
    acc_x:    float = 0.0       # m/s²
    acc_y:    float = 0.0       # m/s²
    acc_z:    float = 0.0       # m/s²


class ObjectFrame(list):
    """One Object Info packet with its packet-level simulation timestamp.

    This intentionally remains a ``list`` subclass for compatibility with the
    existing receiver callbacks and BEV code. Storing the timestamp only on
    each :class:`ObjectData` loses the timestamp when MORAI sends a valid frame
    containing zero objects.

    ``obj_id`` values inside the frame are source object IDs. This type does
    not claim that they are persistent tracks.
    """

    def __init__(self, objects: Iterable[ObjectData] = (), timestamp_ns: int = 0):
        super().__init__(objects)
        self.timestamp_ns = int(timestamp_ns)


@dataclass
class TrafficLightData:
    """Traffic Light Status (SIM → User)"""
    index:  str = ''            # 신호등 ID 문자열
    tl_type: int = 0            # 0=3등(R-Y-G), 1=3등(R-Y-GL), 2=4등, 100=황색전용
    status: int = TL_DEFAULT    # TL_* 상수 참조
