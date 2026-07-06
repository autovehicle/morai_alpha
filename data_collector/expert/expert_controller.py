"""
expert/expert_controller.py
----------------------------
GT_BEV Expert Controller.

GT_BEV 프로세스가 /ctrl_cmd 토픽에 퍼블리시한 CtrlCmd 메시지를
ROSManager가 구독하여 버퍼에 저장한다.
ExpertController.step()은 그 버퍼에서 최신 값을 읽어 반환한다.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class ControlOutput:
    steer:    float  # [-1.0, 1.0]
    throttle: float  # [0.0,  1.0]
    brake:    float  # [0.0,  1.0]


class ExpertController:

    def __init__(self, config: dict = None, ros_manager=None):
        self._ros = ros_manager

    def reset(self, waypoints: np.ndarray):
        pass

    def step(self, ego, tl_states: List, dt: float = 0.1) -> ControlOutput:
        """GT_BEV가 /ctrl_cmd 로 퍼블리시한 최신 제어값을 반환."""
        if self._ros is not None:
            with self._ros._lock:
                return ControlOutput(
                    steer    = self._ros._expert_steer,
                    throttle = self._ros._expert_throttle,
                    brake    = self._ros._expert_brake,
                )
        return ControlOutput(steer=0.0, throttle=0.0, brake=0.0)

    @staticmethod
    def detect_longtail(ego, prev_speed: float, dt: float,
                        tl_states: List, triggers: List[str]) -> bool:
        return False
