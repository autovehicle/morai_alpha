"""
episode_manager.py
------------------
한 episode 실행 담당.

역할:
1. MoraiClient로 MORAI world 시작 / reset
2. ego, NPC 시나리오 배치
3. ROSManager에서 센서 데이터 읽기
4. ExpertController로 expert action 계산
5. DataWriter로 frame 저장
6. collect.py에 summary 반환
"""

import logging
import time
import traceback
from typing import Any, Dict, Optional


logger = logging.getLogger("episode_manager")


class EpisodeManager:
    def __init__(self, ros_mgr, client, expert, writer, config):
        self.ros_mgr = ros_mgr
        self.client = client
        self.expert = expert
        self.writer = writer
        self.config = config

        collection_cfg = config.get("collection", {})

        self.default_duration_sec = float(
            collection_cfg.get("episode_duration_sec", 20.0)
        )
        self.save_hz = float(
            collection_cfg.get("save_hz", 10.0)
        )
        self.warmup_sec = float(
            collection_cfg.get("warmup_sec", 1.0)
        )

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        collect.py에서 호출하는 메인 함수.

        params는 ScenarioParamGenerator가 만든 episode 설정값.
        최소한 아래 구조가 들어있어야 함.

        params = {
            "zone": "urban",
            "scenario": "stop_and_go",
            "episode_id": 1,
            "ego": {
                "x": 53.0,
                "y": 1.2,
                "z": 0.0,
                "yaw": 0.0
            },
            "npcs": [
                {"x": 70.0, "y": 1.2, "z": 0.0, "yaw": 0.0, "speed": 20.0}
            ],
            "duration_sec": 20.0
        }
        """

        frames = 0

        try:
            zone = params.get("zone", "unknown_zone")
            scenario = params.get("scenario", "unknown_scenario")
            episode_id = params.get("episode_id", params.get("ep_id", 0))

            logger.info(
                f"episode 시작: zone={zone}, scenario={scenario}, ep={episode_id}"
            )

            # 1. expert 초기화
            self._safe_call(
                self.expert,
                ["reset", "start_episode", "begin_episode"],
                params,
            )

            # 2. MORAI episode 시작
            # 여기서 morai_client.py 안의 start_episode(params)가 호출됨
            self.client.start_episode(params)

            # 3. ROS topic 안정화 대기
            if self.warmup_sec > 0:
                time.sleep(self.warmup_sec)

            # 4. writer episode 시작
            self._safe_call(
                self.writer,
                ["start_episode", "begin_episode", "open_episode"],
                params,
            )

            duration_sec = float(params.get("duration_sec", self.default_duration_sec))
            dt = 1.0 / max(self.save_hz, 1.0)

            start_time = time.time()
            frame_id = 0

            while time.time() - start_time < duration_sec:
                # 5. ROS에서 현재 센서 frame 읽기
                frame = self._get_ros_frame()

                if frame is None:
                    time.sleep(dt)
                    continue

                if not isinstance(frame, dict):
                    frame = {"raw": frame}

                frame["frame_id"] = frame_id
                frame["timestamp_ns"] = time.time_ns()
                frame["zone"] = zone
                frame["scenario"] = scenario
                frame["episode_id"] = episode_id

                # 6. expert action 계산
                expert_action = self._get_expert_action(params, frame)

                if expert_action is not None:
                    frame["expert"] = expert_action
                    self._apply_expert_action(expert_action)

                # 7. frame 저장
                self._write_frame(params, frame, frame_id)

                frames += 1
                frame_id += 1

                time.sleep(dt)

            # 8. writer episode 종료
            self._safe_call(
                self.writer,
                ["end_episode", "finish_episode", "close_episode"],
                params,
            )

            logger.info(
                f"episode 완료: zone={zone}, scenario={scenario}, ep={episode_id}, frames={frames}"
            )

            return {
                "success": True,
                "frames": frames,
                "reason": "done",
            }

        except Exception as e:
            logger.error(f"episode 실행 실패: {e}")
            logger.error(traceback.format_exc())

            return {
                "success": False,
                "frames": frames,
                "reason": str(e),
            }

    # ------------------------------------------------------------------
    # 아래는 ros_manager / writer / expert 함수명이 조금 달라도 버티게 만든 보조 함수들
    # ------------------------------------------------------------------

    def _safe_call(self, obj, method_names, *args, default=None):
        """
        obj 안에서 method_names 중 존재하는 첫 번째 함수를 호출.
        없으면 default 반환.
        """
        if obj is None:
            return default

        for name in method_names:
            fn = getattr(obj, name, None)
            if callable(fn):
                try:
                    return fn(*args)
                except TypeError:
                    try:
                        return fn()
                    except TypeError:
                        continue

        return default

    def _get_ros_frame(self) -> Optional[Any]:
        """
        ROSManager에서 현재 센서 데이터를 가져옴.
        ros_manager.py의 실제 함수명이 뭐든 어느 정도 대응하도록 여러 이름 시도.
        """
        method_names = [
            "get_latest_frame",
            "get_frame",
            "get_snapshot",
            "get_latest",
            "read_frame",
            "collect_frame",
            "get_data",
            "snapshot",
        ]

        return self._safe_call(
            self.ros_mgr,
            method_names,
            default=None,
        )

    def _get_expert_action(self, params: Dict[str, Any], frame: Dict[str, Any]):
        """
        ExpertController에서 expert action 계산.
        expert_controller.py 함수명에 맞춰 여러 후보를 시도.
        """
        if self.expert is None:
            return None

        method_names = [
            "compute_action",
            "get_action",
            "act",
            "step",
            "control",
            "run",
        ]

        for name in method_names:
            fn = getattr(self.expert, name, None)
            if not callable(fn):
                continue

            # 함수 인자 형태가 다를 수 있어서 순서대로 시도
            for args in [
                (params, frame),
                (frame, params),
                (frame,),
                (params,),
                tuple(),
            ]:
                try:
                    return fn(*args)
                except TypeError:
                    continue

        return None

    def _apply_expert_action(self, action: Any):
        """
        expert action을 MORAI ego 제어로 보냄.
        MoraiClient에 해당 함수가 있으면 호출.
        없으면 그냥 넘어감.
        """
        method_names = [
            "apply_control",
            "control_ego",
            "send_control",
            "set_ego_control",
            "control",
        ]

        self._safe_call(
            self.client,
            method_names,
            action,
            default=None,
        )

    def _write_frame(self, params: Dict[str, Any], frame: Dict[str, Any], frame_id: int):
        """
        DataWriter로 frame 저장.
        data_writer.py 함수명에 맞춰 여러 후보를 시도.
        """
        if self.writer is None:
            return None

        method_names = [
            "write_frame",
            "save_frame",
            "write",
            "save",
            "append_frame",
        ]

        for name in method_names:
            fn = getattr(self.writer, name, None)
            if not callable(fn):
                continue

            for args in [
                (params, frame, frame_id),
                (params, frame),
                (frame, params, frame_id),
                (frame, params),
                (frame,),
            ]:
                try:
                    return fn(*args)
                except TypeError:
                    continue

        logger.warning("DataWriter에 frame 저장 함수가 없음")
        return None