"""MORAI Camera UDP receiver and timestamp-aware JPEG reassembly.

Wire layout (MORAI SIM:Drive 24.R2):

    MOR | total_second | nanosecond | index | size | JPEG chunk | AI/EI

The simulator can split one JPEG frame across several UDP datagrams. This
module groups datagrams by source timestamp and rejects incomplete JPEGs.
"""

import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional

from .base import SensorFrame, SensorReceiver


CAMERA_HEADER = b'MOR'
CAMERA_TAIL_MORE = b'AI'
CAMERA_TAIL_END = b'EI'
CAMERA_FIXED_PREFIX_SIZE = 19
CAMERA_FIXED_SUFFIX_SIZE = 2
CAMERA_MIN_PACKET_SIZE = CAMERA_FIXED_PREFIX_SIZE + CAMERA_FIXED_SUFFIX_SIZE
CAMERA_MAX_DATAGRAM_SIZE = 65_535
INT64_MAX = (2 ** 63) - 1


class CameraPacketError(ValueError):
    """Raised when a datagram violates the MORAI Camera UDP layout."""


@dataclass(frozen=True)
class CameraPacket:
    timestamp_ns: int
    index: int
    declared_size: int
    jpeg_chunk: bytes
    is_final: bool


@dataclass(frozen=True)
class AssembledCameraFrame:
    timestamp_ns: int
    jpeg: bytes
    packet_count: int
    first_index: int
    final_index: int


@dataclass
class _PendingFrame:
    chunks: Dict[int, bytes]
    last_monotonic_ns: int
    final_index: Optional[int] = None


def parse_camera_packet(datagram: bytes) -> CameraPacket:
    """Parse one MORAI Camera UDP datagram.

    MORAI documents the timestamp as two 4-byte integer fields. The official
    receiver example uses native little-endian signed integers for index/size;
    explicit little-endian formats are used here for platform independence.
    """
    if len(datagram) < CAMERA_MIN_PACKET_SIZE:
        raise CameraPacketError('camera datagram is shorter than 21 bytes')
    if datagram[:3] != CAMERA_HEADER:
        raise CameraPacketError('invalid camera packet header')

    total_second, nanosecond = struct.unpack_from('<II', datagram, 3)
    if nanosecond >= 1_000_000_000:
        raise CameraPacketError('camera nanosecond field is out of range')
    timestamp_ns = total_second * 1_000_000_000 + nanosecond
    if timestamp_ns > INT64_MAX:
        raise CameraPacketError('camera timestamp does not fit int64')

    index, declared_size = struct.unpack_from('<ii', datagram, 11)
    if index < 0:
        raise CameraPacketError('camera packet index must be non-negative')
    if declared_size < 0:
        raise CameraPacketError('camera JPEG chunk size must be non-negative')

    jpeg_chunk = datagram[CAMERA_FIXED_PREFIX_SIZE:-CAMERA_FIXED_SUFFIX_SIZE]
    if declared_size != len(jpeg_chunk):
        raise CameraPacketError(
            'camera JPEG chunk size mismatch: declared {0}, received {1}'.format(
                declared_size, len(jpeg_chunk)
            )
        )

    tail = datagram[-CAMERA_FIXED_SUFFIX_SIZE:]
    if tail not in (CAMERA_TAIL_MORE, CAMERA_TAIL_END):
        raise CameraPacketError('invalid camera packet tail')

    return CameraPacket(
        timestamp_ns=timestamp_ns,
        index=index,
        declared_size=declared_size,
        jpeg_chunk=jpeg_chunk,
        is_final=(tail == CAMERA_TAIL_END),
    )


class CameraFrameAssembler:
    """Reassemble timestamped Camera UDP packets without hiding packet loss."""

    def __init__(self, assembly_timeout_ns: int = 1_000_000_000,
                 max_inflight_frames: int = 8):
        if assembly_timeout_ns <= 0:
            raise ValueError('assembly_timeout_ns must be positive')
        if max_inflight_frames <= 0:
            raise ValueError('max_inflight_frames must be positive')
        self.assembly_timeout_ns = int(assembly_timeout_ns)
        self.max_inflight_frames = int(max_inflight_frames)
        self._pending: Dict[int, _PendingFrame] = {}
        self.stats = {
            'packets': 0,
            'duplicates': 0,
            'conflicting_duplicates': 0,
            'incomplete_frames': 0,
            'invalid_jpeg_frames': 0,
            'completed_frames': 0,
        }

    @property
    def pending_frames(self) -> int:
        return len(self._pending)

    def _drop_expired(self, monotonic_ns: int) -> None:
        expired = [
            timestamp_ns
            for timestamp_ns, frame in self._pending.items()
            if monotonic_ns - frame.last_monotonic_ns > self.assembly_timeout_ns
        ]
        for timestamp_ns in expired:
            del self._pending[timestamp_ns]
            self.stats['incomplete_frames'] += 1

    def _enforce_capacity(self) -> None:
        while len(self._pending) >= self.max_inflight_frames:
            oldest_timestamp = min(
                self._pending,
                key=lambda item: self._pending[item].last_monotonic_ns,
            )
            del self._pending[oldest_timestamp]
            self.stats['incomplete_frames'] += 1

    def push(self, packet: CameraPacket,
             monotonic_ns: Optional[int] = None) -> Optional[AssembledCameraFrame]:
        now_ns = time.monotonic_ns() if monotonic_ns is None else int(monotonic_ns)
        self.stats['packets'] += 1
        self._drop_expired(now_ns)

        pending = self._pending.get(packet.timestamp_ns)
        if pending is None:
            self._enforce_capacity()
            pending = _PendingFrame(chunks={}, last_monotonic_ns=now_ns)
            self._pending[packet.timestamp_ns] = pending
        pending.last_monotonic_ns = now_ns

        previous = pending.chunks.get(packet.index)
        if previous is not None:
            if previous == packet.jpeg_chunk:
                self.stats['duplicates'] += 1
                return None
            del self._pending[packet.timestamp_ns]
            self.stats['conflicting_duplicates'] += 1
            self.stats['incomplete_frames'] += 1
            return None
        pending.chunks[packet.index] = packet.jpeg_chunk

        if packet.is_final:
            if pending.final_index is not None and pending.final_index != packet.index:
                del self._pending[packet.timestamp_ns]
                self.stats['incomplete_frames'] += 1
                return None
            pending.final_index = packet.index

        if pending.final_index is None:
            return None

        first_index = 0 if 0 in pending.chunks else 1
        expected_indices = range(first_index, pending.final_index + 1)
        if any(index not in pending.chunks for index in expected_indices):
            return None

        jpeg = b''.join(pending.chunks[index] for index in expected_indices)
        del self._pending[packet.timestamp_ns]
        if not (jpeg.startswith(b'\xff\xd8') and jpeg.endswith(b'\xff\xd9')):
            self.stats['invalid_jpeg_frames'] += 1
            return None

        self.stats['completed_frames'] += 1
        return AssembledCameraFrame(
            timestamp_ns=packet.timestamp_ns,
            jpeg=jpeg,
            packet_count=pending.final_index - first_index + 1,
            first_index=first_index,
            final_index=pending.final_index,
        )


class CameraReceiver(SensorReceiver):
    """Receive one MORAI UDP camera and emit lossless JPEG SensorFrames."""

    def __init__(
        self,
        sensor_id: str,
        bind_ip: str,
        bind_port: int,
        socket_timeout_s: float = 0.2,
        receive_buffer_bytes: int = 8 * 1024 * 1024,
        assembly_timeout_ns: int = 1_000_000_000,
        max_inflight_frames: int = 8,
    ):
        if not sensor_id:
            raise ValueError('sensor_id must be non-empty')
        if not (0 <= int(bind_port) <= 65_535):
            raise ValueError('bind_port must be a valid UDP port')
        self.sensor_id = sensor_id
        self.bind_ip = bind_ip
        self.bind_port = int(bind_port)
        self.socket_timeout_s = float(socket_timeout_s)
        self.receive_buffer_bytes = int(receive_buffer_bytes)
        self.assembler = CameraFrameAssembler(
            assembly_timeout_ns=assembly_timeout_ns,
            max_inflight_frames=max_inflight_frames,
        )
        self._socket: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._callback: Optional[Callable[[SensorFrame], None]] = None
        self._lock = threading.Lock()
        self._stats = {
            'datagrams': 0,
            'malformed_datagrams': 0,
            'socket_errors': 0,
            'callback_errors': 0,
        }

    @property
    def stats(self) -> Mapping[str, int]:
        result = dict(self._stats)
        result.update(self.assembler.stats)
        result['pending_frames'] = self.assembler.pending_frames
        return result

    def start(self, callback: Callable[[SensorFrame], None]) -> None:
        if not callable(callback):
            raise TypeError('callback must be callable')
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError('camera receiver is already running')
            udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                udp_socket.setsockopt(
                    socket.SOL_SOCKET, socket.SO_RCVBUF, self.receive_buffer_bytes
                )
                udp_socket.settimeout(self.socket_timeout_s)
                udp_socket.bind((self.bind_ip, self.bind_port))
            except Exception:
                udp_socket.close()
                raise
            self._socket = udp_socket
            self._callback = callback
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._receive_loop,
                name='morai-camera-{0}'.format(self.sensor_id),
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._stop_event.set()
            udp_socket = self._socket
            thread = self._thread
            self._socket = None
            self._thread = None
            self._callback = None
        if udp_socket is not None:
            udp_socket.close()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, self.socket_timeout_s * 2.0))

    def _receive_loop(self) -> None:
        while not self._stop_event.is_set():
            udp_socket = self._socket
            if udp_socket is None:
                return
            try:
                datagram, source = udp_socket.recvfrom(CAMERA_MAX_DATAGRAM_SIZE)
                received_timestamp_ns = time.time_ns()
                received_monotonic_ns = time.monotonic_ns()
            except socket.timeout:
                continue
            except OSError:
                if not self._stop_event.is_set():
                    self._stats['socket_errors'] += 1
                continue

            self._stats['datagrams'] += 1
            try:
                packet = parse_camera_packet(datagram)
            except CameraPacketError:
                self._stats['malformed_datagrams'] += 1
                continue
            assembled = self.assembler.push(packet, received_monotonic_ns)
            if assembled is None:
                continue

            frame = SensorFrame(
                sensor_id=self.sensor_id,
                modality='camera',
                timestamp_ns=assembled.timestamp_ns,
                payload=assembled.jpeg,
                source_frame_id=None,
                clock_domain='morai_simulation',
                metadata={
                    'encoding': 'jpeg',
                    'packet_count': assembled.packet_count,
                    'first_packet_index': assembled.first_index,
                    'final_packet_index': assembled.final_index,
                    'receive_timestamp_ns': received_timestamp_ns,
                    'source_ip': source[0],
                    'source_port': source[1],
                },
            )
            callback = self._callback
            if callback is None:
                continue
            try:
                callback(frame)
            except Exception:
                self._stats['callback_errors'] += 1
