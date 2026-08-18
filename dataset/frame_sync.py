"""Timestamp-based synchronization for independently arriving streams."""

from bisect import bisect_left
from dataclasses import dataclass
from typing import Any, Dict, Generic, Mapping, Optional, Tuple, TypeVar, Union

from .schema import SourceFrameId, SyncSource, validate_timestamp_ns


T = TypeVar('T')


@dataclass(frozen=True)
class TimedSample(Generic[T]):
    timestamp_ns: int
    payload: T
    source_frame_id: Optional[SourceFrameId] = None
    clock_domain: str = 'morai_simulation'

    def __post_init__(self):
        validate_timestamp_ns(self.timestamp_ns)


@dataclass(frozen=True)
class SynchronizedBundle:
    anchor_stream: str
    timestamp_ns: int
    samples: Mapping[str, TimedSample[Any]]
    offset_ns: Mapping[str, int]

    def sync_sources(self) -> Dict[str, SyncSource]:
        """Build the synchronization section stored in FrameMetadata."""
        return {
            stream: SyncSource(
                timestamp_ns=sample.timestamp_ns,
                offset_ns=self.offset_ns[stream],
                source_frame_id=sample.source_frame_id,
                clock_domain=sample.clock_domain,
            )
            for stream, sample in self.samples.items()
        }


class FrameSynchronizer:
    """Nearest-timestamp synchronizer with bounded per-stream buffers.

    Samples are not interpolated. By default, a non-anchor stream must advance
    to at least the anchor timestamp before its nearest sample is committed;
    this avoids selecting an early sample before a closer future sample arrives.
    Use ``force=True`` only when flushing a stopped sequence.
    """

    def __init__(
        self,
        anchor_stream: str,
        required_streams: Tuple[str, ...],
        tolerance_ns: Mapping[str, int],
        max_buffer_size: int = 256,
    ):
        if not anchor_stream:
            raise ValueError('anchor_stream must be non-empty')
        if anchor_stream not in required_streams:
            raise ValueError('anchor_stream must be a required stream')
        if len(set(required_streams)) != len(required_streams):
            raise ValueError('required_streams must be unique')
        if max_buffer_size <= 0:
            raise ValueError('max_buffer_size must be positive')
        for stream in required_streams:
            if stream == anchor_stream:
                continue
            if stream not in tolerance_ns or int(tolerance_ns[stream]) < 0:
                raise ValueError('every non-anchor stream needs a non-negative tolerance')

        self.anchor_stream = anchor_stream
        self.required_streams = tuple(required_streams)
        self.tolerance_ns = {key: int(value)
                             for key, value in tolerance_ns.items()}
        self.max_buffer_size = int(max_buffer_size)
        self._buffers = {stream: [] for stream in self.required_streams}
        self._clock_domain = None
        self.stats = {
            'duplicates': 0,
            'overflow': 0,
            'stale': 0,
            'unmatched_anchors': 0,
            'matched': 0,
        }

    def add(self, stream: str, sample: TimedSample[Any]) -> bool:
        """Insert a sample in timestamp order; reject duplicate timestamps."""
        if stream not in self._buffers:
            raise KeyError('unknown stream: {0}'.format(stream))
        if self._clock_domain is None:
            self._clock_domain = sample.clock_domain
        elif sample.clock_domain != self._clock_domain:
            raise ValueError(
                'clock domain mismatch: {0} != {1}'.format(
                    sample.clock_domain, self._clock_domain
                )
            )
        buffer = self._buffers[stream]
        timestamps = [item.timestamp_ns for item in buffer]
        index = bisect_left(timestamps, sample.timestamp_ns)
        if index < len(buffer) and buffer[index].timestamp_ns == sample.timestamp_ns:
            self.stats['duplicates'] += 1
            return False
        buffer.insert(index, sample)
        if len(buffer) > self.max_buffer_size:
            buffer.pop(0)
            self.stats['overflow'] += 1
        return True

    def pending(self) -> Dict[str, int]:
        return {stream: len(buffer) for stream, buffer in self._buffers.items()}

    def pop_next(self, force: bool = False) -> Optional[SynchronizedBundle]:
        """Return the oldest ready synchronized bundle, or ``None``."""
        anchor_buffer = self._buffers[self.anchor_stream]

        while anchor_buffer:
            anchor = anchor_buffer[0]
            selected = {self.anchor_stream: (0, anchor)}
            must_wait = False
            anchor_is_hopeless = False

            for stream in self.required_streams:
                if stream == self.anchor_stream:
                    continue
                buffer = self._buffers[stream]
                tolerance = self.tolerance_ns[stream]
                lower_bound = anchor.timestamp_ns - tolerance
                upper_bound = anchor.timestamp_ns + tolerance

                while buffer and buffer[0].timestamp_ns < lower_bound:
                    buffer.pop(0)
                    self.stats['stale'] += 1

                if not buffer:
                    must_wait = True
                    break
                if buffer[0].timestamp_ns > upper_bound:
                    anchor_is_hopeless = True
                    break
                if not force and buffer[-1].timestamp_ns < anchor.timestamp_ns:
                    must_wait = True
                    break

                timestamps = [item.timestamp_ns for item in buffer]
                right_index = bisect_left(timestamps, anchor.timestamp_ns)
                candidate_indices = []
                if right_index < len(buffer):
                    candidate_indices.append(right_index)
                if right_index > 0:
                    candidate_indices.append(right_index - 1)
                index = min(
                    candidate_indices,
                    key=lambda item_index: abs(
                        buffer[item_index].timestamp_ns - anchor.timestamp_ns
                    ),
                )
                candidate = buffer[index]
                if abs(candidate.timestamp_ns - anchor.timestamp_ns) > tolerance:
                    anchor_is_hopeless = candidate.timestamp_ns > upper_bound
                    must_wait = not anchor_is_hopeless
                    break
                selected[stream] = (index, candidate)

            if anchor_is_hopeless:
                anchor_buffer.pop(0)
                self.stats['unmatched_anchors'] += 1
                continue
            if must_wait:
                return None

            anchor_buffer.pop(0)
            samples = {self.anchor_stream: anchor}
            offsets = {self.anchor_stream: 0}
            for stream in self.required_streams:
                if stream == self.anchor_stream:
                    continue
                index, sample = selected[stream]
                buffer = self._buffers[stream]
                if index:
                    self.stats['stale'] += index
                del buffer[:index + 1]
                samples[stream] = sample
                offsets[stream] = sample.timestamp_ns - anchor.timestamp_ns

            self.stats['matched'] += 1
            return SynchronizedBundle(
                anchor_stream=self.anchor_stream,
                timestamp_ns=anchor.timestamp_ns,
                samples=samples,
                offset_ns=offsets,
            )

        return None
