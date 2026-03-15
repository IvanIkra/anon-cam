"""Latency metrics utilities for AnonCam."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
import time
from contextlib import contextmanager


@dataclass
class LatencySample:
    read_ms: float
    build_cfg_ms: float
    engine_ms: float
    ui_ms: float
    total_ms: float
    fps_inst: float
    fps_avg: float


@dataclass
class LatencyRecorder:
    """Sliding window of latency samples with CSV export."""

    path: str = "latency_last_frames.csv"
    window: int = 120
    _history: List[LatencySample] = field(default_factory=list, init=False, repr=False)

    def add(self, **values: float) -> None:
        """Add a new latency sample."""
        sample = LatencySample(
            read_ms=float(values.get("read_ms", 0.0)),
            build_cfg_ms=float(values.get("build_cfg_ms", 0.0)),
            engine_ms=float(values.get("engine_ms", 0.0)),
            ui_ms=float(values.get("ui_ms", 0.0)),
            total_ms=float(values.get("total_ms", 0.0)),
            fps_inst=float(values.get("fps_inst", 0.0)),
            fps_avg=float(values.get("fps_avg", 0.0)),
        )
        self._history.append(sample)
        if len(self._history) > self.window:
            self._history = self._history[-self.window :]
        self._write_csv()

    def context(self) -> "LatencyContext":
        """Create a context object for one frame iteration."""
        return LatencyContext(self)

    def _write_csv(self) -> None:
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                f.write(
                    "read_ms,build_cfg_ms,engine_ms,ui_ms,total_ms,"
                    "fps_inst,fps_avg\n"
                )
                for s in self._history:
                    f.write(
                        f"{s.read_ms:.3f},{s.build_cfg_ms:.3f},{s.engine_ms:.3f},"
                        f"{s.ui_ms:.3f},{s.total_ms:.3f},"
                        f"{s.fps_inst:.3f},{s.fps_avg:.3f}\n"
                    )
        except OSError:
            return


class LatencyContext:
    """Measure named sections inside a single iteration."""

    def __init__(self, recorder: LatencyRecorder):
        self._recorder = recorder
        self._sections: dict[str, float] = {}
        self._t0 = time.perf_counter()

    @contextmanager
    def section(self, name: str):
        t_start = time.perf_counter()
        try:
            yield
        finally:
            t_end = time.perf_counter()
            self._sections[name] = (t_end - t_start) * 1000.0

    def finalize(self, fps_inst: float, fps_avg: float) -> None:
        t_end = time.perf_counter()
        total_ms = (t_end - self._t0) * 1000.0
        self._recorder.add(
            read_ms=self._sections.get("read", 0.0),
            build_cfg_ms=self._sections.get("build_cfg", 0.0),
            engine_ms=self._sections.get("engine", 0.0),
            ui_ms=self._sections.get("ui", 0.0),
            total_ms=total_ms,
            fps_inst=fps_inst,
            fps_avg=fps_avg,
        )


