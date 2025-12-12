import time
import os
import sys
import functools
from typing import Callable, Any, Optional, Dict
from contextlib import ContextDecorator

# Optional imports
try:
    import resource  # POSIX
except Exception:
    resource = None

try:
    import psutil
except Exception:
    psutil = None

# Windows ctypes fallback for peak RSS if neither resource nor psutil is present
_windows_ctypes_fallback_available = sys.platform == "win32"
if _windows_ctypes_fallback_available:
    import ctypes
    from ctypes import wintypes

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    _GetCurrentProcess = ctypes.windll.kernel32.GetCurrentProcess
    _GetCurrentProcess.restype = wintypes.HANDLE
    _GetProcessMemoryInfo = ctypes.windll.psapi.GetProcessMemoryInfo
    _GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
    _GetProcessMemoryInfo.restype = wintypes.BOOL


def _human_bytes(n: Optional[int]) -> str:
    if n is None:
        return "N/A"
    if n < 0:
        return "N/A"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    v = float(n)
    i = 0
    while v >= 1024.0 and i < len(units) - 1:
        v /= 1024.0
        i += 1
    return f"{v:,.2f} {units[i]}"


def _sec_to_hms(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h}h {m:02d}m {s:05.2f}s"


def _get_peak_rss_bytes() -> Optional[int]:
    """
    Return peak resident set size (bytes) if available, otherwise a best-effort current RSS.
    - On POSIX uses resource.getrusage(...).ru_maxrss (Linux: kB -> bytes).
    - Else uses psutil if installed (current RSS or available peak).
    - Else on Windows uses GetProcessMemoryInfo PeakWorkingSetSize via ctypes.
    - Otherwise returns None.
    """
    # POSIX resource (most reliable on Linux/macOS)
    try:
        if resource is not None:
            ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # On Linux ru_maxrss is in kilobytes, on macOS it's bytes.
            if sys.platform.startswith("linux"):
                return int(ru) * 1024
            else:
                return int(ru)
    except Exception:
        pass

    # psutil fallback (gives current RSS; some platforms expose peak fields)
    if psutil is not None:
        try:
            proc = psutil.Process()
            info = proc.memory_info()
            # prefer an explicit peak field if available
            for attr in ("peak_wset", "peak_rss", "peak_working_set", "peak"):
                if hasattr(info, attr):
                    val = getattr(info, attr)
                    if isinstance(val, (int, float)) and val > 0:
                        return int(val)
            # fall back to current RSS
            if hasattr(info, "rss"):
                return int(info.rss)
        except Exception:
            pass

    # Windows ctypes fallback to PeakWorkingSetSize
    if _windows_ctypes_fallback_available:
        try:
            h = _GetCurrentProcess()
            pmc = PROCESS_MEMORY_COUNTERS()
            pmc.cb = ctypes.sizeof(pmc)
            ok = _GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb)
            if ok:
                return int(pmc.PeakWorkingSetSize)
        except Exception:
            pass

    return None


class ResourceReport(ContextDecorator):
    """
    ContextDecorator that can be used as:
      - a decorator:  @resource_report(...)
      - a context:    with resource_report(...):

    It reports wallclock, CPU (user/system/total) and peak RSS.
    """

    def __init__(self, include_children: bool = False, name: Optional[str] = None):
        self.include_children = include_children
        self.name = name or "block"
        self.last_metrics: Optional[Dict[str, Any]] = None
        self._t_start: Optional[float] = None
        self._ot_start: Optional[os.times_result] = None

    # ---- ContextManager API ----
    def __enter__(self):
        self._t_start = time.perf_counter()
        self._ot_start = os.times()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._finalize_and_print(self.name)
        # Do not suppress exceptions
        return False

    # ---- Decorator API (ContextDecorator makes __call__ wrap automatically) ----
    def __call__(self, func: Callable[..., Any] = None, *, include_children: Optional[bool] = None, name: Optional[str] = None):
        # If used as @resource_report without parentheses, func is provided.
        if func is not None and callable(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # configure per-call name/include_children
                self.include_children = self.include_children if include_children is None else include_children
                call_name = func.__name__ if name is None else name
                self.name = call_name
                with self:
                    return func(*args, **kwargs)

            # expose last_metrics via attribute; updated after execution
            wrapper.last_metrics = None  # type: ignore[attr-defined]
            return wrapper

        # Called as resource_report(...): return a fresh context manager for with-usage or decorator factory
        rr = ResourceReport(
            include_children=self.include_children if include_children is None else include_children,
            name=name,
        )

        # When used as @resource_report(...), Python will pass the function to this returned object (callable)
        return rr

    # ---- internals ----
    def _finalize_and_print(self, label: str) -> None:
        if self._t_start is None or self._ot_start is None:
            return
        t_end = time.perf_counter()
        ot_end = os.times()

        wall = max(0.0, t_end - self._t_start)

        user = max(0.0, ot_end[0] - self._ot_start[0])
        system = max(0.0, ot_end[1] - self._ot_start[1])
        if self.include_children:
            user += max(0.0, ot_end[2] - self._ot_start[2])
            system += max(0.0, ot_end[3] - self._ot_start[3])

        total_cpu = user + system

        metrics: Dict[str, Any] = {
            "wall_seconds": wall,
            "wall_hms": _sec_to_hms(wall),
            "user_seconds": user,
            "system_seconds": system,
            "user_cpu_hours": user / 3600.0,
            "system_cpu_hours": system / 3600.0,
            "total_cpu_seconds": total_cpu,
            "total_cpu_hours": total_cpu / 3600.0,
            "peak_rss_bytes": _get_peak_rss_bytes(),
        }
        self.last_metrics = metrics
        prss = metrics["peak_rss_bytes"]

        print("=" * 56)
        print(f"Resource report for {label}")
        print("-" * 56)
        print(f"Wallclock elapsed : {_sec_to_hms(metrics['wall_seconds'])} ({metrics['wall_seconds']:.6f} s)")
        print()
        print("CPU (process):")
        print(f"  user   : {_sec_to_hms(metrics['user_seconds'])} ({metrics['user_seconds']:.6f} s) -> {metrics['user_cpu_hours']:.6f} CPU-hours")
        print(f"  system : {_sec_to_hms(metrics['system_seconds'])} ({metrics['system_seconds']:.6f} s) -> {metrics['system_cpu_hours']:.6f} CPU-hours")
        print(f"  total  : {_sec_to_hms(metrics['total_cpu_seconds'])} ({metrics['total_cpu_seconds']:.6f} s) -> {metrics['total_cpu_hours']:.6f} CPU-hours")
        print()
        if prss is not None:
            print(f"Peak RSS (process): {_human_bytes(prss)} ({prss:,d} bytes)")
        else:
            print("Peak RSS (process): not available on this platform")
        print("=" * 56)


def resource_report(include_children: bool = False, name: str = "Block") -> ResourceReport:
    """
    Factory returning a ResourceReport which works as decorator or context manager.
    Examples:
      @resource_report()
      def work(): ...

      with resource_report(include_children=True):
          ... arbitrary code ...
    """
    return ResourceReport(include_children=include_children, name=name)
