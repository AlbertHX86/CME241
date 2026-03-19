"""Minimal portfolio RL package for the CME241 project."""

from __future__ import annotations

import os

# Force a headless matplotlib backend and tolerate duplicate OpenMP runtimes
# that are common in mixed conda/pip scientific Python environments on Windows.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
