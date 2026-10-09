from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class UsageResult:
    """Results from one back-calculation run."""

    usage: pd.DataFrame
    raw_w: pd.DataFrame
    cell_qc: pd.DataFrame
    run_qc: dict[str, Any]
