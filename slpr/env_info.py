from __future__ import annotations

import os
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Any

import numpy as np


@dataclass
class EnvInfo:
    python: str
    platform: str
    machine: str
    processor: str
    cwd: str
    numpy: str
    argv: list[str]
    env_vars: Dict[str, str]


def collect_env() -> EnvInfo:
    return EnvInfo(
        python=sys.version.split(" ")[0],
        platform=platform.platform(),
        machine=platform.machine(),
        processor=platform.processor(),
        cwd=str(Path.cwd()),
        numpy=np.__version__,
        argv=list(sys.argv),
        env_vars={
            k: v
            for k, v in os.environ.items()
            if k in ("CONDA_DEFAULT_ENV", "PYTHONPATH")
        },
    )


def env_as_dict() -> Dict[str, Any]:
    return asdict(collect_env())
