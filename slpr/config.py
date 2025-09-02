from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class GlobalConfig:
    output_root: Path
    num_cores: Optional[int]
    samples_per_render: int
    zoom_levels: List[int]
    color_mode: str = "gray"  # or "luma"
    seed: Optional[int] = 42
    log_level: str = "INFO"
    # Telemetry knobs
    profiling: bool = True
    # 'coarse' groups fine events; 'fine' keeps original labels
    profiling_granularity: str = "coarse"
    # Aggregate progress updates and only forward every ~N pixels
    progress_update_pixels: int = 25000


@dataclass
class Asset:
    name: str
    path: Path


@dataclass
class AlgoConfig:
    name: str
    pyramid_levels: int
    patch_size: int
    jitter: float
    noise_strength: float


@dataclass
class Config:
    global_cfg: GlobalConfig
    assets: List[Asset]
    configs: List[AlgoConfig]


def _as_int_or_none(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, str) and v.lower() == "auto":
        return None
    return int(v)


def load_config(path: Path) -> Config:
    with path.open("r") as f:
        data: Dict[str, Any] = yaml.safe_load(f)

    g = data.get("global", {})
    global_cfg = GlobalConfig(
        output_root=Path(g.get("output_root", "outputs")),
        num_cores=_as_int_or_none(g.get("num_cores", "auto")),
        samples_per_render=int(g.get("samples_per_render", 8)),
        zoom_levels=[int(z) for z in g.get("zoom_levels", [100, 200])],
        color_mode=str(g.get("color_mode", "gray")),
        seed=(None if g.get("seed") is None else int(g.get("seed", 42))),
        log_level=str(g.get("log_level", "INFO")),
        profiling=bool(g.get("profiling", True)),
        profiling_granularity=str(
            g.get("profiling_granularity", "coarse")
        ),
        progress_update_pixels=int(g.get("progress_update_pixels", 25000)),
    )

    assets_raw = data.get("assets", [])
    assets = [Asset(name=a["name"], path=Path(a["path"])) for a in assets_raw]

    cfgs_raw = data.get("configs", [])
    cfgs: List[AlgoConfig] = []
    for c in cfgs_raw:
        cfgs.append(
            AlgoConfig(
                name=c["name"],
                pyramid_levels=int(c.get("pyramid_levels", 5)),
                patch_size=int(c.get("patch_size", 3)),
                jitter=float(c.get("jitter", 1.0)),
                noise_strength=float(c.get("noise_strength", 0.1)),
            )
        )

    return Config(global_cfg=global_cfg, assets=assets, configs=cfgs)
