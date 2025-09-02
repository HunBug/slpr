from __future__ import annotations

import logging
import json
from pathlib import Path
from time import perf_counter
from contextlib import contextmanager
from typing import Any, Optional

import numpy as np

from joblib import Parallel, delayed  # type: ignore
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib  # type: ignore
from multiprocessing import Manager
from PIL import Image

from slpr.analysis import compare_to_original
from slpr.config import AlgoConfig, Config, GlobalConfig, load_config
from slpr.pyramid import build_laplacian_pyramid, visualize_pyramid
from slpr.render import save_pyramid_images
from slpr.sampling import stochastic_average, stochastic_reconstruct
from slpr.utils import (
    CasePaths,
    load_image,
    now_stamp,
    save_image,
    save_json,
    set_seed,
    rgb_to_gray,
    rgb_to_ycbcr,
    ycbcr_to_rgb,
)
from slpr.logging_utils import setup_logging
from slpr.env_info import env_as_dict
from slpr.progress import GlobalProgressBar, ProgressCounter


class TimingStats:
    def __init__(
        self, enabled: bool = True, granularity: str = "coarse"
    ) -> None:
        self.enabled = enabled
        self.granularity = granularity
        self._totals: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def _map_name(self, name: str) -> str:
        if self.granularity != "coarse":
            return name
        # Group fine-grained events into broader buckets
        if name.startswith("pyramid_"):
            return "pyramid"
        if name.startswith("stochastic_"):
            return "stochastic"
        if name in {"save_image", "save_pyramid_images"}:
            return "save"
        if name in {"metrics_compare"}:
            return "metrics"
        return name

    def add(self, name: str, dt: float) -> None:
        if not self.enabled:
            return
        key = self._map_name(name)
        self._totals[key] = self._totals.get(key, 0.0) + dt
        self._counts[key] = self._counts.get(key, 0) + 1

    def as_dict(self) -> dict[str, dict[str, float] | dict[str, int]]:
        return {"totals": self._totals, "counts": self._counts}


@contextmanager
def record(ts: Optional[TimingStats], name: str):
    if ts is None or not ts.enabled:
        yield
        return
    t0 = perf_counter()
    try:
        yield
    finally:
        ts.add(name, perf_counter() - t0)


class ThrottledAdder:
    def __init__(self, counter: Optional[ProgressCounter], chunk: int) -> None:
        self.counter = counter
        self.chunk = max(1, int(chunk))
        self._acc = 0

    def add(self, n: int) -> None:
        if self.counter is None or n <= 0:
            return
        self._acc += int(n)
        if self._acc >= self.chunk:
            self.counter.add(self._acc)
            self._acc = 0

    def flush(self) -> None:
        if self.counter is not None and self._acc > 0:
            self.counter.add(self._acc)
            self._acc = 0


def _process_case(
    asset_name: str,
    asset_path: Path,
    gcfg: GlobalConfig,
    acfg: AlgoConfig,
    stamp: str,
    shared_counter: Optional[Any] = None,
) -> None:
    log = logging.getLogger(f"case.{asset_name}.{acfg.name}")
    set_seed(gcfg.seed)
    if not asset_path.exists():
        log.warning("Asset not found: %s", asset_path)
        return

    ts = TimingStats(
        enabled=gcfg.profiling, granularity=gcfg.profiling_granularity
    )
    color_mode = gcfg.color_mode
    with record(ts, "load_image"):
        img_any: Any = load_image(asset_path, mode=color_mode)

    # Prepare visualization base (grayscale) and reconstruction inputs
    log.info("Building pyramids: levels=%d", acfg.pyramid_levels)
    if color_mode == "gray":
        img_gray: Any = img_any
        recon_inputs = [img_gray]
        cb: Any | None = None
        cr: Any | None = None
    elif color_mode == "rgb":
        img_gray = rgb_to_gray(img_any)
        r = img_any[:, :, 0]
        gch = img_any[:, :, 1]
        b = img_any[:, :, 2]
        recon_inputs = [r, gch, b]
        cb = None
        cr = None
    elif color_mode == "luma":
        y, cb, cr = rgb_to_ycbcr(img_any)
        img_gray = y
        recon_inputs = [y]
    else:
        raise ValueError(f"Unsupported color_mode: {color_mode}")

    log.info(
        "Building pyramids (vis) on grayscale: levels=%d",
        acfg.pyramid_levels,
    )
    with record(ts, "pyramid_build_vis"):
        lap0, base0, gaussians = build_laplacian_pyramid(
            img_gray, acfg.pyramid_levels
        )
    paths = CasePaths.create(
        Path(gcfg.output_root), stamp, asset_name, acfg.name
    )

    # Save pyramid visuals
    with record(ts, "save_pyramid_images"):
        save_pyramid_images(visualize_pyramid(gaussians), paths.pyramid_dir)

    # 100% recon for metrics
    h, w = img_gray.shape
    log.info(
        "Reconstructing 100%% avg with %d samples", gcfg.samples_per_render
    )

    # Progress aggregation
    ch_factor = 3 if color_mode == "rgb" else 1
    tw_100 = w * ch_factor
    row_counter: Optional[ProgressCounter] = (
        ProgressCounter(shared_counter) if shared_counter else None
    )
    throttled_100 = ThrottledAdder(row_counter, gcfg.progress_update_pixels)

    def make_row_cb(throttled: ThrottledAdder, width: int):
        def _noop(d: int) -> None:
            return None

        if throttled is None:
            return _noop

        def _cb(d: int) -> None:
            # d is rows; multiply by width to get pixels for this channel
            throttled.add(d * width)

        return _cb

    def recon_avg_for_channel(src2d: Any) -> Any:
        with record(ts, "laplacian_build"):
            laps, base, _ = build_laplacian_pyramid(src2d, acfg.pyramid_levels)
        with record(ts, "stochastic_average"):
            return stochastic_average(
                laps,
                base,
                (h, w),
                acfg.patch_size,
                acfg.jitter,
                acfg.noise_strength,
                gcfg.samples_per_render,
                seed=gcfg.seed,
                update_cb=make_row_cb(throttled_100, tw_100),
            )

    if color_mode == "gray":
        recon_100 = recon_avg_for_channel(recon_inputs[0])
        with record(ts, "save_image"):
            save_image(recon_100, paths.output_dir / "recon_zoom_100_avg.png")
        with record(ts, "metrics_compare"):
            metrics = compare_to_original(img_gray, recon_100)
    elif color_mode == "rgb":
        r100 = recon_avg_for_channel(recon_inputs[0])
        g100 = recon_avg_for_channel(recon_inputs[1])
        b100 = recon_avg_for_channel(recon_inputs[2])
        recon_100 = np.stack([r100, g100, b100], axis=2)
        with record(ts, "save_image"):
            save_image(recon_100, paths.output_dir / "recon_zoom_100_avg.png")
        with record(ts, "metrics_compare"):
            metrics = compare_to_original(
                rgb_to_gray(img_any), rgb_to_gray(recon_100)
            )
    else:  # luma
        y100 = recon_avg_for_channel(recon_inputs[0])
        assert cb is not None and cr is not None
        recon_100 = ycbcr_to_rgb(y100, cb, cr)
        with record(ts, "save_image"):
            save_image(recon_100, paths.output_dir / "recon_zoom_100_avg.png")
        with record(ts, "metrics_compare"):
            metrics = compare_to_original(img_gray, y100)

    save_json(metrics, paths.output_dir / "metrics_100.json")
    log.info("Metrics @100%%: %s", metrics)

    # Other zooms
    for z in gcfg.zoom_levels:
        log.info(
            "Zoom %d%%: sampling 1 and averaging %d samples",
            z,
            gcfg.samples_per_render,
        )
        th = int(h * (z / 100.0))
        tw = int(w * (z / 100.0))
        row_counter_z: Optional[ProgressCounter] = (
            ProgressCounter(shared_counter) if shared_counter else None
        )
        throttled_z = ThrottledAdder(
            row_counter_z, gcfg.progress_update_pixels
        )
        rw = tw * (3 if color_mode == "rgb" else 1)

        def recon_pair_for_channel(src2d: Any) -> tuple[Any, Any]:
            with record(ts, "laplacian_build"):
                laps, base, _ = build_laplacian_pyramid(
                    src2d, acfg.pyramid_levels
                )
            with record(ts, "stochastic_reconstruct"):
                s1 = stochastic_reconstruct(
                    laps,
                    base,
                    (th, tw),
                    acfg.patch_size,
                    acfg.jitter,
                    acfg.noise_strength,
                    seed=gcfg.seed,
                    update_cb=make_row_cb(throttled_z, rw),
                )
            with record(ts, "stochastic_average"):
                savg = stochastic_average(
                    laps,
                    base,
                    (th, tw),
                    acfg.patch_size,
                    acfg.jitter,
                    acfg.noise_strength,
                    gcfg.samples_per_render,
                    seed=gcfg.seed,
                    update_cb=make_row_cb(throttled_z, rw),
                )
            return s1, savg

        if color_mode == "gray":
            s1, savg = recon_pair_for_channel(recon_inputs[0])
            with record(ts, "save_image"):
                save_image(
                    s1, paths.output_dir / f"recon_zoom_{z}_sample1.png"
                )
            with record(ts, "save_image"):
                save_image(savg, paths.output_dir / f"recon_zoom_{z}_avg.png")
        elif color_mode == "rgb":
            r1, ravg = recon_pair_for_channel(recon_inputs[0])
            g1, gavg = recon_pair_for_channel(recon_inputs[1])
            b1, bavg = recon_pair_for_channel(recon_inputs[2])
            with record(ts, "save_image"):
                save_image(
                    np.stack([r1, g1, b1], axis=2),
                    paths.output_dir / f"recon_zoom_{z}_sample1.png",
                )
            with record(ts, "save_image"):
                save_image(
                    np.stack([ravg, gavg, bavg], axis=2),
                    paths.output_dir / f"recon_zoom_{z}_avg.png",
                )
        else:  # luma
            y1, yavg = recon_pair_for_channel(recon_inputs[0])
            assert cb is not None and cr is not None
            with record(ts, "save_image"):
                save_image(
                    ycbcr_to_rgb(y1, cb, cr),
                    paths.output_dir / f"recon_zoom_{z}_sample1.png",
                )
            with record(ts, "save_image"):
                save_image(
                    ycbcr_to_rgb(yavg, cb, cr),
                    paths.output_dir / f"recon_zoom_{z}_avg.png",
                )
    # ensure batched progress is flushed per-zoom
    throttled_z.flush()

    # Persist per-case timings
    # Flush any pending progress increments
    throttled_100.flush()
    save_json(ts.as_dict(), paths.output_dir / "timings.json")


def main(config_path: Path) -> None:
    cfg: Config = load_config(config_path)
    stamp = now_stamp()

    # Prepare tasks
    tasks: list[tuple[str, Path, GlobalConfig, AlgoConfig, str]] = []
    for a in cfg.assets:
        for c in cfg.configs:
            tasks.append((a.name, a.path, cfg.global_cfg, c, stamp))

    # Setup logging at run root
    run_root = Path(cfg.global_cfg.output_root) / stamp
    setup_logging(run_root, cfg.global_cfg.log_level)
    log = logging.getLogger("slpr")
    log.info("Starting run: %s", stamp)
    log.info("Config file: %s", str(config_path))
    save_json(env_as_dict(), run_root / "environment.json")
    save_json(
        {
            "global": {
                "output_root": str(cfg.global_cfg.output_root),
                "num_cores": cfg.global_cfg.num_cores,
                "samples_per_render": cfg.global_cfg.samples_per_render,
                "zoom_levels": cfg.global_cfg.zoom_levels,
                "color_mode": cfg.global_cfg.color_mode,
                "seed": cfg.global_cfg.seed,
                "profiling": cfg.global_cfg.profiling,
                "profiling_granularity": cfg.global_cfg.profiling_granularity,
                "progress_update_pixels": (
                    cfg.global_cfg.progress_update_pixels
                ),
            },
            "assets": [
                {"name": a.name, "path": str(a.path)} for a in cfg.assets
            ],
            "configs": [
                {
                    "name": c.name,
                    "pyramid_levels": c.pyramid_levels,
                    "patch_size": c.patch_size,
                    "jitter": c.jitter,
                    "noise_strength": c.noise_strength,
                }
                for c in cfg.configs
            ],
        },
        run_root / "resolved_config.json",
    )

    n_jobs = (
        -1 if cfg.global_cfg.num_cores is None else cfg.global_cfg.num_cores
    )
    total = len(tasks)

    # Compute approximate global pixel workload; account for RGB factor
    asset_sizes: dict[str, tuple[int, int]] = {}
    for a in cfg.assets:
        ap = Path(a.path)
        if ap.exists():
            with Image.open(ap) as img0:
                asset_sizes[a.name] = img0.size  # (w, h)
        else:
            asset_sizes[a.name] = (512, 512)

    approx_total_px = 0
    ch_factor = 3 if cfg.global_cfg.color_mode == "rgb" else 1
    for a in cfg.assets:
        base_w, base_h = asset_sizes[a.name]
        for c in cfg.configs:
            samples = cfg.global_cfg.samples_per_render
            approx_total_px += (
                c.pyramid_levels * base_h * samples * base_w * ch_factor
            )
            for z in cfg.global_cfg.zoom_levels:
                th = int(base_h * (z / 100.0))
                tw = int(base_w * (z / 100.0))
                approx_total_px += (
                    c.pyramid_levels * th * (1 + samples) * tw * ch_factor
                )

    mgr = Manager()
    shared_done = mgr.Value("i", 0)
    gbar = GlobalProgressBar(
        shared_done, approx_total_px, desc="Aggregated pixels"
    )
    gbar.start()
    if n_jobs == 1:
        for t in tqdm(
            tasks, total=total, desc="Processing cases", unit="case"
        ):
            _process_case(*t, shared_counter=shared_done)
    else:
        log.info(
            "Running in parallel with %s jobs",
            n_jobs if n_jobs > 0 else "auto",
        )
        with tqdm_joblib(
            tqdm(total=total, desc="Processing cases", unit="case")
        ):
            Parallel(n_jobs=n_jobs)(
                delayed(_process_case)(*t, shared_counter=shared_done)
                for t in tasks
            )
    gbar.stop()

    # Aggregate and report timing stats
    def aggregate_timings(
        run_root: Path,
    ) -> dict[str, dict[str, float] | dict[str, int]]:
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for asset_dir in run_root.iterdir():
            if not asset_dir.is_dir():
                continue
            for cfg_dir in asset_dir.iterdir():
                if not cfg_dir.is_dir():
                    continue
                tfile = cfg_dir / "timings.json"
                if not tfile.exists():
                    continue
                try:
                    with tfile.open("r") as f:
                        data = json.load(f)
                    t = data.get("totals", {})
                    c = data.get("counts", {})
                    for k, v in t.items():
                        totals[k] = totals.get(k, 0.0) + float(v)
                    for k, v in c.items():
                        counts[k] = counts.get(k, 0) + int(v)
                except Exception as e:  # best-effort aggregate
                    logging.getLogger("slpr").warning(
                        "Failed to read timings %s: %s", str(tfile), e
                    )
        return {"totals": totals, "counts": counts}

    agg = aggregate_timings(run_root)
    save_json(agg, run_root / "run_timings.json")

    totals = agg.get("totals", {})
    counts = agg.get("counts", {})
    items = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    log.info("Timing summary (total seconds, calls, avg per call):")
    for name, total_secs in items:
        cnt = counts.get(name, 0) or 1
        avg = total_secs / cnt
        log.info(
            " - %s: %.3fs total over %d calls (%.3fs avg)",
            name,
            total_secs,
            cnt,
            avg,
        )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.yaml")
    args = p.parse_args()
    main(Path(args.config))
