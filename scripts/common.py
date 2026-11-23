import csv, os, time, platform
import numpy as np

FIELDS = ["machine", "runtime", "device", "precision", "batch", "run",
          "mean_ms", "p50_ms", "p95_ms", "fps", "notes"]


def time_fn(fn, warmup=50, iters=500):
    """Call fn() warmup times (untimed), then iters times, returning per-call ms."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return np.array(times)


def stats(times, batch=1):
    return dict(
        mean_ms=float(times.mean()),
        p50_ms=float(np.percentile(times, 50)),
        p95_ms=float(np.percentile(times, 95)),
        fps=float(batch * 1000 / times.mean()),
    )


def log_result(path, runtime, device, precision, batch, times, notes=""):
    """Append one benchmark row to a CSV (writes the header if the file is new)."""
    row = dict(machine=platform.node(), runtime=runtime, device=device,
               precision=precision, batch=batch,
               run=int(os.environ.get("BENCH_RUN", "1")), notes=notes,
               **stats(times, batch))
    is_new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()})