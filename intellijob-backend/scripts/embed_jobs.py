"""
IntelliJob — Phase B-1: compute job-spec embeddings.

Reads the Phase A CSV (produced by ``seed_jobs.py``), feeds each row's
title + category + description into a SentenceTransformer model, and
writes a single ``.npz`` containing:

    ids          - array of job ids (str)
    vectors      - (N, 384) float32 array of unit-norm embeddings
    model_name   - the HF model id used (also stored in MANIFEST)
    schema_version - the embedder schema version (int)

Plus a small JSON manifest next to it for citation in the dissertation.

This script is intentionally file-based (npz, not pgvector). The
follow-up ``load_embeddings_to_pgvector.py`` script — to be written
when pgvector is installed on this machine — will read the same npz
and bulk-insert into PostgreSQL.

Run with:
    intellijob-backend/venv/Scripts/python.exe scripts/embed_jobs.py
    intellijob-backend/venv/Scripts/python.exe scripts/embed_jobs.py --dry-run
    intellijob-backend/venv/Scripts/python.exe scripts/embed_jobs.py --limit 10

Output:
    intellijob-backend/data/job_embeddings.npz
    intellijob-backend/data/job_embeddings.MANIFEST.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Project root is two levels up: scripts/embed_jobs.py -> intellijob-backend/.
# Make sure `import api.*` works whether this script is invoked as
# ``python scripts/embed_jobs.py`` from intellijob-backend/ OR as
# ``intellijob-backend/venv/.../python scripts/embed_jobs.py``.
_HERE = Path(__file__).resolve().parent
_BACKEND_ROOT = _HERE.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))
os.chdir(_BACKEND_ROOT)

from api.services.embedder import (
    DEFAULT_MODEL_NAME,
    EMBEDDING_DIM,
    EMBEDDING_SCHEMA_VERSION,
    EmbeddingResult,
    embed_jobs,
)
from api.services.jobs_loader import (
    JobRow,
    read_jobs_csv,
    validate_rows,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

#: Project root is two levels up from this script (scripts/embed_jobs.py).
PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Defaults — same as the diss-spec model.
DEFAULT_CSV = PROJECT_ROOT / "data" / "uk_software_jobs.csv"
DEFAULT_NPZ = PROJECT_ROOT / "data" / "job_embeddings.npz"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "job_embeddings.MANIFEST.json"

#: Forward-pass batch size. 64 is CPU-comfortable for 384-dim MiniLM.
DEFAULT_BATCH_SIZE = 64

#: Reproducibility seed.
DEFAULT_SEED = 42

#: Filter behaviour: drop rows that the CSV reader flagged as
 #: bad-quality rather than embedding them (which would just be
 #: noise). The ``--no-filter`` flag disables this for debugging.
DEFAULT_DROP_FLAGGED = True


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(
    *,
    manifest_path: Path,
    npz_path: Path,
    csv_path: Path,
    result: EmbeddingResult,
    rows_total: int,
    rows_dropped: int,
    model_name: str,
    batch_size: int,
    seed: int,
    device: str,
    elapsed_seconds: float,
) -> None:
    """Write a small JSON manifest next to the npz so the embedding
    file is citable in the dissertation (and version-checkable)."""
    manifest: dict[str, Any] = {
        "embedder_schema_version": result.schema_version,
        "embedding_dim": int(result.vectors.shape[1]) if result.count else EMBEDDING_DIM,
        "model_name": model_name,
        "npz_filename": npz_path.name,
        "npz_sha256": _sha256_of_file(npz_path) if npz_path.exists() else None,
        "source_csv": csv_path.name,
        "source_csv_sha256": _sha256_of_file(csv_path) if csv_path.exists() else None,
        "rows_in_csv": rows_total,
        "rows_embedded": int(result.count),
        "rows_dropped": rows_dropped,
        "batch_size": batch_size,
        "seed": seed,
        "device": device,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #


def _filter_rows(
    rows: list[JobRow], drop_flagged: bool
) -> tuple[list[JobRow], int]:
    """Drop rows flagged as bad-quality by validate_rows.

    Returns (kept_rows, dropped_count). If ``drop_flagged`` is False,
    every row is kept and dropped_count is 0.
    """
    if not drop_flagged:
        return rows, 0
    report = validate_rows(rows, require_uk=False)
    kept = [r for i, r in enumerate(rows) if i not in set(report.flagged_indices)]
    return kept, len(rows) - len(kept)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IntelliJob Phase B-1 — embed job specs.")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                   help=f"Input CSV (default: {DEFAULT_CSV})")
    p.add_argument("--output", type=Path, default=DEFAULT_NPZ,
                   help=f"Output npz (default: {DEFAULT_NPZ})")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                   help=f"Output manifest (default: {DEFAULT_MANIFEST})")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME,
                   help=f"HuggingFace model id (default: {DEFAULT_MODEL_NAME})")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                   help=f"Forward batch size (default: {DEFAULT_BATCH_SIZE})")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help=f"Random seed (default: {DEFAULT_SEED})")
    p.add_argument("--device", type=str, default="cpu",
                   help='Torch device (default: "cpu")')
    p.add_argument("--limit", type=int, default=None,
                   help="Embed at most N rows (for smoke testing).")
    p.add_argument("--no-filter", dest="drop_flagged",
                   action="store_false",
                   help="Do not drop rows flagged by validate_rows.")
    p.set_defaults(drop_flagged=DEFAULT_DROP_FLAGGED)
    p.add_argument("--dry-run", action="store_true",
                   help="Embed but do not write npz or manifest.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("IntelliJob Phase B-1 — embed job specs")
    print(f"  csv       : {args.csv}")
    print(f"  npz       : {args.output}")
    print(f"  manifest  : {args.manifest}")
    print(f"  model     : {args.model}")
    print(f"  batch     : {args.batch_size}")
    print(f"  seed      : {args.seed}")
    print(f"  device    : {args.device}")
    if args.limit is not None:
        print(f"  limit     : {args.limit}")
    if args.dry_run:
        print("  mode      : DRY RUN (no files will be written)")
    print()

    if not args.csv.exists():
        sys.exit(f"ERROR: input CSV not found: {args.csv}")

    rows = read_jobs_csv(args.csv)
    print(f"Loaded {len(rows)} rows from {args.csv.name}")

    rows, dropped = _filter_rows(rows, drop_flagged=args.drop_flagged)
    if dropped:
        print(f"Dropped {dropped} flagged row(s)")
    print(f"Embedding {len(rows)} rows")

    if args.limit is not None:
        rows = rows[: args.limit]
        print(f"  (limited to first {len(rows)})")

    t0 = time.monotonic()
    result = embed_jobs(
        rows,
        model_name=args.model,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        show_progress=True,
    )
    elapsed = time.monotonic() - t0

    print()
    print(f"Embedded {result.count} rows in {elapsed:.2f}s")
    print(f"  shape : {result.vectors.shape}")
    print(f"  dtype : {result.vectors.dtype}")
    if result.count:
        norms = np.linalg.norm(result.vectors, axis=1)
        print(f"  norms : min={norms.min():.4f} max={norms.max():.4f} "
              f"mean={norms.mean():.4f}")

    if args.dry_run:
        print("\n--dry-run set; not writing npz or manifest.")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        ids=np.asarray(result.ids, dtype=object),
        vectors=result.vectors,
        model_name=np.asarray(result.model_name),
        schema_version=np.asarray(result.schema_version, dtype=np.int64),
    )
    print(f"  npz  -> {args.output}")

    write_manifest(
        manifest_path=args.manifest,
        npz_path=args.output,
        csv_path=args.csv,
        result=result,
        rows_total=len(read_jobs_csv(args.csv)),
        rows_dropped=dropped,
        model_name=args.model,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        elapsed_seconds=elapsed,
    )
    print(f"  mft  -> {args.manifest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
