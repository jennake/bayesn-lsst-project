import pickle
import sys
import time
from pathlib import Path

import jax.numpy as jnp
from bayesn import SEDmodel

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "Foundation_DR1"
OUTPUT_DIR = PROJECT_ROOT / "results" / "foundation"

FILT_MAP = {
    "g": "g_PS1",
    "r": "r_PS1",
    "i": "i_PS1",
    "z": "z_PS1",
}

Z_QUANTILES = jnp.linspace(0.001, 0.30, 101)


def snid_from_path(path):
    # Strip the "Foundation_DR1_" prefix and ".txt" suffix
    stem = path.stem
    prefix = "Foundation_DR1_"
    if stem.startswith(prefix):
        return stem[len(prefix):]
    return stem


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sn_files = sorted(DATA_DIR.glob("Foundation_DR1_*.txt"))
    print(f"Found {len(sn_files)} SNe in {DATA_DIR}")

    print("Loading T21 BayeSN model...")
    model = SEDmodel(load_model="T21_model")

    n_done = 0
    n_skipped = 0
    n_failed = 0

    for sn_file in sn_files:
        snid = snid_from_path(sn_file)
        chains_file = OUTPUT_DIR / f"{snid}_chains.pkl"

        if chains_file.exists():
            n_skipped += 1
            continue

        print(f"\n=== Fitting {snid} ===", flush=True)
        start = time.time()

        try:
            samples, sn_props = model.fit_from_file(
                str(sn_file),
                filt_map=FILT_MAP,
                photoz=True,
                z_quantiles=Z_QUANTILES,
                chain_method="parallel",
            )
        except Exception as exc:
            print(f"FAILED on {snid}: {exc}", flush=True)
            n_failed += 1
            # Record the failure so we don't silently retry forever without
            # noticing a pattern; keep going with remaining SNe.
            fail_file = OUTPUT_DIR / f"{snid}_FAILED.txt"
            fail_file.write_text(str(exc))
            continue

        elapsed = time.time() - start
        print(f"Fit completed for {snid} in {elapsed:.1f}s", flush=True)
        print("SN properties:", sn_props, flush=True)

        with chains_file.open("wb") as f:
            pickle.dump(samples, f)

        n_done += 1
        print(
            f"Progress: {n_done} done this run, {n_skipped} already done, "
            f"{n_failed} failed, {len(sn_files) - n_done - n_skipped - n_failed} remaining",
            flush=True,
        )

    print("\n=== Batch run summary ===")
    print(f"Fit this run: {n_done}")
    print(f"Already done (skipped): {n_skipped}")
    print(f"Failed: {n_failed}")
    print(f"Total in sample: {len(sn_files)}")

    if n_done + n_skipped < len(sn_files):
        print(
            "\nNot all SNe completed. Resubmit this job to continue "
            "from where it left off (already-fit SNe are skipped)."
        )


if __name__ == "__main__":
    main()
