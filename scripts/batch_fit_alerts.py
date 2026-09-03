import csv
import pickle
import time
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import gzip

import numpy as np
import pandas as pd
import jax.numpy as jnp
from astropy.io import fits
from bayesn import SEDmodel

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ZIP_PATH = PROJECT_ROOT / "data" / "LSST_ALERTS.zip"
HEAD_NAME = "LSST_ALERTS/LSST/LSST_CHUNK01_SPLIT001_HEAD.FITS.gz"
PHOT_NAME = "LSST_ALERTS/LSST/LSST_CHUNK01_SPLIT001_PHOT.FITS.gz"

CANDIDATES_CSV = PROJECT_ROOT / "data" / "selected_alert_snids.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "alerts"
SUMMARY_CSV = OUTPUT_DIR / "alert_fit_summary.csv"

LSST_FILT_MAP = {
    "u": "u_LSST",
    "g": "g_LSST",
    "r": "r_LSST",
    "i": "i_LSST",
    "z": "z_LSST",
    "y": "y_LSST",
}

# Broad redshift prior, matching the notebook's photo-z exploration
Z_QUANTILES = jnp.linspace(0.001, 1.0, 101)

# Any max r_hat above this is flagged "marginal" rather than "converged".
# This is a label only -- marginal fits are still saved, not discarded.
RHAT_WARN_THRESHOLD = 1.05


def load_head_phot():
    with ZipFile(ZIP_PATH, "r") as archive:
        head_bytes = gzip.decompress(archive.read(HEAD_NAME))
        with fits.open(BytesIO(head_bytes), memmap=False) as hdul:
            head_data = hdul[1].data.copy()

        phot_bytes = gzip.decompress(archive.read(PHOT_NAME))
        with fits.open(BytesIO(phot_bytes), memmap=False) as hdul:
            phot_data = hdul[1].data.copy()

    return head_data, phot_data


def get_lightcurve(head_row, phot_table):
    start = int(head_row["PTROBS_MIN"]) - 1
    stop = int(head_row["PTROBS_MAX"])

    lc = pd.DataFrame(phot_table[start:stop])

    lc["BAND"] = (
        lc["BAND"].str.decode("utf-8").str.strip()
        if lc["BAND"].dtype == object
        else lc["BAND"].astype(str).str.strip()
    )

    return lc


def clean_lightcurve(lc):
    lc = lc[
        np.isfinite(lc["MJD"])
        & np.isfinite(lc["FLUXCAL"])
        & np.isfinite(lc["FLUXCALERR"])
        & (lc["FLUXCALERR"] > 0)
    ].copy()

    lc["BAND"] = (
        lc["BAND"].astype(str).str.strip().str.replace("LSST_", "", regex=False)
    )

    return lc


def assess_convergence(samples):
    """Best-effort r_hat check. Returns (status, max_rhat) where status is
    'converged', 'marginal', or 'unknown' (if diagnostics can't be computed
    from the returned samples, e.g. no chain dimension retained)."""
    try:
        from numpyro.diagnostics import summary as numpyro_summary

        diag = numpyro_summary(samples, group_by_chain=True)
        rhats = [
            float(np.max(v["r_hat"]))
            for v in diag.values()
            if "r_hat" in v
        ]
        if not rhats:
            return "unknown", None

        max_rhat = max(rhats)
        status = "converged" if max_rhat <= RHAT_WARN_THRESHOLD else "marginal"
        return status, max_rhat
    except Exception:
        return "unknown", None


def append_summary_row(row, write_header):
    with SUMMARY_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not SUMMARY_CSV.exists()

    print(f"Loading candidates from {CANDIDATES_CSV}")
    candidates = pd.read_csv(CANDIDATES_CSV)
    candidates["SNID"] = candidates["SNID"].astype(str)
    print(f"Loaded {len(candidates)} candidate SNIDs")

    print("Loading HEAD/PHOT tables...")
    head_data, phot_data = load_head_phot()
    all_snids = np.char.strip(np.asarray(head_data["SNID"]).astype(str))

    print("Loading T21 BayeSN model...")
    model = SEDmodel(load_model="T21_model")

    n_done = 0
    n_skipped = 0
    n_failed = 0
    n_not_found = 0

    for _, cand in candidates.iterrows():
        snid = cand["SNID"]
        chains_file = OUTPUT_DIR / f"{snid}_chains.pkl"

        if chains_file.exists():
            n_skipped += 1
            continue

        matches = np.where(all_snids == snid)[0]
        if len(matches) == 0:
            print(f"SNID {snid} not found in this split's HEAD table, skipping")
            n_not_found += 1
            continue

        head_row = head_data[matches[0]]

        lc = get_lightcurve(head_row, phot_data)
        lc = clean_lightcurve(lc)

        if len(lc) == 0:
            print(f"SNID {snid}: no valid photometry after cleaning, skipping")
            n_not_found += 1
            continue

        t = np.asarray(lc["MJD"], dtype=float)
        flux = np.asarray(lc["FLUXCAL"], dtype=float)
        flux_err = np.asarray(lc["FLUXCALERR"], dtype=float)
        filters = np.asarray(lc["BAND"]).astype(str)

        print(f"\n=== Fitting {snid} ({len(lc)} obs) ===", flush=True)
        start = time.time()

        try:
            samples, props = model.fit(
                t,
                flux,
                flux_err,
                filters,
                ebv_mw=float(head_row["MWEBV"]),
                peak_mjd=float(head_row["PEAKMJD"]),
                filt_map=LSST_FILT_MAP,
                photoz=True,
                z_quantiles=Z_QUANTILES,
                chain_method="parallel",
            )
        except Exception as exc:
            print(f"FAILED on {snid}: {exc}", flush=True)
            n_failed += 1
            fail_file = OUTPUT_DIR / f"{snid}_FAILED.txt"
            fail_file.write_text(str(exc))
            append_summary_row(
                {
                    "SNID": snid,
                    "status": "failed",
                    "max_rhat": "",
                    "n_obs": len(lc),
                    "elapsed_s": round(time.time() - start, 1),
                },
                write_header,
            )
            write_header = False
            continue

        elapsed = time.time() - start
        status, max_rhat = assess_convergence(samples)

        print(
            f"Fit completed for {snid} in {elapsed:.1f}s "
            f"[{status}, max r_hat={max_rhat}]",
            flush=True,
        )

        with chains_file.open("wb") as f:
            pickle.dump(samples, f)

        append_summary_row(
            {
                "SNID": snid,
                "status": status,
                "max_rhat": max_rhat if max_rhat is not None else "",
                "n_obs": len(lc),
                "elapsed_s": round(elapsed, 1),
            },
            write_header,
        )
        write_header = False

        n_done += 1
        print(
            f"Progress: {n_done} done this run, {n_skipped} already done, "
            f"{n_failed} failed, {n_not_found} not found/empty, "
            f"{len(candidates) - n_done - n_skipped - n_failed - n_not_found} remaining",
            flush=True,
        )

    print("\n=== Batch run summary ===")
    print(f"Fit this run: {n_done}")
    print(f"Already done (skipped): {n_skipped}")
    print(f"Failed: {n_failed}")
    print(f"Not found / empty after cleaning: {n_not_found}")
    print(f"Total candidates: {len(candidates)}")
    print(f"\nPer-SN convergence status logged to: {SUMMARY_CSV}")

    if n_done + n_skipped + n_not_found < len(candidates):
        print(
            "\nNot all candidates completed. Resubmit this job to continue "
            "from where it left off (already-fit SNe are skipped)."
        )


if __name__ == "__main__":
    main()
