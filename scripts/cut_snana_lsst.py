"""
Cut a SNANA-format LSST simulation (LSST_CHUNK01_SPLIT001..010 _HEAD.FITS/_PHOT.FITS)
down to light curves with a sensible number of observations, while preserving
the exact SNANA directory format so BayeSN can read it unchanged.

Usage:
    python cut_snana_lsst.py \
        --indir  /path/to/LSST_ALERTS/LSST \
        --outdir /path/to/LSST_ALERTS/LSST_CUT \
        --min-obs 8 --max-obs 300 --max-duration 200

Adjust --max-obs / --max-duration after looking at the histograms this script
prints on a first dry-run pass (use --dry-run to just get stats, no files written).
"""

import argparse
import glob
import os
import shutil

import numpy as np
from astropy.table import Table


def find_splits(indir):
    """Find all *_HEAD.FITS(.gz) files in indir, sorted."""
    pats = ["*_HEAD.FITS.gz", "*_HEAD.FITS"]
    files = sorted(set(f for p in pats for f in glob.glob(os.path.join(indir, p))))
    return files


def phot_filename(head_filename):
    base = os.path.basename(head_filename)
    return base.replace("HEAD", "PHOT")


def compute_obs_stats(head, phot):
    """Return n_obs and duration (MJD span) per object, based on PTROBS pointers."""
    n_obs = np.zeros(len(head), dtype=int)
    duration = np.zeros(len(head), dtype=float)
    mjd = np.asarray(phot["MJD"])

    for i, row in enumerate(head):
        lo, hi = int(row["PTROBS_MIN"]) - 1, int(row["PTROBS_MAX"])  # -1: FITS ptrs are 1-indexed
        block_mjd = mjd[lo:hi]
        valid = block_mjd > 0  # excludes the -777 terminator row
        n_obs[i] = valid.sum()
        if valid.any():
            duration[i] = block_mjd[valid].max() - block_mjd[valid].min()
    return n_obs, duration


def cut_split(head_path, phot_path, min_obs, max_obs, max_duration, outdir, dry_run):
    head = Table.read(head_path)
    phot = Table.read(phot_path)

    n_obs, duration = compute_obs_stats(head, phot)

    mask = (n_obs >= min_obs) & (n_obs <= max_obs)
    if max_duration is not None:
        mask &= duration <= max_duration

    kept, total = mask.sum(), len(head)
    print(f"{os.path.basename(head_path)}: keeping {kept}/{total} "
          f"({100*kept/total:.1f}%)  n_obs[kept] median={np.median(n_obs[mask]) if kept else 0:.0f}")

    if dry_run:
        return kept, total, n_obs, duration

    head_cut = head[mask]

    # Rebuild PHOT, preserving each surviving block (data rows + its -777 terminator),
    # and re-point PTROBS_MIN/MAX to the new, compacted row numbers.
    new_blocks = []
    new_min, new_max = [], []
    ptr = 1  # 1-indexed, SNANA convention
    for row in head[mask]:
        lo, hi = int(row["PTROBS_MIN"]) - 1, int(row["PTROBS_MAX"])
        block = phot[lo:hi]
        new_blocks.append(block)
        new_min.append(ptr)
        new_max.append(ptr + len(block) - 1)
        ptr += len(block)

    from astropy.table import vstack
    phot_cut = vstack(new_blocks) if new_blocks else phot[:0]
    head_cut["PTROBS_MIN"] = new_min
    head_cut["PTROBS_MAX"] = new_max

    os.makedirs(outdir, exist_ok=True)
    head_out = os.path.join(outdir, os.path.basename(head_path))
    phot_out = os.path.join(outdir, phot_filename(head_path))
    head_cut.write(head_out, overwrite=True)
    phot_cut.write(phot_out, overwrite=True)

    return kept, total, n_obs, duration


def regenerate_aux_files(indir, outdir, min_obs, max_obs, max_duration, kept_total, orig_total):
    """Copy LIST/README/IGNORE across, pointing LIST at the cut HEAD files, and
    note the cut that was applied in README."""
    for pattern in ("*.LIST", "*.IGNORE"):
        for f in glob.glob(os.path.join(indir, pattern)):
            shutil.copy(f, outdir)

    list_files = sorted(glob.glob(os.path.join(outdir, "*_HEAD.FITS*")))
    list_path = glob.glob(os.path.join(outdir, "*.LIST"))
    if list_path:
        with open(list_path[0], "w") as f:
            f.write("\n".join(os.path.basename(x) for x in list_files) + "\n")

    readme_src = glob.glob(os.path.join(indir, "*.README"))
    note = (f"\n# --- cut_snana_lsst.py ---\n"
            f"# min_obs={min_obs}, max_obs={max_obs}, max_duration={max_duration}\n"
            f"# kept {kept_total}/{orig_total} objects\n")
    if readme_src:
        dst = os.path.join(outdir, os.path.basename(readme_src[0]))
        shutil.copy(readme_src[0], dst)
        with open(dst, "a") as f:
            f.write(note)
    else:
        with open(os.path.join(outdir, "CUT.README"), "w") as f:
            f.write(note)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--min-obs", type=int, default=8)
    ap.add_argument("--max-obs", type=int, default=None,
                     help="Drop objects with more than this many observations")
    ap.add_argument("--max-duration", type=float, default=None,
                     help="Drop objects whose MJD span exceeds this many days")
    ap.add_argument("--dry-run", action="store_true",
                     help="Just print stats/histogram info, write nothing")
    args = ap.parse_args()

    head_files = find_splits(args.indir)
    if not head_files:
        raise SystemExit(f"No *_HEAD.FITS(.gz) files found in {args.indir}")

    all_n_obs, all_dur = [], []
    kept_total, orig_total = 0, 0
    for head_path in head_files:
        phot_path = os.path.join(args.indir, phot_filename(head_path))
        max_obs = args.max_obs if args.max_obs is not None else int(1e9)
        kept, total, n_obs, duration = cut_split(
            head_path, phot_path, args.min_obs, max_obs,
            args.max_duration, args.outdir, args.dry_run,
        )
        all_n_obs.append(n_obs)
        all_dur.append(duration)
        kept_total += kept
        orig_total += total

    all_n_obs = np.concatenate(all_n_obs)
    all_dur = np.concatenate(all_dur)
    print("\n--- overall n_obs percentiles ---")
    for p in (1, 5, 25, 50, 75, 95, 99, 99.9):
        print(f"  p{p:>5}: {np.percentile(all_n_obs, p):.0f}")
    print("--- overall duration (days) percentiles ---")
    for p in (1, 5, 25, 50, 75, 95, 99, 99.9):
        print(f"  p{p:>5}: {np.percentile(all_dur, p):.0f}")
    print(f"\nTotal kept: {kept_total}/{orig_total} ({100*kept_total/orig_total:.1f}%)")

    if not args.dry_run:
        regenerate_aux_files(args.indir, args.outdir, args.min_obs, args.max_obs,
                              args.max_duration, kept_total, orig_total)
        print(f"Wrote cut dataset to {args.outdir}")


if __name__ == "__main__":
    main()
