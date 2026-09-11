import gzip
import pickle
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import jax.numpy as jnp
from astropy.io import fits
from bayesn import SEDmodel

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ZIP_PATH = PROJECT_ROOT / "data" / "LSST_ALERTS.zip"
HEAD_NAME = "LSST_ALERTS/LSST/LSST_CHUNK01_SPLIT001_HEAD.FITS.gz"
PHOT_NAME = "LSST_ALERTS/LSST/LSST_CHUNK01_SPLIT001_PHOT.FITS.gz"

RESULTS_DIR = PROJECT_ROOT / "results" / "alerts"
SUMMARY_CSV = RESULTS_DIR / "alert_fit_summary.csv"
PLOTS_DIR = RESULTS_DIR / "plots"

LSST_FILT_MAP = {
    "u": "u_LSST",
    "g": "g_LSST",
    "r": "r_LSST",
    "i": "i_LSST",
    "z": "z_LSST",
    "y": "y_LSST",
}

COLOURS = {
    "u": "tab:purple",
    "g": "tab:green",
    "r": "tab:red",
    "i": "tab:orange",
    "z": "tab:brown",
    "y": "tab:gray",
}

Z_QUANTILES = jnp.linspace(0.001, 1.0, 101)


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

    float_cols = lc.select_dtypes(include="float").columns
    lc[float_cols] = lc[float_cols].astype("float32")

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


def plot_one(model, snid, status, max_rhat, head_row, phot_data):
    chains_file = RESULTS_DIR / f"{snid}_chains.pkl"
    if not chains_file.exists():
        print(f"{snid}: no chains file, skipping")
        return

    with chains_file.open("rb") as f:
        chains = pickle.load(f)

    plot_samples = dict(chains)
    if "mu" not in plot_samples:
        plot_samples["mu"] = plot_samples["Ds"]
    if "delM" not in plot_samples:
        plot_samples["delM"] = np.zeros_like(plot_samples["Ds"])

    lc = clean_lightcurve(get_lightcurve(head_row, phot_data))
    observed_bands = [b for b in ["u", "g", "r", "i", "z", "y"] if b in lc["BAND"].values]
    model_bands = [[LSST_FILT_MAP[b] for b in observed_bands]]  # note: nested list, per notebook

    # Per-SN state the model needs before reconstructing flux
    model.peak_mjds = np.array([float(head_row["PEAKMJD"])])
    model.ebv = np.array([float(head_row["MWEBV"])])

    model_mjd, model_mean, model_std = model.get_flux_from_chains_photoz(
        chains=plot_samples,
        bands=model_bands,
        num_sne=1,
        n_grid=200,
        num_samples=200,
    )

    model_mjd = np.asarray(model_mjd[0], dtype=np.float64)
    model_mean = np.asarray(model_mean[0], dtype=np.float64)
    model_std = np.asarray(model_std[0], dtype=np.float64)

    z_med = float(np.median(np.asarray(chains["z"]))) if "z" in chains else None

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, band in enumerate(observed_bands):
        use = lc["BAND"] == band
        colour = COLOURS.get(band, "tab:blue")
        ax.errorbar(
            lc.loc[use, "MJD"], lc.loc[use, "FLUXCAL"],
            yerr=lc.loc[use, "FLUXCALERR"],
            fmt="o", ms=5, capsize=2, color=colour, label=f"{band} data",
        )
        ax.plot(model_mjd, model_mean[i], color=colour, linewidth=2, label=f"{band} BayeSN")
        ax.fill_between(
            model_mjd, model_mean[i] - model_std[i], model_mean[i] + model_std[i],
            color=colour, alpha=0.2,
        )

    ax.axhline(0, color="0.6", linewidth=1)
    ax.set_xlabel("MJD")
    ax.set_ylabel("FLUXCAL")
    rhat_str = f"{max_rhat:.2f}" if pd.notna(max_rhat) else "n/a"
    z_str = f", z={z_med:.3f}" if z_med is not None else ""
    ax.set_title(f"{snid}  [{status}, max r_hat={rhat_str}{z_str}]")
    ax.legend(ncol=2, fontsize=8)

    fig.tight_layout()
    out_path = PLOTS_DIR / f"{snid}_lightcurve_fit.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(SUMMARY_CSV)
    summary = summary[summary["SNID"] != "SNID"].copy()
    summary["max_rhat"] = pd.to_numeric(summary["max_rhat"], errors="coerce")

    print("Loading HEAD/PHOT tables...")
    head_data, phot_data = load_head_phot()
    all_snids = np.char.strip(np.asarray(head_data["SNID"]).astype(str))

    print("Loading T21 BayeSN model...")
    model = SEDmodel(load_model="T21_model")

    for _, row in summary.iterrows():
        if row["status"] == "failed":
            continue

        snid = str(row["SNID"])
        matches = np.where(all_snids == snid)[0]
        if len(matches) == 0:
            print(f"{snid}: not found in HEAD table, skipping")
            continue

        head_row = head_data[matches[0]]

        try:
            plot_one(model, snid, row["status"], row["max_rhat"], head_row, phot_data)
        except Exception as exc:
            print(f"{snid}: plotting failed: {exc}")


if __name__ == "__main__":
    main()
