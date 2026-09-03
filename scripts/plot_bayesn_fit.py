import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bayesn import SEDmodel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHAINS_FILE = (
    PROJECT_ROOT
    / "results"
    / "terminal_test"
    / "chains.pkl"
)
OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "terminal_test"
    / "lightcurve_fit.png"
)


def read_snana_lightcurve(path):
    """Read the VARLIST and OBS rows from an SNANA text file."""

    columns = None
    observations = []

    with open(path) as file:
        for line in file:
            line = line.strip()

            if line.startswith("VARLIST:"):
                columns = line.split()[1:]

            elif line.startswith("OBS:"):
                observations.append(
                    line.split()[1:]
                )

    if columns is None:
        raise ValueError(
            "No VARLIST line found in the light-curve file"
        )

    lightcurve = pd.DataFrame(
        observations,
        columns=columns,
    )

    for column in [
        "MJD",
        "FLUXCAL",
        "FLUXCALERR",
    ]:
        lightcurve[column] = pd.to_numeric(
            lightcurve[column]
        )

    return lightcurve


def main():
    print("Loading model...")

    model = SEDmodel(
        load_model="T21_model"
    )

    print("Reading observations...")

    lightcurve = read_snana_lightcurve(
        model.example_lc
    )

    print("Loading posterior chains...")

    with CHAINS_FILE.open("rb") as file:
        chains = pickle.load(file)

    # Values reported by fit_from_file()
    redshift = 0.019253
    ebv_mw = 0.0599

    filter_map = {
        "g": "g_PS1",
        "r": "r_PS1",
        "i": "i_PS1",
        "z": "z_PS1",
    }

    colours = {
        "g": "tab:green",
        "r": "tab:red",
        "i": "tab:orange",
        "z": "tab:purple",
    }

    observed_bands = [
        band
        for band in ["g", "r", "i", "z"]
        if band in lightcurve["FLT"].values
    ]

    model_bands = [
        filter_map[band]
        for band in observed_bands
    ]

    # Rest-frame phases at which BayeSN will be evaluated
    phases = np.linspace(-10, 40, 200)

    print("Generating posterior light curves...")

    phot_grid = model.get_flux_from_chains(
        t=phases,
        bands=model_bands,
        chains=chains,
        zs=np.array([redshift]),
        ebv_mws=np.array([ebv_mw]),
        mag=False,
        num_sne=1,
        mean=False,
    )

    print("Posterior grid shape:", phot_grid.shape)

    model_mean = phot_grid.mean(axis=1)[0]
    model_std = phot_grid.std(axis=1)[0]

    # BayeSN saved the inferred absolute peak MJD in the chains.
    peak_mjd = float(
        np.median(np.asarray(chains["peak_MJD"]))
    )

    # Convert rest-frame phase back to observed MJD.
    model_mjd = (
        peak_mjd
        + phases * (1.0 + redshift)
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for band_index, band in enumerate(observed_bands):
        use = lightcurve["FLT"] == band
        colour = colours[band]

        ax.errorbar(
            lightcurve.loc[use, "MJD"],
            lightcurve.loc[use, "FLUXCAL"],
            yerr=lightcurve.loc[use, "FLUXCALERR"],
            fmt="o",
            ms=5,
            capsize=2,
            color=colour,
            label=f"{band} data",
        )

        mean_flux = model_mean[band_index]
        std_flux = model_std[band_index]

        ax.plot(
            model_mjd,
            mean_flux,
            color=colour,
            linewidth=2,
            label=f"{band} BayeSN",
        )

        ax.fill_between(
            model_mjd,
            mean_flux - std_flux,
            mean_flux + std_flux,
            color=colour,
            alpha=0.2,
        )

    ax.axhline(
        0,
        color="0.6",
        linewidth=1,
    )

    ax.set_xlabel("MJD")
    ax.set_ylabel("FLUXCAL")
    ax.set_title("BayeSN fit: Foundation 2016W")
    ax.legend(ncol=2)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_FILE,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("Saved plot to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
