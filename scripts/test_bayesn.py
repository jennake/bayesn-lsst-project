from bayesn import SEDmodel
import pickle
from pathlib import Path

from bayesn import SEDmodel


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent / "results" / "terminal_test"


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading T21 BayeSN model...")

    model = SEDmodel(
        load_model="T21_model"
    )

    filt_map = {
        "g": "g_PS1",
        "r": "r_PS1",
        "i": "i_PS1",
        "z": "z_PS1",
    }

    print("Starting fit...")

    samples, sn_props = model.fit_from_file(
        model.example_lc,
        filt_map=filt_map,
        chain_method="sequential",
    )

    print("Fit completed")
    print("SN properties:", sn_props)
    print("Parameters:", samples.keys())

    chains_file = OUTPUT_DIR / "chains.pkl"

    with chains_file.open("wb") as file:
        pickle.dump(samples, file)

    print("Saved chains to:", chains_file)


if __name__ == "__main__":
    main()
