from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_CSV = PROJECT_ROOT / "results" / "alerts" / "alert_fit_summary.csv"


def main():
    df = pd.read_csv(SUMMARY_CSV)
    # Guard against a stray duplicated header row (can happen if a job was
    # resumed and re-wrote a header partway through the file)
    df = df[df["SNID"] != "SNID"].copy()
    df["max_rhat"] = pd.to_numeric(df["max_rhat"], errors="coerce")
    df["n_obs"] = pd.to_numeric(df["n_obs"], errors="coerce")
    df["elapsed_s"] = pd.to_numeric(df["elapsed_s"], errors="coerce")

    print(f"Total fits logged: {len(df)}")
    print()
    print("Status breakdown:")
    print(df["status"].value_counts())
    print()

    converged = df[df["status"] == "converged"]
    marginal = df[df["status"] == "marginal"]

    print(f"Converged: {len(converged)}  |  max_rhat range: "
          f"{converged['max_rhat'].min():.3f} - {converged['max_rhat'].max():.3f}")

    if len(marginal):
        print(f"Marginal:  {len(marginal)}  |  max_rhat range: "
              f"{marginal['max_rhat'].min():.3f} - {marginal['max_rhat'].max():.3f}")
        print()
        print("Marginal fits sorted by severity (worst r_hat first):")
        print(
            marginal[["SNID", "max_rhat", "n_obs", "elapsed_s"]]
            .sort_values("max_rhat", ascending=False)
            .to_string(index=False)
        )

    print()
    print("Does convergence correlate with number of observations?")
    print(df.groupby("status")["n_obs"].describe()[["mean", "min", "max"]])


if __name__ == "__main__":
    main()
