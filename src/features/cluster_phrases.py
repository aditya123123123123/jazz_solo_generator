from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

FEATURES_CSV = Path(__file__).parents[2] / "data" / "processed" / "phrase_features_all.csv"
OUTPUT_CSV = Path(__file__).parents[2] / "data" / "processed" / "phrase_clusters_all.csv"
OUTPUTS_DIR = Path(__file__).parents[2] / "outputs"

CLUSTER_FEATURES = [
    "length_beats", "num_notes", "note_density", "pitch_mean",
    "pitch_range", "pitch_std", "rest_ratio", "phrase_duration",
]
N_CLUSTERS = 64


def load_features(path=FEATURES_CSV):
    return pd.read_csv(path)


def cluster_phrases(df, n_clusters=N_CLUSTERS, random_state=42):
    X = df[CLUSTER_FEATURES].values
    X_scaled = StandardScaler().fit_transform(X)
    labels = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit_predict(X_scaled)
    df = df.copy()
    df["phrase_token"] = [f"PHRASE_{i:02d}" for i in labels]
    return df


def print_stats(df):
    comp = df.groupby(["phrase_token", "performer"]).size().unstack(fill_value=0)
    comp["total"] = comp.sum(axis=1)

    sizes = comp["total"]
    print(f"Cluster size distribution:")
    print(f"  min:  {sizes.min()}")
    print(f"  max:  {sizes.max()}")
    print(f"  mean: {sizes.mean():.1f}")
    print()

    parker_col = "Charlie Parker"
    davis_col = "Miles Davis"

    print("Top 3 clusters by Charlie Parker phrase count:")
    print(comp.nlargest(3, parker_col)[[parker_col, davis_col, "total"]].to_string())
    print()

    print("Top 3 clusters by Miles Davis phrase count:")
    print(comp.nlargest(3, davis_col)[[parker_col, davis_col, "total"]].to_string())
    print()

    comp["parker_pct"] = comp[parker_col] / comp["total"]
    comp["davis_pct"] = comp[davis_col] / comp["total"]

    parker_dom = comp[comp["parker_pct"] > 0.70]
    davis_dom = comp[comp["davis_pct"] > 0.70]
    mixed = comp[(comp["parker_pct"] <= 0.70) & (comp["davis_pct"] <= 0.70)]

    print(f"Parker-dominant clusters (>70% Parker): {len(parker_dom)}")
    if not parker_dom.empty:
        print(parker_dom[[parker_col, davis_col, "total", "parker_pct"]].to_string())
    print()

    print(f"Davis-dominant clusters (>70% Davis): {len(davis_dom)}")
    if not davis_dom.empty:
        print(davis_dom[[parker_col, davis_col, "total", "davis_pct"]].to_string())
    print()

    print(f"Mixed clusters (neither Parker nor Davis >70%): {len(mixed)}")


def save_plot(df):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    comp = df.groupby(["phrase_token", "performer"]).size().unstack(fill_value=0)
    comp["total"] = comp.sum(axis=1)
    comp = comp.sort_values("total", ascending=False)

    parker = comp.get("Charlie Parker", pd.Series(0, index=comp.index)).values
    davis = comp.get("Miles Davis", pd.Series(0, index=comp.index)).values
    x = range(len(comp))

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.bar(x, parker, label="Charlie Parker", color="steelblue", alpha=0.8)
    ax.bar(x, davis, bottom=parker, label="Miles Davis", color="coral", alpha=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(comp.index, rotation=90, fontsize=7)
    ax.set_xlabel("Phrase Token")
    ax.set_ylabel("Phrase Count")
    ax.set_title("Phrase Cluster Composition by Artist (sorted by total size)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "phrase_clusters_by_artist.png", dpi=150)
    plt.close(fig)
    print("  phrase_clusters_by_artist.png")


if __name__ == "__main__":
    df = load_features()
    df = cluster_phrases(df)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_CSV}")
    print()

    print_stats(df)

    print("Plot saved to outputs/:")
    save_plot(df)
