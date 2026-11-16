"""Shared plotting style and table writers."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


def save_table(df: pd.DataFrame, path_stem, float_fmt="{:.3f}"):
    """Write CSV (full precision) and Markdown (rounded) versions of a table."""
    df.to_csv(f"{path_stem}.csv", index=False)
    md = df.copy()
    for c in md.columns:
        if pd.api.types.is_float_dtype(md[c]):
            md[c] = md[c].map(lambda v: "—" if pd.isna(v) else float_fmt.format(v))
    md = md.fillna("—").astype(str)
    lines = ["| " + " | ".join(md.columns) + " |", "|" + "---|" * len(md.columns)]
    lines += ["| " + " | ".join(r) + " |" for r in md.values]
    open(f"{path_stem}.md", "w").write("\n".join(lines) + "\n")
