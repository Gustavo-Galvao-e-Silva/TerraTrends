import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Factors affecting growth rate (replace synthetic coefficients with your model output / CSV merge)
FACTORS = (
    "TOT_POP",
    "TOT_MALE",
    "TOT_FEMALE",
    "Unemployment_Rate",
    "Per_Capita_Personal_Income",
    "Real_GDP",
    "Percent_Change_Real_GDP",
    "Bachelor_Degree_or_Higher_Pct",
)
YEARS = (2002, 2003, 2004, 2005)


def factor_label(name: str) -> str:
    return name.replace("_", " ")


rng = np.random.default_rng(42)
raw = rng.normal(0, 0.2, size=(len(YEARS), len(FACTORS)))
raw = np.clip(raw, -0.55, 0.72)
data = [
    {"Year": YEARS[i], "factor": FACTORS[j], "coefficient": round(float(raw[i, j]), 2)}
    for i in range(len(YEARS))
    for j in range(len(FACTORS))
]

df = pd.DataFrame(data)
df["abs_coeff"] = df["coefficient"].abs()

factor_avg = df.groupby("factor")["abs_coeff"].mean().sort_values(ascending=False)
factor_avg.index = factor_avg.index.map(factor_label)

sns.set_theme(style="white", context="talk", font="sans-serif")
mpl.rcParams.update(
    {
        "figure.facecolor": "#f4f2ef",
        "axes.facecolor": "#f4f2ef",
        "axes.edgecolor": "#2d2a26",
        "axes.labelcolor": "#2d2a26",
        "axes.titleweight": "semibold",
        "axes.titlesize": 16,
        "text.color": "#2d2a26",
        "xtick.labelcolor": "#3d3a36",
        "ytick.labelcolor": "#3d3a36",
        "figure.dpi": 120,
        "savefig.facecolor": "#f4f2ef",
        "savefig.edgecolor": "none",
        "font.size": 11,
    }
)

n_fac = len(factor_avg)
crest = sns.color_palette("crest", n_colors=n_fac)
flare = sns.color_palette("flare", n_colors=n_fac)

# --- 1. Horizontal bar: mean absolute contribution ---
fig, ax = plt.subplots(figsize=(12.5, 7.2))
y_pos = np.arange(len(factor_avg))
bars = ax.barh(
    y_pos,
    factor_avg.values,
    height=0.72,
    color=crest,
    edgecolor="white",
    linewidth=1.1,
    zorder=3,
)
ax.set_yticks(y_pos)
ax.set_yticklabels(factor_avg.index, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel("Mean absolute contribution", labelpad=10, fontsize=12)
ax.set_title(
    "Average factor importance for growth rate (LSTM proxy)",
    pad=16,
    fontsize=16,
    fontweight="semibold",
)
ax.xaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
ax.set_axisbelow(True)
sns.despine(ax=ax, left=False, bottom=False, top=True, right=True)
pad = max(factor_avg.max() * 0.06, 0.02)
for bar, v in zip(bars, factor_avg.values):
    ax.text(
        v + pad * 0.15,
        bar.get_y() + bar.get_height() / 2,
        f"{v:.2f}",
        va="center",
        ha="left",
        fontsize=9.5,
        color="#4a4742",
    )
ax.set_xlim(0, factor_avg.max() * 1.2)
plt.tight_layout()
plt.show()

# --- 2. Donut: share of total mean |coefficient| ---
fig, ax = plt.subplots(figsize=(9, 8.5))
sizes = factor_avg.values
explode = [0.02 + 0.008 * i for i in range(len(sizes))]
wedges, texts, autotexts = ax.pie(
    sizes,
    labels=None,
    autopct=lambda p: f"{p:.1f}%" if p > 5 else "",
    pctdistance=0.78,
    explode=explode,
    colors=flare,
    startangle=90,
    counterclock=False,
    wedgeprops=dict(width=0.42, edgecolor="#f4f2ef", linewidth=2),
    textprops=dict(color="#2d2a26", fontsize=9.5),
)
for autotext in autotexts:
    autotext.set_fontweight("semibold")
    autotext.set_color("#1a1816")
ax.legend(
    wedges,
    factor_avg.index,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=False,
    fontsize=9,
    title="Factor",
    title_fontproperties=dict(weight="semibold", size=11),
)
centre = plt.Circle((0, 0), 0.32, fc="#f4f2ef", ec="#d8d4ce", linewidth=1)
ax.add_patch(centre)
ax.text(0, 0.02, "Share of\navg |β|", ha="center", va="center", fontsize=11, color="#5c5852", linespacing=1.2)
ax.set_title("Growth drivers: contribution mix", pad=20, fontsize=16, fontweight="semibold")
ax.set_aspect("equal")
plt.tight_layout()
plt.show()

# --- 3. Waterfall: cumulative mean coefficients ---
wf = df.groupby("factor")["coefficient"].mean().sort_values(ascending=False)
wf.index = wf.index.map(factor_label)
values = wf.values
labels = wf.index
cumulative = np.cumsum(values)
starts = np.insert(cumulative[:-1], 0, 0)

norm = mpl.colors.Normalize(vmin=values.min(), vmax=values.max())
cmap = mpl.colormaps["RdYlGn"]
bar_colors = [cmap(norm(v)) for v in values]

fig, ax = plt.subplots(figsize=(12.5, 7.2))
for i in range(len(values)):
    ax.barh(
        labels[i],
        values[i],
        left=starts[i],
        height=0.68,
        color=bar_colors[i],
        edgecolor="white",
        linewidth=1.05,
        zorder=2,
    )
ax.axvline(0, color="#2d2a26", linewidth=1.2, zorder=1)
ax.set_xlabel("Position along cumulative mean coefficient", labelpad=10, fontsize=12)
ax.set_title(
    "Waterfall: stacked mean factor effects on growth",
    pad=16,
    fontsize=16,
    fontweight="semibold",
)
ax.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax.set_axisbelow(True)
sns.despine(ax=ax, top=True, right=True)
plt.tight_layout()
plt.show()

# --- 4. Heatmap: diverging scale centered at 0 ---
pivot = df.pivot(index="factor", columns="Year", values="coefficient")
pivot.index = pivot.index.map(factor_label)
vmax = np.abs(pivot.to_numpy()).max()

fig, ax = plt.subplots(figsize=(10.5, 8.2))
sns.heatmap(
    pivot,
    cmap=sns.diverging_palette(220, 28, s=75, l=55, as_cmap=True),
    center=0,
    vmin=-vmax,
    vmax=vmax,
    annot=True,
    fmt=".2f",
    annot_kws={"size": 9.5, "weight": "medium", "color": "#1a1816"},
    linewidths=1.2,
    linecolor="#f4f2ef",
    cbar_kws={
        "label": "Coefficient",
        "shrink": 0.82,
        "pad": 0.02,
        "aspect": 22,
    },
    ax=ax,
)
ax.set_xlabel("Year", labelpad=10, fontsize=12)
ax.set_ylabel("")
ax.set_title("Growth-rate factors over time", pad=16, fontsize=16, fontweight="semibold")
plt.setp(ax.get_xticklabels(), rotation=0)
plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9.5)
cbar = ax.collections[0].colorbar
cbar.ax.set_ylabel(cbar.ax.get_ylabel(), labelpad=12)
cbar.outline.set_linewidth(0)
plt.tight_layout()
plt.show()
