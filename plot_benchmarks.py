import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


dense = pd.read_csv("dense_results.csv")


dense_plot = dense[dense["method"].isin(["Baseline", "OptTranspose", "OptBlocked"])]

plt.figure()
for method in ["Baseline", "OptTranspose", "OptBlocked"]:
    df_m = dense_plot[dense_plot["method"] == method]
    plt.plot(df_m["N"], df_m["time_ms"], marker="o", label=method)

plt.xlabel("Matrix size N")
plt.ylabel("Time [ms]")
plt.title("Dense matrix multiplication: time vs N")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "dense_time_vs_N.png")


sparse = pd.read_csv("sparse_results.csv")

plt.figure()
plt.plot(sparse["density"] * 100.0, sparse["dense_time_ms"], marker="o", label="Dense")
plt.plot(sparse["density"] * 100.0, sparse["csr_time_ms"], marker="o", label="CSR")
plt.xlabel("Density [% non-zero]")
plt.ylabel("Time [ms]")
plt.title("Sparse vs dense (N = 1000)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "sparse_time_vs_density.png")

print("Plots saved in 'plots' folder:")
print(" - plots/dense_time_vs_N.png")
print(" - plots/sparse_time_vs_density.png")
