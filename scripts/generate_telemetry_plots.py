import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_telemetry_plot():
    out_dir = root_dir / "docs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("🖥️ Generating AI Data Center Telemetry & Throttling Plot...")

    time_min = np.linspace(0, 60, 180)  # 1 hour simulation

    # GPU Temperatures
    t_alpha = 52.0 + 3.5 * np.sin(time_min / 5.0) + np.random.normal(0, 0.4, len(time_min))
    t_beta = 68.0 + 4.0 * np.cos(time_min / 8.0) + np.random.normal(0, 0.5, len(time_min))

    # Node gamma undergoes thermal surge at t=20 min, LangGraph triggers 15% clock throttling at t=25 min
    t_gamma = 65.0 + 0.8 * np.maximum(0, time_min - 5)
    t_gamma[time_min > 25] = (
        81.5
        - 0.45 * (time_min[time_min > 25] - 25)
        + np.random.normal(0, 0.4, np.sum(time_min > 25))
    )
    t_gamma[time_min > 42] = 73.8 + np.random.normal(0, 0.4, np.sum(time_min > 42))

    t_delta = 45.0 + 2.0 * np.sin(time_min / 10.0) + np.random.normal(0, 0.3, len(time_min))

    # Grid Carbon Intensity (gCO2/kWh)
    carbon_intensity = 160.0 - 55.0 * np.sin(time_min / 15.0)

    fig, axs = plt.subplots(2, 1, figsize=(13, 8), dpi=150, sharex=True)

    # Subplot 1: GPU Node Temperatures & Safety Bounds
    axs[0].plot(
        time_min, t_alpha, label="Node Alpha (H100 - CrossViT)", color="#3b82f6", linewidth=2.0
    )
    axs[0].plot(
        time_min, t_beta, label="Node Beta (H100 - Llama Pretrain)", color="#8b5cf6", linewidth=2.0
    )
    axs[0].plot(
        time_min,
        t_gamma,
        label="Node Gamma (A100 - Dynamic Throttled)",
        color="#ef4444",
        linewidth=2.4,
    )
    axs[0].plot(
        time_min, t_delta, label="Node Delta (A100 - Green Standby)", color="#10b981", linewidth=2.0
    )

    axs[0].axhline(
        80.0,
        color="#dc2626",
        linestyle="--",
        linewidth=1.8,
        label="Hardware Emergency Threshold (80.0°C)",
    )
    axs[0].axvline(
        25.0,
        color="#f59e0b",
        linestyle=":",
        linewidth=2.0,
        label="Agentic FastMCP Action: Clock Throttle -15%",
    )

    axs[0].set_title(
        "AI Cluster Node Telemetry: Real-Time Temperatures & Thermal Mitigation",
        fontsize=13,
        fontweight="bold",
    )
    axs[0].set_ylabel("GPU Temperature (°C)", fontsize=11)
    axs[0].grid(True, linestyle="--", alpha=0.5)
    axs[0].legend(loc="upper right", framealpha=0.9)

    # Subplot 2: Grid Carbon Intensity & Smart Job Dispatch
    axs[1].plot(
        time_min,
        carbon_intensity,
        color="#059669",
        linewidth=2.2,
        label="Grid Carbon Intensity (gCO2 / kWh)",
    )
    axs[1].fill_between(time_min, carbon_intensity, 100, color="#10b981", alpha=0.15)

    # Annotate batch job dispatch during clean energy valley
    axs[1].axvspan(
        15,
        38,
        color="#10b981",
        alpha=0.2,
        label="Green Compute Window (Opportunistic Batch Dispatch)",
    )
    axs[1].set_title(
        "Carbon-Aware AI Workload Scheduling (Green Energy Optimization)",
        fontsize=13,
        fontweight="bold",
    )
    axs[1].set_xlabel("Operational Timeline (minutes)", fontsize=11)
    axs[1].set_ylabel("Carbon Intensity (gCO2/kWh)", fontsize=11)
    axs[1].grid(True, linestyle="--", alpha=0.5)
    axs[1].legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    out_path = out_dir / "cluster_telemetry_simulation.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"✅ Data Center telemetry plot saved to '{out_path}'")


if __name__ == "__main__":
    generate_telemetry_plot()
