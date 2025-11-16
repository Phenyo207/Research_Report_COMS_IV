import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob, os


def load_goal_distance_vs_steps(folder_path, pattern="*.csv", max_episode=8570):
    """
    Load CSVs, compute cumulative total_steps, and extract goal_distance per episode.
    Returns a list of Series for averaging across seeds.
    """
    files = sorted(glob.glob(os.path.join(folder_path, pattern)))
    if len(files) < 3:
        raise ValueError(f"Expected ≥3 CSVs in {folder_path}, found {len(files)}")

    seeds = []
    for f in files[:3]:
        df = pd.read_csv(f)
        if not {"ep_steps", "goal_distance"}.issubset(df.columns):
            raise ValueError(f"{f} missing required columns 'ep_steps' or 'goal_distance'")

        # Truncate to same number of episodes
        df = df.iloc[:max_episode].reset_index(drop=True)

        # Compute cumulative total steps
        df["total_steps"] = df["ep_steps"].cumsum()

        # Smooth goal_distance for readability
        df["goal_distance_smooth"] = df["goal_distance"].rolling(window=500, min_periods=1).mean()

        seeds.append(df[["total_steps", "goal_distance_smooth"]])
    return seeds


def compute_mean_std_steps(seeds):
    """
    Interpolates goal_distance values on a common total_steps axis for averaging.
    """
    # Create a shared total_steps axis (up to the smallest max total_steps)
    min_max_steps = min(s["total_steps"].max() for s in seeds)
    step_axis = np.linspace(0, min_max_steps, 2000)

    interpolated = []
    for s in seeds:
        interp = np.interp(step_axis, s["total_steps"], s["goal_distance_smooth"])
        interpolated.append(interp)

    arr = np.array(interpolated)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    return step_axis, mean, std


def plot_results(results, save_path="goal_distance_vs_steps.pdf"):
    """
    Plot mean ± std for SAC(WVF) and SAC(BASELINE) vs total_steps.
    """
    plt.figure(figsize=(10, 6))
    plt.style.use("default")

    for label, (steps, mean, std) in results.items():
        plt.plot(steps, mean, linewidth=2, label=label)
        plt.fill_between(steps, mean - std, mean + std, alpha=0.25)

    plt.xlabel("Total Steps", fontsize=12)
    plt.ylabel("Goal Distance", fontsize=12)
    plt.legend()
    plt.grid(False)
    plt.tight_layout()

    plt.savefig(save_path, format="pdf")
    print(f"📄 Saved plot to: {os.path.abspath(save_path)}")
    plt.close()


def main():
    results = {}

    # --- SAC(WVF) ---
    sac2_seeds = load_goal_distance_vs_steps("sac2", max_episode=8570)
    steps, mean, std = compute_mean_std_steps(sac2_seeds)
    results["SAC(WVF)"] = (steps, mean, std)

    # --- SAC(BASELINE) ---
    s11_seeds = load_goal_distance_vs_steps("sac_baseline_HER/logs", max_episode=8570)
    steps, mean, std = compute_mean_std_steps(s11_seeds)
    results["logs"] = (steps, mean, std)

    # Plot and save
    plot_results(results)

    print("✅ Compared 'Goal Distance vs Total Steps' for SAC(WVF) and SAC(BASELINE)")
    print("✅ White background, solid lines, mean ± std shading")


if __name__ == "__main__":
    main()
