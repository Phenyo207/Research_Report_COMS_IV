import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os


def load_seeds(folder_path, label, pattern="*.csv", moving_avg_window=1000):
    """
    Load up to 3 CSV files from a folder.
    - For s11: use ep_reward/ep_steps where ep_steps > 1, else ep_reward.
    - For sac2: use avg_reward directly.
    Apply moving average smoothing.
    """
    files = sorted(glob.glob(os.path.join(folder_path, pattern)))
    if len(files) < 3:
        raise ValueError(f"Found {len(files)} files in {folder_path}, expected at least 3.")

    rewards = []
    steps = []

    for f in files[:3]:
        df = pd.read_csv(f)

        if "total_steps" not in df.columns:
            raise ValueError(f"{f} must contain 'total_steps' column")

        # --- s11 logic ---
        if "s11" in folder_path.lower():
            required_cols = {"ep_reward", "ep_steps"}
            missing = required_cols - set(df.columns)
            if missing:
                raise ValueError(f"{f} missing required columns: {missing}")

            r = df["ep_reward"].copy()
            mask = df["ep_steps"] > 1
            r.loc[mask] = df.loc[mask, "ep_reward"] / df.loc[mask, "ep_steps"].replace(0, np.nan)
            r = r.fillna(0)

        # --- sac2 logic ---
        else:
            if "avg_reward" not in df.columns:
                raise ValueError(f"{f} missing 'avg_reward' column")
            r = df["avg_reward"].copy()

        # Apply moving average smoothing
        r = r.rolling(window=moving_avg_window, min_periods=1).mean()

        rewards.append(r)
        steps.append(df["total_steps"])

    return steps, rewards


def compute_mean_std(steps_list, rewards_list):
    """
    Interpolate rewards over a shared total_steps axis for averaging.
    """
    min_step = max(s.min() for s in steps_list)
    max_step = min(s.max() for s in steps_list)
    common_steps = np.linspace(min_step, max_step, 1000)

    interpolated = []
    for s, r in zip(steps_list, rewards_list):
        interp = np.interp(common_steps, s, r)
        interpolated.append(interp)

    mean = np.mean(interpolated, axis=0)
    std = np.std(interpolated, axis=0)

    return common_steps, mean, std


def plot_mean_std(results, save_path="reward_vs_steps.pdf"):
    """
    Plot mean ± std curves for reward vs total_steps.
    """
    plt.figure(figsize=(10, 6))
    plt.style.use("default")

    for steps, mean, std, label in results:
        plt.plot(steps, mean, linewidth=2, label=label)
        plt.fill_between(steps, mean - std, mean + std, alpha=0.25)

    # --- Y-axis range and labels ---
    plt.ylim(-10, 1)
    plt.yticks(np.arange(-10, 1.1, 1))

    plt.xlabel("Total Steps", fontsize=12)
    plt.ylabel("Reward", fontsize=12)
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(save_path, format="pdf")
    print(f"📄 Saved plot to: {os.path.abspath(save_path)}")
    plt.close()


def main():
    folders = {
        "sac_baseline_HER/logs": "SAC_BASELINE(STRAIGHTMAZE)",
        "logs": "SAC_WVF(STRAIGHTMAZE)",
    }

    results = []
    for folder, label in folders.items():
        steps, rewards = load_seeds(folder, label=label, moving_avg_window=1000)
        steps_common, mean, std = compute_mean_std(steps, rewards)
        results.append((steps_common, mean, std, label))

    plot_mean_std(results, "reward_vs_steps.pdf")
    print("✅ Plotted reward vs total_steps for s11 (ep_reward/ep_steps logic) and sac2 (avg_reward).")
    print("✅ Moving average window = 1000, y-axis range = −10 → 1.")


if __name__ == "__main__":
    main()
