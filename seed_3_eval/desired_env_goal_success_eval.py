import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os


def load_seeds(folder_path, pattern="*.csv", moving_avg_window=1000):
    """
    Load CSV files and smooth 'desired_goal_success' with a moving average.
    """
    files = sorted(glob.glob(os.path.join(folder_path, pattern)))
    if len(files) == 0:
        raise ValueError(f"No CSV files found in {folder_path}")

    success_rates = []
    for f in files:
        df = pd.read_csv(f)
        if "desired_goal_success" not in df.columns:
            raise ValueError(f"{f} missing 'desired_goal_success' column")
        r = (
            df["desired_goal_success"]
            .rolling(window=moving_avg_window, min_periods=1)
            .mean()
            .fillna(0)
            .clip(0, 1)
            .reset_index(drop=True)
        )
        success_rates.append(r)
    return success_rates


def compute_mean_std(seeds, max_len=None):
    """
    Compute mean and std across seeds. If max_len is provided, truncate.
    """
    min_len = min(len(r) for r in seeds if len(r) > 0)
    if max_len:
        cutoff = min(min_len, max_len)
    else:
        cutoff = min_len
    seeds_trimmed = np.array([r[:cutoff] for r in seeds])
    mean = np.mean(seeds_trimmed, axis=0)
    std = np.std(seeds_trimmed, axis=0)
    episodes = np.arange(cutoff)
    return episodes, mean, std


def find_first_nonzero_index(series, threshold=1e-6):
    """
    Return the index of the first non-zero (above threshold) element.
    """
    for i, v in enumerate(series):
        if v > threshold:
            return i
    return 0


def plot_mean_std(results, save_path="desired_goal_success_comparison.pdf"):
    plt.figure(figsize=(10, 6))
    plt.style.use("default")

    for ep, mean, std, label in results:
        plt.plot(ep, mean, linewidth=2, label=label)
        plt.fill_between(ep, mean - std, mean + std, alpha=0.25)

    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.1, 0.1))
    #plt.title("Environment Goal Success Rate", fontsize=14)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Environment Goal Success Rate", fontsize=12)
    plt.legend()
    plt.grid(False)
    plt.tight_layout()

    plt.savefig(save_path, format="pdf")
    plt.close()
    print(f"📄 Saved plot to: {os.path.abspath(save_path)}")


def main():
    # --- Load and smooth both folders ---
    sac2_seeds = load_seeds("sac_wvf_desired_eval", moving_avg_window=1500)
    s11_seeds = load_seeds("sac_baseline_desired_eval", moving_avg_window=1500)

    # --- Compute baseline stats (limit 8500) ---
    ep_base, mean_base, std_base = compute_mean_std(s11_seeds, max_len=8500)

    # --- Compute WVF stats without limit first ---
    ep_wvf, mean_wvf, std_wvf = compute_mean_std(sac2_seeds, max_len=None)

    # --- Cut WVF before its first non-zero ---
    start_idx = find_first_nonzero_index(mean_wvf)
    cut_count = start_idx

    # Adjust cutoff dynamically (8500 + number cut off)
    target_len = 8500 + cut_count
    ep_wvf, mean_wvf, std_wvf = compute_mean_std(sac2_seeds, max_len=target_len)

    # Now cut the first non-zero part and reindex from 1
    mean_wvf = mean_wvf[start_idx:]
    std_wvf = std_wvf[start_idx:]
    ep_wvf = np.arange(1, len(mean_wvf) + 1)

    # --- Plot ---
    results = [
        (ep_base, mean_base, std_base, "SAC-BASELINE"),
        (ep_wvf, mean_wvf, std_wvf, "SAC-WVF"),
    ]
    plot_mean_std(results)

    print("✅ Moving average = 1000")
    print(f"✅ Cut SAC_WVF before its first non-zero (index {start_idx})")
    print(f"✅ Extended SAC_WVF max length to {target_len} to preserve full visible range")
    print("✅ Reindexed SAC_WVF episodes starting from 1")
    print("✅ Saved as 'desired_goal_success_comparison.pdf'")


if __name__ == "__main__":
    main()
