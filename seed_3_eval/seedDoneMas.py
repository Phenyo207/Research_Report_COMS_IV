import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os


def load_mastery(folder_path, pattern="*.csv", smooth_window=1000):
    """
    Load CSVs from a folder, extract 'mastery_score', and:
      - Keep first 1000 steps as-is (zeros remain zero)
      - After 1000 steps, forward-fill zeros with the last non-zero value
      - Apply moving average smoothing
    """
    files = sorted(glob.glob(os.path.join(folder_path, pattern)))
    if len(files) < 3:
        raise ValueError(f"Found {len(files)} files in {folder_path}, expected at least 3.")

    mastery_curves = []
    for f in files[:3]:
        df = pd.read_csv(f)
        if "mastery_score" not in df.columns:
            raise ValueError(f"{f} missing 'mastery_score' column")

        mastery = df["mastery_score"].fillna(0).copy()

        # Forward-fill zeros only after the first 1000 steps
        if len(mastery) > 1000:
            post1000 = mastery.iloc[1000:].copy()

            # Replace zeros with NaN so ffill works properly
            post1000 = post1000.replace(0, np.nan)
            post1000 = post1000.ffill().fillna(0)  # fill remaining NaNs (if all zeros) with 0

            mastery.iloc[1000:] = post1000

        # Apply moving average smoothing
        mastery_smooth = mastery.rolling(window=smooth_window, min_periods=1).mean()
        mastery_curves.append(mastery_smooth.reset_index(drop=True))

    return mastery_curves


def load_done_ratio(folder_path, pattern="*.csv"):
    """
    Load CSV files and extract 'done_action_ratio'.
    """
    files = sorted(glob.glob(os.path.join(folder_path, pattern)))
    if len(files) < 3:
        raise ValueError(f"Found {len(files)} files in {folder_path}, expected at least 3.")

    ratios = []
    for f in files[:3]:
        df = pd.read_csv(f)
        if "done_action_ratio" not in df.columns:
            raise ValueError(f"{f} missing 'done_action_ratio' column")
        ratios.append(df["done_action_ratio"].fillna(0).reset_index(drop=True))
    return ratios


def compute_mean_std(seeds):
    """
    Compute mean and standard deviation across seeds.
    """
    min_len = min(len(r) for r in seeds)
    data = np.array([r[:min_len] for r in seeds])
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    episodes = np.arange(min_len)
    return episodes, mean, std


def plot_combined(master_ep, master_mean, master_std,
                  done_ep, done_mean, done_std,
                  save_path="mastery_done_ffill_smoothed.pdf"):
    """
    Plot Mastery Score (forward-filled zeros after step 1000 + smoothed)
    and Done Action Ratio.
    """
    plt.figure(figsize=(10, 6))
    plt.style.use("default")

    plt.plot(master_ep, master_mean, linewidth=2, label="Mastery Score (ffill after 1000)", color="tab:blue")
    plt.fill_between(master_ep, master_mean - master_std, master_mean + master_std, alpha=0.25, color="tab:blue")

    plt.plot(done_ep, done_mean, linewidth=2, label="Done Action Ratio", color="tab:orange")
    plt.fill_between(done_ep, done_mean - done_std, done_mean + done_std, alpha=0.25, color="tab:orange")

    # Mark the change threshold
    #plt.axvline(x=1000, color='gray', linestyle='--', linewidth=1, alpha=0.6, label="f-fill start (1000 steps)")

    plt.ylim(0, 1.1)
    plt.yticks(np.arange(0, 1.2, 0.1))
    plt.title("Mastery Score & Done Action Ratio (SAC_WVF - STRAIGHTMAZE)", fontsize=13)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Metric Value", fontsize=12)
    plt.legend()
    plt.grid(False)
    plt.tight_layout()

    plt.savefig(save_path, format="pdf")
    plt.close()
    print(f"📄 Saved forward-filled + smoothed plot to: {os.path.abspath(save_path)}")


def main():
    folder = "sac2"

    mastery_seeds = load_mastery(folder, smooth_window=1000)
    master_ep, master_mean, master_std = compute_mean_std(mastery_seeds)

    done_seeds = load_done_ratio(folder)
    done_ep, done_mean, done_std = compute_mean_std(done_seeds)

    plot_combined(master_ep, master_mean, master_std, done_ep, done_mean, done_std)

    print("✅ Mastery: forward-filled zeros after 1000 steps (carry last non-zero value)")
    print("✅ Moving average smoothing (window=1000) applied")
    print("✅ Done Action Ratio unchanged")
    print("✅ Full episode range intact, shaded std")
    print("✅ Saved as 'mastery_done_ffill_smoothed.pdf'")


if __name__ == "__main__":
    main()
