import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os


def load_seeds(folder_path, pattern="*.csv", moving_avg_window=1500):
    """
    Load CSV files from a folder.
    - 'mastery_score' is forward-filled between non-zero points, then smoothed with a moving average.
    - 'done_action_ratio' is used directly as-is (no filtering).
    """
    files = sorted(glob.glob(os.path.join(folder_path, pattern)))
    if len(files) < 3:
        raise ValueError(f"Found {len(files)} files in {folder_path}, expected at least 3.")

    mastery_curves = []
    done_ratio_curves = []

    for f in files[:3]:
        df = pd.read_csv(f)
        if "mastery_score" not in df.columns or "done_action_ratio" not in df.columns:
            raise ValueError(f"{f} missing required columns ('mastery_score' or 'done_action_ratio')")

        mastery = df["mastery_score"].copy()

        # --- Forward-fill mastery between non-zero values ---
        last_value = 0
        for i in range(len(mastery)):
            if mastery.iloc[i] > 0:
                last_value = mastery.iloc[i]
            else:
                mastery.iloc[i] = last_value

        # --- Apply moving average smoothing ---
        mastery_smoothed = mastery.rolling(window=moving_avg_window, min_periods=1).mean()

        mastery_curves.append(mastery_smoothed.reset_index(drop=True))
        done_ratio_curves.append(df["done_action_ratio"].reset_index(drop=True))

    return mastery_curves, done_ratio_curves


def compute_mean_std(seeds):
    """
    Compute mean ± std for given list of Series.
    Truncates to minimum length across seeds.
    """
    min_len = min(len(r) for r in seeds)
    seeds_trimmed = np.array([r[:min_len] for r in seeds])

    mean = np.mean(seeds_trimmed, axis=0)
    std = np.std(seeds_trimmed, axis=0)
    episodes = np.arange(min_len)

    return episodes, mean, std


def plot_mastery_and_done_ratio(ep_mastery, mean_mastery, std_mastery,
                                ep_done, mean_done, std_done,
                                label_mastery="Mastery Score",
                                label_done="Done Action Ratio",
                                save_path="mastery_done_ratio.pdf"):
    """
    Plot mastery_score (smoothed + forward-filled) and done_action_ratio (raw)
    on the same figure, and save as PDF.
    """
    plt.figure(figsize=(10, 6))
    plt.style.use("default")  # clean white background

    # Mastery curve
    plt.plot(ep_mastery, mean_mastery, linewidth=2, label=label_mastery, color="tab:blue")
    plt.fill_between(ep_mastery, mean_mastery - std_mastery, mean_mastery + std_mastery,
                     alpha=0.25, color="tab:blue")

    # Done action ratio curve (solid line)
    plt.plot(ep_done, mean_done, linewidth=2, label=label_done, color="tab:orange")
    plt.fill_between(ep_done, mean_done - std_done, mean_done + std_done,
                     alpha=0.25, color="tab:orange")
#
    plt.ylim(0, 1.1)
    plt.yticks(np.arange(0, 1.2, 0.1))

    #plt.title("Mastery Score (Smoothed) and Done Action Ratio Across Seeds", fontsize=14)
    plt.xlabel("Episodes", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.legend()
    plt.grid(False)
    plt.tight_layout()

    # Save as PDF
    plt.savefig(save_path, format="pdf")
    print(f"📄 Saved plot to: {os.path.abspath(save_path)}")

    plt.close()  # free memory


def main():
    folder = "logs"  # Folder containing CSVs

    # Load mastery (forward-filled + smoothed) and done_action_ratio (raw)
    mastery_seeds, done_ratio_seeds = load_seeds(folder, moving_avg_window=1500)

    # Compute stats for mastery and done_action_ratio
    ep_m, mean_m, std_m = compute_mean_std(mastery_seeds)
    ep_d, mean_d, std_d = compute_mean_std(done_ratio_seeds)

    # Plot both curves together and save as PDF
    plot_mastery_and_done_ratio(ep_m, mean_m, std_m, ep_d, mean_d, std_d,
                                save_path="mastery_done_ratio.pdf")

    print("✅ 'mastery_score' forward-filled between non-zero values, smoothed (window=1500)")
    print("✅ 'done_action_ratio' used as-is (no filtering)")
    print("✅ Displayed mean ± std for both metrics as solid lines")
    print("✅ Y-axis scaled 0 → 1.1 with step 0.1")
    print("✅ Clean white background (no grid)")
    print("✅ Saved figure as 'mastery_done_ratio.pdf'")


if __name__ == "__main__":
    main()
