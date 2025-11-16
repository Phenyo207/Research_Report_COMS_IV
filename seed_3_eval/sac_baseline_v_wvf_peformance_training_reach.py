import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os


def load_csvs(folder_path, label, pattern="*.csv", moving_avg_window=100, warmup_cut=300):
    """
    Load all CSVs from a folder (up to 3 recommended).
    - Use avg_reward column.
    - Drop first `warmup_cut` episodes.
    - Renumber remaining episodes to start from 1.
    - Apply moving average smoothing.
    """
    files = sorted(glob.glob(os.path.join(folder_path, pattern)))
    if not files:
        raise ValueError(f"No CSV files found in {folder_path}")

    rewards, episodes = [], []

    for f in files[:3]:
        df = pd.read_csv(f)

        if "avg_reward" not in df.columns:
            raise ValueError(f"{f} missing 'avg_reward' column")

        r = df["avg_reward"].copy()

        # --- Drop warmup episodes and reset index ---
        if len(r) <= warmup_cut:
            raise ValueError(f"{f} has only {len(r)} episodes; cannot drop {warmup_cut}.")
        r = r.iloc[warmup_cut:].reset_index(drop=True)

        # Apply moving average smoothing
        r = r.rolling(window=moving_avg_window, min_periods=1).mean()

        # Renumber episodes: 301→1, 302→2, etc.
        ep = np.arange(1, len(r) + 1)

        rewards.append(r)
        episodes.append(ep)

    return episodes, rewards


def compute_mean_std(episodes_list, rewards_list, max_episode_cap=None):
    """
    Interpolate rewards over a shared episode axis for averaging.
    Optionally cap episode range.
    """
    min_ep = max(e.min() for e in episodes_list)
    max_ep = min(e.max() for e in episodes_list)
    if max_episode_cap:
        max_ep = min(max_ep, max_episode_cap)

    common_episodes = np.linspace(min_ep, max_ep, 1000)
    interpolated = [np.interp(common_episodes, e, r) for e, r in zip(episodes_list, rewards_list)]
    mean = np.mean(interpolated, axis=0)
    std = np.std(interpolated, axis=0)
    return common_episodes, mean, std


def plot_mean_std(results, save_path="avg_reward_vs_episode.pdf"):
    """
    Plot mean ± std for avg_reward vs episode (clean background, no grid).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_alpha(0)  # transparent figure background
    ax.set_facecolor("white")  # white axes background

    for episodes, mean, std, label in results:
        ax.plot(episodes, mean, linewidth=2, label=label)
        ax.fill_between(episodes, mean - std, mean + std, alpha=0.25)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Average Reward", fontsize=12)
    ax.legend()
    ax.grid(False)  # no grid
    ax.set_axisbelow(False)
    plt.tight_layout()
    plt.savefig(save_path, format="pdf", bbox_inches="tight", transparent=True)
    print(f"📄 Saved plot to: {os.path.abspath(save_path)}")
    plt.close()


def main():
    moving_avg_window = 100
    warmup_cut = 300
    folders = {
        "reach1": "SAC-WVF",
        "reach2": "SAC-BASELINE",
    }

    results = []

    # --- Load reach1 first to determine cap ---
    reach1_eps, reach1_rewards = load_csvs(
        "reach1", "SAC-WVF", moving_avg_window=moving_avg_window, warmup_cut=warmup_cut
    )
    max_ep_cap = min(max(e.max() for e in reach1_eps), 10000)
    ep_common, mean, std = compute_mean_std(reach1_eps, reach1_rewards, max_episode_cap=max_ep_cap)
    results.append((ep_common, mean, std, "SAC-WVF"))

    # --- Load reach2 and cap to reach1's max ---
    reach2_eps, reach2_rewards = load_csvs(
        "reach2", "SAC-BASELINE", moving_avg_window=moving_avg_window, warmup_cut=warmup_cut
    )
    ep_common, mean, std = compute_mean_std(reach2_eps, reach2_rewards, max_episode_cap=max_ep_cap)
    results.append((ep_common, mean, std, "SAC-BASELINE"))

    plot_mean_std(results, "avg_reward_vs_episode.pdf")

    print(f"✅ Moving average = {moving_avg_window}")
    print(f"✅ Warmup cut = first {warmup_cut} episodes dropped")
    print(f"✅ Episodes renumbered from {warmup_cut+1} → 1")
    print(f"✅ Capped episodes to {max_ep_cap} (from reach1)")
    print("✅ Background removed and grid disabled")
    print("✅ Saved as 'avg_reward_vs_episode.pdf'")


if __name__ == "__main__":
    main()
