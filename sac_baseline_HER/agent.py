import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from model import Actor, Critic


def soft_update(target, source, tau):
    """Soft update of target network parameters."""
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(tau * s.data + (1.0 - tau) * t.data)


class Agent:
    def __init__(self, state_dim, action_space, goal_dim,
                 gamma=0.98, tau=0.005, lr=3e-4, alpha=0.2,
                 her_k=4, success_threshold=0.05):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.her_k = her_k
        self.success_threshold = success_threshold

        # ---- Networks ----
        self.critic = Critic(state_dim, action_space.shape[0], goal_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_space.shape[0], goal_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)

        self.actor = Actor(state_dim, action_space.shape[0], goal_dim, action_space).to(self.device)
        self.actor_optim = Adam(self.actor.parameters(), lr=lr)

        # ---- Entropy tuning ----
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = Adam([self.log_alpha], lr=lr)
        self.target_entropy = -float(action_space.shape[0])
        self.alpha = alpha

    # ---------------------------------------------------------
    # Reward: +1 if success within threshold, else −0.1
    # ---------------------------------------------------------
    def compute_reward(self, ag, g):
        dist = np.linalg.norm(ag - g)
        return 1.0 if dist <= self.success_threshold else -0.1

    # ---------------------------------------------------------
    # Action selection
    # ---------------------------------------------------------
    def select_action(self, s, g, eval_mode=False):
        s_g = torch.tensor(np.concatenate([s, g]), dtype=torch.float32,
                           device=self.device).unsqueeze(0)
        with torch.no_grad():
            if eval_mode:
                _, _, action = self.actor.sample(s_g, deterministic=True)
            else:
                action, _, _ = self.actor.sample(s_g, deterministic=False)
        return action.detach().cpu().numpy().flatten()

    # ---------------------------------------------------------
    # SAC parameter update
    # ---------------------------------------------------------
    def update(self, replay_buffer, batch_size):
        sg, act, rew, next_sg, done = replay_buffer.sample(batch_size)
        sg = torch.tensor(sg, dtype=torch.float32, device=self.device)
        act = torch.tensor(act, dtype=torch.float32, device=self.device)
        rew = torch.tensor(rew, dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_sg = torch.tensor(next_sg, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device).unsqueeze(-1)

        with torch.no_grad():
            next_a, next_logp, _ = self.actor.sample(next_sg)
            q1_next, q2_next = self.critic_target(next_sg, next_a)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_logp
            target_q = rew + (1 - done) * self.gamma * q_next

        q1, q2 = self.critic(sg, act)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        new_a, logp, _ = self.actor.sample(sg)
        q1_new, q2_new = self.critic(sg, new_a)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * logp - q_new).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp().item()

        soft_update(self.critic_target, self.critic, self.tau)
        return float(critic_loss.item()), float(actor_loss.item())

    # ---------------------------------------------------------
    # HER relabeling for goal-conditioned training
    # ---------------------------------------------------------
    def her_relabel_episode(self, transitions, replay_buffer):
        T = len(transitions)
        for t in range(T):
            s, a, r, s_next, g, ag, next_ag, done = transitions[t]
            for _ in range(self.her_k):
                future = np.random.randint(t, T)
                g_new = transitions[future][6]
                r_new = self.compute_reward(next_ag, g_new)
                replay_buffer.store(s, a, r_new, s_next, g_new, ag, next_ag, done)

    # ---------------------------------------------------------
    # Save checkpoints
    # ---------------------------------------------------------
    def save_checkpoint(self, path="checkpoints"):
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), f"{path}/actor_sac.pt")
        torch.save(self.critic.state_dict(), f"{path}/critic_sac.pt")
        torch.save(self.critic_target.state_dict(), f"{path}/critic_target_sac.pt")
