"""
Original IPPO — standard Independent PPO (no global critic).

Each agent's critic sees only its own local observation.
Same actor structure as Improved IPPO but with a local value function.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from Env.config import EnvConfig


class LocalCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class ActorNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        h = self.net(obs)
        mu = torch.sigmoid(self.mu_head(h))
        std = self.log_std.exp().expand_as(mu)
        return mu, std

    def get_dist(self, obs):
        mu, std = self(obs)
        return Normal(mu, std + 1e-6)


class RolloutBuffer:
    def __init__(self):
        self.obs, self.actions, self.log_probs = [], [], []
        self.rewards, self.dones, self.values = [], [], []

    def store(self, obs, actions, log_probs, rewards, done, values):
        self.obs.append(obs); self.actions.append(actions)
        self.log_probs.append(log_probs); self.rewards.append(rewards)
        self.dones.append(done); self.values.append(values)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.obs)


class IPPO:
    """Standard IPPO with local critic only."""

    def __init__(self, n_agents: int, obs_dim: int = 16, act_dim: int = 2,
                 actor_lr: float = 3e-4, critic_lr: float = 1e-3,
                 lr: float = 3e-4,
                 gamma: float = 0.99, lam: float = 0.95,
                 clip_eps: float = 0.2, entropy_coeff: float = 0.01,
                 n_epochs: int = 4, batch_size: int = 64,
                 cfg: Optional[EnvConfig] = None, device: str = "cpu"):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.entropy_coeff = entropy_coeff
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.cfg = cfg or EnvConfig()
        self.device = torch.device(device)

        self.w1, self.w2, self.w3 = 1.0, 0.1, 0.05

        a_lr = actor_lr if actor_lr != 3e-4 or lr == 3e-4 else lr
        c_lr = critic_lr if critic_lr != 1e-3 or lr == 3e-4 else lr
        self.actor = ActorNet(obs_dim, act_dim).to(self.device)
        self.critic = LocalCritic(obs_dim).to(self.device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=a_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=c_lr)
        self.buffer = RolloutBuffer()

    @torch.no_grad()
    def select_actions(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        obs_t = torch.FloatTensor(obs).to(self.device)
        dist = self.actor.get_dist(obs_t)
        actions = torch.clamp(dist.sample(), 0.0, 1.0)
        log_probs = dist.log_prob(actions).sum(-1)
        return actions.cpu().numpy(), log_probs.cpu().numpy()

    @torch.no_grad()
    def get_values(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.FloatTensor(obs).to(self.device)
        return self.critic(obs_t).cpu().numpy()

    def compute_rewards(self, f1: float, energies: np.ndarray,
                        collisions: np.ndarray) -> np.ndarray:
        r = self.w1 * f1 - self.w2 * energies / self.cfg.E_ref - self.w3 * collisions
        return r.astype(np.float32)

    def compute_gae(self):
        buf = self.buffer
        T = len(buf.rewards)
        advantages = np.zeros((T, self.n_agents), dtype=np.float32)
        gae = np.zeros(self.n_agents, dtype=np.float32)
        for t in reversed(range(T)):
            next_val = buf.values[t + 1] if t + 1 < T else buf.values[-1]
            delta = buf.rewards[t] + self.gamma * next_val * (1 - buf.dones[t]) - buf.values[t]
            gae = delta + self.gamma * self.lam * (1 - buf.dones[t]) * gae
            advantages[t] = gae
        returns = advantages + np.stack(buf.values[:T])
        return advantages, returns

    def update(self) -> Dict[str, float]:
        if len(self.buffer) < 2:
            self.buffer.clear()
            return {"policy_loss": 0, "value_loss": 0}

        advantages, returns = self.compute_gae()
        all_obs = np.stack(self.buffer.obs)
        all_act = np.stack(self.buffer.actions)
        all_lp = np.stack(self.buffer.log_probs)
        T, N = all_obs.shape[:2]
        obs_f = torch.FloatTensor(all_obs.reshape(T * N, -1)).to(self.device)
        act_f = torch.FloatTensor(all_act.reshape(T * N, -1)).to(self.device)
        old_lp = torch.FloatTensor(all_lp.reshape(T * N)).to(self.device)
        adv_f = torch.FloatTensor(advantages.reshape(T * N)).to(self.device)
        ret_f = torch.FloatTensor(returns.reshape(T * N)).to(self.device)
        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)

        pl, vl = [], []
        for _ in range(self.n_epochs):
            idx = np.random.permutation(T * N)
            for s in range(0, T * N, self.batch_size):
                b = idx[s:s + self.batch_size]
                dist = self.actor.get_dist(obs_f[b])
                nlp = dist.log_prob(act_f[b]).sum(-1)
                ent = dist.entropy().sum(-1).mean()
                ratio = (nlp - old_lp[b]).exp()
                s1 = ratio * adv_f[b]
                s2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_f[b]
                p_loss = -torch.min(s1, s2).mean() - self.entropy_coeff * ent
                v_loss = F.mse_loss(self.critic(obs_f[b]), ret_f[b])
                self.actor_optim.zero_grad(); p_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optim.step()
                self.critic_optim.zero_grad(); v_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optim.step()
                pl.append(p_loss.item()); vl.append(v_loss.item())
        self.buffer.clear()
        return {"policy_loss": float(np.mean(pl)), "value_loss": float(np.mean(vl))}

    def train_episode(self, env, protocol, n_windows: int = 10,
                      rng=None) -> Dict:
        if rng is None:
            rng = np.random.default_rng()
        cfg = self.cfg
        obs, info = env.reset()
        nodes = env.nodes
        n = len(nodes)
        ep_r, ep_f1, ep_e = [], [], []

        for w in range(n_windows):
            env.recompute_ground_truth()
            values = self.get_values(obs)
            actions, log_probs = self.select_actions(obs)
            window_result = protocol.run_window(nodes, cfg, rng,
                                                [actions] * cfg.N_slot)
            env.set_discovered_topology(window_result["disc_adj"])
            gt = env.get_ground_truth_topology()
            f1, *_ = protocol.compute_f1(gt, n)
            node_type_by_id = {nd.node_id: nd.node_type for nd in nodes}
            energies = np.array([
                protocol.compute_energy(i, cfg, node_type_by_id.get(i, "ship"))
                for i in range(n)
            ], dtype=np.float32)
            collisions = np.array([protocol.states[i].collisions
                                   if i in protocol.states else 0
                                   for i in range(n)], dtype=np.float32)
            rewards = self.compute_rewards(f1, energies, collisions)
            done = (w == n_windows - 1)
            self.buffer.store(obs, actions, log_probs, rewards, done, values)
            ep_r.append(float(rewards.mean())); ep_f1.append(f1)
            ep_e.append(float(energies.mean()))
            obs, _, term, trunc, info = env.step(actions)
            if term or trunc:
                obs, info = env.reset()

        update_info = self.update()
        return {"mean_reward": np.mean(ep_r), "mean_f1": np.mean(ep_f1),
                "mean_energy": np.mean(ep_e), **update_info}
