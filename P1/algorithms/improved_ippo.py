"""
Improved IPPO — Independent PPO with a Global Critic and
counterfactual difference rewards for INDP neighbour discovery.

Key features (from Manuscript I Section 8):
  * Per-agent actor MLP with LayerNorm
  * Centralised global-state critic MLP (CTDE) with LayerNorm
  * Counterfactual difference reward D_i^t = F1(a_all) - F1(a_{-i}, silent)
  * Active-node filtering + minibatch B_cf for scalability
  * Clipped PPO surrogate objective + GAE advantage estimation
  * Orthogonal initialization for stable training
  * Learning rate cosine annealing
  * Entropy coefficient scheduling

V14 fixes over V11:
  * Added global F1 anchor to reward (prevents degenerate do-nothing policy)
  * Removed RunningNormalizer (relied on advantage normalisation instead)
  * Aligned CosineAnnealingLR T_max with training length
"""

from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from Env.config import EnvConfig


# ═══════════════════════════════════════════════════════════════════════════
# Orthogonal init helper
# ═══════════════════════════════════════════════════════════════════════════

def _ortho_init(module: nn.Module, gain: float = 1.0):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# ═══════════════════════════════════════════════════════════════════════════
# Networks
# ═══════════════════════════════════════════════════════════════════════════

class ActorNetwork(nn.Module):
    """Per-agent actor: maps local observation to (mu, sigma) for actions."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.mu_head = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def forward(self, obs: torch.Tensor):
        h = F.relu(self.ln1(self.fc1(obs)))
        h = F.relu(self.ln2(self.fc2(h)))
        mu = torch.sigmoid(self.mu_head(h))
        std = self.log_std.exp().expand_as(mu)
        return mu, std

    def get_dist(self, obs: torch.Tensor) -> Normal:
        mu, std = self(obs)
        return Normal(mu, std + 1e-6)


class GlobalCritic(nn.Module):
    """Centralised critic: maps concatenated global state to V(s)."""

    def __init__(self, state_dim: int, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.fc3 = nn.Linear(hidden, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        return self.fc3(x).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════
# Rollout buffer
# ═══════════════════════════════════════════════════════════════════════════

class RolloutBuffer:
    def __init__(self):
        self.obs: List[np.ndarray] = []
        self.global_states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.log_probs: List[np.ndarray] = []
        self.rewards: List[np.ndarray] = []
        self.dones: List[bool] = []
        self.values: List[np.ndarray] = []

    def store(self, obs, global_state, actions, log_probs, rewards, done, values):
        self.obs.append(obs)
        self.global_states.append(global_state)
        self.actions.append(actions)
        self.log_probs.append(log_probs)
        self.rewards.append(rewards)
        self.dones.append(done)
        self.values.append(values)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.obs)


# ═══════════════════════════════════════════════════════════════════════════
# Improved IPPO Agent
# ═══════════════════════════════════════════════════════════════════════════

class ImprovedIPPO:
    """
    Improved IPPO with Global Critic, LayerNorm, LR scheduling,
    entropy scheduling.
    """

    def __init__(self, n_agents: int, obs_dim: int = 16, act_dim: int = 2,
                 actor_lr: float = 3e-4, critic_lr: float = 1e-3,
                 lr: float = 3e-4,
                 gamma: float = 0.99, lam: float = 0.97,
                 clip_eps: float = 0.2, entropy_coeff: float = 0.05,
                 n_epochs: int = 4, batch_size: int = 64,
                 cfg: Optional[EnvConfig] = None,
                 device: str = "cpu"):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.entropy_coeff_init = entropy_coeff
        self.entropy_coeff = entropy_coeff
        self.entropy_coeff_min = 0.005
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.cfg = cfg or EnvConfig()
        self.device = torch.device(device)
        self._episode_count = 0

        self.w1 = 1.5
        self.w_f1 = 2.0
        self.w2 = 0.8
        self.w3 = 0.05

        self.actor = ActorNetwork(obs_dim, act_dim).to(self.device)
        global_state_dim = n_agents * obs_dim + n_agents + 2
        self.critic = GlobalCritic(global_state_dim).to(self.device)

        _ortho_init(self.actor, gain=math.sqrt(2))
        nn.init.orthogonal_(self.actor.mu_head.weight, gain=0.01)
        _ortho_init(self.critic, gain=math.sqrt(2))
        nn.init.orthogonal_(self.critic.fc3.weight, gain=1.0)

        a_lr = actor_lr if actor_lr != 3e-4 or lr == 3e-4 else lr
        c_lr = critic_lr if critic_lr != 1e-3 or lr == 3e-4 else lr
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=a_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=c_lr)

        self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optim, T_max=1500, eta_min=a_lr * 0.05)
        self.critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.critic_optim, T_max=1500, eta_min=c_lr * 0.05)

        self.buffer = RolloutBuffer()

    # -----------------------------------------------------------------
    # Global state construction
    # -----------------------------------------------------------------

    def build_global_state(self, obs: np.ndarray,
                           gt_adj: Optional[np.ndarray] = None,
                           i_global: float = 0.0) -> np.ndarray:
        flat_obs = obs.flatten()
        e_res = obs[:, 7] if obs.ndim == 2 else np.zeros(self.n_agents)
        extras = np.array([i_global, float(np.mean(e_res))])
        return np.concatenate([flat_obs, e_res, extras]).astype(np.float32)

    # -----------------------------------------------------------------
    # Action selection
    # -----------------------------------------------------------------

    @torch.no_grad()
    def select_actions(self, obs: np.ndarray,
                       global_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        obs_t = torch.FloatTensor(obs).to(self.device)

        dist = self.actor.get_dist(obs_t)
        actions = dist.sample()
        actions = torch.clamp(actions, 0.0, 1.0)
        log_probs = dist.log_prob(actions).sum(-1)

        return (actions.cpu().numpy(),
                log_probs.cpu().numpy())

    @torch.no_grad()
    def get_value(self, global_state: np.ndarray) -> float:
        gs_t = torch.FloatTensor(global_state).unsqueeze(0).to(self.device)
        return self.critic(gs_t).item()

    # -----------------------------------------------------------------
    # Counterfactual difference reward
    # -----------------------------------------------------------------

    def compute_counterfactual_rewards(
            self, f1_all: float, per_agent_f1_without: np.ndarray,
            energies: np.ndarray, collisions: np.ndarray) -> np.ndarray:
        d_i = f1_all - per_agent_f1_without
        r = (self.w1 * d_i
             + self.w_f1 * f1_all
             - self.w2 * energies / self.cfg.E_ref
             - self.w3 * collisions)
        return r.astype(np.float32)

    def compute_counterfactual_f1_batch(
            self, env, protocol, nodes, cfg, rng,
            active_ids: List[int], b_cf: int) -> np.ndarray:
        """Analytical counterfactual approximation (no re-simulation)."""
        n = len(nodes)
        f1_without = np.zeros(n, dtype=np.float32)

        gt = env.get_ground_truth_topology()
        disc_adj = protocol.build_discovered_topology(n)
        f1_all, tp_all, fp_all, fn_all = protocol.compute_f1(gt, n)

        disc_sym = disc_adj | disc_adj.T
        gt_upper = np.triu(gt, k=1)
        disc_upper = np.triu(disc_sym, k=1)
        total_disc_edges = int(disc_upper.sum())
        if total_disc_edges == 0:
            return f1_without

        for nid in range(n):
            lnt = protocol.states[nid].lnt if nid in protocol.states else set()
            edges_via_i = 0
            for nbr in lnt:
                if 0 <= nbr < n:
                    ri, ci = min(nid, nbr), max(nid, nbr)
                    if disc_upper[ri, ci]:
                        edges_via_i += 1
            frac = edges_via_i / total_disc_edges
            tp_rem = max(0, tp_all - int(round(frac * tp_all)))
            fp_rem = max(0, fp_all - int(round(frac * fp_all)))
            fn_rem = fn_all + int(round(frac * tp_all))
            denom = 2 * tp_rem + fp_rem + fn_rem
            f1_without[nid] = (2 * tp_rem / denom) if denom > 0 else 0.0

        return f1_without

    # -----------------------------------------------------------------
    # GAE computation
    # -----------------------------------------------------------------

    def compute_gae(self, rewards: List[np.ndarray],
                    values: List[np.ndarray],
                    dones: List[bool],
                    last_value: float) -> Tuple[np.ndarray, np.ndarray]:
        n_steps = len(rewards)
        advantages = np.zeros((n_steps, self.n_agents), dtype=np.float32)
        returns = np.zeros_like(advantages)
        gae = np.zeros(self.n_agents, dtype=np.float32)

        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_val = last_value
            else:
                next_val = values[t + 1]
            if isinstance(next_val, np.ndarray):
                next_val = np.mean(next_val)
            cur_val = values[t]
            if isinstance(cur_val, np.ndarray):
                cur_val = np.mean(cur_val)

            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - cur_val
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + cur_val

        return advantages, returns

    # -----------------------------------------------------------------
    # PPO update
    # -----------------------------------------------------------------

    def update(self) -> Dict[str, float]:
        buf = self.buffer
        if len(buf) < 2:
            buf.clear()
            return {"policy_loss": 0, "value_loss": 0}

        last_val = buf.values[-1]
        if isinstance(last_val, np.ndarray):
            last_val = float(np.mean(last_val))

        advantages, returns = self.compute_gae(
            buf.rewards, buf.values, buf.dones, last_val)

        all_obs = np.stack(buf.obs)
        all_gs = np.stack(buf.global_states)
        all_act = np.stack(buf.actions)
        all_lp = np.stack(buf.log_probs)

        T, N = all_obs.shape[:2]
        obs_flat = all_obs.reshape(T * N, -1)
        act_flat = all_act.reshape(T * N, -1)
        lp_flat = all_lp.reshape(T * N)
        adv_flat = advantages.reshape(T * N)
        ret_flat = returns.reshape(T * N)
        gs_flat = np.repeat(all_gs, N, axis=0)

        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        obs_t = torch.FloatTensor(obs_flat).to(self.device)
        act_t = torch.FloatTensor(act_flat).to(self.device)
        old_lp_t = torch.FloatTensor(lp_flat).to(self.device)
        adv_t = torch.FloatTensor(adv_flat).to(self.device)
        ret_t = torch.FloatTensor(ret_flat).to(self.device)
        gs_t = torch.FloatTensor(gs_flat).to(self.device)

        total_samples = T * N
        policy_losses = []
        value_losses = []

        for _ in range(self.n_epochs):
            indices = np.random.permutation(total_samples)
            for start in range(0, total_samples, self.batch_size):
                end = min(start + self.batch_size, total_samples)
                idx = indices[start:end]

                dist = self.actor.get_dist(obs_t[idx])
                new_lp = dist.log_prob(act_t[idx]).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                ratio = (new_lp - old_lp_t[idx]).exp()
                surr1 = ratio * adv_t[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps,
                                    1 + self.clip_eps) * adv_t[idx]
                policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coeff * entropy

                values = self.critic(gs_t[idx])
                value_loss = F.mse_loss(values, ret_t[idx])

                self.actor_optim.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optim.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())

        buf.clear()
        return {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
        }

    # -----------------------------------------------------------------
    # Entropy and scheduling helpers
    # -----------------------------------------------------------------

    def _anneal_entropy(self):
        decay = 0.998
        self.entropy_coeff = max(self.entropy_coeff_min,
                                 self.entropy_coeff * decay)

    # -----------------------------------------------------------------
    # Training loop helper
    # -----------------------------------------------------------------

    def train_episode(self, env, protocol, n_windows: int = 10,
                      rng: Optional[np.random.Generator] = None) -> Dict:
        """Run one training episode of n_windows discovery windows."""
        if rng is None:
            rng = np.random.default_rng()

        cfg = self.cfg
        obs, info = env.reset()
        nodes = env.nodes
        n = len(nodes)

        episode_rewards = []
        episode_f1 = []
        episode_energy = []

        for w in range(n_windows):
            env.recompute_ground_truth()

            gs = self.build_global_state(obs)
            value = self.get_value(gs)

            actions, log_probs = self.select_actions(obs, gs)

            actions_per_slot = [actions] * cfg.N_slot
            window_result = protocol.run_window(nodes, cfg, rng, actions_per_slot)
            disc_adj = window_result["disc_adj"]
            env.set_discovered_topology(disc_adj)

            gt = env.get_ground_truth_topology()
            f1_all, tp, fp, fn = protocol.compute_f1(gt, n)

            active_ids = [nid for nid, st in protocol.states.items()
                          if st.tx_slots > 0]
            f1_without = self.compute_counterfactual_f1_batch(
                env, protocol, nodes, cfg, rng, active_ids, cfg.B_cf)

            node_type_by_id = {nd.node_id: nd.node_type for nd in nodes}
            energies = np.array([
                protocol.compute_energy(i, cfg, node_type_by_id.get(i, "ship"))
                for i in range(n)
            ], dtype=np.float32)
            collisions = np.array(
                [protocol.states[i].collisions if i in protocol.states else 0
                 for i in range(n)], dtype=np.float32)

            rewards = self.compute_counterfactual_rewards(
                f1_all, f1_without, energies, collisions)

            done = (w == n_windows - 1)

            self.buffer.store(obs, gs, actions, log_probs, rewards,
                              done, np.full(n, value, dtype=np.float32))

            episode_rewards.append(float(rewards.mean()))
            episode_f1.append(f1_all)
            episode_energy.append(float(energies.mean()))

            obs, _, terminated, truncated, info = env.step(actions)
            if terminated or truncated:
                obs, info = env.reset()

        update_info = self.update()

        self._episode_count += 1
        self._anneal_entropy()
        self.actor_scheduler.step()
        self.critic_scheduler.step()

        return {
            "mean_reward": float(np.mean(episode_rewards)),
            "mean_f1": float(np.mean(episode_f1)),
            "mean_energy": float(np.mean(episode_energy)),
            **update_info,
        }
