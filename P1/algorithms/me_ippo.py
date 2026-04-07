"""
Memory-Enhanced IPPO (ME-IPPO) — proposed algorithm for INDP neighbour
discovery energy optimisation.

Extends Improved IPPO with a lightweight historical encounter memory
module that guides exploration toward high-yield directions, reducing
blind scanning and directly lowering energy consumption.

Key additions over Improved IPPO:
  * DirectionMemory: K-sector memory of successful discovery directions
  * Augmented observation: actor receives memory confidence vectors
  * Separate actor / critic learning rates
  * Rebalanced reward weights for energy-aware optimisation

V14 fixes over V11:
  * Removed external _apply_mem_bias (caused log-prob mismatch in PPO)
  * Added global F1 anchor to reward (prevents degenerate do-nothing policy)
  * Removed RunningNormalizer (relied on advantage normalisation instead)
  * Partial memory persistence across episodes (0.5 decay instead of reset)
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


K_SECTORS = 8
DELTA_PLUS = 0.15
DELTA_MINUS = 0.03


def _ortho_init(module: nn.Module, gain: float = 1.0):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# ═══════════════════════════════════════════════════════════════════════════
# Direction Memory
# ═══════════════════════════════════════════════════════════════════════════

class DirectionMemory:
    """Per-node K-sector memory tracking successful discovery directions."""

    def __init__(self, n_agents: int, k: int = K_SECTORS):
        self.n_agents = n_agents
        self.k = k
        self.memory = np.zeros((n_agents, k), dtype=np.float32)

    def reset(self):
        self.memory[:] = 0.0

    def update(self, agent_id: int, sector: int, success: bool):
        if success:
            self.memory[agent_id, sector] = min(
                1.0, self.memory[agent_id, sector] + DELTA_PLUS)
        else:
            self.memory[agent_id, sector] = max(
                0.0, self.memory[agent_id, sector] - DELTA_MINUS)

    def update_from_detections(self, detections: Dict[int, List[int]],
                               nodes, n_agents: int):
        """Update memory based on discovery slot results."""
        pos = {nd.node_id: nd.position for nd in nodes}
        detected_set = set()
        for rx_id, tx_ids in detections.items():
            for tx_id in tx_ids:
                detected_set.add((rx_id, tx_id))

        for i in range(n_agents):
            if i not in pos:
                continue
            p_i = pos[i]
            for j in range(n_agents):
                if i == j or j not in pos:
                    continue
                dp = pos[j] - p_i
                angle = math.atan2(dp[1], dp[0])
                if angle < 0:
                    angle += 2 * math.pi
                sector = int(angle / (2 * math.pi) * self.k) % self.k
                success = (i, j) in detected_set or (j, i) in detected_set
                self.update(i, sector, success)

    def get_vectors(self) -> np.ndarray:
        return self.memory.copy()


# ═══════════════════════════════════════════════════════════════════════════
# Networks
# ═══════════════════════════════════════════════════════════════════════════

class ActorNetwork(nn.Module):
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
# Memory-Enhanced IPPO Agent
# ═══════════════════════════════════════════════════════════════════════════

class MemoryEnhancedIPPO:
    """
    ME-IPPO: Improved IPPO + directional encounter memory.

    The memory module appends a K-dimensional confidence vector to each
    agent's observation.  The actor network learns to use memory information
    through the augmented observation, without external action bias.
    """

    def __init__(self, n_agents: int, obs_dim: int = 16, act_dim: int = 2,
                 actor_lr: float = 3e-4, critic_lr: float = 1e-3,
                 gamma: float = 0.99, lam: float = 0.97,
                 clip_eps: float = 0.2, entropy_coeff: float = 0.05,
                 n_epochs: int = 4, batch_size: int = 64,
                 cfg: Optional[EnvConfig] = None,
                 device: str = "cpu"):
        self.n_agents = n_agents
        self.base_obs_dim = obs_dim
        self.obs_dim = obs_dim + K_SECTORS
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
        self.w_ea = 0.2
        self.w3 = 0.05

        self.dir_memory = DirectionMemory(n_agents, K_SECTORS)

        self.actor = ActorNetwork(self.obs_dim, act_dim).to(self.device)
        global_state_dim = n_agents * self.obs_dim + n_agents + 2
        self.critic = GlobalCritic(global_state_dim).to(self.device)

        _ortho_init(self.actor, gain=math.sqrt(2))
        nn.init.orthogonal_(self.actor.mu_head.weight, gain=0.01)
        _ortho_init(self.critic, gain=math.sqrt(2))
        nn.init.orthogonal_(self.critic.fc3.weight, gain=1.0)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optim, T_max=1500, eta_min=actor_lr * 0.05)
        self.critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.critic_optim, T_max=1500, eta_min=critic_lr * 0.05)

        self.buffer = RolloutBuffer()

    def _augment_obs(self, obs: np.ndarray) -> np.ndarray:
        mem_vecs = self.dir_memory.get_vectors()
        n = obs.shape[0]
        if mem_vecs.shape[0] < n:
            pad = np.zeros((n - mem_vecs.shape[0], K_SECTORS), dtype=np.float32)
            mem_vecs = np.concatenate([mem_vecs, pad], axis=0)
        return np.concatenate([obs, mem_vecs[:n]], axis=1).astype(np.float32)

    def build_global_state(self, aug_obs: np.ndarray,
                           gt_adj: Optional[np.ndarray] = None,
                           i_global: float = 0.0) -> np.ndarray:
        flat_obs = aug_obs.flatten()
        e_res = aug_obs[:, 7] if aug_obs.ndim == 2 else np.zeros(self.n_agents)
        extras = np.array([i_global, float(np.mean(e_res))])
        return np.concatenate([flat_obs, e_res, extras]).astype(np.float32)

    @torch.no_grad()
    def select_actions(self, obs: np.ndarray,
                       global_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        obs_t = torch.FloatTensor(obs).to(self.device)
        dist = self.actor.get_dist(obs_t)
        actions = dist.sample()
        actions = torch.clamp(actions, 0.0, 1.0)
        log_probs = dist.log_prob(actions).sum(-1)
        return actions.cpu().numpy(), log_probs.cpu().numpy()

    @torch.no_grad()
    def get_value(self, global_state: np.ndarray) -> float:
        gs_t = torch.FloatTensor(global_state).unsqueeze(0).to(self.device)
        return self.critic(gs_t).item()

    def compute_counterfactual_rewards(
            self, f1_all: float, per_agent_f1_without: np.ndarray,
            energies: np.ndarray, mean_energy: float,
            collisions: np.ndarray) -> np.ndarray:
        d_i = f1_all - per_agent_f1_without
        energy_advantage = mean_energy - energies
        r = (self.w1 * d_i
             + self.w_f1 * f1_all
             - self.w2 * energies / self.cfg.E_ref
             + self.w_ea * energy_advantage / max(self.cfg.E_ref, 1e-6)
             - self.w3 * collisions)
        return r.astype(np.float32)

    def compute_counterfactual_f1_batch(
            self, env, protocol, nodes, cfg, rng,
            active_ids: List[int], b_cf: int) -> np.ndarray:
        n = len(nodes)
        f1_without = np.zeros(n, dtype=np.float32)
        gt = env.get_ground_truth_topology()
        disc_adj = protocol.build_discovered_topology(n)
        f1_all, tp_all, fp_all, fn_all = protocol.compute_f1(gt, n)
        disc_sym = disc_adj | disc_adj.T
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

    def compute_gae(self, rewards, values, dones, last_value):
        n_steps = len(rewards)
        advantages = np.zeros((n_steps, self.n_agents), dtype=np.float32)
        returns = np.zeros_like(advantages)
        gae = np.zeros(self.n_agents, dtype=np.float32)
        for t in reversed(range(n_steps)):
            next_val = last_value if t == n_steps - 1 else values[t + 1]
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
        policy_losses, value_losses = [], []
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

    def _anneal_entropy(self):
        self.entropy_coeff = max(self.entropy_coeff_min,
                                 self.entropy_coeff * 0.998)

    def train_episode(self, env, protocol, n_windows: int = 10,
                      rng: Optional[np.random.Generator] = None) -> Dict:
        if rng is None:
            rng = np.random.default_rng()
        cfg = self.cfg
        obs, info = env.reset()
        nodes = env.nodes
        n = len(nodes)

        if self.dir_memory.n_agents != n:
            self.dir_memory = DirectionMemory(n, K_SECTORS)
        else:
            self.dir_memory.memory *= 0.5

        episode_rewards, episode_f1, episode_energy = [], [], []

        for w in range(n_windows):
            env.recompute_ground_truth()
            aug_obs = self._augment_obs(obs)
            gs = self.build_global_state(aug_obs)
            value = self.get_value(gs)
            actions, log_probs = self.select_actions(aug_obs, gs)

            actions_per_slot = [actions] * cfg.N_slot
            window_result = protocol.run_window(nodes, cfg, rng, actions_per_slot)
            disc_adj = window_result["disc_adj"]
            env.set_discovered_topology(disc_adj)

            for slot_info in window_result.get("slot_infos", []):
                dets = slot_info.get("detections", {})
                self.dir_memory.update_from_detections(dets, nodes, n)

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
            mean_energy = float(energies.mean())
            collisions = np.array(
                [protocol.states[i].collisions if i in protocol.states else 0
                 for i in range(n)], dtype=np.float32)

            rewards = self.compute_counterfactual_rewards(
                f1_all, f1_without, energies, mean_energy, collisions)

            done = (w == n_windows - 1)
            self.buffer.store(aug_obs, gs, actions, log_probs, rewards,
                              done, np.full(n, value, dtype=np.float32))
            episode_rewards.append(float(rewards.mean()))
            episode_f1.append(f1_all)
            episode_energy.append(mean_energy)

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
