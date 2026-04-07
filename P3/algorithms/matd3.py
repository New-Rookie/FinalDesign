"""
Standard MATD3 baseline — Multi-Agent TD3 without temporal encoder
and without hierarchical replay.

Uses a simple MLP encoder on single-step observations and a uniform
replay buffer.  Serves as the ablation baseline for Improved MATD3.
"""

from __future__ import annotations

import copy
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Env.config import EnvConfig
from Env.core_env import MarineIoTEnv

from P3.resource_mgmt.task_offloader import (
    QueueState, simulate_offloading, select_source_buoys,
    find_local_candidates, find_edge_candidates, find_best_satellite,
)
from P3.resource_mgmt.metrics import aggregate_results, compute_reward

OBS_DIM = 18
ACT_DIM = 6


class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.ReLU(),
            nn.Linear(out_dim, out_dim), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(obs))


class TwinCritic(nn.Module):
    def __init__(self, state_dim: int, total_act_dim: int, hidden: int = 256):
        super().__init__()
        inp = state_dim + total_act_dim
        self.q1 = nn.Sequential(
            nn.Linear(inp, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(inp, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state, actions):
        sa = torch.cat([state, actions], dim=-1)
        return self.q1(sa).squeeze(-1), self.q2(sa).squeeze(-1)


class UniformReplay:
    def __init__(self, capacity: int = 50000):
        self.buf: deque = deque(maxlen=capacity)

    def push(self, transition: Tuple):
        self.buf.append(transition)

    def sample(self, batch_size: int):
        if len(self.buf) < batch_size:
            return None
        idxs = np.random.randint(0, len(self.buf), size=batch_size)
        items = [self.buf[i] for i in idxs]
        return list(zip(*items))

    def __len__(self):
        return len(self.buf)


class MATD3:
    """Standard MATD3 baseline for resource management."""

    def __init__(
        self, n_agents: int, cfg: EnvConfig,
        actor_lr: float = 3e-4, critic_lr: float = 1e-3, lr: float = 3e-4,
        gamma: float = 0.99, tau: float = 0.005,
        policy_delay: int = 2, noise_std: float = 0.1,
        noise_clip: float = 0.3, batch_size: int = 256,
        device: str = "cpu",
    ):
        self.n_agents = n_agents
        self.cfg = cfg
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.noise_std = noise_std
        self.noise_clip = noise_clip
        self.batch_size = batch_size
        self.device = torch.device(device)
        self._update_count = 0

        enc_dim = 64
        self.enc_dim = enc_dim
        self.encoder = MLPEncoder(OBS_DIM, enc_dim).to(self.device)
        self.actor = Actor(enc_dim, ACT_DIM).to(self.device)

        global_state_dim = n_agents * enc_dim
        total_act_dim = n_agents * ACT_DIM
        self.critic = TwinCritic(global_state_dim, total_act_dim).to(self.device)

        self.encoder_target = copy.deepcopy(self.encoder)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        a_lr = actor_lr if actor_lr != 3e-4 or lr == 3e-4 else lr
        c_lr = critic_lr if critic_lr != 1e-3 or lr == 3e-4 else lr
        self.actor_optim = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.actor.parameters()), lr=a_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=c_lr)

        self.replay = UniformReplay(capacity=50000)

    def build_observations(
        self, env: MarineIoTEnv, source_ids: List[int],
        local_map: Dict[int, int], edge_map: Dict[int, int],
    ) -> Dict[int, np.ndarray]:
        cfg = self.cfg
        ceilings = cfg.scaled_ceilings()
        obs_dict: Dict[int, np.ndarray] = {}
        for bid in source_ids:
            o = np.zeros(OBS_DIM, dtype=np.float32)
            buoy = env.nodes[bid]
            o[0:3] = buoy.position / 100_000.0
            o[3:6] = buoy.velocity / 100.0
            o[6] = buoy.energy_residual / 100.0
            lid = local_map.get(bid, -1)
            if lid >= 0:
                lp = env.link_phy.get((bid, lid))
                o[7] = min(lp.snr, 1000.0) / 1000.0 if lp else 0.0
                o[8] = min(lp.distance, cfg.R_local_buoy) / cfg.R_local_buoy if lp else 1.0
                lt = env.nodes[lid].node_type
                cl = ceilings.get(lt, ceilings["ship"])
                o[9] = cl["B_max"] / 50e6
                o[10] = cl["F_max"] / 32e9
                o[11] = cl["S_max"] / 50e9
            eid = edge_map.get(bid, -1)
            if eid >= 0:
                et = env.nodes[eid].node_type
                ce = ceilings.get(et, ceilings["ship"])
                o[12] = ce["F_max"] / 32e9
            o[13] = cfg.M_b / max(cfg.M_tot, 1e-12)

            sat_id = find_best_satellite(env, lid, cfg) if lid >= 0 else None
            if sat_id is not None:
                sat_lp = env.link_phy.get((lid, sat_id))
                o[14] = min(sat_lp.snr, 1000.0) / 1000.0 if sat_lp else 0.0
            else:
                o[14] = 0.0

            o[15] = 1.0 if lid >= 0 else 0.0
            o[16] = float(env._step_count) / max(env.max_steps, 1)
            obs_dict[bid] = o
        return obs_dict

    def build_assignments(self, env, source_ids):
        cfg = self.cfg
        local_map, edge_map = {}, {}
        for bid in source_ids:
            cands = find_local_candidates(env, bid, cfg)
            if cands:
                local_map[bid] = cands[0][0]
                e_cands = find_edge_candidates(env, cands[0][0], cfg)
                edge_map[bid] = e_cands[0][0] if e_cands else -1
            else:
                local_map[bid] = -1
                edge_map[bid] = -1
        return local_map, edge_map

    @torch.no_grad()
    def select_actions(self, env, source_ids, local_map, edge_map, explore=True):
        obs_dict = self.build_observations(env, source_ids, local_map, edge_map)
        if not source_ids:
            return {}, np.zeros((0, self.enc_dim), dtype=np.float32)
        obs_batch = np.stack([obs_dict[bid] for bid in source_ids])
        obs_t = torch.FloatTensor(obs_batch).to(self.device)
        enc_batch = self.encoder(obs_t)
        act_batch = self.actor(enc_batch).cpu().numpy()
        enc_np = enc_batch.cpu().numpy()

        actions_out = {}
        for i, bid in enumerate(source_ids):
            a_t = act_batch[i]
            if explore:
                noise = np.random.randn(ACT_DIM).astype(np.float32) * self.noise_std
                a_t = np.clip(a_t + noise, 0.0, 1.0)
            actions_out[bid] = {
                "local_id": local_map.get(bid, -1),
                "edge_id": edge_map.get(bid, -1),
                "alpha_off": float(a_t[0]),
                "bw_frac": float(a_t[1]),
                "f_frac": float(a_t[2]),
                "omega_comm": float(a_t[3]),
                "omega_comp": float(a_t[4]),
                "omega_rem": float(a_t[5]),
            }
        return actions_out, enc_np

    def _pack_encoded_state(self, encoded: np.ndarray) -> np.ndarray:
        packed = np.zeros((self.n_agents, self.enc_dim), dtype=np.float32)
        if encoded.size > 0:
            k = min(self.n_agents, encoded.shape[0])
            packed[:k] = encoded[:k]
        return packed.reshape(-1)

    def _pack_actions(self, actions: Dict[int, Dict[str, float]], source_ids: List[int]) -> np.ndarray:
        packed = np.zeros((self.n_agents, ACT_DIM), dtype=np.float32)
        k = min(self.n_agents, len(source_ids))
        for i, bid in enumerate(source_ids[:k]):
            a = actions.get(bid, {})
            packed[i, 0] = float(a.get("alpha_off", 0.0))
            packed[i, 1] = float(a.get("bw_frac", 0.0))
            packed[i, 2] = float(a.get("f_frac", 0.0))
            packed[i, 3] = float(a.get("omega_comm", 0.5))
            packed[i, 4] = float(a.get("omega_comp", 0.3))
            packed[i, 5] = float(a.get("omega_rem", 0.2))
        return packed.reshape(-1)

    @torch.no_grad()
    def _encode_obs_dict(self, source_ids: List[int], obs_dict: Dict[int, np.ndarray]) -> np.ndarray:
        if not source_ids:
            return np.zeros((0, self.enc_dim), dtype=np.float32)
        obs_batch = np.stack([obs_dict[bid] for bid in source_ids])
        obs_t = torch.FloatTensor(obs_batch).to(self.device)
        return self.encoder(obs_t).cpu().numpy()

    def train_episode(self, env, n_windows=5, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        cfg = self.cfg
        obs, _ = env.reset()
        source_ids = select_source_buoys(env.nodes, cfg.N_src, rng, cfg.source_activation_ratio)
        queue = QueueState()
        ep_rewards, ep_T, ep_E, ep_G = [], [], [], []

        for w in range(n_windows):
            env.recompute_ground_truth()
            queue.reset()
            local_map, edge_map = self.build_assignments(env, source_ids)
            obs_pre = self.build_observations(env, source_ids, local_map, edge_map)
            actions, enc_pre = self.select_actions(env, source_ids, local_map, edge_map, explore=True)
            results = simulate_offloading(env, cfg, source_ids, actions, queue)
            metrics = aggregate_results(results, cfg.Gamma_max)
            reward = compute_reward(metrics, cfg.T_max, cfg.E_max, cfg.Gamma_max)
            ep_rewards.append(reward)
            ep_T.append(metrics["mean_T_total"])
            ep_E.append(metrics["mean_E_total"])
            ep_G.append(metrics["mean_Gamma"])

            actions_env = np.ones((len(env.nodes), 2), dtype=np.float32)
            env.step(actions_env)
            local_map_new, edge_map_new = self.build_assignments(env, source_ids)
            obs_post = self.build_observations(env, source_ids, local_map_new, edge_map_new)
            enc_post = self._encode_obs_dict(source_ids, obs_post)

            global_s = self._pack_encoded_state(enc_pre)
            global_s_next = self._pack_encoded_state(enc_post)
            flat_a = self._pack_actions(actions, source_ids)
            done = 0.0
            self.replay.push((global_s, flat_a, reward, global_s_next, done))
            if len(self.replay) >= self.batch_size:
                self._update()

        return {
            "mean_reward": float(np.mean(ep_rewards)),
            "mean_T_total": float(np.mean(ep_T)),
            "mean_E_total": float(np.mean(ep_E)),
            "mean_Gamma": float(np.mean(ep_G)),
            "policy_loss": getattr(self, "_last_p_loss", 0.0),
            "value_loss": getattr(self, "_last_v_loss", 0.0),
        }

    def eval_window(self, env, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        cfg = self.cfg
        source_ids = select_source_buoys(env.nodes, cfg.N_src, rng, cfg.source_activation_ratio)
        env.recompute_ground_truth()
        local_map, edge_map = self.build_assignments(env, source_ids)
        actions, _ = self.select_actions(env, source_ids, local_map, edge_map, explore=False)
        queue = QueueState()
        results = simulate_offloading(env, cfg, source_ids, actions, queue)
        metrics = aggregate_results(results, cfg.Gamma_max)
        actions_env = np.ones((len(env.nodes), 2), dtype=np.float32)
        env.step(actions_env)
        return metrics

    def _update(self):
        batch = self.replay.sample(self.batch_size)
        if batch is None:
            return
        states, actions, rewards, next_states, dones = batch
        s = torch.FloatTensor(np.stack(states)).to(self.device)
        a = torch.FloatTensor(np.stack(actions)).to(self.device)
        r = torch.FloatTensor(np.array(rewards, dtype=np.float32)).to(self.device)
        s2 = torch.FloatTensor(np.stack(next_states)).to(self.device)
        d = torch.FloatTensor(np.array(dones, dtype=np.float32)).to(self.device)

        with torch.no_grad():
            noise = (torch.randn_like(a) * self.noise_std).clamp(
                -self.noise_clip, self.noise_clip)
            s2_agents = s2.view(-1, self.n_agents, self.enc_dim)
            a2_agents = self.actor_target(s2_agents.reshape(-1, self.enc_dim)).view(-1, self.n_agents, ACT_DIM)
            a2 = (a2_agents.reshape(-1, self.n_agents * ACT_DIM) + noise).clamp(0.0, 1.0)
            q1t, q2t = self.critic_target(s2, a2)
            q_target = r + self.gamma * (1 - d) * torch.min(q1t, q2t)

        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optim.step()
        self._last_v_loss = critic_loss.item()

        self._update_count += 1
        if self._update_count % self.policy_delay == 0:
            s_agents = s.view(-1, self.n_agents, self.enc_dim)
            aa = self.actor(s_agents.reshape(-1, self.enc_dim)).view(-1, self.n_agents * ACT_DIM)
            q1v, _ = self.critic(s, aa)
            actor_loss = -q1v.mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.actor.parameters()), 1.0)
            self.actor_optim.step()
            self._last_p_loss = actor_loss.item()
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.encoder, self.encoder_target)
            self._soft_update(self.critic, self.critic_target)

    def _soft_update(self, src, tgt):
        for sp, tp in zip(src.parameters(), tgt.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)
