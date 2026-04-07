"""
MAPPO — Multi-Agent PPO baseline for link selection (no GCN).

Replaces the GCN encoder with a standard MLP that concatenates local
observation with mean-pooled neighbor features.  Same global critic
and PPO update as GMAPPO.  Isolates the contribution of graph structure.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from Env.config import EnvConfig
from Env.core_env import MarineIoTEnv
from Env.channel import link_class as get_link_class

from P2.link_quality.metrics import (
    compute_let, compute_p_surv, compute_s_ho,
    path_quality, path_stability, link_advantage, compute_lqi,
)
from P2.link_quality.path_manager import PathManager
from P2.link_quality.rf_estimator import LinkQualityEstimator
from P2.algorithms.gmappo import MAX_ACTIONS, NODE_FEAT_DIM, RunningNormalizer


class MLPEncoder(nn.Module):
    """Local obs + mean-pooled neighbor features -> agent embedding."""

    def __init__(self, obs_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )

    def forward(self, local_obs: torch.Tensor,
                nbr_mean: torch.Tensor) -> torch.Tensor:
        x = torch.cat([local_obs, nbr_mean], dim=-1)
        return self.net(x)


class MAPPOActor(nn.Module):
    def __init__(self, embed_dim: int, max_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, max_actions),
        )

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> Categorical:
        logits = self.net(h)
        logits = logits.masked_fill(~mask.bool(), -1e9)
        return Categorical(logits=logits)


class MAPPOCritic(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s).squeeze(-1)


class RolloutBuffer:
    def __init__(self):
        self.obs, self.global_states = [], []
        self.log_probs, self.rewards, self.dones, self.values = [], [], [], []

    def store(self, obs, gs, lp, reward, done, value):
        self.obs.append(obs)
        self.global_states.append(gs)
        self.log_probs.append(lp)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.obs)


class MAPPO:
    """Standard MAPPO — MLP-based, no graph convolution."""

    def __init__(self, n_agents: int, cfg: EnvConfig,
                 estimator: LinkQualityEstimator,
                 actor_lr: float = 3e-4, critic_lr: float = 1e-3,
                 lr: float = 3e-4, gamma: float = 0.99, lam: float = 0.95,
                 clip_eps: float = 0.2, entropy_coeff: float = 0.01,
                 n_epochs: int = 4, batch_size: int = 64,
                 device: str = "cpu"):
        self.n_agents = n_agents
        self.cfg = cfg
        self.estimator = estimator
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.entropy_coeff = entropy_coeff
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = torch.device(device)

        embed_dim = 64
        self.encoder = MLPEncoder(NODE_FEAT_DIM, embed_dim).to(self.device)
        self.actor = MAPPOActor(embed_dim, MAX_ACTIONS).to(self.device)
        self.embed_dim = embed_dim
        gs_dim = embed_dim + 4
        self.critic = MAPPOCritic(gs_dim).to(self.device)

        a_lr = actor_lr if actor_lr != 3e-4 or lr == 3e-4 else lr
        c_lr = critic_lr if critic_lr != 1e-3 or lr == 3e-4 else lr

        params = list(self.encoder.parameters()) + list(self.actor.parameters())
        self.actor_optim = torch.optim.Adam(params, lr=a_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=c_lr)

        self.buffer = RolloutBuffer()
        self.path_mgr = PathManager(cfg)
        self._prev_actions: Dict[int, int] = {}
        self._sinr_histories: Dict[Tuple[int, int], List[float]] = defaultdict(list)
        self._q_cache: Dict[Tuple[int, int], float] = {}
        self._comm_range_cache: Dict[Tuple[str, str], float] = {}
        self.reward_normalizer = RunningNormalizer(clip=5.0)

    def _build_node_features(self, env: MarineIoTEnv) -> np.ndarray:
        nodes = env.nodes
        n = len(nodes)
        feats = np.zeros((n, NODE_FEAT_DIM), dtype=np.float32)
        _type_enc = {"satellite": 0, "uav": 1, "ship": 2, "buoy": 3, "land": 4}
        positions = np.stack([nd.position for nd in nodes])
        velocities = np.stack([nd.velocity for nd in nodes])
        feats[:, 0] = [_type_enc.get(nd.node_type, -1) for nd in nodes]
        feats[:, 1:4] = positions / 100_000.0
        feats[:, 4:7] = velocities / 100.0
        feats[:, 7] = [nd.energy_residual / 100.0 for nd in nodes]
        feats[:, 8] = [nd.tx_power * 1e3 for nd in nodes]
        m = getattr(env, '_phy', None)
        if m is not None and m.SNR.shape[0] == n:
            snr_in = m.SNR.copy()
            np.fill_diagonal(snr_in, np.nan)
            feats[:, 9] = np.nanmean(snr_in, axis=0).astype(np.float32)
        return feats

    def _mean_pool_neighbors(self, feats: np.ndarray,
                             env: MarineIoTEnv) -> np.ndarray:
        n = len(env.nodes)
        pooled = np.zeros_like(feats)
        m = getattr(env, '_phy', None)
        if m is not None and m.SNR.shape[0] == n:
            gamma_lin = self.cfg.gamma_link_linear
            adj = (m.SNR >= gamma_lin).T.astype(np.float32)
            np.fill_diagonal(adj, 0.0)
            counts = adj.sum(axis=1, keepdims=True)
            counts = np.maximum(counts, 1.0)
            pooled = (adj @ feats) / counts
        return pooled

    @torch.inference_mode()
    def select_actions(self, env: MarineIoTEnv, source_ids: List[int],
                       candidate_map: Dict[int, List[int]]
                       ) -> Tuple[Dict[int, int], np.ndarray, np.ndarray]:
        feats = self._build_node_features(env)
        nbr_pool = self._mean_pool_neighbors(feats, env)
        feats_t = torch.FloatTensor(feats).to(self.device)
        nbr_t = torch.FloatTensor(nbr_pool).to(self.device)
        embeddings = self.encoder(feats_t, nbr_t)
        embed_np = embeddings.cpu().numpy()

        action_dict: Dict[int, int] = {}
        all_lp = np.zeros(len(source_ids), dtype=np.float32)

        for idx, bid in enumerate(source_ids):
            candidates = candidate_map.get(bid, [])
            mask = np.zeros(MAX_ACTIONS, dtype=np.float32)
            mask[0] = 1.0
            for ci in range(min(len(candidates), MAX_ACTIONS - 1)):
                mask[ci + 1] = 1.0

            h = embeddings[bid].unsqueeze(0)
            m_t = torch.FloatTensor(mask).unsqueeze(0).to(self.device)
            dist = self.actor(h, m_t)
            action = dist.sample()
            a_idx = action.item()

            if a_idx == 0:
                chosen = self._prev_actions.get(bid, candidates[0]
                                                 if candidates else bid)
            elif a_idx - 1 < len(candidates):
                chosen = candidates[a_idx - 1]
            else:
                chosen = self._prev_actions.get(bid, bid)

            action_dict[bid] = chosen
            all_lp[idx] = dist.log_prob(action).item()

        return action_dict, all_lp, embed_np

    def train_episode(self, env: MarineIoTEnv, n_windows: int = 10,
                      rng: Optional[np.random.Generator] = None) -> Dict:
        if rng is None:
            rng = np.random.default_rng()

        cfg = self.cfg
        obs, info = env.reset()
        nodes = env.nodes
        n = len(nodes)
        source_ids = PathManager.select_source_buoys(nodes, cfg.N_src, rng)
        n_src = len(source_ids)
        self._prev_actions.clear()
        self._sinr_histories.clear()

        ep_rewards, ep_la = [], []

        buf_feats = []
        buf_nbr_pool = []
        buf_gs = []
        buf_actions_idx = []
        buf_log_probs = []
        buf_agent_rewards = []
        buf_values = []
        buf_masks_list = []
        buf_source_bids = []
        buf_dones = []

        for w in range(n_windows):
            env.recompute_ground_truth()
            self._update_sinr_histories(env)
            self._q_cache = {}

            candidate_map = self._build_candidate_map(env, source_ids)

            feats = self._build_node_features(env)
            nbr_pool = self._mean_pool_neighbors(feats, env)
            feats_t = torch.FloatTensor(feats).to(self.device)
            nbr_t = torch.FloatTensor(nbr_pool).to(self.device)

            with torch.inference_mode():
                embeddings = self.encoder(feats_t, nbr_t)

            action_dict = {}
            window_lp = []
            window_actions_idx = []
            window_masks = []
            for idx, bid in enumerate(source_ids):
                candidates = candidate_map.get(bid, [])
                mask = np.zeros(MAX_ACTIONS, dtype=np.float32)
                mask[0] = 1.0
                for ci in range(min(len(candidates), MAX_ACTIONS - 1)):
                    mask[ci + 1] = 1.0
                h = embeddings[bid].unsqueeze(0)
                m_t = torch.FloatTensor(mask).unsqueeze(0).to(self.device)
                with torch.inference_mode():
                    dist = self.actor(h, m_t)
                    action = dist.sample()
                    lp = dist.log_prob(action)
                window_lp.append(lp.item())
                window_actions_idx.append(action.item())
                window_masks.append(mask)
                a_idx = action.item()
                if a_idx == 0:
                    chosen = self._prev_actions.get(bid,
                             candidates[0] if candidates else bid)
                elif a_idx - 1 < len(candidates):
                    chosen = candidates[a_idx - 1]
                else:
                    chosen = self._prev_actions.get(bid, bid)
                action_dict[bid] = chosen

            per_agent_r = np.zeros(n_src, dtype=np.float32)
            per_agent_la = np.zeros(n_src, dtype=np.float32)
            for idx, bid in enumerate(source_ids):
                hop_q, hop_s = self._evaluate_hops(env, bid,
                                                   action_dict.get(bid, bid))
                if hop_q:
                    la = link_advantage(path_quality(hop_q),
                                        path_stability(hop_s),
                                        cfg.w_Q, cfg.w_S)
                else:
                    la = 0.0
                per_agent_la[idx] = la
                did_switch = (self._prev_actions.get(bid) is not None
                              and action_dict.get(bid) != self._prev_actions.get(bid))
                no_path = len(candidate_map.get(bid, [])) == 0
                per_agent_r[idx] = la - cfg.eta_sw * float(did_switch) \
                                   - 0.1 * float(no_path)

            self._prev_actions.update(action_dict)

            mean_la = float(per_agent_la.mean())
            mean_reward = float(per_agent_r.mean())
            ep_rewards.append(mean_reward)
            ep_la.append(mean_la)

            for r_val in per_agent_r:
                self.reward_normalizer.update(float(r_val))
            norm_agent_r = np.array([
                self.reward_normalizer.normalize(float(r))
                for r in per_agent_r], dtype=np.float32)

            embed_np = embeddings.detach().cpu().numpy()
            pooled = embed_np.mean(axis=0)
            n_switch = self._count_switches(action_dict)
            gs = np.concatenate([pooled,
                                 [mean_la, float(n_switch),
                                  float(n), 0.0]]).astype(np.float32)
            value = self._get_value(gs)

            buf_feats.append(feats)
            buf_nbr_pool.append(nbr_pool)
            buf_gs.append(gs)
            buf_actions_idx.append(np.array(window_actions_idx, dtype=np.int64))
            buf_log_probs.append(np.array(window_lp, dtype=np.float32))
            buf_agent_rewards.append(norm_agent_r)
            buf_values.append(value)
            buf_masks_list.append(np.stack(window_masks))
            buf_source_bids.append(list(source_ids))
            buf_dones.append(w == n_windows - 1)

            actions_env = np.ones((n, 2), dtype=np.float32)
            obs, _, term, trunc, _ = env.step(actions_env)
            if term or trunc:
                obs, _ = env.reset()
                nodes = env.nodes

        p_loss_val, v_loss_val = self._ppo_clip_update(
            buf_feats, buf_nbr_pool, buf_gs, buf_actions_idx,
            buf_log_probs, buf_agent_rewards, buf_values, buf_masks_list,
            buf_source_bids, buf_dones)

        self.buffer.clear()
        return {"mean_reward": float(np.mean(ep_rewards)),
                "mean_LA": float(np.mean(ep_la)),
                "policy_loss": p_loss_val,
                "value_loss": v_loss_val}

    def _ppo_clip_update(self, buf_feats, buf_nbr_pool, buf_gs,
                         buf_actions_idx, buf_log_probs, buf_agent_rewards,
                         buf_values, buf_masks_list, buf_source_bids,
                         buf_dones) -> Tuple[float, float]:
        T = len(buf_agent_rewards)
        if T < 2:
            return 0.0, 0.0

        n_src = len(buf_source_bids[0]) if buf_source_bids else 1
        agent_rewards = np.stack(buf_agent_rewards)
        values = np.array(buf_values, dtype=np.float32)
        dones = np.array(buf_dones, dtype=np.float32)

        per_agent_adv = np.zeros((T, n_src), dtype=np.float32)
        for ai in range(n_src):
            gae = 0.0
            for t in reversed(range(T)):
                next_val = values[t + 1] if t + 1 < T else values[-1]
                delta = (agent_rewards[t, ai]
                         + self.gamma * next_val * (1 - dones[t])
                         - values[t])
                gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
                per_agent_adv[t, ai] = gae

        mean_rewards = agent_rewards.mean(axis=1)
        critic_adv = np.zeros(T, dtype=np.float32)
        gae_c = 0.0
        for t in reversed(range(T)):
            next_val = values[t + 1] if t + 1 < T else values[-1]
            delta = mean_rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae_c = delta + self.gamma * self.lam * (1 - dones[t]) * gae_c
            critic_adv[t] = gae_c
        critic_returns = critic_adv + values

        all_adv = per_agent_adv.reshape(-1)
        all_adv = (all_adv - all_adv.mean()) / (all_adv.std() + 1e-8)
        all_ret = np.repeat(critic_returns, n_src)

        all_old_lp = np.concatenate(buf_log_probs)
        all_actions = np.concatenate(buf_actions_idx)
        all_masks = np.concatenate(buf_masks_list)
        all_gs = np.repeat(np.stack(buf_gs), n_src, axis=0)

        old_lp_t = torch.FloatTensor(all_old_lp).to(self.device)
        actions_t = torch.LongTensor(all_actions).to(self.device)
        masks_t = torch.FloatTensor(all_masks).to(self.device)
        adv_t = torch.FloatTensor(all_adv).to(self.device)
        ret_t = torch.FloatTensor(all_ret).to(self.device)
        gs_t = torch.FloatTensor(all_gs).to(self.device)

        all_agent_bids = []
        all_timestep_idx = []
        for t in range(T):
            for bid in buf_source_bids[t]:
                all_agent_bids.append(bid)
                all_timestep_idx.append(t)

        feats_by_t = [torch.FloatTensor(buf_feats[t]).to(self.device) for t in range(T)]
        nbr_by_t = [torch.FloatTensor(buf_nbr_pool[t]).to(self.device) for t in range(T)]
        bids_arr = np.array(all_agent_bids, dtype=np.int64)
        tstep_arr = np.array(all_timestep_idx, dtype=np.int64)

        total_samples = len(all_old_lp)
        p_losses, v_losses = [], []

        for _ in range(self.n_epochs):
            perm = np.random.permutation(total_samples)
            for start in range(0, total_samples, self.batch_size):
                end = min(start + self.batch_size, total_samples)
                idx = perm[start:end]
                B = len(idx)

                batch_tsteps = tstep_arr[idx]
                batch_bids = bids_arr[idx]

                f_batch = torch.stack([feats_by_t[t] for t in batch_tsteps])
                n_batch = torch.stack([nbr_by_t[t] for t in batch_tsteps])
                embed_out = self.encoder(f_batch, n_batch)
                bid_idx_t = torch.LongTensor(batch_bids).to(self.device)
                h_agents = embed_out[torch.arange(B, device=self.device), bid_idx_t]

                m_batch = masks_t[idx]
                dist = self.actor(h_agents, m_batch)
                new_lp = dist.log_prob(actions_t[idx])
                entropy = dist.entropy().mean()

                old_lp_batch = old_lp_t[idx]
                ratio = (new_lp - old_lp_batch).exp()
                adv_batch = adv_t[idx]
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_eps,
                                    1 + self.clip_eps) * adv_batch
                policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coeff * entropy

                self.actor_optim.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) +
                    list(self.actor.parameters()), 0.5)
                self.actor_optim.step()

                v_pred = self.critic(gs_t[idx])
                v_loss = F.mse_loss(v_pred, ret_t[idx])
                self.critic_optim.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optim.step()

                p_losses.append(policy_loss.item())
                v_losses.append(v_loss.item())

        return (float(np.mean(p_losses)) if p_losses else 0.0,
                float(np.mean(v_losses)) if v_losses else 0.0)

    def run_window(self, env: MarineIoTEnv, source_ids: List[int],
                   n_steps: int = 20,
                   rng: Optional[np.random.Generator] = None
                   ) -> Dict[str, float]:
        """Run one evaluation window, compute LA metrics."""
        if rng is None:
            rng = np.random.default_rng()
        cfg = self.cfg
        nodes = env.nodes
        n = len(nodes)

        candidate_map = self._build_candidate_map(env, source_ids)
        action_dict, _, _ = self.select_actions(env, source_ids, candidate_map)
        n_switch = self._count_switches(action_dict)
        self._prev_actions.update(action_dict)

        path_las, path_qs, path_ss = [], [], []
        for bid in source_ids:
            hop_q, hop_s = self._evaluate_hops(env, bid,
                                               action_dict.get(bid, bid))
            if hop_q:
                q_pi = path_quality(hop_q)
                s_pi = path_stability(hop_s)
                la = link_advantage(q_pi, s_pi, cfg.w_Q, cfg.w_S)
                path_las.append(la)
                path_qs.append(q_pi)
                path_ss.append(s_pi)

        mean_la = float(np.mean(path_las)) if path_las else 0.0
        mean_q = float(np.mean(path_qs)) if path_qs else 0.0
        mean_s = float(np.mean(path_ss)) if path_ss else 0.0
        n_outage = sum(1 for bid in source_ids
                       if bid not in action_dict or action_dict[bid] == bid)

        actions_env = np.ones((n, 2), dtype=np.float32)
        for _ in range(n_steps):
            obs, _, term, trunc, _ = env.step(actions_env)
            if term or trunc:
                break
            self._update_sinr_histories(env)

        return {
            "mean_LA": mean_la,
            "mean_Q": mean_q,
            "mean_S": mean_s,
            "n_switch": n_switch,
            "n_outage": n_outage,
            "path_las": path_las,
        }

    @torch.inference_mode()
    def _get_value(self, gs: np.ndarray) -> float:
        gs_t = torch.FloatTensor(gs).unsqueeze(0).to(self.device)
        return self.critic(gs_t).item()

    def _ppo_update(self, n_src: int) -> Dict[str, float]:
        buf = self.buffer
        if len(buf) < 2:
            buf.clear()
            return {"policy_loss": 0.0, "value_loss": 0.0}

        T = len(buf.rewards)
        rewards = np.stack(buf.rewards)
        values = np.stack(buf.values)
        dones = np.array(buf.dones, dtype=np.float32)
        gs_arr = np.stack(buf.global_states)

        advantages = np.zeros_like(rewards)
        gae = np.zeros(n_src, dtype=np.float32)
        for t in reversed(range(T)):
            nv = values[t + 1] if t + 1 < T else values[-1]
            delta = rewards[t] + self.gamma * nv * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values

        flat_adv = (advantages.reshape(-1))
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)
        flat_ret = returns.reshape(-1)
        flat_gs = np.repeat(gs_arr, n_src, axis=0)

        adv_t = torch.FloatTensor(flat_adv).to(self.device)
        ret_t = torch.FloatTensor(flat_ret).to(self.device)
        gs_t = torch.FloatTensor(flat_gs).to(self.device)

        total = len(flat_adv)
        p_l, v_l = [], []
        for _ in range(self.n_epochs):
            idx = np.random.permutation(total)
            for s in range(0, total, self.batch_size):
                b = idx[s:s + self.batch_size]
                p_loss = -adv_t[b].mean()
                v_pred = self.critic(gs_t[b])
                v_loss = F.mse_loss(v_pred, ret_t[b])
                self.actor_optim.zero_grad()
                p_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) +
                    list(self.actor.parameters()), 0.5)
                self.actor_optim.step()
                self.critic_optim.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optim.step()
                p_l.append(p_loss.item())
                v_l.append(v_loss.item())
        buf.clear()
        return {"policy_loss": float(np.mean(p_l)) if p_l else 0.0,
                "value_loss": float(np.mean(v_l)) if v_l else 0.0}

    # ─── shared helpers (same logic as GMAPPO) ────────────────────────

    def _build_candidate_map(self, env, source_ids):
        cmap = {}
        gamma_lin = self.cfg.gamma_link_linear
        ships_uavs = [nd.node_id for nd in env.nodes
                      if nd.node_type in ("ship", "uav")]
        for bid in source_ids:
            scored = []
            for cid in ships_uavs:
                lp = env.link_phy.get((bid, cid))
                if lp and lp.snr >= gamma_lin:
                    scored.append((cid, lp.p_sig))
            scored.sort(key=lambda x: x[1], reverse=True)
            cmap[bid] = [c for c, _ in scored[:self.cfg.K_nbr]]
        return cmap

    def _count_switches(self, ad):
        return sum(1 for b, c in ad.items()
                   if self._prev_actions.get(b) is not None and
                   self._prev_actions[b] != c)

    def _get_comm_range(self, type_i: str, type_j: str) -> float:
        key = (type_i, type_j)
        if key not in self._comm_range_cache:
            from Env.phy import communication_range_estimate
            self._comm_range_cache[key] = communication_range_estimate(
                type_i, type_j, self.cfg)
        return self._comm_range_cache[key]

    def _predict_q(self, lp, lc, tx, rx):
        cached = self._q_cache.get((tx, rx))
        if cached is not None:
            return cached
        if not self.estimator.is_trained:
            sig = max(lp.snr, 1e-30)
            ber = 0.5 * math.erfc(math.sqrt(sig))
            exp = min(int(self.cfg.L_pkt), 64)
            q = max(1e-6, min(1.0, (1.0 - ber) ** exp))
        else:
            hist = self._sinr_histories.get((tx, rx), [lp.snr])
            sa = np.array(hist[-10:])
            q = self.estimator.predict_single(
                lc, float(lp.rssi), float(lp.snr), float(lp.sinr),
                compute_lqi(lp.sinr),
                float(np.mean(sa)), float(np.std(sa)),
                float(lp.rssi), 0.0,
                float(lp.doppler), 0, len(hist))
        self._q_cache[(tx, rx)] = q
        return q

    def _compute_s_ho(self, env, tx_id, rx_id):
        tx_n, rx_n = env.nodes[tx_id], env.nodes[rx_id]
        dp = rx_n.position - tx_n.position
        dv = rx_n.velocity - tx_n.velocity
        r = self._get_comm_range(tx_n.node_type, rx_n.node_type)
        let = compute_let(dp, dv, r)
        hist = self._sinr_histories.get((tx_id, rx_id), [])
        if len(hist) < 2:
            lp = env.link_phy.get((tx_id, rx_id))
            v = lp.snr if lp else 1.0
            hist = [v, v]
        p = compute_p_surv(np.array(hist[-10:]), self.cfg.gamma_ho_linear,
                           self.cfg.N_p, self.cfg.delta_t_sim * 1e-3)
        return compute_s_ho(let, p, self.cfg.tau_req * 1e-3)

    def _evaluate_hops(self, env, bid, first_hop):
        nodes = env.nodes
        hq, hs = [], []
        lp = env.link_phy.get((bid, first_hop))
        if not lp:
            return [], []
        lc = get_link_class(nodes[bid].node_type, nodes[first_hop].node_type)
        hq.append(self._predict_q(lp, lc, bid, first_hop))
        hs.append(self._compute_s_ho(env, bid, first_hop))
        best_sat, best_sig = None, -1.0
        for nd in nodes:
            if nd.node_type != "satellite":
                continue
            lps = env.link_phy.get((first_hop, nd.node_id))
            if lps and lps.snr >= self.cfg.gamma_link_linear and lps.p_sig > best_sig:
                best_sat, best_sig = nd, lps.p_sig
        if not best_sat:
            return [], []
        lps = env.link_phy[(first_hop, best_sat.node_id)]
        lcs = get_link_class(nodes[first_hop].node_type, best_sat.node_type)
        hq.append(self._predict_q(lps, lcs, first_hop, best_sat.node_id))
        hs.append(self._compute_s_ho(env, first_hop, best_sat.node_id))
        best_land, best_sl = None, -1.0
        for nd in nodes:
            if nd.node_type != "land":
                continue
            lpl = env.link_phy.get((best_sat.node_id, nd.node_id))
            if lpl and lpl.p_sig > best_sl:
                best_land, best_sl = nd, lpl.p_sig
        if not best_land:
            return [], []
        lpl = env.link_phy[(best_sat.node_id, best_land.node_id)]
        lcl = get_link_class(best_sat.node_type, best_land.node_type)
        hq.append(self._predict_q(lpl, lcl, best_sat.node_id, best_land.node_id))
        hs.append(self._compute_s_ho(env, best_sat.node_id, best_land.node_id))
        return hq, hs

    def _update_sinr_histories(self, env):
        m = getattr(env, '_phy', None)
        if m is None:
            return
        N = m.SNR.shape[0]
        tx_indices = np.where(m.tx_active)[0]
        node_ids = m.node_ids
        snr_matrix = m.SNR
        for ti in tx_indices:
            tx_id = int(node_ids[ti])
            row = snr_matrix[ti]
            for rj in range(N):
                if ti == rj:
                    continue
                rx_id = int(node_ids[rj])
                key = (tx_id, rx_id)
                hist = self._sinr_histories[key]
                hist.append(float(row[rj]))
                if len(hist) > 20:
                    del hist[:-20]
