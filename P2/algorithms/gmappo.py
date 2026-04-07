"""
GMAPPO — Graph-attention Multi-Agent PPO for MEC-aware link selection.

Architecture (Manuscript II Section 6):
  * 2-layer GCN encoder aggregating neighbor node & edge features
  * Per-agent actor: GCN-encoded obs -> discrete action (next-hop selection)
  * Global critic: concatenated graph summary -> V(s)
  * Action masking: only feasible next-hops selectable
  * Reward: mean_b(LA_pi_b) - eta_sw * N_switch - penalties
  * Residual connections and LayerNorm in GCN for stable training
  * Orthogonal initialization
  * LR cosine annealing
  * Running reward normalization
  * Proper PPO-clip update with GAE
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
from scipy.special import erfc as _erfc_vec

from Env.config import EnvConfig
from Env.core_env import MarineIoTEnv
from Env.channel import link_class as get_link_class

from P2.link_quality.metrics import (
    compute_let, compute_p_surv, compute_s_ho,
    path_quality, path_stability, link_advantage, compute_lqi,
)
from P2.link_quality.path_manager import PathManager, ServicePath
from P2.link_quality.rf_estimator import LinkQualityEstimator


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _ortho_init(module: nn.Module, gain: float = 1.0):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class RunningNormalizer:
    def __init__(self, clip: float = 5.0):
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        self.clip = clip

    def update(self, x: float):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.var += (delta * delta2 - self.var) / self.count

    def normalize(self, x: float) -> float:
        std = max(math.sqrt(self.var), 1e-6)
        return max(-self.clip, min(self.clip, (x - self.mean) / std))


# ═══════════════════════════════════════════════════════════════════════════
# GCN Encoder with residual connections and LayerNorm
# ═══════════════════════════════════════════════════════════════════════════

class GCNLayer(nn.Module):
    """Single graph convolution layer with edge features."""

    def __init__(self, node_dim: int, edge_dim: int, out_dim: int):
        super().__init__()
        self.W_node = nn.Linear(node_dim, out_dim)
        self.W_edge = nn.Linear(edge_dim, out_dim)
        self.W_self = nn.Linear(node_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, h: torch.Tensor, adj: torch.Tensor,
                edge_feat: torch.Tensor) -> torch.Tensor:
        """
        h:         (B, N, node_dim)
        adj:       (B, N, N)  binary adjacency
        edge_feat: (B, N, N, edge_dim)
        returns:   (B, N, out_dim)
        """
        nbr_msg = self.W_node(h).unsqueeze(2).expand(
            h.shape[0], h.shape[1], h.shape[1], self.W_node.out_features)
        edge_msg = self.W_edge(edge_feat)
        combined = (nbr_msg + edge_msg) * adj.unsqueeze(-1)
        degree = adj.sum(dim=-1, keepdim=True).clamp(min=1)
        agg = combined.sum(dim=2) / degree
        out = self.ln(agg + self.W_self(h))
        return F.relu(out)


class GCNEncoder(nn.Module):
    """Two-layer GCN with residual connections."""

    def __init__(self, node_dim: int, edge_dim: int, hidden: int = 64):
        super().__init__()
        self.layer1 = GCNLayer(node_dim, edge_dim, hidden)
        self.layer2 = GCNLayer(hidden, edge_dim, hidden)
        self.input_proj = nn.Linear(node_dim, hidden)

    def forward(self, h: torch.Tensor, adj: torch.Tensor,
                edge_feat: torch.Tensor) -> torch.Tensor:
        h1 = self.layer1(h, adj, edge_feat)
        h2 = self.layer2(h1, adj, edge_feat)
        return h2 + self.input_proj(h)


# ═══════════════════════════════════════════════════════════════════════════
# Actor / Critic
# ═══════════════════════════════════════════════════════════════════════════

class GCNActor(nn.Module):
    """GCN-encoded observation -> discrete action logits with masking."""

    def __init__(self, gcn_dim: int, max_actions: int, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(gcn_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.fc3 = nn.Linear(hidden, max_actions)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> Categorical:
        x = F.relu(self.ln1(self.fc1(h)))
        x = F.relu(self.ln2(self.fc2(x)))
        logits = self.fc3(x)
        logits = logits.masked_fill(~mask.bool(), -1e9)
        return Categorical(logits=logits)


class GlobalCritic(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.fc3 = nn.Linear(hidden, 1)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.ln1(self.fc1(s)))
        x = F.relu(self.ln2(self.fc2(x)))
        return self.fc3(x).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════
# Rollout buffer
# ═══════════════════════════════════════════════════════════════════════════

class RolloutBuffer:
    def __init__(self):
        self.obs, self.global_states, self.actions = [], [], []
        self.log_probs, self.rewards, self.dones, self.values = [], [], [], []
        self.masks = []

    def store(self, obs, gs, actions, lp, reward, done, value, masks):
        self.obs.append(obs)
        self.global_states.append(gs)
        self.actions.append(actions)
        self.log_probs.append(lp)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.masks.append(masks)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.obs)


# ═══════════════════════════════════════════════════════════════════════════
# GMAPPO Agent
# ═══════════════════════════════════════════════════════════════════════════

NODE_FEAT_DIM = 10
EDGE_FEAT_DIM = 5
MAX_ACTIONS = 16

class GMAPPO:
    """Graph-attention Multi-Agent PPO with proper PPO-clip update."""

    def __init__(self, n_agents: int, cfg: EnvConfig,
                 estimator: LinkQualityEstimator,
                 actor_lr: float = 3e-4, critic_lr: float = 1e-3,
                 lr: float = 3e-4, gamma: float = 0.99, lam: float = 0.95,
                 clip_eps: float = 0.2, entropy_coeff: float = 0.02,
                 n_epochs: int = 4, batch_size: int = 64,
                 lr_t_max: int = 1000,
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
        self._episode_count = 0

        gcn_hidden = 64
        self.gcn = GCNEncoder(NODE_FEAT_DIM, EDGE_FEAT_DIM,
                              gcn_hidden).to(self.device)
        self.actor = GCNActor(gcn_hidden, MAX_ACTIONS).to(self.device)

        self.gcn_hidden = gcn_hidden
        global_state_dim = gcn_hidden + 4
        self.critic = GlobalCritic(global_state_dim).to(self.device)

        _ortho_init(self.gcn, gain=math.sqrt(2))
        _ortho_init(self.actor, gain=0.01)
        _ortho_init(self.critic, gain=math.sqrt(2))
        nn.init.orthogonal_(self.critic.fc3.weight, gain=1.0)

        a_lr = actor_lr if actor_lr != 3e-4 or lr == 3e-4 else lr
        c_lr = critic_lr if critic_lr != 1e-3 or lr == 3e-4 else lr

        params = (list(self.gcn.parameters()) +
                  list(self.actor.parameters()))
        self.actor_optim = torch.optim.Adam(params, lr=a_lr)
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(), lr=c_lr)

        self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optim, T_max=lr_t_max, eta_min=a_lr * 0.1)
        self.critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.critic_optim, T_max=lr_t_max, eta_min=c_lr * 0.1)

        self.buffer = RolloutBuffer()
        self.path_mgr = PathManager(cfg)
        self._prev_actions: Dict[int, int] = {}
        self._sinr_histories: Dict[Tuple[int, int], List[float]] = defaultdict(list)
        self.reward_normalizer = RunningNormalizer(clip=5.0)
        self._comm_range_cache: Dict[Tuple[str, str], float] = {}

    # ─── feature construction ─────────────────────────────────────────

    def build_node_features(self, env: MarineIoTEnv) -> np.ndarray:
        """(N, NODE_FEAT_DIM) node feature matrix."""
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

    def build_adj_and_edge_features(
            self, env: MarineIoTEnv) -> Tuple[np.ndarray, np.ndarray]:
        """(N,N) adjacency and (N,N,EDGE_FEAT_DIM) edge features.

        RF predictions are batched per link_class to avoid per-link sklearn
        overhead (~20ms x 368 links = 7.5s -> batched ~30ms total).
        """
        n = len(env.nodes)
        adj = np.zeros((n, n), dtype=np.float32)
        edge_f = np.zeros((n, n, EDGE_FEAT_DIM), dtype=np.float32)
        gamma_lin = self.cfg.gamma_link_linear
        _lc_enc = {"satellite": 0, "uav_terrestrial": 1,
                    "sea_surface": 2, "terrestrial": 3}

        qualifying: List[Tuple[int, int, str, any]] = []
        for (tx_id, rx_id), lp in env.link_phy.items():
            if lp.snr < gamma_lin:
                continue
            lc = get_link_class(env.nodes[tx_id].node_type,
                                env.nodes[rx_id].node_type)
            qualifying.append((tx_id, rx_id, lc, lp))

        q_values = self._batch_predict_q(env, qualifying)

        for k, (tx_id, rx_id, lc, lp) in enumerate(qualifying):
            adj[rx_id, tx_id] = 1.0
            s = self._compute_s_ho_for_link(env, tx_id, rx_id)
            edge_f[rx_id, tx_id, 0] = q_values[k]
            edge_f[rx_id, tx_id, 1] = s
            edge_f[rx_id, tx_id, 2] = min(lp.distance / 100_000.0, 1.0)
            edge_f[rx_id, tx_id, 3] = lp.doppler / 1000.0
            edge_f[rx_id, tx_id, 4] = _lc_enc.get(lc, 0) / 3.0
        return adj, edge_f

    def build_global_state(self, gcn_out: np.ndarray,
                           mean_la: float, n_switch: int) -> np.ndarray:
        pooled = gcn_out.mean(axis=0)
        extras = np.array([mean_la, float(n_switch),
                           float(len(gcn_out)), 0.0], dtype=np.float32)
        return np.concatenate([pooled, extras])

    # ─── action selection ─────────────────────────────────────────────

    @torch.inference_mode()
    def select_actions(
            self, env: MarineIoTEnv, source_ids: List[int],
            candidate_map: Dict[int, List[int]]
    ) -> Tuple[Dict[int, int], np.ndarray, np.ndarray, np.ndarray]:
        """Select next-hop for each source buoy."""
        node_feats = self.build_node_features(env)
        adj, edge_f = self.build_adj_and_edge_features(env)

        h = torch.FloatTensor(node_feats).unsqueeze(0).to(self.device)
        a = torch.FloatTensor(adj).unsqueeze(0).to(self.device)
        ef = torch.FloatTensor(edge_f).unsqueeze(0).to(self.device)
        gcn_out = self.gcn(h, a, ef).squeeze(0)
        gcn_np = gcn_out.cpu().numpy()

        action_dict: Dict[int, int] = {}
        all_lp = np.zeros(len(source_ids), dtype=np.float32)
        all_masks = np.zeros((len(source_ids), MAX_ACTIONS), dtype=np.float32)

        for idx, bid in enumerate(source_ids):
            candidates = candidate_map.get(bid, [])
            mask = np.zeros(MAX_ACTIONS, dtype=np.float32)
            mask[0] = 1.0
            for ci, cid in enumerate(candidates[:MAX_ACTIONS - 1]):
                mask[ci + 1] = 1.0

            h_agent = gcn_out[bid].unsqueeze(0)
            m_t = torch.FloatTensor(mask).unsqueeze(0).to(self.device)
            dist = self.actor(h_agent, m_t)
            action = dist.sample()
            lp = dist.log_prob(action)
            a_idx = action.item()

            if a_idx == 0:
                chosen = self._prev_actions.get(bid, candidates[0]
                                                 if candidates else bid)
            elif a_idx - 1 < len(candidates):
                chosen = candidates[a_idx - 1]
            else:
                chosen = self._prev_actions.get(bid, bid)

            action_dict[bid] = chosen
            all_lp[idx] = lp.item()
            all_masks[idx] = mask

        return action_dict, all_lp, gcn_np, all_masks

    @torch.inference_mode()
    def get_value(self, global_state: np.ndarray) -> float:
        gs_t = torch.FloatTensor(global_state).unsqueeze(0).to(self.device)
        return self.critic(gs_t).item()

    # ─── environment evaluation window ────────────────────────────────

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
        action_dict, _, gcn_np, masks = self.select_actions(
            env, source_ids, candidate_map)

        n_switch = self._count_switches(action_dict)
        self._prev_actions.update(action_dict)

        all_hops = self._evaluate_all_buoys_batched(
            env, source_ids, action_dict)
        path_las, path_qs, path_ss = [], [], []
        for hop_q, hop_s in all_hops:
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
                       if bid not in action_dict or
                       action_dict[bid] == bid)

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

    # ─── training (proper PPO-clip) ──────────────────────────────────

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

        buf_obs = []
        buf_adj = []
        buf_edge_f = []
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

            candidate_map = self._build_candidate_map(env, source_ids)

            node_feats = self.build_node_features(env)
            adj, edge_f = self.build_adj_and_edge_features(env)

            h_t = torch.FloatTensor(node_feats).unsqueeze(0).to(self.device)
            a_t = torch.FloatTensor(adj).unsqueeze(0).to(self.device)
            ef_t = torch.FloatTensor(edge_f).unsqueeze(0).to(self.device)

            with torch.inference_mode():
                gcn_out = self.gcn(h_t, a_t, ef_t).squeeze(0)

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

                h_agent = gcn_out[bid].unsqueeze(0)
                m_t_mask = torch.FloatTensor(mask).unsqueeze(0).to(self.device)
                with torch.inference_mode():
                    dist = self.actor(h_agent, m_t_mask)
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

            all_hops = self._evaluate_all_buoys_batched(
                env, source_ids, action_dict)

            per_agent_r = np.zeros(n_src, dtype=np.float32)
            per_agent_la = np.zeros(n_src, dtype=np.float32)
            for idx, bid in enumerate(source_ids):
                hop_q, hop_s = all_hops[idx]
                if hop_q:
                    q_pi = path_quality(hop_q)
                    s_pi = path_stability(hop_s)
                    la = link_advantage(q_pi, s_pi, cfg.w_Q, cfg.w_S)
                else:
                    la = 0.0
                per_agent_la[idx] = la

                did_switch = (self._prev_actions.get(bid) is not None
                              and action_dict.get(bid) != self._prev_actions.get(bid))
                kept_link = (self._prev_actions.get(bid) is not None
                             and action_dict.get(bid) == self._prev_actions.get(bid))
                no_path = len(candidate_map.get(bid, [])) == 0

                r_i = la - cfg.eta_sw * float(did_switch) \
                      + 0.05 * float(kept_link) \
                      - 0.1 * float(no_path)
                per_agent_r[idx] = r_i

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

            gcn_np = gcn_out.detach().cpu().numpy()
            n_switch = self._count_switches(action_dict)
            gs = self.build_global_state(gcn_np, mean_la, n_switch)
            value = self.get_value(gs)

            buf_obs.append(node_feats)
            buf_adj.append(adj)
            buf_edge_f.append(edge_f)
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
            buf_obs, buf_adj, buf_edge_f, buf_gs, buf_actions_idx,
            buf_log_probs, buf_agent_rewards, buf_values, buf_masks_list,
            buf_source_bids, buf_dones)

        self._episode_count += 1
        self.actor_scheduler.step()
        self.critic_scheduler.step()

        return {
            "mean_reward": float(np.mean(ep_rewards)),
            "mean_LA": float(np.mean(ep_la)),
            "policy_loss": p_loss_val,
            "value_loss": v_loss_val,
        }

    def _ppo_clip_update(self, buf_obs, buf_adj, buf_edge_f, buf_gs,
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

        obs_by_t = [torch.FloatTensor(buf_obs[t]).to(self.device) for t in range(T)]
        adj_by_t = [torch.FloatTensor(buf_adj[t]).to(self.device) for t in range(T)]
        ef_by_t = [torch.FloatTensor(buf_edge_f[t]).to(self.device) for t in range(T)]
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

                h_batch = torch.stack([obs_by_t[t] for t in batch_tsteps])
                a_batch = torch.stack([adj_by_t[t] for t in batch_tsteps])
                ef_batch = torch.stack([ef_by_t[t] for t in batch_tsteps])

                gcn_out = self.gcn(h_batch, a_batch, ef_batch)
                bid_idx_t = torch.LongTensor(batch_bids).to(self.device)
                h_agents = gcn_out[torch.arange(B, device=self.device), bid_idx_t]

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
                    list(self.gcn.parameters()) +
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

    # ─── internal helpers ─────────────────────────────────────────────

    def _build_candidate_map(self, env: MarineIoTEnv,
                             source_ids: List[int]) -> Dict[int, List[int]]:
        """For each source buoy, list candidate first-hop node IDs."""
        cmap: Dict[int, List[int]] = {}
        gamma_lin = self.cfg.gamma_link_linear
        ships_uavs = [nd.node_id for nd in env.nodes
                      if nd.node_type in ("ship", "uav")]
        for bid in source_ids:
            scored = []
            for cid in ships_uavs:
                lp = env.link_phy.get((bid, cid))
                if lp is not None and lp.snr >= gamma_lin:
                    scored.append((cid, lp.p_sig))
            scored.sort(key=lambda x: x[1], reverse=True)
            cmap[bid] = [c for c, _ in scored[:self.cfg.K_nbr]]
        return cmap

    def _count_switches(self, action_dict: Dict[int, int]) -> int:
        count = 0
        for bid, chosen in action_dict.items():
            prev = self._prev_actions.get(bid)
            if prev is not None and prev != chosen:
                count += 1
        return count

    def _get_comm_range(self, type_i: str, type_j: str) -> float:
        """Cached communication range estimate per type-pair."""
        key = (type_i, type_j)
        if key not in self._comm_range_cache:
            from Env.phy import communication_range_estimate
            self._comm_range_cache[key] = communication_range_estimate(
                type_i, type_j, self.cfg)
        return self._comm_range_cache[key]

    def _predict_q(self, lp, lc: str, tx_id: int, rx_id: int) -> float:
        if not self.estimator.is_trained:
            sig = max(lp.snr, 1e-30)
            ber = 0.5 * math.erfc(math.sqrt(sig))
            exp = min(int(self.cfg.L_pkt), 64)
            return max(1e-6, min(1.0, (1.0 - ber) ** exp))

        key = (tx_id, rx_id)
        hist = self._sinr_histories.get(key, [lp.snr])
        sinr_arr = np.array(hist[-10:])
        rssi_arr = np.array([lp.rssi] * len(sinr_arr))

        return self.estimator.predict_single(
            lc, float(lp.rssi), float(lp.snr), float(lp.sinr),
            compute_lqi(lp.sinr),
            float(np.mean(sinr_arr)), float(np.std(sinr_arr)),
            float(np.mean(rssi_arr)), float(np.std(rssi_arr)),
            float(lp.doppler), 0, len(hist))

    def _batch_predict_q(
            self, env: MarineIoTEnv,
            links: List[Tuple[int, int, str, "LinkPHY"]],
    ) -> np.ndarray:
        """Batched Q prediction — groups by link_class and calls predict_prr
        once per class, avoiding per-link sklearn overhead."""
        n_links = len(links)
        if n_links == 0:
            return np.empty(0, dtype=np.float64)

        if not self.estimator.is_trained:
            sig_arr = np.array([
                max(lp.snr, 1e-30)
                for _, _, lc, lp in links], dtype=np.float64)
            ber_arr = 0.5 * _erfc_vec(np.sqrt(sig_arr))
            exp = min(int(self.cfg.L_pkt), 64)
            prr = np.clip((1.0 - ber_arr) ** exp, 1e-6, 1.0)
            return prr

        by_class: Dict[str, List[Tuple[int, np.ndarray]]] = {}
        for k, (tx_id, rx_id, lc, lp) in enumerate(links):
            key = (tx_id, rx_id)
            hist = self._sinr_histories.get(key, [lp.snr])
            sinr_arr = np.array(hist[-10:])
            rssi_arr = np.array([lp.rssi] * len(sinr_arr))
            row = np.array([
                lp.rssi, lp.snr, lp.sinr, compute_lqi(lp.sinr),
                np.mean(sinr_arr), np.std(sinr_arr),
                np.mean(rssi_arr), np.std(rssi_arr),
                lp.doppler, 0.0, float(len(hist)),
            ], dtype=np.float64)
            by_class.setdefault(lc, []).append((k, row))

        out = np.empty(n_links, dtype=np.float64)
        for lc, items in by_class.items():
            indices = [i for i, _ in items]
            features = np.stack([r for _, r in items])
            preds = self.estimator.predict_prr(lc, features)
            for idx, pred in zip(indices, preds):
                out[idx] = pred
        return out

    def _compute_s_ho_for_link(self, env: MarineIoTEnv,
                               tx_id: int, rx_id: int) -> float:
        tx_n = env.nodes[tx_id]
        rx_n = env.nodes[rx_id]
        dp = rx_n.position - tx_n.position
        dv = rx_n.velocity - tx_n.velocity
        r_comm = self._get_comm_range(tx_n.node_type, rx_n.node_type)

        let = compute_let(dp, dv, r_comm)
        key = (tx_id, rx_id)
        hist = self._sinr_histories.get(key, [])
        if len(hist) < 2:
            lp = env.link_phy.get((tx_id, rx_id))
            snr_val = lp.snr if lp else 1.0
            hist = [snr_val, snr_val]
        sinr_arr = np.array(hist[-10:])
        p_surv = compute_p_surv(sinr_arr, self.cfg.gamma_ho_linear,
                                self.cfg.N_p, self.cfg.delta_t_sim * 1e-3)
        tau_req_s = self.cfg.tau_req * 1e-3
        return compute_s_ho(let, p_surv, tau_req_s)

    def _evaluate_hops_for_buoy(
            self, env: MarineIoTEnv, bid: int, first_hop: int
    ) -> Tuple[List[float], List[float]]:
        """Build a minimal path and return per-hop Q and S_HO lists."""
        nodes = env.nodes
        hop_q, hop_s = [], []

        lp = env.link_phy.get((bid, first_hop))
        if lp is None:
            return [], []
        lc = get_link_class(nodes[bid].node_type, nodes[first_hop].node_type)
        hop_q.append(self._predict_q(lp, lc, bid, first_hop))
        hop_s.append(self._compute_s_ho_for_link(env, bid, first_hop))

        sats = [nd for nd in nodes if nd.node_type == "satellite"]
        best_sat, best_sig = None, -1.0
        for sat in sats:
            lp_s = env.link_phy.get((first_hop, sat.node_id))
            if lp_s and lp_s.snr >= self.cfg.gamma_link_linear:
                if lp_s.p_sig > best_sig:
                    best_sat = sat
                    best_sig = lp_s.p_sig
        if best_sat is None:
            return [], []
        lp_s = env.link_phy[(first_hop, best_sat.node_id)]
        lc_s = get_link_class(nodes[first_hop].node_type, best_sat.node_type)
        hop_q.append(self._predict_q(lp_s, lc_s, first_hop, best_sat.node_id))
        hop_s.append(self._compute_s_ho_for_link(env, first_hop, best_sat.node_id))

        lands = [nd for nd in nodes if nd.node_type == "land"]
        best_land, best_sig_l = None, -1.0
        for ld in lands:
            lp_l = env.link_phy.get((best_sat.node_id, ld.node_id))
            if lp_l and lp_l.p_sig > best_sig_l:
                best_land = ld
                best_sig_l = lp_l.p_sig
        if best_land is None:
            return [], []
        lp_l = env.link_phy[(best_sat.node_id, best_land.node_id)]
        lc_l = get_link_class(best_sat.node_type, best_land.node_type)
        hop_q.append(self._predict_q(lp_l, lc_l, best_sat.node_id, best_land.node_id))
        hop_s.append(self._compute_s_ho_for_link(env, best_sat.node_id, best_land.node_id))

        return hop_q, hop_s

    def _evaluate_all_buoys_batched(
            self, env: MarineIoTEnv,
            source_ids: List[int],
            action_dict: Dict[int, int],
    ) -> List[Tuple[List[float], List[float]]]:
        """Evaluate hop paths for all buoys with batched RF prediction."""
        nodes = env.nodes
        gamma_lin = self.cfg.gamma_link_linear
        sats = [nd for nd in nodes if nd.node_type == "satellite"]
        lands = [nd for nd in nodes if nd.node_type == "land"]

        hop_chains: List[Optional[List[Tuple[int, int, str, "LinkPHY"]]]] = []
        for bid in source_ids:
            first_hop = action_dict.get(bid, bid)
            lp = env.link_phy.get((bid, first_hop))
            if lp is None:
                hop_chains.append(None)
                continue
            chain: List[Tuple[int, int, str, "LinkPHY"]] = []
            lc = get_link_class(nodes[bid].node_type, nodes[first_hop].node_type)
            chain.append((bid, first_hop, lc, lp))

            best_sat, best_sig = None, -1.0
            for sat in sats:
                lp_s = env.link_phy.get((first_hop, sat.node_id))
                if lp_s and lp_s.snr >= gamma_lin and lp_s.p_sig > best_sig:
                    best_sat = sat
                    best_sig = lp_s.p_sig
            if best_sat is None:
                hop_chains.append(None)
                continue
            lp_s = env.link_phy[(first_hop, best_sat.node_id)]
            lc_s = get_link_class(nodes[first_hop].node_type, best_sat.node_type)
            chain.append((first_hop, best_sat.node_id, lc_s, lp_s))

            best_land, best_sig_l = None, -1.0
            for ld in lands:
                lp_l = env.link_phy.get((best_sat.node_id, ld.node_id))
                if lp_l and lp_l.p_sig > best_sig_l:
                    best_land = ld
                    best_sig_l = lp_l.p_sig
            if best_land is None:
                hop_chains.append(None)
                continue
            lp_l = env.link_phy[(best_sat.node_id, best_land.node_id)]
            lc_l = get_link_class(best_sat.node_type, best_land.node_type)
            chain.append((best_sat.node_id, best_land.node_id, lc_l, lp_l))
            hop_chains.append(chain)

        all_links: List[Tuple[int, int, str, "LinkPHY"]] = []
        chain_offsets: List[Optional[Tuple[int, int]]] = []
        for chain in hop_chains:
            if chain is None:
                chain_offsets.append(None)
            else:
                start = len(all_links)
                all_links.extend(chain)
                chain_offsets.append((start, start + len(chain)))

        q_all = self._batch_predict_q(env, all_links) if all_links else np.empty(0)

        results: List[Tuple[List[float], List[float]]] = []
        for i, chain in enumerate(hop_chains):
            if chain is None:
                results.append(([], []))
                continue
            start, end = chain_offsets[i]  # type: ignore[misc]
            hop_q = q_all[start:end].tolist()
            hop_s = [self._compute_s_ho_for_link(env, tx, rx)
                     for tx, rx, _, _ in chain]
            results.append((hop_q, hop_s))
        return results

    def _update_sinr_histories(self, env: MarineIoTEnv):
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
