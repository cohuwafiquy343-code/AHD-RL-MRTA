"""AHD-RL 的任务动作与智能体动作策略网络。"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CrossDecoder, TransformerEncoder
from .config import (
    CC, D_CAP, DFF, DM, DROP, NH, NLR, NLT, PHASE_ROBOT_ACCEL,
    ROBOT_FEAT_DIM, ROBOT_TASK_FEAT_DIM, TASK_SELECT_FEAT_DIM,
)


class SchedulingNetwork(nn.Module):
    """分别编码两类观测，再用任务和智能体解码器计算动作分数。"""

    def __init__(self, dropout: Optional[float] = None):
        super().__init__()
        d = DM
        dropout = DROP if dropout is None else float(dropout)
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.dropout = dropout

        # 阶段分离嵌入投影
        self.task_emb_sel = nn.Linear(TASK_SELECT_FEAT_DIM, d)
        self.robot_emb_sel = nn.Linear(ROBOT_FEAT_DIM, d)
        self.task_emb_rob = nn.Linear(ROBOT_TASK_FEAT_DIM, d)
        self.robot_emb_rob = nn.Linear(ROBOT_FEAT_DIM, d)

        # 共享 Transformer 编码器
        self.task_enc = TransformerEncoder(NLT, d, NH, DFF, dropout)
        self.robot_enc = TransformerEncoder(NLR, d, NH, DFF, dropout)
        self.cross_t2r = CrossDecoder(2, d, NH, DFF, dropout)
        self.cross_r2t = CrossDecoder(2, d, NH, DFF, dropout)

        # Ph0 任务选择解码器
        self.W_local = nn.Linear(d * 3, d, bias=False)
        self.W_score_ph0 = nn.Linear(d, d, bias=False)
        self.focus_attn_ph0 = nn.MultiheadAttention(d, NH, dropout=dropout, batch_first=True)
        self.ln_r_global = nn.LayerNorm(d)
        self.ln_r_contrib = nn.LayerNorm(d)
        self.ln_q_ph0 = nn.LayerNorm(d)
        self.ln_kv_ph0 = nn.LayerNorm(d)

        # Ph1/Ph2 机器人选择解码器
        self.ln_ctx = nn.LayerNorm(d)
        self.W_fuse = nn.Linear(d * 2, d, bias=False)
        self.ln_task_fused = nn.LayerNorm(d)
        self.W_gap = nn.Linear(D_CAP, d, bias=False)
        self.W_global = nn.Linear(d * 3, d, bias=False)
        self.W_ind = nn.Linear(d * 2, d, bias=False)
        self.W_stop = nn.Linear(d * 2, d, bias=False)
        self.W_score_ph1 = nn.Linear(d, d, bias=False)
        self.focus_attn_ph1 = nn.MultiheadAttention(d, NH, dropout=dropout, batch_first=True)
        self.ln_q_ph1 = nn.LayerNorm(d)
        self.ln_kv_ph1 = nn.LayerNorm(d)
        self.ln_stop = nn.LayerNorm(d)
        self.stop_token = nn.Parameter(torch.randn(d) * 0.02)

        # 联盟状态（置换不变）
        self.W_proj = nn.Linear(d, d, bias=False)
        self.ln_coal_proj = nn.LayerNorm(d)

    # ── 共享编码 ──────────────────────────────────────────────────────────
    def _encode(
        self,
        task_feat: torch.Tensor,
        robot_feat: torch.Tensor,
        task_pad_mask: Optional[torch.Tensor] = None,
        phase: str = 'sel',
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """分别编码任务与智能体，并进行双向交叉注意力信息传递。"""
        task_emb = self.task_emb_sel if phase == 'sel' else self.task_emb_rob
        robot_emb = self.robot_emb_sel if phase == 'sel' else self.robot_emb_rob
        h_t = self.task_enc(task_emb(task_feat), key_padding_mask=task_pad_mask)
        h_r = self.robot_enc(robot_emb(robot_feat))
        h_t2 = self.cross_t2r(h_t, h_r)
        h_r2 = self.cross_r2t(h_r, h_t, context_key_padding_mask=task_pad_mask)
        return h_t2, h_r2

    # ── 联盟状态更新（外部调用）──────────────────────────────────────────
    def update_coal_state(
            self,
            coal_indices: List[int],
            h_r2_sq: torch.Tensor,
            device: torch.device,
    ) -> torch.Tensor:
        """置换不变联盟状态：对联盟成员嵌入求和后投影。"""
        if not coal_indices:
            return torch.zeros(DM, device=device)
        h_sum = h_r2_sq[coal_indices].sum(dim=0)
        return self.ln_coal_proj(self.W_proj(h_sum))

    # ── Ph0 单步推理 ──────────────────────────────────────────────────────
    def forward_task_select(
            self,
            task_feat: torch.Tensor,
            robot_feat: torch.Tensor,
            mask: torch.Tensor,
            contrib_mask: Optional[torch.Tensor] = None,
            depot_feat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 (logits (N_act,), h_t_tilde (N_act, d))
        站点节点始终拼接在索引0，参与注意力但不可被选择。
        """
        device = task_feat.device
        N_act = task_feat.shape[0]

        # 拼接站点节点（索引0，不可选）
        if depot_feat is not None:
            dep_row = depot_feat.unsqueeze(0)
        else:
            dep_row = torch.zeros(1, task_feat.shape[-1], device=device, dtype=task_feat.dtype)
        tf_ext = torch.cat([dep_row, task_feat], dim=0)  # (N_act+1, TASK_SELECT_FEAT_DIM)

        if contrib_mask is not None:
            M_c = contrib_mask.shape[-1]
            dep_contrib = torch.zeros(1, M_c, dtype=contrib_mask.dtype, device=device)
            contrib_ext = torch.cat([dep_contrib, contrib_mask], dim=0)  # (N_act+1, M)
        else:
            contrib_ext = None

        h_t, h_r = self._encode(tf_ext.unsqueeze(0), robot_feat.unsqueeze(0), phase='sel')
        h_t_sq = h_t.squeeze(0)  # (N_act+1, d)
        h_r_sq = h_r.squeeze(0)
        N_total = h_t_sq.shape[0]
        M = h_r_sq.shape[0]

        load_vec = robot_feat[:, 6].float()
        # 团队摘要按最早可用时间指标加权，任务摘要只关注有能力贡献的智能体。
        w_global = F.softmax(-load_vec, dim=0)
        h_r_global = self.ln_r_global((w_global.unsqueeze(-1) * h_r_sq).sum(0))

        if contrib_ext is not None:
            contrib_float = contrib_ext.float()
            logits_w = -load_vec.unsqueeze(0).expand(N_total, -1)
            logits_w = logits_w + torch.where(
                contrib_float > 0,
                torch.zeros(1, device=device),
                torch.full((1,), -1e9, device=device))
            w_contrib = F.softmax(logits_w, dim=-1)
            h_r_contrib_raw = torch.mm(w_contrib, h_r_sq)
            has_contrib = (contrib_float.sum(-1) > 0).float().unsqueeze(-1)
            h_r_contrib = self.ln_r_contrib(h_r_contrib_raw * has_contrib)
        else:
            h_r_contrib = self.ln_r_contrib(h_r_sq.mean(0).unsqueeze(0).expand(N_total, -1))

        h_r_global_exp = h_r_global.unsqueeze(0).expand(N_total, -1)
        q_local = self.W_local(torch.cat([h_t_sq, h_r_global_exp, h_r_contrib], dim=-1))

        if contrib_ext is not None:
            attn_bias = torch.zeros(N_total, M, dtype=torch.float32, device=device)
            attn_bias[~contrib_ext.bool()] = -1e9
        else:
            attn_bias = None

        q_ln = self.ln_q_ph0(q_local)
        # 对候选任务形成的查询再次聚焦于可贡献的智能体表示。
        h_r_kv = self.ln_kv_ph0(h_r_sq)
        q_enh, _ = self.focus_attn_ph0(
            q_ln.unsqueeze(0), h_r_kv.unsqueeze(0), h_r_kv.unsqueeze(0),
            attn_mask=attn_bias)
        q_enh_out = q_local + q_enh.squeeze(0)

        w_q = self.W_score_ph0(q_enh_out)
        raw = (h_t_sq * w_q).sum(-1) / math.sqrt(DM)
        u = CC * torch.tanh(raw)
        u_out = u.clone()
        u_out[0] = -1e9  # 站点不可选

        # 将原始 mask（针对任务）扩展到 N_total
        mask_ext = torch.zeros(N_total, dtype=torch.bool, device=device)
        mask_ext[1:] = mask
        u_out[~mask_ext] = -1e9

        # 返回时去掉站点节点（索引0）
        return u_out[1:], h_t_sq[1:]

    # ── Ph1/Ph2 单步推理 ─────────────────────────────────────────────────
    def forward_robot_select(
            self,
            task_feat_cur: torch.Tensor,
            robot_feat: torch.Tensor,
            mask: torch.Tensor,
            phase: int,
            task_ctx: Optional[torch.Tensor] = None,
            gap_vec: Optional[torch.Tensor] = None,
            coal_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回智能体动作分数和编码后的智能体表示。"""
        M = robot_feat.shape[0]
        device = robot_feat.device
        d = DM

        h_t = self.task_enc(self.task_emb_rob(task_feat_cur.unsqueeze(0).unsqueeze(0)))
        h_r = self.robot_enc(self.robot_emb_rob(robot_feat.unsqueeze(0)))
        h_t2 = self.cross_t2r(h_t, h_r)
        h_r2 = self.cross_r2t(h_r, h_t)
        h_r2_sq = h_r2.squeeze(0)
        h_t2_0 = h_t2[0, 0]

        e_ctx = self.ln_ctx(task_ctx.to(device).float() if task_ctx is not None
                            else torch.zeros(d, device=device))
        # 已选任务的保留表示与当前观测中的任务表示共同形成任务条件。
        h_task_fused = self.ln_task_fused(
            self.W_fuse(torch.cat([h_t2_0, e_ctx], dim=-1)))

        e_gap = (self.W_gap(gap_vec.to(device).float()) if gap_vec is not None
                 else torch.zeros(d, device=device))

        h_C = (coal_state.to(device).float() if coal_state is not None
               else torch.zeros(d, device=device))
        # 剩余需求与当前联盟表示进入所有候选智能体共享的查询。

        q_shared = self.W_global(torch.cat([h_task_fused, e_gap, h_C], dim=-1))

        q_shared_exp = q_shared.unsqueeze(0).expand(M, -1)
        q_ind = self.W_ind(torch.cat([h_r2_sq, q_shared_exp], dim=-1))

        robot_mask = mask[:M]
        attn_bias = torch.zeros(M, M, dtype=torch.float32, device=device)
        attn_bias[:, ~robot_mask] = -1e9

        q_ln = self.ln_q_ph1(q_ind)
        h_r_kv = self.ln_kv_ph1(h_r2_sq)
        q_enh, _ = self.focus_attn_ph1(
            q_ln.unsqueeze(0), h_r_kv.unsqueeze(0), h_r_kv.unsqueeze(0),
            attn_mask=attn_bias)
        q_enh_out = q_ind + q_enh.squeeze(0)

        w_q = self.W_score_ph1(q_enh_out)
        raw_r = (h_r2_sq * w_q).sum(-1) / math.sqrt(DM)
        u_r = CC * torch.tanh(raw_r)

        if phase == PHASE_ROBOT_ACCEL:
            # 需求已满足时才加入终止动作，其分数与智能体动作一同归一化。
            stop_q = self.W_stop(torch.cat([self.stop_token, q_shared], dim=-1))
            stop_q_enh, _ = self.focus_attn_ph1(
                self.ln_stop(stop_q).unsqueeze(0).unsqueeze(0),
                h_r_kv.unsqueeze(0), h_r_kv.unsqueeze(0))
            stop_q_out = stop_q + stop_q_enh[0, 0]
            w_q_stop = self.W_score_ph1(stop_q_out)
            u_stop = CC * torch.tanh(torch.dot(self.stop_token, w_q_stop) / math.sqrt(DM))
            u_full = torch.cat([u_r, u_stop.unsqueeze(0)], dim=0)
        else:
            u_full = u_r

        u_out = u_full.clone()
        u_out[~mask] = -1e9
        return u_out, h_r2_sq
