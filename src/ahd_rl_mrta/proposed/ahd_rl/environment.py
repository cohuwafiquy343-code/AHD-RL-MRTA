"""生成调度实例，并按 AHD-RL 的任务与智能体动作构造调度。"""

import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import scipy.spatial.distance

from .config import (
    D_CAP, D_MAP, EPS, GROUPS, INF, INSTANCE_TYPES, PHASE_ROBOT_ACCEL,
    PHASE_ROBOT_SATISFY, PHASE_TASK_SELECT, ROBOT_FEAT_DIM,
    ROBOT_TASK_FEAT_DIM, ROBOT_TYPES, TASK_SELECT_FEAT_DIM,
)


def generate_instance(group_id: int, seed: int = None, all_static: bool = False,
                      task_req_mode: str = 'default') -> dict:
    """按组别和随机种子生成智能体、任务及能力需求。

    task_req_mode:
      'default'   : 50% 单智能体、25% 简单协作、25% 困难协作
      'easy_coop' : 50% 单智能体、50% 简单协作
    """
    M, N = GROUPS[group_id]
    n_types = ROBOT_TYPES[group_id]
    rng = np.random.default_rng(seed)

    depot = rng.uniform(0.0, 100.0, 2)

    type_caps = []
    for _ in range(n_types):
        while True:
            cap = rng.integers(0, 4, size=D_CAP)
            if cap.sum() > 0:
                type_caps.append(cap.astype(np.int32))
                break

    type_speeds = [round(float(rng.uniform(10, 20)), 1) for _ in range(n_types)]

    robots_cap = np.zeros((M, D_CAP), dtype=np.int32)
    robot_speeds = np.zeros(M, dtype=np.float64)
    robot_type_ids = np.zeros(M, dtype=np.int32)

    base_per_type = M // n_types
    extra_types = M % n_types
    robots_per_type = [base_per_type + (1 if t < extra_types else 0)
                       for t in range(n_types)]
    j = 0
    for t_idx, count in enumerate(robots_per_type):
        for _ in range(count):
            robots_cap[j] = type_caps[t_idx]
            robot_speeds[j] = type_speeds[t_idx]
            robot_type_ids[j] = t_idx
            j += 1

    task_pos = rng.uniform(0.0, 100.0, (N, 2))

    N1 = N // 2
    N2 = math.ceil(N / 4)
    task_req = np.zeros((N, D_CAP), dtype=np.int32)

    cap_max = robots_cap.max(axis=0)
    cap_sum_all = robots_cap.sum(axis=0)

    # N1 单机器人可完成任务（所有模式共用）
    base_per = N1 // M
    extra_r = N1 % M
    idx = 0
    for j in range(M):
        cnt = base_per + (1 if j < extra_r else 0)
        for _ in range(cnt):
            while True:
                req = np.array(
                    [rng.integers(0, int(robots_cap[j, d]) + 1) for d in range(D_CAP)],
                    dtype=np.int32,
                )
                if req.sum() > 0:
                    task_req[idx] = req
                    break
            idx += 1

    if task_req_mode == 'easy_coop':
        # N - N1 简单协作任务：各维需求 ≤ 各机器人类型能力最大值
        for i in range(N1, N):
            while True:
                req = np.array(
                    [rng.integers(0, int(cap_max[d]) + 1) for d in range(D_CAP)],
                    dtype=np.int32,
                )
                if req.sum() > 0:
                    task_req[i] = req
                    break
    else:  # 'default'：N2 简单协作 + 剩余困难协作
        for i in range(N1, N1 + N2):
            while True:
                req = np.array(
                    [rng.integers(0, int(cap_max[d]) + 1) for d in range(D_CAP)],
                    dtype=np.int32,
                )
                if req.sum() > 0:
                    task_req[i] = req
                    break
        for i in range(N1 + N2, N):
            while True:
                req = np.array(
                    [rng.integers(0, int(cap_sum_all[d]) + 1) for d in range(D_CAP)],
                    dtype=np.int32,
                )
                if req.sum() > 0:
                    task_req[i] = req
                    break

    task_duration = rng.integers(15, 31, N).astype(np.float64)

    if all_static:
        dyn_set: Set[int] = set()
        static_list = list(range(N))
        t_arrival = np.zeros(N, dtype=np.float64)
    else:
        n_dyn = round(N * 0.3)
        perm = rng.permutation(N)
        dyn_set = set(int(x) for x in perm[:n_dyn])
        static_list = [i for i in range(N) if i not in dyn_set]
        T_static_sum = float(task_duration[static_list].sum())
        t_arrival = np.zeros(N, dtype=np.float64)
        for i in dyn_set:
            t_arrival[i] = rng.uniform(T_static_sum / (10.0 * M), T_static_sum / M)

    all_pos = np.vstack([depot.reshape(1, 2), task_pos])
    dist_mat = scipy.spatial.distance.cdist(all_pos, all_pos)

    task_dists = dist_mat[1:, 1:]
    if N > 1:
        D_bar = float(
            (task_dists.sum() - np.trace(task_dists)) / (N * (N - 1)))
    else:
        D_bar = float(dist_mat.mean())
    v_bar = float(robot_speeds.mean())
    T_ref = float(N) / float(M) * D_bar / max(v_bar, EPS)
    T_step = D_bar / max(v_bar, EPS)

    v_max = float(robot_speeds.max())
    c_max = robots_cap.max(axis=0).astype(np.float64)
    c_max = np.where(c_max == 0, 1.0, c_max)
    r_max = task_req.max(axis=0).astype(np.float64)
    r_max = np.where(r_max == 0, 1.0, r_max)
    D_max = float(task_duration.max())

    inst = dict(
        group_id=group_id, M=M, N=N,
        n_types=n_types,
        task_pos=task_pos,
        robots_cap=robots_cap,
        robot_speeds=robot_speeds,
        robot_type_ids=robot_type_ids,
        task_req=task_req,
        task_duration=task_duration,
        dyn_set=dyn_set,
        static_list=static_list,
        t_arrival=t_arrival,
        dist_mat=dist_mat,
        T_ref=T_ref, T_step=T_step,
        v_max=v_max, c_max=c_max, r_max=r_max, D_max=D_max,
        depot=depot,
        all_pos=all_pos.astype(np.float32),
        instance_type=('type1_easy_coop' if all_static and task_req_mode == 'easy_coop'
                       else 'type2_hard_coop' if all_static
                       else 'type3_dynamic_arrival'),
    )
    _build_instance_caches(inst)
    return inst


def generate_instance_by_type(group_id: int, type_key: str, seed: int) -> dict:
    """按场景编号生成一个可直接传入求解器的实例。"""
    spec = INSTANCE_TYPES[str(type_key)]
    base = generate_instance(
        group_id,
        seed=seed,
        all_static=bool(spec['all_static']),
        task_req_mode=str(spec['task_req_mode']),
    )
    base['instance_type'] = spec['name']
    return base


def has_contribution(cap_j: np.ndarray, req_i: np.ndarray) -> bool:
    """判断智能体是否具有当前任务需要的至少一项能力。"""
    return bool(np.any((cap_j > 0) & (req_i > 0)))


def coalition_exec_time(
    coal_set: Set[int],
    robots_cap: np.ndarray,
    task_req_i: np.ndarray,
    task_dur_i: float,
) -> float:
    """按现有仿真规则计算给定联盟的任务执行时间。"""
    req_sum = int(task_req_i.sum())
    if req_sum == 0 or not coal_set:
        return float(task_dur_i)
    contributing = [j for j in coal_set if has_contribution(robots_cap[j], task_req_i)]
    if not contributing:
        return float(task_dur_i)
    cap_sum = int(sum(robots_cap[j].sum() for j in contributing))
    excess = max(0, cap_sum - req_sum)
    sr = min(excess / req_sum, 0.75)
    return float(task_dur_i) * (1.0 - sr)


def coalition_satisfies(
    coal_set: Set[int],
    robots_cap: np.ndarray,
    task_req_i: np.ndarray,
) -> bool:
    """逐项检查联盟能力是否满足任务需求。"""
    if not coal_set:
        return False
    cap = np.zeros(D_CAP, dtype=np.int64)
    for j in coal_set:
        cap += robots_cap[j]
    return bool(np.all(cap >= task_req_i))


# ─────────────────────────────────────────────────────────────
# 2. INSTANCE GENERATION
# ─────────────────────────────────────────────────────────────

def _build_instance_caches(inst: dict) -> None:
    """预计算实例内复用字段（幂等）。

    缓存字段：
      all_pos             (N+1, 2) float32  行0=depot，行i+1=任务i（原始坐标）
      dist_mat            (N+1, N+1) float64 欧氏距离矩阵（旧实例缺失时自动重建）
      robot_task_contrib  (M, N)   bool
    """
    if 'all_pos' not in inst:
        depot = inst['depot']
        all_pos = np.vstack([depot.reshape(1, 2), inst['task_pos']])
        inst['all_pos'] = all_pos.astype(np.float32)

    if 'dist_mat' not in inst:
        all_pos_f = inst['all_pos'].astype(np.float64)
        inst['dist_mat'] = scipy.spatial.distance.cdist(all_pos_f, all_pos_f)

    if 'robot_task_contrib' not in inst:
        caps = inst['robots_cap']
        reqs = inst['task_req']
        inst['robot_task_contrib'] = (
            ((caps[:, None, :] > 0) & (reqs[None, :, :] > 0)).any(axis=2))

# ─────────────────────────────────────────────────────────────
# 2b. SCHEDULE CONSTRUCTION
# ─────────────────────────────────────────────────────────────

class ConstructionEnv:
    """维护已形成的智能体任务序列和当前正在确定的任务联盟。"""

    def __init__(self, inst: dict):
        _build_instance_caches(inst)
        self.inst = inst
        self.M = inst['M']
        self.N = inst['N']
        self._init_speed_het()

    # ── 速度异质性预计算 ────────────────────────────────────────────────
    def _init_speed_het(self) -> None:
        """预计算团队速度差异及各智能体的相对慢速指标。"""
        speeds = self.inst['robot_speeds']
        unique_v = sorted(set(float(v) for v in speeds))
        K = len(unique_v)
        if K >= 2:
            gaps = [unique_v[k + 1] - unique_v[k] for k in range(K - 1)]
            ki = int(np.argmax(gaps))
            v_thresh = unique_v[ki]
            v_top = unique_v[-1]
            self.gamma_het = float(gaps[ki]) / max(v_top, EPS)
            self.phi_slow = np.array(
                [1.0 if float(v) <= v_thresh else 0.0 for v in speeds],
                dtype=np.float32)
        else:
            self.gamma_het = 0.0
            self.phi_slow = np.zeros(self.M, dtype=np.float32)

    # ── 复位 ────────────────────────────────────────────────────────────
    def reset(self) -> None:
        """初始化智能体可用时间、任务状态与当前决策阶段。"""
        inst = self.inst
        M, N = self.M, self.N

        self.t_free = np.zeros(M, dtype=np.float64)
        self.p_last_idx = np.zeros(M, dtype=np.int32)
        self.robot_cum_travel = np.zeros(M, dtype=np.float64)
        self.robot_cum_wait = np.zeros(M, dtype=np.float64)

        self.task_assigned = np.zeros(N, dtype=bool)
        self.task_complete_time = np.full(N, INF)
        self.task_sat_time = np.full(N, INF)
        self.task_coalition = [set() for _ in range(N)]
        self.robot_arrive_time = [{} for _ in range(M)]

        self.robot_task_seq: List[List[int]] = [[] for _ in range(M)]

        self.num_done = 0
        self.done = False
        self.makespan = 0.0

        self.dynamic_unlocked = np.zeros(N, dtype=bool)
        for i in inst['static_list']:
            self.dynamic_unlocked[i] = True

        self.remaining_tasks: Set[int] = set(inst['static_list'])

        self.phase = PHASE_TASK_SELECT
        self.current_task = -1
        self.current_coalition: Set[int] = set()
        self.current_coalition_cap = np.zeros(D_CAP, dtype=np.int64)
        self.current_task_complete_time = INF
        self.coal_late_arr: float = 0.0


        self._unlock_dynamic_tasks()
        if not self.remaining_tasks:
            self._finalize()

    # ── 动态任务解锁 ────────────────────────────────────────────────────
    def _unlock_dynamic_tasks(self) -> None:
        """根据团队当前可用时间开放达到发布时间的动态任务。"""
        inst = self.inst
        t_max = float(self.t_free.max()) if self.M > 0 else 0.0
        for i in inst['dyn_set']:
            if self.dynamic_unlocked[i] or self.task_assigned[i]:
                continue
            if t_max >= float(inst['t_arrival'][i]):
                self.dynamic_unlocked[i] = True
                self.remaining_tasks.add(i)

    # ── 到达时间计算 ────────────────────────────────────────────────────
    def _t_arr(self, j: int, i: int) -> float:
        """机器人 j 到任务 i 的预计到达时刻。"""
        inst = self.inst
        p_from = int(self.p_last_idx[j])
        dist = float(inst['dist_mat'][p_from][i + 1])
        v_j = float(inst['robot_speeds'][j])
        t_arrive = float(self.t_free[j]) + dist / max(v_j, EPS)
        if i in inst['dyn_set']:
            t_arrive = max(t_arrive, float(inst['t_arrival'][i]))
        return t_arrive

    # ── 联盟完成时间计算 ─────────────────────────────────────────────────
    def _compute_coalition_times(
        self,
        i: int,
        coalition: Set[int],
        arr: Dict[int, float],
    ) -> Tuple[float, float]:
        """按成员到达顺序累计执行进度，返回需求满足与完成时刻。"""
        inst = self.inst
        sorted_m = sorted(coalition, key=lambda j: arr[j])
        cum_cap = np.zeros(D_CAP, dtype=np.int64)
        prefix_end = -1
        for k, j in enumerate(sorted_m):
            cum_cap += inst['robots_cap'][j]
            if np.all(cum_cap >= inst['task_req'][i]):
                prefix_end = k
                break
        if prefix_end < 0:
            return INF, INF

        init_coal = set(sorted_m[:prefix_end + 1])
        t_sat = arr[sorted_m[prefix_end]]
        et = coalition_exec_time(
            init_coal, inst['robots_cap'], inst['task_req'][i], inst['task_duration'][i])
        t_complete = t_sat + et

        # 晚到成员的进度追踪
        progress = 0.0
        cur_et = et
        last_seg = t_sat
        cur_coal = set(init_coal)
        for j_late in sorted_m[prefix_end + 1:]:
            t_late = arr[j_late]
            if t_late >= t_complete:
                break
            delta = t_late - last_seg
            if cur_et > EPS:
                progress += delta / cur_et
            progress = min(progress, 1.0 - EPS)
            cur_coal.add(j_late)
            new_et = coalition_exec_time(
                cur_coal, inst['robots_cap'], inst['task_req'][i], inst['task_duration'][i])
            t_complete = t_late + (1.0 - progress) * new_et
            cur_et = new_et
            last_seg = t_late

        return t_sat, t_complete

    def _recompute_current_times(self) -> None:
        """根据当前联盟重新计算任务 i 的完成时刻并更新 current_task_complete_time。"""
        i = self.current_task
        arr = {}
        for j in self.current_coalition:
            arr[j] = float(self.robot_arrive_time[j].get(i, self._t_arr(j, i)))
        t_sat, t_complete = self._compute_coalition_times(i, self.current_coalition, arr)
        self.task_sat_time[i] = t_sat
        self.current_task_complete_time = t_complete

    # ── 加速阶段：计算加入机器人 j 后的新完成时刻 ───────────────────────
    def _completion_time_with_agent(self, i: int, j: int, t_arr_j: float) -> float:
        """估计候选智能体加入后当前任务的完成时刻。"""
        new_coal = self.current_coalition | {j}
        arr = {}
        for jj in new_coal:
            if jj == j:
                arr[jj] = t_arr_j
            else:
                arr[jj] = float(self.robot_arrive_time[jj].get(i, self._t_arr(jj, i)))
        _, t_c = self._compute_coalition_times(i, new_coal, arr)
        return t_c

    def _all_t_arr(self, task_id: int) -> np.ndarray:
        """计算全部智能体到目标任务的预计到达时刻，返回长度为 M 的向量。"""
        inst = self.inst
        dist = inst['dist_mat'][self.p_last_idx, task_id + 1]  # (M,)，fancy indexing
        t_arr = self.t_free + dist / np.maximum(inst['robot_speeds'], EPS)  # (M,)
        if task_id in inst['dyn_set']:
            t_arr = np.maximum(t_arr, float(inst['t_arrival'][task_id]))
        return t_arr.astype(np.float64)

    # ── 特征：速度异质性相关 ─────────────────────────────────────────────
    def _estimate_wait_and_var(
            self, i: int, arr_times: Optional[np.ndarray] = None
    ) -> Tuple[float, float, float, float, int]:
        """估算任务 i 的等待代价等统计量。

        返回 (w_est, sigma_i, w_ref, t_late_ref, n_ref)：
          w_est     : 各成员相对 t_sat 的等待之和（原始值）
          sigma_i   : 参考联盟到达时刻标准差（sqrt(Var)，原始值）
          w_ref     : 各成员相对首到达机器人的额外等待之和（原始值）
          t_late_ref: 参考联盟最晚到达成员的到达时刻
          n_ref     : 参考联盟规模（最小满足联盟成员数）
        """
        inst = self.inst
        req = inst['task_req'][i]
        caps = inst['robots_cap']

        if arr_times is None:
            arr_times = self._all_t_arr(i)

        capable_mask = ((caps > 0) & (req > 0)).any(axis=1)
        cap_idx = np.where(capable_mask)[0]
        if len(cap_idx) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0

        arr_cap = arr_times[cap_idx]
        order = np.argsort(arr_cap)
        cap_idx_sorted = cap_idx[order]
        arr_sorted = arr_cap[order]

        cum_cap = np.cumsum(caps[cap_idx_sorted], axis=0)
        sat_mask = np.all(cum_cap >= req, axis=1)
        if not sat_mask.any():
            return 0.0, 0.0, 0.0, 0.0, 0

        k = int(np.argmax(sat_mask))
        t_sat = float(arr_sorted[k])
        w_est = float(np.sum(np.maximum(0.0, t_sat - arr_sorted[:k])))
        var_arr = float(np.var(arr_sorted[:k + 1])) if k >= 1 else 0.0
        sigma_i = math.sqrt(var_arr)
        w_ref = float(np.sum(arr_sorted[:k + 1] - arr_sorted[0]))
        n_ref = k + 1
        t_late_ref = t_sat
        return w_est, sigma_i, w_ref, t_late_ref, n_ref

    # ── 任务特征（任务选择阶段）──────────────────────────────────────────
    def _task_features_all(self, active_ids: List[int]) -> np.ndarray:
        """构造任务动作观测；第 6、7、12、13、14 列为 NRT 移除的信息。"""
        inst = self.inst
        N_act = len(active_ids)
        feat = np.zeros((N_act, TASK_SELECT_FEAT_DIM), dtype=np.float32)
        pr = self.get_progress_ratio()
        active_ids_arr = np.array(active_ids, dtype=np.int32)

        t_max = float(self.t_free.max()) if self.M > 0 else 0.0

        dist_sub = inst['dist_mat'][
            self.p_last_idx[:, None],
            active_ids_arr[None, :] + 1
        ]
        t_arr_mat = (
                self.t_free[:, None]
                + dist_sub / np.maximum(inst['robot_speeds'][:, None], EPS)
        ).astype(np.float64)

        for idx, i in enumerate(active_ids):
            if i in inst['dyn_set']:
                t_arr_mat[:, idx] = np.maximum(
                    t_arr_mat[:, idx], float(inst['t_arrival'][i]))

        r_max = np.maximum(inst['r_max'], 1e-8).astype(np.float32)

        for idx, i in enumerate(active_ids):
            req = inst['task_req'][i].astype(np.float32)
            feat[idx, 0:5] = req / r_max
            feat[idx, 5] = float(inst['task_duration'][i]) / float(max(inst['D_max'], 1.0))
            w_est, sigma_i, w_ref, t_late_ref, n_ref = self._estimate_wait_and_var(
                i, arr_times=t_arr_mat[:, idx])
            feat[idx, 6] = w_est
            feat[idx, 7] = sigma_i
            feat[idx, 8] = pr
            feat[idx, 9] = 0.0
            feat[idx, 10] = 0.0
            feat[idx, 11] = 0.0
            feat[idx, 12] = t_late_ref - t_max
            feat[idx, 13] = float(n_ref) / max(float(self.M), 1.0)
            feat[idx, 14] = w_ref
            feat[idx, 15] = float(inst['task_pos'][i, 0]) / 100.0
            feat[idx, 16] = float(inst['task_pos'][i, 1]) / 100.0
            feat[idx, 17] = float(inst['dist_mat'][0][i + 1]) / D_MAP

        for k in [6, 7, 12, 14]:
            col = feat[:, k]
            col_min = col.min()
            col_max = col.max()
            feat[:, k] = (col - col_min) / (col_max - col_min + EPS)

        return feat

    # ── 当前任务特征（机器人选择阶段）──────────────────────────────────
    def _task_feature_current(self) -> np.ndarray:
        """构造当前任务观测；第 11、12、17、18、19 列为参考联盟信息。"""
        inst = self.inst
        i = self.current_task
        feat = np.zeros((1, ROBOT_TASK_FEAT_DIM), dtype=np.float32)
        req = inst['task_req'][i].astype(np.float32)
        r_max = np.maximum(inst['r_max'], 1e-8).astype(np.float32)
        gap = np.maximum(0, req - self.current_coalition_cap.astype(np.float32))

        t_max = float(self.t_free.max()) if self.M > 0 else 0.0

        feat[0, 0:5] = req / r_max
        feat[0, 5:10] = gap / r_max
        feat[0, 10] = float(inst['task_duration'][i]) / float(max(inst['D_max'], 1.0))
        w_est, sigma_i, w_ref, t_late_ref, n_ref = self._estimate_wait_and_var(i)
        feat[0, 11] = w_est / float(max(inst['T_step'], 1.0))
        feat[0, 12] = sigma_i / float(max(inst['T_step'], 1.0))
        feat[0, 13] = self.get_progress_ratio()
        feat[0, 14] = 1.0
        if self.current_coalition:
            sr = self._coalition_surplus_ratio(self.current_coalition, i)
            feat[0, 15] = float(sr)
            T_c = self.current_task_complete_time
            best_delta = 0.0
            for j in range(self.M):
                if j in self.current_coalition:
                    continue
                t_arr_j = self._t_arr(j, i)
                if t_arr_j < T_c:
                    tc_p = self._completion_time_with_agent(i, j, t_arr_j)
                    delta = (T_c - tc_p) / max(T_c - self.task_sat_time[i], EPS)
                    best_delta = max(best_delta, delta)
            feat[0, 16] = float(min(best_delta, 1.0))
        feat[0, 17] = (t_late_ref - t_max) / float(max(inst['T_ref'], 1.0))
        feat[0, 18] = float(n_ref) / max(float(self.M), 1.0)
        feat[0, 19] = w_ref / float(max(inst['T_step'], 1.0))
        feat[0, 20] = float(inst['task_pos'][i, 0]) / 100.0
        feat[0, 21] = float(inst['task_pos'][i, 1]) / 100.0
        feat[0, 22] = float(inst['dist_mat'][0][i + 1]) / D_MAP
        return feat[0]

    def _coalition_surplus_ratio(self, coal_set: Set[int], task_id: int) -> float:
        """计算现有仿真规则中的联盟超额能力比例，上限为 0.75。"""
        req_sum = int(self.inst['task_req'][task_id].sum())
        if req_sum == 0:
            return 0.75
        contrib = [j for j in coal_set
                   if has_contribution(self.inst['robots_cap'][j], self.inst['task_req'][task_id])]
        if not contrib:
            return 0.0
        cap_sum = int(sum(self.inst['robots_cap'][j].sum() for j in contrib))
        return min(max(0, cap_sum - req_sum) / req_sum, 0.75)

    # ── 机器人特征（通用）────────────────────────────────────────────────
    def _robot_features(self, i_target: int) -> np.ndarray:
        """构造智能体特征；第 15 列指示智能体是否已进入当前联盟。"""
        inst = self.inst
        M = self.M
        feat = np.zeros((M, ROBOT_FEAT_DIM), dtype=np.float32)

        t_max = float(self.t_free.max()) if M > 0 else 0.0
        t_min = float(self.t_free.min()) if M > 0 else 0.0
        dt = t_max - t_min + EPS

        c_max = inst['robots_cap'].max(axis=0).astype(np.float32)
        feat[:, 0:5] = inst['robots_cap'].astype(np.float32) / np.maximum(c_max, 1.0)
        feat[:, 5] = (inst['robot_speeds'] / 20.0).astype(np.float32)
        feat[:, 6] = np.clip((self.t_free - t_min) / dt, 0.0, 1.0).astype(np.float32)
        feat[:, 10] = self.phi_slow
        feat[:, 11] = (
                (inst['v_max'] - inst['robot_speeds']) / max(float(inst['v_max']), EPS)
        ).astype(np.float32)

        t_free_pos = self.t_free > EPS
        t_safe = np.where(t_free_pos, self.t_free, 1.0)
        feat[:, 13] = np.where(
            t_free_pos, np.clip(self.robot_cum_travel / t_safe, 0.0, 1.0), 0.0
        ).astype(np.float32)
        feat[:, 14] = np.where(
            t_free_pos, np.clip(self.robot_cum_wait / t_safe, 0.0, 1.0), 0.0
        ).astype(np.float32)
        feat[:, 16] = float(self.gamma_het)
        feat[:, 17] = np.clip(
            self.t_free / max(t_max, EPS), 0.0, 1.0
        ).astype(np.float32)

        # 当前位置坐标及到站点距离
        pos_x = (inst['all_pos'][self.p_last_idx, 0] / 100.0).astype(np.float32)
        pos_y = (inst['all_pos'][self.p_last_idx, 1] / 100.0).astype(np.float32)
        depot_dists = inst['dist_mat'][self.p_last_idx, 0].astype(np.float32)

        if i_target >= 0 and self.current_coalition:
            target_pos = inst['all_pos'][i_target + 1]
            target_dep_dist = float(inst['dist_mat'][i_target + 1, 0])
            for j in self.current_coalition:
                pos_x[j] = float(target_pos[0]) / 100.0
                pos_y[j] = float(target_pos[1]) / 100.0
                depot_dists[j] = target_dep_dist

        feat[:, 18] = pos_x
        feat[:, 19] = pos_y
        feat[:, 20] = (depot_dists / D_MAP).astype(np.float32)

        if i_target >= 0:
            # 当前任务已确定时，补充候选智能体的到达、能力贡献和完工时间影响。
            arr_times = self._all_t_arr(i_target).astype(np.float32)
            req_i = inst['task_req'][i_target]
            caps = inst['robots_cap']

            in_coal = np.zeros(M, dtype=bool)
            for j in self.current_coalition:
                in_coal[j] = True
            capable_mask = ((caps > 0) & (req_i > 0)).any(axis=1)
            cand_mask = ~in_coal & capable_mask
            cand_idx = np.where(cand_mask)[0]

            if self.phase == PHASE_ROBOT_SATISFY and len(cand_idx) > 1:
                arr_cand = arr_times[cand_idx]
                arr_min = float(arr_cand.min())
                arr_max = float(arr_cand.max())
                feat[cand_idx, 7] = (arr_cand - arr_min) / (arr_max - arr_min + EPS)
                diff_cand = arr_cand - float(self.coal_late_arr)
                diff_min = float(diff_cand.min())
                diff_max = float(diff_cand.max())
                feat[cand_idx, 8] = (diff_cand - diff_min) / (diff_max - diff_min + EPS)

            gap = np.maximum(0, req_i.astype(np.int64) - self.current_coalition_cap)
            gap_sum = float(gap.sum())
            if gap_sum > 0:
                contrib = np.minimum(caps, gap).sum(axis=1).astype(np.float32)
                feat[:, 9] = contrib / (gap_sum + EPS)

            T_c = self.current_task_complete_time
            d_i = float(inst['task_duration'][i_target])
            if T_c < INF and d_i > EPS:
                for j in cand_idx:
                    t_arr_j = float(arr_times[j])
                    if t_arr_j < T_c:
                        tc_p = self._completion_time_with_agent(i_target, j, t_arr_j)
                        delta = max(0.0, T_c - tc_p)
                        feat[j, 12] = float(min(delta / d_i, 10.0))

            coal_flag = np.zeros(M, dtype=np.float32)
            for j in self.current_coalition:
                if j < M:
                    coal_flag[j] = 1.0
            feat[:, 15] = coal_flag

        return feat

    # ── 掩码计算 ─────────────────────────────────────────────────────────
    def _task_mask(self, active_ids: List[int]) -> np.ndarray:
        """任务选择掩码：存在可行联盟则可选。"""
        inst = self.inst
        if not active_ids:
            return np.zeros(0, dtype=bool)
        total_cap = inst['robots_cap'].sum(axis=0)
        reqs = inst['task_req'][np.asarray(active_ids, dtype=np.int32)]
        return np.all(reqs <= total_cap, axis=1)

    def _robot_satisfy_mask(self) -> np.ndarray:
        """需求满足阶段机器人掩码：对当前缺口有正贡献。"""
        inst = self.inst
        i = self.current_task
        gap = np.maximum(
            0, inst['task_req'][i].astype(np.int64) - self.current_coalition_cap
        )  # (D_CAP,)
        if gap.sum() == 0:
            return np.zeros(self.M, dtype=bool)

        # 向量化：contrib[j] = sum_d min(cap_j[d], gap[d])
        contrib = np.minimum(inst['robots_cap'], gap).sum(axis=1)  # (M,)

        in_coal = np.zeros(self.M, dtype=bool)
        for j in self.current_coalition:
            in_coal[j] = True
        return (contrib > 0) & ~in_coal

    def _robot_accel_mask(self) -> np.ndarray:
        """加速阶段机器人掩码（不含停止动作，索引 M 始终有效）。"""
        inst = self.inst
        i = self.current_task
        T_c = self.current_task_complete_time

        req_i = inst['task_req'][i]
        caps = inst['robots_cap']  # (M, D_CAP)

        # 向量化：有贡献的机器人
        contrib_mask = ((caps > 0) & (req_i > 0)).any(axis=1)  # (M,)

        in_coal = np.zeros(self.M, dtype=bool)
        for j in self.current_coalition:
            in_coal[j] = True

        # 向量化：计算全体到达时间
        arr_times = self._all_t_arr(i)  # (M,)

        # 候选集：不在联盟、有贡献、到达时刻早于 T_c
        candidates = np.where(~in_coal & contrib_mask & (arr_times < T_c))[0]

        mask = np.zeros(self.M, dtype=bool)
        for j in candidates:
            p_from = int(self.p_last_idx[j])
            dist = float(inst['dist_mat'][p_from][i + 1])
            v_j = float(inst['robot_speeds'][j])
            tau_j = dist / max(v_j, EPS)
            tc_p = self._completion_time_with_agent(i, j, float(arr_times[j]))
            delta_exec = T_c - tc_p
            if tau_j < delta_exec:
                mask[j] = True
        return mask

    # ── 统一接口 ─────────────────────────────────────────────────────────
    def get_mask_and_features(self):
        """根据当前阶段返回掩码和特征。

        返回值：
          PHASE_TASK_SELECT  → (active_ids, task_feat, robot_feat, mask)
          PHASE_ROBOT_SATISFY → (task_feat_cur, robot_feat, mask_robot)
          PHASE_ROBOT_ACCEL   → (task_feat_cur, robot_feat, mask_robot_with_stop)
              mask_robot_with_stop 长度 M+1，最后一位为停止动作（始终 True）
        """
        if self.phase == PHASE_TASK_SELECT:
            active_ids = sorted(self.remaining_tasks)
            tf = self._task_features_all(active_ids)
            rf = self._robot_features(-1)
            mask = self._task_mask(active_ids)
            return active_ids, tf, rf, mask

        # 机器人选择阶段（满足 or 加速）
        tf_cur = self._task_feature_current()
        rf = self._robot_features(self.current_task)

        if self.phase == PHASE_ROBOT_SATISFY:
            mask = self._robot_satisfy_mask()
            return tf_cur, rf, mask

        # PHASE_ROBOT_ACCEL：掩码长度 M+1（最后位 = 停止动作）
        mask_base = self._robot_accel_mask()
        mask_ext = np.append(mask_base, True)  # 停止动作始终可选
        return tf_cur, rf, mask_ext

    # ── 动作执行 ─────────────────────────────────────────────────────────
    def step(self, action: int) -> bool:
        """按当前阶段解释动作并推进构造环境。"""
        if self.phase == PHASE_TASK_SELECT:
            self._do_select_task(action)
        elif self.phase == PHASE_ROBOT_SATISFY:
            self._do_select_robot_satisfy(action)
        elif self.phase == PHASE_ROBOT_ACCEL:
            self._do_select_robot_accel(action)
        return self.done

    def _do_select_task(self, task_id: int, task_ctx: Optional[np.ndarray] = None) -> None:
        """记录当前任务，并初始化其待构造的联盟。"""
        self.current_task = task_id
        self.current_coalition = set()
        self.current_coalition_cap = np.zeros(D_CAP, dtype=np.int64)
        self.current_task_complete_time = INF
        self.task_sat_time[task_id] = INF
        self.current_task_ctx = task_ctx

        contrib_js = [
            j for j in range(self.M)
            if has_contribution(self.inst['robots_cap'][j], self.inst['task_req'][task_id])
        ]
        if contrib_js:
            arr_times = self._all_t_arr(task_id)
            self.coal_late_arr = float(min(arr_times[j] for j in contrib_js))
        else:
            self.coal_late_arr = 0.0

        self.phase = PHASE_ROBOT_SATISFY

    def _do_select_robot_satisfy(self, robot_id: int) -> None:
        """把智能体加入联盟；能力满足后转入效率决策阶段。"""
        inst = self.inst
        i = self.current_task
        j = robot_id

        t_arr_j = self._t_arr(j, i)
        p_from = int(self.p_last_idx[j])
        dist = float(inst['dist_mat'][p_from][i + 1])
        v_j = float(inst['robot_speeds'][j])
        self.robot_cum_travel[j] += dist / max(v_j, EPS)
        self.robot_arrive_time[j][i] = t_arr_j

        self.current_coalition.add(j)
        self.current_coalition_cap += inst['robots_cap'][j]
        self.coal_late_arr = max(self.coal_late_arr, t_arr_j)
        self.robot_task_seq[j].append(i)



        if coalition_satisfies(self.current_coalition, inst['robots_cap'], inst['task_req'][i]):
            self._recompute_current_times()
            self._exclude_nonparticipating_members(i)
            self.phase = PHASE_ROBOT_ACCEL

    def _do_select_robot_accel(self, action: int) -> None:
        """选择额外智能体，或通过终止动作确定当前联盟。"""
        inst = self.inst
        i = self.current_task
        M = self.M

        if action == M:
            self._commit_task(i)
            return

        j = action
        t_arr_j = self._t_arr(j, i)
        p_from = int(self.p_last_idx[j])
        dist = float(inst['dist_mat'][p_from][i + 1])
        v_j = float(inst['robot_speeds'][j])
        self.robot_cum_travel[j] += dist / max(v_j, EPS)
        self.robot_arrive_time[j][i] = t_arr_j

        self.current_coalition.add(j)
        self.current_coalition_cap += inst['robots_cap'][j]
        self.coal_late_arr = max(self.coal_late_arr, t_arr_j)
        self.robot_task_seq[j].append(i)
        self._recompute_current_times()
        self._exclude_nonparticipating_members(i)


        if not self._robot_accel_mask().any():
            self._commit_task(i)

    def _exclude_nonparticipating_members(self, i: int) -> Set[int]:
        """撤回任务完成后才会到达的智能体，保持实际联盟和路线记录一致。"""
        T_c = float(self.current_task_complete_time)
        excluded = {
            j for j in self.current_coalition
            if float(self.robot_arrive_time[j][i]) > T_c + EPS
        }
        if not excluded:
            return set()

        old_t_sat = float(self.task_sat_time[i])
        old_t_complete = T_c
        for j in excluded:
            if not self.robot_task_seq[j] or self.robot_task_seq[j][-1] != i:
                raise RuntimeError(
                    "cannot roll back nonparticipating robot {} for task {}".format(j, i)
                )
            self.robot_task_seq[j].pop()

            p_from = int(self.p_last_idx[j])
            dist = float(self.inst['dist_mat'][p_from][i + 1])
            travel_time = dist / max(float(self.inst['robot_speeds'][j]), EPS)
            remaining_travel = float(self.robot_cum_travel[j]) - travel_time
            if remaining_travel < -EPS:
                raise RuntimeError(
                    "negative cumulative travel after rolling back robot {}".format(j)
                )
            self.robot_cum_travel[j] = max(0.0, remaining_travel)
            self.robot_arrive_time[j].pop(i, None)

        self.current_coalition.difference_update(excluded)
        if not self.current_coalition:
            raise RuntimeError("task {} has no participating coalition".format(i))
        if not coalition_satisfies(
                self.current_coalition,
                self.inst['robots_cap'],
                self.inst['task_req'][i]):
            raise RuntimeError(
                "participating coalition no longer satisfies task {}".format(i)
            )
        self.current_coalition_cap = self.inst['robots_cap'][
            sorted(self.current_coalition)
        ].sum(axis=0, dtype=np.int64)
        self._recompute_current_times()
        self.coal_late_arr = max(
            float(self.robot_arrive_time[j][i]) for j in self.current_coalition
        )

        tolerance = 1e-7 * max(1.0, abs(old_t_complete))
        if (abs(float(self.task_sat_time[i]) - old_t_sat) > tolerance
                or abs(float(self.current_task_complete_time)
                       - old_t_complete) > tolerance):
            raise RuntimeError(
                "excluding post-completion arrivals changed task {} timing".format(i)
            )
        return excluded

    def _commit_task(self, i: int) -> None:
        """确定任务联盟，并更新参与智能体的后续可用状态。"""
        inst = self.inst
        self._exclude_nonparticipating_members(i)
        coal = set(self.current_coalition)
        T_c = float(self.current_task_complete_time)
        t_sat = float(self.task_sat_time[i])

        late_members = [
            j for j in coal
            if float(self.robot_arrive_time[j][i]) > T_c + EPS
        ]
        if late_members:
            raise RuntimeError(
                "task {} contains coalition members arriving after completion: {}".format(
                    i, late_members
                )
            )

        self.task_coalition[i] = set(coal)
        self.task_complete_time[i] = T_c
        self.task_assigned[i] = True

        for j in coal:
            t_arr_j = float(self.robot_arrive_time[j].get(i, 0.0))
            wt = max(0.0, t_sat - t_arr_j)
            self.robot_cum_wait[j] += wt
            self.t_free[j] = T_c
            self.p_last_idx[j] = i + 1

        self.remaining_tasks.discard(i)
        self.num_done += 1


        self.phase = PHASE_TASK_SELECT
        self.current_task = -1
        self.coal_late_arr = 0.0
        self._unlock_dynamic_tasks()
        if not self.remaining_tasks:
            self._finalize()

    def _finalize(self) -> None:
        """完成全部任务后计入智能体返回站点的时间。"""
        if self.done:
            return
        inst = self.inst
        for j in range(self.M):
            pi = int(self.p_last_idx[j])
            if pi != 0:
                travel = float(inst['dist_mat'][pi][0]) / float(inst['robot_speeds'][j])
                self.t_free[j] += travel
                self.p_last_idx[j] = 0
        self.makespan = float(self.t_free.max())
        self.done = True

    # ── 辅助 ─────────────────────────────────────────────────────────────
    def get_progress_ratio(self) -> float:
        """返回已确定联盟的任务数占当前已开放任务数的比例。"""
        n_unlocked = int(self.dynamic_unlocked.sum())
        return self.num_done / n_unlocked if n_unlocked > 0 else 0.0
