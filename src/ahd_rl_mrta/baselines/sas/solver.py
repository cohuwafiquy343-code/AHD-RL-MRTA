"""SAS 对比算法：任务转移、联盟形成和信息素更新。"""

import math
import random

import numpy as np

NS = 10
ALPHA = 1.0
BETA = 2.0
RHO = 0.1
P0 = 0.9

# ── Helper utilities ───────────────────────────────────────────────────────────

def _t_arrival(inst, task_idx):
    """读取任务发布时间；静态任务对应零时刻。"""
    arr = inst['t_arrival']
    if isinstance(arr, dict):
        return float(arr.get(task_idx, 0.0))
    return float(arr[task_idx])


def _node_coord(node_idx, inst):
    """返回节点坐标，其中 0 为站点、i+1 为任务 i。"""
    if node_idx == 0:
        return tuple(inst['depot'])
    return tuple(inst['task_pos'][node_idx - 1])


def _unlock_dynamic(t_free, available_tasks, completed, inst):
    """将已达到发布时间的动态任务加入候选集合。"""
    t_now = max(t_free)
    for i in inst['dyn_set']:
        if i not in completed and i not in available_tasks:
            if t_now >= _t_arrival(inst, i):
                available_tasks.add(i)


# ── Core calculation functions ─────────────────────────────────────────────────

def can_contribute(robot_cap, task_req):
    """判断智能体是否能提供至少一项任务所需能力。"""
    for d in range(len(robot_cap)):
        if robot_cap[d] > 0 and task_req[d] > 0:
            return True
    return False


def coalition_satisfies(caps_list, task_req):
    """逐项检查联盟能力是否满足任务需求。"""
    if not caps_list:
        return False
    n_dim = len(task_req)
    total = [0.0] * n_dim
    for cap in caps_list:
        for d in range(n_dim):
            total[d] += cap[d]
    for d in range(n_dim):
        if total[d] < task_req[d]:
            return False
    return True


def compute_exec_time(coalition_robot_ids, task_idx, inst):
    """根据当前联盟的超额能力计算任务执行时长。"""
    robots_cap = inst['robots_cap']
    task_req = inst['task_req'][task_idx]
    task_dur = inst['task_duration'][task_idx]
    cap_sum = 0.0
    for j in coalition_robot_ids:
        if can_contribute(robots_cap[j], task_req):
            cap_sum += sum(robots_cap[j])
    req_sum = sum(task_req)
    sr = 0.0 if req_sum == 0 else min(max(0.0, (cap_sum - req_sum) / req_sum), 0.75)
    return task_dur * (1.0 - sr)


def simulate_task_execution(task_idx, coalition_robot_ids, t_arrive_dict, inst):
    """按成员到达顺序计算任务开始、完成与实际参与联盟。"""
    robots_cap = inst['robots_cap']
    task_req = inst['task_req'][task_idx]

    # 按到达时间排序，同一时刻按智能体编号排序
    sorted_robots = sorted(coalition_robot_ids, key=lambda j: (t_arrive_dict[j], j))

    # 找到首次满足任务需求的成员前缀
    p = len(sorted_robots)
    for k in range(1, len(sorted_robots) + 1):
        if coalition_satisfies([robots_cap[j] for j in sorted_robots[:k]], task_req):
            p = k
            break

    T_sat = t_arrive_dict[sorted_robots[p - 1]]

    # 计算该前缀中先到达成员的等待时间
    wait_times = {}
    for k in range(p - 1):
        j = sorted_robots[k]
        wait_times[j] = T_sat - t_arrive_dict[j]

    # 后续成员到达时按时间段计算任务进度
    active = list(sorted_robots[:p])
    exec_rem = compute_exec_time(active, task_idx, inst)
    t_ref = T_sat
    T_complete = t_ref + exec_rem

    for k in range(p, len(sorted_robots)):
        j = sorted_robots[k]
        t_j = t_arrive_dict[j]
        if t_j >= T_complete or exec_rem < 1e-12:
            break
        frac_left = max(0.0, 1.0 - (t_j - t_ref) / exec_rem)
        active.append(j)
        exec_rem = frac_left * compute_exec_time(active, task_idx, inst)
        t_ref = t_j
        T_complete = t_ref + exec_rem

    # 最终联盟只保留任务完成前实际参与的成员
    valid_coalition = list(active)

    return T_sat, T_complete, valid_coalition, wait_times


# ── Greedy initialization ──────────────────────────────────────────────────────

def greedy_construct(inst):
    """依据智能体位置与任务距离构造初始可行调度。"""
    M = inst['M']
    N = inst['N']
    dist_mat = inst['dist_mat']
    robot_speeds = inst['robot_speeds']
    static_list = inst['static_list']
    dyn_set = inst['dyn_set']

    t_free = [0.0] * M
    pos_idx = [0] * M
    completed = set()
    available_tasks = set(static_list)
    robot_schedules = [[] for _ in range(M)]

    while len(completed) < N:
        _unlock_dynamic(t_free, available_tasks, completed, inst)
        candidates = available_tasks - completed

        if not candidates:
            pending = [_t_arrival(inst, i) for i in dyn_set
                       if i not in completed and i not in available_tasks]
            if not pending:
                break
            t_adv = min(pending)
            for j in range(M):
                t_free[j] = max(t_free[j], t_adv)
            continue

        # 根据智能体平均位置选择最近任务
        avg_x = sum(_node_coord(pos_idx[j], inst)[0] for j in range(M)) / M
        avg_y = sum(_node_coord(pos_idx[j], inst)[1] for j in range(M)) / M

        best_task, best_d = None, float('inf')
        for i in candidates:
            tx, ty = inst['task_pos'][i]
            d = math.sqrt((tx - avg_x) ** 2 + (ty - avg_y) ** 2)
            if d < best_d:
                best_d, best_task = d, i

        task_idx = best_task
        viable = [j for j in range(M)
                  if can_contribute(inst['robots_cap'][j], inst['task_req'][task_idx])]

        if not viable:
            completed.add(task_idx)
            available_tasks.discard(task_idx)
            continue

        t_arr = {}
        for j in viable:
            travel = dist_mat[pos_idx[j]][task_idx + 1] / robot_speeds[j]
            t_arr[j] = max(t_free[j] + travel, _t_arrival(inst, task_idx))

        sorted_viable = sorted(viable, key=lambda j: (t_arr[j], j))
        coalition = []
        for j in sorted_viable:
            coalition.append(j)
            if coalition_satisfies([inst['robots_cap'][k] for k in coalition],
                                   inst['task_req'][task_idx]):
                break

        t_arrive_dict = {j: t_arr[j] for j in coalition}
        T_sat, T_complete, valid_coalition, wait_times = simulate_task_execution(
            task_idx, coalition, t_arrive_dict, inst)

        for j in valid_coalition:
            robot_schedules[j].append(
                (task_idx, t_arrive_dict[j], T_sat, T_complete, wait_times.get(j, 0.0)))
            t_free[j] = T_complete
            pos_idx[j] = task_idx + 1

        completed.add(task_idx)
        available_tasks.discard(task_idx)

    t_final = [t_free[j] + dist_mat[pos_idx[j]][0] / robot_speeds[j]
               for j in range(M) if robot_schedules[j]]
    makespan = max(t_final) if t_final else 0.0
    return makespan, robot_schedules


# ── SAS algorithm functions ────────────────────────────────────────────────────

def build_eta(inst):
    """按节点间距离的倒数构造任务转移启发值。"""
    N = inst['N']
    dist_mat = inst['dist_mat']
    size = N + 1
    eta = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            d = dist_mat[i][j]
            eta[i][j] = 1.0 / d if d > 1e-9 else 1e6
    return eta


def assign_coalition(task_idx, inst, t_free, pos_idx):
    """按预计到达顺序补充智能体，直到满足任务能力需求。"""
    M = inst['M']
    dist_mat = inst['dist_mat']
    robot_speeds = inst['robot_speeds']

    viable = [j for j in range(M)
              if can_contribute(inst['robots_cap'][j], inst['task_req'][task_idx])]
    if not viable:
        return [], {}

    t_arr = {}
    for j in viable:
        travel = dist_mat[pos_idx[j]][task_idx + 1] / robot_speeds[j]
        t_arr[j] = max(t_free[j] + travel, _t_arrival(inst, task_idx))

    sorted_viable = sorted(viable, key=lambda j: (t_arr[j], j))
    coalition = []
    for j in sorted_viable:
        coalition.append(j)
        if coalition_satisfies([inst['robots_cap'][k] for k in coalition],
                               inst['task_req'][task_idx]):
            break

    return coalition, {j: t_arr[j] for j in coalition}


def sas_construct_solution(inst, tau, eta, alpha, beta, p0):
    """由一个蚂蚁依次选择任务并为每个任务确定联盟。"""
    M = inst['M']
    N = inst['N']
    dist_mat = inst['dist_mat']
    robot_speeds = inst['robot_speeds']
    static_list = inst['static_list']
    dyn_set = inst['dyn_set']

    t_free = [0.0] * M
    pos_idx = [0] * M
    completed = set()
    available_tasks = set(static_list)
    robot_schedules = [[] for _ in range(M)]
    ant_pos = 0  # 0 = depot

    while len(completed) < N:
        _unlock_dynamic(t_free, available_tasks, completed, inst)
        candidates = list(available_tasks - completed)

        if not candidates:
            pending = [_t_arrival(inst, i) for i in dyn_set
                       if i not in completed and i not in available_tasks]
            if not pending:
                break
            t_adv = min(pending)
            for j in range(M):
                t_free[j] = max(t_free[j], t_adv)
            continue

        # 计算候选任务的选择权重
        weights = []
        for n in candidates:
            ph = max(tau[ant_pos][n + 1], 1e-300) ** alpha
            heu = eta[ant_pos][n + 1] ** beta
            weights.append(ph * heu)

        total_w = sum(weights)

        if total_w < 1e-15:
            selected = random.choice(candidates)
        elif random.random() < p0:
            selected = candidates[int(np.argmax(weights))]
        else:
            probs = [w / total_w for w in weights]
            r = random.random()
            cumul = 0.0
            selected = candidates[-1]
            for idx, pv in enumerate(probs):
                cumul += pv
                if r <= cumul:
                    selected = candidates[idx]
                    break

        coalition, t_arrive_dict = assign_coalition(selected, inst, t_free, pos_idx)
        if not coalition:
            completed.add(selected)
            available_tasks.discard(selected)
            continue

        T_sat, T_complete, valid_coalition, wait_times = simulate_task_execution(
            selected, coalition, t_arrive_dict, inst)

        for j in valid_coalition:
            robot_schedules[j].append(
                (selected, t_arrive_dict[j], T_sat, T_complete, wait_times.get(j, 0.0)))
            t_free[j] = T_complete
            pos_idx[j] = selected + 1

        completed.add(selected)
        available_tasks.discard(selected)
        ant_pos = selected + 1

    t_final = [t_free[j] + dist_mat[pos_idx[j]][0] / robot_speeds[j]
               for j in range(M) if robot_schedules[j]]
    makespan = max(t_final) if t_final else 0.0
    return makespan, robot_schedules


def _extract_ant_path(robot_schedules):
    """按任务开始时刻从智能体调度记录恢复蚂蚁访问路径。"""
    tasks_by_tsat = []
    for sched in robot_schedules:
        for entry in sched:
            task_idx, _, T_sat, _, _ = entry
            tasks_by_tsat.append((T_sat, task_idx))
    tasks_by_tsat.sort(key=lambda x: (x[0], x[1]))

    seen = set()
    nodes = [0]
    for _, task_idx in tasks_by_tsat:
        node = task_idx + 1
        if node not in seen:
            nodes.append(node)
            seen.add(node)
    nodes.append(0)
    return nodes


def update_pheromone_sas(tau, iter_sched, iter_mk, global_sched, global_mk, rho):
    """蒸发信息素并强化本轮与全局最优路径。"""
    tau *= (1.0 - rho)

    if iter_mk > 1e-12:
        dep = 1.0 / iter_mk
        path = _extract_ant_path(iter_sched)
        for k in range(len(path) - 1):
            u, v = path[k], path[k + 1]
            tau[u][v] += dep
            tau[v][u] += dep

    if global_mk > 1e-12:
        dep = 1.0 / global_mk
        path = _extract_ant_path(global_sched)
        for k in range(len(path) - 1):
            u, v = path[k], path[k + 1]
            tau[u][v] += dep
            tau[v][u] += dep


# 单实例求解入口


def solve(inst, max_evals=1000, ns=NS, alpha=ALPHA, beta=BETA,
          rho=RHO, p0=P0):
    """在评估预算内搜索，返回最优完工时间和智能体序列。"""
    mk, sched = greedy_construct(inst)
    N = inst['N']
    tau = np.full((N + 1, N + 1), 1.0 / mk if mk > 1e-12 else 1.0)
    eta = build_eta(inst)
    for _ in range(0, max_evals, ns):
        iter_ms, iter_sched = min(
            (
                sas_construct_solution(inst, tau, eta, alpha, beta, p0)
                for _ in range(ns)
            ),
            key=lambda item: item[0],
        )
        if iter_ms < mk:
            mk, sched = iter_ms, iter_sched
        update_pheromone_sas(tau, iter_sched, iter_ms, sched, mk, rho)
    return mk, sched
