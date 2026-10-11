"""TACO 对比算法：智能体独立选任务并按信息素构造调度。"""

import random

import numpy as np

NG = 10
ALPHA = 1.0
BETA = 2.0
RHO = 0.1
P0 = 0.9
L = 0.9

# ──────────────────────────────────────────────────────────────────────────────
# 核心计算
# ──────────────────────────────────────────────────────────────────────────────

def can_contribute(robot_cap, task_req):
    """判断智能体是否拥有任务所需的任一非零能力。"""
    for d in range(len(robot_cap)):
        if robot_cap[d] > 0 and task_req[d] > 0:
            return True
    return False


def coalition_satisfies(coalition_caps_list, task_req):
    """检查联盟累计能力是否覆盖任务的每项需求。"""
    if not coalition_caps_list:
        return False
    cap_sum = np.zeros(len(task_req), dtype=np.float64)
    for cap in coalition_caps_list:
        cap_sum += np.asarray(cap, dtype=np.float64)
    return bool(np.all(cap_sum >= np.asarray(task_req, dtype=np.float64)))


def compute_exec_time(coalition_robot_ids, task_idx, inst):
    """计算当前执行联盟对应的任务执行时长。"""
    robots_cap = inst['robots_cap']
    task_req = inst['task_req'][task_idx]
    task_dur = float(inst['task_duration'][task_idx])
    req_sum = float(np.sum(task_req))
    if req_sum == 0.0:
        return task_dur
    cap_sum = 0.0
    for j in coalition_robot_ids:
        if can_contribute(robots_cap[j], task_req):
            cap_sum += float(np.sum(robots_cap[j]))
    surplus = max(0.0, cap_sum - req_sum)
    sr = min(surplus / req_sum, 0.75)
    return task_dur * (1.0 - sr)


def simulate_task_execution(task_idx, coalition_robot_ids, t_arrive_dict, inst):
    """按成员到达顺序推进任务进度并得到完成时刻。"""
    task_req = inst['task_req'][task_idx]
    robots_cap = inst['robots_cap']
    req_f = np.asarray(task_req, dtype=np.float64)

    sorted_robots = sorted(coalition_robot_ids, key=lambda r: t_arrive_dict[r])
    cumulative = np.zeros(len(task_req), dtype=np.float64)
    p = None
    for idx, r in enumerate(sorted_robots):
        cumulative += np.asarray(robots_cap[r], dtype=np.float64)
        if np.all(cumulative >= req_f):
            p = idx + 1
            break

    if p is None:
        raise ValueError(
            f"Coalition {coalition_robot_ids} cannot satisfy task {task_idx}"
        )

    T_sat = t_arrive_dict[sorted_robots[p - 1]]
    wait_times = {sorted_robots[m]: T_sat - t_arrive_dict[sorted_robots[m]]
                  for m in range(p)}

    current_valid = list(sorted_robots[:p])
    et_cur = compute_exec_time(current_valid, task_idx, inst)
    T_complete = T_sat + et_cur
    prog = 0.0
    t_last = T_sat

    for r in sorted_robots[p:]:
        t_arr = t_arrive_dict[r]
        if t_arr >= T_complete:
            continue
        prog  += (t_arr - t_last) / et_cur
        prog = min(prog, 1.0 - 1e-9)
        current_valid.append(r)
        et_cur = compute_exec_time(current_valid, task_idx, inst)
        T_complete = t_arr + (1.0 - prog) * et_cur
        t_last = t_arr

    return T_sat, T_complete, current_valid, wait_times


# ──────────────────────────────────────────────────────────────────────────────
# 共享辅助函数
# ──────────────────────────────────────────────────────────────────────────────

def _unlock_dynamic_tasks(available_tasks, completed, dyn_set, t_arrival_arr, t_now):
    """把已达到发布时间的动态任务加入当前候选集合。"""
    for i in dyn_set:
        if i not in completed and i not in available_tasks:
            if t_now >= float(t_arrival_arr[i]):
                available_tasks.add(i)


def _execute_coalition(task_idx, waiting, robot_schedules, inst,
                        t_free, pos_idx, active_robots, completed):
    """执行满足需求的联盟；撤回未参与执行成员的临时调度记录。"""
    entries = waiting[task_idx]
    robot_ids = [e[0] for e in entries]
    t_arrive_dict = {e[0]: t_free[e[0]] for e in entries}

    try:
        T_sat, T_complete, valid_col, wait_times = simulate_task_execution(
            task_idx, robot_ids, t_arrive_dict, inst
        )
    except ValueError:
        for j, prev_t, prev_pos in entries:
            t_free[j] = prev_t
            pos_idx[j] = prev_pos
            active_robots.add(j)
        del waiting[task_idx]
        completed.add(task_idx)
        return

    valid_set = set(valid_col)
    for j, prev_t, prev_pos in entries:
        if j in valid_set:
            wt = wait_times.get(j, 0.0)
            robot_schedules[j].append(
                (task_idx, t_arrive_dict[j], T_sat, T_complete, wt)
            )
            t_free[j] = T_complete
            pos_idx[j] = task_idx + 1
        else:
            t_free[j] = prev_t
            pos_idx[j] = prev_pos
        active_robots.add(j)

    del waiting[task_idx]
    completed.add(task_idx)


def build_eta_matrix(inst):
    """由节点间距离生成任务转移的启发式矩阵。"""
    dm = inst['dist_mat']
    N = inst['N']
    size = N + 1
    sub = dm[:size, :size]
    mask = sub > 1e-12
    safe = np.where(mask, sub, 1.0)  # 避免距离为零时发生除零
    eta = np.where(mask, 1.0 / safe, 1e6)
    return eta


# ──────────────────────────────────────────────────────────────────────────────
# 贪心初始解
# ──────────────────────────────────────────────────────────────────────────────

def greedy_construct(inst):
    """以最早可用智能体和最近可达任务构造初始调度。"""
    M = inst['M']
    N = inst['N']
    dist_mat = inst['dist_mat']
    robot_speeds = inst['robot_speeds']
    robots_cap = inst['robots_cap']
    task_req = inst['task_req']
    t_arrival_arr = inst['t_arrival']
    dyn_set = inst['dyn_set']
    static_list = inst['static_list']

    t_free = [0.0] * M
    pos_idx = [0] * M
    completed = set()
    available_tasks = set(static_list)
    robot_schedules = [[] for _ in range(M)]

    def arr_time(j, n):
        t = t_free[j] + dist_mat[pos_idx[j]][n + 1] / robot_speeds[j]
        return max(t, float(t_arrival_arr[n]))

    for _ in range(max(N * M * 20, 5000)):
        if len(completed) >= N:
            break

        t_now = max(t_free)
        _unlock_dynamic_tasks(available_tasks, completed, dyn_set, t_arrival_arr, t_now)

        candidates = available_tasks - completed
        if not candidates:
            future = [float(t_arrival_arr[i]) for i in dyn_set
                      if i not in completed and i not in available_tasks]
            if not future:
                break
            a = min(range(M), key=lambda j: t_free[j])
            t_free[a] = max(t_free[a], min(future))
            continue

        a = min(range(M), key=lambda j: t_free[j])
        best_n = min(candidates, key=lambda n: arr_time(a, n))

        coalition_ids = [a]
        t_arrive_dict = {a: arr_time(a, best_n)}

        if not coalition_satisfies([robots_cap[a]], task_req[best_n]):
            others = sorted([j for j in range(M) if j != a],
                            key=lambda j: arr_time(j, best_n))
            for j in others:
                coalition_ids.append(j)
                t_arrive_dict[j] = arr_time(j, best_n)
                if coalition_satisfies([robots_cap[r] for r in coalition_ids],
                                        task_req[best_n]):
                    break

        saved_t = {j: t_free[j]  for j in coalition_ids}
        saved_pos = {j: pos_idx[j] for j in coalition_ids}
        for j in coalition_ids:
            t_free[j] = t_arrive_dict[j]
            pos_idx[j] = best_n + 1

        try:
            T_sat, T_complete, valid_col, wait_times = simulate_task_execution(
                best_n, coalition_ids, t_arrive_dict, inst
            )
        except ValueError:
            for j in coalition_ids:
                t_free[j] = saved_t[j]
                pos_idx[j] = saved_pos[j]
            completed.add(best_n)
            continue

        valid_set = set(valid_col)
        for j in coalition_ids:
            if j in valid_set:
                wt = wait_times.get(j, 0.0)
                robot_schedules[j].append(
                    (best_n, t_arrive_dict[j], T_sat, T_complete, wt)
                )
                t_free[j] = T_complete
                pos_idx[j] = best_n + 1
            else:
                t_free[j] = saved_t[j]
                pos_idx[j] = saved_pos[j]
        completed.add(best_n)

    t_final = [
        t_free[j] + dist_mat[pos_idx[j]][0] / robot_speeds[j]
        for j in range(M) if robot_schedules[j]
    ]
    makespan = max(t_final) if t_final else 1.0
    return makespan, robot_schedules


# ──────────────────────────────────────────────────────────────────────────────
# TACO 调度构造
# ──────────────────────────────────────────────────────────────────────────────

def handle_deadlock(waiting, robot_schedules, inst,
                    t_free, pos_idx, active_robots, completed):
    """当候选任务均在等待联盟时，集中可用能力完成一个任务。"""
    if not waiting:
        return

    robots_cap = inst['robots_cap']
    task_req = inst['task_req']
    dist_mat = inst['dist_mat']
    robot_speeds = inst['robot_speeds']
    t_arrival_arr = inst['t_arrival']

    best_task = max(waiting.keys(), key=lambda t: len(waiting[t]))
    cur_ids = [e[0] for e in waiting[best_task]]

    if coalition_satisfies([robots_cap[r] for r in cur_ids], task_req[best_task]):
        _execute_coalition(best_task, waiting, robot_schedules, inst,
                           t_free, pos_idx, active_robots, completed)
        return

    if cur_ids:
        cap_sum = np.sum(
            [np.asarray(robots_cap[j], dtype=np.float64) for j in cur_ids], axis=0
        )
    else:
        cap_sum = np.zeros(len(task_req[best_task]), dtype=np.float64)
    req = np.asarray(task_req[best_task], dtype=np.float64)

    other_entries = [
        (e[0], e[1], e[2], tid)
        for tid, elist in waiting.items() if tid != best_task
        for e in elist
    ]
    random.shuffle(other_entries)

    for d in range(len(req)):
        while float(cap_sum[d]) < float(req[d]) and other_entries:
            found = False
            for oe_idx, (rid, _prev_t, _prev_pos, from_tid) in enumerate(other_entries):
                if robots_cap[rid][d] > 0:
                    waiting[from_tid] = [e for e in waiting[from_tid] if e[0] != rid]
                    if not waiting[from_tid]:
                        del waiting[from_tid]
                    t_arr_new = (t_free[rid]
                                 + dist_mat[pos_idx[rid]][best_task + 1]
                                 / robot_speeds[rid])
                    t_arr_new = max(t_arr_new, float(t_arrival_arr[best_task]))
                    waiting[best_task].append((rid, t_free[rid], pos_idx[rid]))
                    t_free[rid] = t_arr_new
                    pos_idx[rid] = best_task + 1
                    cap_sum     += np.asarray(robots_cap[rid], dtype=np.float64)
                    other_entries.pop(oe_idx)
                    found = True
                    break
            if not found:
                break
        if np.all(cap_sum >= req):
            break

    # 释放无法继续形成可行联盟的智能体
    for tid in list(waiting.keys()):
        if tid == best_task:
            continue
        cur = [robots_cap[e[0]] for e in waiting[tid]]
        if not coalition_satisfies(cur, task_req[tid]):
            for j, prev_t, prev_pos in waiting[tid]:
                t_free[j] = prev_t
                pos_idx[j] = prev_pos
                active_robots.add(j)
            del waiting[tid]

    cur_ids = [e[0] for e in waiting.get(best_task, [])]
    if cur_ids and coalition_satisfies([robots_cap[r] for r in cur_ids],
                                        task_req[best_task]):
        _execute_coalition(best_task, waiting, robot_schedules, inst,
                           t_free, pos_idx, active_robots, completed)
    else:
        # 释放剩余等待智能体，使调度继续推进
        for tid in list(waiting.keys()):
            for j, prev_t, prev_pos in waiting[tid]:
                t_free[j] = prev_t + 0.01
                pos_idx[j] = prev_pos
                active_robots.add(j)
        waiting.clear()


def taco_construct_solution(inst, tau, eta, alpha, beta, p0, l):
    """根据信息素和启发值生成一条 TACO 调度轨迹。"""
    M = inst['M']
    N = inst['N']
    dist_mat = inst['dist_mat']
    robot_speeds = inst['robot_speeds']
    robots_cap = inst['robots_cap']
    task_req = inst['task_req']
    t_arrival_arr = inst['t_arrival']
    dyn_set = inst['dyn_set']
    static_list = inst['static_list']

    t_free = [0.0] * M
    pos_idx = [0] * M
    completed = set()
    available_tasks = set(static_list)
    waiting = {}
    active_robots = set(range(M))
    robot_schedules = [[] for _ in range(M)]
    skip_counts = [0] * M
    MAX_SKIP = max(N * 3, 10)

    for _ in range(max(N * M * 100, 2000)):
        if len(completed) >= N:
            break

        # 按当前智能体时间开放已发布的动态任务
        t_now = max(t_free)
        _unlock_dynamic_tasks(available_tasks, completed, dyn_set, t_arrival_arr, t_now)

        # 完成已经满足能力需求的联盟安排
        for tid in list(waiting.keys()):
            if tid in completed:
                continue
            cur = [e[0] for e in waiting[tid]]
            if coalition_satisfies([robots_cap[r] for r in cur], task_req[tid]):
                _execute_coalition(tid, waiting, robot_schedules, inst,
                                   t_free, pos_idx, active_robots, completed)
        if len(completed) >= N:
            break

        if not active_robots:
            handle_deadlock(waiting, robot_schedules, inst,
                            t_free, pos_idx, active_robots, completed)
            continue

        a = min(active_robots, key=lambda j: t_free[j])

        # 排除已有可行联盟的任务
        sat_tasks = set()
        for tid, elist in waiting.items():
            if coalition_satisfies([robots_cap[e[0]] for e in elist], task_req[tid]):
                sat_tasks.add(tid)
        candidates = {n for n in available_tasks
                      if n not in completed and n not in sat_tasks}

        if not candidates:
            future = [float(t_arrival_arr[i]) for i in dyn_set
                      if i not in completed and i not in available_tasks]
            if future:
                t_free[a] = max(t_free[a], min(future))
                continue
            if waiting:
                handle_deadlock(waiting, robot_schedules, inst,
                                t_free, pos_idx, active_robots, completed)
            else:
                break
            continue

        # 结合信息素与启发值选择任务
        curr_node = pos_idx[a]
        weights = {}
        for n in candidates:
            w = (float(tau[a][curr_node][n + 1]) ** alpha
                 * float(eta[curr_node][n + 1]) ** beta)
            weights[n] = w

        if random.random() < p0:
            n_star = max(weights, key=weights.get)
        else:
            total = sum(weights.values())
            if total == 0.0:
                n_star = random.choice(list(candidates))
            else:
                r_val = random.random() * total
                cum = 0.0
                n_star = list(candidates)[-1]
                for n, w in weights.items():
                    cum += w
                    if cum >= r_val:
                        n_star = n
                        break

        t_arr_a = (t_free[a] + dist_mat[curr_node][n_star + 1] / robot_speeds[a])
        t_arr_a = max(t_arr_a, float(t_arrival_arr[n_star]))

        force = skip_counts[a] >= MAX_SKIP
        if random.random() < l or force:
            skip_counts[a] = 0
            saved_t = t_free[a]
            saved_pos = pos_idx[a]
            t_free[a] = t_arr_a
            pos_idx[a] = n_star + 1
            active_robots.discard(a)

            if n_star not in waiting:
                waiting[n_star] = []
            waiting[n_star].append((a, saved_t, saved_pos))

            cur = [e[0] for e in waiting[n_star]]
            if coalition_satisfies([robots_cap[r] for r in cur], task_req[n_star]):
                _execute_coalition(n_star, waiting, robot_schedules, inst,
                                   t_free, pos_idx, active_robots, completed)
        else:
            skip_counts[a] += 1
            t_free[a] += 1e-6

    if len(completed) < N:
        return float('inf'), robot_schedules

    t_final = [
        t_free[j] + dist_mat[pos_idx[j]][0] / robot_speeds[j]
        for j in range(M) if robot_schedules[j]
    ]
    makespan = max(t_final) if t_final else 1.0
    return makespan, robot_schedules


def update_pheromone(tau, iter_best_sched, iter_best_mksp,
                     global_best_sched, global_best_mksp, rho, M):
    """先蒸发信息素，再强化本轮和全局最优的智能体路线。"""
    for a in range(M):
        tau[a] *= (1.0 - rho)

    for sched, mksp in [(iter_best_sched, iter_best_mksp),
                        (global_best_sched, global_best_mksp)]:
        if sched is None or mksp == float('inf'):
            continue
        deposit = 1.0 / (mksp + 1e-9)
        for a in range(M):
            prev_u = 0
            for entry in sched[a]:
                v = entry[0] + 1
                tau[a][prev_u][v] = 0.5 * tau[a][prev_u][v] + 0.5 * deposit
                tau[a][v][prev_u] = 0.5 * tau[a][v][prev_u] + 0.5 * deposit
                prev_u = v
            if sched[a]:
                tau[a][prev_u][0] = 0.5 * tau[a][prev_u][0] + 0.5 * deposit
                tau[a][0][prev_u] = 0.5 * tau[a][0][prev_u] + 0.5 * deposit



def solve(inst, max_evals=1000, ng=NG, alpha=ALPHA, beta=BETA,
          rho=RHO, p0=P0, l=L):
    """在评估预算内反复构造轨迹，返回最优完工时间和智能体序列。"""
    mk, sched = greedy_construct(inst)
    M, N = inst['M'], inst['N']
    tau = [np.full((N + 1, N + 1), 1.0 / (mk + 1e-9)) for _ in range(M)]
    eta = build_eta_matrix(inst)
    for _ in range(0, max_evals, ng):
        valid = [
            (value, schedule)
            for value, schedule in (
                taco_construct_solution(inst, tau, eta, alpha, beta, p0, l)
                for _ in range(ng)
            )
            if value < float('inf')
        ]
        iter_ms, iter_sched = (
            min(valid, key=lambda item: item[0])
            if valid else (float('inf'), None)
        )
        if iter_ms < mk:
            mk, sched = iter_ms, iter_sched
        update_pheromone(tau, iter_sched, iter_ms, sched, mk, rho, M)
    return mk, sched
