"""SLPSO 对比算法：粒子更新、局部搜索与单实例调度评估。"""

from copy import deepcopy

import numpy as np


def load_instance_from_rl_dict(inst_dict):
    """将生成器返回的内存实例转换为 SLPSO 使用的字段结构。"""
    M = int(inst_dict['M'])
    N = int(inst_dict['N'])
    K = 5

    if 'depot' in inst_dict:
        depot = np.array(inst_dict['depot'], dtype=np.float64).ravel()[:2]
    elif 'all_pos' in inst_dict:
        depot = np.array(inst_dict['all_pos'][0], dtype=np.float64).ravel()[:2]
    else:
        depot = np.array([50.0, 50.0], dtype=np.float64)

    task_locations = np.array(inst_dict['task_pos'], dtype=np.float64)
    robot_capabilities = np.array(inst_dict['robots_cap'], dtype=np.int32)
    robot_speeds = np.array(inst_dict['robot_speeds'], dtype=np.float64)
    task_requirements = np.array(inst_dict['task_req'], dtype=np.int32)
    task_base_times = np.array(inst_dict['task_duration'], dtype=np.float64)
    dist_matrix = np.array(inst_dict['dist_mat'], dtype=np.float64)

    t_arrival_raw = inst_dict.get('t_arrival', None)
    t_arrival = (np.array(t_arrival_raw, dtype=np.float64)
                 if t_arrival_raw is not None
                 else np.zeros(N, dtype=np.float64))

    robots = [{'cap': robot_capabilities[i], 'speed': float(robot_speeds[i])}
              for i in range(M)]
    tasks = [{'pos': task_locations[i], 'req': task_requirements[i]}
             for i in range(N)]

    return {
        'depot': depot, 'tasks': tasks, 'robots': robots,
        'N': N, 'M': M, 'K': K,
        'dist_matrix': dist_matrix,
        'robot_capabilities': robot_capabilities,
        'robot_speeds': robot_speeds,
        'task_locations': task_locations,
        'task_requirements': task_requirements,
        'task_base_times': task_base_times,
        't_arrival': t_arrival,
    }

def compute_time_factor(coalition, task_req, robot_capabilities):
    """根据联盟超额能力计算任务执行时间系数。"""
    req_sum = int(np.sum(task_req))
    if req_sum == 0:
        return 1.0

    K = len(task_req)
    contrib_cap_sum = 0
    for r in coalition:
        contributes = any(
            int(task_req[k]) > 0 and int(robot_capabilities[r, k]) > 0
            for k in range(K)
        )
        if contributes:
            contrib_cap_sum += int(np.sum(robot_capabilities[r]))

    if contrib_cap_sum <= req_sum:
        return 1.0

    excess_ratio = (contrib_cap_sum - req_sum) / req_sum
    reduction = min(excess_ratio, 0.75)
    return max(0.25, 1.0 - reduction)


# 联盟能力可行性
def is_feasible(coalition, task_id, instance):
    """检查联盟能力是否逐项满足任务需求。"""
    if not coalition:
        return False
    task_req = instance['task_requirements'][task_id]
    robot_caps = instance['robot_capabilities']
    K = instance['K']
    for k in range(K):
        req_k = int(task_req[k])
        if req_k <= 0:
            continue
        total = sum(int(robot_caps[r, k]) for r in coalition)
        if total < req_k:
            return False
    return True


# 按成员到达事件分段计算任务进度
def evaluate(solution, instance):
    """评估任务排列与联盟配置，返回完整调度的完工时间。"""
    perm = solution['perm']
    coalitions = solution['coalitions']
    M = instance['M']
    robots = instance['robots']
    dm = instance['dist_matrix']
    task_req_arr = instance['task_requirements']
    robot_cap_arr = instance['robot_capabilities']
    task_base_times = instance['task_base_times']
    t_arrival = instance.get('t_arrival', None)

    avail = [0.0] * M
    loc = [0] * M

    for tid in perm:
        coal = set(coalitions.get(tid, set()))
        if not coal:
            continue
        if not is_feasible(coal, tid, instance):
            return 1e18

        ti = tid + 1
        base_time = float(task_base_times[tid])
        task_req = task_req_arr[tid]
        t_avail_tid = float(t_arrival[tid]) if t_arrival is not None else 0.0

        arrivals = {r: avail[r] + dm[loc[r]][ti] / robots[r]['speed'] for r in coal}
        sorted_arrivals = sorted(arrivals.items(), key=lambda x: x[1])

        current_coalition = set()
        task_started = False
        fraction_done = 0.0
        last_event_time = 0.0
        task_done_time = None
        backtrack_robots = set()

        for robot_r, arr_t in sorted_arrivals:
            if task_done_time is not None:
                backtrack_robots.add(robot_r)
                continue

            eff_t = max(arr_t, t_avail_tid)

            if task_started:
                elapsed = eff_t - last_event_time
                tf = compute_time_factor(current_coalition, task_req, robot_cap_arr)
                remaining_real = (1.0 - fraction_done) * base_time * tf
                if elapsed >= remaining_real - 1e-10:
                    task_done_time = last_event_time + remaining_real
                    backtrack_robots.add(robot_r)
                else:
                    fraction_done     += elapsed / (base_time * tf)
                    current_coalition.add(robot_r)
                    last_event_time = eff_t
            else:
                current_coalition.add(robot_r)
                if is_feasible(current_coalition, tid, instance):
                    task_started = True
                    last_event_time = eff_t

        if task_done_time is None:
            if task_started:
                tf = compute_time_factor(current_coalition, task_req, robot_cap_arr)
                remaining_real = (1.0 - fraction_done) * base_time * tf
                task_done_time = last_event_time + remaining_real
            else:
                return 1e18

        final_coalition = coal - backtrack_robots
        for r in final_coalition:
            avail[r] = task_done_time
            loc[r] = ti

    makespan = 0.0
    for r in range(M):
        t = avail[r] + dm[loc[r]][0] / robots[r]['speed']
        if t > makespan:
            makespan = t
    return makespan


# 解的表示
def perm_to_adj(perm):
    """把任务排列转为相邻任务边集合，供粒子速度更新使用。"""
    adj = {}
    if not perm:
        adj[-1] = set()
        return adj
    adj[-1] = {perm[0]}
    for i in range(len(perm) - 1):
        adj[perm[i]] = {perm[i + 1]}
    adj[perm[-1]] = set()
    return adj


def generate_greedy_coalition(task_id, instance):
    """为单个任务构造满足能力需求的初始联盟。"""
    K = instance['K']
    M = instance['M']
    robot_caps = instance['robot_capabilities']
    task_req = instance['task_requirements'][task_id]
    coalition = set()
    for k in range(K):
        req_k = int(task_req[k])
        cur = sum(int(robot_caps[r, k]) for r in coalition)
        if cur >= req_k:
            continue
        candidates = [(r, int(robot_caps[r, k])) for r in range(M) if r not in coalition]
        candidates.sort(key=lambda x: -x[1])
        for r, c in candidates:
            coalition.add(r)
            cur += c
            if cur >= req_k:
                break
    return coalition


# 粒子速度更新
def update_velocity(particle, pbest_sol, sbest_sol, omega, c1, c2, N):
    """结合粒子历史最优解与群体参考解更新任务、智能体选择倾向。"""
    r1 = np.random.random()
    r2 = np.random.random()
    coeff1 = c1 * r1
    coeff2 = c2 * r2
    X_adj = perm_to_adj(particle['perm'])
    pb_adj = perm_to_adj(pbest_sol['perm'])
    sb_adj = perm_to_adj(sbest_sol['perm'])

    new_VT = {}
    for j, probs in particle['V_T'].items():
        d = {}
        for t, p in probs.items():
            v = omega * p
            if v > 1e-12:
                d[t] = min(v, 1.0)
        if d:
            new_VT[j] = d

    all_keys = set(pb_adj.keys()) | set(X_adj.keys())
    for j in all_keys:
        d = pb_adj.get(j, set()) - X_adj.get(j, set())
        if d:
            if j not in new_VT:
                new_VT[j] = {}
            for t in d:
                new_VT[j][t] = max(new_VT[j].get(t, 0.0), coeff1)

    all_keys = set(sb_adj.keys()) | set(X_adj.keys())
    for j in all_keys:
        d = sb_adj.get(j, set()) - X_adj.get(j, set())
        if d:
            if j not in new_VT:
                new_VT[j] = {}
            for t in d:
                new_VT[j][t] = max(new_VT[j].get(t, 0.0), coeff2)
    particle['V_T'] = new_VT

    new_VR = {}
    for tid, probs in particle['V_R'].items():
        d = {}
        for r, p in probs.items():
            v = omega * p
            if v > 1e-12:
                d[r] = min(v, 1.0)
        if d:
            new_VR[tid] = d

    pb_coal = pbest_sol['coalitions']
    x_coal = particle['coalitions']
    sb_coal = sbest_sol['coalitions']
    for tid in range(N):
        d = pb_coal.get(tid, set()) - x_coal.get(tid, set())
        if d:
            if tid not in new_VR:
                new_VR[tid] = {}
            for r in d:
                new_VR[tid][r] = max(new_VR[tid].get(r, 0.0), coeff1)
    for tid in range(N):
        d = sb_coal.get(tid, set()) - x_coal.get(tid, set())
        if d:
            if tid not in new_VR:
                new_VR[tid] = {}
            for r in d:
                new_VR[tid][r] = max(new_VR[tid].get(r, 0.0), coeff2)
    particle['V_R'] = new_VR


# 解的构造
def select_nearest(current, candidates, dm):
    """从候选节点中选出与当前位置距离最近者。"""
    ci = 0 if current == -1 else current + 1
    best, best_d = None, float('inf')
    for t in candidates:
        d = dm[ci][t + 1]
        if d < best_d:
            best_d = d
            best = t
    return best


def select_robots_heuristic(task_id, task_idx, coal, cands, inst, avail, loc):
    """按能力贡献和预计到达情况为当前任务补充智能体。"""
    K = inst['K']
    robots = inst['robots']
    dm = inst['dist_matrix']
    req = inst['tasks'][task_id]['req']
    coal = set(coal)
    cands = [r for r in cands if r not in coal]

    while cands:
        if is_feasible(coal, task_id, inst):
            break
        s_vals = {}
        for r in cands:
            tc = coal | {r}
            ma = 0.0
            for rr in tc:
                a = avail[rr] + dm[loc[rr]][task_idx] / robots[rr]['speed']
                if a > ma:
                    ma = a
            s_vals[r] = ma
        s_max = max(s_vals.values()) if s_vals else 1e-9
        if s_max < 1e-9:
            s_max = 1e-9
        best_r, best_h = None, -float('inf')
        for r in cands:
            tc = coal | {r}
            cps = []
            for k in range(K):
                cs = sum(int(robots[rr]['cap'][k]) for rr in tc)
                cps.append(cs / req[k] if req[k] > 0 else 1.0)
            h1p1 = sum(min(c, 1.0) for c in cps) / K
            cbar = sum(cps) / len(cps) if cps else 0
            h1p2 = 1.0 / (sum(abs(c - cbar) for c in cps) + 4)
            h1 = h1p1 + h1p2
            se = max(s_vals[r], 1e-9)
            h3 = s_max / se
            h = h1 + h3
            if h > best_h:
                best_h = h
                best_r = r
        if best_r is not None:
            coal.add(best_r)
            cands.remove(best_r)
        else:
            break
    return coal


def construct_solution(particle, instance):
    """依据粒子的任务与智能体选择倾向构造可评估的调度解。"""
    N = instance['N']
    M = instance['M']
    dm = instance['dist_matrix']
    robots = instance['robots']
    robot_cap_arr = instance['robot_capabilities']
    task_base_times = instance['task_base_times']
    task_req_arr = instance['task_requirements']
    t_arrival = instance.get('t_arrival', None)

    alpha = np.random.uniform(0.01, 1.0)
    VT_cut = {}
    for j, probs in particle['V_T'].items():
        cut = {t for t, p in probs.items() if p >= alpha}
        if cut:
            VT_cut[j] = cut
    VR_cut = {}
    for tid, probs in particle['V_R'].items():
        cut = {r for r, p in probs.items() if p >= alpha}
        if cut:
            VR_cut[tid] = cut

    X_adj = perm_to_adj(particle['perm'])
    new_perm = []
    new_coal = {}
    unvis = set(range(N))
    cur = -1
    avail = [0.0] * M
    loc = [0] * M
    current_time = 0.0  # 用于判断动态任务是否已经发布

    for _ in range(N):
        # 仅考虑已达到发布时间的任务
        if t_arrival is not None:
            avail_tasks = {t for t in unvis if float(t_arrival[t]) <= current_time}
            if not avail_tasks:
                # 如果剩余任务均未发布，推进到最近的发布时间。
                current_time = min(float(t_arrival[t]) for t in unvis)
                avail_tasks = {t for t in unvis if float(t_arrival[t]) <= current_time}
        else:
            avail_tasks = unvis

        UV = VT_cut.get(cur, set()) & avail_tasks
        if UV:
            sel = select_nearest(cur, UV, dm)
        else:
            UX = X_adj.get(cur, set()) & avail_tasks
            if UX:
                sel = select_nearest(cur, UX, dm)
            else:
                sel = select_nearest(cur, avail_tasks, dm)
        if sel is None:
            sel = min(avail_tasks)

        new_perm.append(sel)
        unvis.remove(sel)
        cur = sel

        # 为当前任务构造执行联盟
        tid = sel
        ti = tid + 1
        UV_R = VR_cut.get(tid, set())
        UX_R = set(particle['coalitions'].get(tid, set()))
        all_R = set(range(M))
        coal = set()
        for layer in [list(UV_R), list(UX_R - UV_R), list(all_R - UV_R - UX_R)]:
            if is_feasible(coal, tid, instance):
                break
            if layer:
                coal = select_robots_heuristic(tid, ti, coal, layer, instance, avail, loc)
        if not is_feasible(coal, tid, instance):
            coal = set(range(M))
        new_coal[tid] = coal

        # 根据当前任务安排推进调度时间
        if coal:
            ma = 0.0
            for r in coal:
                a = avail[r] + dm[loc[r]][ti] / robots[r]['speed']
                if a > ma:
                    ma = a
            base_time = float(task_base_times[tid])
            task_req = task_req_arr[tid]
            tf = compute_time_factor(coal, task_req, robot_cap_arr)
            t_avail_tid = float(t_arrival[tid]) if t_arrival is not None else 0.0
            task_start = max(ma, t_avail_tid)
            task_done = task_start + base_time * tf
            for r in coal:
                avail[r] = task_done
                loc[r] = ti
            current_time = max(current_time, task_done)

    return {'perm': new_perm, 'coalitions': new_coal}


# 局部搜索
def estimate_ms(perm, coalitions, instance):
    """评估局部搜索产生的任务排列和联盟配置。"""
    return evaluate({'perm': perm, 'coalitions': coalitions}, instance)


def mutate_coalition(orig_coal, task_id, M, instance):
    """扰动任务联盟；不可行时返回 None。"""
    tc = set(orig_coal)
    op = np.random.randint(0, 3)
    if op == 0:
        av = [r for r in range(M) if r not in tc]
        if av:
            tc.add(int(np.random.choice(av)))
    elif op == 1:
        if len(tc) > 1:
            tc.remove(int(np.random.choice(list(tc))))
    else:
        if tc:
            av = [r for r in range(M) if r not in tc]
            if av:
                tc.remove(int(np.random.choice(list(tc))))
                tc.add(int(np.random.choice(av)))
    if is_feasible(tc, task_id, instance):
        return tc
    return None


def local_search(solution, instance, I=10, J=10, L=10):
    """在任务顺序与联盟成员两个空间中搜索邻近解。"""
    N = instance['N']
    M = instance['M']
    perm = list(solution['perm'])
    coal = {k: set(v) for k, v in solution['coalitions'].items()}
    cur_ms = estimate_ms(perm, coal, instance)

    for _ in range(I):
        if len(perm) < 2:
            break
        rp = int(np.random.randint(0, len(perm)))
        rt = perm[rp]
        orig_c = set(coal[rt])
        tp = perm[:rp] + perm[rp + 1:]
        rm_ms = estimate_ms(tp, coal, instance)
        if rm_ms >= cur_ms:
            continue
        new_c = set(orig_c)
        for _a in range(10):
            res = mutate_coalition(orig_c, rt, M, instance)
            if res is not None:
                new_c = res
                break
        best_ms_i = float('inf')
        best_pos = None
        best_c_i = None
        positions = list(range(len(tp) + 1))
        if len(positions) > J:
            positions = list(np.random.choice(positions, J, replace=False))
        for pos in positions:
            for l in range(L):
                if l == 0:
                    tc = set(new_c)
                else:
                    res = mutate_coalition(new_c, rt, M, instance)
                    tc = res if res is not None else set(new_c)
                test_p = tp[:pos] + [rt] + tp[pos:]
                test_coal = dict(coal)
                test_coal[rt] = tc
                ms_t = estimate_ms(test_p, test_coal, instance)
                if ms_t < best_ms_i:
                    best_ms_i = ms_t
                    best_pos = pos
                    best_c_i = set(tc)
        if best_ms_i < cur_ms and best_pos is not None:
            perm = tp[:best_pos] + [rt] + tp[best_pos:]
            coal[rt] = best_c_i
            cur_ms = best_ms_i

    return {'perm': perm, 'coalitions': coal}, cur_ms


# 存档选择与维护
def select_sbest(particle, archive, gbest):
    """为粒子选择速度更新时使用的群体参考解。"""
    if not archive:
        return gbest
    sa = sorted(archive, key=lambda x: x['ms'])
    top = sa[:max(1, len(sa) // 2)]
    p_adj = perm_to_adj(particle['perm'])
    best_sim, best_sol = -1, top[0]
    for sol in top:
        s_adj = perm_to_adj(sol['perm'])
        same = total = 0
        for key in p_adj:
            ps = p_adj[key]
            ss = s_adj.get(key, set())
            same  += len(ps & ss)
            total += len(ps)
        sim = same / total if total > 0 else 0
        if sim > best_sim:
            best_sim = sim
            best_sol = sol
    return best_sol


def maintain_archive(particles, old_archive, archive_size):
    """保留当前粒子和既有档案中的优良且不同的解。"""
    cands = list(old_archive)
    for p in particles:
        cands.append({'perm': list(p['perm']),
                      'coalitions': {k: set(v) for k, v in p['coalitions'].items()},
                      'ms': p['ms']})
        cands.append({'perm': list(p['pbest_perm']),
                      'coalitions': {k: set(v) for k, v in p['pbest_coal'].items()},
                      'ms': p['pbest_ms']})
    cands.sort(key=lambda x: x['ms'])
    new_arch = []
    seen = set()
    for c in cands:
        key = tuple(c['perm'])
        if key not in seen:
            seen.add(key)
            new_arch.append(c)
            if len(new_arch) >= archive_size:
                break
    return new_arch


# 根据原实验配置，为不同任务规模选取种群与局部搜索参数。
def get_slpso_params(task_count):
    if task_count <= 50:
        return dict(swarm_size=40, archive_size=40, Kls=2,
                    Ils=10, Jls=10, Lls=10)
    if task_count <= 100:
        return dict(swarm_size=30, archive_size=30, Kls=2,
                    Ils=8, Jls=8, Lls=8)
    if task_count <= 200:
        return dict(swarm_size=25, archive_size=25, Kls=2,
                    Ils=6, Jls=6, Lls=6)
    return dict(swarm_size=20, archive_size=20, Kls=1,
                Ils=5, Jls=5, Lls=5)


# SLPSO 单实例求解入口
def slpso(instance, swarm_size=40, max_evals=10000, archive_size=40,
          c1=1.49, c2=1.49, omega_s=0.9, omega_e=0.4,
          Kls=2, Ils=10, Jls=10, Lls=10):
    """在给定评估预算内更新粒子群，并返回找到的最优调度。"""
    N = instance['N']
    M = instance['M']
    eval_cnt = [0]

    def ec_eval(sol):
        """统计一次解评估并计算该解的完工时间。"""
        eval_cnt[0] += 1
        return evaluate(sol, instance)

    particles = []
    gbest = None
    gbest_ms = float('inf')
    all_tasks = set(range(N))

    for _ in range(swarm_size):
        pm = list(np.random.permutation(N))
        cl = {}
        for tid in pm:
            cl[tid] = generate_greedy_coalition(tid, instance)
        ms = ec_eval({'perm': pm, 'coalitions': cl})
        VT = {-1: {t: 1.0 for t in all_tasks}}
        for t in range(N):
            VT[t] = {tt: 1.0 for tt in all_tasks}
        VR = {t: {r: 1.0 for r in range(M)} for t in range(N)}
        p = {
            'perm': pm, 'coalitions': cl,
            'V_T': VT, 'V_R': VR,
            'pbest_perm': list(pm),
            'pbest_coal': {k: set(v) for k, v in cl.items()},
            'pbest_ms': ms, 'ms': ms
        }
        particles.append(p)
        if ms < gbest_ms:
            gbest_ms = ms
            gbest = {'perm': list(pm),
                     'coalitions': {k: set(v) for k, v in cl.items()},
                     'ms': ms}


    archive = []
    while eval_cnt[0] < max_evals:
        progress = min(eval_cnt[0] / max_evals, 1.0)
        omega = omega_s - (omega_s - omega_e) * progress

        for i in range(swarm_size):
            if eval_cnt[0] >= max_evals:
                break
            p = particles[i]
            pb_sol = {'perm': p['pbest_perm'], 'coalitions': p['pbest_coal']}
            sb_sol = select_sbest(p, archive, gbest)
            update_velocity(p, pb_sol, sb_sol, omega, c1, c2, N)
            new_sol = construct_solution(p, instance)
            new_ms = ec_eval(new_sol)
            p['perm'] = new_sol['perm']
            p['coalitions'] = new_sol['coalitions']
            p['ms'] = new_ms
            if new_ms < p['pbest_ms']:
                p['pbest_perm'] = list(new_sol['perm'])
                p['pbest_coal'] = {k: set(v) for k, v in new_sol['coalitions'].items()}
                p['pbest_ms'] = new_ms
            if new_ms < gbest_ms:
                gbest_ms = new_ms
                gbest = {'perm': list(new_sol['perm']),
                         'coalitions': {k: set(v) for k, v in new_sol['coalitions'].items()},
                         'ms': new_ms}

        archive = maintain_archive(particles, archive, archive_size)
        if archive and archive[0]['ms'] < gbest_ms:
            gbest_ms = archive[0]['ms']
            gbest = {'perm': list(archive[0]['perm']),
                     'coalitions': {k: set(v) for k, v in archive[0]['coalitions'].items()},
                     'ms': archive[0]['ms']}


        if archive and eval_cnt[0] < max_evals:
            ls_sols = [deepcopy(archive[0])]
            if len(archive) > 1:
                k_sel = min(Kls, len(archive) - 1)
                idxs = np.random.choice(range(1, len(archive)), k_sel, replace=False)
                for idx in idxs:
                    ls_sols.append(deepcopy(archive[int(idx)]))
            improved = []
            for sol in ls_sols:
                if eval_cnt[0] >= max_evals:
                    break
                imp_sol, _ = local_search(sol, instance, Ils, Jls, Lls)
                final_ms = ec_eval(imp_sol)
                improved.append({'perm': list(imp_sol['perm']),
                                  'coalitions': {k: set(v) for k, v in imp_sol['coalitions'].items()},
                                  'ms': final_ms})
                if final_ms < gbest_ms:
                    gbest_ms = final_ms
                    gbest = {'perm': list(imp_sol['perm']),
                             'coalitions': {k: set(v) for k, v in imp_sol['coalitions'].items()},
                             'ms': final_ms}
            archive.extend(improved)
            archive.sort(key=lambda x: x['ms'])
            new_arch = []
            seen_keys = set()
            for a in archive:
                key = tuple(a['perm'])
                if key not in seen_keys:
                    seen_keys.add(key)
                    new_arch.append(a)
                    if len(new_arch) >= archive_size:
                        break
            archive = new_arch


        if archive and archive[0]['ms'] < gbest_ms:
            gbest_ms = archive[0]['ms']
            gbest = {'perm': list(archive[0]['perm']),
                     'coalitions': {k: set(v) for k, v in archive[0]['coalitions'].items()},
                     'ms': archive[0]['ms']}

    return gbest, gbest_ms
