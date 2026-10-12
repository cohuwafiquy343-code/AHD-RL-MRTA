from pathlib import Path
import random
import numpy as np
import torch
from .proposed.ahd_rl.environment import ConstructionEnv, generate_instance_by_type
from .proposed.ahd_rl.model import SchedulingNetwork
from .proposed.ahd_rl.ablations import ABLATION_VARIANTS, AblationSchedulingNetwork
from .proposed.ahd_rl.policy import run_episode

MODELS = {
    'ahd-rl': 'ahd_rl.pt',
    'ahd-rl-nca': 'ahd_rl_nca.pt',
    'ahd-rl-nrt': 'ahd_rl_nrt.pt',
    'ahd-rl-ncr': 'ahd_rl_ncr.pt',
    'rca-rl': 'rca_rl.pth',
}
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / 'models'


def load_ahd_model(method='ahd-rl', model_dir=DEFAULT_MODEL_DIR, device='cpu'):
    if method != 'ahd-rl' and method not in ABLATION_VARIANTS:
        raise ValueError('unknown AHD-RL variant: {}'.format(method))
    model = SchedulingNetwork() if method == 'ahd-rl' else AblationSchedulingNetwork(method)
    state = torch.load(str(Path(model_dir) / MODELS[method]), map_location='cpu')
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def _load_rca_model(model_dir):
    from .baselines.rca_rl.attention import AttentionNet
    net = AttentionNet(11, 15, 128)
    state = torch.load(str(Path(model_dir) / MODELS['rca-rl']), map_location='cpu')
    net.load_state_dict(state, strict=True)
    return net.eval()


def _solve_rca(instance, net, sample, seed):
    from .baselines.rca_rl.environment import RCATaskEnv
    from .baselines.rca_rl.worker import Worker
    worker = Worker(net, device='cpu')
    env = RCATaskEnv(instance)
    worker.env = env
    with torch.no_grad():
        worker.run_episode(sample=sample,
                           inference_seed=seed, deterministic_order=not sample,
                           max_time=env.max_time, max_decisions=env.max_decisions)
    completed = int(sum(bool(task['finished']) for task in env.task_dic.values()))
    success = bool(env.finished and completed == env.tasks_num)
    return {'method': 'rca-rl', 'makespan': float(env.current_time) if success else None,
            'completed_tasks': completed, 'total_tasks': int(env.tasks_num),
            'success': success}


def _best_policy_result(candidates):
    feasible = [result for result in candidates if result['success']]
    if feasible:
        return min(feasible, key=lambda result: result['makespan'])
    return max(candidates, key=lambda result: result['completed_tasks'])


def solve(instance, method='ahd-rl', model_dir=DEFAULT_MODEL_DIR,
          sample=False, seed=0, max_evals=10000, samples=1):
    if method not in MODELS and method not in ('taco', 'sas', 'slpso'):
        raise ValueError('unknown method: {}'.format(method))
    if samples < 1:
        raise ValueError('samples must be positive')
    if samples > 1 and method not in MODELS:
        raise ValueError('multiple sampled trajectories require a learned policy')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if method == 'ahd-rl' or method in ABLATION_VARIANTS:
        model = load_ahd_model(method, model_dir)
        candidates = []
        for index in range(samples):
            torch.manual_seed(seed + index)
            env = ConstructionEnv(instance)
            with torch.no_grad():
                makespan, actions = run_episode(env, model, greedy=not (sample or samples > 1))
            candidates.append({'method': method, 'makespan': float(makespan),
                               'completed_tasks': int(env.num_done), 'total_tasks': int(env.N),
                               'success': bool(env.num_done == env.N), 'actions': actions})
        return _best_policy_result(candidates)
    if method == 'rca-rl':
        net = _load_rca_model(model_dir)
        candidates = [_solve_rca(instance, net, sample or samples > 1, seed + index)
                      for index in range(samples)]
        return _best_policy_result(candidates)
    if method == 'taco':
        from .baselines.taco.solver import solve as run
        makespan, _ = run(instance, max_evals=max_evals)
    elif method == 'sas':
        from .baselines.sas.solver import solve as run
        makespan, _ = run(instance, max_evals=max_evals)
    else:
        from .baselines.slpso.solver import (
            get_slpso_params, load_instance_from_rl_dict, slpso,
        )
        converted = load_instance_from_rl_dict(instance)
        params = get_slpso_params(converted['N'])
        _, makespan = slpso(converted, max_evals=max_evals, **params)
    completed = int(instance['N']) if np.isfinite(makespan) else 0
    return {'method': method, 'makespan': float(makespan) if completed else None,
            'completed_tasks': completed, 'total_tasks': int(instance['N']),
            'success': bool(completed)}


def solve_generated(group=1, type_key='1', seed=123, method='ahd-rl', **kwargs):
    instance = generate_instance_by_type(group, type_key, seed)
    return solve(instance, method=method, seed=seed, **kwargs)
