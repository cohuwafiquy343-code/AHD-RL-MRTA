import argparse
from .solve import solve, solve_generated
from .example_instances import load_example_instance
from .proposed.ahd_rl.config import GROUPS


def _group_id(value):
    label = str(value).upper()
    if label.startswith('G'):
        label = label[1:]
    if label.isdigit() and int(label) in GROUPS:
        return int(label)
    raise argparse.ArgumentTypeError('group must be G01–G13 or 1–13')


def main():
    p = argparse.ArgumentParser(description='Solve one MRTA instance')
    p.add_argument('--method', choices=['ahd-rl','ahd-rl-nca','ahd-rl-nrt','ahd-rl-ncr',
                                        'rca-rl','taco','sas','slpso'], default='ahd-rl')
    p.add_argument('--group', type=_group_id, default=1,
                   help='paper group G01–G13 or 1–13 (default: G01)')
    p.add_argument('--type', dest='type_key', choices=['1','2','3'], default='1')
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--sample', action='store_true')
    p.add_argument('--samples', type=int, default=1,
                   help='number of sampled trajectories for a learned policy (default: 1)')
    p.add_argument('--max-evals', type=int, default=10000)
    p.add_argument('--instance', help='path to a supplied .npz instance')
    a = p.parse_args()
    if a.instance:
        instance = load_example_instance(a.instance)
        result = solve(instance, method=a.method, seed=a.seed, sample=a.sample,
                       samples=a.samples, max_evals=a.max_evals)
    else:
        result = solve_generated(group=a.group, type_key=a.type_key, seed=a.seed,
                                 method=a.method, sample=a.sample, samples=a.samples,
                                 max_evals=a.max_evals)
    print('{}: completed {}/{}; makespan={}'.format(
        a.method, result['completed_tasks'], result['total_tasks'], result['makespan']))

if __name__ == '__main__':
    main()
