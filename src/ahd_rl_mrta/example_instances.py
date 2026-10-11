"""读取仓库随附的小规模 MRTA 测试实例。"""
from pathlib import Path

import numpy as np


def load_example_instance(path):
    """读取 NumPy 实例，并恢复调度器使用的集合与标量字段。"""
    with np.load(str(Path(path)), allow_pickle=False) as archive:
        instance = {key: archive[key].copy() for key in archive.files}

    for key in ('group_id', 'M', 'N', 'n_types'):
        instance[key] = int(instance[key])
    for key in ('T_ref', 'T_step', 'v_max', 'D_max'):
        instance[key] = float(instance[key])
    instance['instance_type'] = str(instance['instance_type'])
    instance['dyn_set'] = set(int(i) for i in instance['dyn_set'])
    instance['static_list'] = [int(i) for i in instance['static_list']]
    return instance
