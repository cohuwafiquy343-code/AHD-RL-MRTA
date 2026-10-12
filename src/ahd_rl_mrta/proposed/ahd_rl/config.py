import math

D_CAP = 5

# Group IDs follow the paper; each pair is (agents, tasks).
GROUPS = {
    1: (5, 30),
    2: (10, 30),
    3: (15, 30),
    4: (10, 50),
    5: (15, 50),
    6: (10, 100),
    7: (15, 100),
    8: (10, 200),
    9: (15, 200),
    10: (20, 200),
    11: (10, 300),
    12: (15, 300),
    13: (20, 300),
}

ROBOT_TYPES = {
    1: 2,
    2: 3,
    3: 4,
    4: 3,
    5: 4,
    6: 3,
    7: 4,
    8: 3,
    9: 4,
    10: 5,
    11: 3,
    12: 4,
    13: 5,
}
DM = 128
NH = 8
DFF = 256
NLT = 2
NLR = 2

DROP = 0.1
CC = 10.0
PHASE_TASK_SELECT = 0
PHASE_ROBOT_SATISFY = 1
PHASE_ROBOT_ACCEL = 2
TASK_SELECT_FEAT_DIM = 18
ROBOT_TASK_FEAT_DIM = 23
ROBOT_FEAT_DIM = 21
D_MAP = 100.0 * math.sqrt(2)
INF = 1e18
EPS = 1e-8
INSTANCE_TYPES = {
    '1': {'name': 'type1_easy_coop', 'all_static': True,
          'task_req_mode': 'easy_coop'},
    '2': {'name': 'type2_hard_coop', 'all_static': True,
          'task_req_mode': 'default'},
    '3': {'name': 'type3_dynamic_arrival', 'all_static': False,
          'task_req_mode': 'default'},
}
