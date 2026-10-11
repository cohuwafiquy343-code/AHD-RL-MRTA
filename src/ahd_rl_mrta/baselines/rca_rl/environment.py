"""在当前项目的实例上执行 RCA-RL 的任务选择与调度模拟。"""

from __future__ import annotations

import copy
import math
from typing import Dict, List

import numpy as np


EPS = 1e-8


class RCATaskEnv:
    """使用原始单位模拟调度，仅对送入 RCA-RL 网络的观测进行归一化。"""

    def __init__(self, instance: dict):
        self.source_instance = copy.deepcopy(instance)
        self.instance_type = str(instance.get('instance_type', 'unknown'))
        self.rng = None
        self.traits_dim = 5
        self.decision_dim = 30

        self._validate_instance(self.source_instance)
        self.tasks_num = int(self.source_instance['N'])
        self.agents_num = int(self.source_instance['M'])

        self.spatial_scale = 1.0 / 100.0
        duration_max = max(
            float(np.max(self.source_instance['task_duration'])), 1.0)
        self.time_scale = 5.0 / duration_max
        caps = np.asarray(self.source_instance['robots_cap'], dtype=np.float64)
        self.trait_scale = np.maximum(np.max(caps, axis=0), 1.0)

        all_pos = np.vstack([
            np.asarray(self.source_instance['depot'], dtype=np.float64),
            np.asarray(self.source_instance['task_pos'], dtype=np.float64),
        ])
        delta = all_pos[:, None, :] - all_pos[None, :, :]
        self.map_diameter = float(np.linalg.norm(delta, axis=2).max())
        min_speed = max(
            float(np.min(self.source_instance['robot_speeds'])), EPS)
        max_duration = max(
            float(np.max(self.source_instance['task_duration'])), EPS)
        load = math.sqrt(max(1.0, self.tasks_num / max(self.agents_num, 1)))
        self.max_waiting_time = max(
            4.0 * max_duration,
            2.0 * self.map_diameter / min_speed,
        ) * load

        max_release = float(np.max(self.source_instance['t_arrival']))
        serial_work = float(np.sum(self.source_instance['task_duration']))
        travel_bound = (
            2.0 * (self.tasks_num + self.agents_num)
            * self.map_diameter / min_speed)
        self.max_time = max(
            200.0 / max(self.time_scale, EPS),
            max_release + serial_work + travel_bound,
        )
        self.max_decisions = max(300, 2 * self.agents_num * (self.tasks_num + 1))

        self.dynamic_arrival_enabled = bool(
            self.instance_type == 'type3_dynamic_arrival'
            or len(self.source_instance.get('dyn_set', ())) > 0)
        self._build_dictionaries()
        self.init_state()

    @staticmethod
    def _validate_instance(instance: dict) -> None:
        """检查实例字段、数组维度及基本可行性。"""
        required = {
            'M', 'N', 'task_pos', 'robots_cap', 'robot_speeds',
            'robot_type_ids', 'task_req', 'task_duration', 't_arrival',
            'depot',
        }
        missing = sorted(required.difference(instance))
        if missing:
            raise ValueError(
                'Current instance is missing fields: {}'.format(', '.join(missing)))
        m, n = int(instance['M']), int(instance['N'])
        expected = {
            'task_pos': (n, 2),
            'robots_cap': (m, 5),
            'robot_speeds': (m,),
            'robot_type_ids': (m,),
            'task_req': (n, 5),
            'task_duration': (n,),
            't_arrival': (n,),
            'depot': (2,),
        }
        for key, shape in expected.items():
            actual = np.asarray(instance[key]).shape
            if actual != shape:
                raise ValueError(
                    '{} has shape {}, expected {}'.format(key, actual, shape))
        if np.any(np.asarray(instance['robot_speeds']) <= 0):
            raise ValueError('All robot speeds must be positive.')
        if np.any(np.asarray(instance['task_duration']) < 0):
            raise ValueError('Task durations must be non-negative.')
        total_cap = np.asarray(instance['robots_cap']).sum(axis=0)
        if np.any(np.asarray(instance['task_req']) > total_cap):
            raise ValueError('At least one task requirement exceeds total capacity.')

    def _build_dictionaries(self) -> None:
        """把数组格式的实例转换为 RCA-RL 原有的任务与智能体字典。"""
        inst = self.source_instance
        robot_types = np.asarray(inst['robot_type_ids'], dtype=np.int32)
        unique_types = sorted(int(value) for value in np.unique(robot_types))
        type_to_species = {value: idx for idx, value in enumerate(unique_types)}
        self.species_num = len(unique_types)
        self.species_range = (self.species_num, self.species_num)

        type_caps = []
        type_counts = []
        species_dict: Dict = {'source_type_ids': unique_types}
        for source_type in unique_types:
            members = np.where(robot_types == source_type)[0].tolist()
            species = type_to_species[source_type]
            species_dict[species] = members
            type_counts.append(len(members))
            type_caps.append(
                np.asarray(inst['robots_cap'][members[0]], dtype=np.float64))
        species_dict['abilities'] = np.vstack(type_caps)
        species_dict['number'] = type_counts
        self.species_dict = species_dict

        self.task_dic = {}
        task_req = np.asarray(inst['task_req'], dtype=np.float64)
        task_pos = np.asarray(inst['task_pos'], dtype=np.float64)
        duration = np.asarray(inst['task_duration'], dtype=np.float64)
        releases = np.asarray(inst['t_arrival'], dtype=np.float64)
        for task_id in range(self.tasks_num):
            self.task_dic[task_id] = {
                'ID': task_id,
                'requirements': task_req[task_id].copy(),
                'members': [],
                'cost': [],
                'location': task_pos[task_id].copy(),
                'feasible_assignment': False,
                'finished': False,
                'time_start': 0.0,
                'time_finish': 0.0,
                'status': task_req[task_id].copy(),
                'time': float(duration[task_id]),
                'base_time': float(duration[task_id]),
                'release_time': float(releases[task_id]),
                'released': False,
                'sum_waiting_time': 0.0,
                'efficiency': 0.0,
                'abandoned_agent': [],
                'optimized_ability': None,
                'optimized_species': [],
            }

        depot = np.asarray(inst['depot'], dtype=np.float64)
        caps = np.asarray(inst['robots_cap'], dtype=np.float64)
        speeds = np.asarray(inst['robot_speeds'], dtype=np.float64)
        self.agent_dic = {}
        for agent_id in range(self.agents_num):
            species = type_to_species[int(robot_types[agent_id])]
            self.agent_dic[agent_id] = {
                'ID': agent_id,
                'species': species,
                'source_type_id': int(robot_types[agent_id]),
                'abilities': caps[agent_id].copy(),
                'location': depot.copy(),
                'route': [-species - 1],
                'current_task': -species - 1,
                'contributed': False,
                'arrival_time': [0.0],
                'cost': np.zeros(1),
                'travel_time': 0.0,
                'velocity': float(speeds[agent_id]),
                'next_decision': 0.0,
                'depot': depot.copy(),
                'travel_dist': 0.0,
                'sum_waiting_time': 0.0,
                'current_action_index': 0,
                'decision_step': 0,
                'task_waiting_ratio': 1,
                'trajectory': [],
                'angle': 0.0,
                'returned': False,
                'assigned': False,
                'pre_set_route': None,
                'no_choice': False,
            }

        self.depot_dic = {}
        for species in range(self.species_num):
            self.depot_dic[species] = {
                'location': depot.copy(),
                'members': list(self.species_dict[species]),
                'ID': -species - 1,
            }
        self.per_species_range = (
            min(self.species_dict['number']), max(self.species_dict['number']))
        self.tasks_range = (self.tasks_num, self.tasks_num)

    def init_state(self) -> None:
        """重置任务完成状态和智能体路线。"""
        inst = self.source_instance
        task_req = np.asarray(inst['task_req'], dtype=np.float64)
        for task_id, task in self.task_dic.items():
            req = task_req[task_id].copy()
            task.update(
                requirements=req,
                members=[], cost=[], finished=False, status=req.copy(),
                feasible_assignment=False, time_start=0.0, time_finish=0.0,
                time=task['base_time'], sum_waiting_time=0.0, efficiency=0.0,
                abandoned_agent=[], released=task['release_time'] <= 0.0,
            )
        for agent in self.agent_dic.values():
            species = agent['species']
            agent.update(
                route=[-species - 1], location=agent['depot'].copy(),
                contributed=False, next_decision=0.0, travel_time=0.0,
                travel_dist=0.0, arrival_time=[0.0], assigned=False,
                sum_waiting_time=0.0, current_action_index=0,
                decision_step=0, trajectory=[], angle=0.0, returned=False,
                pre_set_route=None, current_task=-species - 1,
                task_waiting_ratio=1, no_choice=False, next_action=0,
            )
        for species, depot in self.depot_dic.items():
            depot['members'] = list(self.species_dict[species])

        self.coalition_matrix = np.zeros(
            (self.agents_num, self.tasks_num), dtype=np.float64)
        self.current_time = 0.0
        self.finished = False
        self.termination_reason = None
        self.decision_count = 0
        self._log_releases()

    def _log_releases(self) -> None:
        """把达到发布时间的任务标记为已开放。"""
        for task in self.task_dic.values():
            if task['release_time'] <= self.current_time + EPS:
                task['released'] = True

    def get_unfinished_tasks(self) -> List[bool]:
        """返回当前已开放且未完成的任务标记。"""
        self._log_releases()
        return [
            bool(task['released']
                 and not task['feasible_assignment']
                 and np.any(task['status'] > 0))
            for task in self.task_dic.values()
        ]

    def get_contributable_task_mask(self, agent_id: int) -> np.ndarray:
        """排除当前智能体不能贡献能力的任务。"""
        agent = self.agent_dic[agent_id]
        mask = np.ones(self.tasks_num, dtype=bool)
        for task in self.task_dic.values():
            if not task['released'] or task['feasible_assignment']:
                continue
            contribution = np.maximum(
                np.minimum(task['status'], agent['abilities']), 0.0)
            if contribution.sum() > 0:
                mask[task['ID']] = False
        return mask

    def get_current_agent_status(self, agent: dict) -> np.ndarray:
        """构造 RCA-RL 网络所需的团队智能体观测。"""
        status = []
        for other in self.agent_dic.values():
            if other['current_task'] >= 0:
                current_task = other['current_task']
                arrival_time = self.get_arrival_time(other['ID'], current_task)
                travel_time = max(arrival_time - self.current_time, 0.0)
                task = self.task_dic[current_task]
                if self.current_time <= task['time_start']:
                    waiting_time = max(self.current_time - arrival_time, 0.0)
                    remaining_time = max(
                        task['time_finish'] - self.current_time, 0.0)
                else:
                    waiting_time = 0.0
                    # Match the feature semantics used by the original model:
                    # after execution starts this field is no longer exposed
                    # as remaining work time.
                    remaining_time = 0.0
            else:
                travel_time = waiting_time = remaining_time = 0.0
            status.append(np.hstack([
                other['abilities'] / self.trait_scale,
                travel_time * self.time_scale,
                remaining_time * self.time_scale,
                waiting_time * self.time_scale,
                (agent['location'] - other['location']) * self.spatial_scale,
                float(other['assigned']),
            ]))
        return np.vstack(status)

    def get_current_task_status(self, agent: dict) -> np.ndarray:
        """构造当前智能体视角下的候选任务观测。"""
        rows = []
        for task in self.task_dic.values():
            travel_time = (
                self.calculate_eulidean_distance(agent, task) / agent['velocity'])
            rows.append(np.hstack([
                task['status'] / self.trait_scale,
                task['requirements'] / self.trait_scale,
                task['base_time'] * self.time_scale,
                travel_time * self.time_scale,
                (agent['location'] - task['location']) * self.spatial_scale,
                float(task['feasible_assignment']),
            ]))
        depot = self.depot_dic[agent['species']]
        depot_travel = (
            self.calculate_eulidean_distance(agent, depot) / agent['velocity'])
        depot_row = np.hstack([
            np.zeros(self.traits_dim),
            -np.ones(self.traits_dim),
            0.0,
            depot_travel * self.time_scale,
            (agent['location'] - agent['depot']) * self.spatial_scale,
            1.0,
        ])
        return np.vstack([depot_row, *rows])

    def _coalition_duration(self, task: dict) -> float:
        """计算当前任务联盟对应的执行时长。"""
        req = np.asarray(task['requirements'], dtype=np.float64)
        req_sum = float(req.sum())
        if req_sum <= 0 or not task['members']:
            return float(task['base_time'])
        contributing = [
            agent_id for agent_id in task['members']
            if np.any((self.agent_dic[agent_id]['abilities'] > 0) & (req > 0))
        ]
        cap_sum = float(sum(
            self.agent_dic[agent_id]['abilities'].sum()
            for agent_id in contributing))
        excess = max(0.0, cap_sum - req_sum)
        speedup = min(excess / req_sum, 0.75)
        return float(task['base_time']) * (1.0 - speedup)

    def task_update(self) -> List[int]:
        """检查联盟可行性、任务完成与等待超时，并返回新可行任务。"""
        newly_feasible = []
        for task in self.task_dic.values():
            if not task['released']:
                continue
            if not task['feasible_assignment']:
                abilities = self.get_abilities(task['members'])
                arrivals = np.array([
                    self.get_arrival_time(member, task['ID'])
                    for member in task['members']
                ], dtype=np.float64)
                task['status'] = task['requirements'] - abilities
                if np.all(task['status'] <= 0) and len(arrivals) > 0:
                    latest = float(np.max(arrivals))
                    earliest = float(np.min(arrivals))
                    if latest - earliest <= self.max_waiting_time + EPS:
                        duration = self._coalition_duration(task)
                        task['time'] = duration
                        task['time_start'] = latest
                        task['time_finish'] = latest + duration
                        task['feasible_assignment'] = True
                        newly_feasible.append(task['ID'])
                    else:
                        expired = arrivals <= latest - self.max_waiting_time
                        for member in np.asarray(task['members'])[expired].tolist():
                            task['members'].remove(int(member))
                            task['abandoned_agent'].append(int(member))
                            self._release_abandoned_agent(int(member))
                else:
                    for member in list(task['members']):
                        waited = self.current_time - self.get_arrival_time(
                            member, task['ID'])
                        if waited + EPS >= self.max_waiting_time:
                            task['members'].remove(member)
                            task['abandoned_agent'].append(member)
                            self._release_abandoned_agent(member)
                    task['status'] = (
                        task['requirements'] - self.get_abilities(task['members']))
            elif not task['finished'] and self.current_time + EPS >= task['time_finish']:
                task['finished'] = True

        all_assigned = all(
            task['feasible_assignment'] for task in self.task_dic.values())
        if all_assigned:
            # 全部任务已有可行联盟后，恢复暂时无动作的智能体，使其能够返回站点。
            for agent in self.agent_dic.values():
                if agent['no_choice']:
                    agent['no_choice'] = False
                    if agent['current_task'] >= 0:
                        task = self.task_dic[agent['current_task']]
                        agent['next_decision'] = max(
                            float(self.current_time), float(task['time_finish']))
                    else:
                        agent['next_decision'] = max(
                            float(self.current_time),
                            float(agent['arrival_time'][-1]))
            for depot in self.depot_dic.values():
                for member in depot['members']:
                    agent = self.agent_dic[member]
                    if (agent['current_task'] < 0
                            and self.current_time + EPS >= agent['arrival_time'][-1]):
                        agent['returned'] = True
        return newly_feasible

    def _release_abandoned_agent(self, agent_id: int) -> None:
        """等待超时的智能体重新进入可决策状态。"""
        agent = self.agent_dic[agent_id]
        agent['assigned'] = False
        agent['no_choice'] = False
        agent['next_decision'] = float(self.current_time)

    def next_decision(self):
        """选出下一决策时刻及在该时刻可决策的智能体。"""
        self._log_releases()
        decision_time = np.asarray(
            self.get_matrix(self.agent_dic, 'next_decision'), dtype=np.float64)
        no_choice = np.asarray(
            self.get_matrix(self.agent_dic, 'no_choice'), dtype=bool)
        effective = np.where(no_choice, np.inf, decision_time)
        if np.all(np.isnan(decision_time)):
            base_next = max(
                max(agent['arrival_time']) if agent['arrival_time'] else 0.0
                for agent in self.agent_dic.values())
            base_agents: List[int] = []
        else:
            base_next = float(np.nanmin(effective))
            if np.isinf(base_next):
                arrivals = np.array([
                    agent['arrival_time'][-1]
                    for agent in self.agent_dic.values()], dtype=np.float64)
                effective = np.where(no_choice, np.inf, arrivals)
                base_next = float(np.nanmin(effective))
            base_agents = np.where(effective == base_next)[0].tolist()

        future_releases = [
            task['release_time'] for task in self.task_dic.values()
            if not task['released'] and task['release_time'] > self.current_time + EPS
        ]
        release_next = min(future_releases) if future_releases else math.inf
        if release_next < base_next - EPS or math.isinf(base_next):
            idle = [
                agent_id for agent_id, agent in self.agent_dic.items()
                if agent['current_task'] < 0
                and agent['arrival_time'][-1] <= release_next + EPS
            ]
            if idle:
                return (idle, []), float(release_next)

        if math.isinf(base_next):
            return ([], []), float(self.current_time)
        blocked = []
        for agent_id in np.where(np.isinf(effective))[0].tolist():
            if base_next >= self.agent_dic[agent_id]['arrival_time'][-1]:
                blocked.append(agent_id)
        return (base_agents, blocked), float(base_next)

    def agent_step(self, agent_id: int, task_id: int, decision_step: int):
        """记录智能体的任务动作、路程与预计到达时间。"""
        selected_id = int(task_id) - 1
        agent = self.agent_dic[agent_id]
        if selected_id >= 0:
            task = self.task_dic[selected_id]
            if not task['released'] or task['feasible_assignment']:
                return -1, False, []
        else:
            task = self.depot_dic[agent['species']]

        task_distance = self.calculate_eulidean_distance(agent, task)
        previous_task = agent['current_task']
        agent['route'].append(task['ID'])
        agent['current_task'] = selected_id
        travel_time = task_distance / agent['velocity']
        agent['travel_time'] = travel_time
        agent['travel_dist'] += task_distance
        if (previous_task >= 0
                and self.task_dic[previous_task]['time_finish'] < self.current_time):
            departure = self.task_dic[previous_task]['time_finish']
        else:
            departure = self.current_time
        arrival = float(departure + travel_time)
        agent['arrival_time'].append(arrival)
        agent['location'] = np.asarray(task['location'], dtype=np.float64).copy()
        agent['decision_step'] = decision_step
        agent['no_choice'] = False
        agent['returned'] = False
        if agent_id not in task['members']:
            task['members'].append(agent_id)
        self.decision_count += 1
        newly_feasible = self.task_update()
        self.agent_update()
        return 0, True, newly_feasible

    def check_finished(self) -> bool:
        """检查全部任务及智能体是否完成调度。"""
        self.task_update()
        decision_agents, current_time = self.next_decision()
        if len(decision_agents[0]) + len(decision_agents[1]) == 0:
            self.current_time = max(float(self.current_time), float(current_time))
            self.task_update()
            finished = bool(
                np.all(self.get_matrix(self.agent_dic, 'returned'))
                and np.all(self.get_matrix(self.task_dic, 'finished')))
            if finished:
                self.termination_reason = 'completed'
            return finished
        return False

    def mark_terminated(self, reason: str) -> None:
        """记录因时间、动作数量或停滞触发的提前终止。"""
        self.termination_reason = str(reason)
        self.finished = False

    # RCA-RL 原有的观测与状态计算，与当前实例的调度逻辑共用同一环境。
    @staticmethod
    def get_matrix(dictionary, key):
        """按字典顺序提取各节点的指定字段。"""
        key_matrix = []
        for value in dictionary.values():
            key_matrix.append(value[key])
        return key_matrix

    @staticmethod
    def calculate_eulidean_distance(agent, task):
        """计算智能体当前位置与任务位置之间的欧氏距离。"""
        return np.linalg.norm(agent['location'] - task['location'])

    def get_unfinished_task_mask(self):
        """标记已完成任务，使其不再进入动作候选集合。"""
        mask = np.logical_not(self.get_unfinished_tasks())
        return mask

    def get_arrival_time(self, agent_id, task_id):
        """读取智能体在当前路线中最近一次抵达该任务的时刻。"""
        arrival_time = self.agent_dic[agent_id]['arrival_time']
        arrival_for_task = np.where(np.array(self.agent_dic[agent_id]['route']) == task_id)[0][-1]
        return float(arrival_time[arrival_for_task])

    def get_abilities(self, members):
        """累计指定智能体集合的能力向量。"""
        if len(members) == 0:
            return np.zeros(self.traits_dim)
        else:
            return np.sum(np.array([self.agent_dic[member]['abilities'] for member in members]), axis=0)

    def agent_update(self):
        """依据任务执行状态更新各智能体的下一次决策时间。"""
        for agent in self.agent_dic.values():
            if agent['current_task'] < 0:
                if np.all(self.get_matrix(self.task_dic, 'feasible_assignment')):
                    agent['next_decision'] = np.nan
                elif not np.isnan(agent['next_decision']):
                    agent['next_decision'] = np.inf
            else:
                current_task = self.task_dic[agent['current_task']]
                if current_task['feasible_assignment']:
                    if agent['ID'] in current_task['members']:
                        agent['next_decision'] = float(current_task['time_finish'])
                        if self.current_time >= float(current_task['time_start']):
                            agent['assigned'] = True
                    else:
                        agent['next_decision'] = (
                            self.get_arrival_time(agent['ID'], current_task['ID'])
                            + self.max_waiting_time
                        )
                        agent['assigned'] = False
                else:
                    agent['next_decision'] = (
                        self.get_arrival_time(agent['ID'], current_task['ID'])
                        + self.max_waiting_time
                    )
                    agent['assigned'] = False

    def agent_observe(self, agent_id):
        """构造当前智能体的任务、团队观测及动作掩码。"""
        agent = self.agent_dic[agent_id]
        mask = self.get_unfinished_task_mask()
        contributable_mask = self.get_contributable_task_mask(agent_id)
        mask = np.logical_or(mask, contributable_mask)
        mask = np.insert(mask, 0, False)
        agents_info = np.expand_dims(self.get_current_agent_status(agent), axis=0)
        tasks_info = np.expand_dims(self.get_current_task_status(agent), axis=0)
        mask = np.expand_dims(mask, axis=0)
        return tasks_info, agents_info, mask
