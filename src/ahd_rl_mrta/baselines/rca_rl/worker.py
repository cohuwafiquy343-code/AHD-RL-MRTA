"""RCA-RL 在单个实例上的策略推理。"""

import random

import numpy as np
import torch
from torch.distributions import Categorical


class Worker:
    """按可决策时间依次调用 RCA-RL 策略。"""

    def __init__(self, network, device='cpu'):
        self.device = device
        self.local_net = network
        self.env = None

    def run_episode(self, sample=False,
                    inference_seed=None, deterministic_order=False,
                    max_time=None, max_decisions=None):
        """运行一条轨迹；sample 控制采样或贪婪选取动作。"""
        current_action_index = 0
        decision_step = 0
        time_limit = float(
            max_time if max_time is not None
            else self.env.max_time)
        decision_limit = int(
            max_decisions if max_decisions is not None
            else getattr(self.env, 'max_decisions', 300))
        seeded_random = random.Random(inference_seed) if inference_seed is not None else None
        if inference_seed is not None:
            np.random.seed(int(inference_seed) % (2 ** 32 - 1))
            torch.manual_seed(int(inference_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(inference_seed))

        while (not self.env.finished
               and self.env.current_time < time_limit
               and current_action_index < decision_limit):
            with torch.no_grad():
                # 同一时刻可决策的智能体依次选择任务。
                release_agents, current_time = self.env.next_decision()
                self.env.current_time = current_time
                had_release = bool(release_agents[0] or release_agents[1])
                self.env._log_releases()
                if deterministic_order:
                    release_agents[0].sort()
                    release_agents[1].sort()
                elif seeded_random is not None:
                    seeded_random.shuffle(release_agents[0])
                else:
                    random.shuffle(release_agents[0])
                while release_agents[0] or release_agents[1]:
                    agent_id = (
                        release_agents[0].pop(0)
                        if release_agents[0] else release_agents[1].pop(0)
                    )
                    agent = self.env.agent_dic[agent_id]
                    task_info, total_agents, mask = self.convert_torch(
                        self.env.agent_observe(agent_id)
                    )
                    no_task_action = bool(mask[0, 1:].all().item())
                    if no_task_action:
                        all_assigned = np.all(
                            self.env.get_matrix(self.env.task_dic, 'feasible_assignment')
                        )
                        if not all_assigned:
                            agent['no_choice'] = True
                            continue
                        if agent['current_task'] < 0:
                            continue
                    index = torch.LongTensor([agent_id]).reshape(1, 1, 1).to(self.device)
                    probs, _ = self.local_net(task_info, total_agents, mask, index)
                    if sample:
                        action = Categorical(probs).sample()
                    else:
                        action = torch.argmax(probs, dim=1)
                    _, doable, _ = self.env.agent_step(agent_id, action.item(), decision_step)
                    agent['current_action_index'] = current_action_index
                    if doable:
                        current_action_index += 1
                self.env.finished = self.env.check_finished()
                decision_step += 1

                if not self.env.finished and not had_release:
                    break

        if not self.env.finished and hasattr(self.env, 'mark_terminated'):
            if current_action_index >= decision_limit:
                reason = 'decision_limit'
            elif self.env.current_time >= time_limit:
                reason = 'time_limit'
            else:
                reason = 'stalled'
            self.env.mark_terminated(reason)

    def convert_torch(self, args):
        """把环境观测转为策略网络所需的浮点张量。"""
        data = []
        for arg in args:
            data.append(torch.tensor(arg, dtype=torch.float).to(self.device))
        return data
