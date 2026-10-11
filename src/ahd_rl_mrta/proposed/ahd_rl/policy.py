"""使用已加载的 AHD-RL 策略，在单个实例上构造调度。"""
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .config import (EPS, PHASE_ROBOT_ACCEL, PHASE_ROBOT_SATISFY,
                     PHASE_TASK_SELECT, TASK_SELECT_FEAT_DIM)
from .environment import ConstructionEnv
from .model import SchedulingNetwork


def run_episode(env: ConstructionEnv, model: SchedulingNetwork,
                greedy: bool = True, device: Optional[torch.device] = None
                ) -> Tuple[float, List[Tuple[int, int]]]:
    """按任务动作与智能体动作推进环境，返回完工时间和动作序列。"""
    if device is None:
        device = next(model.parameters()).device
    env.reset()
    actions: List[Tuple[int, int]] = []
    task_context = None
    coalition_state = None

    with torch.no_grad():
        while not env.done:
            phase = env.phase
            features = env.get_mask_and_features()
            if phase == PHASE_TASK_SELECT:
                active_ids, task_features, robot_features, mask_np = features
                coalition_state = None
            else:
                task_features, robot_features, mask_np = features

            if not mask_np.any():
                # 当前阶段无合法动作时，由环境处理终止或不可行状态。
                env.step(env.M if phase == PHASE_ROBOT_ACCEL else -1)
                if env.phase == PHASE_TASK_SELECT and phase != PHASE_TASK_SELECT:
                    task_context = None
                    coalition_state = None
                continue

            tf = torch.from_numpy(task_features).float().to(device)
            rf = torch.from_numpy(robot_features).float().to(device)
            mask = torch.from_numpy(mask_np).to(device)
            if phase == PHASE_TASK_SELECT:
                # 任务决策使用当前任务、智能体及能力贡献掩码。
                inst = env.inst
                active_array = np.array(active_ids, dtype=np.int32)
                contributions = torch.from_numpy(
                    inst['robot_task_contrib'][:, active_array].T).to(device)
                depot = np.zeros(TASK_SELECT_FEAT_DIM, dtype=np.float32)
                depot[15:17] = np.asarray(inst['depot'], dtype=np.float32) / 100.0
                logits, task_embeddings = model.forward_task_select(
                    tf, rf, mask, contrib_mask=contributions,
                    depot_feat=torch.from_numpy(depot).to(device))
            else:
                # 智能体决策同时使用剩余需求、已选任务表示及当前联盟状态。
                inst = env.inst
                required = inst['task_req'][env.current_task].astype(np.float32)
                gap = np.maximum(0.0, required - env.current_coalition_cap.astype(np.float32))
                gap_vector = torch.from_numpy(
                    (gap / np.maximum(inst['r_max'], EPS)).astype(np.float32)).to(device)
                task_tensor = (torch.from_numpy(task_context.astype(np.float32)).to(device)
                               if task_context is not None else None)
                coalition_tensor = (torch.from_numpy(coalition_state.astype(np.float32)).to(device)
                                    if coalition_state is not None else None)
                logits, robot_embeddings = model.forward_robot_select(
                    tf, rf, mask, phase, task_ctx=task_tensor,
                    gap_vec=gap_vector, coal_state=coalition_tensor)

            probabilities = F.softmax(logits, dim=-1)
            action_index = (int(torch.argmax(probabilities).item()) if greedy else
                            int(torch.distributions.Categorical(probs=probabilities).sample().item()))

            if phase == PHASE_TASK_SELECT:
                # 保存本次任务表示，供该任务后续的智能体动作使用。
                task_context = task_embeddings[action_index].detach().cpu().numpy()
                coalition_state = None
                task_id = active_ids[action_index]
                actions.append((phase, task_id))
                env._do_select_task(task_id, task_ctx=task_context.astype(np.float32))
            else:
                actions.append((phase, action_index))
                env.step(action_index)
                if phase in (PHASE_ROBOT_SATISFY, PHASE_ROBOT_ACCEL) and action_index < env.M:
                    # 只有真实智能体动作改变联盟表示；终止动作不加入成员。
                    indices = list(env.current_coalition)
                    coalition_state = model.update_coal_state(
                        indices, robot_embeddings.detach(), device).cpu().numpy().astype(np.float16)

            if env.phase == PHASE_TASK_SELECT and phase != PHASE_TASK_SELECT:
                task_context = None
                coalition_state = None

    return env.makespan, actions
