"""用论文中的终端奖励与同实例多轨迹基线训练 AHD-RL。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .config import (
    EPS, GROUPS, PHASE_ROBOT_ACCEL, PHASE_TASK_SELECT, TASK_SELECT_FEAT_DIM,
)
from .environment import ConstructionEnv, generate_instance_by_type
from .model import SchedulingNetwork
from .policy import run_episode


def _group_id(value: str) -> int:
    """接收论文组别 G01–G13，也接收对应数值。"""
    label = value.upper()
    if label.startswith("G"):
        label = label[1:]
    if label.isdigit() and int(label) in GROUPS:
        return int(label)
    raise argparse.ArgumentTypeError("group must be G01–G13 or 1–13")


def _stages(groups: Sequence[int]) -> List[List[int]]:
    """按论文的小、中、大规模依次组织所选组别。"""
    ordered = sorted(
        set(groups), key=lambda group: (GROUPS[group][1], GROUPS[group][0])
    )
    stages = (
        [group for group in ordered if GROUPS[group][1] <= 50],
        [group for group in ordered if 50 < GROUPS[group][1] <= 200],
        [group for group in ordered if GROUPS[group][1] > 200],
    )
    return [stage for stage in stages if stage]


def _action_counts(actions: List[Tuple[int, int]]) -> Tuple[int, List[int]]:
    """统计每个任务动作后用于确定联盟的智能体动作数。"""
    counts: List[int] = []
    for phase, _ in actions:
        if phase == PHASE_TASK_SELECT:
            counts.append(0)
        else:
            if not counts:
                raise RuntimeError("agent action appears before a task action")
            counts[-1] += 1
    if not counts or any(count == 0 for count in counts):
        raise RuntimeError("an incomplete task action sequence cannot be trained")
    return len(counts), counts


def _replay_gradient(
    instance: dict,
    model: SchedulingNetwork,
    actions: List[Tuple[int, int]],
    advantage_scale: float,
    device: torch.device,
) -> float:
    """重放采样动作，逐步累积归一化轨迹对数概率的梯度。"""
    task_count, agent_counts = _action_counts(actions)
    env = ConstructionEnv(instance)
    env.reset()
    task_context = None
    coalition_state = None
    action_index = 0
    task_index = -1
    loss_value = 0.0

    while not env.done:
        phase = env.phase
        features = env.get_mask_and_features()
        if phase == PHASE_TASK_SELECT:
            active_ids, task_features, robot_features, mask_np = features
        else:
            task_features, robot_features, mask_np = features

        if not mask_np.any():
            env.step(env.M if phase == PHASE_ROBOT_ACCEL else -1)
            if env.phase == PHASE_TASK_SELECT:
                task_context = None
                coalition_state = None
            continue

        if action_index >= len(actions):
            raise RuntimeError("sampled trajectory ended before environment completion")
        recorded_phase, action = actions[action_index]
        action_index += 1
        if recorded_phase != phase:
            raise RuntimeError("sampled action phase does not match replay")

        task_tensor = torch.from_numpy(task_features).float().to(device)
        robot_tensor = torch.from_numpy(robot_features).float().to(device)
        mask = torch.from_numpy(mask_np).to(device)
        if phase == PHASE_TASK_SELECT:
            task_index += 1
            choice = active_ids.index(action)
            active_array = np.asarray(active_ids, dtype=np.int32)
            contributions = torch.from_numpy(
                instance["robot_task_contrib"][:, active_array].T
            ).to(device)
            depot = np.zeros(TASK_SELECT_FEAT_DIM, dtype=np.float32)
            depot[15:17] = np.asarray(instance["depot"], dtype=np.float32) / 100.0
            logits, embeddings = model.forward_task_select(
                task_tensor, robot_tensor, mask,
                contrib_mask=contributions,
                depot_feat=torch.from_numpy(depot).to(device),
            )
            action_weight = 1.0 / task_count
        else:
            choice = action
            required = instance["task_req"][env.current_task].astype(np.float32)
            gap = np.maximum(
                0.0, required - env.current_coalition_cap.astype(np.float32)
            )
            gap_vector = torch.from_numpy(
                (gap / np.maximum(instance["r_max"], EPS)).astype(np.float32)
            ).to(device)
            logits, embeddings = model.forward_robot_select(
                task_tensor, robot_tensor, mask, phase,
                task_ctx=task_context, gap_vec=gap_vector,
                coal_state=coalition_state,
            )
            action_weight = 1.0 / (task_count * agent_counts[task_index])

        if choice >= len(mask_np) or not mask_np[choice]:
            raise RuntimeError("sampled action is unavailable during replay")
        log_probability = F.log_softmax(logits, dim=-1)[choice]
        step_loss = -advantage_scale * action_weight * log_probability
        step_loss.backward()
        loss_value += float(step_loss.detach())

        if phase == PHASE_TASK_SELECT:
            # 与推理一致，后续动作读取已选任务的固定表示。
            task_context = embeddings[choice].detach()
            coalition_state = None
            env._do_select_task(action, task_ctx=task_context.cpu().numpy())
        else:
            env.step(action)
            if action < env.M and env.phase != PHASE_TASK_SELECT:
                # 联盟表示由本次选择后的成员集合计算，跨动作不保留计算图。
                with torch.no_grad():
                    coalition_state = model.update_coal_state(
                        list(env.current_coalition), embeddings.detach(), device
                    ).to(torch.float16).float()
            if env.phase == PHASE_TASK_SELECT:
                task_context = None
                coalition_state = None

    if action_index != len(actions) or env.num_done != env.N:
        raise RuntimeError("sampled trajectory was not reproduced completely")
    return loss_value


def train(
    groups: Sequence[int] = (1,),
    *,
    epochs_per_stage: int = 1,
    updates_per_epoch: int = 1,
    instances_per_update: int = 4,
    trajectories: int = 8,
    seed: int = 123,
    output: Optional[Path] = None,
    device: Optional[str] = None,
) -> Path:
    """按所选组别训练模型，仅保存最终网络权重。"""
    if not groups or any(group not in GROUPS for group in groups):
        raise ValueError("groups must contain paper group numbers 1–13")
    if min(epochs_per_stage, updates_per_epoch, instances_per_update) < 1:
        raise ValueError("epochs, updates and instances must be positive")
    if trajectories < 2:
        raise ValueError("at least two trajectories are required for the baseline")

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    chosen_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = SchedulingNetwork().to(chosen_device)
    model.eval()  # 采样与梯度重放使用相同的确定性网络前向计算。
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.98)
    scenario_keys = ("1", "2", "3")
    update_number = 0

    for stage in _stages(groups):
        for epoch in range(epochs_per_stage):
            for update in range(updates_per_epoch * len(stage)):
                group = stage[update % len(stage)]
                instances = [
                    generate_instance_by_type(
                        group,
                        scenario_keys[
                            (update_number * instances_per_update + index) % 3
                        ],
                        int(rng.integers(0, 2**31 - 1)),
                    )
                    for index in range(instances_per_update)
                ]
                rewards = torch.empty(instances_per_update, trajectories)
                sampled_actions = []
                for batch_index, instance in enumerate(instances):
                    instance_actions = []
                    for sample_index in range(trajectories):
                        env = ConstructionEnv(instance)
                        makespan, actions = run_episode(env, model, greedy=False)
                        if env.num_done != env.N or not np.isfinite(makespan):
                            raise RuntimeError("training trajectory is incomplete")
                        rewards[batch_index, sample_index] = -makespan
                        instance_actions.append(actions)
                    sampled_actions.append(instance_actions)

                # 每条轨迹与同实例的其他轨迹比较，再按整个更新批次的 RMS 归一化。
                centered = (
                    trajectories * rewards - rewards.sum(dim=1, keepdim=True)
                ) / (trajectories - 1)
                advantages = centered / centered.square().mean().sqrt().clamp_min(EPS)
                optimizer.zero_grad()
                loss = 0.0
                for batch_index, instance in enumerate(instances):
                    for sample_index, actions in enumerate(
                        sampled_actions[batch_index]
                    ):
                        scale = float(advantages[batch_index, sample_index]) / (
                            instances_per_update * trajectories
                        )
                        loss += _replay_gradient(
                            instance, model, actions, scale, chosen_device
                        )
                optimizer.step()
                update_number += 1
                print(
                    "update={} group=G{:02d} reward={:.3f} loss={:.4f}".format(
                        update_number, group, float(rewards.mean()), loss
                    ),
                    flush=True,
                )
            scheduler.step()

    destination = Path(output or "output/ahd_rl.pt")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AHD-RL on generated instances")
    parser.add_argument("--groups", nargs="+", type=_group_id, default=[1])
    parser.add_argument("--epochs-per-stage", type=int, default=1)
    parser.add_argument("--updates-per-epoch", type=int, default=1)
    parser.add_argument("--instances-per-update", type=int, default=4)
    parser.add_argument("--trajectories", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    path = train(
        args.groups,
        epochs_per_stage=args.epochs_per_stage,
        updates_per_epoch=args.updates_per_epoch,
        instances_per_update=args.instances_per_update,
        trajectories=args.trajectories,
        seed=args.seed,
        output=args.output,
        device=args.device,
    )
    print("weights saved to {}".format(path))


if __name__ == "__main__":
    main()
