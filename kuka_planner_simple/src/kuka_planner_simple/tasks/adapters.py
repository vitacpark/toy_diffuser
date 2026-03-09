from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x


def to_torch(x, dtype=None, device=None):
    dtype = dtype or torch.float32
    device = device or torch.device("cpu")
    if isinstance(x, dict):
        return {k: to_torch(v, dtype=dtype, device=device) for k, v in x.items()}
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, dtype=dtype, device=device)


@dataclass
class StageContext:
    cond_idx: int
    stack: int
    place: np.ndarray | int | None = None
    global_place: np.ndarray | None = None
    local_place: np.ndarray | None = None


class BaseTaskAdapter:
    task_name: str = "base"

    def __init__(self, device: torch.device, horizon: int):
        self.device = device
        self.horizon = int(horizon)
        self.future_start = max(0, self.horizon // 2)
        self._mins_t: Optional[torch.Tensor] = None
        self._range_t: Optional[torch.Tensor] = None
        self._obs_dim: Optional[int] = None
        self._contact_indices = np.asarray([14 + 8 * j for j in range(4)], dtype=np.int64)

    def bind_dataset(self, dataset):
        mins = torch.as_tensor(dataset.mins, device=self.device, dtype=torch.float32)
        maxs = torch.as_tensor(dataset.maxs, device=self.device, dtype=torch.float32)
        self._mins_t = mins
        self._range_t = maxs - mins
        self._obs_dim = int(mins.shape[0])

    def _project_state(self, state: np.ndarray) -> np.ndarray:
        if self._obs_dim is None:
            raise ValueError("Dataset stats are not bound. Call bind_dataset(dataset) first.")
        if state.shape[-1] == self._obs_dim:
            return state
        if state.shape[-1] > self._obs_dim:
            return state[..., : self._obs_dim]
        raise ValueError(
            f"State dimension {state.shape[-1]} is smaller than expected obs_dim {self._obs_dim}."
        )

    def make_env(self, save_render: bool):
        raise NotImplementedError

    def reset_env(self, env, seed: int):
        raise NotImplementedError

    def num_stages(self) -> int:
        raise NotImplementedError

    def get_stage_context(self, env) -> StageContext:
        raise NotImplementedError

    def guided_sample(self, ema_model, guide, batch_size: int, conditions, ctx: StageContext, pg: bool, pg_scale: float, guide_step: int):
        raise NotImplementedError

    def fast_guided_sample(self, ema_model, guide, batch_size: int, sub_conditions, samples, ctx: StageContext, diffusion_step: int):
        raise NotImplementedError

    def compute_values(self, samples: torch.Tensor, dataset, ctx: StageContext) -> np.ndarray:
        raise NotImplementedError

    def choose_subtree(self, main_values: np.ndarray, sub_values: np.ndarray) -> Tuple[int, bool]:
        raise NotImplementedError

    def choose_main(self, main_values: np.ndarray) -> int:
        return int(np.argmax(main_values))

    def execute(self, samples: np.ndarray, env, save_render: bool):
        raise NotImplementedError

    def advance_after_stage(self, env):
        return None

    def normalize_state(self, state: np.ndarray, dataset=None) -> torch.Tensor:
        if self._mins_t is None or self._range_t is None:
            if dataset is None:
                raise ValueError("Dataset stats are not bound. Call bind_dataset(dataset) first.")
            self.bind_dataset(dataset)
        state = self._project_state(state)
        x = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        x = (x - self._mins_t) / (self._range_t + 1e-8)
        x = x[None, None, None]
        return (x - 0.5) * 2

    def unnormalize_samples(self, samples: torch.Tensor, dataset=None) -> torch.Tensor:
        if self._mins_t is None or self._range_t is None:
            if dataset is None:
                raise ValueError("Dataset stats are not bound. Call bind_dataset(dataset) first.")
            self.bind_dataset(dataset)
        samples = torch.clamp(samples, -1, 1)
        samples_unscale = (samples + 1) * 0.5
        return samples_unscale * self._range_t + self._mins_t

    def make_sub_plan_index(self, batch_size: int, sample_jump_size: int, sample_ub: int) -> torch.Tensor:
        possible_indices = torch.arange(sample_jump_size, sample_ub, step=sample_jump_size, dtype=torch.long)
        if possible_indices.numel() == 0:
            possible_indices = torch.tensor([sample_jump_size], dtype=torch.long)
        picks = min(8, int(possible_indices.numel()))
        chosen = possible_indices[torch.randperm(possible_indices.numel())[:picks]]
        chosen, _ = torch.sort(chosen)
        repeat_factor = int(math.ceil(batch_size / max(1, picks)))
        return chosen.repeat(repeat_factor)[:batch_size]

    def _contact_vec(self, sample: np.ndarray) -> np.ndarray:
        return sample[self._contact_indices]

    def _state_joint_contact_vec(self, state: np.ndarray) -> np.ndarray:
        return np.concatenate([state[:7], state[self._contact_indices]], dtype=np.float32)


class ConditionalStackAdapter(BaseTaskAdapter):
    task_name = "conditional_stack"

    def __init__(self, device: torch.device, horizon: int):
        super().__init__(device=device, horizon=horizon)
        counter = 0
        self.map_tuple: Dict[Tuple[int, int], int] = {}
        for i in range(4):
            for j in range(4):
                if i == j:
                    continue
                self.map_tuple[(i, j)] = counter
                counter += 1

    def make_env(self, save_render: bool):
        from gym_stacking.env import StackEnv

        return StackEnv(conditional=True)

    def reset_env(self, env, seed: int):
        from gym_stacking.utils import set_client

        set_client(env.client)
        return env.reset(seed=seed)

    def num_stages(self) -> int:
        return 3

    def get_stage_context(self, env) -> StageContext:
        stack = int(env.goal[env.progress + 1])
        place = int(env.goal[env.progress])
        cond_idx = self.map_tuple[(stack, place)]
        return StageContext(cond_idx=cond_idx, stack=stack, place=place)

    def guided_sample(self, ema_model, guide, batch_size: int, conditions, ctx: StageContext, pg: bool, pg_scale: float, guide_step: int):
        return ema_model.guided_conditional_sample(
            guide,
            batch_size,
            conditions,
            ctx.cond_idx,
            ctx.stack,
            int(ctx.place),
            pg=pg,
            pg_scale=pg_scale,
            guide_step=guide_step,
        )

    def fast_guided_sample(self, ema_model, guide, batch_size: int, sub_conditions, samples, ctx: StageContext, diffusion_step: int):
        return ema_model.fast_guided_conditional_sample(
            guide,
            batch_size,
            sub_conditions,
            samples,
            ctx.cond_idx,
            ctx.stack,
            int(ctx.place),
            diffusion_step,
        )

    def compute_values(self, samples: torch.Tensor, dataset, ctx: StageContext) -> np.ndarray:
        unnorm = self.unnormalize_samples(samples, dataset)
        stack_xy = unnorm[..., self.future_start :, 7 + ctx.stack * 8 : 9 + ctx.stack * 8]
        place_idx = int(ctx.place)
        place_xy = unnorm[..., self.future_start :, 7 + place_idx * 8 : 9 + place_idx * 8]
        values = -torch.abs(stack_xy - place_xy).mean(dim=-1).mean(dim=-1)
        return to_np(values)

    def choose_subtree(self, main_values: np.ndarray, sub_values: np.ndarray) -> Tuple[int, bool]:
        best_idx = int(np.argmax(main_values))
        best_sub_idx = int(np.argmax(sub_values))
        is_sub_best = bool(main_values[best_idx] < sub_values[best_sub_idx])
        if is_sub_best:
            best_idx = best_sub_idx
        return best_idx, is_sub_best

    def execute(self, samples: np.ndarray, env, save_render: bool):
        import pybullet as p
        from gym_stacking.env import get_env_state
        from gym_stacking.utils import (
            add_fixed_constraint,
            get_movable_joints,
            remove_fixed_constraint,
            set_client,
            set_velocity,
        )

        set_client(env.client)

        robot = env.robot
        joints = get_movable_joints(robot)
        gains = np.ones(len(joints))
        cubes = env.cubes
        link = 8

        near = 0.001
        far = 4.0
        projection_matrix = p.computeProjectionMatrixFOV(60.0, 1.0, near, far)
        location = np.array([0.8, 1.5, 2.4])
        end = np.array([0.0, 0.0, 0.0])
        view_matrix = p.computeViewMatrix(location, end, [0, 0, 1])

        states = [get_env_state(robot, cubes, env.attachments)]
        rewards = 0.0
        frames: List[np.ndarray] = []

        for sample in samples[1:]:
            p.setJointMotorControlArray(
                bodyIndex=robot,
                jointIndices=joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=sample[:7],
                positionGains=gains,
            )

            for j in range(4):
                contact = sample[14 + j * 8]
                if contact > 0.5:
                    add_fixed_constraint(cubes[j], robot, link)
                    env.attachments[j] = 1
                else:
                    remove_fixed_constraint(cubes[j], robot, link)
                    set_velocity(cubes[j], linear=[0, 0, 0], angular=[0, 0, 0, 0])
                    env.attachments[j] = 0

            for _ in range(10):
                p.stepSimulation()

            states.append(get_env_state(robot, cubes, env.attachments))
            if save_render:
                _, _, im, _, _ = p.getCameraImage(
                    width=1024,
                    height=1024,
                    viewMatrix=view_matrix,
                    projectionMatrix=projection_matrix,
                )
                frames.append(np.array(im).reshape((1024, 1024, 4))[:, :, :3])

            env.get_state()
            rewards += float(env.compute_reward())

        env.attachments[:] = 0
        env.get_state()
        rewards += float(env.compute_reward())
        state = get_env_state(robot, cubes, env.attachments)
        return state, states, frames, rewards


class Pick2PutAdapter(BaseTaskAdapter):
    task_name = "pick2put"

    def make_env(self, save_render: bool):
        from gym_stacking.pick_env import PickandPutEnv

        return PickandPutEnv(conditional=True, save_render=save_render)

    def reset_env(self, env, seed: int):
        from gym_stacking.utils import set_client

        set_client(env.client)
        np.random.seed(seed)
        return env.reset()

    def num_stages(self) -> int:
        return 4

    def get_stage_context(self, env) -> StageContext:
        stack = int(env.goal[env.progress])
        place = np.asarray(env.put_place[stack][:2], dtype=np.float32)
        return StageContext(cond_idx=0, stack=stack, place=place)

    def guided_sample(self, ema_model, guide, batch_size: int, conditions, ctx: StageContext, pg: bool, pg_scale: float, guide_step: int):
        return ema_model.guided_conditional_sample(
            guide,
            batch_size,
            conditions,
            ctx.cond_idx,
            ctx.stack,
            ctx.place,
            pg=pg,
            pg_scale=pg_scale,
            guide_step=guide_step,
        )

    def fast_guided_sample(self, ema_model, guide, batch_size: int, sub_conditions, samples, ctx: StageContext, diffusion_step: int):
        return ema_model.fast_guided_conditional_sample(
            guide,
            batch_size,
            sub_conditions,
            samples,
            ctx.cond_idx,
            ctx.stack,
            ctx.place,
            diffusion_step,
        )

    def compute_values(self, samples: torch.Tensor, dataset, ctx: StageContext) -> np.ndarray:
        unnorm = to_np(self.unnormalize_samples(samples, dataset))
        place_xy = np.repeat(ctx.place.reshape((1, 1, 2)), self.horizon - self.future_start, axis=1)
        return -np.abs(unnorm[..., self.future_start :, 7 + ctx.stack * 8 : 9 + ctx.stack * 8] - place_xy).mean(axis=2).mean(axis=1)

    def choose_subtree(self, main_values: np.ndarray, sub_values: np.ndarray) -> Tuple[int, bool]:
        merged = main_values + sub_values
        best_idx = int(np.argmax(merged))
        is_sub_best = bool(main_values[best_idx] < sub_values[best_idx])
        return best_idx, is_sub_best

    def execute(self, samples: np.ndarray, env, save_render: bool):
        from gym_stacking.pick_env import get_env_state
        from gym_stacking.utils import set_client

        set_client(env.client)

        states = [get_env_state(env.robot, env.cubes, env.attachments)]
        rewards = 0.0
        frames: List[np.ndarray] = []
        dists: List[float] = []

        for ind, sample in enumerate(samples[1:]):
            joint_pos = sample[:7]
            action = np.concatenate([joint_pos, self._contact_vec(sample)], dtype=np.float32)

            _, reward, _, _ = env.step(action)
            if save_render and env.save_render:
                frames.append(env.render())

            states.append(get_env_state(env.robot, env.cubes, env.attachments))

            if ind < samples.shape[0] - 2:
                actual_vec = self._state_joint_contact_vec(states[-1])
                pred_vec = self._state_joint_contact_vec(samples[ind + 2])
                dists.append(float(np.linalg.norm(actual_vec - pred_vec)))

            rewards += float(reward)

        env.attachments[:] = 0
        env.get_state()
        rewards += float(env.compute_reward())
        state = get_env_state(env.robot, env.cubes, env.attachments)

        if dists and max(dists) > 1.02:
            rewards = 0.0

        return state, states, frames, rewards

    def advance_after_stage(self, env):
        env.progress += 1


class PnwpAdapter(Pick2PutAdapter):
    task_name = "pnwp"

    def make_env(self, save_render: bool):
        from gym_stacking.pick_env_pnwp import PickandPutEnv

        return PickandPutEnv(conditional=True, save_render=save_render)

    def get_stage_context(self, env) -> StageContext:
        stack = int(env.goal[env.progress])
        global_place = np.asarray(env.global_put_place[stack][:2], dtype=np.float32)
        local_place = np.asarray(env.local_put_place[stack][:2], dtype=np.float32)
        return StageContext(cond_idx=0, stack=stack, global_place=global_place, local_place=local_place)

    def guided_sample(self, ema_model, guide, batch_size: int, conditions, ctx: StageContext, pg: bool, pg_scale: float, guide_step: int):
        return ema_model.guided_conditional_sample(
            guide,
            batch_size,
            conditions,
            ctx.cond_idx,
            ctx.stack,
            ctx.global_place,
            ctx.local_place,
            pg=pg,
            pg_scale=pg_scale,
            guide_step=guide_step,
        )

    def fast_guided_sample(self, ema_model, guide, batch_size: int, sub_conditions, samples, ctx: StageContext, diffusion_step: int):
        return ema_model.fast_guided_conditional_sample(
            guide,
            batch_size,
            sub_conditions,
            samples,
            ctx.cond_idx,
            ctx.stack,
            ctx.global_place,
            ctx.local_place,
            diffusion_step,
        )

    def compute_values(self, samples: torch.Tensor, dataset, ctx: StageContext) -> np.ndarray:
        unnorm = to_np(self.unnormalize_samples(samples, dataset))
        tail = self.horizon - self.future_start
        local_xy = np.repeat(ctx.local_place.reshape((1, 1, 2)), tail, axis=1)
        global_xy = np.repeat(ctx.global_place.reshape((1, 1, 2)), tail, axis=1)
        mid_place = np.array(
            [
                (ctx.local_place[0] + 3.0 * ctx.global_place[0]) / 4.0,
                (ctx.local_place[1] + 3.0 * ctx.global_place[1]) / 4.0,
            ],
            dtype=np.float32,
        )
        mid_xy = np.repeat(mid_place.reshape((1, 1, 2)), tail, axis=1)

        stack_xy = unnorm[..., self.future_start :, 7 + ctx.stack * 8 : 9 + ctx.stack * 8]
        values = -(
            np.abs(stack_xy - local_xy).mean(axis=2).mean(axis=1)
            - 1.5 * np.abs(stack_xy - mid_xy).mean(axis=2).mean(axis=1)
            + 2.0 * np.abs(stack_xy - global_xy).mean(axis=2).mean(axis=1)
        )
        return values

    def choose_subtree(self, main_values: np.ndarray, sub_values: np.ndarray) -> Tuple[int, bool]:
        # Match the original tdp-kuka pnwp logic:
        # compare best main vs best sub independently.
        best_idx = int(np.argmax(main_values))
        best_sub_idx = int(np.argmax(sub_values))
        is_sub_best = bool(main_values[best_idx] < sub_values[best_sub_idx])
        if is_sub_best:
            best_idx = best_sub_idx
        return best_idx, is_sub_best

    def execute(self, samples: np.ndarray, env, save_render: bool):
        import pybullet as p
        from gym_stacking.pick_env_pnwp import get_env_state
        from gym_stacking.utils import get_movable_joints, set_client

        set_client(env.client)

        states = [get_env_state(env.robot, env.cubes, env.attachments)]
        rewards = 0.0
        frames: List[np.ndarray] = []
        joints = get_movable_joints(env.robot)
        gains = np.ones(len(joints))

        near = 0.001
        far = 4.0
        projection_matrix = p.computeProjectionMatrixFOV(60.0, 1.0, near, far)
        location = np.array([0.8, 1.5, 2.4])
        end = np.array([0.0, 0.0, 0.0])
        view_matrix = p.computeViewMatrix(location, end, [0, 0, 1])

        for sample in samples[1:]:
            p.setJointMotorControlArray(
                bodyIndex=env.robot,
                jointIndices=joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=sample[:7],
                positionGains=gains,
            )

            action = np.concatenate([sample[:7], self._contact_vec(sample)], dtype=np.float32)
            _, reward, _, _ = env.step(action)

            if save_render and env.save_render:
                _, _, im, _, _ = p.getCameraImage(
                    width=1024,
                    height=1024,
                    viewMatrix=view_matrix,
                    projectionMatrix=projection_matrix,
                )
                frames.append(np.array(im).reshape((1024, 1024, 4))[:, :, :3])

            states.append(get_env_state(env.robot, env.cubes, env.attachments))
            rewards += float(reward)

        env.attachments[:] = 0
        env.get_state()
        rewards += float(env.compute_reward())
        state = get_env_state(env.robot, env.cubes, env.attachments)

        return state, states, frames, rewards


def make_task_adapter(task_name: str, device: torch.device, horizon: int) -> BaseTaskAdapter:
    name = str(task_name)
    adapter_cls = TASK_ADAPTERS.get(name)
    if adapter_cls is None:
        raise ValueError(f"Unsupported task `{task_name}`. Supported={supported_task_names()}")
    return adapter_cls(device=device, horizon=horizon)


TASK_ADAPTERS = {
    "conditional_stack": ConditionalStackAdapter,
    "pick2put": Pick2PutAdapter,
    "pnwp": PnwpAdapter,
}


def supported_task_names() -> List[str]:
    return list(TASK_ADAPTERS.keys())
