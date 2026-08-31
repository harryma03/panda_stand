"""Print one Panda jump trajectory without opening the viewer."""

import isaacgym  # noqa: F401 -- must be imported before torch
import torch

from legged_gym.envs import *  # noqa: F401,F403 -- registers tasks
from legged_gym.utils import get_args, task_registry


def main(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.env.test = True
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_towards_goal = False
    env_cfg.domain_rand.jump_start_assistance = False
    env_cfg.commands.resampling_time = 1.0e7

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    train_cfg.runner.resume = True
    runner, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = runner.get_inference_policy(device=env.device)
    obs = env.get_observations()
    origin_x = env.root_states[0, 0].item()
    previous_contact = None
    rows = []

    for step in range(int(env.max_episode_length)):
        with torch.no_grad():
            actions = policy(obs.detach())
        obs, _, _, done, _ = env.step(actions.detach())
        contact = (env.contact_forces[0, env.feet_indices, 2] > 5.0).tolist()
        height = env.root_states[0, 2].item()
        descending = env.root_states[0, 9].item() < 0.0
        changed = contact != previous_contact
        important = changed or step % 5 == 0 or (descending and any(contact))
        if important:
            foot_z = env.rigid_state[0, env.feet_indices, 2]
            rows.append(
                (
                    step,
                    env._get_phase()[0].item(),
                    env.root_states[0, 0].item() - origin_x,
                    height,
                    env.root_states[0, 7].item(),
                    env.root_states[0, 9].item(),
                    env.base_euler_xyz[0, 1].item(),
                    env.base_ang_vel[0, 1].item(),
                    "".join("1" if value else "0" for value in contact),
                    env.contact_forces[0, env.feet_indices, 2].amax().item(),
                    foot_z[:2].mean().item(),
                    foot_z[2:].mean().item(),
                )
            )
        previous_contact = contact
        if done[0]:
            break

    print(
        "step phase     x     z   vx_g   vz_g pitch pitch_rate "
        "contact max_fz front_z rear_z"
    )
    for row in rows:
        if row[1] < 0.40:
            continue
        print(
            f"{row[0]:4d} {row[1]:5.2f} {row[2]:5.2f} {row[3]:5.2f} "
            f"{row[4]:6.2f} {row[5]:6.2f} {row[6]:5.2f} {row[7]:10.2f} "
            f"{row[8]:>7s} {row[9]:6.0f} {row[10]:7.3f} {row[11]:6.3f}"
        )


if __name__ == "__main__":
    main(get_args())
