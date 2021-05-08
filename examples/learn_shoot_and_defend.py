"""Script demonstrating the joint use of simulation and control.

The simulation is run by a `CtrlAviary` or `VisionAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Example
-------
In a terminal, run as:

    $ python fly.py

Notes
-----
The drones move, at different altitudes, along cicular trajectories 
in the X-Y plane, around point (0, -.3).

"""
import os
import time
import argparse
from datetime import datetime
import gym
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import ray
from ray.tune import register_env
from ray.rllib.agents import ppo

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.multi_agent_rl.ShootAndDefend import ShootAndDefend
from gym_pybullet_drones.utils.utils import sync, str2bool

def select_policy(agent_id):
    if agent_id == 1:
        return "defender"
    else:
        return "shooter"

if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary or VisionAviary and DSLPIDControl')
    parser.add_argument('--drone',              default="cf2x",     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--physics',            default="pyb",      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui',                action="store_true",      help='Whether to use PyBullet GUI (default: True)')
    parser.add_argument('--record_video',       default=False,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--aggregate',          default=True,       type=str2bool,      help='Whether to aggregate physics steps (default: False)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=240,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=100,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--checkpoint',         required=False,                         help='Path to ray checkpoint that can be re-loaded.')
    parser.add_argument('--lstm',               default=False,                         help='Use an LSTM? (default: False)')
    parser.add_argument('--num_workers', default=16, required=False, help='The number of workers ray should use to perform rollouts. Defaults to 16.')
    ARGS = parser.parse_args()

    #### Initialize the simulation #############################
    R = .3
    num_drones = 2
    AGGR_PHY_STEPS = int(ARGS.simulation_freq_hz/ARGS.control_freq_hz) if ARGS.aggregate else 1

    # env = gym.make("shoot-and-defend-v0")
    # # env = gym.make("meetup-aviary-v0")
    # print("[INFO] Action space:", env.action_space)
    # print("[INFO] Observation space:", env.observation_space)
    # # check_env(env,
    # #           warn=True,
    # #           skip_render_check=True
    # # )

    ### Create the environment with or without video capture ##
    env = ShootAndDefend(
        drone_model=ARGS.drone,
        neighbourhood_radius=10,
        freq=ARGS.simulation_freq_hz,
        aggregate_phy_steps=AGGR_PHY_STEPS,
        gui=ARGS.gui,
        record=ARGS.record_video,
    )

    ray.init(local_mode=False)
    obs_space = env.observation_space
    action_space = env.action_space
    register_env("shoot-and-defend-v0", lambda _: ShootAndDefend())
    config = ppo.DEFAULT_CONFIG.copy()
    config["num_workers"] = args.num_workers
    config["framework"] = "torch"
    config["env"] = "shoot-and-defend-v0"
    config["train_batch_size"] = 28800
    config["sgd_minibatch_size"] = 28800//8
    config["lr"] = 1e-4
    config["gamma"] = 0.9995
    config["num_sgd_iter"] = 10
    config["multiagent"] = {
        "policies_to_train": ["shooter", "defender"],
        "policies": {
            "shooter": (
                None,
                obs_space[env.shooter_id],
                action_space[env.shooter_id],
                {
                    "model": {
                        "fcnet_hiddens": [128, 64],
                        "fcnet_activation": "relu",
                        "use_lstm" : ARGS.lstm
                    }
                }
            ),
            "defender": (
                None,
                obs_space[env.defender_id],
                action_space[env.defender_id],
                {
                    "model": {
                        "fcnet_hiddens": [128, 64],
                        "fcnet_activation": "relu",
                        "use_lstm" : ARGS.lstm
                    }
                }
            )
        },
        "policy_mapping_fn": select_policy
    }
    if ARGS.lstm:
    	print("Using LSTM!!")
    	
    print(config)
    
    agent = ppo.PPOTrainer(config)
    if ARGS.checkpoint:
        agent.restore(ARGS.checkpoint)

    for i in range(10000):
        print("Iteration number:", i)
        results = agent.train()
        print("[INFO] {:d}: episode_reward max {:f} min {:f} mean {:f}".format(
                i,
                results["episode_reward_max"],
                results["episode_reward_min"],
                results["episode_reward_mean"]
            )
        )

        if i % 10 == 0:
            checkpoint = agent.save()
            print("Checkpoint saved at:", checkpoint)

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Run the simulation ####################################
    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ/ARGS.control_freq_hz))
    # Action of [0, 0, 0, 0] puts the drone into a hover position
    action = {
        0: np.array([0,0,0,0]),
        1: np.array([0,0,0,0, 0]),
    }
    done = {"__all__": False}
    START = time.time()
    i = 0
    # for i in range(0, int(ARGS.duration_sec*env.SIM_FREQ), AGGR_PHY_STEPS):
    while not done["__all__"]:

        #### Make it rain rubber ducks #############################
        # if i/env.SIM_FREQ>5 and i%10==0 and i/env.SIM_FREQ<10: p.loadURDF("duck_vhacd.urdf", [0+random.gauss(0, 0.3),-0.5+random.gauss(0, 0.3),3], p.getQuaternionFromEuler([random.randint(0,360),random.randint(0,360),random.randint(0,360)]), physicsClientId=PYB_CLIENT)

        #### Step the simulation ###################################
        obs, reward, done, info = env.step(action)

        # print("Obs out of environment:", obs)
        #### Compute control at the desired frequency ##############
        if i % CTRL_EVERY_N_STEPS == 0:
            #### Compute control for the current way point #############
            for j in enumerate(env.DRONE_IDS):
                pass

        #### Printout ##############################################
        if i%env.SIM_FREQ == 0:
            env.render()
            #### Print matrices with the images captured by each drone #

        #### Sync the simulation ###################################
        if ARGS.gui:
            # sync(i, START, env.TIMESTEP)
            # time.sleep(env.TIMESTEP)
            pass

        if done["__all__"]:
            print(done)
            print("Sim terminated")
            break

        i += 1

    #### Close the environment #################################
    env.close()
