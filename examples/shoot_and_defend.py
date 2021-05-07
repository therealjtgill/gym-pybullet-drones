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
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.envs.multi_agent_rl.ShootAndDefend import ShootAndDefend
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

def select_policy(agent_id):
    if agent_id == 0:
        return "defender"
    else:
        return "shooter"


if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary or VisionAviary and DSLPIDControl')
    parser.add_argument('--drone',              default="cf2x",     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--physics',            default="pyb",      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--vision',             default=False,      type=str2bool,      help='Whether to use VisionAviary (default: False)', metavar='')
    parser.add_argument('--gui',                default=True,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=False,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=True,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--aggregate',          default=True,       type=str2bool,      help='Whether to aggregate physics steps (default: False)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=240,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=100,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--checkpoint',         required=False,                         help='Path to ray checkpoint that can be re-loaded.')
    ARGS = parser.parse_args()

    #### Initialize the simulation #############################
    H = .1
    H_STEP = .05
    R = .3
    num_drones = 2
    INIT_XYZS = np.array([[R*np.cos((i/6)*2*np.pi+np.pi/2), R*np.sin((i/6)*2*np.pi+np.pi/2)-R, H+i*H_STEP] for i in range(num_drones)])
    INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/num_drones] for i in range(num_drones)])
    AGGR_PHY_STEPS = int(ARGS.simulation_freq_hz/ARGS.control_freq_hz) if ARGS.aggregate else 1

    #### Initialize a circular trajectory ######################
    PERIOD = 10
    NUM_WP = ARGS.control_freq_hz*PERIOD
    TARGET_POS = np.zeros((NUM_WP,3))
    for i in range(NUM_WP):
        TARGET_POS[i, :] = R*np.cos((i/NUM_WP)*(2*np.pi)+np.pi/2)+INIT_XYZS[0, 0], R*np.sin((i/NUM_WP)*(2*np.pi)+np.pi/2)-R+INIT_XYZS[0, 1], 0
    wp_counters = np.array([int((i*NUM_WP/6)%NUM_WP) for i in range(num_drones)])

    #### Create the environment with or without video capture ##
    env = ShootAndDefend(
        drone_model=ARGS.drone,
        neighbourhood_radius=10,
        freq=ARGS.simulation_freq_hz,
        aggregate_phy_steps=AGGR_PHY_STEPS,
        gui=ARGS.gui,
        record=ARGS.record_video,
    )

    obs_space = env.observation_space
    action_space = env.action_space

    if ARGS.checkpoint is not None:
        ray.init(local_mode=False)
        register_env("shoot-and-defend-v0", lambda _: ShootAndDefend())
        config = ppo.DEFAULT_CONFIG.copy()
        config["framework"] = "torch"
        config["env"] = "shoot-and-defend-v0"
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
                            "fcnet_activation": "relu"
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
                            "fcnet_activation": "relu"
                        }
                    }
                )
            },
            "policy_mapping_fn": select_policy
        }
        agent = ppo.PPOTrainer(config, env="shoot-and-defend-v0")
        agent.restore(ARGS.checkpoint)
        # print(agent.get_policy("shooter").model)
        # print(agent.get_policy("defender").model)
        defender_policy = agent.get_policy("defender")
        shooter_policy = agent.get_policy("shooter")

    # import pdb
    # pdb.set_trace()

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=int(ARGS.simulation_freq_hz/AGGR_PHY_STEPS),
                    num_drones=num_drones
                    )

    #### Initialize the controllers ############################
    if ARGS.drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [DSLPIDControl(drone_model=ARGS.drone) for i in range(num_drones)]
    elif ARGS.drone in [DroneModel.HB]:
        ctrl = [SimplePIDControl(drone_model=ARGS.drone) for i in range(num_drones)]

    #### Run the simulation ####################################
    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ/ARGS.control_freq_hz))
    # Action of [0, 0, 0, 0] puts the drone into a hover position
    action = {
        0: np.array([0,0,0,0,0]),
        1: np.array([0,0,0,0]),
    }
    done = {"__all__": False}
    START = time.time()
    obs = env.reset()
    # for i in range(0, int(ARGS.duration_sec*env.SIM_FREQ), AGGR_PHY_STEPS):
    while not done["__all__"]:

        #### Make it rain rubber ducks #############################
        # if i/env.SIM_FREQ>5 and i%10==0 and i/env.SIM_FREQ<10: p.loadURDF("duck_vhacd.urdf", [0+random.gauss(0, 0.3),-0.5+random.gauss(0, 0.3),3], p.getQuaternionFromEuler([random.randint(0,360),random.randint(0,360),random.randint(0,360)]), physicsClientId=PYB_CLIENT)
        if ARGS.checkpoint is not None:
            action = {
                env.shooter_id: shooter_policy.compute_single_action(obs[env.shooter_id])[0],
                env.defender_id: defender_policy.compute_single_action(obs[env.defender_id])[0]
            }
        #### Step the simulation ###################################
        obs, reward, done, info = env.step(action)

        # print("Obs out of environment:", obs)
        if i >= 250:
            action[0][4] = 1
        #### Compute control at the desired frequency ##############
        if i % CTRL_EVERY_N_STEPS == 0:
            #### Compute control for the current way point #############
            for j in enumerate(env.DRONE_IDS):
                pass

        #### Printout ##############################################
        if i%env.SIM_FREQ == 0:
            env.render()
            #### Print matrices with the images captured by each drone #
            if ARGS.vision:
                for j in range(num_drones):
                    print(obs[str(j)]["rgb"].shape, np.average(obs[str(j)]["rgb"]),
                          obs[str(j)]["dep"].shape, np.average(obs[str(j)]["dep"]),
                          obs[str(j)]["seg"].shape, np.average(obs[str(j)]["seg"])
                          )

        #### Sync the simulation ###################################
        if ARGS.gui:
            # sync(i, START, env.TIMESTEP)
            time.sleep(12/240)

        if done["__all__"]:
            print(done)
            print("Sim terminated")
            break

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    # logger.save()
    # logger.save_as_csv("pid") # Optional CSV save

    #### Plot the simulation results ###########################
    # if ARGS.plot:
    #     logger.plot()
