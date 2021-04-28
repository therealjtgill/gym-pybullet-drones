"""
Learning script for shoot and defend. This script is a sandbox for trial/error.

Agents based on ray[rllib]'s implimentation of PPO.


Run:
-----
To run this script, use
    python train_shoot_and_defend-sandbox.py

Notes:
------

"""

# Flag for verbose debug outputs or not
DEBUG = True

# ============== Import statements =================
# Import Ray libraries
import ray
from ray import tune
# Import the ShootAndDefend enviroinment
from gym_pybullet_drones.envs.multi_agent_rl.ShootAndDefend import ShootAndDefend
# ===================================================


def debug_statement(msg):
    if DEBUG:
        print("[Debug] " + msg)
    else:
        pass

def main():
    # Initialize Ray
    debug_statement("Initializing Ray ... ")
    ray.shutdown()
    ray.init(ignore_reinit_error=True)
    debug_statement("... initialized")

    # Register the environment
    debug_statement("Registering ShootAndDefend environment ... ")
    env_name = "shoot_and_defend-v0"
    ray.tune.register_env(env_name, lambda _: ShootAndDefend())
    debug_statement("... registered")


# Driver to call the main program
if __name__ == '__main__':
    main()
