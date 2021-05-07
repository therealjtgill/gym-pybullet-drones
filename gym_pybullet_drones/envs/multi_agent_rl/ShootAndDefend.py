import math
import numpy as np
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE
import pybullet as pb
import pybullet_data

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.envs.multi_agent_rl.BaseMultiagentAviary import BaseMultiagentAviary

class ShootAndDefend(BaseMultiagentAviary):
    """
    Multi-agent RL problem: one agent shoots a ball, the other tries to block it.
    """

    ################################################################################

    def __init__(
        self,
        drone_model: DroneModel=DroneModel.CF2X,
        neighbourhood_radius: float=np.inf,
        freq: int=240,
        aggregate_phy_steps: int=1,
        gui=False,
        record=False, 
        obs: ObservationType=ObservationType.KIN,
        field_dims: np.array=np.array([4, 4, 4]),
        shooter_box_dims: np.array=np.array([1, 4, 4]),
        defender_box_dims: np.array=np.array([1, 4, 4]),
        space_multiplier: float=2
    ):
        """
        Initialization of a multi-agent RL environment.

        Using the generic multi-agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        freq : int, optional
            The frequency (Hz) at which the physics engine steps.
        aggregate_phy_steps : int, optional
            The number of physics steps within one call to `BaseAviary.step()`.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        """

        num_drones = 2

        self.shooter_id = 0
        self.defender_id = 1
        self.ball_id = -1
        self.space_multiplier = space_multiplier

        box_size_increaser = np.array([self.space_multiplier, self.space_multiplier, 1])

        self.defender_box = spaces.Box(
            high=box_size_increaser*np.array([0.75, 2.0, 3]),
            low=box_size_increaser*np.array([-0.75, 0.5, 0.0]),
            dtype=np.float32
        )

        self.shooter_box = spaces.Box(
            high=box_size_increaser*np.array([0.75, -0.5, 3]),
            low=box_size_increaser*np.array([-0.75, -2.0, 0.0]),
            dtype=np.float32
        )

        # Zero attitude initially
        self.rpy_box = spaces.Box(
            high=np.array([0*np.pi/60, 0*np.pi/60, np.pi/2]),
            low=np.array([0*-np.pi/60, 0*-np.pi/60, np.pi/2]),
            dtype=np.float32
        )
        
        self.field_box = spaces.Box(
            high=box_size_increaser*np.array([1.0, 2.0, 3]),
            low=box_size_increaser*np.array([-1.0, -2.0, 0.0]),
            dtype=np.float32
        )

        self.goal_box = spaces.Box(
            high=box_size_increaser*np.array([0.6, 5.0, 2]),
            low=box_size_increaser*np.array([-0.6, 2.0, 0]),
            dtype=np.float32
        )

        self.ball_launched = False
        self.done_funcs = [
            self._shooterOutsideBox,
            self._defenderOutsideBox,
            self._goalScored,
            self._ballOutOfBounds,
            self._ballStationary,
            self._defenderCrashed,
            self._shooterCrashed,
            self._timeExpired,
        ]

        defender_pos = self.defender_box.sample()
        shooter_pos = self.shooter_box.sample()

        super().__init__(
            drone_model=drone_model,
            num_drones=num_drones,
            neighbourhood_radius=neighbourhood_radius,
            initial_xyzs=np.vstack([shooter_pos, defender_pos]),
            initial_rpys=np.vstack([self.rpy_box.sample(), self.rpy_box.sample()]),
            physics=Physics.PYB,
            freq=freq,
            aggregate_phy_steps=aggregate_phy_steps,
            gui=gui,
            record=record,
            obs=obs,
            act=ActionType.RPM
        )

    ################################################################################

    def _actionSpace(self):
        """
        Overridden from the BaseMultiagentAviary class. The defender has the
        standard four actions (one for each motor), but the shooter has four
        motor RPM commands and a single shoot command. The shoot command
        releases a ball from the shooter's grasp with the shooter's velocity at
        release time as the ball's initial velocity.

        Returns
        -------
        dict[int, ndarray]
            A Dict() of Box() of size 1, 3, or 3, depending on the action type,
            indexed by drone Id in integer format.

        """
        defender_action_size = 4
        shooter_action_size = 5
        shooter_action_space = spaces.Box(
            low=np.array([-1, -1, -1, -1, 0]),
            high=np.ones(shooter_action_size),
            dtype=np.float32
        )
        defender_action_space = spaces.Box(
            low=-1*np.ones(defender_action_size),
            high=np.ones(defender_action_size),
            dtype=np.float32
        )
        action_space = spaces.Dict(
            {
                self.shooter_id: shooter_action_space,
                self.defender_id: defender_action_space
            }
        )
        return action_space

    ################################################################################

    def _observationSpace(self):
        """
        Returns the observation space of the environment.

        [obs[0:3], obs[7:10], obs[10:13], obs[13:16]]
        Observations are [pos, rpy, vel, ang_vel] per drone and for the ball.

        Returns
        -------
        dict[int, ndarray]
            A Dict with NUM_DRONES entries indexed by Id in integer format,
            each a Box() is shape (36,) with 12 observed states per entity in
            the environment.
        """
        #### OBS SPACE OF SIZE 12 per entity
        return spaces.Dict(
            {
                i: spaces.Box(
                    low=np.array([-1]*(12 + 12 + 6)),
                    high=np.array([1]*(12 + 12 + 6)),
                    dtype=np.float32
                )
                for i in range(self.NUM_DRONES)
            }
        )

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value(s).

        Returns
        -------
        dict[int, float]
            The reward value for each drone.

        """
        # Smooth reward for shooting ball towards goal, cosine similarity.
        # Replace ball out of bounds penalty for shooter with smooth reward.
        # Reward shooter for collision with ball?
        rewards = {}
        shooter_rewards = \
            0*self._defenderOutsideBox() + \
            0*self._defenderCrashed() + \
            500*self._goalScored() + \
            -10*self._shooterCrashed() + \
            -10*self._shooterOutsideBox() + \
            0*-10*self._ballOutOfBounds() + \
            -5*self._ballStationary() + \
            -8*self._timeExpired() + \
            1e-2*self._shooterAttitudeReward() + \
            1e-2*self._shooterBallDistanceReward()

        defender_rewards = \
            0*self._shooterOutsideBox() + \
            0*self._shooterCrashed() + \
            -500*self._goalScored() + \
            -10*self._defenderCrashed() + \
            -10*self._defenderOutsideBox() + \
            0*10*self._ballOutOfBounds() + \
            5*self._ballStationary() + \
            -8*self._timeExpired() + \
            1e-2*self._defenderAttitudeReward() + \
            1e-2*self._defenderBallDistanceReward()

        rewards[self.shooter_id] = shooter_rewards
        rewards[self.defender_id] = defender_rewards
        return rewards

    ################################################################################
    
    def reset(self):
        defender_pos = self.defender_box.sample()
        shooter_pos = self.shooter_box.sample()
        defender_rpy = self.rpy_box.sample()
        shooter_rpy = self.rpy_box.sample()
        self.ball_launched = False
        self.INIT_XYZS = np.vstack([shooter_pos, defender_pos])
        self.INIT_RPYS = np.vstack([shooter_rpy, defender_rpy])
        super(ShootAndDefend, self).reset()
        return self._computeObs()

    def _preprocessAction(self, actions):
        if actions[self.shooter_id][4] > 0.5:
            self._launchBall()
        movement_actions = {k: v[0:4] for k, v in actions.items()}
        return super()._preprocessAction(movement_actions)

    def _shooterOutsideBox(self):
        ret_val = not self.shooter_box.contains(self.pos[self.shooter_id])
        if ret_val:
            print("Shooter outside box!")
        return ret_val

    def _defenderOutsideBox(self):
        ret_val = not self.defender_box.contains(self.pos[self.defender_id])
        if ret_val:
            print("Defender outside box!")
        return ret_val

    def _goalScored(self):
        ret_val = self.goal_box.contains(self._getBallState()[0:3])
        if ret_val:
            print("Goal scored!")
        return ret_val

    def _ballOutOfBounds(self):
        ball_pos = self._getBallState()[0:3]
        ret_val = not self.field_box.contains(ball_pos) and not self._goalScored()
        # if ret_val:
        #     print("Ball out of bounds!")
        return ret_val

    def _ballStationary(self):
        ball_vel = self._getBallState()[10:13]
        ret_val = (np.linalg.norm(ball_vel) < 1e-6) and self.ball_launched
        # if ret_val:
        #     print("Ball stationary!")
        return ret_val

    def _shooterCrashed(self):
        return self.pos[self.shooter_id][2] <= 1e-6

    def _defenderCrashed(self):
        return self.pos[self.defender_id][2] <= 1e-6

    def _timeExpired(self):
        ret_val = self.step_counter/self.SIM_FREQ > self.EPISODE_LEN_SEC
        # if ret_val:
        #     print("Timer expired!")
        return ret_val

    def _computeAttitudeReward(self, rpy):
        max_roll_pitch = np.pi/6
        reward = 0
        if np.abs(rpy[0]) > max_roll_pitch:
            reward += -1*np.abs(rpy[0])
        else:
            reward += 2*(1 - np.abs(rpy[0])/max_roll_pitch)
        if np.abs(rpy[1]) > max_roll_pitch:
            reward += -1*np.abs(rpy[1])
        else:
            reward += 2*(1 - np.abs(rpy[1])/max_roll_pitch)
        return reward

    def _shooterAttitudeReward(self):
        rpy = self.rpy[self.shooter_id]
        return self._computeAttitudeReward(rpy)

    def _defenderAttitudeReward(self):
        rpy = self.rpy[self.defender_id]
        return self._computeAttitudeReward(rpy)

    def _computeBallDistanceToGoalReward(self):
        ball_pos = self._getBallState()[0:3]
        goal_center_pos = (self.goal_box.high + self.goal_box.low)/2.0
        ball_dist_to_goal = np.linalg.norm(ball_pos - goal_center_pos)
        field_length = self.field_box.high[1] - self.field_box.low[1]
        return ball_dist_to_goal/field_length

    def _shooterBallDistanceReward(self):
        return self._computeBallDistanceToGoalReward()

    def _defenderBallDistanceReward(self):
        return -1.0*self._computeBallDistanceToGoalReward()

    def _computeDone(self):
        """
        Computes the current done value(s).

        Returns
        -------
        dict[int | "__all__", bool]
            Dictionary with the done value of each drone and 
            one additional boolean value for key "__all__".

        """
        dones = [f() for f in self.done_funcs]
        done = {i: any(dones) for i in range(self.NUM_DRONES)}
        done["__all__"] = True if True in done.values() else False
        if done["__all__"]:
            print("Num steps:", self.step_counter)
        return done

    ################################################################################
    
    def _computeInfo(self):
        """
        Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[int, dict[]]
            Dictionary of empty dictionaries.

        """
        return {i: {} for i in range(self.NUM_DRONES)}

    ################################################################################

    def _clipAndNormalizeState(self, state):
        """
        Normalizes a drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.

        """
        MAX_LIN_VEL_XY = 3 
        MAX_LIN_VEL_Z = 1

        MAX_XY = MAX_LIN_VEL_XY*self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z*self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi # Full range

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = state[13:16]/(np.linalg.norm(state[13:16]) + 1e-6)

        norm_and_clipped = np.hstack(
            [
                normalized_pos_xy,
                normalized_pos_z,
                state[3:7],
                normalized_rp,
                normalized_y,
                normalized_vel_xy,
                normalized_vel_z,
                normalized_ang_vel,
                state[16:20]
            ]
        ).reshape(20,)

        return norm_and_clipped
    
    ################################################################################
    
    def _clipAndNormalizeStateWarning(
        self,
        state,
        clipped_pos_xy,
        clipped_pos_z,
        clipped_rp,
        clipped_vel_xy,
        clipped_vel_z,
    ):
        """
        Debugging printouts associated to `_clipAndNormalizeState`.

        Print a warning if values in a state vector is out of the clipping range.
        
        """
        if not(clipped_pos_xy == np.array(state[0:2])).all():
            print("[WARNING] it", self.step_counter, "in FlockAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0], state[1]))
        if not(clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter, "in FlockAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not(clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter, "in FlockAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[7], state[8]))
        if not(clipped_vel_xy == np.array(state[10:12])).all():
            print("[WARNING] it", self.step_counter, "in FlockAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10], state[11]))
        if not(clipped_vel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter, "in FlockAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))

    def _computeObs(self):
        """
        Kinematic states for each drone and the ball.
        """
        obs_12 = np.zeros((self.NUM_DRONES,12))
        # for i in self.DRONE_IDS:
        for i in range(self.NUM_DRONES):
            obs = self._clipAndNormalizeState(self._getDroneStateVector(i))
            obs_12[i, :] = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16]]).reshape(12,)
        obs_ball = self._getBallObs()[[0, 1, 2, 7, 8, 9]]
        # Flatten out the observations made by each drone.
        obs_dict = {
            i: np.concatenate(
                [np.reshape(obs_12, -1), obs_ball]
            ) for i in range(self.NUM_DRONES)
        }
        return obs_dict

    def _getBallState(self):
        ball_state = np.zeros(20)
        if not self.ball_launched:
            shooter_state = self._getDroneStateVector(self.shooter_id)
            R_b2gs = np.array(
                pb.getMatrixFromQuaternion(shooter_state[3:7])
            ).reshape((3,3))
            ball_pos = 1.5*R_b2gs[:, 0]*self.L + shooter_state[0:3]
            ball_state = shooter_state
            ball_state[0:3] = ball_pos
        else:
            ball_pos, ball_quat = pb.getBasePositionAndOrientation(self.ball_id, physicsClientId=self.CLIENT)
            ball_rpy = pb.getEulerFromQuaternion(ball_quat)
            ball_vel, ball_ang_v = pb.getBaseVelocity(self.ball_id, physicsClientId=self.CLIENT)
            ball_state = np.hstack(
                [ball_pos, ball_quat, ball_rpy, ball_vel, ball_ang_v, np.zeros(4)]
            )
        return ball_state

    def _getBallObs(self):
        ball_obs = np.zeros(12)
        ball_state = self._getBallState()
        ball_obs_full = self._clipAndNormalizeState(ball_state)
        ball_obs = np.hstack(
            [ball_obs_full[0:3], ball_obs_full[7:10], np.zeros(3), np.zeros(3)]
        ).reshape(12,)
        return ball_obs

    def _launchBall(self):
        if self.ball_launched:
            return
        # print("Ball launched!!")
        shooter_state = self._getDroneStateVector(self.shooter_id)
        R_b2gs = np.array(
            pb.getMatrixFromQuaternion(shooter_state[3:7])
        ).reshape((3,3))
        ball_pos = 1.5*R_b2gs[:, 0]*self.L + shooter_state[0:3]
        self.ball_id = pb.loadURDF("sphere2.urdf",
            ball_pos,
            pb.getQuaternionFromEuler([0,0,0]),
            globalScaling=0.0625,
            physicsClientId=self.CLIENT
        )
        pb.changeDynamics(self.ball_id, -1, mass=0.1)
        # shooter_vel, _ = pb.getBaseVelocity(
        #     self.shooter_id,
        #     physicsClientId=self.CLIENT
        # )
        # # print("Shooter vel:", shooter_vel)
        # shooter_vel = np.array(shooter_vel)
        # ball_force_unit = shooter_vel/(np.linalg.norm(shooter_vel) + 1e-6)
        ball_force = self.space_multiplier*100*R_b2gs[:, 0]
        # print("Force on ball:", ball_force)

        pb.applyExternalForce(
            self.ball_id,
            -1,
            forceObj=ball_force.tolist(),
            posObj=[0, 0, 0],
            flags=pb.LINK_FRAME,
            physicsClientId=self.CLIENT
        )
        self.ball_launched = True
