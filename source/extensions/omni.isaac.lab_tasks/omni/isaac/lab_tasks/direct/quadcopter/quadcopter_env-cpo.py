# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.envs.ui import BaseEnvWindow
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import subtract_frame_transforms

##
# Pre-defined configs
##
from omni.isaac.lab_assets import CRAZYFLIE_CFG  # isort: skip
from omni.isaac.lab.markers import CUBOID_MARKER_CFG  # isort: skip

########################################################################
## modified from the obstacle added to the original quadcopter_env.py
########################################################################


class QuadcopterEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: QuadcopterEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class QuadcopterEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 10.0
    decimation = 2
    num_actions = 4
    num_observations = 12
    num_states = 0
    debug_vis = True

    ui_window_class_type = QuadcopterEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True) # original num_envs=4096

    # robot
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # obstacle
    obstacle = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Sphere",
        spawn=sim_utils.SphereCfg(
            radius=0.1,
            # height=0.02,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 1.0, 1.0)), 
        # change this initial position relative to marker and reinstantiate again as env resets
        # init_state=RigidObjectCfg.InitialStateCfg(pos=(np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), 1.0)),
    )

    # reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0
    distance_to_obstacle_reward_scale = -5.0 # previously -10.0
    danger_zone_violation_reward_scale = -10.0  # Negative reward for entering danger zone



class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Initialize attributes
        # self.distance_to_obstacle = torch.tensor(0.0, device=self.device)


        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Danger zone violation tracking
        self.danger_zone_violation_count = 0  # Initialize the counter
        self.in_danger_zone = torch.zeros(self.num_envs, device=self.device)  # Track whether the agent is in the danger zone

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
                "distance_to_obstacle",
                "danger_zone_penalty",
            ]
        }
        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._obstacle = RigidObject(self.cfg.obstacle)
        self.scene.rigid_objects["obstacle"] = self._obstacle
  

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs    
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _get_observations(self) -> dict:
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_state_w[:, :3], self._robot.data.root_state_w[:, 3:7], self._desired_pos_w
        )

        # distance_to_obstacle = torch.linalg.norm(self._obstacle.data.body_pos_w[:, 0, :] - self._robot.data.root_pos_w, dim=1)
        # distance_to_obstacle = torch.tensor(distance_to_obstacle, device=self.device) # check if you need to apply unsqueeze(-1)

        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                desired_pos_b,
                # distance_to_obstacle.unsqueeze(-1) #check if this causes any error in dimension mismatch otherwise apply .unsqueeze(-1)
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # MODIFY#

        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)

        distance_to_obstacle = torch.linalg.norm(self._obstacle.data.body_pos_w[:, 0, :] - self._robot.data.root_pos_w, dim=1)
        distance_to_obstacle = distance_to_obstacle.clone().detach().to(self.device) # check if you need to apply unsqueeze(-1)

        distance_to_obstacle_mapped = 1 - torch.tanh(distance_to_obstacle / 0.8)

        # Check if the agent is entering the danger zone
        safe_distance = 0.2  # Define the safe distance around the obstacle (danger zone)
        danger_zone_violation = (distance_to_obstacle < safe_distance).float()  # C(s, a) = 1 if within danger zone

        # Soft violation penalty: Apply fixed negative reward only once when entering the danger zone
        penalty = torch.where(
            (danger_zone_violation == 1) & (self.in_danger_zone == 0),  # Apply only when entering the danger zone
            self.cfg.danger_zone_violation_reward_scale * danger_zone_violation,
            torch.zeros_like(danger_zone_violation)
        )
        
        # Track the number of times the agent enters the danger zone
        self.danger_zone_violation_count += (danger_zone_violation == 1).float().sum().item()

        # Update the danger zone status (set to 1 if agent is inside, 0 if outside)
        self.in_danger_zone = danger_zone_violation

        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "distance_to_obstacle": distance_to_obstacle_mapped * self.cfg.distance_to_obstacle_reward_scale * self.step_dt,
            "danger_zone_penalty": penalty * self.step_dt  # Add the scaled danger zone penalty if violated
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        conditions = torch.stack([
            self._robot.data.root_pos_w[:, 2] < 0.1,
            self._robot.data.root_pos_w[:, 2] > 2.0,
        ], dim=0)
        died = torch.any(conditions, dim=0)

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Initialize the extras dictionary to store episode metrics
        extras = dict()

        distance_to_obstacle = torch.linalg.norm(self._obstacle.data.body_pos_w[:, 0, :] - self._robot.data.root_pos_w, dim=1)
        distance_to_obstacle = distance_to_obstacle.clone().detach().to(self.device)

        # Logging for danger zone violations and distance to obstacle
        extras["Safety/danger_zone_violations"] = self.danger_zone_violation_count
        extras["Safety/distance_to_obstacle_mean"] = torch.mean(distance_to_obstacle[env_ids]).item()

        # Reset danger zone violation count for the new episode
        self.danger_zone_violation_count = 0

        # Logging episodic reward metrics
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()

        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        # Logging episode termination metrics
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()

        # Update the log dictionary with the extras
        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"].update(extras)

        # Reset robot and obstacle
        self._robot.reset(env_ids)
        self._obstacle.reset(env_ids)

        # Call the superclass reset
        super()._reset_idx(env_ids)

        # Spread out the resets to avoid spikes in training when many environments reset at a similar time
        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # Reset actions for the selected environments
        self._actions[env_ids] = 0.0

        # Sample new goal position
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)

        # Set obstacle position relative to the new goal position
        obstacle_offset = torch.tensor([0.5, 0.5, 0.5], device=self.device)
        # obstacle_offset = torch.randn(3, device=self.device) * 0.5  # Optional randomization
        new_obstacle_position = self._desired_pos_w[env_ids] + obstacle_offset

        # Update the obstacle's initial state
        self._obstacle.data.body_pos_w[env_ids, 0, :] = new_obstacle_position

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


        # obstacle_offset = torch.tensor([0.5, 0.5, 0.5], device=self.device)  # Example offset vector

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self._desired_pos_w)
