from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base import BaseAlgorithm, ControlType

class MC(BaseAlgorithm):
    def __init__(
            self,
            num_of_action: int,
            action_range: list,
            discretize_state_weight: list,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float,
    ) -> None:
        """
        Initialize the Monte Carlo algorithm.

        Args:
            num_of_action (int): Number of possible actions.
            action_range (list): Scaling factor for actions.
            discretize_state_weight (list): Scaling factor for discretizing states.
            learning_rate (float): Learning rate for Q-value updates.
            initial_epsilon (float): Initial value for epsilon in epsilon-greedy policy.
            epsilon_decay (float): Rate at which epsilon decays.
            final_epsilon (float): Minimum value for epsilon.
            discount_factor (float): Discount factor for future rewards.
        """
        super().__init__(
            control_type=ControlType.MONTE_CARLO,
            num_of_action=num_of_action,
            action_range=action_range,
            discretize_state_weight=discretize_state_weight,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
        )
        
    def update(
        self,
        obs_dis,
        action_idx,
        reward,
        done
    ):
        """
        Update Q-values using Monte Carlo.

        This method applies the Monte Carlo update rule to improve policy decisions by updating the Q-table.
        
        Args:
            obs_dis (tuple): Discretized observation.
            action_idx (int): action index [action in discrete].
            reward (float): reward value.
            done (bool): episode termination flag.
        """
        
        self.obs_hist.append(obs_dis)
        self.action_hist.append(action_idx)
        self.reward_hist.append(reward)
        
        if done:
            G_cum = 0 # return
            for t in reversed(range(len(self.obs_hist))):
                G_cum = self.discount_factor * G_cum + self.reward_hist[t]
                # print(G_cum)
                if (self.obs_hist[t], self.action_hist[t]) not in list(zip(self.obs_hist[:t], self.action_hist[:t])):   # if First Visit
                    self.n_values[self.obs_hist[t]][self.action_hist[t]] += 1
                    self.q_values[self.obs_hist[t]][self.action_hist[t]] += (G_cum - self.q_values[self.obs_hist[t]][self.action_hist[t]]) / self.n_values[self.obs_hist[t]][self.action_hist[t]]
            
            self.obs_hist.clear()
            self.action_hist.clear()
            self.reward_hist.clear()