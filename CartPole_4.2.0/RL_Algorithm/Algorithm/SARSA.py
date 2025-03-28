from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base import BaseAlgorithm, ControlType

class SARSA(BaseAlgorithm):
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
        Initialize the SARSA algorithm.

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
            control_type=ControlType.SARSA,
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
        next_obs_dis,
        reward,
        done
    ):
        """
        Update Q-values using SARSA .

        This method applies the SARSA update rule to improve policy decisions by updating the Q-table.
        """
        if done: # not boostraping
            self.q_values[obs_dis][action_idx] += self.lr * (reward - self.q_values[obs_dis][action_idx])
        else:
            _next_action_idx = self.get_discretize_action(next_obs_dis)
            self.q_values[obs_dis][action_idx] += self.lr * (reward + self.discount_factor * self.q_values[next_obs_dis][_next_action_idx] - self.q_values[obs_dis][action_idx])
        pass