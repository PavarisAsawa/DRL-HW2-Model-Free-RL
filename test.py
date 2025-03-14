import numpy as np
import torch
from collections import defaultdict
import sys
import os
import json
import numpy as np

def random_scaled_tensor(value_range):
    rand_tensor = torch.rand((1,1))  
    scaled_tensor = (rand_tensor * 2 - 1) * value_range
    return scaled_tensor

def discretize_state(obs: dict):
    """
    Discretize the observation state.

    Args:
        obs (dict): Observation dictionary containing policy states.

    Returns:
        Tuple[pose_cart:int, pose_pole:int, vel_cart:int, vel_pole:int]: Discretized state.
    """

    # ========= put your code here =========#
    # cart pose range : [-4.8 , 4.8]
    # pole pose range : [-pi , pi]
    # cart vel  range : [-inf , inf]
    # pole vel range  : [-inf , inf]
    
    # define number of value
    pose_cart_bin = 5
    pose_pole_bin = 5
    vel_cart_bin = 10
    vel_pole_bin = 0
    # Clipping value
    pose_cart_bound = 4.5
    pose_pole_bound = np.pi
    vel_cart_bound = 10
    vel_pole_bound = 10
    
    # get observation term from continuos space
    pose_cart_raw, pose_pole_raw , vel_cart_raw , vel_pole_raw = obs['policy'][0, 0] , obs['policy'][0, 1] , obs['policy'][0, 2] , obs['policy'][0, 3]

    pose_cart_clip = torch.clip(pose_cart_raw , -pose_cart_bound ,pose_cart_bound)
    pose_pole_clip = torch.clip(pose_pole_raw , -pose_pole_bound ,pose_pole_bound)
    vel_cart_clip = torch.clip(vel_cart_raw , -vel_cart_bound ,vel_cart_bound)
    vel_pole_clip = torch.clip(vel_pole_raw , -vel_pole_bound ,vel_pole_bound)

    # linspace value
    pose_cart_grid = torch.linspace(-pose_cart_bound , pose_cart_bound , pose_cart_bin)
    pose_pole_grid = torch.linspace(-pose_pole_bound , pose_pole_bound , pose_pole_bin)
    vel_cart_grid = torch.linspace(-vel_cart_bound , vel_cart_bound , vel_cart_bin)
    vel_pole_grid = torch.linspace(-vel_pole_bound , vel_pole_bound , vel_pole_bin)

    # digitalize to range
    pose_cart_dig = torch.bucketize(pose_cart_clip,pose_cart_grid)
    pose_pole_dig = torch.bucketize(pose_pole_clip,pose_pole_grid) 
    vel_cart_dig = torch.bucketize(vel_cart_clip,vel_cart_grid)
    vel_pose_dig = torch.bucketize(vel_pole_clip,vel_pole_grid)

    return ( int(pose_cart_dig), int(pose_pole_dig), int(vel_cart_dig),  int(vel_pose_dig))

tensor = torch.tensor([[0,0,0,50]])
obs = {'policy' : tensor}

# print(discretize_state(obs))

# Save reward history
# os.makedirs("reward_value", exist_ok=True)
# reward_file = os.path.join("reward_value", "reward_history.json")
# with open(reward_file, "w") as f:
#     json.dump([5,4,3,1,5,6,8,7], f)

# print(discretize_state(obs))

update_rand = np.random.randint(0, 2)
if update_rand:
    print("hi")
else: print(update_rand)