import numpy as np
import torch

def discretize_state(obs: dict):
    """
    Discretize the observation state.

    Args:
        obs (dict): Observation dictionary containing policy states.

    Returns:
        Tuple[pose_cart:int, pose_pole:int, vel_cart:int, vel_pole:int]: Discretized state.
    """

    # ========= put your code here =========#
    # cart pose range : [-4.5 , 4.5]
    # pole pose range : [-pi , pi]
    # cart vel  range : [-20 , 20 ] real value is [-inf , inf]
    # pole vel range  : [-20 , 20 ] real value is [-inf , inf]
    
    # define bin
    pose_cart_bin , pose_pole_bin , vel_cart_bin , vel_pole_bin = 100 , 720 , 100 , 100

    # get observation term from continuos space
    pose_cart_raw, pose_pole_raw , vel_cart_raw , vel_pole_raw = obs['policy'][0, 0] , obs['policy'][0, 1] , obs['policy'][0, 2] , obs['policy'][0, 3]

    # Clipping value
    pose_cart_bound = 4.5
    pose_pole_bound = np.pi
    vel_cart_bound = 20
    vel_pole_bound = 20

    pose_cart_clip = np.clip(pose_cart_raw , -pose_cart_bound ,pose_cart_bound)
    pose_pole_clip = np.clip(pose_pole_raw , -pose_pole_bound ,pose_pole_bound)
    vel_cart_clip = np.clip(vel_cart_raw , -vel_cart_bound ,vel_cart_bound)
    vel_pole_clip = np.clip(vel_pole_raw , -vel_pole_bound ,vel_pole_bound)

    # scaled value
    pose_cart_scaled, pose_pole_scaled , vel_cart_scaled , vel_pole_scaled = pose_cart_clip, pose_pole_clip , vel_cart_clip , vel_pole_clip

    # linspace value
    pose_cart_grid = np.linspace(-pose_cart_bound , pose_cart_bound , pose_cart_bin)
    pose_pole_grid = np.linspace(-pose_pole_bound , pose_pole_bound , pose_pole_bin)
    vel_cart_grid = np.linspace(-vel_cart_bound , vel_cart_bound , vel_cart_bin)
    vel_pole_grid = np.linspace(-vel_pole_bound , vel_pole_bound , vel_pole_bin)

    # digitalize to range
    pose_cart_dig = np.digitize(x=pose_cart_scaled,bins=pose_cart_grid)
    pose_pole_dig = np.digitize(x=pose_pole_clip,bins=pose_pole_grid)
    vel_cart_dig = np.digitize(x=vel_cart_clip,bins=vel_cart_grid)
    vel_pose_dig = np.digitize(x=vel_pole_clip,bins=vel_pole_grid)

    return ( int(pose_cart_dig), int(pose_pole_dig), int(vel_cart_dig),  int(vel_pose_dig))


# # Create a tensor
# tensor_data = torch.tensor([[-4.8 , 0 , 20 , -10]])

# # Create a dictionary
# obs = {"policy": tensor_data}

# print(discretize_state(obs))

print(np.random.rand())