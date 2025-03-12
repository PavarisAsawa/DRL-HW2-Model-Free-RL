import numpy as np
import torch
from collections import defaultdict
import sys
import os
import json

def random_scaled_tensor(value_range):
    rand_tensor = torch.rand((1,1))  
    scaled_tensor = (rand_tensor * 2 - 1) * value_range
    return scaled_tensor

# สร้าง directory "q_value" หากไม่มีอยู่แล้ว
# os.makedirs("reward_value", exist_ok=True)

# reward_file = os.path.join("reward_value", "reward_history.json")
# with open(reward_file, "w") as f:
#     json.dump([2,4,6,7,1,3,4,1,2,6,4,3,7,9,78,94,6,3213], f)
a = np.random.randint(0, 2)
if a:
    print("a")
else:
    print("b")