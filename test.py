import numpy as np
import torch
from collections import defaultdict
import sys
import os
import json
import numpy as np
from plotting_reward import *

algorithm = "MC"
save_number = "2"
episode = 5000

json_file = os.path.join("q_value", "Stabilize", f"{algorithm}", f"{algorithm}{save_number}", f"{algorithm}_{save_number}_{episode}_5_12_4_8.json")
q_values = load_q_values(json_file)

x_index = 0
y_index = 1

x_range = 8
y_range = 8

# plot_reward_grouped("reward_value/MC_r_0.json", group_size=100)
plot_q_values_3d(q_values=q_values , x_index=x_index,y_index=y_index,x_max=x_range,y_max=y_range)

# a = [456,23,1,564,5614,321,456,456,13,1,54,56,987]

# for t in reversed(range(len(a))):
#     print(t)
