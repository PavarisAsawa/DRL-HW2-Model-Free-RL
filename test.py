import numpy as np
import torch
from collections import defaultdict
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # สำหรับ 3D plot
from evaluate import *

q_learn_q_value_file_5000 = "q_value/Stabilize/Q_Learning/Q_Learning0/Q_Learning_0_5000_5_12_4_8.json"
plot_q_values_3d_dual(load_q_values(q_learn_q_value_file_5000), (0,1), (4,8),(2,3), (4,4))
