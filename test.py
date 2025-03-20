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
# plot_q_values_3d_dual(load_q_values(q_learn_q_value_file_5000), (0,1), (4,8),(2,3), (4,4))


# print(load_q_values(q_learn_q_value_file_5000).items())
Z = np.zeros_like(X, dtype=float)

for state, values in load_q_values(q_learn_q_value_file_5000).items():
    # เลือกองค์ประกอบที่ต้องการสำหรับแกน x และ y
    x_val = state[8]
    y_val = state[8]
    if 8 <= x_val <= 8 and 8 <= y_val <= 8:
        max_val = max(values) if values else 0
        # คำนวณตำแหน่งใน grid (สมมติว่า state มีค่าเป็นตัวเลขที่ตรงกับ index ใน grid)
        
        xi = int(x_val - 8)
        yi = int(y_val - 8)
        Z[yi, xi] = max_val