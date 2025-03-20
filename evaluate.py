import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # สำหรับ 3D plot

def load_q_values(json_file):
    """
    โหลด Q-values จากไฟล์ JSON และแปลง key ให้เป็น tuple ของ float
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    q_values = {}
    for state_str, values in data['q_values'].items():
        # ลบวงเล็บและแยก string เป็นตัวเลข
        state = tuple(map(float, state_str.strip("()").split(", ")))
        q_values[state] = values
    return q_values

def plot_q_values_3d(q_values, x_index, y_index, x_max, y_max):
    """
    สร้างกราฟ 3 มิติจาก Q-values โดย:
      - x_index, y_index: เลือก index ของ state tuple ที่จะใช้เป็นแกน x และ y 
      - x_range, y_range: (min, max) ของ state ในมิติที่เลือก (ทั้งสองแกน)
      - ถ้า key ไม่อยู่ในช่วงที่กำหนด ให้ค่าเป็น 0
  
    แกน z คือค่าสูงสุด (max) ใน list ของ Q‑values สำหรับ key นั้น ๆ
    """
    x_min, x_max = [0 , x_max]
    y_min, y_max = [0 , y_max]

    # สร้าง grid สำหรับแกน x และ y
    x_vals = np.arange(x_min, x_max + 1)
    y_vals = np.arange(y_min, y_max + 1)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.zeros_like(X, dtype=float)

    # วนลูปผ่าน Q-values และเติมค่าใน grid ถ้า key อยู่ในช่วงที่กำหนด
    for state, values in q_values.items():
        # เลือกองค์ประกอบที่ต้องการสำหรับแกน x และ y
        x_val = state[x_index]
        y_val = state[y_index]
        if x_min <= x_val <= x_max and y_min <= y_val <= y_max:
            max_val = max(values) if values else 0
            # คำนวณตำแหน่งใน grid (สมมติว่า state มีค่าเป็นตัวเลขที่ตรงกับ index ใน grid)
            
            xi = int(x_val - x_min)
            yi = int(y_val - y_min)
            Z[yi, xi] = max_val

    # Plot 3D surface
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_xlabel(f"State dimension {x_index}")
    ax.set_ylabel(f"State dimension {y_index}")
    ax.set_zlabel("Max Q value")
    ax.set_title("3D Plot of Q-values")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.show()
    
def plot_reward(json_file):
    """
    โหลด reward จากไฟล์ JSON และแสดงกราฟ
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    rewards = data
    plt.figure(figsize=(8, 6))
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward per Episode")
    plt.show()
    
def plot_reward_grouped(json_file, group_size=100):
    """
    โหลด reward จากไฟล์ JSON แล้วแบ่งเป็นกลุ่ม ๆ ทีละ group_size episode
    จากนั้นพล็อตกราฟโดยใช้ค่าเฉลี่ยของ reward ในแต่ละกลุ่ม

    Args:
        json_file (str): path ของไฟล์ JSON ที่เก็บ reward
        group_size (int): จำนวน episode ต่อกลุ่ม (default คือ 100)
    """
    # โหลด rewards จาก JSON file (สมมุติว่าเป็น list ของ reward)
    with open(json_file, 'r') as f:
        rewards = json.load(f)

    # คำนวณค่าเฉลี่ยของ reward ในแต่ละกลุ่ม
    avg_rewards = []
    episodes = []
    for i in range(0, len(rewards), group_size):
        group = rewards[i:i+group_size]
        avg = np.mean(group)
        avg_rewards.append(avg)
        episodes.append(i)

    # สร้างกราฟ
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, avg_rewards, marker='o', linestyle='-')
    plt.xlabel("Episode")
    plt.ylabel("Average Reward (per {} episodes)".format(group_size))
    plt.title("Average Reward per {} Episodes".format(group_size))
    plt.grid(True)
    plt.ylim(bottom=0)  # กำหนดให้แกน y เริ่มต้นที่ 0
    plt.show()

def plot_multiple_reward_files(json_files, group_size=100):
    """
    Loads rewards from multiple JSON files and plots the average reward per group
    for each file on a single graph.

    Args:
        json_files (list of str): List of paths to JSON files that store rewards (each as a list of rewards).
        group_size (int): Number of episodes per group for averaging (default is 100).
    """
    plt.figure(figsize=(10, 6))
    
    for json_file in json_files:
        # Load rewards from the JSON file
        with open(json_file, 'r') as f:
            rewards = json.load(f)
        
        # Compute average rewards per group
        avg_rewards = []
        episodes = []
        for i in range(0, len(rewards), group_size):
            group = rewards[i:i+group_size]
            avg = np.mean(group)
            avg_rewards.append(avg)
            episodes.append(i)  # starting index of the group

        # Plot with label derived from file name
        label = os.path.basename(json_file)
        plt.plot(episodes, avg_rewards, marker='o', linestyle='-', label=label)
    
    plt.xlabel("Episode")
    plt.ylabel("Average Reward (per {} episodes)".format(group_size))
    plt.title("Average Reward per {} Episodes".format(group_size))
    plt.grid(True)
    plt.ylim(bottom=0)  # Set y-axis to start at 0
    plt.legend()
    plt.show()
    
    
def plot_q_values_3d_dual(q_values, left_pair, left_range, right_pair, right_range):
    """
    Create two 3D plots side-by-side from Q-values:
      - Left plot uses state indices specified in left_pair (tuple: (x_index, y_index))
        with range left_range (tuple: (x_max, y_max))
      - Right plot uses state indices specified in right_pair (tuple: (x_index, y_index))
        with range right_range (tuple: (x_max, y_max))
    The z-axis in each plot is the maximum Q-value from the list for that state.
    If a state key is not in the specified range, its value is set to 0.

    Additionally, axis labels are provided according to the following mapping:
      0: Cart Pose
      1: Pole Pose
      2: Cart Velocity
      3: Pole Velocity

    Args:
        q_values (dict): Dictionary with keys as state tuples and values as list of Q-values.
        left_pair (tuple): (x_index, y_index) for the left plot.
        left_range (tuple): (x_max, y_max) for the left plot.
        right_pair (tuple): (x_index, y_index) for the right plot.
        right_range (tuple): (x_max, y_max) for the right plot.
    """
    # Mapping from state dimension index to label.
    dim_labels = {
        0: "Cart Pose",
        1: "Pole Pose",
        2: "Cart Velocity",
        3: "Pole Velocity"
    }
    
    # Left plot settings
    left_x_index, left_y_index = left_pair
    left_x_min, left_x_max = 0, left_range[0]  # left_range[0] is the number of states for dimension x
    left_y_min, left_y_max = 0, left_range[1]  # left_range[1] is the number of states for dimension y
    left_x_vals = np.arange(left_x_min, left_x_max)  # now it will create exactly left_range[0] points: 0 to left_range[0]-1
    left_y_vals = np.arange(left_y_min, left_y_max)
    X_left, Y_left = np.meshgrid(left_x_vals, left_y_vals)
    Z_left = np.zeros_like(X_left, dtype=float)
    
    # Fill grid for left plot using indices specified in left_pair
    for state, values in q_values.items():
        x_val = state[left_x_index]
        y_val = state[left_y_index]
        if left_x_min <= x_val < left_x_max and left_y_min <= y_val < left_y_max:
            max_val = max(values) if values else 0
            xi = int(x_val - left_x_min)
            yi = int(y_val - left_y_min)
            Z_left[yi, xi] = max_val

    # Right plot settings
    right_x_index, right_y_index = right_pair
    right_x_min, right_x_max = 0, right_range[0]
    right_y_min, right_y_max = 0, right_range[1]
    right_x_vals = np.arange(right_x_min, right_x_max)
    right_y_vals = np.arange(right_y_min, right_y_max)
    X_right, Y_right = np.meshgrid(right_x_vals, right_y_vals)
    Z_right = np.zeros_like(X_right, dtype=float)
    
    # Fill grid for right plot using indices specified in right_pair
    for state, values in q_values.items():
        if len(state) <= max(right_x_index, right_y_index):
            continue
        x_val = state[right_x_index]
        y_val = state[right_y_index]
        if right_x_min <= x_val < right_x_max and right_y_min <= y_val < right_y_max:
            max_val = max(values) if values else 0
            xi = int(x_val - right_x_min)
            yi = int(y_val - right_y_min)
            Z_right[yi, xi] = max_val

    # Create figure with two subplots side-by-side
    fig = plt.figure(figsize=(16, 6))
    
    # Left subplot
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(X_left, Y_left, Z_left, cmap='viridis', edgecolor='none')
    ax1.set_xlabel(f"{dim_labels.get(left_x_index, f'Dim {left_x_index}')}")
    ax1.set_ylabel(f"{dim_labels.get(left_y_index, f'Dim {left_y_index}')}")
    ax1.set_zlabel("Max Q-value")
    ax1.set_title(f"3D Q-values (indices {left_pair[0]}, {left_pair[1]})")
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
    
    # Right subplot
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(X_right, Y_right, Z_right, cmap='viridis', edgecolor='none')
    ax2.set_xlabel(f"{dim_labels.get(right_x_index, f'Dim {right_x_index}')}")
    ax2.set_ylabel(f"{dim_labels.get(right_y_index, f'Dim {right_y_index}')}")
    ax2.set_zlabel("Max Q-value")
    ax2.set_title(f"3D Q-values (indices {right_pair[0]}, {right_pair[1]})")
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
    
    plt.tight_layout()
    plt.show()
    
def plot_q_values_3d_and_top(q_values, x_index, y_index, x_range, y_range):
    """
    Plots Q-values in both 3D (surface) and top-down (2D heatmap) views 
    for a single pair of state dimensions (x_index, y_index).

    The z-axis / color represents the maximum Q-value across all actions for that state.

    Args:
        q_values (dict): Keys are state tuples, values are lists of Q-values (one for each action).
        x_index (int): Index of the state dimension to use for the x-axis.
        y_index (int): Index of the state dimension to use for the y-axis.
        x_range (int): The number of discrete states in the x dimension (0 to x_range-1).
        y_range (int): The number of discrete states in the y dimension (0 to y_range-1).
    """

    # Mapping from state dimension index to label (customize as needed)
    dim_labels = {
        0: "Cart Pose",
        1: "Pole Pose",
        2: "Cart Velocity",
        3: "Pole Velocity"
    }

    # Set up the grid
    x_min, x_max = 0, x_range
    y_min, y_max = 0, y_range
    x_vals = np.arange(x_min, x_max)
    y_vals = np.arange(y_min, y_max)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.zeros_like(X, dtype=float)

    # Fill in the Z array with max Q-values for each state in the given range
    for state, values in q_values.items():
        # Check if state has enough dimensions
        if len(state) <= max(x_index, y_index):
            continue

        x_val = state[x_index]
        y_val = state[y_index]

        if x_min <= x_val < x_max and y_min <= y_val < y_max:
            max_val = max(values) if values else 0
            xi = int(x_val - x_min)
            yi = int(y_val - y_min)
            Z[yi, xi] = max_val

    # Create a figure with two subplots side-by-side
    fig = plt.figure(figsize=(12, 5))

    # -------------------
    # 1) 3D surface plot
    # -------------------
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax1.set_xlabel(dim_labels.get(x_index, f"Dim {x_index}"))
    ax1.set_ylabel(dim_labels.get(y_index, f"Dim {y_index}"))
    ax1.set_zlabel("Max Q-value")
    ax1.set_title(f"3D Q-values (Indices {x_index}, {y_index})")
    fig.colorbar(surf, ax=ax1, shrink=0.6, aspect=10)

    # -------------------
    # 2) Top-down 2D heatmap
    # -------------------
    ax2 = fig.add_subplot(1, 2, 2)
    # imshow expects y on the vertical axis and x on the horizontal,
    # so we transpose Z or set origin='lower'.
    im = ax2.imshow(
        Z, origin='lower', 
        cmap='viridis', 
        extent=[x_min, x_max, y_min, y_max],
        aspect='auto'
    )
    ax2.set_xlabel(dim_labels.get(x_index, f"Dim {x_index}"))
    ax2.set_ylabel(dim_labels.get(y_index, f"Dim {y_index}"))
    ax2.set_title(f"Top View (Indices {x_index}, {y_index})")
    fig.colorbar(im, ax=ax2, shrink=0.6, aspect=10)

    plt.tight_layout()
    plt.show()
    

def get_heatmap_data(q_values, x_index, y_index, x_max, y_max):
    """
    Build a 2D array Z (of shape [y_max, x_max]) where each cell contains
    the maximum Q-value for the corresponding (x_val, y_val) in the state space.
    
    Args:
        q_values (dict): Dictionary with keys as state tuples and values as list of Q-values.
        x_index (int): Index of the state tuple for the x-axis.
        y_index (int): Index of the state tuple for the y-axis.
        x_max (int): Number of discrete states along the x-axis (states 0 to x_max-1).
        y_max (int): Number of discrete states along the y-axis (states 0 to y_max-1).
    
    Returns:
        X, Y, Z: Arrays suitable for plotting with imshow. Z[y, x] contains the max Q-value.
    """
    # Create grid arrays from 0 to x_max-1 and 0 to y_max-1
    x_vals = np.arange(0, x_max)
    y_vals = np.arange(0, y_max)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Initialize Z with zeros
    Z = np.zeros((y_max, x_max), dtype=float)
    
    # Fill Z with maximum Q-value for each state, if available
    for state, values in q_values.items():
        if len(state) <= max(x_index, y_index):
            continue
        x_val = state[x_index]
        y_val = state[y_index]
        if 0 <= x_val < x_max and 0 <= y_val < y_max:
            max_val = max(values) if values else 0
            xi = int(x_val)
            yi = int(y_val)
            Z[yi, xi] = max_val
    return X, Y, Z

def plot_q_values_heatmaps_multiple_files(
    file_list,
    left_pair, left_range,
    right_pair, right_range,
    episodes_list=None
):
    """
    Plot heatmaps for multiple Q-value files. Each file produces two heatmaps side by side:
      - Left heatmap uses state indices specified in left_pair (tuple: (x_index, y_index))
        with grid size defined by left_range (tuple: (x_max, y_max)).
      - Right heatmap uses state indices specified in right_pair (tuple: (x_index, y_index))
        with grid size defined by right_range (tuple: (x_max, y_max)).
    
    If you have N files, the function creates an N-row x 2-column grid of subplots.
    Each row is optionally labeled with an entry from episodes_list (e.g., episode number).
    
    Args:
        file_list (list of str): Paths to JSON files containing Q-values.
        left_pair (tuple): (x_index, y_index) for the left heatmap.
        left_range (tuple): (x_max, y_max) for the left heatmap.
        right_pair (tuple): (x_index, y_index) for the right heatmap.
        right_range (tuple): (x_max, y_max) for the right heatmap.
        episodes_list (list of str or None): Labels for each row (e.g., "Episode 1000"). If None, no extra labels are added.
    """
    # Mapping for state dimension labels.
    dim_labels = {
        0: "Cart Pose",
        1: "Pole Pose",
        2: "Cart Velocity",
        3: "Pole Velocity"
    }
    
    # Ensure episodes_list exists and has the same length as file_list
    if episodes_list is None:
        episodes_list = ["" for _ in file_list]
    elif len(episodes_list) < len(file_list):
        episodes_list += [""] * (len(file_list) - len(episodes_list))
    
    n_files = len(file_list)
    fig, axes = plt.subplots(n_files, 2, figsize=(10, 3 * n_files))
    
    # If only one file is provided, reshape axes to 2D
    if n_files == 1:
        axes = np.array([axes])
    
    for i, json_file in enumerate(file_list):
        # Load Q-values from JSON
        with open(json_file, 'r') as f:
            data = json.load(f)
        raw_q_values = data['q_values']
        # Convert keys to tuple of floats
        q_dict = {}
        for state_str, vals in raw_q_values.items():
            st = tuple(map(float, state_str.strip("()").split(", ")))
            q_dict[st] = vals
        
        # Build heatmap data for left pair
        X_left, Y_left, Z_left = get_heatmap_data(
            q_dict,
            x_index=left_pair[0],
            y_index=left_pair[1],
            x_max=left_range[0],
            y_max=left_range[1]
        )
        
        # Build heatmap data for right pair
        X_right, Y_right, Z_right = get_heatmap_data(
            q_dict,
            x_index=right_pair[0],
            y_index=right_pair[1],
            x_max=right_range[0],
            y_max=right_range[1]
        )
        
        # Plot left heatmap
        im_left = axes[i, 0].imshow(Z_left, origin='lower', cmap='viridis', aspect='auto')
        fig.colorbar(im_left, ax=axes[i, 0], fraction=0.046, pad=0.04)
        axes[i, 0].set_xlabel(dim_labels.get(left_pair[0], f"Dim {left_pair[0]}"))
        axes[i, 0].set_ylabel(dim_labels.get(left_pair[1], f"Dim {left_pair[1]}"))
        axes[i, 0].set_title(f"{os.path.basename(json_file)} (Left: {left_pair})")
        
        # Plot right heatmap
        im_right = axes[i, 1].imshow(Z_right, origin='lower', cmap='viridis', aspect='auto')
        fig.colorbar(im_right, ax=axes[i, 1], fraction=0.046, pad=0.04)
        axes[i, 1].set_xlabel(dim_labels.get(right_pair[0], f"Dim {right_pair[0]}"))
        axes[i, 1].set_ylabel(dim_labels.get(right_pair[1], f"Dim {right_pair[1]}"))
        axes[i, 1].set_title(f"{os.path.basename(json_file)} (Right: {right_pair})")
        
        # Add row label if provided (displaying episode info)
        if episodes_list[i]:
            # Here we set the ylabel of the left subplot to include the episode label.
            current_ylabel = axes[i, 0].get_ylabel()
            axes[i, 0].set_ylabel(f"{episodes_list[i]}\n{current_ylabel}")
    
    plt.tight_layout()
    plt.show()