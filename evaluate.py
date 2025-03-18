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
    
    
