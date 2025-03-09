import numpy as np

def discretize_value(value, min_val, max_val, num_bins):
    bins = np.linspace(min_val, max_val, num_bins + 1)
    index = np.digitize(value, bins) - 1
    return np.clip(index, 0, num_bins - 1)

# ตัวอย่าง: แบ่งค่า x ในช่วง [-3, 3] ออกเป็น 10 bins
x = 10
bin_index = discretize_value(x, -3, 3, 10)
print("x =", x, "อยู่ใน bin index =", bin_index)