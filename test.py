import numpy as np
import torch
from collections import defaultdict


def random_scaled_tensor(value_range):
    rand_tensor = torch.rand((1,1))  
    scaled_tensor = (rand_tensor * 2 - 1) * value_range
    return scaled_tensor


# print(torch.rand((1,1)))
# print(random_scaled_tensor(10))
# print(torch.tensor([5 * ((20) / (5 )) -10]))
print(torch.tensor([[1]]))