import json
import os
import torch
def load_q_value(path, filename):
    output = {}
    full_path = os.path.join(path, filename)        
    with open(full_path, 'r') as file:
        data = json.load(file)
        data_q_values = data['q_values']
        for state, action_values in data_q_values.items():
            state = state.replace('(', '')
            state = state.replace(')', '')
            tuple_state = tuple(map(float, state.split(', ')))

            output[tuple_state] = action_values.copy()

    return output


def plotting_q(path : str,filename : str):
    array = torch.tensor([[]])
    q_val = load_q_value(path ,filename)
    # print(q_val)
    for i in q_val:
        xy  = i[0:2]
        z   = max(i)
        print(xy)



plotting_q("q_value/Q_Learning","QL_q_8.json")
# print(load_q_value("q_value/Q_Learning","QL_q_8.json"))
