import json

# Assuming the JSON file is called 'data.json'
with open('reward_value/QL_r_3.json', 'r') as file:
    data = json.load(file)

# Now you can access the list of values
print(data)  # This will print the entire list
