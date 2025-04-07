import numpy as np

def dataset_info(data):
    print("=====================================")
    print(f' data shape: {data.shape}')
    # print(data["dataset"].value_counts()) 
    print(data["K"].value_counts())
    print(data["algorithm"].value_counts())
    print(data["seed"].value_counts())
    print("=====================================")

def parse_string_to_list(rewards):
    rewards = rewards.replace("[", "").replace("]", "").replace("\n", " ")    
    rewards = rewards.replace("'", " ")
    rewards = rewards.replace("  ", " ")
    rewards = rewards.replace("  ", " ")
    rewards = rewards.replace("  ", " ")
    rewards = rewards.replace("  ", " ")
    # remove the last space
    if rewards[-1] == " ":
        rewards = rewards[:-1]
    if rewards[0] == " ":
        rewards = rewards[1:]
    rewards = rewards.replace("  ", " ").split(" ")
    # print(rewards)
    # remove "''" from the list
    rewards = [x for x in rewards if x != ""]
    rewards = np.array(rewards).astype(float)
    return rewards