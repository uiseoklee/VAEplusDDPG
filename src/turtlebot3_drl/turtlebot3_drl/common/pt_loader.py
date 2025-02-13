import torch

def load_weights(file_path):
    # Load the .pt file
    checkpoint = torch.load(file_path)
    
    # Print the keys of the state_dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    print("Keys in the state_dict:")
    for key in state_dict.keys():
        print(key)

if __name__ == "__main__":
    # Replace 'your_model_weights.pt' with the path to your .pt file
    file_path = '/home/dmsgv2/turtlebot3_drlnav/src/turtlebot3_drl/model/dmsgv2/ddpg_52_stage_6/actor_stage6_episode100.pt'
    load_weights(file_path)