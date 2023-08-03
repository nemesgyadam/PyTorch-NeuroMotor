import torch 
import torch.nn as nn
def print_parameters(model):
    total_params = 0
    param_info = [(name, p.numel()) for name, p in model.named_parameters() if p.requires_grad]
    max_len = max([len(name) for name, _ in param_info])
    
    for name, param in param_info:
        total_params += param
        print(f'{name:.<{max_len}} --> {param}')
    
    print(f'\n{"Total Parameter Count:":.<{max_len}} --> {total_params}')

def print_weights_statistics(model):
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear):  # replace this with other types if you want to handle other layers
            print(f"Layer: {name}")
            print("Mean of weights: ", layer.weight.data.mean().item())
            print("Std of weights: ", layer.weight.data.std().item())
            print("--------------------")