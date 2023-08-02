import torch 

def print_parameters(model):
    total_params = 0
    param_info = [(name, p.numel()) for name, p in model.named_parameters() if p.requires_grad]
    max_len = max([len(name) for name, _ in param_info])
    
    for name, param in param_info:
        total_params += param
        print(f'{name:.<{max_len}} --> {param}')
    
    print(f'\n{"Total Parameter Count:":.<{max_len}} --> {total_params}')