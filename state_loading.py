

def update_state_dict(original_state_dict, dataparallel=False):
    """ 
    Add or keep 'module' in name of state dict keys, depending on whether model is
    using DataParallel or not
    """
    sample_layer = [k for k in original_state_dict.keys()][0]
    state_dict = {}
    if not dataparallel and 'module' in sample_layer:
        for key, value in original_state_dict.items():
            # remove module. from start of layer names
            state_dict[key[7:]] = value
    elif dataparallel and 'module' not in sample_layer:
        for key, value in original_state_dict.items():
            # add module. to beginning of layer names
            state_dict['module.{}'.format(key)] = value
    else:
        state_dict = original_state_dict
    return state_dict


def load_state(model, PATH, device, dataparallel=False):
    """
    Import the learned weights from the `.pt` file located at PATH into the model.

    model: (nn.Module) Model
    PATH: (string) Path of the pt. file

    return None
    """
    state_dict = torch.load(PATH, map_location=device)
    state_dict = update_state_dict(state_dict, dataparallel=dataparallel)
    model.load_state_dict(state_dict)
    model.eval()
    return None
