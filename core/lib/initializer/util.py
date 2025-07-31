
def get_parameter_by_name(model, parameter_name):
    for name, param in model.named_parameters():
        if name == parameter_name:
            return param
    return None
