
def get_parameter_by_name(model, parameter_name, get_module=False):
    if get_module:
        # Return the module instead of parameter
        for name, module in model.named_modules():
            if name == parameter_name:
                return module
        return None
    else:
        # Return the parameter as before
        for name, param in model.named_parameters():
            if name == parameter_name:
                return param
        return None
