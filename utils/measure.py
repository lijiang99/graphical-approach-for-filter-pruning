def measure_conv2d_layer(layer):
    params = layer.count_params()
    weights = layer.get_weights()[0]
    kernel_h, kernel_w, in_channels, out_channels = weights.shape
    feature_h = layer.output_shape[1]
    feature_w = layer.output_shape[2]
    flops = kernel_h * kernel_w * in_channels * out_channels * feature_h * feature_w
    return params, flops

def measure_dense_layer(layer):
    params = layer.count_params()
    weights = layer.get_weights()[0]
    flops = weights.shape[0] * weights.shape[1]
    return params, flops

def measure_model(model):
    conv_layers_name, dense_layers_name = [], []
    for layer in model.layers:
        if "conv2d" in layer.name:
            conv_layers_name.append(layer.name)
        elif "dense" in layer.name:
            dense_layers_name.append(layer.name)
    params, flops = 0, 0
    for name in conv_layers_name:
        tmp_params, tmp_flops = measure_conv2d_layer(model.get_layer(name=name))
        params += tmp_params
        flops += tmp_flops
    for name in dense_layers_name:
        tmp_params, tmp_flops = measure_dense_layer(model.get_layer(name=name))
        params += tmp_params
        flops += tmp_flops
    return params, flops