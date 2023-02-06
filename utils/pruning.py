import numpy as np

class PruneModelWeights:
    def __init__(self, model):
        self.__model = model
    
    def conv_mask(self, name, input_mask=None, output_mask=None):
        layer_weights = self.__model.get_layer(name=name).get_weights()
        weights = layer_weights[0]
        biases = layer_weights[1] if len(layer_weights) > 1 else None
        if input_mask is not None:
            weights = np.delete(arr=weights, obj=input_mask, axis=2)
        if output_mask is not None:
            weights = np.delete(arr=weights, obj=output_mask, axis=3)
        if biases is not None:
            biases = np.delete(arr=biases, obj=output_mask, axis=0)
        return [weights, biases] if biases is not None else [weights]
    
    def bn_mask(self, name, mask):
        weights_list = self.__model.get_layer(name=name).get_weights()
        for i in range(len(weights_list)):
            weights_list[i] = np.delete(arr=weights_list[i], obj=mask, axis=0)
        return weights_list
    
    def dense_mask(self, name, mask, last=True):
        weights, biases = self.__model.get_layer(name=name).get_weights()
        weights = np.delete(arr=weights, obj=mask, axis=0)
        if last == False:
            weights = np.delete(arr=weights, obj=mask, axis=1)
            biases = np.delete(arr=biases, obj=mask, axis=0)
        return [weights, biases]

class PruneVGGWeights(PruneModelWeights):
    def __init__(self, model, idx_map):
        super().__init__(model)
        self.__idx_map = idx_map
    
    # prune conv and bn layers
    def __prune_conv_bn__(self):
        weights = dict()
        bn_layers_name = ["batch_normalization"] + ["batch_normalization_"+str(i) for i in range(1,13)]
        input_mask = None
        for conv_layer_name, bn_layer_name, mask in zip(self.__idx_map.keys(), bn_layers_name, self.__idx_map.values()):
            output_mask = mask
            weights[conv_layer_name] = self.conv_mask(name=conv_layer_name, input_mask=input_mask, output_mask=output_mask)
            weights[bn_layer_name] = self.bn_mask(name=bn_layer_name, mask=mask)
            input_mask = output_mask
        weights["batch_normalization_13"] = self.bn_mask(name="batch_normalization_13", mask=input_mask)
        return weights
    
    # prune dense layers
    def __prune_dense__(self):
        weights = dict()
        weights["dense"] = self.dense_mask(name="dense", mask=list(self.__idx_map.values())[-1], last=False)
        weights["dense_1"] = self.dense_mask(name="dense_1", mask=list(self.__idx_map.values())[-1])
        return weights
    
    def get_pruned_weights(self):
        weights = dict()
        weights.update(self.__prune_conv_bn__())
        weights.update(self.__prune_dense__())
        return weights

class PruneResNetWeights(PruneModelWeights):
    def __init__(self, model, depth, idx_map):
        super().__init__(model)
        self.__num_res_blocks = int((depth - 2) / 6)
        self.__idx_map = idx_map
    
    # prune conv layers in the first stack
    def __prune_first_stack_conv__(self):
        weights = dict()
        output_mask = restrict_mask = self.__idx_map["conv2d"]
        weights["conv2d"] = self.conv_mask(name="conv2d", input_mask=None, output_mask=output_mask)
        for res_block in range(self.__num_res_blocks):
            # the first conv layer in the residual unit
            layer_name = "conv2d_"+str(res_block * 2 + 1)
            input_mask = output_mask
            output_mask = self.__idx_map[layer_name]
            weights[layer_name] = self.conv_mask(name=layer_name, input_mask=input_mask, output_mask=output_mask)
            # the second conv layer in the residual unit
            layer_name = "conv2d_"+str(res_block * 2 + 2)
            input_mask = output_mask
            output_mask = restrict_mask
            weights[layer_name] = self.conv_mask(name=layer_name, input_mask=input_mask, output_mask=output_mask)
        return weights
    
    # prune conv layers in the rest stacks
    def __prune_rest_stacks_conv__(self):
        weights = dict()
        output_mask = last_stack_output_mask = self.__idx_map["conv2d"]
        idx = 2 * self.__num_res_blocks
        for stack in [1,2]:
            restrict_mask = self.__idx_map["conv2d_"+str(stack*(self.__num_res_blocks*2+1)+2)]
            for res_block in range(self.__num_res_blocks):
                # the first conv layer in the residual unit
                idx += 1
                layer_name = "conv2d_"+str(idx)
                input_mask = output_mask
                output_mask = self.__idx_map[layer_name]
                weights[layer_name] = self.conv_mask(name=layer_name, input_mask=input_mask, output_mask=output_mask)
                # the second conv layer in the residual unit
                idx +=1
                layer_name = "conv2d_"+str(idx)
                input_mask = output_mask
                output_mask = restrict_mask
                weights[layer_name] = self.conv_mask(name=layer_name, input_mask=input_mask, output_mask=output_mask)
                # the shortcut conv layer in the residual unit
                if res_block == 0:
                    idx += 1
                    layer_name = "conv2d_"+str(idx)
                    input_mask = last_stack_output_mask
                    output_mask = restrict_mask
                    weights[layer_name] = self.conv_mask(name=layer_name, input_mask=input_mask, output_mask=output_mask)
            output_mask = last_stack_output_mask = restrict_mask
        return weights
    
    # prune bn layers in the first stack
    def __prune_first_stack_bn__(self):
        weights = dict()
        restrict_mask = self.__idx_map["conv2d"]
        weights["batch_normalization"] = self.bn_mask(name="batch_normalization", mask=restrict_mask)
        for res_block in range(self.__num_res_blocks):
            # the first bn layer in the residual unit
            idx = res_block * 2 + 1
            conv_layer_name = "conv2d_"+str(idx)
            bn_layer_name = "batch_normalization_"+str(idx)
            mask = self.__idx_map[conv_layer_name]
            weights[bn_layer_name] = self.bn_mask(name=bn_layer_name, mask=mask)
            # the second bn layer in the residual unit
            bn_layer_name = "batch_normalization_"+str(idx+1)
            weights[bn_layer_name] = self.bn_mask(name=bn_layer_name, mask=restrict_mask)
        return weights
    
    # prune bn layers in the rest stacks
    def __prune_rest_stacks_bn__(self):
        weights = dict()
        bn_idx = conv_idx = 2 * self.__num_res_blocks
        for stack in [1,2]:
            restrict_mask = self.__idx_map["conv2d_"+str(stack*(self.__num_res_blocks*2+1)+2)]
            for res_block in range(self.__num_res_blocks):
                # the first bn layer in the residual unit
                conv_idx += 1
                bn_idx += 1
                conv_layer_name = "conv2d_"+str(conv_idx)
                bn_layer_name = "batch_normalization_"+str(bn_idx)
                mask = self.__idx_map[conv_layer_name]
                weights[bn_layer_name] = self.bn_mask(name=bn_layer_name, mask=mask)
                # the second bn layer in the residual unit
                conv_idx += 1
                bn_idx += 1
                bn_layer_name = "batch_normalization_"+str(bn_idx)
                weights[bn_layer_name] = self.bn_mask(name=bn_layer_name, mask=restrict_mask)
                if res_block == 0:
                    conv_idx += 1
        return weights
    
    # prune dense layer
    def __prune_dense__(self):
        weights = dict()
        mask = self.__idx_map["conv2d_"+str(2*(self.__num_res_blocks*2+1)+2)]
        weights["dense"] = self.dense_mask(name="dense", mask=mask)
        return weights
    
    def get_pruned_weights(self):
        weights = dict()
        weights.update(self.__prune_first_stack_conv__())
        weights.update(self.__prune_rest_stacks_conv__())
        weights.update(self.__prune_first_stack_bn__())
        weights.update(self.__prune_rest_stacks_bn__())
        weights.update(self.__prune_dense__())
        return weights

class PruneDenseNetWeights(PruneModelWeights):
    def __init__(self, model, idx_map, num_layers=12, grow_rate=12, num_filters=16, num_blocks=3, compression=1):
        super().__init__(model)
        self.__idx_map = idx_map
        self.__num_layers = num_layers
        self.__grow_rate = grow_rate
        self.__num_filters = num_filters
        self.__num_blocks = num_blocks
        self.__compression = compression
    
    # get masked input channels of conv layers
    def __input_channel_mask__(self):
        in_idx_map = dict()
        last_input_mask, last_output_mask = [], []
        offset = 0
        for block in range(0, self.__num_blocks):
            beg_idx, end_idx = (self.__grow_rate + 1) * block, (self.__grow_rate + 1) * (block + 1)
            for layer_name, output_mask in zip(list(self.__idx_map.keys())[beg_idx:end_idx],
                                               list(self.__idx_map.values())[beg_idx:end_idx]):
                if layer_name == "conv2d_"+str(beg_idx) or layer_name == "conv2d":
                    in_idx_map[layer_name] = last_input_mask + list(np.array(last_output_mask) + offset)
                    offset = 0
                    last_input_mask = []
                else:
                    in_idx_map[layer_name] = last_input_mask + list(np.array(last_output_mask) + offset)
                    tmp_offset = int(self.__num_layers * block * self.__grow_rate * self.__compression) + self.__num_filters
                    offset = (offset + self.__grow_rate) if layer_name != "conv2d_"+str(beg_idx+1) else (offset + tmp_offset)
                    last_input_mask = in_idx_map[layer_name]
                last_output_mask = output_mask
        return in_idx_map
    
    # prune conv layers
    def __prune_conv__(self):
        weights = dict()
        in_idx_map, out_idx_map = self.__input_channel_mask__(), self.__idx_map
        for layer_name in out_idx_map.keys():
            input_mask, output_mask = in_idx_map[layer_name], out_idx_map[layer_name]
            weights[layer_name] = self.conv_mask(name=layer_name, input_mask=input_mask, output_mask=output_mask)
        return weights
    
    # prune bn layers
    def __prune_bn__(self):
        weights = dict()
        bn_idx_map, in_idx_map = dict(), self.__input_channel_mask__()
        num_bns = len(in_idx_map)
        for i in range(0, num_bns-1):
            bn_layer_name = ("batch_normalization_"+str(i)) if i != 0 else "batch_normalization"
            bn_idx_map[bn_layer_name] = in_idx_map["conv2d_"+str(i+1)]
        bn_layer_name = "batch_normalization_"+str(num_bns-1)
        conv_layer_name = "conv2d_"+str(num_bns-1)
        offset = (self.__num_layers * self.__num_blocks - 1) * self.__grow_rate + self.__num_filters
        bn_idx_map[bn_layer_name] = in_idx_map[conv_layer_name] + list(np.array(self.__idx_map[conv_layer_name]) + offset)
        for layer_name in bn_idx_map.keys():
            weights[layer_name] = self.bn_mask(name=layer_name, mask=bn_idx_map[layer_name])
        return weights
    
    # prune dense layers
    def __prune_dense__(self):
        weights = dict()
        in_idx_map = self.__input_channel_mask__()
        conv_layer_name = list(in_idx_map.keys())[-1]
        offset = (self.__num_layers * self.__num_blocks - 1) * self.__grow_rate + self.__num_filters
        mask = in_idx_map[conv_layer_name] + list(np.array(self.__idx_map[conv_layer_name]) + offset)
        weights["dense"] = self.dense_mask(name="dense", mask=mask)
        return weights
    
    def get_pruned_weights(self):
        weights = dict()
        weights.update(self.__prune_conv__())
        weights.update(self.__prune_bn__())
        weights.update(self.__prune_dense__())
        return weights