from tensorflow import keras
from tensorflow.keras.datasets import cifar10
import numpy as np
from scipy import fftpack
import pandas as pd
import igraph

# load and preprocess the cifar10 dataset
def load_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

# randomly sample the outputs of the convolutional layer
def feature_sampling(output, samples_num):
    samples_index = np.random.choice(output.shape[0], size=samples_num, replace=False)
    return output[samples_index,:,:,:]

# get conv layers whose filters need to be selected
def get_filters_selection_layer(model, depth=None):
    layers_name = []
    if depth is None:
        for layer in model.layers:
            if "conv" in layer.name:
                layers_name.append(layer.name)
        return layers_name
    # for resnet
    num_res_blocks = int((depth - 2) / 6)
    layers_name.append("conv2d")
    for res_block in range(num_res_blocks):
        layers_name.append("conv2d_"+str(res_block * 2 + 1))
    idx = 2 * num_res_blocks
    for stack in [1,2]:
        for res_block in range(num_res_blocks):
            idx += 1
            layers_name.append("conv2d_"+str(idx))
            idx += 1
            if res_block == 0:
                idx += 1
                layers_name.append("conv2d_"+str(idx))
    return layers_name

# get the number of masked filters in all conv layers
def get_mask_nums(num_map, depth=None):
    if depth is None:
        return list(num_map.values())
    # for resnet
    num_res_blocks = int((depth - 2) / 6)
    mask_nums, restrict_mask_num = [num_map["conv2d"]], num_map["conv2d"]
    for res_block in range(num_res_blocks):
        layer_name = "conv2d_"+str(res_block * 2 + 1)
        mask_nums.append(num_map[layer_name])
        mask_nums.append(restrict_mask_num)
    idx = 2 * num_res_blocks
    for stack in [1,2]:
        restrict_mask_num = num_map["conv2d_"+str(stack*(num_res_blocks*2+1)+2)]
        for res_block in range(num_res_blocks):
            idx += 1
            layer_name = "conv2d_"+str(idx)
            mask_nums.append(num_map[layer_name])
            idx += 1
            mask_nums.append(restrict_mask_num)
            if res_block == 0:
                idx += 1
                layer_name = "conv2d_"+str(idx)
                mask_nums.append(num_map[layer_name])
    return mask_nums

# select the filters according to the specified threshold
class FilterSelection:
    def __init__(self, samples, threshold=0.7):
        self.__samples = samples
        self.__samples_num = samples.shape[0]
        self.__filters_num = samples.shape[3]
        self.__threshold = threshold
    
    # calculate pHash
    def __pHash__(self, feature):
        dct = fftpack.dct(fftpack.dct(feature, axis=0), axis=1)
        dct_low_freq = dct[:8,:8]
        avg = np.mean(dct_low_freq)
        return (dct_low_freq >= avg).astype(int)
    
    # calculate average similarity
    def get_similarity(self):
        result = {"index1": [], "index2": [], "similarity": []}
        for i in range(self.__samples_num):
            features = self.__samples[i,:,:,:]
            for j in range(self.__filters_num):
                feature1 = features[:,:,j]
                for k in range(j+1, self.__filters_num):
                    feature2, hamming_distance, similarity = features[:,:,k], None, None
                    hamming_distance = np.sum(self.__pHash__(feature1) ^ self.__pHash__(feature2))
                    if feature2.shape[0] < 8:
                        similarity = 1 - hamming_distance * 1.0 / (feature2.shape[0] ** 2)
                    if feature2.shape[0] >= 8:
                        similarity = 1 - hamming_distance * 1.0 / 64
                    result["index1"].append(j)
                    result["index2"].append(k)
                    result["similarity"].append(similarity)
        return pd.DataFrame(result).groupby(["index1", "index2"]).mean().reset_index()
    
    # create graph based on a fixed threshold
    def __create_graph__(self):
        df = self.get_similarity()
        df = df[df["similarity"] >= self.__threshold]
        g = igraph.Graph()
        g.add_vertices([i for i in range(self.__filters_num)])
        g.add_edges([(i, j) for i, j in zip(df["index1"], df["index2"])])
        g.vs["label"] = [i for i in range(self.__filters_num)]
        g.es["label"] = [round(i,2) for i in df["similarity"]]
        return g
    
    # select vertices (saved or deleted) from graph
    def __filter_vertices__(self):
        g = self.__create_graph__()
        all_vertices, isolated_vertices = set(g.vs["label"]), set(g.vs.select(_degree=0)["label"])
        saved_vertices, deleted_vertices = set(), set()
        for vertex in (all_vertices - isolated_vertices):
            # case1: vertex has been saved or deleted
            if vertex in (saved_vertices | deleted_vertices):
                continue
            neighbors = g.neighbors(vertex)
            # case2: vertex has one neighbor, and the neighbor does not connect to others
            if len(neighbors) == 1 and g.degree(neighbors[0]) == 1:
                saved_vertices.add(vertex)
                deleted_vertices = deleted_vertices | set(neighbors)
                continue
            # case3: vertex has one neighbor, and the neighbor connects to others
            if len(neighbors) == 1 and g.degree(neighbors[0]) != 1:
                deleted_vertices.add(vertex)
                continue
            # case4: vertex has more than one neighbor
            neighbors = set(neighbors) - (saved_vertices | deleted_vertices)
            # case4.1: neighbors all have been saved or deleted
            if len(neighbors) == 0:
                saved_vertices.add(vertex)
                continue
            # case4.2: one or more than one neighbor have not been saved or deleted
            max_degree, max_weight = g.degree(vertex), sum(g.es.select(_source=vertex)["label"])
            saved_vertex = vertex
            for neighbor in neighbors:
                tmp_degree, tmp_weight = g.degree(neighbor), sum(g.es.select(_source=neighbor)["label"])
                if (tmp_degree > max_degree) or (tmp_degree == max_degree and tmp_weight > max_weight):
                    max_degree, max_weight, saved_vertex = tmp_degree, tmp_weight, neighbor
            saved_vertices.add(saved_vertex)
            deleted_vertices = deleted_vertices | ((neighbors | {vertex}) - {saved_vertex})
        return (isolated_vertices, saved_vertices | isolated_vertices, deleted_vertices)
    
    # get the index of filters needed to be deleted
    def get_deleted_filters(self):
        return list(self.__filter_vertices__()[2])
