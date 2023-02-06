import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import warnings
warnings.filterwarnings("ignore")
import argparse
from utils.model import vgg16, resnet, densenet
from utils.processing import load_cifar10, feature_sampling, FilterSelection
from utils.processing import get_filters_selection_layer, get_mask_nums
from utils.pruning import PruneVGGWeights, PruneResNetWeights, PruneDenseNetWeights
from utils.measure import measure_model
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from datetime import datetime

parser = argparse.ArgumentParser(description="CIFAR-10 Pruning and Fine-tuning")

parser.add_argument("--arch", type=str, default="resnet20", help="model architecture")
parser.add_argument("--pretrain-dir", type=str, default="./models/pre-trained", help="pre-trained model saving directory")
parser.add_argument("--save-dir", type=str, default="./models/fine-tuned", help="pruned model saving directory")
parser.add_argument("--threshold", type=float, default=0.7, help="similarity threshold")
parser.add_argument("--samples", type=int, default=100, help="number of input samples to calculate average similarity")
parser.add_argument("--epochs", type=int, default=160, help="number of fine-tuning epochs")
parser.add_argument("--batch-size", type=int, default=128, help="batch size")
parser.add_argument("--learning-rate", type=float, default=1e-3, help="initial learning rate")
parser.add_argument("--decay-step", type=str, default="80,120", help="learning rate decay")
parser.add_argument("--gpu", type=str, default="0", help="GPU id to use")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# load cifar10 dataset and pre-trained model
(x_train, y_train), (x_test, y_test) = load_cifar10()
input_shape = x_train.shape[1:]
model = keras.models.load_model(f"{args.pretrain_dir}/{args.arch}.h5")

# learning rate schedule
def lr_schedule(epoch):
    lr = args.learning_rate
    decay_step = [int(step) for step in args.decay_step.split(",")]
    for step in decay_step:
        lr = lr*1e-1 if epoch >= step else lr
    return lr

def main():
    # select redundant filters from conv layers
    depth = int(args.arch.split("resnet")[1]) if "resnet" in args.arch else None
    layers_name = get_filters_selection_layer(model, depth=depth)
    idx_map, num_map, now_time = dict(), dict(), datetime.now()
    print(f"{now_time.strftime('%Y/%m/%d %H:%M:%S')} | => selecting redundant filters of model: {args.arch}")
    samples = feature_sampling(x_train, samples_num=args.samples)
    for layer_name in layers_name:
        layer_model = Model(inputs=model.input, outputs=model.get_layer(name=layer_name).output)
        filter_selection = FilterSelection(samples=layer_model.predict(samples, verbose=0), threshold=args.threshold)
        idx_map[layer_name] = filter_selection.get_deleted_filters()
        num_map[layer_name] = len(idx_map[layer_name])
        print(f"select {num_map[layer_name]:#>4} filters from layer: {layer_name}")
    
    # create the pruned model based on selection results
    mask_nums, now_time = get_mask_nums(num_map=num_map, depth=depth), datetime.now()
    print(f"{now_time.strftime('%Y/%m/%d %H:%M:%S')} | => creating model: pruned-{args.arch}")
    if args.arch == "vgg16":
        pruned_model = vgg16(input_shape=input_shape, mask_nums=mask_nums)
        pmw = PruneVGGWeights(model=model, idx_map=idx_map)
    elif args.arch == "densenet40":
        pruned_model = densenet(input_shape=input_shape, mask_nums=mask_nums)
        pmw = PruneDenseNetWeights(model=model, idx_map=idx_map)
    elif "resnet" in args.arch:
        pruned_model = resnet(input_shape=input_shape, depth=depth, mask_nums=mask_nums)
        pmw = PruneResNetWeights(model=model, depth=depth, idx_map=idx_map)
    
    optimizer = Adam(learning_rate=lr_schedule(0))
    pruned_model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    
    # load pruned weights to the according layers
    weights_map = pmw.get_pruned_weights()
    for layer_name, weights in weights_map.items():
        pruned_model.get_layer(name=layer_name).set_weights(weights)
        print(f"load pruned weights to layer: {layer_name}")
    
    # prepare pruned model saving directory
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    model_name = f"pruned_{args.arch}_threshold_{args.threshold:.2f}.h5"
    filepath = os.path.join(args.save_dir, model_name)
    
    # prepare callbacks for pruned model saving and learning rate adjustment
    checkpoint = ModelCheckpoint(filepath=filepath, monitor="val_accuracy", verbose=2, save_best_only=True)
    lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    callbacks = [checkpoint, lr_scheduler, lr_reducer]
    
    # preprocessing and realtime data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-06,
        rotation_range=0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.0,
        zoom_range=0.0,
        channel_shift_range=0.0,
        fill_mode="nearest",
        cval=0.0,
        horizontal_flip=True,
        vertical_flip=False,
        rescale=None,
        preprocessing_function=None,
        data_format=None,
        validation_split=0.0
    )
    datagen.fit(x_train)
    
    # fine-tune the pruned model on cifar10
    beg_time = datetime.now()
    print(f"{beg_time.strftime('%Y/%m/%d %H:%M:%S')} | => fine-tuning model: pruned-{args.arch}")
    history = pruned_model.fit(datagen.flow(x_train, y_train, batch_size=args.batch_size),
                               validation_data=(x_test, y_test), epochs=args.epochs, verbose=2, callbacks=callbacks)
    end_time = datetime.now()
    print(f"{end_time.strftime('%Y/%m/%d %H:%M:%S')} | => end of fine-tuning, time-consuming: {end_time-beg_time}")
    
    # evaluate pruning performance
    now_time = datetime.now()
    print(f"{now_time.strftime('%Y/%m/%d %H:%M:%S')} | => evaluating model: pruned-{args.arch}")
    initial_acc = round(model.evaluate(x_test, y_test, verbose=0)[1]*100, 2)
    initial_params, initial_flops = measure_model(model)
    acc = round(max(history.history["val_accuracy"])*100, 2)
    params, flops = measure_model(pruned_model)
    params_drop = (initial_params - params) / initial_params * 100.0
    flops_drop = (initial_flops - flops) / initial_flops * 100.0
    print(f"threshold: {args.threshold:>.2f} -",
          f"acc, drop: {acc:>.2f}%, {initial_acc-acc:>.2f}% -",
          f"params, drop: {params/1e6:>.2f}M, {params_drop:>.1f}% -",
          f"flops, drop: {flops/1e6:>.2f}M, {flops_drop:>.1f}%")
    now_time = datetime.now()
    print(f"{now_time.strftime('%Y/%m/%d %H:%M:%S')} | => done")

if __name__ == "__main__":
    main()
