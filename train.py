import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import warnings
warnings.filterwarnings("ignore")
import argparse
from utils.model import vgg16, resnet, densenet
from utils.processing import load_cifar10
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from datetime import datetime

parser = argparse.ArgumentParser(description="CIFAR-10 Training")

parser.add_argument("--arch", type=str, default="resnet20", help="model architecture")
parser.add_argument("--save-dir", type=str, default="./models/pre-trained", help="model saving directory")
parser.add_argument("--epochs", type=int, default=200, help="number of training epochs")
parser.add_argument("--batch-size", type=int, default=128, help="batch size")
parser.add_argument("--optimizer", type=str, default="adam", help="optimizer")
parser.add_argument("--learning-rate", type=float, default=1e-3, help="initial learning rate")
parser.add_argument("--decay-step", type=str, default="80,120,160,180", help="learning rate decay")
parser.add_argument("--gpu", type=str, default="0", help="GPU id to use")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# load cifar10 dataset
(x_train, y_train), (x_test, y_test) = load_cifar10()
input_shape = x_train.shape[1:]

# learning rate schedule
def lr_schedule(epoch):
    lr = args.learning_rate
    decay_step = [int(step) for step in args.decay_step.split(",")]
    for step in decay_step:
        lr = lr*1e-1 if epoch >= step else lr
    return lr

def main():
    # create model
    now_time = datetime.now()
    print(f"{now_time.strftime('%Y/%m/%d %H:%M:%S')} | => creating model: {args.arch}")
    if args.arch == "vgg16":
        model = vgg16(input_shape=input_shape)
    elif args.arch == "densenet40":
        model = densenet(input_shape=input_shape)
    elif "resnet" in args.arch:
        model = resnet(input_shape=input_shape, depth=int(args.arch.split("resnet")[1]))
    
    if args.optimizer == "adam":
        optimizer = Adam(learning_rate=lr_schedule(0))
    elif args.optimizer == "sgd":
        optimizer = SGD(learning_rate=lr_schedule(0), momentum=0.9)
    
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    
    # prepare model saving directory
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    model_name = f"{args.arch}.h5"
    filepath = os.path.join(args.save_dir, model_name)
    
    # prepare callbacks for model saving and learning rate adjustment
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
    
    # fit the model on cifar10
    beg_time = datetime.now()
    print(f"{beg_time.strftime('%Y/%m/%d %H:%M:%S')} | => training model: {args.arch}")
    history = model.fit(datagen.flow(x_train, y_train, batch_size=args.batch_size),
                        validation_data=(x_test, y_test), epochs=args.epochs, verbose=2, callbacks=callbacks)
    end_time = datetime.now()
    print(f"{end_time.strftime('%Y/%m/%d %H:%M:%S')} | => end of training, time-consuming: {end_time-beg_time}")
    acc, now_time = round(max(history.history["val_accuracy"])*100, 2), datetime.now()
    print(f"{now_time.strftime('%Y/%m/%d %H:%M:%S')} | => done, best accuracy: {acc:>.2f}%")
    
if __name__ == "__main__":
    main()
