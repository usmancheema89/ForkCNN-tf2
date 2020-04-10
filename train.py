import os
import argparse
import numpy as np

import tensorflow as tf
from keras.utils import to_categorical, multi_gpu_model

from forkcnn.get_model import get_model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
curr_path = os.getcwd()

parser = argparse.ArgumentParser()
# Network selection
parser.add_argument("--dataset", "-d", help="Please enter the data to train the model", default="IRIS")
parser.add_argument("--model", "-m", help="Pleae enter the name of model architecture (`vgg16`, `resnet50`, `senet50`",
                    default='senet50')
parser.add_argument("--stream", "-s", help="Please enter number of streams, (1 or 2)", default=2)
parser.add_argument("--merge_point", "-mp", help="percentage of network to split into two streams. possible values ("
                                                 "30, 50, 70)", default=30)
parser.add_argument("--pooling", "-p", help="pooling to use. (avg or max)", default="max")
parser.add_argument("--merge-style", "-ms", help="Please enter the method to merge two streams"
                                                 "(options: `addition`, `concatenate`", default='concatenate')
# GPU settings
parser.add_argument("--multi-gpu", "-mg", help="If you want to use mulitple gpus. Turn on this flag", default=False)
parser.add_argument("--num-gpus", "-ng", help="Enter the number of gpus to use", default=2)
parser.add_argument("--gpu", "-g", help="Enter the gpu ID to train on (according to nvidia-smi)", default=1)

args = parser.parse_args()

if args.dataset == "IRIS":
    DATA_PATH = os.path.join(curr_path, "data", "IRIS Data/")
# TODO for other datasets


# Loading data
visible_data = np.load(DATA_PATH + 'Vis Images.npy')
thermal_data = np.load(DATA_PATH + 'The Images.npy')
labels = np.load(DATA_PATH + 'Labels.npy')
nb_classes = len(np.unique(labels))
y_train = to_categorical(labels)

if args.stream == 2:
    data = [visible_data, thermal_data]
else:
    data = visible_data
# Model initiation
with tf.device('/cpu:0'):
    model = get_model(model=args.model, include_top=True, input_1_tensor=None, input_shape=(256, 256, 3),
                      stream=args.stream, pooling=args.pooling, classes=nb_classes, merge_point=args.merge_point,
                      merge_style=args.merge_style)
if args.multi_gpu:
    prallel_model = multi_gpu_model(model, gpus=args.num_gpus)
else:
    parallel_model = model

# Compile model
optimizer = 'Adam'
loss = 'categorical_crossentropy'
# loss = TripletSemiHardLoss()
loss_weights = None
sample_weight_mode = None
weighted_metrics = None
target_tensors = None
metrics = ['accuracy']

parallel_model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights, sample_weight_mode=sample_weight_mode,
                       weighted_metrics=weighted_metrics, target_tensors=target_tensors, metrics=metrics)

print(model.summary())
# Train model

train = parallel_model.fit(data, y_train, batch_size=32, epochs=1, verbose=True, validation_split=0.3)
