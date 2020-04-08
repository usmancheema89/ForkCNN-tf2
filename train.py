import os
import argparse

import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.utils import to_categorical
from tensorflow_addons.losses import TripletSemiHardLoss

from forkcnn.get_model import get_model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
curr_path = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", help="Please enter the data to train the model", default="IRIS")
parser.add_argument("--model", "-m", help="Pleae enter the name of model architecture (`vgg16`, `resnet50`, `senet50`",
                    default='vgg16')
parser.add_argument("--stream", "-s", help="Please enter number of streams, (1 or 2)", default=2)
parser.add_argument("--merge_point", "-mp", help="percentage of network to split into two streams. possible values ("
                                                 "30, 50, 70)", default=30)
parser.add_argument("--pooling", "-p", help="pooling to use. (avg or max)", default="max")

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

# Model initiation

model = get_model(model=args.model, include_top=False, input_1_tensor=None, input_shape=(256, 256, 3),
                  stream=args.stream, pooling=args.pooling, classes=nb_classes, merge_point=args.merge_point,
                  merge_style='concatenate')

# Compile model
optimizer = 'Adam'
loss = 'categorical_crossentropy'
# loss = TripletSemiHardLoss()
loss_weights = None
sample_weight_mode = None
weighted_metrics = None
target_tensors = None
metrics = ['accuracy']
model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights, sample_weight_mode=sample_weight_mode,
              weighted_metrics=weighted_metrics, target_tensors=target_tensors, metrics=metrics)

print(model.summary())
# Train model

train = model.fit([visible_data, thermal_data], y_train, batch_size=32, epochs=32, verbose=True, validation_split=0.3)
