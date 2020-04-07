"""
Generic setup of the data sources and the model training. 

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py
and also on 
    https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py

"""


from keras.models         import Model
from keras.layers         import Dense, Dropout, Flatten, Input
from keras.utils.np_utils import to_categorical
from keras.callbacks      import EarlyStopping, Callback,ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.layers         import Conv2D, MaxPooling2D, Add
from keras                import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model
from time import time
import os
import numpy as np
from keras import optimizers
from sklearn.model_selection import train_test_split


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def train_and_score(nb_classes,model_name):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network
        dataset (str): Dataset to use for training/evaluating

    """
    ## setting network parameters
    batch_size = 64
    epoch = 50
    activation  = "relu"
    optimizer = optimizers.SGD(lr=0.003)
    

    img_rows, img_cols, img_channels = 128, 128, 3




    print("Compling Keras model")

    thermal_input = Input(shape=(img_rows,img_cols,3),name='thermal_input')
    visible_input = Input(shape=(img_rows,img_cols,3),name='visible_input')
    
    def streams(input,activation):
        next_input = Conv2D(64, kernel_size=(3,3), activation=activation, padding='same')(input)
        next_input = Conv2D(64, kernel_size=(3,3), activation=activation, padding='same')(next_input)

        next_input = MaxPooling2D(pool_size=(2,2), padding="same", strides=(2,2))(next_input)

        next_input = Conv2D(128, kernel_size=(3,3), activation=activation, padding='same')(next_input)
        next_input = Conv2D(128, kernel_size=(3,3), activation=activation, padding='same')(next_input)

        next_input = MaxPooling2D(pool_size=(2,2), padding="same", strides=(2,2))(next_input)

        next_input = Conv2D(256, kernel_size=(3,3), activation=activation, padding='same')(next_input)
        next_input = Conv2D(256, kernel_size=(3,3), activation=activation, padding='same')(next_input)
        next_input = Conv2D(256, kernel_size=(3,3), activation=activation, padding='same')(next_input)

        next_input = MaxPooling2D(pool_size=(2,2), padding="same", strides=(2,2))(next_input)
        return next_input
    
    thermal_output = streams(thermal_input,activation)
    Visible_output = streams(visible_input,activation)

    next_input = Add()([thermal_output,Visible_output])

    next_input = Conv2D(512, kernel_size=(3,3), activation=activation, padding='same')(next_input)
    next_input = Conv2D(512, kernel_size=(3,3), activation=activation, padding='same')(next_input)
    next_input = Conv2D(512, kernel_size=(3,3), activation=activation, padding='same')(next_input)

    next_input = MaxPooling2D(pool_size=(2,2), padding="same", strides=(2,2))(next_input)

    next_input = Conv2D(512, kernel_size=(3,3), activation=activation, padding='same')(next_input)
    next_input = Conv2D(512, kernel_size=(3,3), activation=activation, padding='same')(next_input)
    next_input = Conv2D(512, kernel_size=(3,3), activation=activation, padding='same')(next_input)

    next_input = MaxPooling2D(pool_size=(2,2), padding="same", strides=(2,2))(next_input)

    output = Flatten()(next_input)
    output = Dense(4096, activation=activation)(output)
    output = Dropout(0.5)(output)
    output = Dense(nb_classes, activation='softmax')(output)
    

    model = Model(inputs = [thermal_input,visible_input], outputs=[output])
    model = multi_gpu_model(model, gpus=2) #in this case the number of GPus is 2
    model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])


    ######################################## Defining Optional Callbacks ####################################################
    # history = LossHistory()

    # In your case, you can see that your training loss is not dropping - which means you are learning nothing after each epoch.
    # It look like there's nothing to learn in this model, aside from some trivial linear-like fit or cutoff value.
    early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=5, verbose=1, mode='auto',restore_best_weights=True)

    # Save the model according to the conditions
    # filepath_checkpoint = "chekcpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5"
    # checkpoint = ModelCheckpoint(filepath_checkpoint, monitor='val_acc', verbose=1, save_best_only=True,save_weights_only=True, mode='auto', period=10)

    tensorboard = TensorBoard(
        log_dir='logs\\' + model_name,
        histogram_freq=0,
        write_images=True,
        write_graph=True,
        batch_size=batch_size
    )

    ###################################### Training the Model ############################################################


    ThermImages = np.load('E:\Work\CNN\IRIS VisThermal data\IRIS_Thermal_data.npy')
    VisImages = np.load('E:\Work\CNN\IRIS VisThermal data\IRIS_Visible_data.npy')
    labelsdata = np.load('E:\Work\CNN\IRIS VisThermal data\IRIS_Labels.npy')
    labels = to_categorical(labelsdata)

    ## spliting total data for test and validation
    split = train_test_split(ThermImages, VisImages, labels, test_size=0.25, random_state=42)
    (trainThermImages, splitThermImages, trainVisImages, splitVisImages, TrainLabels, splitLabels) = split

    test_train_split = train_test_split(splitThermImages, splitVisImages, splitLabels, test_size=0.01, random_state=42)
    (valThermImages, testThermImages, valVisImages, testVisImages, valLabels, testLabels) = test_train_split

    print("[INFO] training model...")
    model.fit(
    [trainThermImages, trainVisImages], TrainLabels,
    validation_data=([valThermImages, valVisImages], valLabels),
    epochs = epoch, batch_size = batch_size,callbacks = [tensorboard])

    # make predictions on the testing data
    print("[INFO reading Test Data...]")
    Test_ThermImages = np.load('E:\Work\CNN\IRIS VisThermal data\Test_Thermal_data.npy')
    Test_VisImages = np.load('E:\Work\CNN\IRIS VisThermal data\Test_Visible_data.npy')
    Test_labelsdata = np.load('E:\Work\CNN\IRIS VisThermal data\Test_Labels.npy')
    Test_labels = to_categorical(Test_labelsdata)
    print("[INFO] predicting Labels...")
    # preds = model.predict([testThermImages, testVisImages])
    results = model.evaluate([Test_ThermImages, Test_VisImages],Test_labels, batch_size=16)
    print(results)

    path = "E://Work//CNN//IRIS VisThermal data//"
    filepath_model = (path + model_name)
    model.save(filepath_model, overwrite= True)
    # ###################################### Testing the model #####################################################
    K.clear_session()

    
    return 

if __name__=="__main__":

    classes = 28
    savefilename = "IRIS_Fork_SGD_lr.003_Multi Add.h5"
    train_and_score(classes, savefilename)