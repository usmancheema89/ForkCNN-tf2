import io, os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from keras.utils.np_utils import to_categorical

def _normalize_img(img, label):
    img = tf.cast(img, tf.float32) / 255.
    return (img, label)

def load_numpytotensor_Data():
    # def getTrainData(train_name,label_name):
    root = r'E:\Work\Multi Modal Face Recognition\Numpy Data\IRIS Data'
    data_path = os.path.join(root,'IRIS Vis Images.npy')
    label_path = os.path.join(root,'IRIS Labels.npy')
    data = np.load(data_path)                      # @mobeen Load numpy array 
    labels = np.load(label_path)
    # data = data.reshape(data.shape[0], 720, 1280, 1)
    idx = np.random.permutation(len(data))         # generate random index array
    x,y = data[idx], labels[idx]               # use randon index array to shuffle labels and images (apply to both vis and the images)
    
    (train_data, test_data, train_labels, test_labels) = train_test_split(x, y,test_size = 0.2,random_state = 42)
    # train_dataset = tf.data.Dataset.from_tensor_slices((traindata,trainlabels))
    # test_dataset = tf.data.Dataset.from_tensor_slices((testdata,testlabels))
    # train_dataset = my_train_dataset.shuffle(100).batch(16)
    # train_dataset = train_dataset.map(_normalize_img)

    # test_dataset = my_test_dataset.batch(16)
    # test_dataset = test_dataset.map(_normalize_img)
    # train_labels = to_categorical(train_labels)
    scaled_train_data = tf.keras.utils.normalize( train_data/ (train_data.max()) )
    scaled_test_data = tf.keras.utils.normalize( test_data/ (test_data.max()) )
    return scaled_train_data, scaled_test_data, train_labels, test_labels


def create_model():
    acti = 'relu'
    # x = tf.keras.layers.Lambda(my_print)(input_tf)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation = acti )(input_tf)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation = acti )(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2,padding = 'same',strides = (2,2))(x)

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation = acti )(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation = acti )(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2,padding = 'same',strides = (2,2))(x)

    x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation = acti )(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation = acti )(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation = acti )(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2,padding = 'same',strides = (2,2))(x)

    x = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation = acti )(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation = acti )(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation = acti )(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2,padding = 'same',strides = (2,2))(x)

    x = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation = acti )(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation = acti )(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation = acti )(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2,padding = 'same',strides = (2,2))(x)

    
# tutorial network tail
    # x = tf.keras.layers.Dropout(0.1)(x) # [run:True pre:0.8213 post:0.9710] 
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same')(x) # def = 32 [run:True, 0.9598][run: False:NAN] [run:64 0.97]
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(0.1)(x)     # [0.8213, 0.9598] @0.1 [run:0.3 post:0.9673] [run:False post:0.999 ]
# tutorial tail
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128)(x) # No activation on final dense layer #try 256
    x = tf.keras.layers.Lambda(lambda y: tf.math.l2_normalize(y, axis=1))(x) # L2 normalize embeddings

    return x

def my_print(x):
    # y = tf.keras.backend.sum(x)
    # tf.print(y, [y], 'Sum')
    tf.print(x, [x], "Printing Nodes")
    return x

# def VGG16_Emb():
    #     normaliz_l2 = lambda y: tf.math.l2_normalize(y, axis=1)

    #     activation = 'elu' # tf.keras.activations.elu()
    #     next_input = layers.Conv2D(64, kernel_size=(3,3), activation=activation, padding='same')(input_tf)
    #     next_input = layers.Conv2D(64, kernel_size=(3,3), activation=activation, padding='same')(next_input)
    #     next_input = layers.BatchNormalization()(next_input)
    #     # next_input = layers.MaxPooling2D(pool_size=(2,2), padding="same", strides=(2,2))(next_input)

    #     # next_input = layers.Conv2D(128, kernel_size=(3,3), activation=activation, padding='same')(next_input)
    #     # next_input = layers.Conv2D(128, kernel_size=(3,3), activation=activation, padding='same')(next_input)
    #     # next_input = layers.BatchNormalization()(next_input)
    #     # next_input = layers.MaxPooling2D(pool_size=(2,2), padding="same", strides=(2,2))(next_input)
        
    #     # next_input = layers.Conv2D(256, kernel_size=(3,3), activation=activation, padding='same')(next_input)
    #     # next_input = layers.Conv2D(256, kernel_size=(3,3), activation=activation, padding='same')(next_input)
    #     # next_input = layers.Conv2D(256, kernel_size=(3,3), activation=activation, padding='same')(next_input)
    #     # next_input = layers.BatchNormalization()(next_input)
    #     # next_input = layers.MaxPooling2D(pool_size=(2,2), padding="same", strides=(2,2))(next_input)

    #     # next_input = layers.Conv2D(512, kernel_size=(3,3), activation=activation, padding='same')(next_input)
    #     # next_input = layers.Conv2D(512, kernel_size=(3,3), activation=activation, padding='same')(next_input)
    #     next_input = layers.Conv2D(512, kernel_size=(3,3), activation=activation, padding='same')(next_input)
    #     next_input = layers.BatchNormalization()(next_input)
    #     next_input = layers.MaxPooling2D(pool_size=(2,2), padding="same", strides=(2,2))(next_input)

    #     next_input = layers.Conv2D(512, kernel_size=(3,3), activation=activation, padding='same')(next_input)
    #     next_input = layers.Conv2D(512, kernel_size=(3,3), activation=activation, padding='same')(next_input)
    #     next_input = layers.Conv2D(512, kernel_size=(3,3), activation=activation, padding='same')(next_input)
    #     next_input = layers.BatchNormalization()(next_input)
    #     next_input = layers.MaxPooling2D(pool_size=(2,2), padding="same", strides=(2,2))(next_input)

    #     output = layers.Flatten()(next_input)
    #     output = layers.Dropout(0.1)(output)
    #     # output = layers.Dense(4096, activation=activation)(output)
    #     output = layers.Dense(128)(output) # No activation on final dense layer
    #     # output = layers.Lambda(my_print)(output)
    #     output = layers.Lambda(normaliz_l2)(output) # L2 normalize embeddings

    #     # output = layers.Dense(29, activation='softmax')(output)

    #     return output

my_train_dataset, my_test_dataset, my_train_labels, my_test_labels  = load_numpytotensor_Data()

input_t = (256,256,3)
input_tf = tf.keras.Input(shape=input_t)
# Compile the model
sgd = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.0, nesterov=False) # [___]
adam = tf.keras.optimizers.Adam(0.001, epsilon = 0.1) # [0.9703]
model = tf.keras.Model(inputs = input_tf, outputs=create_model())
model.compile(optimizer= sgd, loss=tfa.losses.TripletSemiHardLoss())
# model.compile(optimizer= tf.keras.optimizers.Adam(0.0001, epsilon = 0.1), loss='categorical_crossentropy',metrics = ['accuracy'])

# print(model.summary())
# Train the network
history = model.fit(x = my_train_dataset , y = my_train_labels, epochs=50,batch_size = 64)
# Evaluate the network
results = model.predict(my_test_dataset)

# # Save test embeddings for visualization in projector
np.savetxt("vecs.tsv", results, delimiter='\t')

out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for labels in my_test_labels:
    out_m.write(str(labels) + "\n")
out_m.close()
print('Doe')
# try:
#   from google.colab import files
#   files.download('vecs.tsv')
#   files.download('meta.tsv')
# except:
#   pass













######################################################################################################################
## GET MNIST
    # train_mnist, test_mnist = tfds.load(name="mnist", split=['train', 'test'], as_supervised=True)
    # train_mnist = train_mnist.shuffle(1024).batch(32)
    # train_mnist = train_mnist.map(_normalize_img)

    # test_mnist = test_mnist.batch(32)
    # test_mnist = test_mnist.map(_normalize_img)

# input_t = (256,256,3)
# input_shape = layers.Input(shape=input_t)
# input_tf = tf.keras.Input(shape=(256,256,3))
# classes = 29

# def _normalize_img(img,label):
#     img = tf.cast(img, tf.float32) / 255.
#     return (img,label)


# def create_CNN():
#     next_input = layers.Conv2D(64, kernel_size=(3,3))(input_shape)
#     next_input = layers.Conv2D(64, kernel_size=(3,3))(next_input)
#     next_input = layers.MaxPooling2D(pool_size=(4,4))(next_input)
#     next_input = layers.Conv2D(64, kernel_size=(3,3))(next_input)

#     next_input = layers.Flatten()(next_input)
#     # tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)) # L2 normalize embeddings
#     next_input = layers.Dense(128, activation=tf.nn.relu)(next_input)
#     output = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(next_input)       #EMBEDDING LAYER
#     # output = layers.Dense(classes, activation='softmax')(next_input)

#     return output

# def tf_model():
#     # model = tf.keras.Sequential([
#     # tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(256,256,3)),
#     # tf.keras.layers.MaxPooling2D(pool_size=2),
#     # tf.keras.layers.Dropout(0.3),
#     # tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'),
#     # tf.keras.layers.MaxPooling2D(pool_size=2),
#     # tf.keras.layers.Dropout(0.3),
#     # tf.keras.layers.Flatten(),
#     # tf.keras.layers.Dense(256, activation=None), # No activation on final dense layer
#     # tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)) # L2 normalize embeddings
#     # ])

#     x = tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu')(input_tf)
#     x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
#     x = tf.keras.layers.Dropout(0.3)(x)
#     x = tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu')(x)
#     x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
#     x = tf.keras.layers.Dropout(0.3)(x)
#     x = tf.keras.layers.Flatten()(x)
#     x = tf.keras.layers.Dense(128, activation=None)(x) # No activation on final dense layer
#     x = tf.keras.layers.Lambda(lambda y: tf.math.l2_normalize(y, axis=1))(x) # L2 normalize embeddings


#     return x

# def load_Data():
#     # def getTrainData(train_name,label_name):
#     root = r'E:\Work\Multi Modal Face Recognition\Numpy Data\IRIS Data'
#     data_path = os.path.join(root,'IRIS Vis Images.npy')
#     label_path = os.path.join(root,'IRIS Labels.npy')
#     data = np.load(data_path)                      # @mobeen Load numpy array 
#     labels = np.load(label_path)
    
#     idx = np.random.permutation(len(data))         # generate random index array
#     x,y = data[idx], labels[idx]               # use randon index array to shuffle labels and images (apply to both vis and the images)

#     (traindata, testdata, trainlabels, testlabels) = train_test_split(x, y,test_size = 0.2,random_state = 42)

#     traindata = map(_normalize_img,traindata,trainlabels)
#     testdata = map(_normalize_img,testdata,testlabels)
#     # y = to_categorical(y)
    
#     # print(y.shape)
    

#     return traindata, testdata, trainlabels, testlabels

# ## GET DATA ##
# (traindata, testdata, trainlabels, testlabels) = load_Data()

# ## Create Model ##
# model = tf.keras.Model(inputs = input_tf, outputs=tf_model())
# model.compile(optimizer = Adam(),loss= [TripletSemiHardLoss()])

# ## Train Model ##
# history = model.fit(traindata,y=trainlabels,batch_size=16,epochs=3)

# ## Evaluate the Model ##
# predictions = model.predict(x = testdata)

# ## Check Embbedings ##
# # Save test embeddings for visualization in projector
# np.savetxt("vecs.tsv", predictions, delimiter='\t')

# out_m = io.open('meta.tsv', 'w', encoding='utf-8')
# for labels in testlabels:
#     out_m.write(str(labels) + "\n")
# out_m.close()


# try:
#   from google.colab import files
#   files.download('vecs.tsv')
#   files.download('meta.tsv')
# except:
#   pass

















# layers = create_CNN()
# model = Model(inputs = input_shape,outputs=layers)
# # model.compile(loss='categorical_crossentropy',optimizer=Adam(0.001),metrics=['accuracy'])    #TRIPLET LOSS COMPILATION
# model.compile(loss=TripletSemiHardLoss(),optimizer=Adam(0.001),metrics=['accuracy'])    #TRIPLET LOSS COMPILATION

# print("[INFO] training model...")
# model.fit(data, labels, epochs = 2,validation_split= 0.3)              #TRIPLET LOSS TRAINING, BATCH SIZE SHOULD BE 1

# # print(model.summary())