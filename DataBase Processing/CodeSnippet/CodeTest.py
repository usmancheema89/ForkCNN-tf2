import cv2, os
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


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
# scaled_train_data = tf.keras.utils.normalize( train_data/ (train_data.max()) )
# scaled_test_data = tf.keras.utils.normalize( test_data/ (test_data.max()) )

xnan = np.ma.filled(train_labels.astype(float), np.nan)
isnan = np.isnan(xnan)
if np.any(isnan):
    print('DATAFU')




# img_path = r'E:\Work\Multi Modal Face Recognition\Image Databases\I2BVSD\thermal_cropped\P1\P1_1.pgm'


# img = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
# img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
# # cv2.imshow("Img",img)
# # cv2.waitKey(0)
# # print(len(img.shape))

# # img = preprocessing.minmax_scale(img.ravel(), feature_range=(0,255)).reshape(img.shape)
# # img = img.astype('uint8',copy = False)

# print(img.dtype)
# print(img.shape)

# cv2.imwrite('outfile.jpg',img)

# inimg = cv2.imread('outfile.jpg',2)
# print(inimg.dtype)
