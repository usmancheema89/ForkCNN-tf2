import json
import cv2
import numpy as np
from sklearn import preprocessing

def read_file(pairs_file):

    for key in pairs_file:
        vis_path = pairs_file[key][1]
        the_path = pairs_file[key][2]
        vis_img = get_img(vis_path)
        the_img = get_img(the_path)

        print('Label: ', pairs_file[key][0], 'Vis Path: ', vis_path, 'The Path: ', the_path)
        
        label.append(pairs_file[key][0])
        vis_imgs.append(vis_img)
        the_imgs.append(the_img)

def get_img(img_path):

    img = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        # img = preprocessing.minmax_scale(img.ravel(), feature_range=(0,255)).reshape(img.shape)
        # img = img.astype('uint8',copy = False)

    if len(img.shape) == 3:
        img = resize_img(img)
        img = crop_img(img)
    else:
        print('channel length error')
        exit(1)



    return img

def crop_img(img):

    half = 256/2
    (h,w) = img.shape[:2]
    topx, topy = int(h/2-half), int(w/2-half)
    botx, boty = int(h/2+half), int(w/2+half)
    img_crop = img[topx:botx, topy:boty]

    if img_crop.shape != (256,256,3):
        print(img.shape)
        exit(1)

    return img_crop

def resize_img(img):
    if img.shape[0] <= img.shape[1]:
        if img.shape[0] < 256:
            diff = 256 - img.shape[0]
            mag = (diff/img.shape[0]) + 1
            shape1 = int(img.shape[1] * mag)
            dim = (shape1,256)
            img = cv2.resize(img,dim, interpolation = cv2.INTER_AREA)
        if img.shape[0] > 256:
            diff = abs(256 - img.shape[0])
            mag = 1 - (diff/img.shape[0])
            shape1 = int(img.shape[1]* mag)
            dim = (shape1,256)
            img = cv2.resize(img,dim, interpolation = cv2.INTER_AREA)

    elif img.shape[1] < img.shape[0]:
        if img.shape[1] < 256:
            diff = 256 - img.shape[1]
            mag = (diff/img.shape[1]) + 1
            shape0 = int(img.shape[0] * mag)
            dim = (256, shape0)
            img = cv2.resize(img,dim, interpolation = cv2.INTER_AREA)
        if img.shape[1] > 256:
            diff = abs(256 - img.shape[1])
            mag = 1 - (diff/img.shape[1])
            shape0 = int(img.shape[0] * mag)
            dim = (256, shape0)
            img = cv2.resize(img,dim, interpolation = cv2.INTER_AREA)


    return img

def read_Json(file_path):
    with open(file_path) as data:
        files_dic = json.load(data)                     # files_dic key = subjectname_imagename_i.bmp, dict[key][0] = file path, dict[key][1] = subject id 

    return files_dic

def write_numpy(data, name):

    data_array = np.array(data)
    np.save(name,data_array)



label = []
vis_imgs = []
the_imgs = []

pairs_file_name = 'I2BVSD Img Pairs.txt'
pairs_file = read_Json(pairs_file_name)
read_file(pairs_file)
write_numpy(label,'I2BVSD Labels.npy')
write_numpy(vis_imgs,'I2BVSD Vis Images.npy')
write_numpy(the_imgs,'I2BVSD The Images.npy')

