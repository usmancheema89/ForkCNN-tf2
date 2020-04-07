import cv2
from sklearn import preprocessing


img_path = r'E:\Work\Multi Modal Face Recognition\Image Databases\I2BVSD\thermal_cropped\P1\P1_1.pgm'


img = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
# cv2.imshow("Img",img)
# cv2.waitKey(0)
# print(len(img.shape))

# img = preprocessing.minmax_scale(img.ravel(), feature_range=(0,255)).reshape(img.shape)
# img = img.astype('uint8',copy = False)

print(img.dtype)
print(img.shape)

cv2.imwrite('outfile.jpg',img)

inimg = cv2.imread('outfile.jpg',2)
print(inimg.dtype)
