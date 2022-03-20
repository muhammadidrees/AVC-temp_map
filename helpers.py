import cv2
import numpy as np

# size of image
IMG_SIZE = (255, 300)

# path of template picture
template_file = "data/template.jpg"

def normalize(img):
    # resize image to small size
    img_size = cv2.resize(img, IMG_SIZE)

    # convert to double value gray scale
    img_gray = im2grayDouble(img_size)

    # normalize image by subtracting mean
    img_mean = imMinusMean(img_gray)

    return img_mean


def template():
    # Read template file and resize
    temp_image = cv2.imread(template_file) 
    
    # resize image to small size
    temp_norm = normalize(temp_image)

    # create template from image
    # crop_img = img[y:y+h, x:x+w]
    template = temp_norm[58:155, 90:169]

    return template

def im2grayDouble(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    info = np.iinfo(gray.dtype) 
    return gray.astype(np.float64) / info.max


def imfilter(img, kernel):
    out = cv2.filter2D(img, -1, kernel)
    return out

def imMax(img):
    maxMat = np.zeros(img.shape)
    maxMat[np.where(img == np.max(img))] = 1
    return maxMat

def imMinusMean(img):
    mean = np.mean(img)
    return img - mean

def imshow(img):
    cv2.imshow("image", img)
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 