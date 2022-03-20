import cv2
import helpers as h

image_file = "data/orient/image2.jpg"

def main():
    # read image file
    image = cv2.imread(image_file) 
    
    # normalize image
    norm_img = h.normalize(image)

    # apply template as filter
    img_f = h.imfilter(norm_img, h.template())

    # conver to max image
    img_max = h.imMax(img_f)

    # show image
    h.imshow(img_max)

if __name__ == '__main__':
    main()