import cv2
import numpy
import pytesseract
from PIL import Image
from matplotlib import pyplot as plot

#Open Saved Photo
img_file = 'images/camera-test.jpg'

image = cv2.imread(img_file)

def display_img(img_path):
    dpi = 80
    img_data = plot.imread(img_path)
    height, width = img_data.shape[:2]

    figsize = width / float(dpi), height / float(dpi)
    fig = plot.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(img_data, cmap='gray')
    plot.show()

display_img(img_file)

#Invert Image/Photo
inverted_img = cv2.bitwise_not(image)
cv2.imwrite('processed_images/inverted_camera-test.jpg', inverted_img)
display_img('processed_images/inverted_camera-test.jpg')

#Binarization
def grayscale (img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray_image = grayscale(image)
cv2.imwrite('processed_images/gray.jpg', gray_image)

display_img('processed_images/gray.jpg')


try:
        img = Image.open(img_file)
        ocr = pytesseract.image_to_string(img)
        print('Scanned result for {}: \n{}'.format(img_file, ocr))

#    img.save("processed_images/proc_camera-test.jpg")
except IOError:
    pass