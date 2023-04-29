import cv2
import pytesseract
import numpy as np
from PIL import Image
from matplotlib import pyplot as plot

#Open Saved Photo
img_file = 'images/capture-test.jpg'

image = cv2.imread(img_file)


#Display image
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
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray_image = grayscale(image)
cv2.imwrite('processed_images/gray.jpg', gray_image)

display_img('processed_images/gray.jpg')

thresh, img_black_white = cv2.threshold(gray_image, 45, 29, cv2.THRESH_BINARY)
cv2.imwrite('processed_images/black_white_image.jpg', img_black_white)

display_img('processed_images/black_white_image.jpg')

#Noise Removal
def noise_removal(image):
     
     kernel = np.ones((1, 1), np.uint8)
     image = cv2.dilate(image, kernel, iterations=1)
     kernel = np.ones((1, 1), np.uint8)
     image = cv2.erode(image, kernel, iterations=1)
     image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
     image = cv2.medianBlur(image, 3)
     return (image)

no_noise = noise_removal(img_black_white)
cv2.imwrite('processed_images/no_noise.jpg', no_noise)

display_img('processed_images/no_noise.jpg')


#Erosion
def thin_font(image):
     image = cv2.bitwise_not(image)
     kernel = np.ones((2,2), np.uint8)
     image = cv2.erode(image, kernel, iterations=1)
     image = cv2.bitwise_not(image)
     return (image)

eroded_image = thin_font(no_noise)
cv2.imwrite('processed_images/eroded_image.jpg', eroded_image)

display_img('processed_images/eroded_image.jpg')

#Dilation
def thick_font(image):
     image = cv2.bitwise_not(image)
     kernel = np.ones((2,2), np.uint8)
     image = cv2.dilate(image, kernel, iterations=1)
     image = cv2.bitwise_not(image)
     return (image)

dilated_image = thick_font(no_noise)
cv2.imwrite('processed_images/dilated_image.jpg', dilated_image)

display_img('processed_images/dilated_image.jpg')

#Removing borders
def remove_borders(image):
    contours, heiarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted = sorted(contours, key=lambda x:cv2.contourArea(x))
    cnt = cntsSorted[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = image[y:y+h, x:x+w]
    return (crop)

no_borders = remove_borders(no_noise)
cv2.imwrite('processed_images/removed_borders.jpg', no_borders)

display_img('processed_images/removed_borders.jpg')

#Adding missing borders
color = [255, 255, 255]
top, bottom, left, right = [150]*4

image_with_borders = cv2.copyMakeBorder(no_borders, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
cv2.imwrite("processed_images/image_with_borders.jpg", image_with_borders)

display_img("processed_images/image_with_borders.jpg")

try:
        img = Image.open(img_file)
        ocr = pytesseract.image_to_string(img)
        print('Scanned result for {}: \n{}'.format(img_file, ocr))
#    img.save("processed_images/proc_camera-test.jpg")
except IOError:
    pass