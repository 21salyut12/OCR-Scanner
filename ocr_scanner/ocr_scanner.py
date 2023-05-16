import cv2, pytesseract, argparse, imutils
import numpy as np
from matplotlib import pyplot as plot
from ocr_functions import *
from PIL import Image


# ocr_functions.py version
display_img(img_file)

img = cv2.imread("C:/Users/Radu/Downloads/capture-test.jpg")
ocr = pytesseract.image_to_string(img)
print("Scanned result for {} using ocr_functions.py: \n{}".format(img, ocr))


# simpler ocr with PIL
im_file = "C:/Users/Radu/Downloads/capture-test.jpg"

im = Image.open(im_file)
ocr_pil = pytesseract.image_to_string(im)
print('Scanned result for {} using the PIL library: \n{}'.format(img_file,ocr_pil))