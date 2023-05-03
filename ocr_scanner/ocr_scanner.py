import cv2, pytesseract, argparse, imutils
import numpy as np
from matplotlib import pyplot as plot
from ocr_functions import *

display_img(img_file)

img = cv2.imread('images/capture-test.jpg')
ocr = pytesseract.image_to_string(img)
print('Scanned result for {}: \n{}'.format('images/capture-test.jpg', ocr))
