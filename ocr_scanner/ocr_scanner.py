import pytesseract
from PIL import Image

img_files_list = ["images/camera-test.jpg", "images/test_image1.jpg",
                  "images/test_image2.jpg", "images/test_image3.png"]

try:
    for img_file in img_files_list:
        img = Image.open(img_file)
        ocr = pytesseract.image_to_string(img)
        print('Scanned result for {}: \n{}'.format(img_file, ocr))

#    img.save("processed_images/proc_camera-test.jpg")
except IOError:
    pass

