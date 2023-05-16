import unittest, cv2
import numpy as np
import os.path


class TestImageProcessing(unittest.TestCase):
    def setUp(self):
        self.img_file = 'C:/Users/Radu/Downloads/capture-test.jpg'
        self.processed_dir = 'processed_images'
        
    def test_invert_image(self):
        # Test if the image is successfully inverted
        image = cv2.imread(self.img_file)
        inverted_image = cv2.bitwise_not(image)
        inverted_file = os.path.join(self.processed_dir, 'inverted_capture-test.jpg')
        cv2.imwrite(inverted_file, inverted_image)
        
        self.assertTrue(os.path.exists(inverted_file))
        
    def test_binarization(self):
        # Test if the image is successfully binarized
        image = cv2.imread(self.img_file)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binarized_image = cv2.threshold(gray_image, 45, 29, cv2.THRESH_BINARY)[1]
        binarized_file = os.path.join(self.processed_dir, 'binarized_capture-test.jpg')
        cv2.imwrite(binarized_file, binarized_image)
        
        self.assertTrue(os.path.exists(binarized_file))
        
    def test_noise_removal(self):
        # Test if noise is successfully removed from the image
        image = cv2.imread(self.img_file)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binarized_image = cv2.threshold(gray_image, 45, 29, cv2.THRESH_BINARY)[1]
        noise_removed_image = cv2.medianBlur(binarized_image, 3)
        noise_removed_file = os.path.join(self.processed_dir, 'noise_removed_capture-test.jpg')
        cv2.imwrite(noise_removed_file, noise_removed_image)
        
        self.assertTrue(os.path.exists(noise_removed_file))
        
    # Add more test methods for other image processing functions
        
    def tearDown(self):
        # Delete all processed images created during testing
        for file in os.listdir(self.processed_dir):
            os.remove(os.path.join(self.processed_dir, file))

if __name__ == '__main__':
    unittest.main()
