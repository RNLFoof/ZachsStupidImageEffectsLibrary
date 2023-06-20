import inspect
import os
import unittest

from PIL import Image
import sys
sys.path.append("..") # Adds higher directory to python modules path.
import zsil.colors



class MyTestCase(unittest.TestCase):
    def assert_image_matches_expectation(self, img: Image):
        stack = inspect.stack()
        path = os.path.join("expected_output", stack[1].function)
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, os.path.split(img.filename)[1])

        if os.path.exists(path):
            saved_image = Image.open(path)
            self.assertEqual(saved_image, img)
        else:
            img.save(path)

    def test_get_colors_dict(self):
        self.assert_image_matches_expectation(Image.open("test_images/hippo.png"))


if __name__ == '__main__':
    unittest.main()
