import tempfile
from PIL import Image
import cv2
import os


# A stupid trick to force palette into cv2
def cv2palette(image, palette):
    img_E = Image.fromarray(image)
    img_E.putpalette(palette)
    with tempfile.TemporaryDirectory() as tmppath:
        full_path = os.path.join(tmppath, 'temp.png')
        img_E.save(full_path)
        image = cv2.imread(full_path)
    return image