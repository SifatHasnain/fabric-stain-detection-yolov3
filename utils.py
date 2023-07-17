import base64
import numpy as np
import cv2


def load_image(data):
    base64_image = base64.b64decode(data)
    image_np = np.fromstring(base64_image, dtype=np.uint8)
    image = cv2.imdecode(image_np, flags=cv2.IMREAD_COLOR)
    return image

def convert_image_to_base64(image_array):
    image_string = base64.b64encode(image_array).decode("utf-8")
    return image_string
