import os
from datetime import datetime
import time
import cv2
import darknet
from random import randint

from utils import load_image, convert_image_to_base64


def image_detection(image, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    image = darknet.draw_boxes(detections, image_resized, class_colors)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections


def convert2relative(image, bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h = bbox
    height, width, _ = image.shape
    return x/width, y/height, w/width, h/height


def save_annotations(name, image, detections, class_names):
    """
    Files saved with image_name.txt and relative coordinates
    """
    file_name = os.path.splitext(name)[0] + ".txt"
    with open(file_name, "w") as f:
        for label, confidence, bbox in detections:
            x, y, w, h = convert2relative(image, bbox)
            label = class_names.index(label)
            f.write("{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(label, x, y, w, h, float(confidence)))


def detect_stain(image, save_labels=False, save_image=False):
    network, class_names, class_colors = darknet.load_network(
        'cfg/custom.cfg',
        'obj/obj.data',
        'backup/custom.weights'
    )

    image = load_image(image)

    start_time = time.time()
    annotated_image, detections = image_detection(
        image, network, class_names, class_colors, 0.25
    )

    if save_image:
        filename = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(randint(000, 999))
        cv2.imwrite(filename+'.jpg', annotated_image)

    if save_labels:
        filename = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(randint(000, 999))
        save_annotations(filename, image, detections, class_names)

    darknet.print_detections(detections, True)

    print("Execution time(Stain Detection)(s): ", time.time() - start_time)

    annotated_image_string = convert_image_to_base64(annotated_image)

    return annotated_image_string, detections
