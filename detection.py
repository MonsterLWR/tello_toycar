import numpy as np
import pathlib
import tensorflow as tf

from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile


class DetectionModel:
    def __init__(self, model, category_map):
        self.model = model
        self.category_map = category_map

    def detect(self, image):
        image = np.asarray(image)
        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        # Run inference
        output_dict = self.model(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(output_dict.pop('num_detections'))
        output_dict = {key: value[0, :num_detections].numpy()
                       for key, value in output_dict.items()}
        output_dict['num_detections'] = num_detections

        # detection_classes should be ints.
        output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

        name_list = []
        for cls_ind in output_dict['detection_classes']:
            if cls_ind in self.category_map.keys():
                name_list.append(self.category_map[cls_ind]['name'])
            else:
                name_list.append('???')
        boxes = boxes_float_boxes_to_int(output_dict['detection_boxes'], image.shape[0], image.shape[1])
        return boxes, output_dict['detection_scores'], name_list


def boxes_float_boxes_to_int(boxes, height, width):
    int_boxes = []
    for box in boxes:
        # x1, y1, x2, y2 = box
        y1, x1, y2, x2 = box
        int_box = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)
        int_boxes.append(int_box)
    return int_boxes


def load_model(model_dir='./export'):
    # base_url = 'http://download.tensorflow.org/models/object_detection/'
    # model_file = model_name + '.tar.gz'
    # model_dir = tf.keras.utils.get_file(
    #     fname=model_name,
    #     origin=base_url + model_file,
    #     untar=True)

    model_dir = pathlib.Path(model_dir) / "saved_model"

    model = tf.saved_model.load(str(model_dir))
    model = model.signatures['serving_default']

    return model


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    output_dict = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict


def show_inference(model, image_path):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = np.array(Image.open(image_path))
    # Actual detection.
    output_dict = run_inference_for_single_image(model, image_np)
    print(output_dict)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8)

    # display(Image.fromarray(image_np))
    # Image.save(image_np)

    return image_np


if __name__ == '__main__':
    # load_model()

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = 'export/label_map.pbtxt'
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
    PATH_TO_TEST_IMAGES_DIR = pathlib.Path('test_images')
    TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))

    detection_model = load_model()
    print(detection_model.inputs)
    # detection_model.output_dtypes
    # detection_model.output_shapes
    image_path = 'test_images/test.JPG'
    image = show_inference(detection_model, image_path)
    plt.imshow(image)
    plt.show()

    # run_inference_for_single_image()
    # image_nps = []
    # for image_path in TEST_IMAGE_PATHS:
    #     image_nps.append(show_inference(detection_model, image_path))
    #
    # ax1 = plt.subplot(121)
    # ax2 = plt.subplot(122)
    #
    # plt.axes(ax1)
    # plt.imshow(image_nps[0])
    # plt.axes(ax2)
    # plt.imshow(image_nps[1])
    #
    # plt.show()
