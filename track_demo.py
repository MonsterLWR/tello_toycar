from simple_multi_tracker import SimpleMultiTracker
from imutils.video import VideoStream
from imutils.video import FPS
from my_tracker.utils import pop_up_box
import imutils
import time
import cv2
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
import pathlib
from detection import DetectionModel, load_model, run_inference_for_single_image
import tensorflow as tf
import logging

WAIT = 0
DETECTING = 1
TRACKING = 2
INPUTTING = 3


def track(detection_model, class_set, input_vedio=None, output_vedio=None, frame_width=400, skip_frames=2,
          confidence=0.4, verbose=True):
    if input_vedio is None:
        print("[INFO] starting video stream...")
        vs = VideoStream(src=0).start()
        time.sleep(2.0)
    else:
        print("[INFO] opening video file...")
        vs = cv2.VideoCapture(input_vedio)

    W = None
    H = None
    writer = None
    fps = None
    track_id = None

    totalFrames = 0
    display_str = 'waiting'
    mode = WAIT
    multi_tracker = SimpleMultiTracker(debug=verbose)
    # loop over frames from the video stream
    while True:
        # grab the next frame and handle if we are reading from either
        # VideoCapture or VideoStream
        frame = vs.read()
        frame = frame[1] if input_vedio is not None else frame

        # if we are viewing a video and we did not grab a frame then we
        # have reached the end of the video
        if input_vedio is not None and frame is None:
            break

        # resize the frame to have a maximum width of frame_width
        frame = imutils.resize(frame, width=frame_width)

        # if the frame dimensions are empty, set them
        if W is None or H is None:
            (H, W) = frame.shape[:2]
            # detection_model.performDetect(None, initOnly=True)

        # if we are supposed to be writing a video to disk, initialize
        # the writer
        if output_vedio is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MP4V")
            writer = cv2.VideoWriter(output_vedio, fourcc, 30,
                                     (W, H), True)

        if mode != WAIT:
            # check to see if we should run a more computationally expensive
            # object detection method to aid our tracker
            if totalFrames % skip_frames == 0:
                raw_boxes, scores, classes = detection_model.detect(frame)
                # print(classes)

                # loop over the detections
                boxes = []
                for i in range(len(scores)):
                    if scores[i] > confidence:
                        if classes[i] in class_set:
                            boxes.append(raw_boxes[i])

                if totalFrames == 0:
                    rects = multi_tracker.init_tracker(frame, boxes)
                else:
                    rects = multi_tracker.update_tracker(frame, boxes)

                if mode == TRACKING:
                    if multi_tracker.tracked[track_id]:
                        # if tracked after detection, init a new single tracker
                        multi_tracker.init_single_tracker(track_id, frame)
                    else:
                        multi_tracker.update_single_tracker(frame)
            # otherwise, we should utilize our object *trackers* rather than
            # object *detectors* to obtain a higher frame processing throughput
            else:
                if mode == TRACKING:
                    multi_tracker.update_single_tracker(frame)
                rects = multi_tracker.appearing_boxes()

            if rects is not None:
                for rect in rects.values():
                    (startX, startY, endX, endY) = rect
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                                  (0, 255, 0), 2)

                # loop over the tracked objects
                for (objectID, box) in rects.items():
                    (startX, startY, endX, endY) = box
                    cX = int((startX + endX) / 2.0)
                    cY = int((startY + endY) / 2.0)
                    centroid = (cX, cY)

                    # draw both the ID of the object and the centroid of the
                    # object on the output frame
                    text = "ID {}".format(objectID)
                    cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.circle(frame, (centroid[0], centroid[1]), 2, (0, 0, 255), -1)

            if totalFrames == 0:
                cv2.imshow("Frame", frame)
                # selected_IDs.append(pop_up_box())
                fps = FPS().start()

            # increment the total number of frames processed thus far and
            # then update the FPS counter
            totalFrames += 1
            fps.update()

            if verbose:
                multi_tracker.show_info('Frame{}'.format(totalFrames))

        if writer is not None:
            writer.write(frame)

        cv2.putText(frame, display_str, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imshow("Frame", frame)

        if mode == WAIT:
            key = cv2.waitKey(33) & 0xFF
        else:
            key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        elif key == ord("d"):
            mode = DETECTING
            display_str = 'DETECTING'
        elif key == 13:
            # enter key
            if mode == DETECTING:
                mode = INPUTTING
                display_str = 'INPUTTING:'
            elif mode == INPUTTING:
                track_id = int(display_str.split(':')[-1])
                mode = TRACKING
                display_str = 'TRACKING:{}'.format(track_id)
                multi_tracker.init_single_tracker(track_id, frame)
        elif ord('0') <= key <= ord('9') and mode == INPUTTING:
            num = key - ord('0')
            display_str += str(num)
        else:
            if key != 255:
                print('unused key:{}'.format(key))

    # stop the timer and display FPS information
    if fps is not None:
        fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # check to see if we need to release the video writer pointer
    if writer is not None:
        writer.release()

    # if we are not using a video file, stop the camera video stream
    if input_vedio is None:
        vs.stop()

    # otherwise, release the video file pointer
    else:
        vs.release()

    # close any open windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('track_demo')

    PATH_TO_LABELS = 'export/label_map.pbtxt'
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    detection_model = load_model()
    logger.info('---warming up detection model---')
    run_inference_for_single_image(detection_model, cv2.imread('./test_images/IMG_20200428_144451.jpg'))

    detection_model = DetectionModel(detection_model, category_index)

    track(detection_model, ['toycar'], 'test.mp4', 'out1.mp4', skip_frames=3, verbose=False)
