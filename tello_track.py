import threading
import time
import imutils
import cv2
import numpy as np

from easytello import Tello
from pid import PID
from queue import Queue
from object_detection.utils import label_map_util
from simple_multi_tracker import SimpleMultiTracker
from detection import DetectionModel, load_model, run_inference_for_single_image

WAIT = 0
DETECTING = 1
TRACKING = 2
INPUTTING = 3


class TelloTracker:
    def __init__(self, detect_model, tello, frame_width=768, skip_frames=5, action_frames=5, confidence=0.4,
                 frame_rate=30, pool_size=30, class_set=['toycar'], output_video=None):
        self.detect_model = detect_model
        self.tello = tello
        self.frame_width = frame_width
        self.skip_frames = skip_frames
        self.confidence = confidence
        self.output_video = output_video
        self.frame_rate = frame_rate
        self.action_frames = action_frames
        self.class_set = class_set
        self.pool_size = pool_size

        # self.frame = None
        self.init_area = None
        self.init_y_ratio = None
        self.area_foward_pid = PID(setPoint=1.0, sample_time=0.1, P=50, I=0, D=0)
        # self.foward_pid = PID(setPoint=1.0, sample_time=0.1, P=30, I=5, D=10)
        self.rotate_pid = PID(setPoint=0.5, sample_time=0.1, P=100, I=20, D=30)
        # self.rotate_pid = PID(setPoint=0.5, sample_time=0.1, P=150, I=30, D=60)
        self.y_foward_pid = PID(setPoint=0.0, sample_time=0.1, P=100, I=20, D=30)
        self.frame_pool = Queue()
        self.stopEvent = threading.Event()

        self.horizontal_speed = 0
        self.forward_speed = 0
        self.accelerator = 0
        self.rotate = 0

        # meters
        self.distance = 0.5
        self.degree = 10

        self.video_thread = threading.Thread(target=self._videoLoop, args=())
        # self.video_thread.daemon = True
        self.video_thread.start()

        # self.battery_command_thread = threading.Thread(target=self._sendingCommand)
        # self.battery_command_thread.daemon = True
        # self.battery_command_thread.start()

    def _get_pooled_frame(self):
        if self.frame_pool.qsize() != 0:
            return self.frame_pool.get()

    def _videoLoop(self):
        """
        a thread used to read frame from tello object
        """
        try:
            # time.sleep(0.5)
            # self.sending_command_thread.start()
            # self.sending_command_thread.start()
            while not self.stopEvent.is_set():
                # read the frame for GUI show
                # self.frame = np.array(self.tello.read()).astype(int)
                frame = self.tello.frame
                if frame is None or frame.size == 0:
                    # print('no frame!')
                    continue
                # self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # print('poll frame!')
                if self.frame_pool.qsize() > self.pool_size:
                    self.frame_pool.get()
                    print('lost frame!')
                self.frame_pool.put(frame)
                # control frame rate
                time.sleep(1 / self.frame_rate)
        except RuntimeError as e:
            print("[INFO] caught a RuntimeError")

    def _close(self):
        """
        set the stop event, cleanup the camera
        """
        print("[INFO] closing...")
        self.stopEvent.set()
        # del self.tello

    def _regionCheck(self, region, min=-50, max=50):
        if region < min:
            region = min
        elif region > max:
            region = max
        return int(region)

    def _pid_init_lasttime(self):
        self.area_foward_pid.init_last_time()
        self.y_foward_pid.init_last_time()
        self.rotate_pid.init_last_time()

    def _pid_clear(self):
        self.area_foward_pid.clear()
        self.y_foward_pid.clear()
        self.rotate_pid.clear()

    def _sendingCommand(self):
        """
        start a while loop that sends 'command' to tello every 30 second
        """

        while True:
            self.tello.get_battery()
            # self.tello.send_command('command')
            time.sleep(30)

    def track(self, verbose=True):
        # 初始要追踪的目标ID
        track_id = None
        multi_tracker = SimpleMultiTracker(debug=verbose)
        display_str = 'waiting'
        mode = WAIT
        width, height = None, None
        writer = None
        total_frames = 0

        # loop over frames from the video stream
        while True:
            # grab the next frame
            frame = self._get_pooled_frame()
            if frame is None or frame.size == 0:
                continue

            # resize the frame to have a maximum width of frame_width
            frame = imutils.resize(frame, width=self.frame_width)

            # if the frame dimensions are empty, set them
            if width is None or height is None:
                (height, width) = frame.shape[:2]

            # if we are supposed to be writing a video to disk, initialize the writer
            if self.output_video is not None and writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MP4V")
                # fourcc = cv2.VideoWriter_fourcc(*"H264")
                writer = cv2.VideoWriter(self.output_video, fourcc, 30,
                                         (width, height), True)

            if mode != WAIT:
                # check to see if we should run a more computationally expensive
                # object detection method to aid our tracker
                if total_frames % self.skip_frames == 0:
                    raw_boxes, scores, classes = self.detect_model.detect(frame)
                    print(scores[:5], classes[:5], raw_boxes[:5])

                    # loop over the detections
                    boxes = []
                    for i in range(len(scores)):
                        # filter out weak detections by requiring a minimum onfidence
                        if scores[i] > self.confidence:
                            # if the class label is not a person, ignore it
                            if classes[i] in self.class_set:
                                boxes.append(raw_boxes[i])

                    if total_frames == 0:
                        rects = multi_tracker.init_tracker(frame, boxes)
                    else:
                        rects = multi_tracker.update_tracker(frame, boxes)

                    if mode == TRACKING:
                        if track_id not in multi_tracker.tracked:
                            display_str = 'LOST TARGET'
                            mode = DETECTING
                        else:
                            if multi_tracker.tracked[track_id]:
                                # if tracked after detection, init a new single tracker
                                multi_tracker.init_single_tracker(track_id, frame)
                            else:
                                multi_tracker.update_single_tracker(frame)
                else:
                    if mode == TRACKING:
                        if track_id not in multi_tracker.tracked:
                            display_str = 'LOST TARGET'
                            mode = DETECTING
                        else:
                            multi_tracker.update_single_tracker(frame)
                    rects = multi_tracker.appearing_boxes()

                if verbose:
                    multi_tracker.show_info('total frame:{}'.format(total_frames))
                # draw rectangles of corresponding objects
                for rect in rects.values():
                    (startX, startY, endX, endY) = rect
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

                # draw both the ID of the object and the centroid of the object on the output frame
                for (objectID, box) in rects.items():
                    (startX, startY, endX, endY) = box
                    cX = int((startX + endX) / 2.0)
                    cY = int((startY + endY) / 2.0)
                    centroid = (cX, cY)

                    text = "ID {}".format(objectID)
                    cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.circle(frame, (centroid[0], centroid[1]), 2, (0, 0, 255), -1)

                # decide what action to choose
                if total_frames % self.action_frames == 0:
                    box = multi_tracker.appearing_tracking_box()
                    if box is not None:
                        (startX, startY, endX, endY) = box
                        area = (endX - startX) * (endY - startY)

                        cX = (startX + endX) / 2.0
                        cY = (startY + endY) / 2.0
                        x_ratio = cX / width
                        y_ratio = cY / height

                        if self.init_area is None:
                            # it's the first time to get a box, so we record it as a target for later use.
                            self.init_area = area
                            self.init_y_ratio = y_ratio
                            self._pid_init_lasttime()
                        else:
                            # using pid algorithms to calculate a proper argument to send to tello
                            area_ratio = float(area) / float(self.init_area)
                            y_error = y_ratio - self.init_y_ratio

                            cur_time = time.time()
                            area_forward_speed = self.area_foward_pid.update(area_ratio, cur_time)
                            y_forward_speed = self.y_foward_pid.update(y_error, cur_time)
                            self.forward_speed = 0.5 * area_forward_speed + y_forward_speed * 0.5
                            # self.foward_speed = self.foward_pid.update(y_ratio, cur_time)
                            self.rotate = self.rotate_pid.update(1 - x_ratio, cur_time)

                            # too close to tracking object
                            if endY > 0.9 * height and self.forward_speed > 0:
                                self.forward_speed = -0.3 * self.forward_speed

                            if self.forward_speed is not None and self.horizontal_speed is not None:
                                self.tello.control(foward_speed=self._regionCheck(self.forward_speed),
                                                   rotate=self._regionCheck(self.rotate))
                    else:
                        # no tracking object, stop any moving
                        self.tello.control()
                        self._pid_clear()
                        self._pid_init_lasttime()

                # increment the total number of frames processed thus far
                total_frames += 1

            cv2.putText(frame, display_str, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            # show the output frame
            cv2.imshow('frame', frame)

            # if the `q` key was pressed, break from the loop
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == 32:
                # space key
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
                    if track_id in multi_tracker.tracked.keys():
                        multi_tracker.init_single_tracker(track_id, frame)
                    else:
                        mode = DETECTING
                        display_str = 'NO SUCH ID'
            elif ord('0') <= key <= ord('9') and mode == INPUTTING:
                num = key - ord('0')
                display_str += str(num)
            elif key == ord("t"):
                self.tello.takeoff()
            elif key == ord("w"):
                self.tello.forward(20)
            elif key == ord("a"):
                self.tello.ccw(30)
            elif key == ord("s"):
                self.tello.back(20)
            elif key == ord("d"):
                self.tello.cw(30)
            elif key == ord('l'):
                self.tello.land()
            elif key == ord('e'):
                self.tello.emergency()
            else:
                if key != 255:
                    print('unused key:{}'.format(key))

            # check to see if we should write the frame to disk
            if writer is not None:
                writer.write(frame)

            # elif key == ord("u"):
            #     self.foward_speed += 10
            #     self.tello.set_control(self.horizontal_speed, self.foward_speed, self.accelerator, self.rotate)
            # elif key == ord("j"):
            #     self.foward_speed -= 10
            #     self.tello.set_control(self.horizontal_speed, self.foward_speed, self.accelerator, self.rotate)
            # elif key == ord("h"):
            #     self.horizontal_speed -= 10
            #     self.tello.set_control(self.horizontal_speed, self.foward_speed, self.accelerator, self.rotate)
            # elif key == ord("k"):
            #     self.horizontal_speed += 10
            #     self.tello.set_control(self.horizontal_speed, self.foward_speed, self.accelerator, self.rotate)

        # check to see if we need to release the video writer pointer
        if writer is not None:
            writer.release()

        # close any open windows
        cv2.destroyAllWindows()
        self._close()


class dumy_tello():
    def __init__(self):
        self.frame = np.random.randn(300, 300)


if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO)
    # logger = logging.getLogger('track_demo')

    PATH_TO_LABELS = 'export/label_map.pbtxt'
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    detection_model = load_model()
    print('---warming up detection model---')
    run_inference_for_single_image(detection_model, cv2.imread('./test_images/IMG_20200428_144451.jpg'))

    detection_model = DetectionModel(detection_model, category_index)

    tello = Tello()
    tello.command()
    tello.streamon()
    # tello.get_battery()
    # a = dumy_tello()
    # a.frame = np.random.randn(300, 300)
    tello_tracker = TelloTracker(detection_model, tello, output_video='out.mp4')
    try:
        tello_tracker.track(verbose=True)
        tello.land()
        time.sleep(0.75)
        tello.emergency()
    except Exception as e:
        tello.emergency()
        raise e
