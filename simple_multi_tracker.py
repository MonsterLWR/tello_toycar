# coding=utf-8
from scipy.spatial import distance as dist
from my_tracker.utils import *
import numpy as np
import cv2


class SimpleMultiTracker:
    def __init__(self, tracker_type='kcf', **kwargs):
        self.nextTrackerID = 0
        self.trackers = {}
        self.boxes = {}
        self.counts = {}
        # 检测算法未检测到目标的次数
        self.disappear = {}
        # 目标是否被追踪到
        self.tracked = {}
        # 是否使用相关滤波器进行匹配
        # self.use_CF = use_CF
        # self.template_denominator = {}
        # self.template_numerator = {}
        # self.last_frame = None
        # self.filter_size = kwargs.setdefault('filter_size', (64, 64))
        # cv2.imshow('g', gauss_response(self.filter_size[0], self.filter_size[1]))

        self.maxDistance = kwargs.setdefault('maxDistance', 200)
        self.maxDisappear = kwargs.setdefault('maxDisappear', 6)
        self.areaThreshold = kwargs.setdefault('areaThreshold', 50)
        # psr为8可能是追踪目标，也可能不是追踪目标。结合距离来判断是否为追踪目标。
        # self.psrThreshold = kwargs.setdefault('psrThreshold', 7.0)
        # 用于判断追踪的是否是同一目标
        # self.updatePsr = kwargs.setdefault('updatePsr', 12.0)
        # self.areaChangeRatio_max = kwargs.setdefault('areaChangeRatio_max', 3.0)
        # self.areaChangeRatio_min = kwargs.setdefault('areaChangeRatio_min', 0.5)
        # self.distancePenalty = self.updatePsr - self.psrThreshold

        # 每次追踪的目标匹配为检测到的目标时，由于框的大小不同，可能导致psr也比较小。
        # 该值用于每次追踪的目标匹配为检测到的目标时，可以无视psr的值，更新滤波器的次数。
        # self.safeUpdateCount = kwargs.setdefault('safeUpdateCount', 0)

        self.tracker_type = tracker_type

        self.debug = kwargs.setdefault('debug', True)

        OPENCV_OBJECT_TRACKERS = {
            "csrt": cv2.TrackerCSRT_create,
            "kcf": cv2.TrackerKCF_create,
            "boosting": cv2.TrackerBoosting_create,
            "mil": cv2.TrackerMIL_create,
            "tld": cv2.TrackerTLD_create,
            "medianflow": cv2.TrackerMedianFlow_create,
            "mosse": cv2.TrackerMOSSE_create
        }
        self.tracker = OPENCV_OBJECT_TRACKERS[tracker_type]

        self.single_tracker = None
        self.single_tracker_id = None
        # self.pool = multiprocessing.Pool()

    def __register(self, frame, box):
        """通过frame,box,新建并初始化一个tracker"""
        # tracker = self.new_tracker()
        # tracker.init(frame, convert_to_wh_box(box))
        # tracker.init(frame, box)
        # self.trackers[self.nextTrackerID] = tracker

        self.tracked[self.nextTrackerID] = True
        self.disappear[self.nextTrackerID] = 0
        self.boxes[self.nextTrackerID] = box
        # self.counts[self.nextTrackerID] = self.safeUpdateCount
        self.nextTrackerID += 1

    def init_single_tracker(self, trackerID, frame):
        self.single_tracker = self.tracker()
        self.single_tracker.init(frame, convert_to_wh_box(self.boxes[trackerID]))
        self.single_tracker_id = trackerID

    def update_single_tracker(self, frame):
        # this function won't affect disappear variable
        if self.single_tracker_id in self.tracked.keys():
            success, box = self.single_tracker.update(frame)
            if success:
                self.boxes[self.single_tracker_id] = convert_from_wh_box(box)
                self.tracked[self.single_tracker_id] = True

    def __renew_tracker(self, trackerID, frame, box):
        """重置tracker"""
        # tracker = self.new_tracker()
        # tracker.init(frame, convert_to_wh_box(box))
        # self.trackers[trackerID] = tracker

        self.tracked[trackerID] = True
        self.disappear[trackerID] = 0
        self.boxes[trackerID] = box
        # self.counts[trackerID] = self.safeUpdateCount

    def __deregister(self, trackerID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        # del self.trackers[trackerID]
        del self.boxes[trackerID]
        del self.disappear[trackerID]
        del self.tracked[trackerID]
        # del self.counts[trackerID]

    def init_tracker(self, frame, boxes):
        """初始化manager,注册可能作为追踪目标的trackerID"""
        boxes = discard_boxes(boxes, self.areaThreshold, frame.shape[0], frame.shape[1])

        # if len(self.trackers.keys()) == 0:
        for box in boxes:
            self.__register(frame, box)
            # self.last_frame = frame.copy()

        return self.appearing_boxes()

    def update_tracker(self, frame, boxes):
        existing_IDs = list(self.boxes.keys())
        existing_boxes = list(self.boxes.values())
        existing_centroids = compute_centroids(existing_boxes)
        input_centroids = compute_centroids(boxes)

        unusedRows = range(len(existing_boxes))
        unusedCols = range(len(boxes))
        # 计算目前的跟踪目标的中点和检测目标的中点的距离矩阵
        D = dist.cdist(existing_centroids, input_centroids)
        if D.size > 0:
            # 根据每一行上的最小值对行索引进行排序
            rows = np.min(D, axis=1).argsort()
            # 得到每一行上最小值对应的列索引，并依据rows重新排序
            cols = np.argmin(D, axis=1)[rows]

            # self.__show_table(D, existing_IDs)

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                if D[row, col] > self.maxDistance:
                    continue

                self.__renew_tracker(existing_IDs[row], frame, boxes[col])

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

        # 没有对应检测到的目标 的tracker
        for row in unusedRows:
            ID = existing_IDs[row]
            self.disappear[ID] += 1
            if self.disappear[ID] >= self.maxDisappear:
                # 超出设定的消失次数，销毁tracker
                self.__deregister(ID)

        # 注册新的目标
        unusedCols = list(unusedCols)
        for col in unusedCols:
            box = boxes[col]
            self.__register(frame, box)

        # def update_trackers(self, frame):
        #     """使用tracker追踪目标"""
        #     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #     for (ID, tracker) in self.trackers.items():
        #         (success, box) = tracker.update(frame)
        #         # print(success, box)
        #         if success:
        #             self.boxes[ID] = clip_box(convert_from_wh_box(box), frame.shape[0], frame.shape[1])
        #             self.tracked[ID] = True
        #         else:
        #             # 追踪失败
        #             self.tracked[ID] = False
        #     # self.last_frame = frame.copy()
        return self.appearing_boxes()

    def appearing_boxes(self):
        boxes = {}
        for ID, tracked in self.tracked.items():
            if tracked:
                boxes[ID] = self.boxes[ID]
        return boxes

    def appearing_tracking_box(self):
        if self.single_tracker_id in self.tracked.keys() and self.tracked[self.single_tracker_id]:
            return self.boxes[self.single_tracker_id]
        else:
            return None

    def show_info(self, mes_title):
        print('--------------------{}--------------------'.format(mes_title))
        # print('trackers:{}'.format(self.trackers))
        print('tracked:{}'.format(self.tracked))
        print('boxes:{}'.format(self.boxes))
        print('disappear:{}'.format(self.disappear))

    # def __update_template(self, ID, gray_frame, rate=0.125):
    #     # 更新滤波器模板
    #     startX, startY, endX, endY = self.boxes[ID]
    #     hi = gray_frame[startY:endY, startX:endX]
    #     hi = cv2.resize(hi, self.filter_size)
    #     numerator, denominator = correlation_filter(hi, self.fft_gauss_response)
    #
    #     self.template_numerator[ID] = rate * numerator + (1 - rate) * self.template_numerator[ID]
    #     self.template_denominator[ID] = rate * denominator + (1 - rate) * self.template_denominator[ID]

    # def __set_templates(self, gray_frame):
    #     # 对要追踪的id
    #     for ID in self.tracked.keys():
    #         # 对于追踪到的目标构建相关滤波模板
    #         # 没有追踪到的目标还需要依靠现有的滤波器来寻找目标
    #         if self.tracked[ID]:
    #             startX, startY, endX, endY = self.boxes[ID]
    #             hi = gray_frame[startY:endY, startX:endX]
    #             hi = cv2.resize(hi, self.filter_size)
    #             self.template_numerator[ID], self.template_denominator[ID] = correlation_filter(hi,
    #                                                                                             self.fft_gauss_response)
