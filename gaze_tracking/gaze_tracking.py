from __future__ import division
import os
import cv2
import dlib
import numpy as np
from .eye import Eye
from .calibration import Calibration


class GazeTracking(object):
    """
    This class tracks the user's gaze.
    It provides useful information like the position of the eyes
    and pupils and allows to know if the eyes are open or closed
    """

    def __init__(self):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.calibration = Calibration()
        
        # 시선 방향 감지 설정
        self.direction_threshold = 0.1  # 방향 감지 임계값
        self.smoothing_factor = 0.3    # 부드러운 전환을 위한 계수
        self.last_direction = 'center'  # 마지막 감지된 방향
        self.direction_history = []     # 방향 기록
        self.history_size = 5          # 기록 유지 크기

        # _face_detector is used to detect faces
        self._face_detector = dlib.get_frontal_face_detector()

        # _predictor is used to get facial landmarks of a given face
        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.abspath(os.path.join(cwd, "trained_models/shape_predictor_68_face_landmarks.dat"))
        self._predictor = dlib.shape_predictor(model_path)

    @property
    def pupils_located(self):
        """Check that the pupils have been located"""
        try:
            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)
            return True
        except Exception:
            return False

    def _analyze(self):
        """Detects the face and initialize Eye objects"""
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_detector(frame)

        try:
            landmarks = self._predictor(frame, faces[0])
            self.eye_left = Eye(frame, landmarks, 0, self.calibration)
            self.eye_right = Eye(frame, landmarks, 1, self.calibration)

        except IndexError:
            self.eye_left = None
            self.eye_right = None

    def refresh(self, frame):
        """Refreshes the frame and analyzes it.

        Arguments:
            frame (numpy.ndarray): The frame to analyze
        """
        self.frame = frame
        self._analyze()

    def pupil_left_coords(self):
        """Returns the coordinates of the left pupil"""
        if self.pupils_located:
            x = self.eye_left.origin[0] + self.eye_left.pupil.x
            y = self.eye_left.origin[1] + self.eye_left.pupil.y
            return (x, y)

    def pupil_right_coords(self):
        """Returns the coordinates of the right pupil"""
        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_right.pupil.x
            y = self.eye_right.origin[1] + self.eye_right.pupil.y
            return (x, y)

    def horizontal_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        horizontal direction of the gaze. The extreme right is 0.0,
        the center is 0.5 and the extreme left is 1.0
        """
        if self.pupils_located:
            # 왼쪽 눈과 오른쪽 눈의 동공 위치를 정규화
            pupil_left = self.eye_left.pupil.x / (self.eye_left.center[0] * 2 - 10)
            pupil_right = self.eye_right.pupil.x / (self.eye_right.center[0] * 2 - 10)
            
            # 두 눈의 평균값 계산
            ratio = (pupil_left + pupil_right) / 2
            
            # 부드러운 전환을 위한 스무딩 적용
            if hasattr(self, 'last_ratio'):
                ratio = (ratio * self.smoothing_factor) + (self.last_ratio * (1 - self.smoothing_factor))
            self.last_ratio = ratio
            
            return ratio

    def vertical_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        vertical direction of the gaze. The extreme top is 0.0,
        the center is 0.5 and the extreme bottom is 1.0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.y / (self.eye_left.center[1] * 2 - 10)
            pupil_right = self.eye_right.pupil.y / (self.eye_right.center[1] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def get_direction(self):
        """현재 시선 방향을 반환합니다."""
        if not self.pupils_located:
            return 'unknown'
            
        ratio = self.horizontal_ratio()
        
        # 방향 결정
        if ratio <= 0.40:
            direction = 'right'
        elif ratio >= 0.70:
            direction = 'left'
        else:
            direction = 'center'
            
        # 방향 기록 업데이트
        self.direction_history.append(direction)
        if len(self.direction_history) > self.history_size:
            self.direction_history.pop(0)
            
        # 가장 많이 감지된 방향 반환
        if len(self.direction_history) >= 3:
            direction_counts = {
                'left': self.direction_history.count('left'),
                'right': self.direction_history.count('right'),
                'center': self.direction_history.count('center')
            }
            return max(direction_counts, key=direction_counts.get)
            
        return direction

    def is_right(self):
        """Returns true if the user is looking to the right"""
        return self.get_direction() == 'right'

    def is_left(self):
        """Returns true if the user is looking to the left"""
        return self.get_direction() == 'left'

    def is_center(self):
        """Returns true if the user is looking to the center"""
        return self.get_direction() == 'center'

    def is_blinking(self):
        """Returns true if the user closes his eyes"""
        if self.pupils_located:
            blinking_ratio = (self.eye_left.blinking + self.eye_right.blinking) / 2
            return blinking_ratio > 3.8

    def annotated_frame(self):
        """Returns the main frame with pupils highlighted"""
        frame = self.frame.copy()

        if self.pupils_located:
            # 동공 위치 표시
            color = (0, 255, 0)
            x_left, y_left = self.pupil_left_coords()
            x_right, y_right = self.pupil_right_coords()
            cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
            cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
            cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
            cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)
            
            # 현재 방향 표시
            direction = self.get_direction()
            cv2.putText(frame, f"Direction: {direction}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 수평 비율 표시
            ratio = self.horizontal_ratio()
            cv2.putText(frame, f"Ratio: {ratio:.2f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame
