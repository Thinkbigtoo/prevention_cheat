# --- 라이브러리 임포트 ---
import cv2
import numpy as np
import math
import time
import threading
from collections import deque
from picamera2 import Picamera2
import mediapipe as mp
import mariadb
from gpiozero import LED
import pigpio
import RPi.GPIO as GPIO
import atexit

# --- GPIO 핀 설정 ---
TRIG = 23            # 초음파 센서 트리거 핀
ECHO = 24            # 초음파 센서 에코 핀
RED_LED_PIN = 17     # 빨간 LED 핀
GREEN_LED_PIN = 18   # 초록 LED 핀

# --- 하드웨어 초기화 ---
red_led = LED(RED_LED_PIN)
green_led = LED(GREEN_LED_PIN)
pi = pigpio.pi()
picam2 = Picamera2()

# --- GPIO 설정 ---
GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

# --- MariaDB 연결 함수 ---
def connect_db():
    try:
        conn = mariadb.connect(
            user="kgupi",
            password="capstone",
            host="172.20.10.5",
            port=3306,
            database="TestMonitoringDB"
        )
        return conn
    except mariadb.Error as e:
        print(f"Error connecting to MariaDB: {e}")
        return None

# --- 초음파 센서를 통한 거리 측정 함수 ---
def measure_distance():
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    start_time = time.time()
    stop_time = time.time()

    while GPIO.input(ECHO) == 0:
        start_time = time.time()
    while GPIO.input(ECHO) == 1:
        stop_time = time.time()

    duration = stop_time - start_time
    distance = duration * 17150  # 왕복 시간 → 거리(cm) 변환
    return distance

# --- DB에 상태 정보 기록 함수 ---
def insert_status(conn, status):
    try:
        print(f"[DB LOG] Trying to insert status: {status}")
        cursor = conn.cursor()
        cursor.execute("INSERT INTO status_table (status_value, timestamp) VALUES (?, NOW())", (status,))
        conn.commit()
        print("[DB LOG] Insert successful.")
    except mariadb.Error as e:
        print(f"Error inserting into MariaDB: {e}")

# --- 종료 시 GPIO 정리 및 리소스 해제 함수 ---
def cleanup():
    print("프로그램 종료: 모든 LED OFF")
    red_led.off()
    green_led.off()
    GPIO.cleanup()
    pi.stop()

# --- 빨간 LED 잠시 켜기 (경고 시 사용) ---
def turn_on_red_led_temporarily():
    red_led.on()
    green_led.off()
    time.sleep(5)
    red_led.off()
    green_led.on()

# --- 헬퍼 함수 ---
relative = lambda landmark, shape: (int(landmark.x * shape[1]), int(landmark.y * shape[0]))
relativeT = lambda landmark, shape: (int(landmark.x * shape[1]), int(landmark.y * shape[0]), 0)

# --- 전역 변수 ---
gaze_history = deque(maxlen=30)
warning_start_time = None
warning_direction = None
start_time = time.time()

# --- 시간 포맷 변환 함수 ---
def format_time(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

# --- 손 탐지 및 경고 처리 함수 ---
def detect_hands(hand_results, frame, mp_drawing, mp_hands):
    hand_text = ""
    hand_count = 0

    if hand_results.multi_hand_landmarks:
        hand_count = len(hand_results.multi_hand_landmarks)
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 두 손 미만 감지 시 부정행위 의심
    if hand_count < 2:
        hand_text = "Hands - Possible cheating"
        print(f"{format_time(time.time() - start_time)} - 3")
        threading.Thread(target=turn_on_red_led_temporarily).start()
        if conn:
            insert_status(conn, "Hands - Possible Cheating")

    return hand_text

# --- 시선 추적 및 경고 처리 함수 ---
def gaze(frame, points):
    global gaze_history, warning_start_time, warning_direction

    try:
        # 얼굴 특징점 정의
        image_points = np.array([
            relative(points.landmark[4], frame.shape),      # 코끝
            relative(points.landmark[152], frame.shape),    # 턱
            relative(points.landmark[263], frame.shape),    # 왼쪽 눈
            relative(points.landmark[33], frame.shape),     # 오른쪽 눈
            relative(points.landmark[287], frame.shape),    # 왼쪽 입꼬리
            relative(points.landmark[57], frame.shape)      # 오른쪽 입꼬리
        ], dtype="double")

        image_points1 = np.array([
            relativeT(points.landmark[4], frame.shape),
            relativeT(points.landmark[152], frame.shape),
            relativeT(points.landmark[263], frame.shape),
            relativeT(points.landmark[33], frame.shape),
            relativeT(points.landmark[287], frame.shape),
            relativeT(points.landmark[57], frame.shape)
        ], dtype="double")

        model_points = np.array([
            (0.0, 0.0, 0.0),
            (0, -63.6, -12.5),
            (-43.3, 32.7, -26),
            (43.3, 32.7, -26),
            (-28.9, -28.9, -24.1),
            (28.9, -28.9, -24.1)
        ])

        Eye_ball_center_left = np.array([[29.05], [32.7], [-39.5]])

        focal_length = frame.shape[1]
        center = (frame.shape[1] / 2, frame.shape[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        dist_coeffs = np.zeros((4, 1))
        success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                     dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        left_pupil = relative(points.landmark[468], frame.shape)
        _, transformation, _ = cv2.estimateAffine3D(image_points1, model_points)

        if transformation is not None:
            pupil_world_cord = transformation @ np.array([[left_pupil[0], left_pupil[1], 0, 1]]).T
            S = Eye_ball_center_left + (pupil_world_cord - Eye_ball_center_left) * 10

            eye_pupil2D, _ = cv2.projectPoints((int(S[0]), int(S[1]), int(S[2])),
                                               rotation_vector, translation_vector, camera_matrix, dist_coeffs)

            head_pose, _ = cv2.projectPoints((int(pupil_world_cord[0]), int(pupil_world_cord[1]), int(40)),
                                             rotation_vector, translation_vector, camera_matrix, dist_coeffs)

            gaze_point = left_pupil + (eye_pupil2D[0][0] - left_pupil) - (head_pose[0][0] - left_pupil)

            # 시선 선 그리기 및 방향 분석
            p1 = (int(left_pupil[0]), int(left_pupil[1]))
            p2 = (int(gaze_point[0]), int(gaze_point[1]))
            line_length = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
            dx = p2[0] - p1[0]

            cv2.line(frame, p1, p2, (0, 0, 255), 2)
            cv2.putText(frame, f"dx: {dx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            current_direction = None
            if line_length >= 50:
                if dx >= 50:
                    current_direction = "left"
                    print(f"{format_time(time.time() - start_time)} - 4")
                    threading.Thread(target=turn_on_red_led_temporarily).start()
                    if conn:
                        insert_status(conn, "Left")
                elif dx <= -80:
                    current_direction = "right"
                    print(f"{format_time(time.time() - start_time)} - 5")
                    threading.Thread(target=turn_on_red_led_temporarily).start()
                    if conn:
                        insert_status(conn, "Right")

            # 경고 이력 업데이트
            gaze_history.append(current_direction)
            current_time = time.time()

            if warning_start_time is not None and current_time - warning_start_time < 1.5:
                warning_text = f"Looking {warning_direction.capitalize()} - Possible Cheating"
                cv2.putText(frame, warning_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                warning_start_time = None
                warning_direction = None

            if warning_start_time is None:
                left_count = sum(1 for x in gaze_history if x == "left")
                right_count = sum(1 for x in gaze_history if x == "right")

                if left_count >= 2:
                    warning_start_time = current_time
                    warning_direction = "left"
                elif right_count >= 2:
                    warning_start_time = current_time
                    warning_direction = "right"

    except Exception as e:
        print(f"Gaze estimation error: {str(e)}")

# --- 메인 감지 루프 함수 ---
def main():
    global conn
    global picam2
    # DB 연결
    conn = connect_db()
    if conn:
        print("[DB LOG] MariaDB 연결 성공")
    else:
        print("[DB ERROR] MariaDB 연결 실패. 종료합니다.")
        return

    # LED 상태 초기화
    green_led.on()
    red_led.off()

    # 카메라 설정 및 시작
    try:
        picam2.configure(picam2.create_preview_configuration(main={"format": 'RGB888', "size": (640, 480)}))
        picam2.start()
    except Exception as e:
        print(f"[CAMERA ERROR] 카메라 시작 실패: {e}")
        return

    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh, \
        mp_hands.Hands() as hands:

        try:
            while True:
                frame = picam2.capture_array()
                image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                results_face = face_mesh.process(frame)
                results_hands = hands.process(frame)

                if results_face.multi_face_landmarks:
                    gaze(image, results_face.multi_face_landmarks[0])

                hand_text = detect_hands(results_hands, image, mp_drawing, mp_hands)
                if hand_text:
                    cv2.putText(image, hand_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow("Test Monitoring", image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            if conn:
                conn.close()
            picam2.stop()
            picam2.close()
            cv2.destroyAllWindows()
            green_led.off()

# --- 거리 감지 후 메인 감지 실행 ---
def distance_camera():
    while True:
        print("200cm 이내로 접근을 기다리는 중...")
        distance = measure_distance()
        print(f"측정 거리: {distance:.2f}cm")
        if distance <= 200:
            print("감지됨! 카메라 실행 중...")
            main()
            break # 한 번만 실행 후 종료

# --- 프로그램 시작 ---
if __name__ == "__main__":
    atexit.register(cleanup)  # 종료 시 자원 정리 등록
    distance_camera()         # 거리 감지부터 시작
