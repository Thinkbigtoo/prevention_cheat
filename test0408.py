import cv2
import mediapipe as mp
import math
import mariadb
import sys
# === 유클리드 거리 계산 함수 ===
def euclidean_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# === 시선 방향 탐지 ===
def detect_gaze_direction(landmarks, w, h):
    iris = landmarks.landmark[468]
    left = landmarks.landmark[33]
    right = landmarks.landmark[133]

    iris_ratio = (iris.x - left.x) / (right.x - left.x)

    if iris_ratio < 0.38:
        return "Looking LEFT - Possible cheating"
    elif iris_ratio > 0.62:
        return "Looking RIGHT - Possible cheating"
    return ""

# === 손 탐지 ===
def detect_hands(hand_results, frame, mp_drawing, mp_hands):
    hand_text = ""
    hand_count = 0

    if hand_results.multi_hand_landmarks:
        hand_count = len(hand_results.multi_hand_landmarks)
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if hand_count < 2:
        hand_text = "Hands - Possible cheating"

    return hand_text

# === 경고 텍스트 표시 ===
def draw_alerts(frame, alerts):
    y_offset = 50
    for alert in alerts:
        if alert:
            cv2.putText(frame, alert, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            y_offset += 50

# === 프레임 처리 ===
def process_frame(frame, face_mesh, hands, mp_drawing, mp_hands):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_results = face_mesh.process(rgb)
    hand_results = hands.process(rgb)

    warning_text = ""
    hand_text = ""

    if face_results.multi_face_landmarks:
        for landmarks in face_results.multi_face_landmarks:
            warning_text = detect_gaze_direction(landmarks, w, h)

            # 시선 중심 (홍채 표시)
            iris = landmarks.landmark[468]
            iris_coord = (int(iris.x * w), int(iris.y * h))
            cv2.circle(frame, iris_coord, 3, (255, 0, 255), -1)

    hand_text = detect_hands(hand_results, frame, mp_drawing, mp_hands)

    draw_alerts(frame, [warning_text, hand_text])
    return frame

# === 메인 루프 ===
def main():
    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hands = mp_hands.Hands()

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame, face_mesh, hands, mp_drawing, mp_hands)
        cv2.imshow('Exam Monitor - Eye and Hand Tracking', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC 키
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
