import cv2
import mediapipe as mp
import time

print("1. 正在初始化 MediaPipe Face Mesh...")
mp_face_mesh = mp.solutions.face_mesh
# static_image_mode=False 會讓它針對影片流進行追蹤優化
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

print("2. 正在開啟攝影機...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("無法開啟鏡頭，嘗試切換索引...")
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

p_time = 0

print("3. 開始執行！請面向鏡頭。按 Esc 鍵退出。")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # 翻轉影像並轉為 RGB
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 執行臉部網格偵測
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 畫出 468 點網格 (Tesselation)
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            )

    # 計算 FPS
    c_time = time.time()
    fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
    p_time = c_time
    cv2.putText(image, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Face Mesh Test', image)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("程式結束")
