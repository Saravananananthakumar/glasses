import cv2
import mediapipe as mp
import numpy as np
 
# Load glasses image (transparent PNG)
glasses_img = cv2.imread("glasses/glass1.png", cv2.IMREAD_UNCHANGED)
 
# Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                   refine_landmarks=True,
                                   min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5)
 
# Get webcam feed
cap = cv2.VideoCapture(0)
 
while True:
    ret, frame = cap.read()
    if not ret:
        break
 
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
 
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Eye landmarks (outer left and right corners)
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]
 
            left_x = int(left_eye.x * w)
            left_y = int(left_eye.y * h)
            right_x = int(right_eye.x * w)
            right_y = int(right_eye.y * h)
 
            # Compute center and angle
            eye_center = ((left_x + right_x) // 2, (left_y + right_y) // 2)
            width = int(np.hypot(right_x - left_x, right_y - left_y) * 2.0)
            aspect_ratio = glasses_img.shape[1] / glasses_img.shape[0]
            height = int(width / aspect_ratio)
 
            # Resize glasses
            resized_glasses = cv2.resize(glasses_img, (width, height))
 
            # Overlay
            y1 = int(eye_center[1] - height // 2)
            x1 = int(eye_center[0] - width // 2)
            y2 = y1 + height
            x2 = x1 + width
 
            # Clip coordinates
            y1, y2 = max(0, y1), min(h, y2)
            x1, x2 = max(0, x1), min(w, x2)
 
            # Extract alpha channel
            alpha_s = resized_glasses[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
 
            for c in range(3):
                frame[y1:y2, x1:x2, c] = (alpha_s * resized_glasses[:, :, c] +
                                          alpha_l * frame[y1:y2, x1:x2, c])
 
    cv2.imshow("Virtual Try-On", frame)
 
    key = cv2.waitKey(1)
    if key == 27:  # ESC to quit
        break
 
cap.release()
cv2.destroyAllWindows()