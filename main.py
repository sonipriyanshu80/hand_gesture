import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

print("Hand Gesture Recognition Started!")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    
    # MediaPipe needs RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    finger_count = 0
    gesture_name = "No Hand Detected"
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Draw the hand landmarks on screen
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )
        
        landmarks = hand_landmarks.landmark
        
        # Get positions of finger tips and joints
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        index_tip = landmarks[8]
        index_pip = landmarks[6]
        middle_tip = landmarks[12]
        middle_pip = landmarks[10]
        ring_tip = landmarks[16]
        ring_pip = landmarks[14]
        pinky_tip = landmarks[20]
        pinky_pip = landmarks[18]
        
        # Check if it's right hand (for thumb detection)
        is_right_hand = thumb_tip.x < thumb_ip.x
        
        # Count fingers - thumb uses x coordinate, others use y coordinate
        if is_right_hand:
            if thumb_tip.x > thumb_ip.x:
                finger_count += 1
        else:
            if thumb_tip.x < thumb_ip.x:
                finger_count += 1
        
        if index_tip.y < index_pip.y:
            finger_count += 1
        
        if middle_tip.y < middle_pip.y:
            finger_count += 1
        
        if ring_tip.y < ring_pip.y:
            finger_count += 1
        
        if pinky_tip.y < pinky_pip.y:
            finger_count += 1
        
        # Recognize gestures
        if finger_count == 0:
            gesture_name = "Fist"
        elif finger_count == 1:
            gesture_name = "One Finger"
        elif finger_count == 2:
            gesture_name = "Victory"
        elif finger_count == 5:
            gesture_name = "Open Palm"
        else:
            gesture_name = f"{finger_count} Fingers"
        
        # Check for thumbs up gesture
        thumb_up = False
        if is_right_hand:
            thumb_up = thumb_tip.x > thumb_ip.x
        else:
            thumb_up = thumb_tip.x < thumb_ip.x
        
        other_fingers_down = (
            index_tip.y > index_pip.y and
            middle_tip.y > middle_pip.y and
            ring_tip.y > ring_pip.y and
            pinky_tip.y > pinky_pip.y
        )
        
        if thumb_up and other_fingers_down:
            gesture_name = "Thumbs Up"
    
    cv2.putText(
        frame,
        f"Gesture: {gesture_name}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    
    cv2.putText(
        frame,
        f"Fingers: {finger_count}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )
    
    cv2.imshow("Hand Gesture Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
print("Program ended")
