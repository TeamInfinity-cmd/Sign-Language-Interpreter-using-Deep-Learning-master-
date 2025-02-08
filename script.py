import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)


def get_hand_gesture(landmarks):
    """
    Determine the hand gesture based on the hand landmarks.
    For now, it's only a basic check for gestures like 'A', 'B', 'C', 'D', 'E', etc.
    You can add more gestures based on the ASL alphabet.
    """
    # Example for gesture 'A' (closed fist with the thumb pointing upwards)
    if (landmarks[4].y < landmarks[3].y and landmarks[4].y < landmarks[2].y and
            landmarks[8].y > landmarks[7].y and landmarks[12].y > landmarks[11].y and
            landmarks[16].y > landmarks[15].y and landmarks[20].y > landmarks[19].y):
        return 'hello'

    # Example for gesture 'B' (open hand, fingers straight)
    if (landmarks[4].y < landmarks[3].y and landmarks[8].y > landmarks[7].y and
            landmarks[12].y > landmarks[11].y and landmarks[16].y > landmarks[15].y and
            landmarks[20].y > landmarks[19].y):
        return 'B'

    # Gesture 'C' (curved hand like a 'C')
    if (landmarks[4].y > landmarks[3].y and landmarks[8].y > landmarks[7].y and
            landmarks[12].y > landmarks[11].y and landmarks[16].y > landmarks[15].y and
            landmarks[20].y > landmarks[19].y):
        return 'sorry'

    # Gesture 'D' (Index finger pointing up)
    if (landmarks[8].y < landmarks[6].y and landmarks[12].y > landmarks[11].y and
            landmarks[16].y > landmarks[15].y and landmarks[20].y > landmarks[19].y):
        return 'ONE'

    # Gesture 'E' (Fist with fingers slightly curled)
    if (landmarks[4].y < landmarks[3].y and landmarks[8].y < landmarks[7].y and
            landmarks[12].y < landmarks[11].y and landmarks[16].y < landmarks[15].y and
            landmarks[20].y < landmarks[19].y):
        return 'BYE'

    # Gesture 'F' (Thumb and index finger forming a circle)
    if (landmarks[4].y < landmarks[3].y and landmarks[8].y < landmarks[7].y and
            landmarks[12].y < landmarks[11].y and landmarks[16].y > landmarks[15].y and
            landmarks[20].y < landmarks[19].y):
        return 'NO'

    # Gesture 'G' (Thumb and index finger extended, others curled in)
    if (landmarks[4].y < landmarks[3].y and landmarks[8].y < landmarks[7].y and
            landmarks[12].y > landmarks[11].y and landmarks[16].y < landmarks[15].y and
            landmarks[20].y < landmarks[19].y):
        return 'Yes'
    if (landmarks[8].y < landmarks[7].y and landmarks[12].y < landmarks[11].y and
            landmarks[4].y < landmarks[3].y and landmarks[16].y > landmarks[15].y and
            landmarks[20].y > landmarks[19].y):
        return 'Peace'
    if (landmarks[8].y < landmarks[7].y and landmarks[12].y < landmarks[11].y and
            landmarks[4].y < landmarks[3].y and landmarks[16].y > landmarks[15].y and
            landmarks[20].y > landmarks[19].y):
        return 'your under my genjutsu'
    if (landmarks[20].y < landmarks[19].y and
            landmarks[4].y < landmarks[3].y and
            landmarks[8].y < landmarks[7].y and
            landmarks[12].y < landmarks[11].y and
            landmarks[16].y < landmarks[15].y):
        return 'I'

    # If no gesture matches, return 'Unknown'
    return 'Unknown'


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Convert the image to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(frame_rgb)

    # Initialize gesture variable
    gesture = "Unknown"

    # If hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get hand gesture (e.g., 'A', 'B', 'C', etc.)
            gesture = get_hand_gesture(hand_landmarks.landmark)

            # Display the recognized gesture on the frame
            cv2.putText(frame, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame with landmarks and gesture text
    cv2.imshow('Sign Language Translator', frame)

    # Break the loop on pressing the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):cu[;

    44aq+]
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
