import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
import threading
import queue
import subprocess
import time

# --- Speech ---
speech_queue = queue.Queue()
ps_process = None

def init_speech():
    global ps_process
    ps_process = subprocess.Popen(
        ['powershell', '-NoProfile', '-Command', '-'],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1
    )
    ps_process.stdin.write(
        'Add-Type -AssemblyName System.Speech\n'
        '$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer\n'
        '$speak.Rate = 3\n'  # faster rate = less lag feel
        '$speak.Volume = 100\n'
    )
    ps_process.stdin.flush()
    time.sleep(1)  # let it load once

def speech_worker():
    init_speech()
    while True:
        text = speech_queue.get()
        if text is None:
            break
        try:
            ps_process.stdin.write(f'$speak.Speak("{text}")\n')
            ps_process.stdin.flush()
        except Exception as e:
            print(f"Speech error: {e}")
        speech_queue.task_done()

threading.Thread(target=speech_worker, daemon=True).start()

def speak(text):
    # Clear any pending speech first — speak latest word immediately
    while not speech_queue.empty():
        try:
            speech_queue.get_nowait()
            speech_queue.task_done()
        except:
            pass
    speech_queue.put(text)
    print(f"[Speaking]: {text}")

# --- Load model ---
interpreter = tf.lite.Interpreter(model_path="training/isl_model.tflite")
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
signs = open("training/labels.txt").read().strip().split("\n")
print("Signs loaded:", signs)

# --- MediaPipe ---
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.75)

# --- Settings ---
SEQ_LENGTH       = 30
CONFIDENCE_MIN   = 0.90   # raised to 90% — only accept very confident predictions
HAND_PRESENT_MIN = 20     # hand must be detected for 20 consecutive frames before predicting

# --- State ---
sequence          = deque(maxlen=SEQ_LENGTH)
sentence          = []
last_prediction   = None
debounce_count    = 0
current_sign      = ""
current_conf      = 0.0
no_hand_count     = 0
hand_present_count = 0    # counts consecutive frames WITH hand

def normalize(frame):
    frame = frame - frame[0]
    scale = np.max(np.abs(frame))
    if scale > 0:
        frame = frame / scale
    return frame.flatten()

def predict(seq):
    input_data = np.array(seq, dtype=np.float32)[np.newaxis]
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    return np.argmax(output), np.max(output)

# --- Webcam ---
cap = cv2.VideoCapture(0)
print("\nISL Recogniser running!")
print("Controls: Q=quit  C=clear  S=speak sentence\n")
speak("Ready")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame   = cv2.flip(frame, 1)
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        # Hand IS detected
        no_hand_count       = 0
        hand_present_count += 1

        lm = results.multi_hand_landmarks[0].landmark
        keypoints = np.array([[p.x, p.y, p.z] for p in lm])
        keypoints = normalize(keypoints)
        sequence.append(keypoints)

        for hand_lm in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

    else:
        # No hand detected
        hand_present_count = 0
        no_hand_count     += 1
        sequence.append(np.zeros(63))
        current_sign = ""
        current_conf = 0.0

        # Auto speak when hand removed for 1.5 seconds
        if no_hand_count == 45 and sentence:
            speak(" ".join(sentence))

    # Only predict when hand has been present for enough frames
    # This stops random predictions when hand first enters frame
    if len(sequence) == SEQ_LENGTH and hand_present_count >= HAND_PRESENT_MIN:
        idx, confidence = predict(list(sequence))

        if confidence >= CONFIDENCE_MIN:
            current_sign = signs[idx]
            current_conf = confidence

            if current_sign != last_prediction:
                sentence.append(current_sign)
                last_prediction = current_sign
                debounce_count  = 0
                speak(current_sign)
                print(f"Recognised: {current_sign} ({confidence*100:.0f}%)")
            else:
                debounce_count += 1
                if debounce_count > 60:
                    debounce_count  = 0
                    last_prediction = None
        else:
            # Confidence too low — show what it thinks but don't add to sentence
            current_sign = f"? {signs[idx]}"
            current_conf = confidence

    if len(sentence) > 6:
        sentence = sentence[-6:]

    # --- Draw UI ---
    h, w = frame.shape[:2]

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 65), (20, 20, 20), -1)
    if current_sign.startswith("?"):
        sign_color = (0, 165, 255)   # orange = uncertain
    elif current_conf >= 0.90:
        sign_color = (0, 255, 150)   # green = confident
    else:
        sign_color = (255, 255, 255) # white = neutral
    sign_text = f"{current_sign}  ({current_conf*100:.0f}%)" if current_conf > 0 else "Show a sign..."
    cv2.putText(frame, sign_text, (15, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, sign_color, 2)

    # Hand status indicator
    status      = "Hand detected" if hand_present_count > 0 else "No hand"
    status_col  = (0, 255, 0) if hand_present_count > 0 else (0, 0, 255)
    cv2.putText(frame, status, (w-180, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_col, 1)

    # Bottom bar
    cv2.rectangle(frame, (0, h-70), (w, h), (20, 20, 20), -1)
    sentence_text = " ".join(sentence) if sentence else "Sentence appears here..."
    cv2.putText(frame, sentence_text, (15, h-25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 150), 2)

    # Confidence bar
    if current_conf > 0:
        bar_h     = int((h - 90) * current_conf)
        bar_color = (0, 255, 0) if current_conf >= 0.90 else (0, 165, 255)
        cv2.rectangle(frame, (w-22, h-70-bar_h), (w-5, h-70), bar_color, -1)

    # Controls
    cv2.putText(frame, "Q=quit  C=clear  S=speak", (15, h-75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

    cv2.imshow("ISL Sign Recogniser", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('c'):
        sentence.clear()
        last_prediction = None
        current_sign    = ""
        print("Sentence cleared")
    if key == ord('s') and sentence:
        speak(" ".join(sentence))

cap.release()
cv2.destroyAllWindows()

if sentence:
    final = " ".join(sentence)
    print(f"\nFinal sentence: {final}")
    speak(final)
    time.sleep(3)

print("Done!")
