import cv2
import mediapipe as mp
import numpy as np
import os
import time

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# --- SETTINGS ---
SIGNS        = ["HELLO", "THANK YOU", "WATER", "FOOD", "HELP", "YES", "NO", "PLEASE"]
NUM_SEQUENCES = 80    # recordings per sign
SEQ_LENGTH    = 30    # frames per recording
DATASET_PATH  = "data_collection/dataset"

def collect_sign(sign_name):
    os.makedirs(f"{DATASET_PATH}/{sign_name}", exist_ok=True)
    cap = cv2.VideoCapture(0)
    print(f"\n--- Collecting: {sign_name} ---")

    for seq in range(NUM_SEQUENCES):
        print(f"  Sequence {seq+1}/{NUM_SEQUENCES} — get ready, press SPACE to record")

        # Wait for spacebar
        while True:
            ret, frame = cap.read()
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            if results.multi_hand_landmarks:
                for lm in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

            # Instructions on screen
            cv2.putText(frame, f"Sign: {sign_name}  Seq: {seq+1}/{NUM_SEQUENCES}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, "Press SPACE to start recording",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.imshow("ISL Collector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 32: break   # spacebar
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

        # Record SEQ_LENGTH frames
        frames = []
        for f in range(SEQ_LENGTH):
            ret, frame = cap.read()
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0].landmark
                keypoints = np.array([[p.x, p.y, p.z] for p in lm])  # (21, 3)
                for lmk in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, lmk, mp_hands.HAND_CONNECTIONS)
            else:
                keypoints = np.zeros((21, 3))  # no hand detected — zero pad

            frames.append(keypoints)

            # Show recording indicator
            cv2.putText(frame, f"RECORDING... {f+1}/{SEQ_LENGTH}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.imshow("ISL Collector", frame)
            cv2.waitKey(1)

        # Save the sequence
        np.save(f"{DATASET_PATH}/{sign_name}/seq_{seq}.npy", np.array(frames))
        print(f"  Saved seq_{seq}.npy ✓")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✓ Done collecting {sign_name}!")

# --- RUN ---
if __name__ == "__main__":
    for sign in SIGNS:
        input(f"\nReady to collect [{sign}]? Press Enter to start...")
        collect_sign(sign)
    print("\n All signs collected! Check data_collection/dataset/")