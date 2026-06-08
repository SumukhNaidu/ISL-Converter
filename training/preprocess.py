import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Settings
DATASET_PATH = "data_collection/dataset"
SEQ_LENGTH   = 30

# Auto-read all sign folders
SIGNS = sorted(os.listdir(DATASET_PATH))
print("Signs found:", SIGNS)

# Build label map
label_map = {sign: i for i, sign in enumerate(SIGNS)}
print("Label map:", label_map)

# Load all sequences
sequences, labels = [], []

for sign in SIGNS:
    sign_path = os.path.join(DATASET_PATH, sign)
    files = sorted(os.listdir(sign_path))
    print(f"Loading {sign}: {len(files)} sequences")

    for file in files:
        filepath = os.path.join(sign_path, file)
        data = np.load(filepath)           # shape: (30, 21, 3)

        # Normalize — subtract wrist, scale to [-1, 1]
        normalized = []
        for frame in data:
            frame = frame - frame[0]       # subtract wrist position
            scale = np.max(np.abs(frame))
            if scale > 0:
                frame = frame / scale      # scale to [-1, 1]
            normalized.append(frame.flatten())  # (63,)

        sequences.append(normalized)       # (30, 63)
        labels.append(label_map[sign])

X = np.array(sequences)                   # (total, 30, 63)
y = to_categorical(labels, num_classes=len(SIGNS))  # one-hot encoded

print("\nDataset shape:", X.shape)         # should be (640, 30, 63)
print("Labels shape:", y.shape)            # should be (640, 8)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# Save for training
np.save("training/X_train.npy", X_train)
np.save("training/X_test.npy",  X_test)
np.save("training/y_train.npy", y_train)
np.save("training/y_test.npy",  y_test)
np.save("training/signs.npy",   np.array(SIGNS))

print("\nSaved all files to training/ — ready to train!")