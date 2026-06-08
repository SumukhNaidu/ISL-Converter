import numpy as np

data = np.load("data_collection/dataset/HELLO/seq_0.npy")

print("Shape:", data.shape)
# Should print: (30, 21, 3)
# 30 frames, 21 landmarks, 3 coords each

print("Wrist position (frame 1):", data[0][0])
# Should print 3 non-zero decimal numbers like (0.51, 0.83, 0.002)

print("Any empty frames?", (data.sum(axis=(1,2)) == 0).sum())
# Should print 0 — means hand was detected in all frames