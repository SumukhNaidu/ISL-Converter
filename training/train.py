import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Load preprocessed data
X_train = np.load("training/X_train.npy")
X_test  = np.load("training/X_test.npy")
y_train = np.load("training/y_train.npy")
y_test  = np.load("training/y_test.npy")
SIGNS   = np.load("training/signs.npy")

NUM_SIGNS = len(SIGNS)
print(f"Training on {NUM_SIGNS} signs: {SIGNS}")
print(f"Input shape: {X_train.shape}")

# Build LSTM model
model = Sequential([
    LSTM(64,  return_sequences=True,  input_shape=(30, 63)),
    Dropout(0.3),
    LSTM(128, return_sequences=False),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(NUM_SIGNS, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "training/best_model.h5",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)


# Train
print("\nTraining started...")
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# Evaluate on test set
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {accuracy*100:.2f}%")
print(f"Test loss:     {loss:.4f}")

# Save final model
model.save("training/isl_model.h5")
np.save("training/history.npy", history.history)
print("\nModel saved to training/isl_model.keras")