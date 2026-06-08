import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("training/isl_model.h5")
print("Model loaded successfully")

# Convert to TFLite — with SELECT_TF_OPS for LSTM support
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# These two lines fix the LSTM error
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter._experimental_lower_tensor_list_ops = False

tflite_model = converter.convert()

# Save the .tflite file
with open("training/isl_model.tflite", "wb") as f:
    f.write(tflite_model)

print(f"TFLite model saved — size: {len(tflite_model) / 1024:.1f} KB")

# Verify it works
interpreter = tf.lite.Interpreter(model_path="training/isl_model.tflite")
interpreter.allocate_tensors()
print("TFLite model verified and working!")

# Print input/output shapes
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(f"\nInput shape:  {input_details[0]['shape']}")
print(f"Output shape: {output_details[0]['shape']}")

# Save signs list for Flutter
signs = np.load("training/signs.npy")
with open("training/labels.txt", "w") as f:
    for sign in signs:
        f.write(sign + "\n")
print(f"\nLabels saved: {list(signs)}")
print("\nDone! Ready for Flutter.")