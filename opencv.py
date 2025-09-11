import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

# Load labels
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Load TFLite model
interpreter = Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get input shape
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

# Open camera (0 for USB webcam, Pi Camera works too)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize & preprocess
    img = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(img, axis=0).astype(np.float32)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Get predicted label
    predicted_idx = int(np.argmax(output_data))
    predicted_label = labels[predicted_idx]

    # Put label text on frame
    cv2.putText(frame, f"{predicted_label}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Show output window
    cv2.imshow("Driver Monitoring", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
