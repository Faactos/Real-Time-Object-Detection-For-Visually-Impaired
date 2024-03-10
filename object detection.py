import cv2
import numpy as np
import os
import pygame
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

pygame.mixer.init()

# Opencv DNN
net = cv2.dnn.readNet("C:\\Users\\vidul\\dnn_model\\yolov3 (1).weights", "C:\\Users\\vidul\\dnn_model\\yolov3.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1 / 255)

# Initialize camera
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Load class Lists
classes = []
with open("C:\\Users\\vidul\\dnn_model\\coco.names", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

print("Objects Lists")
print(classes)

# Assigning unique colors to different classes
class_colors = np.random.uniform(0, 255, size=(len(classes), 3))
import os

# Get the directory of the Python script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the path for the audio_files directory
audio_files_dir = os.path.join(script_dir, "audio_files")

# Create the directory if it doesn't exist
if not os.path.exists(audio_files_dir):
    os.makedirs(audio_files_dir)

# Initialize Pygame mixer
pygame.mixer.init()


# Function to speak the object name
def speak_object(class_name):
    global last_time
    audio_file = os.path.join("audio_files", class_name + ".mp3")
    try:
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        last_time = time.time()
    except Exception as e:
        print("Error playing audio:", e)


# Function to estimate distance based on object size
def estimate_distance(bbox):
    # Simplified approach: Assuming fixed object size and distance
    known_object_size = 100  # Example: object size in pixels at a known distance (e.g., 1 meter)
    known_distance = 1.0  # Example: distance to the object in meters

    object_size = max(bbox[2], bbox[3])  # Use the maximum dimension of the bounding box as the object size
    distance = known_object_size * known_distance / object_size

    return distance


# Prepare data for linear regression
X = []  # Features (independent variables)
y = []  # Distance (dependent variable)

try:
    while True:
        ret, frame = cam.read()

        # Check if camera reading is successful
        if not ret:
            print("Error: Camera frame could not be read")
            break

        # Object Detection
        detected_classes, scores, bboxes = model.detect(frame, confThreshold=0.5)
        print("Classes:", detected_classes)

        # Check if objects are detected
        if len(detected_classes) > 0:
            for detection, score, bbox in zip(detected_classes, scores, bboxes):
                class_id = detection
                class_name = classes[class_id]

                # Get color for the class
                color = [int(c) for c in class_colors[class_id]]

                # Draw bounding box and object name
                label = f"{class_name}: {score:.2f}"
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color, 2)
                cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Speak object name
                speak_object(class_name)

                # Estimate distance to object
                distance = estimate_distance(bbox)
                cv2.putText(frame, f"Distance: {distance:.2f} meters", (bbox[0], bbox[1] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Collect data for linear regression
                X.append([bbox[2], bbox[3]])  # Features: object width and height
                y.append(distance)  # Distance

        # Display the frame
        cv2.imshow("Frame", frame)

        # Check for keyboard input
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

except KeyboardInterrupt:
    pass

# Release camera and close window
cam.release()
cv2.destroyAllWindows()

# Perform linear regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
