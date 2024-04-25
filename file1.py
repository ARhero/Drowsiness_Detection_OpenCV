from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

prototxt = "MobileNetSSD_deploy.prototxt.txt"
model = "MobileNetSSD_deploy.caffemodel"

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)

print("[INFO] starting video stream...")
vs = VideoStream(src=1).start()
time.sleep(2.0)
fps = FPS().start()
counter = 0
conf = 0.2

# Placeholder function for robot control
def control_robot(direction):
    print("Robot moving", direction)

def get_distance_to_camera(y_start, y_end):
    # Placeholder function to compute distance to camera based on object size in the frame
    return 10  # Replace this with your implementation

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=600)

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
        0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()

    direction_text = "Stop"

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > conf:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label = "{}: {:.2f}%".format(CLASSES[idx],
                confidence * 100)
            if CLASSES[idx] == "person":
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                center = (int((startX + endX) / 2), int((startY + endY) / 2))
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

                distance = get_distance_to_camera(startY, endY)  # Calculate distance to camera
                if center[0] < w // 3:
                    direction_text = "Left"
                    control_robot("left")
                elif center[0] > 2 * w // 3:
                    direction_text = "Right"
                    control_robot("right")
                elif distance < 200:  # Adjust threshold as needed
                    direction_text = "Front"
                    control_robot("Front")    
                else:
                    direction_text = "Stop"
                    control_robot("Stop")

    # Display direction text on the frame
    cv2.putText(frame, direction_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    fps.update()

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()
vs.stop()
