import cv2
import numpy as np

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes =[]
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

image = cv2.imread("room.jpg")
height, width, _ = image.shape
blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
net.setInput(blob)

output_layers_names = net.getUnconnectedOutLayersNames()
layer_outputs = net.forward(output_layers_names)
np.random.seed(42)
colors = np.random.uniform(0, 255, size=(len(classes), 3))

boxes = []
confidences = []
class_ids = []

for output in layer_outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.7:
            print(f"Detected Ckass ID: {class_id} with confidence {confidence}")
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)


import cv2
import numpy as np

# 1. Load the Model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# 2. Setup Colors
np.random.seed(42)
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# 3. Prepare the Image
image = cv2.imread("room.jpg")
height, width, _ = image.shape
blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
net.setInput(blob)

# 4. Run Detection
output_layers_names = net.getUnconnectedOutLayersNames()
layer_outputs = net.forward(output_layers_names)

boxes = []
confidences = []
class_ids = []

for output in layer_outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.7:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# 5. Apply Non-Maximum Suppression
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# 6. Draw Results
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        
        # Pull the specific ID and Color for THIS box
        class_id = class_ids[i]
        label = classes[class_id]
        # FIX: Ensure we use 'color' (singular) and convert to a tuple
        color = tuple([int(c) for c in colors[class_id]])
        
        # FIX: Get the confidence for THIS box only
        confidence_pct = int(confidences[i] * 100)
        display_text = f"{label} {confidence_pct}%"

        # Draw using the specific color we picked
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

cv2.imshow("Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
