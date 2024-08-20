# object-detection-using-yolov4
import numpy as np
import cv2 as cv2
import os
import time

%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

#Routine to fix colors in image
def fixColor(image):
    return(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) 
labelsFile = r"C:\Users\jhans\Desktop\ML Internship\opencv-intro\images\soccer.jpg"
with open(labelsFile, encoding='latin-1') as file:
    LABELS = file.read().strip().split("\n")
print("No. of supported classes", len(LABELS))
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")
import requests

url = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights"
output_path = "model_data/yolov4.weights"
response = requests.get(url, stream=True)

if response.status_code == 200:
    with open(output_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print(f"Downloaded {output_path} successfully.")
else:
    print(f"Failed to download file. Status code: {response.status_code}")
import requests

url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg"
output_path = "model_data/yolov4.cfg"

response = requests.get(url)

if response.status_code == 200:
    with open(output_path, 'wb') as file:
        file.write(response.content)
    print(f"Downloaded {output_path} successfully.")
else:
    print(f"Failed to download file. Status code: {response.status_code}")
weights="model_data/yolov4.weights"
config="model_data/yolov4.cfg"
net = cv2.dnn.readNetFromDarknet(config, weights)
#Take a look at names of all layers in the model
ln = net.getLayerNames()
print (len(ln), ln )
net.getUnconnectedOutLayers()!pip install opencv-python-headless
import cv2

# Replace with your actual paths
weights_path = "model_data/yolov4.weights"
config_path = "model_data/yolov4.cfg"

# Load YOLO model
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

# Check if net is successfully loaded
if net is None:
    print("Error: Failed to load YOLO model.")
else:
    # Get the indices of the output layers
    layer_names = net.getLayerNames()
    output_layer_indices = net.getUnconnectedOutLayers()

    # Extract output layer names based on indices
    output_layers = [layer_names[i - 1] for i in output_layer_indices]

    print("Output layer names:")
    print(output_layers)
img_path = r"C:\Users\jhans\Desktop\ML Internship\opencv-intro\images\soccer.jpg"
img = cv2.imread(img_path)
img=cv2.resize(img, (608, 608))
plt.imshow(fixColor(img))
print (img.shape)
(H, W) = img.shape[:2] 
blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (608, 608), swapRB=True, crop=False)
print ("Shape of blob", blob.shape)
plt.imshow(fixColor(blob[0, 0, :, :]))
split_blob=np.hstack([ blob[0, 0, :, :],blob[0, 1, :, :], blob[0, 2, :, :],])
plt.imshow(fixColor(split_blob))
net.setInput(blob)
import cv2
import numpy as np
import time

# Replace with your actual paths
config_path = "model_data/yolov4.cfg"
weights_path = "model_data/yolov4.weights"
img_path = r"C:\Users\jhans\Desktop\ML Internship\opencv-intro\images\soccer.jpg"

# Load YOLO model
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

# Load image
img = cv2.imread(img_path)

# Check if image is loaded successfully
if img is None:
    print(f"Error: Unable to load image '{img_path}'")
else:
    # Resize image (if necessary)
    img = cv2.resize(img, (608, 608))

    # Construct blob from image (assuming YOLOv4)
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (608, 608), swapRB=True, crop=False)

    # Set input for the network
    net.setInput(blob)

    # Perform forward pass
    layerOutputs = net.forward(net.getUnconnectedOutLayersNames())

    # Iterate through each layer's output
    for i, output in enumerate(layerOutputs):
        # Print the shape of each output
        print(f"Shape of output {i+1}: {output.shape}")

    # Example: Measure time taken
    t0 = time.time()
    # Your operations here...
    t = time.time()
    print('Time elapsed:', t - t0)
print (len(layerOutputs))
print (len(layerOutputs[0]))
print (len(layerOutputs[0][0]))
print (layerOutputs[0][0])
boxes = []
confidences = []
classIDs = []
for output in layerOutputs:
    print ("Shape of each output", output.shape)
    # loop over each of the detections
    for detection in output:
        # extract the class ID and confidence (i.e., probability)
        # of the current object detection
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]

        # filter out weak predictions by ensuring the detected
        # probability is greater than the minimum probability
        if confidence > 0.3:
            # scale the bounding box coordinates back relative to
            # the size of the image, keeping in mind that YOLO
            # actually returns the center (x, y)-coordinates of
            # the bounding box followed by the boxes' width and
            # height
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype("int")

            # use the center (x, y)-coordinates to derive the top
            # and and left corner of the bounding box
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))

            # update our list of bounding box coordinates,
            # confidences, and class IDs
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)
            print (LABELS[classID], detection[4], confidence)
print (len(boxes))
# apply non-maxima suppression to suppress weak, overlapping
# bounding boxes
idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
print (len(idxs))
# ensure at least one detection exists
if len(idxs) > 0:
	# loop over the indexes we are keeping
	for i in idxs.flatten():
		# extract the bounding box coordinates
		(x, y) = (boxes[i][0], boxes[i][1])
		(w, h) = (boxes[i][2], boxes[i][3])

		# draw a bounding box rectangle and label on the frame
		color = [int(c) for c in COLORS[classIDs[i]]]
		cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
		text = "{}: {:.4f}".format(LABELS[classIDs[i]],
			confidences[i])
		cv2.putText(img, text, (x, y - 5),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
plt.imshow(fixColor(img))
import cv2
import matplotlib.pyplot as plt

def draw_boxes(image, boxes, color=(0, 255, 0), label=None):
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        if label:
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

test_dataset = [
    {"image_path": r"C:\Users\jhans\Desktop\ML Internship\opencv-intro\images\soccer.jpg", "annotations": [{"class": "person", "bbox": [50, 50, 200, 200]}]},
    # Add more images and annotations as needed
]

for data in test_dataset:
    input_image_path = data["image_path"]
    ground_truth_annotations = data["annotations"]
    
    # Load image
    img = cv2.imread(input_image_path)
    
    # Check if image is loaded successfully
    if img is None:
        print(f"Error: Unable to load image '{input_image_path}'")
        continue
    
    # Preprocess image if necessary (resize, normalize, etc.)
    img_resized = cv2.resize(img, (608, 608))  # Example resize
    
    # Perform inference with YOLO model
    blob = cv2.dnn.blobFromImage(img_resized, 1/255.0, (608, 608), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(output_layers)
    
    # Extract bounding boxes and confidence scores from YOLO output
    boxes = []
    confidences = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.5:  # Confidence threshold
                box = detection[0:4] * np.array([img_resized.shape[1], img_resized.shape[0], img_resized.shape[1], img_resized.shape[0]])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
    
    # Non-maxima suppression to eliminate redundant overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            detections.append([x, y, w, h])
    
    # Draw ground truth and detected boxes
    draw_boxes(img_resized, [ann["bbox"] for ann in ground_truth_annotations], color=(0, 255, 0), label="Ground Truth")
    draw_boxes(img_resized, detections, color=(0, 0, 255), label="Detection")
    
    # Save and show the image
    output_image_path = input_image_path.replace('.jpg', '_output.jpg')
    cv2.imwrite(output_image_path, img_resized)
    plt.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    plt.show()
import cv2
import matplotlib.pyplot as plt
import numpy as np

def calculate_iou(boxA, boxB):
    # Calculate intersection coordinates
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    # Calculate intersection area
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Calculate box areas
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    # Calculate IoU
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def draw_boxes(image, boxes, color=(0, 255, 0), label=None):
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        if label:
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

test_dataset = [
    {"image_path": r"C:\Users\jhans\Desktop\ML Internship\opencv-intro\images\soccer.jpg", "annotations": [{"class": "person", "bbox": [50, 50, 200, 200]}]},
    # Add more images and annotations as needed
]

total_tp = 0
total_fp = 0
total_fn = 0

for data in test_dataset:
    input_image_path = data["image_path"]
    ground_truth_annotations = data["annotations"]
    
    # Load image
    img = cv2.imread(input_image_path)
    
    # Check if image is loaded successfully
    if img is None:
        print(f"Error: Unable to load image '{input_image_path}'")
        continue
    
    # Preprocess image if necessary (resize, normalize, etc.)
    img_resized = cv2.resize(img, (608, 608))  # Example resize
    
    # Perform inference with YOLO model
    blob = cv2.dnn.blobFromImage(img_resized, 1/255.0, (608, 608), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(output_layers)
    
    # Extract bounding boxes and confidence scores from YOLO output
    boxes = []
    confidences = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.5:  # Confidence threshold
                box = detection[0:4] * np.array([img_resized.shape[1], img_resized.shape[0], img_resized.shape[1], img_resized.shape[0]])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
    
    # Non-maxima suppression to eliminate redundant overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            detections.append([x, y, w, h])
    
    # Calculate metrics
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for gt_annotation in ground_truth_annotations:
        gt_bbox = gt_annotation["bbox"]
        gt_class = gt_annotation["class"]
        gt_detected = False
        
        for detection in detections:
            iou = calculate_iou(gt_bbox, detection)
            if iou > 0.5:  # IoU threshold for considering a true positive
                true_positives += 1
                gt_detected = True
                break
        
        if not gt_detected:
            false_negatives += 1
    
    false_positives = len(detections) - true_positives
    
    total_tp += true_positives
    total_fp += false_positives
    total_fn += false_negatives
    
    # Draw ground truth and detected boxes
    draw_boxes(img_resized, [ann["bbox"] for ann in ground_truth_annotations], color=(0, 255, 0), label="Ground Truth")
    draw_boxes(img_resized, detections, color=(0, 0, 255), label="Detection")
    
    # Save and show the image
    output_image_path = input_image_path.replace('.jpg', '_output.jpg')
    cv2.imwrite(output_image_path, img_resized)
    plt.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    plt.show()

# Calculate overall precision, recall, and F1 score
precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"Total True Positives: {total_tp}")
print(f"Total False Positives: {total_fp}")
print(f"Total False Negatives: {total_fn}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1_score:.2f}")
import matplotlib.pyplot as plt

# Example data
categories = ['Category A', 'Category B', 'Category C']
values = [10, 20, 15]

# Plotting
plt.bar(categories, values)
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Graph')
plt.show()
