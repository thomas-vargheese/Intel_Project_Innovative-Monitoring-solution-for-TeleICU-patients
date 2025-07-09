import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from object_detection.utils import label_map_util
import time

# Path to the label map file
PATH_TO_LABELS = PATH_TO_LABELS = r'D:/Object_detection/obj_det/HM/intel/Intel_pythoncode/label_map.pbtxt'


# Path to the saved model file
PATH_TO_SAVED_MODEL = 'teleicu_model.h5'

# Load the label map
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Define the custom MSE function
@tf.keras.utils.register_keras_serializable()
def custom_mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Load the Keras model with the custom object scope
model = tf.keras.models.load_model(PATH_TO_SAVED_MODEL, custom_objects={'custom_mse': custom_mse})

# Function to run inference
def run_inference_for_single_image(model, image):
    original_size = image.shape[:2]  # Get original image size (height, width)
    image_resized = cv2.resize(image, (150, 150))  # Resize the image to match the input size expected by the model
    image_normalized = image_resized.astype(np.float32) / 255.0  # Normalize the image
    input_tensor = tf.convert_to_tensor(image_normalized)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = model.predict(input_tensor)

    # Assuming model output is four values per detection: [ymin, xmin, ymax, xmax]
    detection_boxes = detections[..., :4]

    # Transform the coordinates back to the original image size
    detection_boxes[:, [0, 2]] *= original_size[0]
    detection_boxes[:, [1, 3]] *= original_size[1]

    detection_scores = np.ones((detection_boxes.shape[0],))  # Placeholder for scores
    detection_classes = np.ones((detection_boxes.shape[0],), dtype=np.int64)  # Placeholder for classes
    num_detections = len(detection_boxes)

    # Debugging: Print detection boxes
    print(f"Detection boxes: {detection_boxes}")

    # Convert to dictionary format expected by visualization utils
    detections = {
        'detection_boxes': detection_boxes,
        'detection_scores': detection_scores,
        'detection_classes': detection_classes,
        'num_detections': num_detections
    }

    return detections

# Function to draw bounding boxes and labels on the frame
def draw_bounding_boxes(frame, detections, category_index):
    im_height, im_width, _ = frame.shape

    for i in range(detections['num_detections']):
        box = detections['detection_boxes'][i]
        class_id = detections['detection_classes'][i]
        score = detections['detection_scores'][i]

        ymin, xmin, ymax, xmax = box
        display_str = f"{category_index[class_id]['name']}: {int(score * 100)}%"

        # Convert normalized coordinates to pixel values
        left = int(xmin)
        right = int(xmax)
        top = int(ymin)
        bottom = int(ymax)

        # Draw bounding box
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 0), 2)

        # Calculate text position (top-left corner of the bounding box)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(display_str, font, 0.5, 1)[0]
        text_x = left
        text_y = top - 10 if top - 10 > 10 else top + 10

        # Draw text without bounding box
        cv2.putText(frame, display_str, (text_x, text_y), font, 0.5, (255, 255, 255), 1)

# Real-time prediction with option to run for fixed number of frames or duration
def real_time_prediction(model, video_path=None, num_frames=None, duration=None):
    if video_path:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    start_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video file reached or failed to grab frame.")
            break

        # Run inference
        detections = run_inference_for_single_image(model, frame)

        # Debugging: Print detection results
        print(f"Detections: {detections}")

        # Draw bounding boxes and labels
        draw_bounding_boxes(frame, detections, category_index)

        cv2.imshow('Real-time Prediction', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

        if num_frames and frame_count >= num_frames:
            break
        if duration and (time.time() - start_time) >= duration:
            break

    cap.release()
    cv2.destroyAllWindows()

# Main execution for real-time prediction
if __name__ == "__main__":
    # Path to the video file, if any
    video_path = r'D:/Object_detection/obj_det/HM/intel/Intel_pythoncode/004.mp4'  # Update with the correct path to your video file

    # Run real-time prediction with the TensorFlow model
    real_time_prediction(model, video_path=video_path)
