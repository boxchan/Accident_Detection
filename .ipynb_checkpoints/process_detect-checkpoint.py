import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

import tempfile
import subprocess
import re

# hyper params
batch_size = 100
img_height = 300
img_width = 300

training_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'data_collision/train/',
    image_size= (img_height, img_width),
    batch_size=batch_size

)
testing_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'data_collision/test/',
    image_size= (img_height, img_width),
    batch_size=batch_size)
validation_ds =  tf.keras.preprocessing.image_dataset_from_directory(
 'data_collision/val/',
    image_size= (img_height, img_width),
    batch_size=batch_size)

class_names = training_ds.class_names
# print(class_names)
AUTOTUNE = tf.data.experimental.AUTOTUNE
training_ds = training_ds.cache().prefetch(buffer_size=AUTOTUNE)
testing_ds = testing_ds.cache().prefetch(buffer_size=AUTOTUNE)
img_shape = (img_height, img_width, 3)


# run detection of how many persons in the image
def run_detection(image_path):
    command = f"python ./yolov5-master/detect.py --source {image_path}"
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    # print("Command Output:\n", result.stderr)
    
    matches = re.findall(r"(\d+) persons?", result.stderr)
    
    total_persons_detected = sum(int(match) for match in matches)
    if total_persons_detected > 0:
        print(f"Person detected in the image. Total persons: {total_persons_detected}.")
        return True
    else:
        print("No person detected in the image.")
        return False

# test if there is an accident in the image
def predict_frame(img, model):
    img_array = tf.keras.utils.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    prediction=(model.predict(img_batch) < 0.5).astype("int32")
    # print("prediction in predict_frame", prediction)
    if(prediction[0][0]==0):
        return(("Accident Detected", 1))
    else:
        return(("No Accident", 0))

# import the trained model and parameters
trained_model = tf.keras.models.load_model("AccidentDetectionModel.h5")
acc_vector = []

# test the trained model
# n = 0
# for images, labels in testing_ds.take(1):
#     predictions = trained_model.predict(images)
#     predict_labels = []  # class_names
#     prdind = []  # index
    
#     for p in predictions:
#         predict_labels.append(class_names[np.argmax(p)])
#         prdind.append(np.argmax(p))
    
#     acc_vector = (np.array(prdlbl) == labels)
#     print('accuracy: ', np.mean(acc_vector))
    
#     n+=1


if __name__ == "__main__":
    # image_path = "./yolov5-master/data/images/bus.jpg"
    # run_detection(image_path)
    
    images=[]  # list of image frames
    pred_labels=[]  # [("accident detected", 1)] list of labels of each image frame, if collision detected
    
    humans_detected = []  # humans detected in each image frame
    if_accident_detected = False
    sec_count = 0  #  count seconds after collision
    alert_threshold = 3  # alert after 3 seconds
    
    # read the video stream
    cap= cv2.VideoCapture('data_collision/videoplayback_demo.mp4')
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    fps = cap.get(cv2.CAP_PROP_FPS)  # get the frames per second (fps) of the video stream
    frame_counts = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # total frame counts of the video stream
    print("fps, frame_counts: ", fps, frame_counts)

    for i in range(frame_counts):
        ret, frame = cap.read()  # ret: if frame is successfully grabbed; frame_shape = (360.640,3)
        if not ret:
            print("Error: ret is false.")
            break
        if i%fps==0:  # every second
            print(i)
            # print("ret, frame: ", ret, frame, frame.shape)
            # resize the image frame from (360,640) to (300,300)
            resized_frame=tf.keras.preprocessing.image.smart_resize(frame, (img_height, img_width), interpolation='bilinear')
            images.append(frame)
            collision_detect = predict_frame(resized_frame, trained_model)
            pred_labels.append(collision_detect)
            # accident detected == True
            if collision_detect[1]:  
                print("Accident Detected at ", i/fps, " seconds")
                if_accident_detected = True
                # save the frame per second for object detection
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                    cv2.imwrite(tmp_file.name, frame)
                    humans_detected.append(run_detection(tmp_file.name))
            
            # algorithm to roughly determin the severity of the accident: after threshold seconds but no persons are detected, alert
            if sec_count >= alert_threshold:
                if sum(humans_detected) == 0:
                    print("Alarm sent to 111!")
                    print("Alarm at ", i/fps, " seconds")
                else:  # >0
                    print("alarm not sent since there are others to help.")
                # reset
                sec_count = 0
                humans_detected = []
                if_accident_detected = False
            # count the number of seconds while accident happened
            if if_accident_detected:
                sec_count+=1
    cap.release()
