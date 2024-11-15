import contextlib
import cv2
from flask import Flask, render_template, Response, request
from keras.models import load_model
from emoji import emojize
from emotion_recognition.prediction import get_face_from_frame, get_emotions_from_face
from keras.applications.resnet import ResNet50
from emotion_recognition.Models.common_functions import (
    create_model,
    get_data,
    fit,
    evaluation_model,
    saveModel,
)
import face_recognition_api
import pickle
import os
import warnings
import numpy as np

def magnify_results(emotions):
    return "".join(list(map(magnify_emotion, emotions)))

def magnify_emotion(emotion):
    return f"<p>{emotions_with_smiley[emotion[0]]} :{int(emotion[1] * 100)} %</p>"

# Set initial variables
switch, out, capture, rec_frame = 1, 0, 0, 0
face_shape = (80, 80)

# Load the emotion recognition model
parameters = {
    "shape": [80, 80],
    "nbr_classes": 7,
    "batch_size": 8,
    "epochs": 50,
    "number_of_last_layers_trainable": 10,
    "learning_rate": 0.001,
    "nesterov": True,
    "momentum": 0.9,
}
model = create_model(architecture=ResNet50, parameters=parameters)
model.load_weights("emotion_recognition/Models/trained_models/resnet50_ferplus.h5")

# Load the face recognition classifier
fname = 'classifier.pkl'
if os.path.isfile(fname):
    with open(fname, 'rb') as f:
        (le, clf) = pickle.load(f)
else:
    print("Classifier '{}' does not exist".format(fname))
    quit()

# Initialize Flask
app = Flask(__name__, template_folder="./templates", static_folder="./staticFiles")

# Camera setup
camera = cv2.VideoCapture(0)
class_cascade = cv2.CascadeClassifier("./emotion_recognition/ClassifierForOpenCV/frontalface_default.xml")

# Define emoji representations for emotions
emotions_with_smiley = {
    "happy": f"{emojize(':face_with_tears_of_joy:')} HAPPY",
    "angry": f"{emojize(':pouting_face:')} ANGRY",
    "fear": f"{emojize(':fearful_face:')} FEAR",
    "neutral": f"{emojize(':neutral_face:')} NEUTRAL",
    "sad": f"{emojize(':loudly_crying_face:')} SAD",
    "surprise": f"{emojize(':face_screaming_in_fear:')} SURPRISE",
    "disgust": f"{emojize(':nauseated_face:')} DISGUST",
}

frame_count = 0  # Initialize a frame counter
def gen_frames():  # Generate frame by frame from the camera
    global frame_count
    while camera.isOpened():
        success, frame = camera.read()
        if success:
            frame_count += 1
            # Resize frame to reduce resolution for better performance
            frame = cv2.resize(frame, (640, 480))
            
            if frame_count % 2 == 0:  # Process every 2nd frame
                # Create a smaller version for face detection
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                # Detect faces
                face_locations = face_recognition_api.face_locations(small_frame)
                face_encodings = face_recognition_api.face_encodings(small_frame, face_locations)
                
                predictions = []
                if face_encodings:
                    closest_distances = clf.kneighbors(face_encodings, n_neighbors=1)
                    is_recognized = [closest_distances[0][i][0] <= 0.5 for i in range(len(face_locations))]

                    predictions = [
                        (le.inverse_transform([int(pred)])[0].title(), loc) if rec else ("Unknown", loc)
                        for pred, loc, rec in zip(clf.predict(face_encodings), face_locations, is_recognized)
                    ]

                # Detect emotions and draw bounding boxes
                for name, (top, right, bottom, left) in predictions:
                    # Scale back up face locations to original frame size
                    top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4

                    # Draw bounding box around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                    # Add label with the person's name
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

                    # Extract face region for emotion detection
                    face_frame = frame[top:bottom, left:right]
                    face_frame_resized, face_region = get_face_from_frame(face_frame, face_shape, class_cascade=class_cascade)
                    emotions = get_emotions_from_face(face_region, model)
                    
                    # Display emotions near the bounding box
                    if emotions:
                        emotion_display = magnify_results(emotions)
                        cv2.putText(frame, emotion_display, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Flip the frame after drawing text and bounding boxes
            # display_frame = cv2.flip(frame, 1)

            # Encode the flipped frame for streaming
            with contextlib.suppress(Exception):
                ret, buffer = cv2.imencode(".jpg", frame)
                frame = buffer.tobytes()
                yield b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/requests", methods=["POST", "GET"])
def tasks():
    global switch, camera
    if request.method == "GET":
        return render_template("index.html")
    elif request.method == "POST":
        if request.form.get("stop") == "Stop/Start":
            if switch == 1:
                switch = 0
                camera.release()
                cv2.destroyAllWindows()
            else:
                camera = cv2.VideoCapture(0)
                switch = 1
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/time_feed")
def time_feed():
    def generate():
        success, frame = camera.read()
        if success:
            frame, face = get_face_from_frame(
                cv2.flip(frame, 1), face_shape, class_cascade=class_cascade
            )
            emotions = get_emotions_from_face(face, model)
            return magnify_results(emotions) if emotions else "no faces found"

    return Response(generate(), mimetype="text")

if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=3134)
    finally:
        camera.release()
        cv2.destroyAllWindows()
