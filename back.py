from flask import Flask, render_template, Response, request
import cv2
from threading import Thread
import numpy as np
import dlib
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import smtplib
from time import time
import torch

app = Flask(__name__)

# ObjectDetection class for YOLO object detection
class ObjectDetection:
    def __init__(self):
        self.model = YOLO("yolov8n")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def predict(self, im0):
        results = self.model(im0)
        return results

# FaceRecognition class for face recognition
class FaceRecognition:
    def __init__(self):
        # Load reference images and face recognition models
        self.reference_images = []  # List to store reference images
        self.reference_face_encodings = []  # List to store face encodings

        # Load dlib models
        self.detector = dlib.get_frontal_face_detector()
        self.sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

    def add_reference_image(self, image_path):
        # Add a reference image
        reference_image = cv2.imread(image_path)
        if reference_image is not None:
            self.reference_images.append(reference_image)
            face_encoding = self.calculate_face_encoding(reference_image)
            self.reference_face_encodings.append(face_encoding)
            return True
        else:
            return False

    def calculate_face_encoding(self, image):
        # Detect faces in the image
        faces = self.detector(image, 1)

        if len(faces) == 0:
            print("No faces found in the image.")
            return None

        # Compute face encodings
        face_encodings = []
        for face in faces:
            shape = self.sp(image, face)
            face_encoding = self.facerec.compute_face_descriptor(image, shape)
            face_encodings.append(np.array(face_encoding))

        return face_encodings

    def compare_faces(self, face_to_compare):
        # Compare face_to_compare with each known face encoding
        matches = []
        for known_face_encoding in self.reference_face_encodings:
            distance = np.linalg.norm(known_face_encoding - face_to_compare)
            # Lower distance indicates more similar faces
            if distance < 0.6:
                matches.append(True)
            else:
                matches.append(False)
        return matches

# Email sender function
def send_email(to_email, from_email, object_detected=1, image_path=None):
    password = "your_password"
    message = MIMEMultipart()
    message['From'] = from_email
    message['To'] = to_email
    message['Subject'] = "Security Alert"
    # Add in the message body
    message_body = f'ALERT - {object_detected} objects has been detected!!'
    message.attach(MIMEText(message_body, 'plain'))

    # Attach the image if available
    if image_path:
        with open(image_path, 'rb') as img_file:
            msg_image = MIMEImage(img_file.read())
            msg_image.add_header('Content-Disposition', 'attachment', filename='detected_person.jpg')
            message.attach(msg_image)

    server = smtplib.SMTP('smtp.gmail.com: 587')
    server.starttls()
    server.login(from_email, password)
    server.sendmail(from_email, to_email, message.as_string())
    server.quit()

# Initialize ObjectDetection and FaceRecognition classes
object_detector = ObjectDetection()
face_recognizer = FaceRecognition()

# Flag to control program execution
program_active = False

# Function to handle object detection and face recognition
def run_detection():
    global program_active
    cap = cv2.VideoCapture(0)
    while program_active:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Perform face recognition
        input_face_encodings = face_recognizer.calculate_face_encoding(frame)
        if input_face_encodings is not None:
            for face_encoding in input_face_encodings:
                matches = face_recognizer.compare_faces(face_encoding)
                if True in matches:
                    print("Match found!")
                else:
                    print("No match found.")
                    # Perform object detection if no match found
                    results = object_detector.predict(frame)
                    # Process detection results and send email
                    # (Implementation left to the user)

        # Display the frame
        cv2.imshow('Camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Thread for program execution
program_thread = Thread(target=run_detection)

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start')
def start_program():
    global program_active
    program_active = True
    program_thread.start()
    return 'Program started.'

@app.route('/stop')
def stop_program():
    global program_active
    program_active = False
    return 'Program stopped.'

@app.route('/add_reference_image', methods=['POST'])
def add_reference_image():
    # Get reference image from form data
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    # Save file and add reference image
    file.save(file.filename)
    if face_recognizer.add_reference_image(file.filename):
        return 'Reference image added successfully'
    else:
        return 'Failed to add reference image'

if __name__ == '__main__':
    app.run(debug=True)
