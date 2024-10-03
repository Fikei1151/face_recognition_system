from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import face_recognition
import numpy as np
from models import db, Person, Emotion
from repvgg import init, detect_emotion
from PIL import Image
import torch
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://user:password@db:5432/facial_emotion_recognition'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

db.init_app(app)

# Create tables if they don't exist
with app.app_context():
    db.create_all()

# Initialize the emotion detection model
device = "cuda" if torch.cuda.is_available() else "cpu"
init(device)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    # Load and detect faces
    image = face_recognition.load_image_file(file_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    if len(face_encodings) > 0:
        pil_image = Image.open(file_path).convert("RGB")
        np_image = np.array(pil_image)
        
        # Detect emotion using the RepVGG model
        emotions = detect_emotion([np_image])
        
        detected_faces = []
        for encoding, emotion in zip(face_encodings, emotions):
            # Fetch all known faces from the database
            all_persons = Person.query.all()

            # Compare with each known face in the database
            matched_person = None
            for person in all_persons:
                if face_recognition.compare_faces([np.array(person.face_encoding)], encoding)[0]:
                    matched_person = person
                    break
            
            if matched_person:
                # If the person is recognized, show their name and emotion
                detected_faces.append({'name': matched_person.name, 'encoding': encoding.tolist(), 'emotion': emotion[0]})
            else:
                # If the person is not recognized, ask for the name
                return render_template('name_input.html', encoding=encoding.tolist(), emotion=emotion[0])

        # Pass the detected faces and emotions to a new page
        # Render show_faces.html with the detected faces and image path
        return render_template('show_faces.html', faces=detected_faces, image_path=file_path)

    
    return redirect(url_for('index'))
@app.route('/show_faces', methods=['POST'])
def show_faces():
    # Handle name updates and other logic for the faces displayed in the form
    for i, name in enumerate(request.form.getlist('name')):
        encoding = np.array(eval(request.form.getlist('encoding')[i]))  # Convert string back to numpy array
        existing_person = Person.query.filter_by(face_encoding=encoding.tolist()).first()

        if not existing_person:
            # Add new person if they don't exist in the database
            new_person = Person(name=name, face_encoding=encoding)
            db.session.add(new_person)
        else:
            # Update the name of the existing person
            existing_person.name = name

        db.session.commit()

    return redirect(url_for('index'))

@app.route('/save_person', methods=['POST'])
def save_person():
    name = request.form['name']
    encoding = request.form['encoding']
    encoding = np.array(eval(encoding))  # Convert string back to numpy array
    new_person = Person(name=name, face_encoding=encoding)
    db.session.add(new_person)
    db.session.commit()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
