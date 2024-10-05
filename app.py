from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import cv2
import face_recognition
import numpy as np
from models import db, Person, Emotion
from repvgg import init, detect_emotion
from PIL import Image
import torch
from werkzeug.utils import secure_filename
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from celery import Celery
from datetime import datetime
import time

# Initialize Flask app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://user:password@db:5432/facial_emotion_recognition'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Initialize SQLAlchemy
db.init_app(app)

# Initialize the emotion detection model
device = "cuda" if torch.cuda.is_available() else "cpu"
init(device)

# Initialize Dash app
dash_app = Dash(__name__, server=app, url_base_pathname='/dashboard/', external_stylesheets=[dbc.themes.BOOTSTRAP])

# Initialize Celery
def make_celery(app):
    celery = Celery(
        app.import_name,
        backend='redis://redis:6379/0',
        broker='redis://redis:6379/0'
    )
    celery.conf.update(app.config)
    return celery

celery = make_celery(app)

with app.app_context():
    db.create_all()

# SQLAlchemy setup for session
engine = create_engine('postgresql://user:password@db:5432/facial_emotion_recognition')
Session = sessionmaker(bind=engine)
session = Session()

# File type restrictions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'JPG', 'PNG'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Reusable function for processing image(s) and saving to DB
def process_and_save_image(file_path):
    print(f"Processing file: {file_path}")
    
    # Load image and detect faces
    image = face_recognition.load_image_file(file_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    print(f"Detected {len(face_encodings)} faces")

    detected_faces = []
    
    if len(face_encodings) > 0:
        pil_image = Image.open(file_path).convert("RGB")
        np_image = np.array(pil_image)

        # Detect emotion with RepVGG model
        emotions = detect_emotion([np_image])
        print(f"Emotions detected: {emotions}")

        for encoding, emotion in zip(face_encodings, emotions):
            all_persons = Person.query.all()
            matched_person = None
            
            # Compare encodings with known persons
            for person in all_persons:
                if face_recognition.compare_faces([np.array(person.face_encoding)], encoding)[0]:
                    matched_person = person
                    break
            
            if matched_person:
                # Save emotion to DB for known person
                new_emotion = Emotion(person_id=matched_person.id, emotion=emotion[0], timestamp=datetime.utcnow())
                db.session.add(new_emotion)
                db.session.commit()
                detected_faces.append({'name': matched_person.name, 'encoding': encoding.tolist(), 'emotion': emotion[0]})
            else:
                # Return for user to input name if unknown
                return render_template('name_input.html', encoding=encoding.tolist(), emotion=emotion[0])

    return detected_faces

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        file.save(file_path)
    except Exception as e:
        return jsonify({"error": f"Cannot save file: {str(e)}"}), 500

    # Process and save image, returning the detected faces
    detected_faces = process_and_save_image(file_path)
    
    if detected_faces:
        return render_template('show_faces.html', faces=detected_faces, image_path=file_path)
    else:
        return "No faces detected."

@app.route('/upload_multiple', methods=['POST'])
def upload_multiple():
    files = request.files.getlist('files')
    saved_files = []

    for file in files:
        if file.filename == '':
            continue

        if not allowed_file(file.filename):
            continue

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        try:
            file.save(file_path)
            saved_files.append(file_path)
        except Exception as e:
            continue

    all_faces = []
    for file_path in saved_files:
        detected_faces = process_and_save_image(file_path)
        all_faces.extend(detected_faces)

    return render_template('show_faces_multi.html', faces=all_faces)

@app.route('/show_faces_multi')
def show_faces_multi():
    return render_template('show_faces_multi.html')

@app.route('/upload_multiple_page', methods=['GET'])
def upload_multiple_page():
    return render_template('upload_multiple.html')

@app.route('/save_person', methods=['POST'])
def save_person():
    name = request.form['name']
    encoding = request.form['encoding']
    encoding = np.array(eval(encoding))  # Convert string back to array
    new_person = Person(name=name, face_encoding=encoding)
    db.session.add(new_person)
    db.session.commit()
    return redirect(url_for('index'))
# Dash dashboard layout and callback
def get_data_by_person(person_name):
    records = session.query(Emotion, Person).join(Person).filter(Person.name == person_name).all()
    data = {
        'name': [record.Person.name for record in records],
        'emotion': [record.Emotion.emotion for record in records],
        'timestamp': [record.Emotion.timestamp for record in records]
    }
    return data

def get_all_data():
    records = session.query(Emotion, Person).join(Person).all()
    data = {
        'name': [record.Person.name for record in records],
        'emotion': [record.Emotion.emotion for record in records],
        'timestamp': [record.Emotion.timestamp for record in records]
    }
    return data

def generate_dashboard_layout():
    person_options = [{'label': 'All', 'value': 'all'}]
    all_records = session.query(Person).all()
    for record in all_records:
        person_options.append({'label': record.name, 'value': record.name})

    layout = html.Div([
        dbc.NavbarSimple(
            children=[
                dbc.NavItem(dcc.Link('Home', href='/')),
                dbc.NavItem(dcc.Link('Dashboard', href='/dashboard/')),
            ],
            brand='Facial Emotion Dashboard',
            color='primary',
            dark=True,
        ),
        dbc.Row([
            dbc.Col(html.H2("Emotion Statistics Dashboard")),
            dbc.Col(dcc.Dropdown(id='person-select', options=person_options, value='all', clearable=False)),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='emotion-stats-graph')),
        ]),
    ])
    return layout

# Callback for Dash to update graph
@dash_app.callback(
    Output('emotion-stats-graph', 'figure'),
    [Input('person-select', 'value')]
)
def update_graph(selected_person):
    if selected_person == 'all':
        data = get_all_data()
    else:
        data = get_data_by_person(selected_person)
    
    if not data['timestamp']:
        return px.line()

    fig = px.line(
        x=data['timestamp'],
        y=data['emotion'],
        color=data['name'],
        title=f"Emotion Trends for {'All' if selected_person == 'all' else selected_person}"
    )
    fig.update_layout(xaxis_title='Timestamp', yaxis_title='Emotion')
    return fig

dash_app.layout = generate_dashboard_layout()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
