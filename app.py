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


# Initialize the emotion detection model
device = "cuda" if torch.cuda.is_available() else "cpu"
init(device)

# ประเภทไฟล์ที่อนุญาตให้อัปโหลด
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'JPG', 'PNG'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
from celery import Celery  # เพิ่มการนำเข้า Celery

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        print("No file part in request")
        return redirect(url_for('index'))

    file = request.files['file']
    print(f"Uploaded file: {file.filename}")

    if file.filename == '':
        print("No selected file")
        return redirect(url_for('index'))

    # ตรวจสอบประเภทไฟล์ที่อนุญาต
    if not allowed_file(file.filename):
        print("Invalid file type")
        return "ไฟล์ไม่ถูกต้อง อนุญาตเฉพาะไฟล์รูปภาพเท่านั้น"

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # บันทึกไฟล์
    try:
        file.save(file_path)
        print(f"File saved to {file_path}")
    except Exception as e:
        print(f"File saving failed: {str(e)}")
        return f"ไม่สามารถบันทึกไฟล์ได้: {str(e)}"
    
    # โหลดและตรวจจับใบหน้า
    try:
        image = face_recognition.load_image_file(file_path)
        print("Image loaded successfully")
    except Exception as e:
        print(f"Error loading image: {str(e)}")
        return f"เกิดข้อผิดพลาดในการโหลดรูปภาพ: {str(e)}"

    # ตรวจจับตำแหน่งของใบหน้า
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    print(f"Detected {len(face_encodings)} faces")

    if len(face_encodings) > 0:
        try:
            pil_image = Image.open(file_path).convert("RGB")
            np_image = np.array(pil_image)

            # ตรวจจับอารมณ์ด้วยโมเดล RepVGG
            emotions = detect_emotion([np_image])
            print(f"Emotions detected: {emotions}")
        except Exception as e:
            print(f"Error detecting emotion: {str(e)}")
            return f"เกิดข้อผิดพลาดในการตรวจจับอารมณ์: {str(e)}"
        
        detected_faces = []
        for encoding, emotion in zip(face_encodings, emotions):
            all_persons = Person.query.all()
            matched_person = None
            for person in all_persons:
                if face_recognition.compare_faces([np.array(person.face_encoding)], encoding)[0]:
                    matched_person = person
                    break
            
            if matched_person:
                # ถ้ารู้จักบุคคลนี้ ให้แสดงชื่อและอารมณ์
                detected_faces.append({'name': matched_person.name, 'encoding': encoding.tolist(), 'emotion': emotion[0]})
            else:
                # ถ้าไม่รู้จัก ให้กรอกชื่อ
                return render_template('name_input.html', encoding=encoding.tolist(), emotion=emotion[0])

        # แสดงหน้าที่มีรูป ชื่อ และอารมณ์
        return render_template('show_faces.html', faces=detected_faces, image_path=file_path)

    print("No faces detected")
    return "No faces detected in the uploaded image."

@app.route('/show_faces', methods=['POST'])
def show_faces():
    # อัปเดตชื่อบุคคลจากหน้า show_faces
    for i, name in enumerate(request.form.getlist('name')):
        encoding = np.array(eval(request.form.getlist('encoding')[i]))  # แปลงข้อมูลกลับเป็น numpy array
        existing_person = Person.query.filter_by(face_encoding=encoding.tolist()).first()

        if not existing_person:
            # เพิ่มบุคคลใหม่ถ้าไม่พบในฐานข้อมูล
            new_person = Person(name=name, face_encoding=encoding)
            db.session.add(new_person)
        else:
            # อัปเดตชื่อของบุคคลที่มีอยู่แล้ว
            existing_person.name = name

        db.session.commit()

    return redirect(url_for('index'))

@app.route('/save_person', methods=['POST'])
def save_person():
    name = request.form['name']
    encoding = request.form['encoding']
    encoding = np.array(eval(encoding))  # แปลงข้อมูลกลับเป็น numpy array
    new_person = Person(name=name, face_encoding=encoding)
    db.session.add(new_person)
    db.session.commit()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)