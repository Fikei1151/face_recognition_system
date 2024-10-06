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
import base64
import io


# Initialize Flask app
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = (
    "postgresql://user:password@db:5432/facial_emotion_recognition"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["UPLOAD_FOLDER"] = "static/uploads/"

# Initialize SQLAlchemy
db.init_app(app)

# Initialize the emotion detection model
device = "cuda" if torch.cuda.is_available() else "cpu"
init(device)

# Initialize Dash app
dash_app = Dash(
    __name__,
    server=app,
    url_base_pathname="/dashboard/",
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)


# Initialize Celery
def make_celery(app):
    celery = Celery(
        app.import_name, backend="redis://redis:6379/0", broker="redis://redis:6379/0"
    )
    celery.conf.update(app.config)
    return celery


celery = make_celery(app)

# Create tables in the database
with app.app_context():
    db.create_all()

# SQLAlchemy setup for session
engine = create_engine("postgresql://user:password@db:5432/facial_emotion_recognition")
Session = sessionmaker(bind=engine)
session = Session()

# File type restrictions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "JPG", "PNG", "webp", "avif"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# Reusable function for processing image(s)
def process_images(file_paths):
    all_faces = []
    unknown_faces = []

    for file_path in file_paths:
        print(f"Processing file: {file_path}")

        # Load image
        image = face_recognition.load_image_file(file_path)
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        if len(face_encodings) > 0:
            pil_image = Image.open(file_path).convert("RGB")
            np_image = np.array(pil_image)
            detected_faces = []

            for encoding in face_encodings:
                matched_person = None
                all_persons = Person.query.all()

                # Compare encodings
                for person in all_persons:
                    if face_recognition.compare_faces(
                        [np.array(person.face_encoding)], encoding
                    )[0]:
                        matched_person = person
                        break

                if matched_person:
                    detected_faces.append(
                        {
                            "name": matched_person.name,
                            "emotion": "Unknown",
                            "image_path": file_path,
                        }
                    )
                else:
                    unknown_faces.append(
                        {"encoding": encoding.tolist(), "image_path": file_path}
                    )

            # Detect emotion for known faces
            if detected_faces:
                emotions = detect_emotion([np_image])
                for idx, emotion in enumerate(emotions):
                    detected_faces[idx]["emotion"] = emotion[0]

            all_faces.extend(detected_faces)

        time.sleep(1)

    return all_faces, unknown_faces


# Flask routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    # ตรวจสอบว่ามีการส่งภาพจากเว็บแคมหรือไม่
    if "captured_image" in request.form:
        captured_image = request.form.get("captured_image")

        # แปลงภาพจาก base64 กลับเป็นไฟล์รูปภาพ
        image_data = base64.b64decode(captured_image.split(",")[1])
        image = Image.open(io.BytesIO(image_data))

        # สร้างชื่อไฟล์และบันทึกลงโฟลเดอร์ static/uploads
        filename = (
            "webcam_image.png"  # คุณสามารถเปลี่ยนเป็นชื่อไดนามิกได้ เช่น timestamp หรือ uuid
        )
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        image.save(file_path)
        print(f"Webcam image saved to {file_path}")

    # ตรวจสอบว่ามีการอัปโหลดไฟล์ปกติหรือไม่
    elif "file" in request.files:
        file = request.files["file"]

        if file.filename == "":
            print("No selected file")
            return redirect(url_for("index"))

        if not allowed_file(file.filename):
            print("Invalid file type")
            return "ไฟล์ไม่ถูกต้อง อนุญาตเฉพาะไฟล์รูปภาพเท่านั้น"

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        # บันทึกไฟล์ปกติ
        try:
            file.save(file_path)
            print(f"File saved to {file_path}")
        except Exception as e:
            print(f"File saving failed: {str(e)}")
            return f"ไม่สามารถบันทึกไฟล์ได้: {str(e)}"

    else:
        return "No image captured or file uploaded."

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
            emotions = detect_emotion([np_image])  # สมมติว่าคุณมีฟังก์ชันนี้เพื่อรับอารมณ์
            print(f"Emotions detected: {emotions}")
        except Exception as e:
            print(f"Error detecting emotion: {str(e)}")
            return f"เกิดข้อผิดพลาดในการตรวจจับอารมณ์: {str(e)}"

        detected_faces = []
        for encoding, emotion in zip(face_encodings, emotions):
            all_persons = Person.query.all()  # สมมติว่าคุณมีโมเดล Person
            matched_person = None
            for person in all_persons:
                if face_recognition.compare_faces(
                    [np.array(person.face_encoding)], encoding
                )[0]:
                    matched_person = person
                    break

            if matched_person:
                # บันทึกอารมณ์ลงในฐานข้อมูล
                new_emotion = Emotion(  # สมมติว่าคุณมีโมเดล Emotion
                    person_id=matched_person.id,
                    emotion=emotion[0],
                    timestamp=datetime.utcnow(),
                )
                db.session.add(new_emotion)
                db.session.commit()

                detected_faces.append(
                    {
                        "name": matched_person.name,
                        "encoding": encoding.tolist(),
                        "emotion": emotion[0],
                    }
                )
            else:
                # ถ้าไม่รู้จัก ให้กรอกชื่อ
                return render_template(
                    "name_input.html", encoding=encoding.tolist(), emotion=emotion[0]
                )

        # แสดงหน้าที่มีรูป ชื่อ และอารมณ์
        return render_template(
            "show_faces.html", faces=detected_faces, image_path=file_path
        )

    print("No faces detected")
    return "No faces detected in the uploaded image."


@app.route("/upload_webcam", methods=["POST"])
def upload_webcam():
    # ตรวจสอบว่ามีภาพที่ถ่ายจากเว็บแคม (base64) หรือไม่
    if "captured_image" in request.form:
        captured_image = request.form.get("captured_image")

        # ตรวจสอบว่าภาพ base64 ถูกส่งมาถูกต้อง
        if captured_image:
            print("Captured Image Received")
        else:
            print("No Image Received")

        # แปลงภาพจาก base64 กลับเป็นไฟล์รูปภาพ
        image_data = base64.b64decode(captured_image.split(",")[1])
        image = Image.open(io.BytesIO(image_data))

        # สร้างชื่อไฟล์และบันทึกลงโฟลเดอร์ static/uploads
        filename = (
            "webcam_image.png"  # คุณสามารถเปลี่ยนให้เป็นชื่อไดนามิกได้เช่น timestamp หรือ uuid
        )
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        image.save(file_path)
        print(f"Webcam image saved to {file_path}")

        # ส่งเส้นทางรูปภาพไปยังหน้าเว็บเพจเพื่อแสดง
        return render_template("show_captured_image.html", image_path=file_path)

    else:
        return "No image captured."


@app.route("/upload_multiple", methods=["POST"])
def upload_multiple():
    files = request.files.getlist("files")
    saved_files = []

    # Save uploaded files
    for file in files:
        if file.filename == "":
            continue

        if not allowed_file(file.filename):
            continue

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        try:
            file.save(file_path)
            saved_files.append(file_path)
        except Exception as e:
            print(f"Error saving file: {str(e)}")
            continue

    # Process the images if there are any saved files
    if len(saved_files) > 0:
        all_faces = []
        unknown_faces = []

        for file_path in saved_files:
            try:
                image = face_recognition.load_image_file(file_path)
                print(f"Processing file: {file_path}")
            except Exception as e:
                print(f"Error loading image: {str(e)}")
                continue

            face_locations = face_recognition.face_locations(image)
            face_encodings = face_recognition.face_encodings(image, face_locations)

            # If faces are found in the image
            if len(face_encodings) > 0:
                try:
                    pil_image = Image.open(file_path).convert("RGB")
                    np_image = np.array(pil_image)
                    emotions = detect_emotion(
                        [np_image]
                    )  # Detect emotions for each face
                except Exception as e:
                    print(f"Error detecting emotion: {str(e)}")
                    continue

                # Process each face in the image
                detected_faces = []
                for encoding, emotion in zip(face_encodings, emotions):
                    all_persons = Person.query.all()
                    matched_person = None

                    # Compare this face encoding with all known persons
                    for person in all_persons:
                        if face_recognition.compare_faces(
                            [np.array(person.face_encoding)], encoding
                        )[0]:
                            matched_person = person
                            break

                    if matched_person:
                        # If person is matched, update emotion in the database
                        new_emotion = Emotion(
                            person_id=matched_person.id,
                            emotion=emotion[0],
                            timestamp=datetime.utcnow(),
                        )
                        db.session.add(new_emotion)
                        db.session.commit()  # Make sure to commit here!

                        detected_faces.append(
                            {
                                "name": matched_person.name,
                                "encoding": encoding.tolist(),
                                "emotion": emotion[0],
                            }
                        )
                    else:
                        # If no match, store for later input
                        unknown_faces.append(
                            {
                                "encoding": encoding.tolist(),
                                "emotion": emotion[0],
                                "image_path": file_path,
                            }
                        )

                all_faces.extend(detected_faces)

        # Return template with known and unknown faces
        return render_template(
            "show_faces_multi.html", faces=all_faces, unknown_faces=unknown_faces
        )

    return "No files uploaded."


@app.route("/show_faces_multi")
def show_faces_multi():
    return render_template("show_faces_multi.html")


@app.route("/upload_multiple_page", methods=["GET"])
def upload_multiple_page():
    return render_template("upload_multiple.html")


@app.route("/save_person", methods=["POST"])
def save_person():
    name = request.form["name"]
    encoding = np.array(eval(request.form["encoding"]))  # แปลงข้อมูลกลับเป็น numpy array

    new_person = Person(name=name, face_encoding=encoding)
    db.session.add(new_person)
    db.session.commit()

    return redirect(url_for("index"))


# Dash dashboard layout and callback
def get_data_by_person(person_name):
    records = (
        session.query(Emotion, Person)
        .join(Person)
        .filter(Person.name == person_name)
        .all()
    )
    data = {
        "name": [record.Person.name for record in records],
        "emotion": [record.Emotion.emotion for record in records],
        "timestamp": [record.Emotion.timestamp for record in records],
    }
    return data


def get_all_data():
    records = session.query(Emotion, Person).join(Person).all()
    data = {
        "name": [record.Person.name for record in records],
        "emotion": [record.Emotion.emotion for record in records],
        "timestamp": [record.Emotion.timestamp for record in records],
    }
    print(data["name"])
    return data


def generate_dashboard_layout():
    # Query ข้อมูลจากฐานข้อมูล (สมมติว่ามี Person และ session)
    all_records = session.query(Person).all()

    # สร้าง list ของชื่อจาก all_records
    names = [record.name for record in all_records]

    # ใช้ set เพื่อทำให้ชื่อไม่ซ้ำกัน
    unique_names = list(set(names))

    # สร้าง options สำหรับ dropdown โดยไม่มีชื่อซ้ำ
    person_options = [{"label": "All", "value": "all"}]
    for name in unique_names:
        person_options.append({"label": name, "value": name})

    layout = html.Div(
        [
            # Navbar ใช้ธีมเหมือนหน้า HTML อื่นๆ
            dbc.NavbarSimple(
                children=[
                    # ปรับลิงก์ Home ให้ชี้ไปยัง route "/"
                    dbc.NavItem(html.A("Home", href="/", className="nav-link")),
                    dbc.NavItem(
                        html.A("Dashboard", href="/dashboard/", className="nav-link")
                    ),
                ],
                brand="Facial Emotion Dashboard",
                color="primary",
                dark=True,
            ),
            # เพิ่ม container และ row สำหรับจัด layout
            dbc.Container(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                html.H2("Emotion Statistics Dashboard"),
                                className="text-center mt-4",
                            ),
                            dbc.Col(
                                dcc.Dropdown(
                                    id="person-select",
                                    options=person_options,
                                    value="all",
                                    clearable=False,
                                ),
                                width=4,
                                className="mt-4",
                            ),
                        ]
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Graph(id="emotion-stats-graph"),
                                width=12,
                                className="mt-4",
                            ),
                        ]
                    ),
                    # เพิ่มตารางแสดงผลอารมณ์หลักของแต่ละวัน
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Div(id="emotion-summary-table"),
                                width=12,
                                className="mt-4",
                            ),
                        ]
                    ),
                ],
                className="mt-5",
            ),
        ]
    )
    return layout

def extract_emotion_type(emotion_with_percentage):
    # ตัดส่วนที่เป็นเปอร์เซ็นต์ออกจากประเภทอารมณ์
    if "(" in emotion_with_percentage:
        return emotion_with_percentage.split(" (")[0]  # เก็บเฉพาะประเภทอารมณ์
    return emotion_with_percentage



# Function to calculate the dominant emotion for each day
def calculate_dominant_emotion(data):
    # สร้าง dictionary สำหรับเก็บผลลัพธ์รายวัน
    daily_emotion_summary = {}

    # สมมติว่าข้อมูล timestamp และ emotion ถูกจับคู่กัน
    for ts, emotion in zip(data["timestamp"], data["emotion"]):
        # แปลง ts (datetime) เป็น string เพื่อใช้วันที่
        # print(data["emotion"])
        day = ts.strftime("%Y-%m-%d")  # เปลี่ยน datetime เป็นรูปแบบ 'YYYY-MM-DD'

        # สร้างรายการสำหรับวันนั้นถ้ายังไม่มี
        if day not in daily_emotion_summary:
            daily_emotion_summary[day] = []

        daily_emotion_summary[day].append(emotion)

    # สรุปผลอารมณ์หลักของแต่ละวัน (โดยนับอารมณ์ที่เกิดบ่อยที่สุดในแต่ละวัน)
    dominant_emotions = {}
    for day, emotions in daily_emotion_summary.items():
        dominant_emotions[day] = max(set(emotions), key=emotions.count)

    return dominant_emotions

@dash_app.callback(
    [
        Output("emotion-stats-graph", "figure"),
        Output("emotion-summary-table", "children"),
    ],
    [Input("person-select", "value")],
)
def update_dashboard(selected_person):
    # ถ้าเลือก All แสดงข้อมูลของทุกคน
    if selected_person == "all":
        data = get_all_data()

        # ตรวจสอบว่ามีข้อมูล timestamp หรือไม่
        if not data["timestamp"]:
            return px.line(), ""

        # ทำความสะอาดข้อมูลอารมณ์ โดยดึงเฉพาะประเภทอารมณ์
        cleaned_emotions = [
            extract_emotion_type(emotion) for emotion in data["emotion"]
        ]

        # พล็อตกราฟสำหรับทุกคน โดยแสดงสีแยกตามชื่อ
        fig = px.line(
            x=data["timestamp"],  # แกน X เป็น timestamp
            y=cleaned_emotions,  # แกน Y เป็นอารมณ์ที่ถูกทำความสะอาดแล้ว
            color=data["name"],  # แสดงสีแยกตามชื่อบุคคล
            title="Emotion Trends for All People",
        )
        fig.update_layout(xaxis_title="Timestamp", yaxis_title="Emotion")

        # คำนวณอารมณ์หลักของแต่ละวันสำหรับทุกคน
        dominant_emotions = calculate_dominant_emotion(
            {
                "timestamp": data["timestamp"],
                "emotion": cleaned_emotions,  # ใช้ข้อมูลอารมณ์ที่ถูกทำความสะอาดแล้ว
            }
        )

        # สร้างตารางแสดงผลอารมณ์หลักของแต่ละวันสำหรับทุกคน
        summary_table = dbc.Table(
            # Header
            [html.Thead(html.Tr([html.Th("Date"), html.Th("Dominant Emotion")]))] +
            # Body
            [
                html.Tbody(
                    [
                        html.Tr([html.Td(day), html.Td(emotion)])
                        for day, emotion in dominant_emotions.items()
                    ]
                )
            ],
            bordered=True,
            hover=True,
            striped=True,
        )

    # ถ้าเลือกบุคคลใดบุคคลหนึ่ง แสดงข้อมูลของบุคคลนั้นเท่านั้น
    else:
        data = get_data_by_person(selected_person)

        # ตรวจสอบว่ามีข้อมูล timestamp หรือไม่
        if not data["timestamp"]:
            return px.line(), ""

        # ทำความสะอาดข้อมูลอารมณ์ โดยดึงเฉพาะประเภทอารมณ์
        cleaned_emotions = [
            extract_emotion_type(emotion) for emotion in data["emotion"]
        ]

        # พล็อตกราฟสำหรับบุคคลที่เลือก
        fig = px.line(
            x=data["timestamp"],  # แกน X เป็น timestamp
            y=cleaned_emotions,  # แกน Y เป็นอารมณ์ที่ถูกทำความสะอาดแล้ว
            color=data["name"],  # แสดงสีแยกตามชื่อบุคคล
            title=f"Emotion Trends for {selected_person}",
        )
        fig.update_layout(xaxis_title="Timestamp", yaxis_title="Emotion")

        # คำนวณอารมณ์หลักของแต่ละวันสำหรับบุคคลนั้น
        dominant_emotions = calculate_dominant_emotion(
            {
                "timestamp": data["timestamp"],
                "emotion": cleaned_emotions,  # ใช้ข้อมูลอารมณ์ที่ถูกทำความสะอาดแล้ว
            }
        )

        # สร้างตารางแสดงผลอารมณ์หลักของแต่ละวันสำหรับบุคคลนั้น
        summary_table = dbc.Table(
            # Header
            [html.Thead(html.Tr([html.Th("Date"), html.Th("Dominant Emotion")]))] +
            # Body
            [
                html.Tbody(
                    [
                        html.Tr([html.Td(day), html.Td(emotion)])
                        for day, emotion in dominant_emotions.items()
                    ]
                )
            ],
            bordered=True,
            hover=True,
            striped=True,
        )

    return fig, summary_table


dash_app.layout = generate_dashboard_layout()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
