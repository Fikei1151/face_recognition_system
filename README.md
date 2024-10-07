# Face Recognition System

Member
1. Nicha   Vikromrotjananan 6410110175
2. Fikree  Hajiyusof        6410110697
3. Satid   Deepeng          6410110724

## Overview
This repository implements a Facial Emotion Recognition System using a Flask web application. The system can detect and recognize human faces from uploaded images or images captured via webcam, identify known individuals, and classify their facial emotions using a deep learning model. The recognized emotions and the corresponding timestamp are saved in a PostgreSQL database for further analysis.

## Features
- Face Detection and Recognition: Detects faces in uploaded images and identifies known persons.
- Emotion Classification: Classifies emotions for detected faces using a deep learning model based on RepVGG.
- Image Upload and Webcam Capture: Supports both image file uploads and webcam captures.
- Emotion Logging: Saves emotion data with timestamps to a PostgreSQL database.
- Dashboard Visualization: Provides a visualization dashboard using Dash and Plotly to display emotion trends and statistics.
- Multi-image Processing: Supports the uploading and processing of multiple images simultaneously.
- User Registration: Allows the registration of new users by saving their face encodings in the database.
