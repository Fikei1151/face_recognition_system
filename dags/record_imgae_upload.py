import os
import pandas as pd
from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

###############################################
# Parameters
###############################################
folder_to_check = "/usr/local/static/uploads"
output_folder = "/usr/local/static/uploads/output"  # โฟลเดอร์สำหรับเก็บไฟล์ CSV
log_file_path = os.path.join(output_folder, 'upload_log.csv')  # กำหนดเส้นทางไฟล์ log

###############################################
# Custom Function Definitions
###############################################
# ฟังก์ชันสำหรับบันทึกข้อมูลการอัปโหลด
def log_upload(file_name):
    os.makedirs(output_folder, exist_ok=True)  # สร้างโฟลเดอร์ถ้ายังไม่มี

    # ถ้าไฟล์ log ยังไม่มี ให้สร้างใหม่
    if not os.path.exists(log_file_path):
        df = pd.DataFrame(columns=['file_name', 'upload_time'])
    else:
        df = pd.read_csv(log_file_path)

    # เพิ่มข้อมูลการอัปโหลดใหม่
    new_entry = {'file_name': file_name, 'upload_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    df = df.append(new_entry, ignore_index=True)

    # บันทึก DataFrame กลับไปยังไฟล์ CSV
    df.to_csv(log_file_path, index=False)

# ฟังก์ชันสำหรับตรวจสอบไฟล์ในโฟลเดอร์และบันทึกเฉพาะไฟล์ใหม่หรือไฟล์ที่ถูกอัปโหลดซ้ำ
def check_file_existence():
    # อ่านข้อมูลไฟล์ log เดิม
    existing_log = []
    if os.path.exists(log_file_path):
        existing_log = pd.read_csv(log_file_path)['file_name'].tolist()

    # อ่านไฟล์และกรองเฉพาะไฟล์ที่ไม่ใช่ output โฟลเดอร์
    files_found = [file for file in os.listdir(folder_to_check) if file != "output" and not os.path.isdir(os.path.join(folder_to_check, file))]

    # ตรวจสอบไฟล์ใหม่หรือไฟล์ที่ถูกอัปโหลดซ้ำ
    for file in files_found:
        if file not in existing_log:
            print(f"New file found: {file}. Logging upload.")
            log_upload(file)  # บันทึกข้อมูลการอัปโหลดไฟล์ใหม่
        else:
            print(f"File {file} already exists in log.")

###############################################
# DAG Definition
###############################################
now = datetime.now()

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(now.year, now.month, now.day),
    "email": ["airflow@airflow.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1)
}

# สร้าง DAG
dag = DAG(
    dag_id="record_time_image_dag",
    description="Custom DAG for checking file existence and processing data.",
    default_args=default_args,
    schedule_interval=timedelta(seconds=1)
)

# สร้าง Task ต่าง ๆ ใน DAG
start = DummyOperator(task_id="start", dag=dag)

# Task ตรวจสอบไฟล์
file_check = PythonOperator(
    task_id="check_image_existence",
    python_callable=check_file_existence,
    dag=dag
)

# Task สุดท้าย
end = DummyOperator(task_id="end", dag=dag)

###############################################
# Task Dependencies
###############################################
# กำหนดลำดับการทำงานของ Task
start >> file_check >> end
