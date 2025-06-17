import base64
import os
from datetime import datetime
from time import time
import cv2
import flet as ft
from scipy.spatial import distance
import dlib
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import serial
import serial.tools.list_ports

# Настройки Arduino
ARDUINO_PORT = None  # Автоматическое определение
BAUD_RATE = 9600
ser = None

# Пути к моделям
shape_predictor_path = 'face_model/shape_predictor_68_face_landmarks.dat'
face_rec_model_path = 'face_model/dlib_face_recognition_resnet_model_v1.dat'
base_path = 'face_bd'
log_file = 'detection_log.txt'

# Глобальные переменные
is_running = False
cap = None
face_descriptors = []
faces = []
file_picker = ft.FilePicker()

# Инициализация детектора и моделей
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(shape_predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

def init_arduino_connection():
    """Инициализация соединения с Arduino для управления реле"""
    global ser, ARDUINO_PORT
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if 'Arduino' in port.description or 'CH340' in port.description:
            ARDUINO_PORT = port.device
            break
    
    if ARDUINO_PORT:
        try:
            ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
            print(f"Connected to Arduino on {ARDUINO_PORT}")
            return True
        except serial.SerialException as e:
            print(f"Failed to connect to Arduino: {e}")
            return False
    else:
        print("Arduino not found. Relay control will be disabled.")
        return False

def control_relay(state):
    if ser and ser.is_open:
        try:
            ser.write(b'1' if state else b'0')
        except serial.SerialException as e:
            print("Arduino error:", e)

def putText_rus(img, text, pos, color=(0, 255, 0), font_size=20):
    """Рисует текст с поддержкой кириллицы на изображении"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    draw.text(pos, text, font=font, fill=color[::-1])
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def log_detection(name):
    """Логирование обнаруженных лиц"""
    with open(log_file, 'a', encoding='utf-8') as log:
        log.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Detected: {name}\n")

def load_face_descriptors():
    """Загрузка базы лиц и их дескрипторов"""
    global face_descriptors, faces
    face_descriptors = []
    faces = os.listdir(base_path)

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    if not faces:
        print(f"В папке {base_path} нет сохранённых лиц")
        return False

    for face in faces:
        img_path = os.path.join(base_path, face)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Ошибка загрузки изображения {img_path}")
            continue

        dets = detector(img, 1)
        if len(dets) == 0:
            print(f"Лицо не найдено на изображении {img_path}")
            continue

        for d in dets:
            shape = sp(img, d)
            face_descriptors.append(facerec.compute_face_descriptor(img, shape))

    print(f"Загружено {len(face_descriptors)} лиц.")
    return True

def compare_faces(frame, threshold=0.6):
    """Сравнение лиц и поиск совпадений"""
    dets = detector(frame, 0)

    if len(dets) == 0:
        control_relay(False)
        return None, None, False

    for d in dets:
        shape = sp(frame, d)
        main_descriptor = facerec.compute_face_descriptor(frame, shape)
        distances = [distance.euclidean(main_descriptor, fd) for fd in face_descriptors]

        min_dist = min(distances)
        closest_face_idx = distances.index(min_dist)
        is_match = min_dist <= threshold

        control_relay(is_match)
        return min_dist, faces[closest_face_idx], is_match

    control_relay(False)
    return None, None, False

def resize_image(image, max_width=1920, max_height=1080):
    """Сжатие изображения"""
    h, w = image.shape[:2]
    if w > max_width or h > max_height:
        scale = min(max_width / w, max_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h))
    return image

def image_to_base64(image):
    """Конвертация изображения в base64"""
    _, encoded_img = cv2.imencode('.png', image)
    return base64.b64encode(encoded_img).decode('utf-8')

def exit_mode(page):
    """Выход из текущего режима"""
    global is_running, cap
    is_running = False
    if cap is not None:
        cap.release()
        cap = None
    
    control_relay(False)
    
    for control in page.controls[:]:
        page.controls.remove(control)

    start_interface(page)
    page.update()

def start_webcam(page, image_area, match_area, status_text, exit_button):
    """Запуск веб-камеры"""
    global is_running, cap
    is_running = True
    cap = cv2.VideoCapture(0)
    exit_button.visible = True
    page.update()

    last_detected = None
    last_detection_time = 0

    while is_running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = resize_image(frame)
        min_dist, closest_face, is_match = compare_faces(frame)
        current_time = time()

        if is_match:
            face_name = os.path.splitext(closest_face)[0]
            
            if last_detected != closest_face or current_time - last_detection_time > 5:
                log_detection(closest_face)
                status_text.value = f"Найдено: {face_name}"
                status_text.color = ft.Colors.GREEN

                match_img_path = os.path.join(base_path, closest_face)
                match_img = cv2.imread(match_img_path)
                match_img = resize_image(match_img)
                match_area.src_base64 = image_to_base64(match_img)

            last_detected = closest_face
            last_detection_time = current_time
            frame = putText_rus(frame, f"Найдено: {face_name}", (10, 30), (0, 255, 0), 20)
        else:
            if current_time - last_detection_time > 5:
                last_detected = None
                status_text.value = "Лицо не найдено"
                status_text.color = ft.Colors.RED
                match_area.src_base64 = None
                frame = putText_rus(frame, "Лицо не найдено", (10, 30), (0, 0, 255), 20)

        image_area.src_base64 = image_to_base64(frame)
        page.update()

    if cap is not None:
        cap.release()

def process_selected_image(e, page, image_area, match_area, status_text, exit_button):
    """Обработка выбранного изображения"""
    if e.files:
        image_path = e.files[0].path
        img = cv2.imread(image_path)

        if img is None:
            status_text.value = "Ошибка загрузки изображения"
            status_text.color = ft.Colors.RED
            page.update()
            return

        img = resize_image(img)
        min_dist, closest_face, is_match = compare_faces(img)

        if is_match:
            face_name = os.path.splitext(closest_face)[0]
            status_text.value = f"Совпадение: {face_name} (Расстояние: {min_dist:.2f})"
            status_text.color = ft.Colors.GREEN
            log_detection(closest_face)

            match_img_path = os.path.join(base_path, closest_face)
            match_img = cv2.imread(match_img_path)
            match_img = resize_image(match_img)
            match_area.src_base64 = image_to_base64(match_img)
        else:
            status_text.value = "Совпадений не найдено"
            status_text.color = ft.Colors.RED
            match_area.src_base64 = None

        image_area.src_base64 = image_to_base64(img)
        page.update()

def pick_image(page, image_area, match_area, status_text, exit_button):
    """Выбор изображения для анализа"""
    file_picker.on_result = lambda e: process_selected_image(e, page, image_area, match_area, status_text, exit_button)
    file_picker.pick_files("Выберите изображение", allowed_extensions=["jpg", "jpeg", "png"])
    exit_button.visible = True
    page.update()

def process_selected_video(e, page, image_area, match_area, status_text, exit_button):
    """Обработка выбранного видео"""
    global is_running, cap
    if e.files:
        is_running = True
        video_path = e.files[0].path
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        last_detected = None

        while is_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = resize_image(frame)
            frame_count += 1

            if frame_count % 30 == 0:
                min_dist, closest_face, is_match = compare_faces(frame)

                if is_match:
                    face_name = os.path.splitext(closest_face)[0]
                    
                    if last_detected != closest_face:
                        log_detection(closest_face)
                        status_text.value = f"Найдено: {face_name}"
                        status_text.color = ft.Colors.GREEN

                        match_img_path = os.path.join(base_path, closest_face)
                        match_img = cv2.imread(match_img_path)
                        match_img = resize_image(match_img)
                        match_area.src_base64 = image_to_base64(match_img)

                    last_detected = closest_face
                    frame = putText_rus(frame, f"Найдено: {face_name}", (10, 30), (0, 255, 0), 20)
                else:
                    if last_detected:
                        status_text.value = "Лицо не найдено"
                        status_text.color = ft.Colors.RED
                        match_area.src_base64 = None
                    frame = putText_rus(frame, "Лицо не найдено", (10, 30), (0, 0, 255), 20)

            image_area.src_base64 = image_to_base64(frame)
            page.update()

        if cap is not None:
            cap.release()

def pick_video(page, image_area, match_area, status_text, exit_button):
    """Выбор видео для анализа"""
    file_picker.on_result = lambda e: process_selected_video(e, page, image_area, match_area, status_text, exit_button)
    file_picker.pick_files("Выберите видео", allowed_extensions=["mp4", "avi", "mov"])
    exit_button.visible = True
    page.update()

def start_interface(page: ft.Page):
    """Инициализация интерфейса"""
    init_arduino_connection()
    
    page.title = "Распознавание лиц с Arduino"
    page.window_width = 1400
    page.window_height = 800
    page.window_resizable = False
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.padding = 20

    page.overlay.append(file_picker)

    button_style = ft.ButtonStyle(
        padding=20,
        bgcolor=ft.Colors.BLUE_700,
        color=ft.Colors.WHITE,
        overlay_color=ft.Colors.BLUE_900,
    )

    image_area = ft.Image(src=None, width=800, height=600, fit=ft.ImageFit.CONTAIN)
    match_area = ft.Image(src=None, width=400, height=600, fit=ft.ImageFit.CONTAIN)
    status_text = ft.Text(size=20, weight="bold")
    exit_button = ft.ElevatedButton(
        text="Выход",
        icon=ft.Icons.EXIT_TO_APP,
        style=button_style,
        width=150,
        height=60,
        visible=False,
        on_click=lambda e: exit_mode(page)
    )

    webcam_button = ft.ElevatedButton(
        text="Камера",
        icon=ft.Icons.CAMERA,
        style=button_style,
        width=200,
        height=80,
        on_click=lambda e: start_webcam(page, image_area, match_area, status_text, exit_button),
    )

    image_button = ft.ElevatedButton(
        text="Изображение",
        icon=ft.Icons.IMAGE,
        style=button_style,
        width=200,
        height=80,
        on_click=lambda e: pick_image(page, image_area, match_area, status_text, exit_button),
    )

    video_button = ft.ElevatedButton(
        text="Видео",
        icon=ft.Icons.VIDEO_FILE,
        style=button_style,
        width=200,
        height=80,
        on_click=lambda e: pick_video(page, image_area, match_area, status_text, exit_button),
    )

    page.add(
        ft.Row(
            [
                ft.Column([
                    ft.Text("Выберите метод:", size=24, weight="bold"),
                    ft.Container(webcam_button),
                    ft.Container(image_button),
                    ft.Container(video_button),
                    ft.Container(exit_button),
                    ft.Container(status_text),
                    ft.Container(match_area)
                ]),
                ft.Container(image_area,)
            ]
        )
    )

if __name__ == "__main__":
    try:
        if not load_face_descriptors():
            print("Не удалось загрузить базу лиц. Создайте папку 'face_db' и добавьте изображения.")
        
        ft.app(target=start_interface)
    except Exception as e:
        print(f"Ошибка запуска приложения: {e}")
        control_relay(False)
        exit(1)