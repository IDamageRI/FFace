import base64
import os
from datetime import datetime
import cv2
import flet as ft
from scipy.spatial import distance
import dlib
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import serial
import serial.tools.list_ports
import time
import threading

# ===== КОНСТАНТЫ И ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ =====
shape_predictor_path = 'face_model/shape_predictor_68_face_landmarks.dat'
face_rec_model_path = 'face_model/dlib_face_recognition_resnet_model_v1.dat'
smile_cascade_path = 'haarcascade_smile.xml'  # Haar каскад для улыбки
base_path = 'face_bd'
neutral_faces_path = 'neutral'  # Папка с нейтральными лицами
smiling_faces_path = 'smiling'  # Папка с улыбающимися лицами
log_file = 'detection_log.txt'

is_running = False
cap = None
face_descriptors = []
faces = []
file_picker = ft.FilePicker()
arduino_serial = None
smile_cascade = None

# ===== ИНИЦИАЛИЗАЦИЯ МОДЕЛЕЙ =====
def initialize_models():
    """Инициализация всех моделей"""
    global sp, facerec, detector, smile_cascade
    
    # Модели для распознавания лиц
    sp = dlib.shape_predictor(shape_predictor_path)
    facerec = dlib.face_recognition_model_v1(face_rec_model_path)
    detector = dlib.get_frontal_face_detector()
    
    # Haar каскад для детекции улыбки
    try:
        if not os.path.exists(smile_cascade_path):
            download_haar_cascade()
        smile_cascade = cv2.CascadeClassifier(smile_cascade_path)
        print("Каскад улыбки загружен успешно")
    except Exception as e:
        print(f"Ошибка загрузки каскада улыбки: {e}")
        smile_cascade = None

def download_haar_cascade():
    """Скачивание Haar каскада если отсутствует"""
    import urllib.request
    print("Скачивание Haar каскада для улыбки...")
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_smile.xml"
    try:
        urllib.request.urlretrieve(url, smile_cascade_path)
        print("Каскад успешно скачан")
    except Exception as e:
        print(f"Ошибка скачивания каскада: {e}")

# ===== ARDUINO ФУНКЦИИ =====
def connect_to_arduino():
    """Подключение к Arduino"""
    global arduino_serial
    ports = serial.tools.list_ports.comports()
    
    for port in ports:
        try:
            arduino_serial = serial.Serial(port.device, 9600, timeout=1)
            time.sleep(2)
            print(f"Успешно подключено к Arduino на {port.device}")
            return arduino_serial
        except Exception as e:
            print(f"Ошибка подключения к {port.device}: {e}")
    
    print("Arduino не найдена. Режим эмуляции.")
    return None

def send_to_arduino(command):
    """Отправка команды на Arduino"""
    global arduino_serial
    if arduino_serial and arduino_serial.is_open:
        try:
            arduino_serial.write(command.encode())
            print(f"Отправлено на Arduino: {command}")
        except Exception as e:
            print(f"Ошибка отправки на Arduino: {e}")

# ===== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =====
def putText_rus(img, text, pos, color=(0, 255, 0), font_size=20):
    """Отображение русского текста на изображении"""
    try:
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        draw.text(pos, text, font=font, fill=color[::-1])
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except:
        return img

def log_detection(name, status, smile_detected=False):
    """Логирование событий"""
    try:
        with open(log_file, 'a', encoding='utf-8') as log:
            smile_status = " с улыбкой" if smile_detected else ""
            log.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {status}: {name}{smile_status}\n")
    except:
        pass

def resize_image(image, max_width=800, max_height=600):
    """Изменение размера изображения"""
    h, w = image.shape[:2]
    if w > max_width or h > max_height:
        scale = min(max_width / w, max_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h))
    return image

def image_to_base64(image):
    """Конвертация изображения в base64"""
    try:
        _, encoded_img = cv2.imencode('.png', image)
        return base64.b64encode(encoded_img).decode('utf-8')
    except:
        return ""

def list_available_cameras(max_tested=5):
    """Проверяет доступные камеры"""
    available_cameras = []
    for i in range(max_tested):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

# ===== ФУНКЦИИ РАСПОЗНАВАНИЯ ЛИЦ =====
def load_face_descriptors():
    """Загрузка дескрипторов лиц из базы"""
    global face_descriptors, faces
    
    face_descriptors = []
    faces = []
    
    # Создаем папки если их нет
    os.makedirs(neutral_faces_path, exist_ok=True)
    os.makedirs(smiling_faces_path, exist_ok=True)
    
    # Загрузка нейтральных лиц
    neutral_faces = [f for f in os.listdir(neutral_faces_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    for face in neutral_faces:
        img_path = os.path.join(neutral_faces_path, face)
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        
        if img is None:
            continue
            
        dets = detector(img, 1)
        if len(dets) == 0:
            continue
            
        for d in dets:
            shape = sp(img, d)
            face_descriptor = facerec.compute_face_descriptor(img, shape)
            face_descriptors.append(face_descriptor)
            faces.append(face)
    
    print(f"Загружено {len(face_descriptors)} нейтральных лиц")

def compare_faces(frame, threshold=0.5):
    """Сравнение лиц с базой"""
    dets = detector(frame, 0)
    if len(dets) == 0:
        return None, None, False

    for d in dets:
        shape = sp(frame, d)
        main_descriptor = facerec.compute_face_descriptor(frame, shape)
        
        if not face_descriptors:
            return None, None, False
            
        distances = [distance.euclidean(main_descriptor, fd) for fd in face_descriptors]
        min_dist = min(distances)
        closest_face_idx = distances.index(min_dist)
        is_match = min_dist <= threshold
        
        return min_dist, faces[closest_face_idx], is_match
    
    return None, None, False

# ===== ДЕТЕКЦИЯ УЛЫБКИ =====
def detect_smile(face_roi):
    """Обнаружение улыбки на области лица с использованием Haar каскадов"""
    if smile_cascade is None:
        return False, 0.0
    
    try:
        # Конвертируем в grayscale
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Детекция улыбки
        smiles = smile_cascade.detectMultiScale(
            gray,
            scaleFactor=1.8,
            minNeighbors=20,
            minSize=(25, 25)
        )
        
        # Если найдены улыбки, возвращаем True
        smile_detected = len(smiles) > 0
        confidence = len(smiles) / 10.0  # Простая уверенность на основе количества обнаружений
        
        return smile_detected, min(confidence, 1.0)
        
    except Exception as e:
        print(f"Ошибка детекции улыбки: {e}")
        return False, 0.0

# ===== ДВУХФАКТОРНАЯ АУТЕНТИФИКАЦИЯ =====
def two_factor_authentication(identity, frame, face_location):
    """Проверка второго фактора - улыбки"""
    top, right, bottom, left = face_location
    face_roi = frame[top:bottom, left:right]
    
    # Детекция улыбки
    smile_detected, smile_confidence = detect_smile(face_roi)
    
    # Проверка наличия улыбающегося изображения в базе
    smiling_img_path = os.path.join(smiling_faces_path, identity)
    has_smiling_reference = os.path.exists(smiling_img_path)
    
    # Если есть эталонная улыбка, можно добавить дополнительную проверку
    if has_smiling_reference and smile_detected:
        # Можно добавить сравнение с эталоном здесь
        pass
    
    return smile_detected, smile_confidence

# ===== ОСНОВНЫЕ ФУНКЦИИ РЕЖИМОВ =====
def start_webcam(page, image_area, match_area, status_text, exit_button, camera_index=0):
    """Запуск веб-камеры с двухфакторной аутентификацией"""
    global is_running, cap
    is_running = True
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        status_text.value = f"Ошибка открытия камеры {camera_index}"
        status_text.color = ft.Colors.RED
        page.update()
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    exit_button.visible = True
    page.update()

    last_detected = None
    last_detection_time = 0
    arduino_triggered = False
    auth_stage = "face_detection"  # Стадии: face_detection, smile_verification
    current_identity = None

    while is_running:
        ret, frame = cap.read()
        if not ret:
            status_text.value = "Ошибка чтения кадра"
            status_text.color = ft.Colors.RED
            page.update()
            break

        frame = resize_image(frame)
        display_frame = frame.copy()
        
        # Детекция лиц
        dets = detector(frame, 0)
        
        if dets:
            for d in dets:
                top, right, bottom, left = d.top(), d.right(), d.bottom(), d.left()
                
                # Рисуем рамку вокруг лица
                color = (0, 255, 0) if auth_stage == "smile_verification" else (255, 255, 0)
                cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                
                if auth_stage == "face_detection":
                    # Первый фактор: распознавание лица
                    min_dist, closest_face, is_match = compare_faces(frame)
                    
                    if is_match:
                        current_identity = os.path.splitext(closest_face)[0]
                        status_text.value = f"Распознан: {current_identity}. Улыбнитесь!"
                        status_text.color = ft.Colors.BLUE
                        
                        # Переходим к проверке улыбки
                        auth_stage = "smile_verification"
                        last_detection_time = time.time()
                        
                        # Показываем нейтральное фото
                        match_img_path = os.path.join(neutral_faces_path, closest_face)
                        if os.path.exists(match_img_path):
                            match_img = cv2.imdecode(np.fromfile(match_img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                            match_img = resize_image(match_img)
                            match_area.src_base64 = image_to_base64(match_img)
                
                elif auth_stage == "smile_verification" and current_identity:
                    # Второй фактор: проверка улыбки
                    smile_detected, smile_conf = two_factor_authentication(
                        closest_face, frame, (top, right, bottom, left)
                    )
                    
                    if smile_detected:
                        # Успешная двухфакторная аутентификация
                        status_text.value = f"Доступ разрешен: {current_identity}"
                        status_text.color = ft.Colors.GREEN
                        log_detection(current_identity, "Успешная аутентификация", True)
                        
                        if not arduino_triggered:
                            send_to_arduino('1')
                            arduino_triggered = True
                        
                        # Показываем улыбающееся фото если есть
                        smile_img_path = os.path.join(smiling_faces_path, closest_face)
                        if os.path.exists(smile_img_path):
                            smile_img = cv2.imdecode(np.fromfile(smile_img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                            smile_img = resize_image(smile_img)
                            match_area.src_base64 = image_to_base64(smile_img)
                        
                        display_frame = putText_rus(display_frame, f"Улыбка подтверждена!", 
                                                  (left, top - 30), (0, 255, 0), 20)
                    
                    # Таймаут проверки улыбки (10 секунд)
                    elif time.time() - last_detection_time > 10:
                        auth_stage = "face_detection"
                        status_text.value = "Таймаут. Распознайте лицо снова"
                        status_text.color = ft.Colors.ORANGE
                        current_identity = None
                        
                        if arduino_triggered:
                            send_to_arduino('0')
                            arduino_triggered = False
        
        else:
            # Лицо не найдено
            if auth_stage != "face_detection":
                auth_stage = "face_detection"
                status_text.value = "Лицо не найдено"
                status_text.color = ft.Colors.RED
                current_identity = None
            
            if arduino_triggered:
                send_to_arduino('0')
                arduino_triggered = False

        # Отображаем текущую стадию аутентификации
        stage_text = "Поиск лица" if auth_stage == "face_detection" else "Улыбнитесь!"
        display_frame = putText_rus(display_frame, stage_text, (10, 30), 
                                  (0, 255, 0) if auth_stage == "smile_verification" else (255, 255, 0), 20)

        image_area.src_base64 = image_to_base64(display_frame)
        page.update()
        
        # Небольшая задержка для снижения нагрузки на CPU
        time.sleep(0.03)

    if cap:
        cap.release()
    if arduino_triggered:
        send_to_arduino('0')

def exit_mode(page):
    """Выход из режима"""
    global is_running, cap
    is_running = False
    if cap:
        cap.release()
        cap = None
    
    # Очистка и перезагрузка интерфейса
    page.clean()
    start_interface(page)
    page.update()

# ===== ИНТЕРФЕЙС =====
def start_interface(page: ft.Page):
    """Основной интерфейс приложения"""
    page.title = "Система двухфакторной аутентификации по лицу"
    page.window.maximized = True
    page.window.maximizable = False
    page.window.resizable = False
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.padding = 20
    page.bgcolor = "#111524"
    
    # Инициализация моделей
    initialize_models()
    load_face_descriptors()
    
    # Подключение Arduino
    global arduino_serial
    arduino_serial = connect_to_arduino()
    
    # Создание элементов интерфейса
    image_area = ft.Image(src=f"None", width=1000, height=700, fit=ft.ImageFit.CONTAIN)
    match_area = ft.Image(src=f"None", width=300, height=400)
    status_text = ft.Text(size=18, weight="bold", width=280, color="white")
    
    # Кнопки и элементы управления
    exit_button = ft.ElevatedButton(
        text="Выход", 
        icon=ft.Icons.EXIT_TO_APP, 
        width=150, 
        height=50, 
        visible=False,
        style=ft.ButtonStyle(bgcolor=ft.Colors.RED, color=ft.Colors.WHITE),
        on_click=lambda e: exit_mode(page)
    )
    
    webcam_button = ft.ElevatedButton(
        text="Запуск аутентификации", 
        icon=ft.Icons.CAMERA, 
        width=250, 
        height=60,
        style=ft.ButtonStyle(bgcolor=ft.Colors.BLUE, color=ft.Colors.WHITE),
        on_click=lambda e: start_webcam(page, image_area, match_area, status_text, exit_button)
    )
    
    # Информационная панель
    info_text = ft.Text(
        "Система двухфакторной аутентификации:\n"
        "1. Распознавание лица\n"
        "2. Подтверждение улыбкой",
        size=14,
        color=ft.Colors.WHITE70,
        text_align=ft.TextAlign.CENTER
    )
    
    # Компоновка интерфейса
    page.add(
        ft.Row([
            ft.Column([
                ft.Text("🔐 Двухфакторная аутентификация", 
                       size=24, weight="bold", color="white"),
                ft.Divider(height=20),
                info_text,
                ft.Divider(height=30),
                webcam_button,
                ft.Divider(height=20),
                status_text,
                ft.Divider(height=20),
                ft.Text("Эталонное фото:", size=16, color="white"),
                match_area,
                exit_button
            ], 
            width=350,
            alignment=ft.MainAxisAlignment.START, 
            spacing=15),
            
            ft.VerticalDivider(width=20),
            
            ft.Column([
                ft.Text("Видео с камеры", size=18, color="white"),
                image_area
            ], 
            alignment=ft.MainAxisAlignment.CENTER, 
            expand=True)
        ], 
        spacing=20, 
        expand=True)
    )

# ===== ЗАПУСК ПРИЛОЖЕНИЯ =====
if __name__ == "__main__":
    try:
        ft.app(target=start_interface)
    except Exception as e:
        print(f"Ошибка запуска приложения: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if arduino_serial and arduino_serial.is_open:
            arduino_serial.close()