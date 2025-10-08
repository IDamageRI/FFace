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
from queue import Queue
import logging

# Настройка логирования
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

# Пути к моделям
shape_predictor_path = 'face_model/shape_predictor_68_face_landmarks.dat'
face_rec_model_path = 'face_model/dlib_face_recognition_resnet_model_v1.dat'
base_path = 'face_bd'
log_file = 'detection_log.txt'

# Глобальные переменные
is_running = False
is_rtsp_running = False
cap = None
face_descriptors = []
faces = []
file_picker = ft.FilePicker()
arduino_serial = None
rtsp_frame_queue = Queue(maxsize=1)
rtsp_url = "rtsp://admin:admin123@192.168.0.2:554/cam/realmonitor?channel=1&subtype=0"

# Инициализация детектора и моделей
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(shape_predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

def connect_to_arduino():
    """Поиск и подключение к Arduino"""
    global arduino_serial
    ports = serial.tools.list_ports.comports()
    
    for port in ports:
        try:
            arduino_serial = serial.Serial(port.device, 9600, timeout=1)
            time.sleep(2)
            logging.info(f"Успешно подключено к Arduino на {port.device}")
            return arduino_serial
        except (serial.SerialException, serial.SerialTimeoutException) as e:
            logging.error(f"Ошибка подключения к {port.device}: {e}")
            continue
    
    logging.warning("Arduino не найдена. Проверьте подключение.")
    return None

def send_to_arduino(command):
    """Отправка команды на Arduino"""
    global arduino_serial
    if arduino_serial and arduino_serial.is_open:
        try:
            arduino_serial.write(command.encode())
            logging.info(f"Отправлено на Arduino: {command}")
        except serial.SerialException as e:
            logging.error(f"Ошибка отправки на Arduino: {e}")

def putText_rus(img, text, pos, color=(0, 255, 0), font_size=20):
    """Добавление русского текста на изображение"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    draw.text(pos, text, font=font, fill=color[::-1])
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def log_detection(name):
    """Логирование обнаруженных лиц"""
    with open(log_file, 'a', encoding='utf-8') as log:
        log.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Распознан: {os.path.splitext(name)[0]}\n")

def load_face_descriptors():
    """Загрузка базы лиц и их дескрипторов"""
    global face_descriptors, faces
    face_descriptors = []
    faces = os.listdir(base_path)

    if not faces:
        raise ValueError(f"В папке {base_path} нет сохранённых лиц")

    for face in faces:
        img_path = os.path.join(base_path, face)
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            logging.error(f"Ошибка загрузки изображения {img_path}")
            continue

        dets = detector(img, 1)
        if len(dets) == 0:
            logging.warning(f"Лицо не найдено на изображении {img_path}")
            continue

        for d in dets:
            shape = sp(img, d)
            face_descriptors.append(facerec.compute_face_descriptor(img, shape))

    logging.info(f"Загружено {len(face_descriptors)} лиц.")

def compare_faces(frame, threshold=0.5):
    """Сравнение лиц и поиск совпадений"""
    dets = detector(frame, 0)

    if len(dets) == 0:
        return None, None, False

    for d in dets:
        shape = sp(frame, d)
        main_descriptor = facerec.compute_face_descriptor(frame, shape)
        distances = [distance.euclidean(main_descriptor, fd) for fd in face_descriptors]

        min_dist = min(distances)
        closest_face_idx = distances.index(min_dist)
        is_match = min_dist <= threshold

        return min_dist, faces[closest_face_idx], is_match

    return None, None, False

def resize_image(image, max_width=800, max_height=600):
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
    global is_running, is_rtsp_running, cap
    is_running = False
    is_rtsp_running = False
    if cap is not None:
        cap.release()
        cap = None

    for control in page.controls[:]:
        page.controls.remove(control)

    start_interface(page)
    page.update()

def list_available_cameras(max_tested=5):
    """Проверяет доступные камеры и возвращает список работающих"""
    available_cameras = []
    for i in range(max_tested):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

def rtsp_capture_thread():
    """Поток для захвата RTSP-потока"""
    global is_rtsp_running, rtsp_frame_queue
    
    # Настройки для стабильного подключения
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;65536"
    
    while is_rtsp_running:
        try:
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 30000)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
            
            if not cap.isOpened():
                logging.error("RTSP: Ошибка подключения. Повтор через 5 сек...")
                time.sleep(5)
                continue

            logging.info("RTSP: Подключение установлено")
            
            while is_rtsp_running:
                ret, frame = cap.read()
                if not ret:
                    logging.warning("RTSP: Потеря кадра. Переподключение...")
                    break
                    
                frame = resize_image(frame)
                
                # Очистка очереди
                while not rtsp_frame_queue.empty():
                    try:
                        rtsp_frame_queue.get_nowait()
                    except:
                        pass
                
                rtsp_frame_queue.put(frame)
                
        except Exception as e:
            logging.error(f"RTSP: Ошибка - {str(e)}")
            time.sleep(3)
        finally:
            if 'cap' in locals():
                cap.release()

def start_rtsp_camera(page, image_area, match_area, status_text, exit_button):
    """Запуск IP-камеры по RTSP с ограничением частоты распознавания"""
    global is_rtsp_running
    
    load_face_descriptors()
    is_rtsp_running = True
    exit_button.visible = True
    page.update()
    
    rtsp_thread = threading.Thread(target=rtsp_capture_thread, daemon=True)
    rtsp_thread.start()
    
    last_detected = None
    arduino_triggered = False
    last_detection_time = 0  # Время последнего распознавания
    last_frame_time = time.time()
    
    while is_rtsp_running:
        try:
            if not rtsp_frame_queue.empty():
                frame = rtsp_frame_queue.get()
                
                # Ограничение FPS для интерфейса (~30 кадров/сек)
                if time.time() - last_frame_time < 0.033:
                    continue
                
                last_frame_time = time.time()
                current_time = time.time()
                
                # Распознавание только если прошло более 3 секунд с последнего
                if current_time - last_detection_time >= 3:
                    # Распознавание лиц
                    min_dist, closest_face, is_match = compare_faces(frame)
                    last_detection_time = current_time  # Обновляем время последнего распознавания
                else:
                    # Используем предыдущие результаты
                    is_match = last_detected is not None
                    closest_face = last_detected if is_match else None
                
                if is_match:
                    face_name = os.path.splitext(closest_face)[0]
                    
                    # Всегда обновляем левую панель при обнаружении
                    match_img_path = os.path.join(base_path, closest_face)
                    match_img = cv2.imdecode(np.fromfile(match_img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                    match_img = resize_image(match_img)
                    match_area.src_base64 = image_to_base64(match_img)
                    
                    if last_detected != closest_face:
                        if arduino_triggered:
                            send_to_arduino('0')
                            time.sleep(0.025)
                        
                        send_to_arduino('1')
                        arduino_triggered = True
                        log_detection(closest_face)
                        
                        status_text.value = f"Найдено: {face_name}"
                        status_text.color = ft.Colors.GREEN
                    
                    last_detected = closest_face
                    frame = putText_rus(frame, f"Найдено: {face_name}", (10, 30), (0, 255, 0), 20)
                else:
                    if last_detected:
                        status_text.value = "Лицо не найдено"
                        status_text.color = ft.Colors.RED
                        match_area.src_base64 = None
                        
                        if arduino_triggered:
                            send_to_arduino('0')
                            arduino_triggered = False                                
                    
                    frame = putText_rus(frame, "Лицо не найдено", (10, 30), (0, 0, 255), 20)                                                                                                                                                                                                            
                
                image_area.src_base64 = image_to_base64(frame)
                page.update()
            
            time.sleep(0.01)
            
        except Exception as e:
            logging.error(f"Ошибка в основном цикле RTSP: {str(e)}")
            time.sleep(1)
    
    if arduino_triggered:
        send_to_arduino('0')

def start_webcam(page, image_area, match_area, status_text, exit_button, camera_index=0):
    """Запуск веб-камеры"""
    global is_running, cap
    
    load_face_descriptors()
    is_running = True
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        status_text.value = f"Ошибка: не удалось открыть камеру {camera_index}"
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
    last_frame_time = time.time()

    while is_running:
        ret, frame = cap.read()
        if not ret:
            status_text.value = "Ошибка чтения кадра с камеры"
            status_text.color = ft.Colors.RED
            page.update()
            break

        # Ограничение FPS для интерфейса
        if time.time() - last_frame_time < 0.033:
            continue
        
        last_frame_time = time.time()
        frame = resize_image(frame)
        
        min_dist, closest_face, is_match = compare_faces(frame)
        current_time = time.time()

        if is_match:
            face_name = os.path.splitext(closest_face)[0]
            
            if last_detected != closest_face:
                if arduino_triggered:
                    send_to_arduino('0')
                    arduino_triggered = False
                    time.sleep(0.025)
                
                send_to_arduino('1')
                arduino_triggered = True
                
                log_detection(closest_face)
                status_text.value = f"Найдено: {face_name}"
                status_text.color = ft.Colors.GREEN

                match_img_path = os.path.join(base_path, closest_face)
                match_img = cv2.imdecode(np.fromfile(match_img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                match_img = resize_image(match_img)
                match_area.src_base64 = image_to_base64(match_img)

            last_detected = closest_face
            last_detection_time = current_time
            
        else:
            if current_time - last_detection_time > 5 or last_detected:
                last_detected = None
                status_text.value = "Лицо не найдено"
                status_text.color = ft.Colors.RED
                match_area.src_base64 = None
                frame = putText_rus(frame, "Лицо не найдено", (10, 30), (0, 0, 255), 20)
                
                if arduino_triggered:
                    send_to_arduino('0')
                    arduino_triggered = False

        image_area.src_base64 = image_to_base64(frame)
        page.update()

    if cap is not None:
        cap.release()
    
    if arduino_triggered:
        send_to_arduino('0')

def process_selected_image(e, page, image_area, match_area, status_text, exit_button):
    """Обработка выбранного изображения"""
    if e.files:
        send_to_arduino('0')
        time.sleep(0.1)
        
        image_path = e.files[0].path
        img = cv2.imread(image_path)

        if img is None:
            status_text.value = "Ошибка загрузки изображения"
            status_text.color = ft.Colors.RED
            page.update()
            return

        load_face_descriptors()
        img = resize_image(img)
        min_dist, closest_face, is_match = compare_faces(img)

        if is_match:
            face_name = os.path.splitext(closest_face)[0]
            status_text.value = f"Совпадение: {face_name}"
            status_text.color = ft.Colors.GREEN
            log_detection(closest_face)
            send_to_arduino('1')

            match_img_path = os.path.join(base_path, closest_face)
            match_img = cv2.imdecode(np.fromfile(match_img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
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
        arduino_triggered = False

        load_face_descriptors()

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
                        if arduino_triggered:
                            send_to_arduino('0')
                            arduino_triggered = False
                            time.sleep(0.1)
                        
                        send_to_arduino('1')
                        arduino_triggered = True
                        
                        log_detection(closest_face)
                        status_text.value = f"Найдено: {face_name}"
                        status_text.color = ft.Colors.GREEN

                        match_img_path = os.path.join(base_path, closest_face)
                        match_img = cv2.imdecode(np.fromfile(match_img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                        match_img = resize_image(match_img)
                        match_area.src_base64 = image_to_base64(match_img)

                    last_detected = closest_face
                    frame = putText_rus(frame, f"Найдено: {face_name}", (10, 30), (0, 255, 0), 20)
                else:
                    if last_detected:
                        status_text.value = "Лицо не найдено"
                        status_text.color = ft.Colors.RED
                        match_area.src_base64 = None
                        
                        if arduino_triggered:
                            send_to_arduino('0')
                            arduino_triggered = False
                            
                    frame = putText_rus(frame, "Лицо не найдено", (10, 30), (0, 0, 255), 20)

            image_area.src_base64 = image_to_base64(frame)
            page.update()

        if cap is not None:
            cap.release()
            
        if arduino_triggered:
            send_to_arduino('0')

def pick_video(page, image_area, match_area, status_text, exit_button):
    """Выбор видео для анализа"""
    file_picker.on_result = lambda e: process_selected_video(e, page, image_area, match_area, status_text, exit_button)
    file_picker.pick_files("Выберите видео", allowed_extensions=["mp4", "avi", "mov"])
    exit_button.visible = True
    page.update()

def start_interface(page: ft.Page):
    """Основной интерфейс приложения"""
    page.title = "Система распознавания лиц"
    page.window_maximized = True
    page.window_maximizable = False
    page.window_resizable = False
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.padding = 20
    page.bgcolor = "#111524"
    
    log_entries = ft.ListView(expand=True, spacing=5, auto_scroll=True)
    log_container = ft.Container(
        content=log_entries,
        border=ft.border.all(1, ft.Colors.GREY_800),
        border_radius=10,
        padding=10,
        width=400,
        height=750,
        bgcolor="#11158",
    )
    
    def add_log_entry(message):
        log_entries.controls.append(ft.Text(message, size=12, color=ft.Colors.WHITE, selectable=True))
        page.update()
    
    def update_logs():
        last_position = 0
        while True:
            try:
                with open('detection_log.txt', 'r', encoding='utf-8') as f:
                    f.seek(0, 2)
                    file_size = f.tell()
                    
                    if file_size < last_position:
                        last_position = 0
                        log_entries.controls.clear()
                        page.update()
                    
                    if file_size > last_position:
                        f.seek(last_position)
                        new_lines = f.readlines()
                        last_position = f.tell()
                        
                        if new_lines:
                            for line in new_lines:
                                line = line.strip()
                                if line:
                                    add_log_entry(line)
            except Exception as e:
                logging.error(f"Ошибка чтения лога: {e}")
            
            time.sleep(1)

    threading.Thread(target=update_logs, daemon=True).start()

    page.overlay.append(file_picker)

    button_style = ft.ButtonStyle(
        padding=20,
        bgcolor=ft.Colors.BLUE_700,
        color=ft.Colors.WHITE,
        overlay_color=ft.Colors.BLUE_900,
    )

    image_area = ft.Image(src=f'None', width=1200, height=900, fit=ft.ImageFit.CONTAIN)
    match_area = ft.Image(src=f'None', width=350, height=300)
    status_text = ft.Text(size=20, weight="bold", width=300)
    
    camera_dropdown = ft.Dropdown(
        options=[],
        label="Выберите камеру",
        width=200,
        visible=False,
    )
     
    def update_camera_list():
        cameras = list_available_cameras()
        options = []
        for cam_idx in cameras:
            options.append(ft.dropdown.Option(text=f"Камера {cam_idx}", key=str(cam_idx)))
        camera_dropdown.options = options
        if cameras:
            camera_dropdown.value = str(cameras[-1])
        camera_dropdown.visible = True
        page.update()
     
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
        on_click=lambda e: [
            update_camera_list(),
            ft.Text("Выберите камеру из списка и нажмите 'Запустить'", color="white")
        ],
    )
    
    start_button = ft.ElevatedButton(
        text="Запустить",
        icon=ft.Icons.PLAY_ARROW,
        style=button_style,
        width=200,
        height=80,
        visible=False,
        on_click=lambda e: start_webcam(
            page=page,
            image_area=image_area,
            match_area=match_area,
            status_text=status_text,
            exit_button=exit_button,
            camera_index=int(camera_dropdown.value) if camera_dropdown.value else 0
        ),
    )
    
    rtsp_button = ft.ElevatedButton(
        text="IP-камера (RTSP)",
        icon=ft.Icons.SETTINGS_ETHERNET,
        style=button_style,
        width=200,
        height=80,
        on_click=lambda e: start_rtsp_camera(
            page=page,
            image_area=image_area,
            match_area=match_area,
            status_text=status_text,
            exit_button=exit_button
        ),
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

    def on_camera_selected(e):
        start_button.visible = bool(camera_dropdown.value)
        page.update()
    
    camera_dropdown.on_change = on_camera_selected

    page.add(
        ft.Row(
            [
                ft.Column(
                    [
                        ft.Text("Выберите метод:", size=24, weight="bold", color="white"),
                        webcam_button,
                        camera_dropdown,
                        start_button,
                        rtsp_button,
                        image_button,
                        video_button,
                        exit_button,
                        status_text,
                        match_area
                    ],
                    alignment=ft.MainAxisAlignment.START,
                    spacing=10,
                ),
                
                ft.Column(
                    [image_area],
                    alignment=ft.MainAxisAlignment.CENTER,
                    expand=True
                ),
                
                ft.Column(
                    [
                        ft.Text("Журнал событий", size=24, weight="bold", color=ft.Colors.WHITE),
                        log_container
                    ],
                    width=400,
                    alignment=ft.MainAxisAlignment.START,
                )
            ],
            spacing=20,
            expand=True,
        )
    )

if __name__ == "__main__":
    try:
        # Инициализация моделей
        load_face_descriptors()
        
        # Подключение Arduino
        arduino_serial = connect_to_arduino()
        if arduino_serial:
            print(f"Успешно подключено к Arduino на {arduino_serial.port}")
        else:
            print("Arduino не найдена. Проверьте подключение.")

        # Запуск интерфейса
        ft.app(target=start_interface)
        
        # Очистка ресурсов
        if arduino_serial and arduino_serial.is_open:
            arduino_serial.close()
    except Exception as e:
        print(f"Ошибка запуска приложения: {e}")
        if arduino_serial and arduino_serial.is_open:
            arduino_serial.close()
        exit(1)

        