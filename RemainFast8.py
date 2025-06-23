import base64
import os
import time
import cv2
import dlib
import numpy as np
import flet as ft
import serial
import serial.tools.list_ports
from datetime import datetime
from scipy.spatial import distance
from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import ThreadPoolExecutor

# Конфигурация
TARGET_FPS = 20
RESIZE_FACTOR = 0.5
ARDUINO_PORT = 'COM4'
ARDUINO_BAUDRATE = 9600

# Пути к моделям
shape_predictor_path = 'face_model/shape_predictor_68_face_landmarks.dat'
face_rec_model_path = 'face_model/dlib_face_recognition_resnet_model_v1.dat'
base_path = 'face_bd'
log_file = 'detection_log.txt'

# Инициализация моделей dlib
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(shape_predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

class FaceRecognitionApp:
    def __init__(self):
        self.is_running = False
        self.cap = None
        self.face_descriptors = []
        self.faces = []
        self.arduino_serial = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.last_match = None
        self.file_picker = ft.FilePicker()

    def putText_rus(self, img, text, pos, color=(0, 255, 0), font_size=20):
        """Отображение русского текста на изображении"""
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        draw.text(pos, text, font=font, fill=color[::-1])
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def connect_to_arduino(self):
        """Подключение к Arduino"""
        try:
            self.arduino_serial = serial.Serial(ARDUINO_PORT, ARDUINO_BAUDRATE, timeout=1)
            time.sleep(2)
            print(f"Connected to Arduino on {ARDUINO_PORT}")
            return True
        except serial.SerialException as e:
            print(f"Arduino connection error: {e}")
            return False

    def send_to_arduino(self, command):
        """Отправка команды на Arduino"""
        if self.arduino_serial and self.arduino_serial.is_open:
            try:
                self.arduino_serial.write(command.encode())
                print(f"Sent to Arduino: {command}")
            except serial.SerialException as e:
                print(f"Arduino send error: {e}")

    def load_face_descriptors(self):
        """Загрузка базы лиц"""
        self.face_descriptors = []
        self.faces = [f for f in os.listdir(base_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        for face in self.faces:
            img = cv2.imread(os.path.join(base_path, face))
            if img is not None:
                dets = detector(img, 1)
                if dets:
                    shape = sp(img, dets[0])
                    self.face_descriptors.append(facerec.compute_face_descriptor(img, shape))
        
        print(f"Loaded {len(self.face_descriptors)} faces")

    def process_frame(self, frame, threshold=0.4):
        """Обработка кадра"""
        small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
        dets = detector(small_frame, 0)
        
        if not dets:
            return None, None, False
        
        shape = sp(small_frame, dets[0])
        descriptor = facerec.compute_face_descriptor(small_frame, shape)
        distances = [distance.euclidean(descriptor, fd) for fd in self.face_descriptors]
        
        min_dist = min(distances)
        closest_face = self.faces[distances.index(min_dist)]
        is_match = min_dist <= threshold
        
        return min_dist, closest_face, is_match

    async def webcam_loop(self, page, image_area, match_area, status_text, exit_button):
        """Цикл обработки веб-камеры"""
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        
        frame_time = 1 / TARGET_FPS
        
        while self.is_running:
            start_time = time.time()
            
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Асинхронная обработка кадра
            future = self.executor.submit(self.process_frame, frame)
            min_dist, closest_face, is_match = future.result()
            
            # Обновление интерфейса
            if is_match:
                face_name = os.path.splitext(closest_face)[0]
                
                if self.last_match != closest_face:
                    if self.last_match is not None:
                        self.send_to_arduino('0')
                        await asyncio.sleep(0.025)
                    
                    self.send_to_arduino('1')
                    self.last_match = closest_face
                    
                    status_text.value = f"Найдено: {face_name}"
                    status_text.color = ft.Colors.GREEN
                    
                    match_img = cv2.imread(os.path.join(base_path, closest_face))
                    match_area.src_base64 = self.image_to_base64(cv2.resize(match_img, (400, 400)))
                    
                    frame = self.putText_rus(frame, f"Найдено: {face_name}", (10, 30), (0, 255, 0))
            else:
                if self.last_match:
                    self.send_to_arduino('0')
                    self.last_match = None
                    
                    status_text.value = "Лицо не найдено"
                    status_text.color = ft.Colors.RED
                    match_area.src_base64 = None
                    
                    frame = self.putText_rus(frame, "Лицо не найдено", (10, 30), (0, 0, 255))
            
            image_area.src_base64 = self.image_to_base64(cv2.resize(frame, (800, 600)))
            
            # Поддержание FPS
            processing_time = time.time() - start_time
            await asyncio.sleep(max(0, frame_time - processing_time))
            
            await page.update_async()
        
        self.cap.release()
        if self.last_match:
            self.send_to_arduino('0')

    def image_to_base64(self, image):
        """Конвертация изображения в base64"""
        _, encoded_img = cv2.imencode('.png', image)
        return base64.b64encode(encoded_img).decode('utf-8')

    def start_interface(self, page: ft.Page):
        """Инициализация интерфейса"""
        page.title = "Face Recognition"
        page.window_width = 1400
        page.window_height = 800
        page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
        page.vertical_alignment = ft.MainAxisAlignment.CENTER
        
        # Элементы интерфейса
        image_area = ft.Image(width=800, height=600, fit=ft.ImageFit.CONTAIN)
        match_area = ft.Image(width=400, height=600, fit=ft.ImageFit.CONTAIN)
        status_text = ft.Text(size=20, weight="bold")
        exit_button = ft.ElevatedButton(
            text="Выход",
            visible=False,
            on_click=lambda e: self.exit_mode(page)
        )

        async def start_webcam(e):
            self.is_running = True
            exit_button.visible = True
            await self.webcam_loop(page, image_area, match_area, status_text, exit_button)
        
        webcam_button = ft.ElevatedButton(
            text="Камера",
            on_click=start_webcam
        )

        page.add(
            ft.Row([
                ft.Column([
                    webcam_button,
                    exit_button,
                    status_text,
                    match_area
                ]),
                image_area
            ])
        )

    def exit_mode(self, page):
        """Завершение работы"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        page.controls.clear()
        self.start_interface(page)
        page.update()

if __name__ == "__main__":
    import asyncio
    
    app = FaceRecognitionApp()
    app.load_face_descriptors()
    app.connect_to_arduino()
    
    try:
        ft.app(target=app.start_interface)
    finally:
        app.is_running = False
        if app.arduino_serial:
            app.arduino_serial.close()
        app.executor.shutdown()