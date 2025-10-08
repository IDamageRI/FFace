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

# ===== КОНСТАНТЫ И ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ =====
shape_predictor_path = 'face_model/shape_predictor_68_face_landmarks.dat'
face_rec_model_path = 'face_model/dlib_face_recognition_resnet_model_v1.dat'
smile_cascade_path = 'haarcascade_smile.xml'  # Haar каскад для улыбки
base_path = 'face_bd'
neutral_faces_path = 'face_bd/neutral'  # Папка с нейтральными лицами
smiling_faces_path = 'face_bd/smiling'  # Папка с улыбающимися лицами
log_file = 'detection_log.txt'

is_running = False
is_rtsp_running = False
cap = None
face_descriptors = []
faces = []
file_picker = ft.FilePicker()
arduino_serial = None
smile_cascade = None
rtsp_frame_queue = Queue(maxsize=1)
rtsp_url = "rtsp://admin:admin123@192.168.0.2:554/cam/realmonitor?channel=1&subtype=0"

# Настройка логирования
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

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
    except Exception as e:
        logging.error(f"Ошибка записи в лог: {e}")

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
    
    logging.info(f"Загружено {len(face_descriptors)} нейтральных лиц")

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
# ===== УЛУЧШЕННАЯ ДЕТЕКЦИЯ УЛЫБКИ =====
class SmileDetector:
    def __init__(self):
        self.smile_history = []  # История детекций для стабильности
        self.last_smile_time = 0
        self.smile_cooldown = 2.0  # Задержка между детекциями (сек)
        self.min_smile_duration = 0.5  # Минимальная продолжительность улыбки
        self.smile_start_time = 0
        self.confidence_threshold = 0.4  # Повышенный порог уверенности
        
    def detect_smile(self, face_roi):
        """Улучшенное обнаружение улыбки с защитой от ложных срабатываний"""
        try:
            if face_roi.size == 0:
                return False, 0.0
                
            # Проверка размера области лица
            height, width = face_roi.shape[:2]
            if height < 80 or width < 80:  # Увеличили минимальный размер
                return False, 0.0
            
            # Предобработка
            processed_face = self.preprocess_face(face_roi)
            gray = cv2.cvtColor(processed_face, cv2.COLOR_BGR2GRAY)
            
            current_time = time.time()
            
            # Проверка кулдауна
            if current_time - self.last_smile_time < self.smile_cooldown:
                return False, 0.0
            
            # Комбинированная детекция
            haar_confidence = self.detect_smile_haar(gray)
            landmark_confidence = self.analyze_lips_with_landmarks(face_roi)
            brightness_confidence = self.analyze_brightness_change(gray)
            
            # Взвешенное среднее с приоритетом landmarks
            total_confidence = (
                haar_confidence * 0.3 + 
                landmark_confidence * 0.5 + 
                brightness_confidence * 0.2
            )
            
            # Пороговая логика с гистерезисом
            smile_detected = total_confidence > self.confidence_threshold
            
            # Отслеживание продолжительности улыбки
            if smile_detected:
                if self.smile_start_time == 0:
                    self.smile_start_time = current_time
                elif current_time - self.smile_start_time >= self.min_smile_duration:
                    # Улыбка подтверждена после минимальной продолжительности
                    self.last_smile_time = current_time
                    self.smile_start_time = 0  # Сброс таймера
                    return True, total_confidence
                else:
                    # Улыбка еще не достигла минимальной продолжительности
                    return False, total_confidence
            else:
                self.smile_start_time = 0  # Сброс если улыбка пропала
                return False, total_confidence
                
        except Exception as e:
            logging.error(f"Ошибка детекции улыбки: {e}")
            return False, 0.0
    
    def preprocess_face(self, face_roi):
        """Улучшенная предобработка лица"""
        try:
            # Увеличение резкости с меньшей агрессивностью
            kernel = np.array([[0, -0.25, 0], [-0.25, 2, -0.25], [0, -0.25, 0]])
            sharpened = cv2.filter2D(face_roi, -1, kernel)
            
            # Нормализация освещения
            lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))  # Уменьшили clipLimit
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            
            return enhanced
        except:
            return face_roi
    
    def detect_smile_haar(self, gray_face):
        """Детекция улыбки с помощью Haar каскадов с оптимизированными параметрами"""
        if smile_cascade is None:
            return 0.0
            
        try:
            # Более консервативные параметры для уменьшения ложных срабатываний
            smiles = smile_cascade.detectMultiScale(
                gray_face,
                scaleFactor=1.8,
                minNeighbors=20,  # Увеличили для большей стабильности
                minSize=(35, 35),  # Увеличили минимальный размер
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            confidence = min(len(smiles) * 0.15, 1.0)  # Понизили вес каждого обнаружения
            return confidence
        except:
            return 0.0
    
    def analyze_lips_with_landmarks(self, face_roi):
        """Улучшенный анализ губ с дополнительными проверками"""
        try:
            dets = detector(face_roi, 1)
            if len(dets) == 0:
                return 0.0
                
            shape = sp(face_roi, dets[0])
            landmarks = np.array([[p.x, p.y] for p in shape.parts()])
            
            # Индексы точек губ (48-67)
            outer_lips = landmarks[48:60]
            inner_lips = landmarks[60:68]
            
            # Проверка качества landmarks
            if not self.validate_landmarks(outer_lips, inner_lips):
                return 0.0
            
            # Основные метрики
            lip_width = np.linalg.norm(outer_lips[6] - outer_lips[0])
            lip_height = np.linalg.norm(outer_lips[3] - outer_lips[9])
            
            if lip_height <= 0:
                return 0.0
                
            lip_ratio = lip_width / lip_height
            
            # Дополнительные метрики для надежности
            mouth_openness = self.calculate_mouth_openness(inner_lips)
            lip_curvature = self.analyze_lip_curvature(outer_lips)
            
            # Комбинированная уверенность
            ratio_confidence = min(lip_ratio / 2.8, 1.0)  # Понизили порог
            openness_confidence = min(mouth_openness / 15.0, 1.0)
            curvature_confidence = min(lip_curvature * 5.0, 1.0)
            
            total_confidence = (ratio_confidence * 0.5 + 
                             openness_confidence * 0.3 + 
                             curvature_confidence * 0.2)
            
            return total_confidence
            
        except Exception as e:
            logging.error(f"Ошибка анализа губ: {e}")
            return 0.0
    
    def validate_landmarks(self, outer_lips, inner_lips):
        """Проверка валидности landmarks"""
        if len(outer_lips) != 12 or len(inner_lips) != 8:
            return False
            
        # Проверка на выбросы (слишком большие расстояния между точками)
        for i in range(len(outer_lips) - 1):
            dist = np.linalg.norm(outer_lips[i] - outer_lips[i + 1])
            if dist > 100:  # Максимальное разумное расстояние
                return False
                
        return True
    
    def calculate_mouth_openness(self, inner_lips):
        """Вычисление степени открытости рта"""
        if len(inner_lips) < 6:
            return 0.0
            
        # Расстояние между верхней и нижней губой
        upper_lip = inner_lips[3]  # Центр верхней губы
        lower_lip = inner_lips[5]  # Центр нижней губы
        
        return abs(upper_lip[1] - lower_lip[1])
    
    def analyze_lip_curvature(self, lip_points):
        """Анализ кривизны губ"""
        if len(lip_points) < 3:
            return 0.0
            
        x_coords = lip_points[:, 0]
        y_coords = lip_points[:, 1]
        
        if max(x_coords) - min(x_coords) > 0:
            # Полиномиальная аппроксимация для лучшей оценки кривизны
            try:
                z = np.polyfit(x_coords, y_coords, 2)
                curvature = abs(z[0])  # Коэффициент при x^2
                return curvature * 1000  # Масштабирование
            except:
                return 0.0
        
        return 0.0
    
    def analyze_brightness_change(self, gray_face):
        """Более консервативный анализ изменения яркости"""
        try:
            height, width = gray_face.shape
            if height < 3 or width < 3:
                return 0.0
                
            # Увеличиваем области для анализа
            mouth_region = gray_face[int(height*0.6):, :]
            forehead_region = gray_face[:int(height*0.3), :]
            
            if mouth_region.size == 0 or forehead_region.size == 0:
                return 0.0
                
            mouth_brightness = np.mean(mouth_region)
            forehead_brightness = np.mean(forehead_region)
            
            if forehead_brightness > 10:  # Минимальная яркость
                brightness_ratio = mouth_brightness / forehead_brightness
                # Более консервативные пороги
                confidence = max(0, min((brightness_ratio - 0.9) / 0.3, 1.0))
                return confidence
                
            return 0.0
        except:
            return 0.0

# Инициализация детектора улыбки
smile_detector = SmileDetector()

# Замените вызовы detect_smile на smile_detector.detect_smile
def two_factor_authentication(identity, frame, face_location):
    """Улучшенная проверка второго фактора - улыбки"""
    top, right, bottom, left = face_location
    
    # Увеличиваем область для захвата большего контекста
    expand = 25
    top = max(0, top - expand)
    left = max(0, left - expand)
    bottom = min(frame.shape[0], bottom + expand)
    right = min(frame.shape[1], right + expand)
    
    face_roi = frame[top:bottom, left:right]
    
    if face_roi.size == 0:
        return False, 0.0
    
    # Используем улучшенный детектор улыбки
    smile_detected, smile_confidence = smile_detector.detect_smile(face_roi)
    
    return smile_detected, smile_confidence


    # ===== Предобработка =====
def preprocess_face(face_roi):
    """Предобработка области лица для лучшей детекции"""
    try:
        # Увеличение резкости
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(face_roi, -1, kernel)
        
        # Увеличение контраста
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        return enhanced
    except:
        return face_roi

# ===== ДВУХФАКТОРНАЯ АУТЕНТИФИКАЦИЯ =====
def two_factor_authentication(identity, frame, face_location):
    """Улучшенная проверка второго фактора - улыбки"""
    top, right, bottom, left = face_location
    
    # Увеличиваем область для захвата большего контекста
    expand = 20
    top = max(0, top - expand)
    left = max(0, left - expand)
    bottom = min(frame.shape[0], bottom + expand)
    right = min(frame.shape[1], right + expand)
    
    face_roi = frame[top:bottom, left:right]
    
    if face_roi.size == 0:
        return False, 0.0
    
    # Предобработка
    processed_face = preprocess_face(face_roi)
    
    # Детекция улыбки с улучшенным методом
    smile_detected, smile_confidence = detect_smile(processed_face)
    
    # Дополнительная проверка по времени для стабильности
    if smile_detected and smile_confidence > 0.5:
        # Можно добавить проверку продолжительности улыбки
        return True, smile_confidence
    
    return smile_detected, smile_confidence


# ===== RTSP ФУНКЦИИ =====
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
    """Запуск IP-камеры по RTSP с двухфакторной аутентификацией"""
    global is_rtsp_running
    
    load_face_descriptors()
    is_rtsp_running = True
    exit_button.visible = True
    page.update()
    
    rtsp_thread = threading.Thread(target=rtsp_capture_thread, daemon=True)
    rtsp_thread.start()
    
    last_detected = None
    arduino_triggered = False
    last_detection_time = 0
    last_frame_time = time.time()
    auth_stage = "face_detection"
    current_identity = None
    
    while is_rtsp_running:
        try:
            if not rtsp_frame_queue.empty():
                frame = rtsp_frame_queue.get()
                
                # Ограничение FPS для интерфейса (~30 кадров/сек)
                if time.time() - last_frame_time < 0.033:
                    continue
                
                last_frame_time = time.time()
                current_time = time.time()
                
                display_frame = frame.copy()
                
                # Детекция лиц
                dets = detector(frame, 0)
                
                if dets:
                    for d in dets:
                        top, right, bottom, left = d.top(), d.right(), d.bottom(), d.left()
                        
                        # Рисуем рамку вокруг лица
                        color = (0, 255, 0) if auth_stage == "smile_verification" else (255, 255, 0)
                        cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                        
                        if auth_stage == "face_detection" and current_time - last_detection_time >= 3:
                            # Первый фактор: распознавание лица
                            min_dist, closest_face, is_match = compare_faces(frame)
                            
                            if is_match:
                                current_identity = os.path.splitext(closest_face)[0]
                                status_text.value = f"Распознан: {current_identity}. Улыбнитесь!"
                                status_text.color = ft.Colors.BLUE
                                
                                # Переходим к проверке улыбки
                                auth_stage = "smile_verification"
                                last_detection_time = current_time
                                
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
                            elif current_time - last_detection_time > 10:
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
            
            time.sleep(0.01)
            
        except Exception as e:
            logging.error(f"Ошибка в основном цикле RTSP: {str(e)}")
            time.sleep(1)
    
    if arduino_triggered:
        send_to_arduino('0')

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
    last_frame_time = time.time()

    while is_running:
        ret, frame = cap.read()
        if not ret:
            status_text.value = "Ошибка чтения кадра"
            status_text.color = ft.Colors.RED
            page.update()
            break

        # Ограничение FPS для интерфейса
        if time.time() - last_frame_time < 0.033:
            continue
            
        last_frame_time = time.time()
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

    if cap:
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
            log_detection(face_name, "Распознано на изображении")
            send_to_arduino('1')

            match_img_path = os.path.join(neutral_faces_path, closest_face)
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
        auth_stage = "face_detection"
        current_identity = None

        load_face_descriptors()

        while is_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = resize_image(frame)
            frame_count += 1
            display_frame = frame.copy()

            if frame_count % 30 == 0:
                dets = detector(frame, 0)
                
                if dets:
                    for d in dets:
                        top, right, bottom, left = d.top(), d.right(), d.bottom(), d.left()
                        
                        if auth_stage == "face_detection":
                            min_dist, closest_face, is_match = compare_faces(frame)
                            
                            if is_match:
                                current_identity = os.path.splitext(closest_face)[0]
                                auth_stage = "smile_verification"
                                
                                match_img_path = os.path.join(neutral_faces_path, closest_face)
                                if os.path.exists(match_img_path):
                                    match_img = cv2.imdecode(np.fromfile(match_img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                                    match_img = resize_image(match_img)
                                    match_area.src_base64 = image_to_base64(match_img)
                        
                        elif auth_stage == "smile_verification" and current_identity:
                            smile_detected, smile_conf = two_factor_authentication(
                                closest_face, frame, (top, right, bottom, left)
                            )
                            
                            if smile_detected:
                                if not arduino_triggered:
                                    send_to_arduino('1')
                                    arduino_triggered = True
                                    log_detection(current_identity, "Успешная аутентификация в видео", True)
                                
                                status_text.value = f"Доступ разрешен: {current_identity}"
                                status_text.color = ft.Colors.GREEN
                                display_frame = putText_rus(display_frame, f"Улыбка подтверждена!", 
                                                          (left, top - 30), (0, 255, 0), 20)
                            
                            # Упрощенная логика для видео
                            auth_stage = "face_detection"

            image_area.src_base64 = image_to_base64(display_frame)
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

def exit_mode(page):
    """Выход из режима"""
    global is_running, is_rtsp_running, cap
    is_running = False
    is_rtsp_running = False
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
    
    # Создание журнала событий
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

    # Создание элементов интерфейса
    image_area = ft.Image(src=f"None", width=1000, height=700, fit=ft.ImageFit.CONTAIN)
    match_area = ft.Image(src=f"None", width=300, height=400)
    status_text = ft.Text(size=18, weight="bold", width=280, color="white")
    
    # Элементы выбора камеры
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
    
    # Кнопки
    button_style = ft.ButtonStyle(
        padding=20,
        bgcolor=ft.Colors.BLUE_700,
        color=ft.Colors.WHITE,
        overlay_color=ft.Colors.BLUE_900,
    )

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
        text="Камера",
        icon=ft.Icons.CAMERA,
        style=button_style,
        width=200,
        height=80,
        on_click=lambda e: update_camera_list(),
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

# ===== ЗАПУСК ПРИЛОЖЕНИЯ =====
if __name__ == "__main__":
    ft.app(target=start_interface)