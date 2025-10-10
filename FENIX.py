 # Facial Emotion Neural Identification compleX
import base64
import os
from datetime import datetime
import cv2
import flet as ft
from scipy.spatial import distance
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import serial
import serial.tools.list_ports
import time
import threading
from queue import Queue, Empty
import logging
import urllib.request
import sys
import subprocess
import dlib  

#pip install dlib-19.24.1-cp311-cp311-win_amd64.whl
# Автоматически определяем корень проекта & догружаем модуль
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
required_libs = ["cv2", "flet", "numpy", "PIL", "dlib", "scipy", "serial"]
for lib in required_libs:
    try:
        __import__(lib)
    except ImportError:
        print(f"[!] Устанавливаю отсутствующий модуль: {lib}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib])


# ---------------- Конфигурация и логирование ----------------
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ---------------- НАСТРОЙКИ ТОЧНОСТИ ДЕТЕКЦИИ УЛЫБОК ----------------
# Настройки для точной детекции улыбок (можно изменять для настройки чувствительности)
SMILE_DETECTION_CONFIG = {
    'scale_factor': 1.5,           # Масштабирующий фактор (1.1-2.0, меньше = точнее)
    'min_neighbors': 25,            # Минимальные соседи (20-50, больше = меньше ложных срабатываний)
    'min_size': (20, 20),          # Минимальный размер улыбки
    'max_size': (200, 200),        # Максимальный размер улыбки
    'confidence_threshold': 0.4,   # Порог уверенности (0.3-0.8, больше = строже)
    'cooldown_time': 1.0,           # Время между детекциями в секундах
    'min_face_size': 40,           # Минимальный размер области лица
    'brightness_weight': 0.2,      # Вес анализа яркости (0.0-0.5)
    'base_confidence_weight': 0.8  # Вес базовой уверенности (0.5-1.0)
}

# Файлы/папки/модели
# Все пути теперь от BASE_DIR:
shape_predictor_path = os.path.join(BASE_DIR, 'face_model', 'shape_predictor_68_face_landmarks.dat')
face_rec_model_path = os.path.join(BASE_DIR, 'face_model', 'dlib_face_recognition_resnet_model_v1.dat')
smile_cascade_path = os.path.join(BASE_DIR, 'haarcascade_smile.xml')# Haar каскад для улыбки (скачаем при необходимости)
neutral_faces_path = os.path.join(BASE_DIR, 'face_bd', 'neutral')# нейтральные фото (для демонстрации/эталонов)
smiling_faces_path = os.path.join(BASE_DIR, 'face_bd', 'smiling')# улыбающиеся эталоны
log_file = os.path.join(BASE_DIR, 'detection_log.txt')

# Основная БД лиц (фото для идентификации)
base_path = 'face_bd'         

# RTSP URL (по умолчанию, редактируй в интерфейсе если нужно)
rtsp_url = "rtsp://admin:admin123@192.168.0.2:554/cam/realmonitor?channel=1&subtype=0"

# Глобальные переменные
is_running = False
is_rtsp_running = False
cap = None
face_descriptors = []   # дескрипторы (в том же порядке, что и faces)
faces = []              # имена файлов (с расширением)
file_picker = ft.FilePicker()
arduino_serial = None
rtsp_frame_queue = Queue(maxsize=1)
smile_cascade = None

# dlib модели (будут инициализированы)
detector = None
sp = None
facerec = None

# ---------------- Утилиты ----------------
def ensure_dirs():
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(neutral_faces_path, exist_ok=True)
    os.makedirs(smiling_faces_path, exist_ok=True)

def putText_rus(img, text, pos, color=(0, 255, 0), font_size=20):
    """Добавляет русскоязычный текст на OpenCV-изображение используя PIL."""
    try:
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        draw.text(pos, text, font=font, fill=color[::-1])
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logging.error(f"putText_rus error: {e}")
        return img

def resize_image(image, max_width=1200, max_height=900):
    """Пропорциональное уменьшение изображения до указанных размеров."""
    if image is None:
        return image
    h, w = image.shape[:2]
    if w > max_width or h > max_height:
        scale = min(max_width / w, max_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h))
    return image

def image_to_base64(image):
    """Конвертация изображения в base64 для отображения в Flet."""
    try:
        _, encoded_img = cv2.imencode('.png', image)
        return base64.b64encode(encoded_img).decode('utf-8')
    except Exception as e:
        logging.error(f"image_to_base64 error: {e}")
        return None

def log_detection(name, status, smile_detected=False):
    """Логирование событий аутентификации в файл dtection_log.txt и в отдельный лог."""
    try:
        smile_status = " с улыбкой" if smile_detected else ""
        line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {status}: {name}{smile_status}\n"
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(line)
        logging.info(line.strip())
    except Exception as e:
        logging.error(f"log_detection error: {e}")

def list_available_cameras(max_tested=6):
    """Проверка доступных локальных камер (0..max_tested-1)."""
    available_cameras = []
    for i in range(max_tested):
        try:
            c = cv2.VideoCapture(i)
            if c.isOpened():
                available_cameras.append(i)
            c.release()
        except:
            pass
    return available_cameras

# ---------------- Инициализация моделей ----------------
def download_haar_cascade():
    """Скачивает haarcascade_smile.xml из репозитория OpenCV, если файл отсутствует."""
    try:
        logging.info("Скачиваем Haar каскад для улыбки...")
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_smile.xml"
        urllib.request.urlretrieve(url, smile_cascade_path)
        logging.info("Каскад улыбки успешно скачан.")
    except Exception as e:
        logging.error(f"Ошибка скачивания каскада улыбки: {e}")

def initialize_models():
    """Инициализация dlib моделей и каскада улыбки."""
    global detector, sp, facerec, smile_cascade
    ensure_dirs()

    try:
        detector = dlib.get_frontal_face_detector()
        sp = dlib.shape_predictor(shape_predictor_path)
        facerec = dlib.face_recognition_model_v1(face_rec_model_path)
        logging.info("Dlib модели успешно загружены.")
    except Exception as e:
        logging.error(f"Ошибка инициализации dlib моделей: {e}")
        raise

    # Инициализация Haar каскада улыбки
    try:
        if not os.path.exists(smile_cascade_path):
            download_haar_cascade()
        smile_cascade = cv2.CascadeClassifier(smile_cascade_path)
        if smile_cascade.empty():
            logging.warning("Каскад улыбки загружен, но пуст (ошибка).")
        else:
            logging.info("Каскад улыбки загружен успешно.")
    except Exception as e:
        logging.error(f"Ошибка инициализации каскада улыбки: {e}")
        smile_cascade = None

# ---------------- Загрузка БД лиц ----------------
def load_face_descriptors():
    """Загружает дескрипторы лиц из base_path и список файлов."""
    global face_descriptors, faces
    face_descriptors = []
    faces = []

    ensure_dirs()
    files = [f for f in os.listdir(base_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    faces = files

    if not files:
        logging.warning(f"В папке {base_path} не найдено изображений эталонов.")
        return

    for face in files:
        img_path = os.path.join(base_path, face)
        try:
            # Используем imdecode чтобы корректно читать пути с юникодом/пробелами
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                logging.error(f"Не удалось прочитать изображение {img_path}")
                continue
            dets = detector(img, 1)
            if len(dets) == 0:
                logging.warning(f"Лицо не найдено на эталоне {img_path}")
                continue
            # Берём первое лицо
            d = dets[0]
            shape = sp(img, d)
            fd = facerec.compute_face_descriptor(img, shape)
            face_descriptors.append(fd)
        except Exception as e:
            logging.error(f"Ошибка при обработке {img_path}: {e}")

    logging.info(f"Загружено {len(face_descriptors)} дескрипторов из {base_path}")

# ---------------- Сравнение лиц ----------------
def compare_faces(frame, threshold=0.5):
    """Сравнивает лицо(а) в кадре с loaded descriptors, возвращает (min_dist, closest_face_filename, is_match)."""
    try:
        dets = detector(frame, 0)
        if len(dets) == 0 or not face_descriptors:
            return None, None, False

        # Если несколько лиц, сравниваем только первое (можно расширить)
        d = dets[0]
        shape = sp(frame, d)
        main_descriptor = facerec.compute_face_descriptor(frame, shape)
        distances = [distance.euclidean(main_descriptor, fd) for fd in face_descriptors]
        min_dist = min(distances)
        closest_face_idx = distances.index(min_dist)
        is_match = min_dist <= threshold
        return min_dist, faces[closest_face_idx], is_match
    except Exception as e:
        logging.error(f"compare_faces error: {e}")
        return None, None, False

# ---------------- ДЕТЕКЦИЯ УЛЫБКИ ----------------
# Глобальные переменные для временной фильтрации
smile_detection_history = []
last_smile_time = 0
smile_cooldown = 1.5  # секунды между детекциями

def detect_smile(face_roi):
    """Улучшенное обнаружение улыбки с защитой от ложных срабатываний.
       Возвращает (smile_detected: bool, confidence: float 0..1).
    """
    global smile_detection_history, last_smile_time
    
    if smile_cascade is None:
        return False, 0.0
    
    try:
        # Проверка размера области лица
        height, width = face_roi.shape[:2]
        min_face_size = SMILE_DETECTION_CONFIG['min_face_size']
        if height < min_face_size or width < min_face_size:
            return False, 0.0
        
        # Проверка кулдауна
        current_time = time.time()
        cooldown_time = SMILE_DETECTION_CONFIG['cooldown_time']
        if current_time - last_smile_time < cooldown_time:
            return False, 0.0
        
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Используем настраиваемые параметры для детекции
        smiles = smile_cascade.detectMultiScale(
            gray,
            scaleFactor=SMILE_DETECTION_CONFIG['scale_factor'],
            minNeighbors=SMILE_DETECTION_CONFIG['min_neighbors'],
            minSize=SMILE_DETECTION_CONFIG['min_size'],
            maxSize=SMILE_DETECTION_CONFIG['max_size'],
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Анализ качества детекции
        if len(smiles) == 0:
            return False, 0.0
        
        # Фильтрация по размеру и позиции
        valid_smiles = []
        for (x, y, w, h) in smiles:
            # Проверяем, что улыбка находится в нижней части лица
            face_center_y = height // 2
            if y > face_center_y and w > 25 and h > 25:
                valid_smiles.append((x, y, w, h))
        
        if len(valid_smiles) == 0:
            return False, 0.0
        
        # Вычисляем уверенность на основе количества и качества детекций
        base_confidence = min(len(valid_smiles) / 3.0, 1.0)
        
        # Дополнительная проверка: анализируем яркость в области улыбки
        brightness_confidence = analyze_smile_brightness(gray, valid_smiles)
        
        # Комбинированная уверенность с настраиваемыми весами
        base_weight = SMILE_DETECTION_CONFIG['base_confidence_weight']
        brightness_weight = SMILE_DETECTION_CONFIG['brightness_weight']
        total_confidence = (base_confidence * base_weight + brightness_confidence * brightness_weight)
        
        # Порог уверенности для предотвращения ложных срабатываний
        confidence_threshold = SMILE_DETECTION_CONFIG['confidence_threshold']
        
        smile_detected = total_confidence > confidence_threshold
        
        # Отладочная информация
        logging.info(f"Smile detection: confidence={total_confidence:.2f}, threshold={confidence_threshold:.2f}, detected={smile_detected}")
        
        if smile_detected:
            last_smile_time = current_time
            # Добавляем в историю для дополнительной фильтрации
            smile_detection_history.append(current_time)
            # Ограничиваем историю последними 5 детекциями
            if len(smile_detection_history) > 5:
                smile_detection_history.pop(0)
        
        return smile_detected, total_confidence
        
    except Exception as e:
        logging.error(f"detect_smile error: {e}")
        return False, 0.0

def analyze_smile_brightness(gray_face, smiles):
    """Анализирует изменение яркости в области улыбки для дополнительной проверки."""
    try:
        if len(smiles) == 0:
            return 0.0
        
        # Берем первую (наиболее вероятную) улыбку
        x, y, w, h = smiles[0]
        
        # Извлекаем область улыбки
        smile_roi = gray_face[y:y+h, x:x+w]
        if smile_roi.size == 0:
            return 0.0
        
        # Вычисляем среднюю яркость
        mean_brightness = np.mean(smile_roi)
        
        # Нормализуем яркость (улыбка обычно ярче)
        brightness_confidence = min(mean_brightness / 150.0, 1.0)
        
        return brightness_confidence
        
    except Exception as e:
        logging.error(f"analyze_smile_brightness error: {e}")
        return 0.0

# ---------------- Arduino ----------------
def connect_to_arduino():
    """Попытка подключиться ко всем доступным COM-портам."""
    global arduino_serial
    ports = serial.tools.list_ports.comports()
    for port in ports:
        try:
            arduino_serial = serial.Serial(port.device, 9600, timeout=1)
            time.sleep(2)
            logging.info(f"Успешно подключено к Arduino на {port.device}")
            return arduino_serial
        except (serial.SerialException, serial.SerialTimeoutException) as e:
            logging.warning(f"Ошибка подключения к {port.device}: {e}")
            continue
    logging.info("Arduino не найдена. Переходим в режим эмуляции.")
    return None

def send_to_arduino(command):
    """Отправляет строковую команду на Arduino (например '1' или '0')."""
    global arduino_serial
    if arduino_serial and arduino_serial.is_open:
        try:
            arduino_serial.write(command.encode())
            logging.info(f"Отправлено на Arduino: {command}")
        except Exception as e:
            logging.error(f"send_to_arduino error: {e}")

# ---------------- RTSP (IP-камера) ----------------
def rtsp_capture_thread(local_rtsp_url):
    """Поток захвата RTSP, кладёт последний кадр в очередь rtsp_frame_queue."""
    global is_rtsp_running, rtsp_frame_queue
    # Попытки обеспечить устойчивое подключение
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;65536"
    while is_rtsp_running:
        cap_local = None
        try:
            cap_local = cv2.VideoCapture(local_rtsp_url, cv2.CAP_FFMPEG)
            cap_local.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 30000)
            cap_local.set(cv2.CAP_PROP_BUFFERSIZE, 3)
            if not cap_local.isOpened():
                logging.warning("RTSP: не удалось подключиться, повтор через 5 с")
                time.sleep(5)
                continue
            logging.info("RTSP: подключение установлено")
            while is_rtsp_running:
                ret, frame = cap_local.read()
                if not ret or frame is None:
                    logging.warning("RTSP: кадр не получен, переподключение")
                    break
                frame = resize_image(frame)
                # поддерживаем только последний кадр в очереди
                try:
                    if not rtsp_frame_queue.empty():
                        rtsp_frame_queue.get_nowait()
                except:
                    pass
                try:
                    rtsp_frame_queue.put_nowait(frame)
                except:
                    pass
                time.sleep(0.01)
        except Exception as e:
            logging.error(f"RTSP capture error: {e}")
            time.sleep(3)
        finally:
            try:
                if cap_local is not None:
                    cap_local.release()
            except:
                pass

# ---------------- Основные режимы: RTSP и Webcam с Smile ----------------
def start_rtsp_camera(page, image_area, match_area, status_text, exit_button, local_rtsp_url=rtsp_url):
    """Запуск распознавания на RTSP-потоке с двухфакторной проверкой."""
    global is_rtsp_running, rtsp_frame_queue
    load_face_descriptors()
    is_rtsp_running = True
    exit_button.visible = True
    page.update()

    # Старт потока захвата
    rtsp_thread = threading.Thread(target=rtsp_capture_thread, args=(local_rtsp_url,), daemon=True)
    rtsp_thread.start()

    last_detected = None
    arduino_triggered = False
    last_detection_time = 0
    auth_stage = "face_detection"
    current_identity = None
    smile_start_time = 0

    while is_rtsp_running:
        try:
            try:
                frame = rtsp_frame_queue.get(timeout=1)
            except Empty:
                # нет кадров — продолжаем цикл
                continue

            # Ограничиваем обновление интерфейса ~30 FPS
            # (Можно улучшить синхронизацию)
            frame = resize_image(frame)
            display_frame = frame.copy()

            # Если стадия face_detection, пытаемся распознать
            if auth_stage == "face_detection":
                min_dist, closest_face, is_match = compare_faces(frame)
                if is_match:
                    current_identity = os.path.splitext(closest_face)[0]
                    status_text.value = f"Распознан: {current_identity}. Улыбнитесь!"
                    status_text.color = ft.colors.BLUE
                    # показать эталонное нейтральное фото, если есть
                    neutral_path = os.path.join(neutral_faces_path, closest_face)
                    if os.path.exists(neutral_path):
                        match_img = cv2.imdecode(np.fromfile(neutral_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if match_img is not None:
                            match_area.src_base64 = image_to_base64(resize_image(match_img))
                    auth_stage = "smile_verification"
                    smile_start_time = time.time()
                    last_detection_time = time.time()
                else:
                    # Если нет совпадения, показываем статус
                    status_text.value = "Лицо не найдено"
                    status_text.color = ft.colors.RED
                    match_area.src_base64 = None

            elif auth_stage == "smile_verification" and current_identity:
                # находим ROI по детекции лиц - возьмём первый детект
                dets = detector(frame, 0)
                if dets:
                    d = dets[0]
                    top, right, bottom, left = d.top(), d.right(), d.bottom(), d.left()
                    # корректируем границы в рамках изображения
                    h, w = frame.shape[:2]
                    top = max(0, top); left = max(0, left)
                    bottom = min(h, bottom); right = min(w, right)
                    face_roi = frame[top:bottom, left:right]
                    smile_detected, smile_conf = detect_smile(face_roi)
                    if smile_detected:
                        # Успешная двухфакторная аутентификация
                        status_text.value = f"Доступ разрешен: {current_identity}"
                        status_text.color = ft.colors.GREEN
                        log_detection(current_identity, "Успешная аутентификация", True)
                        # показать улыбающееся фото из папки smiling если есть
                        smile_path = os.path.join(smiling_faces_path, closest_face)
                        if os.path.exists(smile_path):
                            smile_img = cv2.imdecode(np.fromfile(smile_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                            if smile_img is not None:
                                match_area.src_base64 = image_to_base64(resize_image(smile_img))
                        # индикатор на Arduino
                        if not arduino_triggered:
                            send_to_arduino('1')
                            arduino_triggered = True
                        display_frame = putText_rus(display_frame, "Улыбка подтверждена!", (10, 60), (0,255,0), 20)
                        # После успешной аутентификации возвращаемся в face_detection через небольшую паузу
                        current_identity = None
                        auth_stage = "face_detection"
                        last_detection_time = time.time()
                    else:
                        # Показываем уверенность для отладки
                        remaining_time = 10 - (time.time() - smile_start_time)
                        status_text.value = f"Улыбнитесь! Уверенность: {smile_conf:.2f}, осталось: {remaining_time:.1f}с"
                        status_text.color = ft.colors.YELLOW
                        # Проверка таймаута
                        if time.time() - smile_start_time > 10:
                            status_text.value = "Таймаут улыбки. Попробуйте снова."
                            status_text.color = ft.colors.ORANGE
                            log_detection(current_identity, "Таймаут улыбки", False)
                            current_identity = None
                            auth_stage = "face_detection"
                            if arduino_triggered:
                                send_to_arduino('0')
                                arduino_triggered = False
                else:
                    # Если лицо перестало быть видимым — отменяем стадию
                    if time.time() - smile_start_time > 2:
                        status_text.value = "Лицо потеряно"
                        status_text.color = ft.colors.RED
                        current_identity = None
                        auth_stage = "face_detection"
                        if arduino_triggered:
                            send_to_arduino('0')
                            arduino_triggered = False

            # Отрисовка информации на кадре
            stage_text = "Поиск лица" if auth_stage == "face_detection" else "Улыбнитесь!"
            display_frame = putText_rus(display_frame, stage_text, (10, 30),
                                       (0, 255, 0) if auth_stage == "smile_verification" else (255,255,0), 20)

            image_area.src_base64 = image_to_base64(display_frame)
            page.update()
            time.sleep(0.01)

        except Exception as e:
            logging.error(f"Ошибка в RTSP основном цикле: {e}")
            time.sleep(1)

    # выход из режима
    if arduino_triggered:
        send_to_arduino('0')

def start_webcam(page, image_area, match_area, status_text, exit_button, camera_index=0):
    """Запуск локальной веб-камеры с face+smile двухфакторной аутентификацией."""
    global is_running, cap
    load_face_descriptors()
    is_running = True
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        status_text.value = f"Ошибка: не удалось открыть камеру {camera_index}"
        status_text.color = ft.colors.RED
        page.update()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    exit_button.visible = True
    page.update()

    last_detected = None
    arduino_triggered = False
    last_frame_time = time.time()
    auth_stage = "face_detection"
    current_identity = None
    smile_start_time = 0
    closest_face_last = None

    while is_running:
        ret, frame = cap.read()
        if not ret:
            status_text.value = "Ошибка чтения кадра с камеры"
            status_text.color = ft.colors.RED
            page.update()
            break

        # Ограничение интерфейсного FPS
        if time.time() - last_frame_time < 0.033:
            continue
        last_frame_time = time.time()

        frame = resize_image(frame)
        display_frame = frame.copy()

        try:
            if auth_stage == "face_detection":
                min_dist, closest_face, is_match = compare_faces(frame)
                if is_match:
                    current_identity = os.path.splitext(closest_face)[0]
                    closest_face_last = closest_face
                    status_text.value = f"Распознан: {current_identity}. Улыбнитесь!"
                    status_text.color = ft.colors.BLUE
                    # показать нейтральное фото, если есть
                    neutral_path = os.path.join(neutral_faces_path, closest_face)
                    if os.path.exists(neutral_path):
                        mimg = cv2.imdecode(np.fromfile(neutral_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if mimg is not None:
                            match_area.src_base64 = image_to_base64(resize_image(mimg))
                    auth_stage = "smile_verification"
                    smile_start_time = time.time()
                else:
                    status_text.value = "Лицо не найдено"
                    status_text.color = ft.colors.RED
                    match_area.src_base64 = None

            elif auth_stage == "smile_verification" and current_identity:
                # определяем ROI по детекции лиц
                dets = detector(frame, 0)
                if dets:
                    d = dets[0]
                    top, right, bottom, left = d.top(), d.right(), d.bottom(), d.left()
                    h, w = frame.shape[:2]
                    top = max(0, top); left = max(0, left)
                    bottom = min(h, bottom); right = min(w, right)
                    face_roi = frame[top:bottom, left:right]
                    smile_detected, smile_conf = detect_smile(face_roi)
                    if smile_detected:
                        status_text.value = f"Доступ разрешен: {current_identity}"
                        status_text.color = ft.colors.GREEN
                        log_detection(current_identity, "Успешная аутентификация", True)
                        # показать улыбающееся изображение, если есть
                        smile_path = os.path.join(smiling_faces_path, closest_face_last) if closest_face_last else None
                        if smile_path and os.path.exists(smile_path):
                            smimg = cv2.imdecode(np.fromfile(smile_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                            if smimg is not None:
                                match_area.src_base64 = image_to_base64(resize_image(smimg))
                        if not arduino_triggered:
                            send_to_arduino('1')
                            arduino_triggered = True
                        display_frame = putText_rus(display_frame, "Улыбка подтверждена!", (10, 60), (0,255,0), 20)
                        # Вернуться к поиску лица
                        current_identity = None
                        auth_stage = "face_detection"
                    else:
                        # Показываем уверенность для отладки
                        remaining_time = 10 - (time.time() - smile_start_time)
                        status_text.value = f"Улыбнитесь! Уверенность: {smile_conf:.2f}, осталось: {remaining_time:.1f}с"
                        status_text.color = ft.colors.YELLOW
                        if time.time() - smile_start_time > 10:
                            status_text.value = "Таймаут улыбки. Повторите распознавание."
                            status_text.color = ft.colors.ORANGE
                            log_detection(current_identity, "Таймаут улыбки", False)
                            current_identity = None
                            auth_stage = "face_detection"
                            if arduino_triggered:
                                send_to_arduino('0')
                                arduino_triggered = False
                else:
                    if time.time() - smile_start_time > 2:
                        status_text.value = "Лицо потеряно"
                        status_text.color = ft.colors.RED
                        current_identity = None
                        auth_stage = "face_detection"
                        if arduino_triggered:
                            send_to_arduino('0')
                            arduino_triggered = False

            # Отрисовка стадии на кадре
            stage_text = "Поиск лица" if auth_stage == "face_detection" else "Улыбнитесь!"
            display_frame = putText_rus(display_frame, stage_text, (10, 30),
                                       (0,255,0) if auth_stage=="smile_verification" else (255,255,0), 20)

            image_area.src_base64 = image_to_base64(display_frame)
            page.update()
        except Exception as e:
            logging.error(f"Ошибка в основном цикле webcam: {e}")
            time.sleep(0.01)

    # Очистка
    if cap is not None:
        cap.release()
    if arduino_triggered:
        send_to_arduino('0')

# ---------------- Обработка выбора файла/видео ----------------
def process_selected_image(e, page, image_area, match_area, status_text, exit_button):
    if e.files:
        image_path = e.files[0].path
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            status_text.value = "Ошибка загрузки изображения"
            status_text.color = ft.colors.RED
            page.update()
            return
        load_face_descriptors()
        img = resize_image(img)
        min_dist, closest_face, is_match = compare_faces(img)
        if is_match:
            face_name = os.path.splitext(closest_face)[0]
            status_text.value = f"Совпадение: {face_name}"
            status_text.color = ft.colors.GREEN
            log_detection(closest_face, "Найдено (файл)", False)
            send_to_arduino('1')
            match_img_path = os.path.join(base_path, closest_face)
            if os.path.exists(match_img_path):
                match_img = cv2.imdecode(np.fromfile(match_img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                match_area.src_base64 = image_to_base64(resize_image(match_img))
        else:
            status_text.value = "Совпадений не найдено"
            status_text.color = ft.colors.RED
            match_area.src_base64 = None
        image_area.src_base64 = image_to_base64(img)
        exit_button.visible = True
        page.update()

def process_selected_video(e, page, image_area, match_area, status_text, exit_button):
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
            # распознаем каждые N кадров (например каждые 30)
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
                        log_detection(closest_face, "Найдено (видео)", False)
                        status_text.value = f"Найдено: {face_name}"
                        status_text.color = ft.colors.GREEN
                        match_img_path = os.path.join(base_path, closest_face)
                        if os.path.exists(match_img_path):
                            match_img = cv2.imdecode(np.fromfile(match_img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                            match_area.src_base64 = image_to_base64(resize_image(match_img))
                    last_detected = closest_face
                    frame = putText_rus(frame, f"Найдено: {face_name}", (10, 30), (0,255,0), 20)
                else:
                    if last_detected:
                        status_text.value = "Лицо не найдено"
                        status_text.color = ft.colors.RED
                        match_area.src_base64 = None
                        if arduino_triggered:
                            send_to_arduino('0')
                            arduino_triggered = False
                    frame = putText_rus(frame, "Лицо не найдено", (10, 30), (0,0,255), 20)
            image_area.src_base64 = image_to_base64(frame)
            page.update()
        if cap is not None:
            cap.release()
        if arduino_triggered:
            send_to_arduino('0')

# ---------------- Интерфейс ----------------
def exit_mode(page):
    """Выход из режима — останавливает потоки и возвращает стартовый интерфейс."""
    global is_running, is_rtsp_running, cap
    is_running = False
    is_rtsp_running = False
    try:
        if cap is not None:
            cap.release()
    except:
        pass
    # Очистим интерфейс и перезапустим UI
    for control in page.controls[:]:
        try:
            page.controls.remove(control)
        except:
            pass
    start_interface(page)
    page.update()

def pick_image(page, image_area, match_area, status_text, exit_button):
    file_picker.on_result = lambda e: process_selected_image(e, page, image_area, match_area, status_text, exit_button)
    file_picker.pick_files("Выберите изображение", allowed_extensions=["jpg","jpeg","png"])
    exit_button.visible = True
    page.update()

def pick_video(page, image_area, match_area, status_text, exit_button):
    file_picker.on_result = lambda e: process_selected_video(e, page, image_area, match_area, status_text, exit_button)
    file_picker.pick_files("Выберите видео", allowed_extensions=["mp4","avi","mov"])
    exit_button.visible = True
    page.update()

def update_smile_settings(confidence_threshold, min_neighbors, cooldown_time):
    """Обновляет настройки детекции улыбок."""
    global SMILE_DETECTION_CONFIG
    SMILE_DETECTION_CONFIG['confidence_threshold'] = confidence_threshold
    SMILE_DETECTION_CONFIG['min_neighbors'] = min_neighbors
    SMILE_DETECTION_CONFIG['cooldown_time'] = cooldown_time
    logging.info(f"Настройки детекции улыбок обновлены: threshold={confidence_threshold}, neighbors={min_neighbors}, cooldown={cooldown_time}")

def start_interface(page: ft.Page):
    """Создаёт интерфейс Flet и подключает все кнопки."""
    page.title = "Face ID FENIX" # Facial Emotion Neural Identification compleX
    page.window_maximized = True
    page.window_maximizable = True
    page.window_resizable = True
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.padding = 15
    page.bgcolor = "#111524"
    
    # Функция для переключения полноэкранного режима
    def toggle_fullscreen(e):
        if page.window_full_screen:
            page.window_full_screen = False
            page.bgcolor = "#111524"
        else:
            page.window_full_screen = True
            page.bgcolor = "#000000"  # Черный фон в полноэкранном режиме
        page.update()
    
    # Кнопка полноэкранного режима
    fullscreen_button = ft.IconButton(
        icon=ft.icons.FULLSCREEN,
        tooltip="Полный экран",
        on_click=toggle_fullscreen,
        icon_color=ft.colors.WHITE70
    )

    # Инициализация моделей и загрузка дескрипторов
    try:
        initialize_models()
        load_face_descriptors()
    except Exception as e:
        logging.error(f"Ошибка инициализации моделей при старте интерфейса: {e}")

    # Подключаем Arduino (попытка)
    global arduino_serial
    try:
        arduino_serial = connect_to_arduino()
    except Exception as e:
        logging.error(f"Ошибка при подключении к Arduino: {e}")
        arduino_serial = None

    # GUI элементы
    image_area = ft.Image(src='None', width=1000, height=750, fit=ft.ImageFit.CONTAIN)
    match_area = ft.Image(src='None', width=400, height=350, fit=ft.ImageFit.CONTAIN)
    status_text = ft.Text(size=16, weight="bold", width=300, color=ft.colors.WHITE)
    info_text = ft.Text(
        "🔐 Двухфакторная аутентификация: лицо + улыбка",
        size=11,
        color=ft.colors.WHITE70
    )
    
    # Упрощенные настройки детекции улыбок
    smile_settings_text = ft.Text("Настройки детекции:", size=12, weight="bold", color=ft.colors.WHITE)
    
    confidence_slider = ft.Slider(
        min=0.3, max=0.9, value=SMILE_DETECTION_CONFIG['confidence_threshold'], 
        divisions=12, label="Точность: {value:.1f}",
        on_change=lambda e: None
    )
    
    cooldown_slider = ft.Slider(
        min=0.5, max=3.0, value=SMILE_DETECTION_CONFIG['cooldown_time'],
        divisions=25, label="Время: {value:.1f}с",
        on_change=lambda e: None
    )
    
    def apply_smile_settings(e):
        # Автоматически вычисляем min_neighbors на основе точности
        auto_neighbors = int(20 + (confidence_slider.value - 0.3) * 50)
        update_smile_settings(
            confidence_slider.value,
            auto_neighbors,
            cooldown_slider.value
        )
        status_text.value = f"Настройки: точность={confidence_slider.value:.1f}, время={cooldown_slider.value:.1f}с"
        status_text.color = ft.colors.GREEN
        page.update()
    
    apply_settings_button = ft.ElevatedButton(
        text="Применить", icon=ft.icons.SETTINGS, width=120, height=35,
        on_click=apply_smile_settings
    )

    exit_button = ft.ElevatedButton(
        text="Выход", icon=ft.icons.EXIT_TO_APP, width=150, height=50, visible=False,
        on_click=lambda e: exit_mode(page)
    )

    camera_dropdown = ft.Dropdown(options=[], label="Выберите камеру", width=200, visible=False)

    def update_camera_list():
        cams = list_available_cameras()
        options = [ft.dropdown.Option(text=f"Камера {i}", key=str(i)) for i in cams]
        camera_dropdown.options = options
        camera_dropdown.value = options[-1].key if options else None
        camera_dropdown.visible = True
        page.update()

    webcam_button = ft.ElevatedButton(
        text="Камера", icon=ft.icons.CAMERA, width=140, height=45,
        on_click=lambda e: [update_camera_list()]
    )

    start_cam_button = ft.ElevatedButton(
        text="Запуск", icon=ft.icons.PLAY_ARROW, width=140, height=45, visible=False,
        on_click=lambda e: start_webcam(
            page=page, image_area=image_area, match_area=match_area, status_text=status_text,
            exit_button=exit_button, camera_index=int(camera_dropdown.value) if camera_dropdown.value else 0
        )
    )

    rtsp_button = ft.ElevatedButton(
        text="IP-камера", icon=ft.icons.SETTINGS_ETHERNET, width=140, height=45,
        on_click=lambda e: start_rtsp_camera(page=page, image_area=image_area, match_area=match_area, 
                                             status_text=status_text, exit_button=exit_button)
    )

    image_button = ft.ElevatedButton(
        text="Фото", icon=ft.icons.IMAGE, width=140, height=45,
        on_click=lambda e: pick_image(page, image_area, match_area, status_text, exit_button)
    )

    video_button = ft.ElevatedButton(
        text="Видео", icon=ft.icons.VIDEO_FILE, width=140, height=45,
        on_click=lambda e: pick_video(page, image_area, match_area, status_text, exit_button)
    )

    def on_camera_selected(e):
        start_cam_button.visible = bool(camera_dropdown.value)
        page.update()
    camera_dropdown.on_change = on_camera_selected

    # Лог — live view (чтение file detection_log.txt)
    log_entries = ft.ListView(expand=True, spacing=3, auto_scroll=True)
    log_container = ft.Container(content=log_entries, width=350, height=600, padding=8, border=ft.border.all(1, ft.colors.GREY_800))

    def add_log_entry(msg):
        log_entries.controls.append(ft.Text(msg, size=12, color=ft.colors.WHITE, selectable=True))
        page.update()

    def update_logs_loop():
        pos = 0
        while True:
            try:
                if not os.path.exists(log_file):
                    time.sleep(1)
                    continue
                with open(log_file, 'r', encoding='utf-8') as lf:
                    lf.seek(pos)
                    lines = lf.readlines()
                    pos = lf.tell()
                    for ln in lines:
                        add_log_entry(ln.strip())
            except Exception as e:
                logging.error(f"update_logs_loop error: {e}")
            time.sleep(1)

    threading.Thread(target=update_logs_loop, daemon=True).start()

    page.overlay.append(file_picker)

    # Разметка
    left_column = ft.Column([
        ft.Row([ft.Text("FENIX Face ID", size=18, weight="bold", color="white"), fullscreen_button], 
               alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
        ft.Divider(height=8),
        info_text,
        ft.Divider(height=8),
        smile_settings_text,
        confidence_slider,
        cooldown_slider,
        apply_settings_button,
        ft.Divider(height=8),
        ft.Row([webcam_button, rtsp_button], spacing=5),
        camera_dropdown,
        start_cam_button,
        ft.Row([image_button, video_button], spacing=5),
        exit_button,
        ft.Divider(height=8),
        status_text,
        match_area
    ], spacing=8, width=320)

    middle_column = ft.Column([ft.Text("Видео с камеры", size=16, color=ft.colors.WHITE), 
                               image_area], alignment=ft.MainAxisAlignment.CENTER, expand=True)

    right_column = ft.Column([ft.Text("Журнал событий", size=16, color=ft.colors.WHITE), log_container], width=350)

    page.add(ft.Row([left_column, ft.VerticalDivider(width=15), middle_column, 
                     ft.VerticalDivider(width=15), right_column], spacing=15, expand=True))

# ---------------- Запуск приложения ----------------
if __name__ == "__main__":
    try:
        initialize_models()
        load_face_descriptors()
        arduino_serial = connect_to_arduino()
        if arduino_serial:
            print(f"Arduino подключен: {arduino_serial.port}")
        else:
            print("Arduino не найден (эмуляция).")
        ft.app(target=start_interface)
    except Exception as e:
        logging.exception(f"Ошибка запуска приложения: {e}")
    finally:
        try:
            if arduino_serial and arduino_serial.is_open:
                arduino_serial.close()
        except:
            pass
