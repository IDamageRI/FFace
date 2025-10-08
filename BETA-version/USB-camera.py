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

# Настройки для Arduino
ARDUINO_PORT = None  # Будет автоматически определен
BAUD_RATE = 9600
ser = None

# Функция для инициализации соединения с Arduino
def init_arduino_connection():
    global ser, ARDUINO_PORT
    # Автоматическое определение порта Arduino
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
        print("Arduino not found. The app will run without LED control.")
        return False

# Функция для управления светодиодом
def control_led(state):
    """Управление светодиодом на Arduino"""
    if ser and ser.is_open:
        try:
            if state:
                ser.write(b'1')  # Включить светодиод
            else:
                ser.write(b'0')  # Выключить светодиод
        except serial.SerialException as e:
            print(f"Error controlling LED: {e}")

# ... (остальные импорты и функции остаются без изменений до функции compare_faces)

def compare_faces(frame, threshold=0.6):
    """Сравнение лиц и поиск совпадений"""
    dets = detector(frame, 0)

    if len(dets) == 0:
        control_led(False)  # Выключить светодиод, если лицо не обнаружено
        return None, None, False

    for d in dets:
        shape = sp(frame, d)
        main_descriptor = facerec.compute_face_descriptor(frame, shape)
        distances = [distance.euclidean(main_descriptor, fd) for fd in face_descriptors]

        min_dist = min(distances)
        closest_face_idx = distances.index(min_dist)
        is_match = min_dist <= threshold

        if is_match:
            control_led(True)  # Включить светодиод при совпадении
        else:
            control_led(False)  # Выключить светодиод при несовпадении

        return min_dist, faces[closest_face_idx], is_match

    control_led(False)  # Выключить светодиод, если лицо не распознано
    return None, None, False

# ... (остальной код остается без изменений до функции start_interface)

def start_interface(page: ft.Page):
    # Инициализация соединения с Arduino
    init_arduino_connection()
    
    page.title = "Распознавание лиц с Arduino"
    page.window_width = 1400
    page.window_height = 800
    page.window_resizable = False
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.padding = 20

    # ... (остальная часть функции start_interface без изменений)

def exit_mode(page):
    """Выход из текущего режима"""
    global is_running, cap
    is_running = False
    if cap is not None:
        cap.release()
        cap = None
    
    # Выключить светодиод при выходе
    control_led(False)
    
    for control in page.controls[:]:
        page.controls.remove(control)

    start_interface(page)
    page.update()

# ... (остальной код без изменений)

if __name__ == "__main__":
    try:
        sp = dlib.shape_predictor(shape_predictor_path)
        facerec = dlib.face_recognition_model_v1(face_rec_model_path)
        detector = dlib.get_frontal_face_detector()

        load_face_descriptors()

        ft.app(target=start_interface)
    except Exception as e:
        print(f"Ошибка запуска приложения: {e}")
        # Выключить светодиод при ошибке
        control_led(False)
        exit(1)