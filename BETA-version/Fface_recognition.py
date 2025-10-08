import os
import numpy as np
from PIL import Image
from scipy.spatial import distance
import dlib

# Подключаем предобученные модели
shape_predictor_path = 'face_model/shape_predictor_68_face_landmarks.dat'
face_rec_model_path = 'face_model/dlib_face_recognition_resnet_model_v1.dat'

# Проверяем наличие моделей
if not os.path.exists(shape_predictor_path):
    raise FileNotFoundError(f"Shape predictor model not found at {shape_predictor_path}")
if not os.path.exists(face_rec_model_path):
    raise FileNotFoundError(f"Face recognition model not found at {face_rec_model_path}")

sp = dlib.shape_predictor(shape_predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)
detector = dlib.get_frontal_face_detector()

def load_image(img_path):
    try:
        img = Image.open(img_path).convert('RGB')
        return np.array(img)
    except Exception as e:
        raise ValueError(f"Error opening image {img_path}: {e}")

def load_face_descriptors(base_path):
    face_descriptors = []
    faces = os.listdir(base_path)

    if not faces:
        raise ValueError(f"No faces found in the directory {base_path}")

    for face in faces:
        print(f"Processing file: {face}")
        img_path = os.path.join(base_path, face)

        try:
            img = load_image(img_path)
        except Exception as e:
            print(f"Error reading image {img_path}: {e}")
            continue

        dets = detector(img, 1)
        if len(dets) == 0:
            print(f"No faces detected in the image {img_path}")
            continue

        for k, d in enumerate(dets):
            shape = sp(img, d)
            face_descriptors.append(facerec.compute_face_descriptor(img, shape))
    return face_descriptors, faces

def compare_faces(img_or_frame, face_descriptors, faces, threshold=0.6):
    if isinstance(img_or_frame, str):
        img = load_image(img_or_frame)
    else:
        img = img_or_frame
    dets = detector(img, 1)
    if len(dets) == 0:
        raise ValueError(f"No faces detected in the image")

    for k, d in enumerate(dets):
        shape = sp(img, d)
        main_descriptor = facerec.compute_face_descriptor(img, shape)

    distances = [distance.euclidean(main_descriptor, fd) for fd in face_descriptors]
    min_dist = min(distances)
    closest_face_idx = distances.index(min_dist)

    return min_dist, faces[closest_face_idx], min_dist <= threshold

def get_face_landmarks(img_or_frame):
    if isinstance(img_or_frame, str):
        img = load_image(img_or_frame)
    else:
        img = img_or_frame
    dets = detector(img, 1)
    if len(dets) == 0:
        raise ValueError(f"No faces detected in the image")

    landmarks = []
    for k, d in enumerate(dets):
        shape = sp(img, d)
        landmarks.append([(point.x, point.y) for point in shape.parts()])
    return img, landmarks
