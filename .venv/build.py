# build.py
import os
from PyInstaller.building.build_main import Analysis, EXE
from PyInstaller.building.datastruct import TOC

# Пути к вашим ресурсам
model_files = [
    ('face_model/shape_predictor_68_face_landmarks.dat', 'face_model'),
    ('face_model/dlib_face_recognition_resnet_model_v1.dat', 'face_model'),
]

# Автоматически добавляем все файлы из face_bd
face_db_files = [
    (os.path.join('face_bd', f), 'face_bd')
    for f in os.listdir('face_bd')
    if os.path.isfile(os.path.join('face_bd', f))
]

a = Analysis(
    ['main.py'],
    pathex=[os.getcwd()],
    binaries=[],
    datas=model_files + face_db_files,
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
)

exe = EXE(
    a,
    a.scripts,
    exclude_binaries=True,
    name='FaceRecognitionApp',
    debug=False,
    strip=False,
    upx=True,
    console=False,  # Измените на True если нужна консоль
    icon='icon.ico'  # Необязательно
)