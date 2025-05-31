@echo off
pip install -r requirements.txt
pyinstaller build.py --noconfirm --clean
xcopy /E /I /Y "face_model" "dist\main\face_model"
xcopy /E /I /Y "face_bd" "dist\main\face_bd"
pause