@echo off
chcp 65001 >nul
setlocal enableextensions enabledelayedexpansion

echo =============================================================
echo        FENIX - One-Click Installer (Windows x64)
echo =============================================================
echo.
cd /d "%~dp0"

REM --- 1) Проверка Python 3.11 ---
set "REQUIRED_MAJOR=3"
set "REQUIRED_MINOR=11"
set "PY_EXE="

REM Пытаемся py -3.11
py -3.11 --version >nul 2>&1
if not errorlevel 1 (
  set "PY_EXE=py -3.11"
) else (
  for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set CUR_PY_VER=%%v
  for /f "tokens=1,2 delims=." %%a in ("%CUR_PY_VER%") do (
    set CUR_MAJ=%%a
    set CUR_MIN=%%b
  )
  if "%CUR_MAJ%"=="%REQUIRED_MAJOR%" if "%CUR_MIN%"=="%REQUIRED_MINOR%" (
    set "PY_EXE=python"
  )
)

if "%PY_EXE%"=="" (
  echo [WARN] Python %REQUIRED_MAJOR%.%REQUIRED_MINOR% не найден в PATH.
  echo Установите Python 3.11 вручную (с включением Add to PATH),
  echo затем перезапустите этот установщик.
  echo Либо используйте installer_fenix.bat (пункт: скачать Python 3.11).
  pause
  exit /b 1
)

echo [OK] Использую интерпретатор: %PY_EXE%

REM --- 2) Создание/пересоздание .venv ---
if exist .venv (
  echo Удаляю старую .venv ...
  rmdir /s /q .venv
)
%PY_EXE% -m venv .venv
if errorlevel 1 (
  echo [ERROR] Не удалось создать .venv
  pause
  exit /b 1
)

REM --- 3) Активация и обновление pip ---
call .venv\Scripts\activate
python -m pip install --upgrade pip wheel setuptools

REM --- 4) Установка зависимостей ---
if exist requirements.txt (
  echo Устанавливаю зависимости из requirements.txt ...
  pip install -r requirements.txt
) else (
  echo requirements.txt не найден, устанавливаю основные пакеты...
  pip install opencv-python flet numpy Pillow scipy pyserial cmake
)
if errorlevel 1 (
  echo [ERROR] Ошибка при установке зависимостей
  pause
  exit /b 1
)

REM --- 5) Установка dlib из локального whl (для Python 3.11) ---
if exist dlib-19.24.1-cp311-cp311-win_amd64.whl (
  echo Устанавливаю локальный dlib wheel...
  pip install dlib-19.24.1-cp311-cp311-win_amd64.whl
) else (
  echo [WARN] Локальный dlib .whl не найден. Попробую pip install dlib (может быть долго).
  pip install dlib==19.24.1
)

REM --- 6) Запуск приложения ---
if exist FENIX.py (
  echo Запуск FENIX ...
  python FENIX.py
) else (
  echo [ERROR] FENIX.py не найден.
)

echo Готово.
pause
endlocal