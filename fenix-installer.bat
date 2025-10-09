@echo off
chcp 65001 >nul
title 🔥 Установщик FENIX - Facial Emotion Neural Identification compleX
color 0A

REM === Автоматический переход в директорию скрипта ===
cd /d "%~dp0"

:: ==== ФУНКЦИЯ ПРОГРЕСС-БАР ====
:progress
setlocal enabledelayedexpansion
set /a total=%1
set /a delay=%2
set /a step=0
set "bar="
for /L %%i in (1,1,%total%) do (
    set /a step=%%i*100/%total%
    set "bar=!bar!#"
    <nul set /p=" [!step!%%] !bar! " 
    timeout /nobreak /t %delay% >nul
    cls
    echo =============================================================
    echo           🚀 УСТАНОВКА И ЗАПУСК СИСТЕМЫ FENIX
    echo =============================================================
    echo.
    echo Установка выполняется, подождите...
    echo [!step!%%] !bar!
)
endlocal
exit /b

echo =============================================================
echo           🚀 УСТАНОВКА И ЗАПУСК СИСТЕМЫ FENIX
echo =============================================================
echo.

REM --- Проверка Python ---
python --version >nul 2>&1
if errorlevel 1 (
    echo [❌] Python не найден! Установите Python 3.11 и добавьте его в PATH.
    pause
    exit /b
)

REM --- Проверка версии Python (3.11+) ---
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
for /f "tokens=1,2 delims=." %%a in ("%PYVER%") do (
    set MAJOR=%%a
    set MINOR=%%b
)
if %MAJOR% LSS 3 (
    echo [⚠️] Требуется Python 3.11 или новее! Обнаружено: %PYVER%.
    pause
    exit /b
)
if %MAJOR%==3 if %MINOR% LSS 11 (
    echo [⚠️] Требуется Python 3.11 или новее! Обнаружено: %PYVER%.
    pause
    exit /b
)

REM --- Удаление старой среды ---
if exist ".venv" (
    echo [🧹] Удаляю старую виртуальную среду...
    rmdir /s /q ".venv"
)

REM --- Создание новой среды ---
echo [⚙️] Создаю новую виртуальную среду .venv ...
call :progress 20 1
python -m venv .venv

REM --- Проверка создания ---
if not exist ".venv\Scripts\activate" (
    echo [❌] Ошибка: виртуальная среда не создана!
    pause
    exit /b
)

REM --- Активация среды ---
echo [▶️] Активирую среду...
call .venv\Scripts\activate

REM --- Обновление pip ---
echo [🔄] Обновляю pip...
call :progress 25 1
python -m pip install --upgrade pip

REM --- Установка зависимостей ---
if exist requirements.txt (
    echo [📦] Устанавливаю зависимости из requirements.txt...
    call :progress 30 1
    pip install -r requirements.txt
) else (
    echo [⚠️] Файл requirements.txt не найден — пропускаю установку.
)

REM --- Основные библиотеки (страховка) ---
echo [🔧] Проверяю и доустанавливаю основные пакеты...
call :progress 15 1
pip install opencv-python pillow flet numpy scipy pyserial

REM --- Установка dlib из локального файла ---
if exist dlib-19.24.1-cp311-cp311-win_amd64.whl (
    echo [🧠] Устанавливаю dlib из локального файла...
    call :progress 15 1
    pip install dlib-19.24.1-cp311-cp311-win_amd64.whl
) else (
    echo [⚠️] Файл dlib-19.24.1-cp311-cp311-win_amd64.whl не найден!
    echo Установите вручную при необходимости.
)

echo.
echo =============================================================
echo [✅] УСТАНОВКА СИСТЕМЫ FENIX ЗАВЕРШЕНА
echo =============================================================
echo.

REM --- Запуск проекта ---
echo [🚀] Запуск FENIX...
python FENIX.py

pause
