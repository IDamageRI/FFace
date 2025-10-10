@echo off
chcp 65001 >nul
title FENIX - Uninstaller
cd /d "%~dp0"

echo =============================================================
echo             FENIX - Uninstaller
echo =============================================================
echo.
set /p confirm="Удалить .venv, логи и временные файлы? (Y/N): "
if /I "%confirm%" NEQ "Y" (
    echo Отмена.
    pause
    exit /b
)

REM remove .venv
if exist ".venv" (
    echo Удаляю виртуальное окружение .venv ...
    rmdir /s /q ".venv"
)

REM remove caches and logs
echo Удаляю логи и временные файлы ...
if exist "__pycache__" rmdir /s /q "__pycache__"
if exist "app.log" del /f /q "app.log"
if exist "detection_log.txt" del /f /q "detection_log.txt"
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"
if exist "*.spec" del /f /q "*.spec"

set /p deldata="Также удалить каталоги данных (face_bd) и модели (face_model)? (Y/N): "
if /I "%deldata%"=="Y" (
    if exist "face_bd" rmdir /s /q "face_bd"
    if exist "face_model" rmdir /s /q "face_model"
)

echo Удаление завершено.
pause
