@echo off
chcp 65001 >nul
title FENIX - Uninstaller
cd /d "%~dp0"

echo =============================================================
echo             FENIX - Uninstaller
echo =============================================================
echo.
set /p confirm="Do you really want to uninstall FENIX and delete its files? (Y/N): "
if /I "%confirm%" NEQ "Y" (
    echo Aborted.
    pause
    exit /b
)

REM remove .venv
if exist ".venv" (
    echo Removing virtual environment...
    rmdir /s /q ".venv"
)

REM remove database and key
if exist "face_catalog.db" del /f /q "face_catalog.db"
if exist "secret.key" del /f /q "secret.key"

REM logs
if exist "detection_log.txt" del /f /q "detection_log.txt"
if exist "app.log" del /f /q "app.log"

echo.
set /p delall="Also delete face_bd and face_model directories? (Y/N): "
if /I "%delall%"=="Y" (
    if exist "face_bd" rmdir /s /q "face_bd"
    if exist "face_model" rmdir /s /q "face_model"
)

echo Uninstallation complete.
pause
