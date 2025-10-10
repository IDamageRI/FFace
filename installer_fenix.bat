@echo off
chcp 65001 >nul
title FENIX Installer - интерактивный установщик
color 0A

REM Перейти в папку скрипта
cd /d "%~dp0"

:: -----------------------
:: Функции/процедуры
:: -----------------------

:pause_and_return
echo.
echo Нажмите любую клавишу для возврата в меню...
pause >nul
goto menu

:check_python
echo =============================================================
echo Проверка установленного Python
echo =============================================================
echo.
python --version 2>nul >tmp_python_ver.txt
if errorlevel 1 (
    REM python not found, try py launcher
    del tmp_python_ver.txt >nul 2>&1
    py -3.11 --version 2>nul >tmp_python_ver.txt
    if errorlevel 1 (
        del tmp_python_ver.txt >nul 2>&1
        echo Python не найден в PATH, и py -3.11 не доступен.
        echo Если хотите, выберите пункт "Загрузить и установить Python 3.11".
        goto pause_and_return
    )
)

for /f "tokens=2 delims= " %%v in (tmp_python_ver.txt) do set PYVER=%%v
del tmp_python_ver.txt >nul 2>&1

echo Обнаружен Python: %PYVER%
for /f "tokens=1,2 delims=." %%a in ("%PYVER%") do (
    set PY_MAJOR=%%a
    set PY_MINOR=%%b
)
echo Версия: major=%PY_MAJOR% minor=%PY_MINOR%
if "%PY_MAJOR%"=="3" if "%PY_MINOR%"=="11" (
    echo [OK] Установлен Python 3.11 — подходит для FENIX.
) else (
    echo [WARN] Установлена несовместимая версия Python. Для корректной работы необходим Python 3.11.
    echo Можете установить 3.11 вручную или выбрать пункт "Загрузить и установить Python 3.11".
)
goto pause_and_return

:download_python_prompt
echo =============================================================
echo Скачивание и установка Python 3.11 (Windows x64)
echo =============================================================
echo.
echo ВНИМАНИЕ: установка потребует прав администратора. Продолжить? (Y/N)
set /p yn="> "
if /i "%yn%"=="Y" goto download_python
echo Операция отменена.
goto pause_and_return

:download_python
set PY_URL=https://www.python.org/ftp/python/3.11.0/python-3.11.0-amd64.exe
set PY_INSTALLER=python-3.11.0-amd64.exe

echo Скачивание установщика...
powershell -Command "Write-Host 'Downloading...'; Invoke-WebRequest -Uri '%PY_URL%' -OutFile '%PY_INSTALLER%'" 2>nul
if not exist "%PY_INSTALLER%" (
    echo Ошибка: не удалось скачать установщик %PY_INSTALLER%.
    echo Проверьте подключение к интернету и запустите скачивание вручную.
    goto pause_and_return
)

echo Запуск установки Python 3.11 (тихая установка)...
echo Если UAC появится — подтвердите запуск.
"%PY_INSTALLER%" /quiet InstallAllUsers=1 PrependPath=1 Include_test=0 SimpleInstall=1
if errorlevel 1 (
    echo Ошибка при установке Python. Убедитесь, что запустили файл с правами администратора.
    del "%PY_INSTALLER%" >nul 2>&1
    goto pause_and_return
)

echo Установка завершена. Удаляю установщик...
del "%PY_INSTALLER%" >nul 2>&1

echo Установка Python завершена. Возможно потребуется перезапуск системы, чтобы PATH обновился.
goto pause_and_return

:create_venv
echo =============================================================
echo Создание виртуальной среды .venv (Python 3.11)
echo =============================================================
echo.

REM Сначала попробуем использовать py -3.11 (если есть)
py -3.11 --version >nul 2>&1
if errorlevel 0 (
    echo Использую py -3.11 для создания venv...
    py -3.11 -m venv .venv
    if exist ".venv\Scripts\activate" (
        echo [OK] .venv создан с помощью py -3.11.
        goto venv_created
    ) else (
        echo [WARN] Создание .venv с помощью py -3.11 не удалось. Попробую python.
    )
)

REM Если py -3.11 нет или не получилось, проверим python
python --version >tmp_python_ver.txt 2>&1
if errorlevel 1 (
    echo Python не найден. Выберите сначала установку Python 3.11 в меню.
    del tmp_python_ver.txt >nul 2>&1
    goto pause_and_return
)
for /f "tokens=2 delims= " %%v in (tmp_python_ver.txt) do set PYVER=%%v
del tmp_python_ver.txt >nul 2>&1
for /f "tokens=1,2 delims=." %%a in ("%PYVER%") do (
    set PY_MAJOR=%%a
    set PY_MINOR=%%b
)
if "%PY_MAJOR%"=="3" if "%PY_MINOR%"=="11" (
    echo Использую python (версия %PYVER%) для создания venv...
    python -m venv .venv
    if exist ".venv\Scripts\activate" (
        echo [OK] .venv создан.
        goto venv_created
    ) else (
        echo [ERROR] Не удалось создать .venv с python.
        goto pause_and_return
    )
) else (
    echo Установлена несовместимая версия Python (%PYVER%). Нужен Python 3.11.
    goto pause_and_return
)

:venv_created
echo.
echo Виртуальная среда .venv успешно создана.
goto pause_and_return

:install_deps
cls
echo =============================================================
echo Установка зависимостей в .venv
echo =============================================================
echo.

if not exist ".venv\Scripts\activate" (
    echo [⚠️] .venv не найден. Сначала создайте виртуальную среду (пункт 1).
    pause
    goto menu
)

echo [ℹ️] Активирую виртуальное окружение...
call .venv\Scripts\activate

echo [⬆️] Обновляю pip...
python -m pip install --upgrade pip

echo [📦] Установка зависимостей...

if exist requirements.txt (
    echo [📃] Установка из requirements.txt ...
    pip install -r requirements.txt
) else (
    echo [⚙️] requirements.txt не найден. Устанавливаю основные пакеты...
    pip install opencv-python pillow flet numpy scipy pyserial
)

if exist dlib-19.24.1-cp311-cp311-win_amd64.whl (
    echo [🧠] Устанавливаю локальный dlib wheel...
    pip install dlib-19.24.1-cp311-cp311-win_amd64.whl
) else (
    echo [⚠️] Локальный dlib wheel не найден. Установите dlib вручную при необходимости.
)

echo [✅] Установка зависимостей завершена.
pause
goto menu

:run_fenix
echo =============================================================
echo Запуск FENIX (в активной .venv, если она есть)
echo =============================================================
echo.
if exist ".venv\Scripts\activate" (
    call .venv\Scripts\activate
)
if exist "FENIX.py" (
    python FENIX.py
) else (
    echo Файл FENIX.py не найден в текущей папке: %cd%
)
goto pause_and_return

:uninstall_prompt
echo =============================================================
echo УДАЛЕНИЕ FENIX - Uninstall
echo =============================================================
echo.
echo Вы уверены, что хотите удалить .venv, БД, ключ и логи? (Y/N)
set /p ans="> "
if /i "%ans%"=="Y" goto uninstall_do
echo Операция отменена.
goto pause_and_return

:uninstall_do
echo Удаляю .venv, face_catalog.db, secret.key, detection_log.txt, app.log (если есть)...
if exist ".venv" rmdir /s /q ".venv"
if exist "face_catalog.db" del /f /q "face_catalog.db"
if exist "secret.key" del /f /q "secret.key"
if exist "detection_log.txt" del /f /q "detection_log.txt"
if exist "app.log" del /f /q "app.log"
echo Удаление завершено.
goto pause_and_return

:menu
cls
echo =============================================================
echo            FENIX - Interactive Installer
echo =============================================================
echo.
echo 1. Проверить установленный Python (версию)
echo 2. Создать виртуальную среду .venv (требует Python 3.11)
echo 3. Скачать и установить Python 3.11 (опционально)
echo 4. Установить зависимости (pip install -r requirements.txt) в .venv
echo 5. Запустить FENIX
echo 6. Удалить (uninstall) - .venv, БД, ключи, логи
echo 0. Выход
echo.
set /p choice="Выберите пункт (0-6): "

if "%choice%"=="1" goto check_python
if "%choice%"=="2" goto create_venv_choice
if "%choice%"=="3" goto download_python_prompt
if "%choice%"=="4" goto install_deps
if "%choice%"=="5" goto run_fenix
if "%choice%"=="6" goto uninstall_prompt
if "%choice%"=="0" goto exit_script

echo Неверный выбор, попробуйте снова.
goto menu

:create_venv_choice
echo Вы уверены, что хотите создать/пересоздать .venv? (существующая .venv будет удалена) (Y/N)
set /p c="> "
if /i "%c%"=="Y" (
    goto create_venv_confirm
) else (
    goto menu
)

:create_venv_confirm
if exist ".venv" (
    echo Удаляю старую .venv ...
    rmdir /s /q ".venv"
)
goto create_venv

:exit_script
echo Выход. Спасибо!
pause >nul
exit /b