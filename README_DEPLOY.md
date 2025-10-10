# FENIX Face ID — установка и запуск (Windows)

## Быстрый старт (ZIP из GitHub)
1. Скачайте ZIP с проектом и распакуйте в папку без кириллицы и пробелов (например, C:\FENIX).
2. Убедитесь, что установлен Python 3.11 (x64) с галкой "Add to PATH".
   - Если Python 3.11 нет — запустите `installer_fenix.bat` и выберите пункт скачивания Python 3.11, либо поставьте вручную с python.org.
3. Запустите `installer_fenix2.bat` (one‑click):
   - создаст `.venv`;
   - установит зависимости из `requirements.txt`;
   - поставит `dlib` из локального `dlib-19.24.1-cp311-cp311-win_amd64.whl` (или через pip);
   - запустит `FENIX.py`.

## Состав инсталляторов
- `installer_fenix2.bat` — one‑click: Python 3.11 → .venv → зависимости → dlib → запуск.
- `installer_fenix.bat` — интерактивное меню: проверка/скачивание Python 3.11, создание .venv, установка зависимостей, запуск, удаление.
- `uninstaller_fenix.bat` — деинсталляция: удаляет `.venv`, логи, кэш, сборки; по желанию — `face_bd` и `face_model`.
- `run_fenix.bat` — запуск, если `.venv` уже создана.

## Частые вопросы
- Окно .bat сразу закрывается: запускайте .bat двойным кликом — в конце предусмотрен `pause`. Если падает раньше — скрипт покажет сообщение и выполнит `pause` на ошибках.
- Dlib не ставится из интернета: кладите файл `dlib-19.24.1-cp311-cp311-win_amd64.whl` в корень проекта (он включен).
- У меня другой Python: приложение поддерживает только Python 3.11. Поставьте 3.11 и добавьте в PATH.

## Ручная установка (альтернатива)
```
py -3.11 -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
pip install dlib-19.24.1-cp311-cp311-win_amd64.whl
python FENIX.py
```

## Удаление
Запустите `uninstaller_fenix.bat`. По запросу можно удалить папки `face_bd` и `face_model`.
