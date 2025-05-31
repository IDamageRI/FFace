import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk


def select_input():
    window = tk.Tk()
    window.title("Select Input")
    window.geometry("300x200")

    style = ttk.Style()
    style.theme_use('clam')

    def choose_image():
        window.destroy()
        process_image()

    def choose_video():
        window.destroy()
        process_video()

    def choose_webcam():
        window.destroy()
        process_webcam()

    image_button = ttk.Button(window, text="Process Image", command=choose_image)
    video_button = ttk.Button(window, text="Process Video", command=choose_video)
    webcam_button = ttk.Button(window, text="Process Webcam", command=choose_webcam)

    image_button.pack(pady=10)
    video_button.pack(pady=10)
    webcam_button.pack(pady=10)

    window.mainloop()


def process_image():
    file_path = get_file_path(filetypes=[("Image files", "*.jpg *.jpeg *.png *.svg *.webp"), ("All files", "*.*")])
    if file_path:
        # Добавьте код обработки изображения здесь
        print("Selected image:", file_path)


def process_video():
    file_path = get_file_path(filetypes=[("Video files", "*.avi *.mp4 *.mkv *.mov"), ("All files", "*.*")])
    if file_path:
        # Добавьте код обработки видео здесь
        print("Selected video:", file_path)


def process_webcam():
    # Добавьте код обработки веб-камеры здесь
    print("Processing webcam feed...")


def get_file_path(filetypes=None):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=filetypes)
    return file_path


if __name__ == "__main__":
    select_input()
