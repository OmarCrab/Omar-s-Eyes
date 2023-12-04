import time
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
import telebot
from telebot import types
import os
from datetime import datetime
import shutil

bot = telebot.TeleBot('YOUR_TOKEN')

model = YOLO("./runs/Omar-s-eye/weights/best.pt")

target_directory = './input_images'
os.makedirs(target_directory, exist_ok=True)

output_directory = './output_images'
os.makedirs(output_directory, exist_ok=True)

input_directory = './runs/detect'
os.makedirs(input_directory, exist_ok=True)

thread_pool = ThreadPoolExecutor()

@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.chat.id, 'Привет! Чем я могу помочь?')
    markup = types.ReplyKeyboardMarkup()
    btn1 = types.KeyboardButton('Прислать отчёт')
    markup.row(btn1)
    bot.send_message(message.chat.id, 'Чтобы прислать отчёт, нажмите "Прислать отчёт".', reply_markup=markup)

@bot.message_handler(func=lambda message: True)
def on_click(message):
    if message.text == 'Прислать отчёт':
        thread_pool.submit(send_report, message)

def send_report(message):
    try:
        current_date = datetime.now().date()
        bot.send_message(message.chat.id, f'Принято. Вот отчёт за {current_date}')
        move_images_and_remove_folders(input_directory)
        send_images_from_folder(message.chat.id, output_directory)
        time.sleep(1)
        clear_directory(target_directory)
        clear_directory(output_directory)
    except Exception as e:
        print(f"Ошибка: {e}")

def send_images_from_folder(chat_id, folder_path):
    try:
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'rb') as image:
                bot.send_photo(chat_id, image)
    except Exception as e:
        print(f"Ошибка при отправке изображений: {e}")

def clear_directory(directory_path):
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Ошибка при удалении файла {file_path}: {e}")

@bot.message_handler(content_types=['photo', 'text'])
def handle_messages(message):
    if message.content_type == 'photo':
        file_id = message.photo[-1].file_id
        path = bot.get_file(file_id)
        downloaded_file = bot.download_file(path.file_path)

        # Create a unique file name for the downloaded image
        extn = '.' + str(path.file_path).split('.')[-1]
        name = os.path.join(target_directory, f'photo_{message.photo[-1].file_id}{extn}')

        # Save the downloaded image to the specified directory
        with open(name, 'wb') as new_file:
            new_file.write(downloaded_file)
            thread_pool.submit(getting_check, name)

def getting_check(downloaded_file):
    try:
        model.predict(
            source=downloaded_file,
            show_labels=False,
            save=True,
            name='Augury',
            imgsz=1280,
            conf=0.2,
            show=False
        )
    except Exception as e:
        print(f"Ошибка: {e}")

def move_images_and_remove_folders(base_path):
    # Директория для перемещения изображений
    output_images_path = os.path.join(output_directory)
    os.makedirs(output_images_path, exist_ok=True)

    # Перенос изображений из папок Augury в директорию output_images
    for folder_name in os.listdir(base_path):
        if folder_name.startswith('Augury'):
            folder_path = os.path.join(base_path, folder_name)
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                shutil.move(file_path, output_images_path)

    # Удаление папок Augury
    for folder_name in os.listdir(base_path):
        if folder_name.startswith('Augury'):
            folder_path = os.path.join(base_path, folder_name)
            shutil.rmtree(folder_path)

    print("Завершено!")


if __name__ == '__main__':
    bot.polling(none_stop=True)
