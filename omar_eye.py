import telebot
import cv2
import numpy as np
import keras
from keras.models import load_model

bot = telebot.TeleBot('TOKEN') 

model = keras.models.load_model('Augury/Omar_eye_1.h5')

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    try:
        # Получаем информацию о фотографии
        file_id = message.photo[-1].file_id
        file_info = bot.get_file(file_id)
        file = bot.download_file(file_info.file_path)
        with open('input_photo.jpg', 'wb') as photo:
            photo.write(file)

        # Загрузка и обработка фотографии перед подачей в нейросеть
        img = cv2.imread('input_photo.jpg')
        new_size = (128, 128)  # Новый размер
        resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)

        # Подача обработанного изображения в нейросеть и получение результата
        prediction = model.predict(np.array([resized_img]))
        bot.reply_to(message, str(np.argmax(prediction)))
        response_message = prediction
        # Отправляем результат обратно в чат
        bot.reply_to(message, response_message)

    except Exception as e:
        bot.reply_to(message, "Произошла ошибка при обработке фотографии.")

if __name__ == "__main__":
    bot.polling(none_stop=True)
