import logging
import os

import albumentations as A
import cv2

# Настройка журналирования
logging.basicConfig(filename='augmentation.log', level=logging.INFO)

# Создание функции аугментатора
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussianBlur(p=0.2),
    A.Rotate(limit=30, p=0.5),
])

# Вход/выход
input_path = "dataset/input"
output_path = "dataset/output"
augmentations_per_image = 100  # Количество аугментаций на каждое изображение

# Считать все файлы из директории входа
for filename in os.listdir(input_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(input_path, filename)

        # Загрузка изображения
        image = cv2.imread(img_path)

        # Создание директории для аугментированных изображений
        base_output_filename, file_extension = os.path.splitext(filename)
        output_dir = os.path.join(output_path, base_output_filename)
        os.makedirs(output_dir, exist_ok=True)

        for i in range(augmentations_per_image):
            # Применение аугментации
            transformed = transform(image=image)
            transformed_image = transformed["image"]

            # Генерация уникального имени файла для каждой аугментации
            output_filename = f"augmented_{base_output_filename}_{i}{file_extension}"

            # Путь к сохраненному аугментированному изображению
            output = os.path.join(output_dir, output_filename)
            cv2.imwrite(output, transformed_image)

            # Сообщение о том, что обработка изображения завершилась
            logging.info(f"Сохранено аугментированное изображение: {output}")
