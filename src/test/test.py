import os
import sys
import unittest
import pandas as pd

sys.path.insert(1, os.path.join(os.getcwd(), "src"))

from trainer import Trainer

class TestTrainer(unittest.TestCase):

    def setUp(self):
        """Подготовка перед каждым тестом"""
        # Создаем экземпляр класса Trainer
        self.trainer = Trainer()

    def test_train(self):
        """Тестирование обучения модели"""
        accuracy = self.trainer.train()
        print(f"Точность модели после обучения: {accuracy}")
        self.assertTrue(accuracy > 0, "Точность должна быть больше 0")

    def test_save_and_load_model(self):
        """Тестирование сохранения и загрузки модели"""
        # Сохраняем модель
        self.trainer.train()
        self.trainer.save_model()
        # Проверяем, что файл модели существует
        self.assertTrue(os.path.exists(self.trainer.model_path), "Файл модели не был сохранен")

        # Загружаем модель
        self.trainer.load_model()
        self.assertTrue(self.trainer.is_fitted, "Модель не была загружена корректно")

    def test_predict(self):
        """Тестирование предсказания на новых данных"""
        # Загружаем модель (чтобы тестирование было воспроизводимо)
        self.trainer.train()
        self.trainer.save_model()
        self.trainer.load_model()

        # Данные для предсказания (из предоставленных данных)
        data_for_predict = pd.DataFrame({
            'Culmen Length (mm)': [39.5],
            'Culmen Depth (mm)': [17.4],
            'Flipper Length (mm)': [186],
            'Body Mass (g)': [3800],
            'Sex': ['FEMALE'],
            'Island': ['Torgersen']
        })

        # Предсказание на новых данных
        prediction = self.trainer.predict(data_for_predict)

        # Проверка того, что предсказание прошло успешно
        self.assertIsNotNone(prediction, "Предсказание не должно быть пустым")
        print(f"Предсказанный вид пингвина: {prediction}")

    def tearDown(self):
        """Удаление файлов или сброс после тестов"""
        if os.path.exists(self.trainer.model_path):
            os.remove(self.trainer.model_path)


if __name__ == "__main__":
    unittest.main()
