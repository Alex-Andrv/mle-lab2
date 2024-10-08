import os

from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

from trainer import Trainer

# Инициализация Flask приложения
app = Flask(__name__)


# Получаем абсолютный путь к текущему файлу
current_file_path = os.path.abspath(__file__)

# Получаем директорию, в которой находится текущий файл
current_directory = os.path.dirname(current_file_path)
# Загрузка модели
model = Trainer()
model.train()
model.save_model()
# Эндпоинт для предсказания
@app.route('/predict', methods=['POST'])
def predict():
    """
    Эндпоинт для предсказания класса пингвина.
    Ожидается, что входные данные будут переданы в формате JSON.
    Пример:
    {
        "Culmen Length (mm)": 39.1,
        "Culmen Depth (mm)": 18.7,
        "Flipper Length (mm)": 181,
        "Body Mass (g)": 3750,
        "Sex": "MALE",
        "Island": "Torgersen"
    }
    """
    try:
        # Получение данных от клиента
        data = request.get_json()
        print("chane")

        # Извлечение данных
        culmen_length = data['Culmen Length (mm)']
        culmen_depth = data['Culmen Depth (mm)']
        flipper_length = data['Flipper Length (mm)']
        body_mass = data['Body Mass (g)']
        sex = data['Sex']
        island = data['Island']

        df = pd.DataFrame({
            'Culmen Length (mm)': [culmen_length],
            'Culmen Depth (mm)': [culmen_depth],
            'Flipper Length (mm)': [flipper_length],
            'Body Mass (g)': [body_mass],
            'Sex': [sex],
            'Island': [island]
        })

        predicted_class = model.predict(df)

        # Возврат результата
        return jsonify({
            'predicted_species': predicted_class
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Запуск приложения
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)
