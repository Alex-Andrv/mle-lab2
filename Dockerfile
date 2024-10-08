# Базовый образ с Python 3.9
FROM python:3.9-slim

# Устанавливаем рабочую директорию внутри контейнера

# Копируем файл с требованиями (зависимостями)
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем остальные файлы в контейнер
COPY . .

# Открываем порт 5000 для приложения Flask
EXPOSE 5001

# Запуск приложения
RUN ["python", "src/preprocess.py", "python", "src/app.py"]
