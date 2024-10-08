#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Импорт необходимых библиотек
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Шаг 1. Загрузка данных
data = pd.read_csv('data/raw/penguins_lter.csv')

# Шаг 2. Предобработка данных
# Выбираем только необходимые столбцы
data = data[['Species', 'Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)', 'Sex', 'Island']]

# Удаляем строки с пропущенными значениями
data.dropna(inplace=True)

# Преобразуем категориальные переменные в числовые значения
label_encoder_species = LabelEncoder()
data['Species'] = label_encoder_species.fit_transform(data['Species'])

label_encoder_sex = LabelEncoder()
data['Sex'] = label_encoder_sex.fit_transform(data['Sex'])

# Преобразуем категориальную переменную 'Island' с помощью one-hot encoding
data = pd.get_dummies(data, columns=['Island'], drop_first=True)

# Выбираем признаки для обучения
X = data.drop(columns='Species')
y = data['Species']

# Шаг 3. Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Шаг 4. Обучение модели
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Шаг 5. Предсказание и оценка качества модели
y_pred = model.predict(X_test)

# Оценка точности
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Отчет по классификации
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder_species.classes_))

# Шаг 6. Визуализация
# Визуализируем соотношение предсказаний и реальных данных
plt.figure(figsize=(8, 6))
plt.scatter(X_test['Culmen Length (mm)'], X_test['Culmen Depth (mm)'], c=y_pred, cmap='viridis', label="Predictions")
plt.xlabel('Culmen Length (mm)')
plt.ylabel('Culmen Depth (mm)')
plt.title('Logistic Regression Predictions')
plt.colorbar(label='Predicted Species')
plt.show()


# In[ ]:




