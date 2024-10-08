import configparser
import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sqlalchemy import create_engine
from logger import Logger

SHOW_LOG = True


class Trainer:
    def __init__(self, model=None, model_path="penguin_model.pkl", db_engine=None):
        """
        Инициализация тренера
        """
        self.model = model if model else LogisticRegression(max_iter=2000)
        self.scaler = StandardScaler()
        self.label_encoder_species = LabelEncoder()
        self.label_encoder_sex = LabelEncoder()
        self.model_path = model_path
        self.is_fitted = False

        logger = Logger(SHOW_LOG)
        self.log = logger.get_logger(__name__)
        self.config = configparser.ConfigParser()
        self.config.read("config.ini")
        self.database_config = configparser.ConfigParser()
        self.database_config.read("database.ini")
        db_config = self.database_config['DATABASE']
        # Установка подключения к базе данных
        db_url = f"postgresql://{db_config['USER']}:{db_config['PASSWORD']}@{db_config['HOST']}:{db_config['PORT']}/{db_config['NAME']}"
        self.db_engine = db_engine if db_engine else create_engine(db_url)

        # Загрузка данных из базы данных
        self.X_train = self.load_data_from_db("x_train")
        self.y_train = self.load_data_from_db("y_train")
        self.X_test = self.load_data_from_db("x_test")
        self.y_test = self.load_data_from_db("y_test")

        self.project_path = os.path.join(os.getcwd(), "experiments")
        self.model_path = os.path.join(self.project_path, "penguin_model.pkl")
        self.log.info("MultiModel is ready")

    def load_data_from_db(self, table_name):
        """
        Загрузка данных из базы данных.
        :param table_name: Имя таблицы в базе данных
        :return: pandas DataFrame с данными
        """
        try:
            query = f"SELECT * FROM {table_name}"
            data = pd.read_sql(query, self.db_engine)
            self.log.info(f"Данные успешно загружены из таблицы {table_name}")
            return data
        except Exception as e:
            self.log.error(f"Ошибка при загрузке данных из таблицы {table_name}: {e}")
            return None

    def preprocess_x(self, data, is_train=True):
        """
        Предобработка признаков X:
        - Оставляем только необходимые столбцы
        - Преобразование категориальных переменных
        - Масштабирование данных
        """
        data = data[['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)', 'Sex']]

        numeric_columns = ['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)']
        data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

        categorical_columns = ['Sex']
        for column in categorical_columns:
            data[column] = data[column].fillna(data[column].mode()[0])

        if is_train:
            data['Sex'] = self.label_encoder_sex.fit_transform(data['Sex'])
        else:
            data['Sex'] = self.label_encoder_sex.transform(data['Sex'])

        if is_train:
            X_scaled = self.scaler.fit_transform(data)
        else:
            X_scaled = self.scaler.transform(data)

        return X_scaled

    def preprocess_y(self, data):
        """
        Предобработка целевой переменной y:
        - Преобразование категориальных переменных
        """
        y = self.label_encoder_species.fit_transform(data['Species'])
        self.species_labels = self.label_encoder_species.classes_
        return y

    def train(self):
        """
        Обучение модели
        """
        X_train_scaled, y_train = self.preprocess_x(self.X_train), self.preprocess_y(self.y_train)

        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True

        X_test_scaled, y_test = self.preprocess_x(self.X_test), self.preprocess_y(self.y_test)

        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        self.log.info(f"Точность модели: {accuracy:.2f}")

        return accuracy

    def save_model(self):
        """
        Сохранение модели на диск
        """
        if self.is_fitted:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'label_encoder_species': self.label_encoder_species,
                'label_encoder_sex': self.label_encoder_sex
            }, self.model_path)
            self.log.info(f"Модель сохранена в {self.model_path}")
        else:
            self.log.info("Модель не обучена. Сохранение невозможно.")

    def load_model(self):
        """
        Загрузка модели с диска
        """
        if os.path.exists(self.model_path):
            data = joblib.load(self.model_path)
            self.model = data['model']
            self.scaler = data['scaler']
            self.label_encoder_species = data['label_encoder_species']
            self.label_encoder_sex = data['label_encoder_sex']
            self.is_fitted = True
            self.log.info(f"Модель загружена из {self.model_path}")
        else:
            self.log.info(f"Модель по пути {self.model_path} не найдена.")

    def predict(self, X):
        """
        Предсказание на новых данных
        """
        if os.path.exists(self.model_path):
            data = joblib.load(self.model_path)
            self.model = data['model']
            self.scaler = data['scaler']
            self.label_encoder_species = data['label_encoder_species']
            self.label_encoder_sex = data['label_encoder_sex']
            self.is_fitted = True
            self.log.info(f"Модель загружена из {self.model_path}")

            X_preprocessed = self.preprocess_x(X, is_train=False)

            return self.label_encoder_species.inverse_transform(self.model.predict(X_preprocessed))[0]
        else:
            self.log.info(f"Модель по пути {self.model_path} не найдена.")


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
    trainer.save_model()
