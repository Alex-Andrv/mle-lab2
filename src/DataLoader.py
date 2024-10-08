import pandas as pd
import configparser
from sqlalchemy import create_engine, text
import os


class DataLoader:
    def __init__(self, config_path='config.ini',  database_config_path='database.ini'):
        """
        Initialize the DataLoader with the path to the config file.
        """
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.database_config = configparser.ConfigParser()
        self.database_config.read(database_config_path)
        self.db_engine = None

        # Check if there is a database configuration
        if 'DATABASE' in self.database_config:
            db_config = self.database_config['DATABASE']
            db_url = f"postgresql://{db_config['USER']}:{db_config['PASSWORD']}@{db_config['HOST']}:{db_config['PORT']}/{db_config['NAME']}"
            self.db_engine = create_engine(db_url)
            print("Database connection initialized.")
        else:
            print("No database configuration found.")

    def load_csv_data(self, data_key):
        """
        Load CSV data based on the key from the config file.
        :param data_key: The section/key from the config file to load the data (e.g., 'DATA', 'SPLIT_DATA').
        :return: Pandas DataFrame of the loaded data.
        """
        try:
            data_path = self.config['DATA'][data_key]
            if os.path.exists(data_path):
                data = pd.read_csv(data_path)
                print(f"Data loaded from {data_path}")
                return data
            else:
                raise FileNotFoundError(f"The file {data_path} does not exist.")
        except KeyError:
            raise KeyError(f"{data_key} not found in the config file.")

    def load_split_data(self):
        """
        Load the training and testing data as specified in the config file under SPLIT_DATA.
        :return: X_train, y_train, X_test, y_test DataFrames.
        """
        try:
            X_train_path = self.config['SPLIT_DATA']['x_train']
            y_train_path = self.config['SPLIT_DATA']['y_train']
            X_test_path = self.config['SPLIT_DATA']['x_test']
            y_test_path = self.config['SPLIT_DATA']['y_test']

            X_train = pd.read_csv(X_train_path)
            y_train = pd.read_csv(y_train_path)
            X_test = pd.read_csv(X_test_path)
            y_test = pd.read_csv(y_test_path)

            print("Split data loaded successfully.")
            return X_train, y_train, X_test, y_test
        except KeyError as e:
            raise KeyError(f"Missing key in SPLIT_DATA section of config: {e}")

    def save_to_db(self, data, table_name, if_exists='replace'):
        """
        Save the data to PostgreSQL.
        :param data: DataFrame to save.
        :param table_name: Table name to save the data to.
        :param if_exists: Behavior if the table already exists (default is 'replace').
        """
        if self.db_engine:
            data.to_sql(table_name, self.db_engine, if_exists=if_exists, index=False)
            print(f"Data saved to PostgreSQL table {table_name}.")
        else:
            print("No database connection available. Data not saved.")

    def load_from_db(self, table_name):
        """
        Load data from PostgreSQL table.
        :param table_name: The name of the table to load the data from.
        :return: DataFrame of the loaded data.
        """
        if self.db_engine:
            data = pd.read_sql(f"SELECT * FROM {table_name}", self.db_engine)
            print(f"Data loaded from PostgreSQL table {table_name}.")
            return data
        else:
            print("No database connection available. Data not loaded.")
            return None

    def clear_table(self, table_name):
        """
        Clear all data from a given table if it exists.
        :param table_name: The name of the table to clear.
        """
        if self.db_engine:
            with self.db_engine.connect() as connection:
                # Check if the table exists in the database
                # Пример запроса на проверку существования таблицы
                query = text("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = :table_name)")
                result = connection.execute(query, {"table_name": "x_data"}).scalar()

                if result:
                    connection.execute(text(f"DELETE FROM {table_name};"))
                    print("Table exists")
                else:
                    print("Table does not exist")

        else:
            print("No database connection available. Table not cleared.")


# Example usage
if __name__ == '__main__':
    # Initialize DataLoader
    loader = DataLoader()

    try:
        loader.clear_table('x_data')
        loader.clear_table('y_data')
        loader.clear_table('x_train')
        loader.clear_table('y_train')
        loader.clear_table('x_test')
        loader.clear_table('y_test')
    except Exception as e:
        print(e)

    # Load raw data from a CSV file based on the 'DATA' section in the config file
    try:
        X_data = loader.load_csv_data('x_data')
        y_data = loader.load_csv_data('y_data')
    except Exception as e:
        print(e)

    # Load split training and testing data
    try:
        X_train, y_train, X_test, y_test = loader.load_split_data()
    except Exception as e:
        print(e)

    # Example to save processed data to the PostgreSQL database
    try:
        loader.save_to_db(X_train, 'x_train')
        loader.save_to_db(y_train, 'y_train')
        loader.save_to_db(X_test, 'x_test')
        loader.save_to_db(y_test, 'y_test')

        loader.save_to_db(X_data, 'x_data')
        loader.save_to_db(y_data, 'y_data')
    except Exception as e:
        print(e)

    # Example to load preprocessed data from the PostgreSQL database
    try:
        db_data = loader.load_from_db('train_data')
    except Exception as e:
        print(e)
