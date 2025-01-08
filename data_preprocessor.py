import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessor:
    def __init__(self, date_column=None, numerical_columns=None, holiday_column=None, sort_columns=None):
        """
        Initialize the preprocessor with configurable parameters.
        """
        self.date_column = date_column
        self.numerical_columns = numerical_columns or []
        self.holiday_column = holiday_column
        self.sort_columns = sort_columns or []
        self.scaler = MinMaxScaler()

    def process_date(self, data):
        """
        Process the date column into datetime format and add time features.
        """
        print("Processing date column...")
        try:
            if self.date_column:
                data[self.date_column] = pd.to_datetime(data[self.date_column])
                data['Year'] = data[self.date_column].dt.year
                data['Month'] = data[self.date_column].dt.month
                data['Week'] = data[self.date_column].dt.isocalendar().week
        except KeyError as e:
            print(f"KeyError in processing date column: {e}")
        except Exception as e:
            print(f"Unexpected error in processing date column: {e}")
        return data

    def normalize_columns(self, data):
        """
        Normalize specified numerical columns.
        """
        print("Normalizing numerical columns...")
        try:
            if self.numerical_columns:
                data[self.numerical_columns] = self.scaler.fit_transform(data[self.numerical_columns])
        except KeyError as e:
            print(f"KeyError in normalizing columns: {e}")
        except Exception as e:
            print(f"Unexpected error in normalizing columns: {e}")
        return data

    def encode_holiday(self, data):
        """
        Encode the holiday column as binary.
        """
        print("Encoding holiday column...")
        try:
            if self.holiday_column:
                data[self.holiday_column] = data[self.holiday_column].astype(int)
        except KeyError as e:
            print(f"KeyError in encoding holiday column: {e}")
        except Exception as e:
            print(f"Unexpected error in encoding holiday column: {e}")
        return data

    def add_seasonality_features(self, data):
        """
        Add seasonality features for months and weeks.
        """
        print("Adding seasonality features...")
        try:
            if 'Month' in data.columns:
                data['Month_Sin'] = np.sin(2 * np.pi * data['Month'] / 12.0)
                data['Month_Cos'] = np.cos(2 * np.pi * data['Month'] / 12.0)
            if 'Week' in data.columns:
                data['Week_Sin'] = np.sin(2 * np.pi * data['Week'] / 52.0)
                data['Week_Cos'] = np.cos(2 * np.pi * data['Week'] / 52.0)
        except KeyError as e:
            print(f"KeyError in adding seasonality features: {e}")
        except Exception as e:
            print(f"Unexpected error in adding seasonality features: {e}")
        return data

    def sort_data(self, data):
        """
        Sort the data by specified columns.
        """
        print("Sorting data...")
        try:
            if self.sort_columns:
                data.sort_values(by=self.sort_columns, inplace=True)
        except KeyError as e:
            print(f"KeyError in sorting data: {e}")
        except Exception as e:
            print(f"Unexpected error in sorting data: {e}")
        return data

    def preprocess(self, data):
        """
        Preprocess the data based on initialized configurations.
        """
        print("Starting preprocessing...")
        try:
            # Step 1: Process Date and Add Time Features
            data = self.process_date(data)

            # Step 2: Add Seasonality Features
            data = self.add_seasonality_features(data)

            # Step 3: Normalize Numerical Columns
            data = self.normalize_columns(data)

            # Step 4: Encode Holiday Column
            data = self.encode_holiday(data)

            # Step 5: Sort Data
            data = self.sort_data(data)
        except Exception as e:
            print(f"Unexpected error during preprocessing: {e}")
        print("Preprocessing complete!")
        return data