import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split


class FeatureEngineering:
    """
    A class to engineer features for sales prediction and use LSTM for future sales forecasting.
    """

    def __init__(self, data):
        """
        Initialize with the dataset.
        :param data: The input DataFrame.
        """
        self.data = data.copy()
        self.scaler = MinMaxScaler()
        self.sales_scaler = MinMaxScaler()
        self.model = None

    def create_lag_features(self, lags=[1, 4, 12]):
        """
        Create lag features for sales data.
        :param lags: List of lag periods (e.g., 1 week, 4 weeks, 12 weeks).
        """
        print("Creating lag features...")
        for lag in lags:
            self.data[f'Sales_Lag_{lag}'] = (
                self.data.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(lag)
            )
        return self

    def create_rolling_features(self, windows=[4, 12]):
        """
        Create rolling mean and standard deviation features.
        :param windows: List of window sizes for rolling calculations.
        """
        print("Creating rolling statistics...")
        for window in windows:
            rolling_means = (
                self.data.groupby(['Store', 'Dept'])['Weekly_Sales']
                .rolling(window)
                .mean()
                .reset_index(level=[0, 1], drop=True)  # Align indices with the main DataFrame
            )
            self.data[f'Rolling_Mean_{window}'] = rolling_means

            rolling_stds = (
                self.data.groupby(['Store', 'Dept'])['Weekly_Sales']
                .rolling(window)
                .std()
                .reset_index(level=[0, 1], drop=True)  # Align indices with the main DataFrame
            )
            self.data[f'Rolling_Std_{window}'] = rolling_stds
        return self

    def create_seasonality_features(self):
        """
        Add seasonality features based on time data.
        """
        print("Creating seasonality features...")
        self.data['Is_End_of_Year'] = self.data['Month'].isin([11, 12]).astype(int)
        self.data['Week_of_Year_Sin'] = np.sin(2 * np.pi * self.data['Week'] / 52.0)
        self.data['Week_of_Year_Cos'] = np.cos(2 * np.pi * self.data['Week'] / 52.0)
        return self

    def create_store_features(self):
        """
        Add store-level aggregated features.
        """
        print("Creating store-level features...")
        store_aggregates = self.data.groupby('Store')['Weekly_Sales'].agg(['mean', 'sum']).reset_index()
        store_aggregates.columns = ['Store', 'Store_Average_Sales', 'Store_Total_Sales']
        self.data = self.data.merge(store_aggregates, on='Store', how='left')
        return self

    def create_trend_features(self):
        """
        Create trend features based on week-over-week sales differences.
        """
        print("Creating trend features...")
        self.data['Sales_Trend_1'] = self.data.groupby(['Store', 'Dept'])['Weekly_Sales'].diff()
        self.data['Sales_Trend_4'] = self.data.groupby(['Store', 'Dept'])['Weekly_Sales'].diff(4)
        return self

    def handle_missing_values(self):
        """
        Fill missing values with appropriate strategies.
        """
        print("Handling missing values...")
        self.data.fillna(0, inplace=True)
        return self

    def prepare_lstm_data(self, sequence_length=12):
        """
        Prepare data for LSTM training.
        :param sequence_length: Number of time steps to look back for prediction.
        """
        print("Preparing data for LSTM...")

        # Ensure missing values are handled before scaling
        self.handle_missing_values()

        # Normalize Weekly_Sales
        self.data['Weekly_Sales'] = self.sales_scaler.fit_transform(self.data[['Weekly_Sales']])

        # Normalize input features
        feature_columns = [col for col in self.data.columns if col not in ['Weekly_Sales', 'Date', 'Store', 'Dept']]
        scaled_features = self.scaler.fit_transform(self.data[feature_columns])

        # Create sequences for LSTM
        X, y = [], []
        for i in range(sequence_length, len(scaled_features)):
            X.append(scaled_features[i - sequence_length:i])  # Sequence of past `sequence_length` steps
            y.append(self.data['Weekly_Sales'].iloc[i])  # Corresponding target

        X, y = np.array(X), np.array(y)
        print(f"Shape of input data (X): {X.shape}")  # Expected: (num_samples, sequence_length, num_features)
        print(f"Shape of target data (y): {y.shape}")  # Expected: (num_samples,)
        return X, y

    def train_lstm(self, sequence_length=12, epochs=10, batch_size=32):
        """
        Train an LSTM model to predict sales.
        :param sequence_length: Number of time steps to look back for prediction.
        :param epochs: Number of training epochs.
        :param batch_size: Batch size for training.
        """
        print("Training LSTM model...")
        X, y = self.prepare_lstm_data(sequence_length)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = Sequential([
            Input(shape=(X_train.shape[1], X_train.shape[2])),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(64),
            Dropout(0.2),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping]
        )
        print("LSTM model training complete!")

    def predict_sales(self, future_periods=4):
        """
        Predict future sales using the trained LSTM model.
        :param future_periods: Number of weeks to predict into the future.
        """
        if not self.model:
            raise ValueError("Model not trained. Call `train_lstm` before predicting.")

        print(f"Predicting sales for {future_periods} weeks...")
        X, _ = self.prepare_lstm_data()
        predictions = self.model.predict(X[-future_periods:])
        original_scale_predictions = self.sales_scaler.inverse_transform(predictions)
        return original_scale_predictions

    def execute_all(self):
        """
        Execute all feature engineering steps in order.
        """
        print("Starting feature engineering...")
        self.create_lag_features()
        self.create_rolling_features()
        self.create_seasonality_features()
        self.create_store_features()
        self.create_trend_features()
        self.handle_missing_values()
        print("Feature engineering complete!")
        return self.data