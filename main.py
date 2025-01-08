# Import necessary libraries
import pandas as pd
from data_preprocessor import DataPreprocessor  # Import the preprocessing class
from data_visualizer import DataVisualizer  # Import the visualization class
from feature_engineering import FeatureEngineering  # Import the feature engineering class
from sklearn.preprocessing import MinMaxScaler


def main():
    """
    Main function to handle data preprocessing, feature engineering, LSTM modeling, and visualization.
    """
    # Step 1: Load the Dataset
    file_path = "retail_sales_dataset.csv"  # Replace with your dataset's actual path
    print("Step 1: Loading dataset...")
    try:
        data = pd.read_csv(file_path)
        print(f"Dataset loaded successfully! Shape: {data.shape}")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Check the path and try again.")
        return

    # Step 2: Preprocess the Dataset
    print("\nStep 2: Setting preprocessing parameters...")
    date_column = "Date"  # Column containing date information
    numerical_columns = [
        "Temperature", "Fuel_Price", "CPI", "Unemployment", "Size",
        "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"
    ]
    holiday_column = "IsHoliday"  # Column indicating if the week is a holiday
    sort_columns = ["Date", "Store", "Dept"]  # Columns to sort by

    # Initialize the DataPreprocessor class
    print("Step 3: Initializing the DataPreprocessor class...")
    preprocessor = DataPreprocessor(
        date_column=date_column,
        numerical_columns=numerical_columns,
        holiday_column=holiday_column,
        sort_columns=sort_columns
    )
    print("DataPreprocessor initialized successfully.")

    # Preprocess the dataset
    print("\nStep 4: Preprocessing data...")
    processed_data = preprocessor.preprocess(data)
    print("Data preprocessing complete!")

    # Display first few rows of the processed dataset for verification
    print("\nSample of the processed dataset:")
    print(processed_data.head())

    # Save the processed dataset
    preprocessed_output_path = "processed_sales_data.csv"
    print(f"\nStep 5: Saving the processed dataset to {preprocessed_output_path}...")
    processed_data.to_csv(preprocessed_output_path, index=False)
    print("Processed data saved successfully!")

    # Step 5a: Extract Scaling Metrics (debugging step)
    print("\nExtracting scaling metrics for Weekly_Sales...")
    try:
        sales_scaler = MinMaxScaler()
        sales_scaler.fit(processed_data[['Weekly_Sales']])

        # Min and Max values used for scaling
        min_value = sales_scaler.data_min_[0]
        max_value = sales_scaler.data_max_[0]
        print(f"Min value used for scaling Weekly_Sales: {min_value}")
        print(f"Max value used for scaling Weekly_Sales: {max_value}")

        # Check the range of scaled Weekly_Sales values
        scaled_min = processed_data['Weekly_Sales'].min()
        scaled_max = processed_data['Weekly_Sales'].max()
        print(f"Scaled Weekly_Sales range: {scaled_min} to {scaled_max}")
    except Exception as e:
        print(f"Error during scaling metrics extraction: {e}")
        return

    # Step 6: Apply Feature Engineering
    print("\nStep 6: Applying feature engineering...")
    feature_engineer = FeatureEngineering(processed_data)
    try:
        enhanced_data = feature_engineer.execute_all()
        print("\nFeature engineering complete!")
    except Exception as e:
        print(f"Error during feature engineering: {e}")
        return

    # Step 8: Train LSTM Model
    print("\nStep 8: Training LSTM model...")
    try:
        feature_engineer.train_lstm(sequence_length=12, epochs=5, batch_size=32)
    except Exception as e:
        print(f"Error during LSTM model training: {e}")
        return

    # Step 9: Predict Future Sales
    print("\nStep 9: Predicting future sales...")
    try:
        predictions = feature_engineer.predict_sales(future_periods=4)
        print("\nFuture sales predictions for 4 weeks:")
        print(predictions)

        # Debugging prediction output
        if predictions is None or len(predictions) == 0:
            print("Error: Predictions are empty.")
            return
        print(f"Predictions Shape: {predictions.shape}")
    except Exception as e:
        print(f"Error during predictions: {e}")
        return

    # Create corresponding future dates for predictions
    try:
        last_date = pd.to_datetime(enhanced_data['Date'].max())
        prediction_dates = [last_date + pd.Timedelta(weeks=i) for i in range(1, 5)]
        if len(prediction_dates) == 0:
            print("Error: Prediction dates are empty.")
            return
    except Exception as e:
        print(f"Error while creating prediction dates: {e}")
        return

    # Step 10: Visualize the Data
    print("\nStep 10: Visualizing the data...")
    try:
        visualizer = DataVisualizer(
            data=enhanced_data,
            predictions=predictions,
            prediction_periods=prediction_dates
        )
        visualizer.visualize_all(recent_weeks=8)  # Use the recent_weeks parameter to visualize last 8 weeks
    except Exception as e:
        print(f"Error during visualization: {e}")
        return

    print("\nAll tasks complete!")


# Execute the main function
if __name__ == "__main__":
    print("Starting the script...")
    main()
    print("Script execution complete!")