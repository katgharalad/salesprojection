import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the processed data (replace with the actual path if necessary)
file_path = 'processed_sales_data.csv'

try:
    # Load the processed data
    processed_data = pd.read_csv(file_path)
    print("Processed data loaded successfully!")

    # Display a sample of the processed Weekly_Sales column
    print("\nSample of processed Weekly_Sales column:")
    print(processed_data['Weekly_Sales'].head())

    # Create a MinMaxScaler instance
    sales_scaler = MinMaxScaler()

    # Fit the scaler on the scaled Weekly_Sales column (assumes scaling was applied)
    sales_scaler.fit(processed_data[['Weekly_Sales']])

    # Extract Min and Max values used for scaling
    min_value = sales_scaler.data_min_[0]
    max_value = sales_scaler.data_max_[0]
    print(f"\nMin value used for scaling Weekly_Sales: {min_value}")
    print(f"Max value used for scaling Weekly_Sales: {max_value}")

    # Check the range of scaled Weekly_Sales values in the dataset
    scaled_min = processed_data['Weekly_Sales'].min()
    scaled_max = processed_data['Weekly_Sales'].max()
    print(f"\nScaled Weekly_Sales range: {scaled_min} to {scaled_max}")

except FileNotFoundError:
    print(f"Error: File not found at {file_path}. Ensure the file exists and the path is correct.")
except KeyError:
    print("Error: 'Weekly_Sales' column not found in the processed data.")