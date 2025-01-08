import matplotlib.pyplot as plt  # Ensure that matplotlib is imported
import seaborn as sns
import pandas as pd

class DataVisualizer:
    """
    A class to handle visualization of sales data, including historical and predicted trends.
    """

    def __init__(self, data, predictions=None, prediction_periods=None):
        """
        Initialize the DataVisualizer class.
        :param data: Processed DataFrame with historical data.
        :param predictions: Predicted sales values (inverse-transformed to original scale).
        :param prediction_periods: Corresponding dates for predicted values.
        """
        self.data = data
        self.predictions = predictions
        self.prediction_periods = prediction_periods

    def plot_sales_over_time(self):
        """
        Plot historical and predicted sales over time for comparison.
        """
        plt.figure(figsize=(12, 6))  # Creating a plot

        # Plot historical sales
        historical_sales = self.data.groupby('Date')['Weekly_Sales'].sum()
        plt.plot(historical_sales.index, historical_sales.values, label="Historical Sales", color='blue', marker='o')

        # Plot predicted sales if available
        if self.predictions is not None and self.prediction_periods is not None:
            plt.plot(
                self.prediction_periods,
                self.predictions.flatten(),
                label="Predicted Sales",
                color='red',
                linestyle='--',
                marker='o'
            )

        plt.title("Sales Trends Over Time")
        plt.xlabel("Date")
        plt.ylabel("Total Weekly Sales")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_recent_vs_predicted(self, recent_weeks=8):
        """
        Compare recent historical sales with predicted sales for the next period.
        :param recent_weeks: Number of weeks to include in the recent historical data.
        """
        plt.figure(figsize=(12, 6))

        # Extract recent historical data
        recent_data = self.data.tail(recent_weeks)
        plt.plot(recent_data['Date'], recent_data['Weekly_Sales'], label="Recent Historical Sales", color='blue', marker='o')

        # Add predicted sales
        if self.predictions is not None and self.prediction_periods is not None:
            plt.plot(
                self.prediction_periods,
                self.predictions.flatten(),
                label="Predicted Sales",
                color='red',
                linestyle='--',
                marker='o'
            )

        plt.title(f"Recent ({recent_weeks} weeks) vs Predicted Sales")
        plt.xlabel("Date")
        plt.ylabel("Weekly Sales")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_seasonality(self):
        """
        Plot seasonality trends (Monthly sales).
        """
        plt.figure(figsize=(12, 6))
        monthly_sales = self.data.groupby('Month')['Weekly_Sales'].sum()
        sns.barplot(x=monthly_sales.index, y=monthly_sales.values, palette="viridis", dodge=False)
        plt.title("Monthly Sales Trends")
        plt.xlabel("Month")
        plt.ylabel("Total Sales")
        plt.show()

    def plot_correlation_matrix(self):
        """
        Plot correlation matrix for key numerical features.
        """
        numerical_columns = [
            "Weekly_Sales", "Temperature", "Fuel_Price", "CPI",
            "Unemployment", "Size"
        ]
        plt.figure(figsize=(10, 8))
        corr = self.data[numerical_columns].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix")
        plt.show()

    def plot_sales_by_store(self):
        """
        Plot total sales by store to identify high-performing stores.
        """
        plt.figure(figsize=(12, 6))
        store_sales = self.data.groupby('Store')['Weekly_Sales'].sum().sort_values(ascending=False)
        sns.barplot(x=store_sales.index, y=store_sales.values, palette="plasma", dodge=False)
        plt.title("Total Sales by Store")
        plt.xlabel("Store")
        plt.ylabel("Total Sales")
        plt.xticks(rotation=90)
        plt.show()

    def visualize_all(self, recent_weeks=8):
        """
        Visualize all relevant plots for sales data analysis.
        :param recent_weeks: Number of recent weeks to include in the recent vs predicted plot.
        """
        print("Visualizing historical and predicted sales trends...")
        self.plot_sales_over_time()
        self.plot_recent_vs_predicted(recent_weeks=recent_weeks)  # Use the recent_weeks parameter
        self.plot_seasonality()
        self.plot_correlation_matrix()
        self.plot_sales_by_store()