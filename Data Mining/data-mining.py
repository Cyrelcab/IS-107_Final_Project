import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.dates as mdates
import json
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

# Function to save results to JSON
def save_results_to_json(results, time_granularity):
    """
    Save the results dictionary to a JSON file based on time granularity.
    """
    file_path = f'{time_granularity}.json'
    with open(file_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)
    print(f"Results saved to {file_path}")


# Database connection function
def connect_to_db(user, password, host, port, db_name):
    """
    Establishes a connection to the PostgreSQL database.
    """
    try:
        engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{db_name}")
        print("Database connection successful!")
        return engine
    except Exception as e:
        print("Database connection failed:", e)
        return None


# Predictive analysis function with time granularity
def predictive_analysis(engine, time_granularity="monthly"):
    """
    Performs predictive analysis using linear regression for sales forecasting.
    Time granularity can be 'daily', 'weekly', or 'monthly'.
    """
    try:
        # Modified query to work with the new schema
        query = """
            SELECT 
                dt."date",
                dt.year,
                dt.month,
                dt.quarter,
                SUM(fs."Quantity") as total_items_sold,
                SUM(fs."total_amount") as total_sales,
                COUNT(DISTINCT fs."InvoiceNo") as number_of_transactions
            FROM Fact_Sales fs
            JOIN Dim_Time dt ON fs."date" = dt."date"
            GROUP BY dt."date", dt.year, dt.month, dt.quarter
            ORDER BY dt."date";
        """
        df = pd.read_sql(query, engine)
        
        # Convert date to datetime
        df["date"] = pd.to_datetime(df["date"])

        # Set the 'date' column as index for resampling
        df.set_index("date", inplace=True)

        # Aggregate the sales data based on the desired time granularity
        if time_granularity == "daily":
            df_resampled = df.resample('D').sum()  # Daily sales aggregation
        elif time_granularity == "weekly":
            df_resampled = df.resample('W').sum()  # Weekly sales aggregation
        elif time_granularity == "monthly":
            df_resampled = df.resample('M').sum()  # Monthly sales aggregation
        else:
            raise ValueError("Invalid time granularity. Choose from 'daily', 'weekly', or 'monthly'.")

        # Prepare data for regression
        df_resampled["month"] = df_resampled.index.month
        df_resampled["year"] = df_resampled.index.year
        X = df_resampled[["month", "year"]]  # Predictor variables (time-based features)
        y = df_resampled["total_sales"]  # Target variable

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict on test data
        y_pred = model.predict(X_test)

        # After model training, calculate total predicted sales
        total_predicted_sales = sum(y_pred)
        total_actual_sales = sum(y_test)
        print(f"Total Predicted Sales ({time_granularity}): ${total_predicted_sales:,.2f}")
        print(f"Total Actual Sales ({time_granularity}): ${total_actual_sales:,.2f}")
        print(f"Prediction Difference: ${(total_predicted_sales - total_actual_sales):,.2f}")

        # Perform sales forecasting using ARIMA
        forecast = forecast_sales(df_resampled, time_granularity)

        # Update results dictionary to include sales predictions instead of MSE
        results = {
            "time_granularity": time_granularity,
            "historical_sales": {
                str(date): float(sales)
                for date, sales in df_resampled["total_sales"].items()
            },
            "forecasted_sales": forecast.tolist(),
            "total_predicted_sales": float(total_predicted_sales),
            "total_actual_sales": float(total_actual_sales),
            "prediction_difference": float(total_predicted_sales - total_actual_sales)
        }

        # Save results to JSON with time granularity-specific filename
        save_results_to_json(results, time_granularity)
        
        return model
    except Exception as e:
        print("Error in predictive analysis:", e)
        return None


def forecast_sales(df_resampled, time_granularity):
    """
    Perform sales forecasting using ARIMA for future sales prediction.
    """
    try:
        print(f"Performing sales forecasting ({time_granularity.capitalize()})...")
        
        # ARIMA model: Adjust parameters to ensure stationarity
        sales_series = df_resampled['total_sales']
        model = ARIMA(sales_series, order=(1, 1, 1))  # Changed from (5,1,0) to (1,1,1) for better stationarity
        model_fit = model.fit()

        # Forecast sales for the next 12 periods (change according to granularity)
        forecast_periods = 12  # Number of periods to forecast
        forecast = model_fit.forecast(steps=forecast_periods)

        # Create forecast dates
        if time_granularity == "daily":
            forecast_dates = pd.date_range(df_resampled.index[-1], periods=forecast_periods+1, freq='D')[1:]
        elif time_granularity == "weekly":
            forecast_dates = pd.date_range(df_resampled.index[-1], periods=forecast_periods+1, freq='W')[1:]
        elif time_granularity == "monthly":
            forecast_dates = pd.date_range(df_resampled.index[-1], periods=forecast_periods+1, freq='M')[1:]

        # Plot and save the forecast figure
        plt.figure(figsize=(10, 6))
        plt.plot(df_resampled.index, df_resampled['total_sales'], label="Actual Sales", color='blue')
        plt.plot(forecast_dates, forecast, label="Forecasted Sales", color='red', linestyle='--')
        plt.title(f'Sales Forecasting ({time_granularity.capitalize()})')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(f'sales_forecast_{time_granularity}.png')
        plt.show()
        plt.close()  # Close the figure to free memory

        return forecast

    except Exception as e:
        print("Error in sales forecasting:", e)


def customer_segmentation(engine, n_clusters=3):
    """
    Performs customer segmentation using K-means clustering based on customer behavior.
    """
    try:
        # Modified query to work with the new schema
        query = """
            SELECT 
                dc."CustomerID" as customer_id,
                COUNT(DISTINCT fs."InvoiceNo") as total_orders,
                SUM(fs."total_amount") as total_spent,
                AVG(fs."total_amount") as avg_order_value,
                MAX(dt."date") - MIN(dt."date") as customer_lifetime
            FROM Fact_Sales fs
            JOIN Dim_Customer dc ON fs."CustomerID" = dc."CustomerID"
            JOIN Dim_Time dt ON fs."date" = dt."date"
            WHERE dc."CustomerID" IS NOT NULL
            GROUP BY dc."CustomerID"
        """
        df = pd.read_sql(query, engine)
        
        # Convert customer_lifetime to days if it's a timedelta
        if 'customer_lifetime' in df.columns:
            df['customer_lifetime'] = df['customer_lifetime'].dt.days
        
        # Prepare features for clustering
        features = ['total_orders', 'total_spent', 'avg_order_value', 'customer_lifetime']
        X = df[features]
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Remove the optimal cluster calculation since we're using fixed clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['segment'] = kmeans.fit_predict(X_scaled)
        
        # Map numeric segments to meaningful labels
        segment_labels = {
            0: 'Moderate-Value Customers',
            1: 'High-Value Customers',
            2: 'Low-Value Customers'
        }
        
        # Add segment labels to the dataframe
        df['segment_label'] = df['segment'].map(segment_labels)
        
        # Calculate segment characteristics with labels
        segment_analysis = df.groupby('segment_label').agg({
            'customer_id': 'count',
            'total_orders': 'mean',
            'total_spent': 'mean',
            'avg_order_value': 'mean',
            'customer_lifetime': 'mean'
        }).round(2)
        
        # Prepare results with labeled segments
        results = {
            "n_clusters": n_clusters,
            "segment_sizes": df['segment_label'].value_counts().to_dict(),
            "segment_characteristics": segment_analysis.to_dict(),
            "cluster_centers": {
                feature: centers.tolist() 
                for feature, centers in zip(features, kmeans.cluster_centers_.T)
            }
        }
        
        # Update visualization with labels
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(df['total_spent'], df['total_orders'], 
                            c=df['segment'], cmap='viridis')
        plt.title('Customer Segmentations by K-Means Clustering')
        plt.xlabel('Total Sales')
        plt.ylabel('Order Frequency')
        
        # Fix the legend
        handles = scatter.legend_elements()[0]
        plt.legend(handles, segment_labels.values(), title='Customer Segments')
        plt.tight_layout()
        
        # Save the figure
        plt.savefig('customer_segments.png')
        plt.show()
        plt.close()  # Close the figure to free memory
        
        return results
        
    except Exception as e:
        print("Error in customer segmentation:", e)
        return None


# Main function
def main():
    # Load environment variables
    load_dotenv()
    
    # Database credentials from environment variables
    user = os.getenv('DB_USER', 'postgres')
    password = os.getenv('DB_PASS')
    host = os.getenv('DB_HOST', 'localhost')
    port = os.getenv('DB_PORT', '5432')
    db_name = os.getenv('DB_NAME', 'retail_store_db')

    # Establish connection
    engine = connect_to_db(user, password, host, port, db_name)
    if not engine:
        return

    # Perform predictive analysis with different granularities
    print("Performing daily sales forecasting...")
    predictive_analysis(engine, time_granularity="daily")

    print("Performing weekly sales forecasting...")
    predictive_analysis(engine, time_granularity="weekly")

    print("Performing monthly sales forecasting...")
    predictive_analysis(engine, time_granularity="monthly")

    # Add customer segmentation analysis
    print("\nPerforming customer segmentation analysis...")
    segmentation_results = customer_segmentation(engine)
    if segmentation_results:
        save_results_to_json(segmentation_results, 'segmentation')  # Will save as segmentation.json


if __name__ == "__main__":
    main()
