import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import json

def create_figured_directories():
    """Create standardized directory structure for all predictions"""
    base_dir = 'figures'
    directories = {
        'customer_segmentation': ['charts', 'data'],
        'product_forecast': {
            'daily_forecast': ['charts', 'data'],
            'weekly_forecast': ['charts', 'data'],
            'monthly': ['charts', 'data']
        },
        'sales_forecast': ['charts', 'data']
    }

    for main_dir, subdirs in directories.items():
        if isinstance(subdirs, list):
            for subdir in subdirs:
                os.makedirs(os.path.join(base_dir, main_dir, subdir), exist_ok=True)
        else:
            for forecast_type, forecast_subdirs in subdirs.items():
                for subdir in forecast_subdirs:
                    os.makedirs(os.path.join(base_dir, main_dir, forecast_type, subdir), exist_ok=True)

    return base_dir

#function to load data and create connection to the database
def load_data():
    #Load the environment variables
    load_dotenv()

    # Get credentials from environment variables
    db_host = os.getenv('DB_HOST')
    db_port = os.getenv('DB_PORT')
    db_name = os.getenv('DB_NAME')
    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASS')

    # Create a database connection
    engine = create_engine(f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')

    #SQL Query
    query = """
    SELECT
        c."CustomerID",
        SUM(s."total_amount") AS total_sales,
        COUNT(s."InvoiceNo") AS order_frequency
    FROM
        Dim_Customer c
    JOIN
        Fact_Sales s ON c."CustomerID" = s."CustomerID"
    GROUP BY
        c."CustomerID"
    """

    return pd.read_sql(query, engine)

#function to process the datas
def process_data(df):
    #Load model and scaler
    k_means = joblib.load('customer_segmentation_model.pkl')
    scaler = joblib.load('scaler.pkl')

    #Scale data and predict clusters
    df_scaled = scaler.transform(df[['total_sales', 'order_frequency']])
    df['Cluster'] = k_means.predict(df_scaled)

    #Assign cluster labels
    cluster_centers = k_means.cluster_centers_
    cluster_labels = {
        0: 'High-Value Customers' if cluster_centers[0][0] > cluster_centers[1][0] else 'Low-Value Customers',
        1: 'Low-Value Customers' if cluster_centers[0][0] > cluster_centers[1][0] else 'High-Value Customers',
        2: 'Moderate-Value Customers' 
    }

    df['Cluster Label'] = df['Cluster'].map(cluster_labels)

    return df, cluster_labels

#function to create and save different visualizations
def create_save_visualizations(df, cluster_labels, base_dir):
    charts_dir = os.path.join(base_dir, 'customer_segmentation', 'charts')

    #scatter plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df['total_sales'], df['order_frequency'], c=df['Cluster'], cmap='viridis')
    handles, _ = scatter.legend_elements()
    legend_labels = [cluster_labels[i] for i in range(3)]
    plt.legend(handles, legend_labels, title='Customer Segments')
    plt.xlabel('Total Sales')
    plt.ylabel('Order Frequency')
    plt.title('Customer Segmentations by K-Means Clustering')
    plt.savefig(os.path.join(charts_dir, 'scatter_plot.png'))
    plt.close()

    #Bar Chart
    plt.figure(figsize=(10, 6))
    segment_counts = df['Cluster Label'].value_counts()
    plt.bar(segment_counts.index, segment_counts.values, color=['blue', 'green', 'orange'])
    plt.xlabel('Customer Segments')
    plt.ylabel('Number of Customers')
    plt.title('Customer Segmentation Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'bar_chart.png'))
    plt.close()

     # Pie Chart
    plt.figure(figsize=(10, 6))
    plt.pie(segment_counts, labels=segment_counts.index, autopct='%1.1f%%', colors=['blue', 'green', 'orange'])
    plt.title('Customer Segmentation Proportions')
    plt.savefig(os.path.join(charts_dir, 'pie_chart.png'))
    plt.close()

    return segment_counts

def save_data_json(df, cluster_labels, segment_counts, base_dir):
    data_dir = os.path.join(base_dir, 'customer_segmentation', 'data')
    
    # Scatter plot data
    scatter_data = {
        "data": [{
            'CustomerID': int(row['CustomerID']),
            'total_sales': float(row['total_sales']),
            'order_frequency': int(row['order_frequency']),
            'Cluster Label': row['Cluster Label']
        } for _, row in df[['CustomerID', 'total_sales', 'order_frequency', 'Cluster Label']].iterrows()],
        "chart_type": "scatter",
        "x_axis": "total_sales",
        "y_axis": "order_frequency",
        "legend": [cluster_labels[i] for i in range(3)]
    }
    
    # Bar chart data
    bar_data = {
        "data": {k: int(v) for k, v in segment_counts.to_dict().items()},
        "chart_type": "bar",
        "x_axis": "Customer Segments",
        "y_axis": "Number of Customers"
    }
    
    # Pie chart data
    pie_data = {
        "data": [{"label": label, "value": int(value)} 
                for label, value in zip(segment_counts.index, segment_counts.values)],
        "chart_type": "pie",
        "title": "Customer Segmentation Proportions"
    }
    
    # Save all data files
    with open(os.path.join(data_dir, 'scatter_plot_data.json'), 'w') as f:
        json.dump(scatter_data, f, indent=4)
    with open(os.path.join(data_dir, 'bar_chart_data.json'), 'w') as f:
        json.dump(bar_data, f, indent=4)
    with open(os.path.join(data_dir, 'pie_chart_data.json'), 'w') as f:
        json.dump(pie_data, f, indent=4)

def main():
    # Create directory structure
    base_dir = create_figured_directories()
    
    # Load and process data
    print("Loading data...")
    df = load_data()
    
    print("Processing data...")
    df, cluster_labels = process_data(df)
    
    # Create and save visualizations
    print("Creating visualizations...")
    segment_counts = create_save_visualizations(df, cluster_labels, base_dir)
    
    # Save data files
    print("Saving data files...")
    save_data_json(df, cluster_labels, segment_counts, base_dir)
    
    print(f"\nAnalysis complete. Results saved in '{base_dir}/customer_segmentation'")
    print("├── charts/")
    print("│   ├── scatter_plot.png")
    print("│   ├── bar_chart.png")
    print("│   └── pie_chart.png")
    print("└── data/")
    print("    ├── scatter_plot_data.json")
    print("    ├── bar_chart_data.json")
    print("    └── pie_chart_data.json")

if __name__ == "__main__":
    main() 