import os
from flask import Flask, request, render_template, send_from_directory, url_for
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy import stats
import numpy as np
import json
import requests
import csv
import time

app = Flask(__name__, static_folder='static')

def fetch_car_data(page, headers, url, brand, model):
    querystring = {
        "filter": [
            f'{{"key":"make","type":"nested","values":["{brand}"]}}',
            f'{{"key":"models","type":"nested","values":["{model}"]}}'
        ],
        "page": str(page)
    }
    response = requests.get(url, headers=headers, params=querystring)
    if response.status_code == 200:
        return response.json().get("cars", [])
    else:
        print(f"Failed to fetch data for page {page}. Status code: {response.status_code}")
        return []

def clean_price(s):
    if s:
        try:
            s_clean = s.replace('kr', '').replace(' ', '').strip()
            price = float(s_clean)
            return price
        except ValueError:
            return None
    return None

def clean_reg_date(s):
    if s:
        try:
            reg_date = float(s)
            return reg_date
        except ValueError:
            return None
    return None

def clean_mileage(s):
    if s:
        try:
            mileage = float(s)
            return mileage
        except ValueError:
            return None
    return None

def process_car_data(cars):
    car_list = []
    for car in cars:
        price = clean_price(car.get("price", {}).get("amount"))
        reg_date = clean_reg_date(car.get("car", {}).get("regDate"))
        mileage = clean_mileage(car.get("car", {}).get("mileage"))
        car_info = {
            "dealId": car.get("dealId"),
            "heading": car.get("heading"),
            "type": car.get("seller", {}).get("type"),
            "price": price,
            "region": car.get("car", {}).get("location", {}).get("region"),
            "fuel": car.get("car", {}).get("fuel"),
            "gearbox": car.get("car", {}).get("gearbox"),
            "regDate": reg_date,
            "mileage": mileage,
            "link": car.get("link")  # Extract the link information
        }
        car_list.append(car_info)
    return car_list

def write_to_csv(car_list, csv_file, csv_columns):
    try:
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for car in car_list:
                writer.writerow(car)
        print(f"Data successfully written to {csv_file}")
    except IOError:
        print("I/O error")

def find_brand_and_model(model_name):
    url = "https://api.blocket.se/motor-search-service/v2/search/car/filters"
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Referer": "https://www.blocket.se/",
        "Origin": "https://www.blocket.se",
        "DNT": "1",
        "Connection": "keep-alive",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
        "Priority": "u=4",
        "TE": "trailers"
    }
    
    response = requests.get(url, headers=headers)
    data = json.loads(response.text)

    for attribute in data.get('attributes', []):
        if attribute['key'] == 'make':
            for value in attribute['values']:
                brand = value['value']
                if 'subGroup' in value:
                    for subgroup in value['subGroup']:
                        model = subgroup['value']
                        if model_name.lower() in (model.lower(), brand.lower()):
                            return brand, model
    return None, None

def load_data(filename):
    df = pd.read_csv(filename)
    return df

def clean_data(df):
    df.dropna(inplace=True)
    df['price'] = df['price'].astype(float)
    df['regDate'] = df['regDate'].astype(float)
    df['mileage'] = df['mileage'].astype(float)
    
    # Filter out extreme values in 'price' and 'mileage' columns using Z-score method
    z_scores = np.abs(stats.zscore(df[['price', 'mileage']]))
    df_filtered = df[(z_scores < 1.96).all(axis=1)]  # Filtering outliers for both columns

    return df_filtered

def fit_regression_and_rank(df):
    # Fit regression model for price vs regDate
    X_regDate = df[['regDate']]
    y_price = df['price']
    model_regDate = LinearRegression().fit(X_regDate, y_price)
    regDate_pred = model_regDate.predict(X_regDate)
    df['regDate_residuals'] = y_price - regDate_pred

    # Fit regression model for price vs mileage
    X_mileage = df[['mileage']]
    model_mileage = LinearRegression().fit(X_mileage, y_price)
    mileage_pred = model_mileage.predict(X_mileage)
    df['mileage_residuals'] = y_price - mileage_pred

    # Rank the cars based on the absolute residuals for both models
    df['regDate_rank'] = df['regDate_residuals'].rank(ascending=True)
    df['mileage_rank'] = df['mileage_residuals'].rank(ascending=True)

    # Combine the rankings with equal weight
    df['combined_rank'] = (df['regDate_rank'] + df['mileage_rank']) / 2
    df['final_rank'] = df['combined_rank'].rank()

    return df

def select_best_deals(df):
    # Select the top 10% best deals
    best_deals = df.nsmallest(int(len(df) * 0.1), 'final_rank')
    return best_deals

def visualize_data(df, best_deals):
    # Separate numerical and categorical features
    numerical_features = ['regDate', 'mileage']
    categorical_features = ['type', 'region', 'fuel', 'gearbox']

    # Combine all features for iteration
    all_features = numerical_features + categorical_features

    # Create a figure and multiple subplots
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))  # Adjusted grid to fit 2 columns wide and 3 rows long

    # Plot features against price
    for i, feature in enumerate(all_features):
        ax = axs[i//2, i%2]
        if feature in numerical_features:
            sns.regplot(x=feature, y='price', data=df, ax=ax, scatter_kws={'alpha':0.5}, label='Original Data')
            sns.scatterplot(x=feature, y='price', data=best_deals, ax=ax, color='red', label='Best Deals')
        else:
            sns.scatterplot(x=feature, y='price', data=df, ax=ax, alpha=0.5, label='Original Data')
            sns.scatterplot(x=feature, y='price', data=best_deals, ax=ax, color='red', label='Best Deals')
        ax.set_title(f'{feature} vs Price')
        ax.set_xlabel(feature)
        ax.set_ylabel('Price')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
        ax.grid(True)
        ax.legend()

    # Adjust the layout and show the plots
    plt.tight_layout()
    
    static_folder = os.path.join(os.path.dirname(__file__), 'static')
    if not os.path.exists(static_folder):
        os.makedirs(static_folder)

    plot_file_path = os.path.join(static_folder, 'price_plots.png')
    plt.savefig(plot_file_path)
    plt.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    model_name = request.form['model_name']
    brand, model = find_brand_and_model(model_name)
    if not brand or not model:
        return f"Model {model_name} not found."
    
    url = "https://api.blocket.se/motor-search-service/v2/search/car"
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Referer": "https://www.blocket.se/",
        "Authorization": "Bearer 21b1908bebb063b52254529affc4d461ce2218c0",
        "Origin": "https://www.blocket.se",
        "DNT": "1",
        "Connection": "keep-alive",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
        "Priority": "u=4",
        "TE": "trailers"
    }
    start_time = time.time()
    car_list = []
    for page in range(1, 11):
        cars = fetch_car_data(page, headers, url, brand, model)
        if cars:
            car_list.extend(process_car_data(cars))
    csv_file = "car_data.csv"
    csv_columns = ["dealId", "link", "heading", "type", "price", "region", "fuel", "gearbox", "regDate", "mileage"]
    write_to_csv(car_list, csv_file, csv_columns)
    
    df = load_data(csv_file)
    df = clean_data(df)
    
    df_ranked = fit_regression_and_rank(df)
    best_deals = select_best_deals(df_ranked)
    visualize_data(df, best_deals)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    
    return render_template('result.html', ranked_cars=df_ranked[['dealId', 'heading', 'price', 'regDate', 'mileage', 'final_rank', 'link']].sort_values(by='final_rank').to_dict(orient='records'))

@app.route('/static/<path:filename>')
def serve_static(filename):
    static_dir = os.path.join(os.getcwd(), 'static')
    return send_from_directory(static_dir, filename)

if __name__ == '__main__':
    app.run(debug=True)
