import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy import stats
import numpy as np
import os

def load_data(filename):
    df = pd.read_csv(filename)
    return df

def clean_data(df):
    df.dropna(inplace=True)
    df['price'] = df['price'].astype(float)
    df['regDate'] = df['regDate'].astype(float)
    df['mileage'] = df['mileage'].astype(float)
    z_scores = np.abs(stats.zscore(df[['price', 'mileage']]))
    df_filtered = df[(z_scores < 1.96).all(axis=1)]
    return df_filtered

def fit_regression_and_rank(df):
    X_regDate = df[['regDate']]
    y_price = df['price']
    model_regDate = LinearRegression().fit(X_regDate, y_price)
    regDate_pred = model_regDate.predict(X_regDate)
    df['regDate_residuals'] = y_price - regDate_pred

    X_mileage = df[['mileage']]
    model_mileage = LinearRegression().fit(X_mileage, y_price)
    mileage_pred = model_mileage.predict(X_mileage)
    df['mileage_residuals'] = y_price - mileage_pred

    df['regDate_rank'] = df['regDate_residuals'].rank(ascending=True)
    df['mileage_rank'] = df['mileage_residuals'].rank(ascending=True)

    df['combined_rank'] = (df['regDate_rank'] + df['mileage_rank']) / 2
    df['final_rank'] = df['combined_rank'].rank()

    return df

def select_best_deals(df):
    best_deals = df.nsmallest(int(len(df) * 0.1), 'combined_rank')
    return best_deals

def visualize_data(df, best_deals):
    numerical_features = ['regDate', 'mileage']
    categorical_features = ['type', 'region', 'fuel', 'gearbox']
    all_features = numerical_features + categorical_features

    fig, axs = plt.subplots(3, 2, figsize=(15, 15))

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

    plt.tight_layout()
    static_folder = os.path.join(os.path.dirname(__file__), 'static')
    if not os.path.exists(static_folder):
        os.makedirs(static_folder)
    plot_file_path = os.path.join(static_folder, 'price_plots.png')
    plt.savefig(plot_file_path)
    plt.close()

def save_to_file(df, filename):
    df.to_csv(filename, index=False, sep='\t')

def main():
    filename = 'car_data.csv'
    df = load_data(filename)
    df = clean_data(df)
    df_ranked = fit_regression_and_rank(df)
    best_deals = select_best_deals(df_ranked)
    visualize_data(df, best_deals)
    print(df_ranked[['dealId', 'price', 'regDate', 'mileage', 'regDate_residuals', 'mileage_residuals', 'final_rank']].sort_values(by='final_rank').to_string(index=False))

if __name__ == "__main__":
    main()
