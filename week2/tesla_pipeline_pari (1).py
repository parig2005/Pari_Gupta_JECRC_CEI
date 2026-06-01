import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from prophet import Prophet

def _generate_mock_dataset(filepath):
    """Helper function to create fallback data if CSV is missing."""
    print(f"Dataset missing at '{filepath}'. Generating sample data...")
    q_dates = [f"Q{i%4+1} {2015 + i//4}" for i in range(40)]
    delivs = [10000 + i*5000 + (i%4)*2000 for i in range(40)]
    prods = [d + 1000 for d in delivs]
    pd.DataFrame({
        'Quarter': q_dates, 
        'Total_Deliveries': delivs, 
        'Total_Production': prods
    }).to_csv(filepath, index=False)

def standardize_dates(data, date_column):
    """Transform string quarters (e.g., 'Q1 2015') into pandas datetime objects."""
    df_clean = data.copy()
    
    # Map quarters to specific end-of-month dates
    mapping = {'Q1': '03-31', 'Q2': '06-30', 'Q3': '09-30', 'Q4': '12-31'}
    for q_str, date_str in mapping.items():
        df_clean[date_column] = df_clean[date_column].astype(str).str.replace(q_str, date_str)
        
    df_clean[date_column] = pd.to_datetime(df_clean[date_column])
    return df_clean.sort_values(date_column).reset_index(drop=True)

def engineer_features(data, target):
    """Generate historical lags and rolling averages for the ML model."""
    df_out = data.copy()
    
    # Create lag features
    df_out['prev_1q_lag'] = df_out[target].shift(1)
    df_out['prev_1y_lag'] = df_out[target].shift(4)
    
    # Create rolling window statistics
    df_out['mov_avg_2q'] = df_out[target].rolling(window=2).mean()
    df_out['mov_std_4q'] = df_out[target].rolling(window=4).std()
    
    # Time indices
    df_out['q_index'] = df_out['Quarter'].dt.quarter
    df_out['y_index'] = df_out['Quarter'].dt.year
    
    return df_out.dropna().reset_index(drop=True)

def build_xgboost_model(features_df, target_name, drop_features):
    """Train XGBoost using chronological cross-validation to prevent leakage."""
    X_features = features_df.drop(columns=[target_name] + drop_features)
    X_features = X_features.select_dtypes(include=[np.number])
    y_target = features_df[target_name]
    
    # Chronological 80/20 Train-Test Split
    train_size = int(len(features_df) * 0.8)
    X_tr, X_te = X_features.iloc[:train_size], X_features.iloc[train_size:]
    y_tr, y_te = y_target.iloc[:train_size], y_target.iloc[train_size:]
    
    xgb_model = XGBRegressor(objective='reg:squarederror', random_state=99)
    search_space = {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    }
    
    time_splitter = TimeSeriesSplit(n_splits=3)
    tuner = GridSearchCV(
        estimator=xgb_model, 
        param_grid=search_space, 
        cv=time_splitter, 
        scoring='neg_mean_absolute_error', 
        n_jobs=-1
    )
    
    tuner.fit(X_tr, y_tr)
    final_model = tuner.best_estimator_
    
    # Evaluate performance
    preds = final_model.predict(X_te)
    err_rmse = np.sqrt(mean_squared_error(y_te, preds))
    err_mae = mean_absolute_error(y_te, preds)
    
    print(f"[XGBoost Evaluation] RMSE: {err_rmse:.2f} | MAE: {err_mae:.2f}")
    return final_model

def run_prophet_forecast(data, date_col, target_col, periods_ahead=8):
    """Execute Prophet forecasting model for upcoming periods."""
    df_prophet = data[[date_col, target_col]].copy()
    df_prophet.columns = ['ds', 'y']
    
    m = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(df_prophet)
    
    future_df = m.make_future_dataframe(periods=periods_ahead, freq='Q')
    predictions = m.predict(future_df)
    
    # Visualize and save
    m.plot(predictions)
    plt.title('Vehicle Deliveries: Future Forecast')
    plt.savefig('forecast_results.png')
    print("=> Forecast plot saved as 'forecast_results.png'")
    return predictions

def main():
    csv_file = "tesla_deliveries.csv"
    
    if not os.path.exists(csv_file):
        _generate_mock_dataset(csv_file)
        
    print("Step 1/4: Formatting datetime columns...")
    raw_df = pd.read_csv(csv_file)
    processed_df = standardize_dates(raw_df, 'Quarter')
    
    print("Step 2/4: Generating temporal features...")
    ml_data = engineer_features(processed_df, 'Total_Deliveries')
    
    print("Step 3/4: Optimizing Regression Model...")
    best_xgb = build_xgboost_model(
        features_df=ml_data, 
        target_name='Total_Deliveries', 
        drop_features=['Quarter', 'Total_Production']
    )
    
    print("Step 4/4: Executing Time-Series Forecasting...")
    forecast_df = run_prophet_forecast(processed_df, 'Quarter', 'Total_Deliveries', periods_ahead=8)
    
    print("\n[SUCCESS] Pipeline execution finished.")

if __name__ == "__main__":
    main()
