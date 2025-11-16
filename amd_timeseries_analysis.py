import json
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import acf
import warnings
warnings.filterwarnings('ignore')

# Constants
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10
MINUTES_IN_FUTURE = 90

def load_amd_data(filepath: str) -> pd.DataFrame:
    """Load AMD dataset and prepare for analysis."""
    df = pd.read_csv(filepath)
    # Convert time column to datetime
    df['time_entry_ts'] = pd.to_datetime(df['time_entry_ts'])
    # Sort by time
    df = df.sort_values('time_entry_ts').reset_index(drop=True)
    # Rename columns for consistency with timeseries.py
    df.rename(columns={
        'high_price': 'HIGH',
        'low_price': 'LOW',
        'close_price': 'CLOSE',
        'open_price': 'OPEN'
    }, inplace=True)
    return df

def local_TRANGE(df: pd.DataFrame) -> pd.Series:
    """
    Calculate True Range: max(high−low, |high−prev_close|, |low−prev_close|)
    Based on the implementation from timeseries.py
    """
    high, low, close = df["HIGH"], df["LOW"], df["CLOSE"]
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR) using exponential moving average.
    """
    tr = local_TRANGE(df)
    # Use exponential moving average for ATR calculation
    atr = tr.ewm(span=period, adjust=False).mean()
    return atr

def create_both_atr_columns(df: pd.DataFrame, minutes_future: int = 90) -> pd.DataFrame:
    """
    Create both atr_0 (current ATR) and atr_{minutes_future} columns.
    This ensures both targets are on the same dataset with the same rows.
    """
    # Calculate ATR first
    df['ATR'] = calculate_atr(df)

    # Create current ATR column (atr_0)
    df['atr_0'] = df['ATR'].copy()

    # Create future ATR column by shifting ATR values backwards
    # This gives us "what will the ATR be in X minutes"
    df[f'atr_{minutes_future}'] = df['ATR'].shift(-minutes_future)

    # Remove rows with NaN values in EITHER column
    # This ensures both atr_0 and atr_90 use the exact same rows
    df_clean = df.dropna(subset=['atr_0', f'atr_{minutes_future}']).copy()

    print(f"Original rows: {len(df)}")
    print(f"Rows after creating atr_0 and atr_{minutes_future}: {len(df_clean)}")
    print(f"Rows removed: {len(df) - len(df_clean)}")

    return df_clean

def split_data(df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.20) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform a 70-20-10 split sorted by time_entry_ts ascending.
    """
    # Ensure data is sorted by time
    df = df.sort_values('time_entry_ts').reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print(f"\nData Split Summary:")
    print(f"Total samples: {n}")
    print(f"Train samples: {len(train_df)} ({len(train_df)/n*100:.1f}%)")
    print(f"Validation samples: {len(val_df)} ({len(val_df)/n*100:.1f}%)")
    print(f"Test samples: {len(test_df)} ({len(test_df)/n*100:.1f}%)")

    return train_df, val_df, test_df

def extract_seasonality_acf(series: pd.Series, max_lag: int = None) -> Tuple[np.ndarray, int]:
    """
    Implement ACF to extract seasonality from the series.
    Returns ACF values and detected seasonal period.
    """
    if max_lag is None:
        max_lag = min(len(series) // 4, 500)  # Reasonable default

    # Calculate ACF
    acf_values, confint = acf(series.dropna(), nlags=max_lag, alpha=0.05, fft=True)

    # Find significant peaks in ACF (excluding lag 0)
    # Look for the first significant peak after lag 0
    seasonal_period = None
    threshold = 1.96 / np.sqrt(len(series))

    for lag in range(1, len(acf_values)):
        if acf_values[lag] > threshold:
            # Check if this is a peak
            if lag > 1 and lag < len(acf_values) - 1:
                if acf_values[lag] > acf_values[lag-1] and acf_values[lag] > acf_values[lag+1]:
                    seasonal_period = lag
                    break

    if seasonal_period is None:
        # If no clear peak, look for the maximum ACF value after lag 0
        seasonal_period = np.argmax(acf_values[1:]) + 1

    print(f"\nACF Analysis:")
    print(f"Detected seasonal period: {seasonal_period}")
    print(f"ACF value at seasonal period: {acf_values[seasonal_period]:.4f}")

    return acf_values, seasonal_period

def create_fourier_features(data: pd.DataFrame, period: int, n_fourier_pairs: int) -> pd.DataFrame:
    """
    Create Fourier features for modeling seasonality.
    Runs 1 to n_fourier_pairs K Fourier pairs.
    """
    features = pd.DataFrame(index=data.index)

    # Create time index
    t = np.arange(len(data))

    # Create Fourier features
    for k in range(1, n_fourier_pairs + 1):
        omega = 2 * np.pi * k / period
        features[f'sin_{k}'] = np.sin(omega * t)
        features[f'cos_{k}'] = np.cos(omega * t)

    # Add linear trend (slope extraction)
    features['trend'] = t

    return features

def extract_trend_slope(series: pd.Series) -> Tuple[float, float]:
    """
    Extract linear trend slope from the series.
    """
    t = np.arange(len(series))
    slope, intercept, r_value, p_value, std_err = stats.linregress(t, series)

    print(f"\nTrend Analysis:")
    print(f"Slope: {slope:.6f}")
    print(f"Intercept: {intercept:.4f}")
    print(f"R-squared: {r_value**2:.4f}")

    return slope, intercept

def implement_naive_baseline(train_series: pd.Series, val_series: pd.Series, test_series: pd.Series,
                            horizon: int = 1) -> Dict[str, float]:
    """
    Implement naive baseline model (persistence model).
    The prediction is simply the value from 'horizon' steps ago.
    """
    # Validation predictions
    val_pred = train_series.iloc[-horizon:].values.tolist() + val_series.iloc[:-horizon].values.tolist()
    val_pred = pd.Series(val_pred[:len(val_series)], index=val_series.index)

    # Test predictions
    test_pred = val_series.iloc[-horizon:].values.tolist() + test_series.iloc[:-horizon].values.tolist()
    test_pred = pd.Series(test_pred[:len(test_series)], index=test_series.index)

    # Calculate metrics
    val_mae = mean_absolute_error(val_series, val_pred)
    val_mse = mean_squared_error(val_series, val_pred)
    val_rmse = np.sqrt(val_mse)

    test_mae = mean_absolute_error(test_series, test_pred)
    test_mse = mean_squared_error(test_series, test_pred)
    test_rmse = np.sqrt(test_mse)

    results = {
        'val_mae': val_mae,
        'val_mse': val_mse,
        'val_rmse': val_rmse,
        'test_mae': test_mae,
        'test_mse': test_mse,
        'test_rmse': test_rmse
    }

    print(f"\nNaive Baseline Results (horizon={horizon}):")
    print(f"Validation - MAE: {val_mae:.4f}, MSE: {val_mse:.4f}, RMSE: {val_rmse:.4f}")
    print(f"Test - MAE: {test_mae:.4f}, MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}")

    return results

def implement_ar_fourier_model(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                               target_col: str, period: int, max_k: int = 8, lag: int = 1) -> Dict:
    """
    Implement AR-Fourier model with hyperparameter tuning.
    Tests K from 1 to max_k and selects the best based on validation performance.
    """
    results = []
    best_k = 1
    best_val_rmse = float('inf')
    best_model = None

    for k in range(1, max_k + 1):
        # Create features for each dataset
        train_fourier = create_fourier_features(train_df, period, k)
        val_fourier = create_fourier_features(val_df, period, k)
        test_fourier = create_fourier_features(test_df, period, k)

        # Add AR component (lagged target)
        train_fourier[f'lag_{lag}'] = train_df[target_col].shift(lag)
        val_fourier[f'lag_{lag}'] = val_df[target_col].shift(lag)
        test_fourier[f'lag_{lag}'] = test_df[target_col].shift(lag)

        # For validation, use the last value from train as the first lag
        if lag <= len(train_df):
            val_fourier[f'lag_{lag}'].iloc[:lag] = train_df[target_col].iloc[-lag:].values

        # For test, use the last value from val as the first lag
        if lag <= len(val_df):
            test_fourier[f'lag_{lag}'].iloc[:lag] = val_df[target_col].iloc[-lag:].values

        # Remove NaN rows from training
        train_fourier = train_fourier.dropna()
        train_y = train_df[target_col].loc[train_fourier.index]

        # Train model
        model = Ridge(alpha=1.0)
        model.fit(train_fourier, train_y)

        # Validate
        val_pred = model.predict(val_fourier)
        val_y = val_df[target_col]

        val_mae = mean_absolute_error(val_y, val_pred)
        val_mse = mean_squared_error(val_y, val_pred)
        val_rmse = np.sqrt(val_mse)

        # Test
        test_pred = model.predict(test_fourier)
        test_y = test_df[target_col]

        test_mae = mean_absolute_error(test_y, test_pred)
        test_mse = mean_squared_error(test_y, test_pred)
        test_rmse = np.sqrt(test_mse)

        results.append({
            'K': k,
            'val_mae': val_mae,
            'val_mse': val_mse,
            'val_rmse': val_rmse,
            'test_mae': test_mae,
            'test_mse': test_mse,
            'test_rmse': test_rmse
        })

        # Keep track of best model
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_k = k
            best_model = model

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    print(f"\nAR-Fourier Model Results:")
    print(f"Best K (by validation RMSE): {best_k}")
    print("\nAll K results:")
    print(results_df)

    best_results = results_df[results_df['K'] == best_k].iloc[0]

    return {
        'best_k': best_k,
        'results_df': results_df,
        'best_model': best_model,
        'best_metrics': best_results.to_dict()
    }

def plot_results(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                 target_col: str, predictions: Dict = None):
    """
    Plot the time series data and predictions if available.
    """
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    # Plot 1: Full time series with splits
    ax1 = axes[0]
    ax1.plot(train_df.index, train_df[target_col], label='Train', color='blue', alpha=0.7)
    ax1.plot(val_df.index, val_df[target_col], label='Validation', color='green', alpha=0.7)
    ax1.plot(test_df.index, test_df[target_col], label='Test', color='red', alpha=0.7)
    ax1.set_title(f'{target_col} - Time Series with Train/Val/Test Split')
    ax1.set_xlabel('Index')
    ax1.set_ylabel(target_col)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: ACF
    ax2 = axes[1]
    series = pd.concat([train_df[target_col], val_df[target_col], test_df[target_col]])
    acf_values, _ = extract_seasonality_acf(series, max_lag=100)
    lags = np.arange(len(acf_values))
    ax2.stem(lags, acf_values, linefmt='b-', markerfmt='bo', basefmt='k-')
    ax2.set_title('Autocorrelation Function (ACF)')
    ax2.set_xlabel('Lag')
    ax2.set_ylabel('ACF')
    ax2.grid(True, alpha=0.3)

    # Add confidence bounds
    n = len(series)
    confidence_bound = 1.96 / np.sqrt(n)
    ax2.axhline(y=confidence_bound, color='r', linestyle='--', alpha=0.5, label='95% Confidence')
    ax2.axhline(y=-confidence_bound, color='r', linestyle='--', alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('amd_analysis_plots.png', dpi=100)
    plt.show()
    print("Plots saved to 'amd_analysis_plots.png'")

def plot_results_combined(results_atr_90: Dict, results_atr_0: Dict):
    """
    Plot combined results for both ATR_90 and ATR_0 analysis.
    """
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))

    # ATR_90 plots (left column)
    train_90 = results_atr_90['train_df']
    val_90 = results_atr_90['val_df']
    test_90 = results_atr_90['test_df']
    target_90 = results_atr_90['target_col']

    # Plot 1: ATR_90 time series
    ax1 = axes[0, 0]
    ax1.plot(train_90.index, train_90[target_90], label='Train', color='blue', alpha=0.7)
    ax1.plot(val_90.index, val_90[target_90], label='Validation', color='green', alpha=0.7)
    ax1.plot(test_90.index, test_90[target_90], label='Test', color='red', alpha=0.7)
    ax1.set_title('ATR_90 - Time Series with Train/Val/Test Split')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('ATR_90')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: ATR_90 ACF
    ax2 = axes[1, 0]
    series_90 = pd.concat([train_90[target_90], val_90[target_90], test_90[target_90]])
    acf_values_90, _ = extract_seasonality_acf(series_90, max_lag=100)
    lags = np.arange(len(acf_values_90))
    ax2.stem(lags, acf_values_90, linefmt='b-', markerfmt='bo', basefmt='k-')
    ax2.set_title('ATR_90 - Autocorrelation Function (ACF)')
    ax2.set_xlabel('Lag')
    ax2.set_ylabel('ACF')
    ax2.grid(True, alpha=0.3)
    n_90 = len(series_90)
    confidence_bound_90 = 1.96 / np.sqrt(n_90)
    ax2.axhline(y=confidence_bound_90, color='r', linestyle='--', alpha=0.5, label='95% Confidence')
    ax2.axhline(y=-confidence_bound_90, color='r', linestyle='--', alpha=0.5)
    ax2.legend()

    # ATR_0 plots (right column)
    train_0 = results_atr_0['train_df']
    val_0 = results_atr_0['val_df']
    test_0 = results_atr_0['test_df']
    target_0 = results_atr_0['target_col']

    # Plot 3: ATR_0 time series
    ax3 = axes[0, 1]
    ax3.plot(train_0.index, train_0[target_0], label='Train', color='blue', alpha=0.7)
    ax3.plot(val_0.index, val_0[target_0], label='Validation', color='green', alpha=0.7)
    ax3.plot(test_0.index, test_0[target_0], label='Test', color='red', alpha=0.7)
    ax3.set_title('ATR_0 - Time Series with Train/Val/Test Split')
    ax3.set_xlabel('Index')
    ax3.set_ylabel('ATR_0')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: ATR_0 ACF
    ax4 = axes[1, 1]
    series_0 = pd.concat([train_0[target_0], val_0[target_0], test_0[target_0]])
    acf_values_0, _ = extract_seasonality_acf(series_0, max_lag=100)
    lags = np.arange(len(acf_values_0))
    ax4.stem(lags, acf_values_0, linefmt='g-', markerfmt='go', basefmt='k-')
    ax4.set_title('ATR_0 - Autocorrelation Function (ACF)')
    ax4.set_xlabel('Lag')
    ax4.set_ylabel('ACF')
    ax4.grid(True, alpha=0.3)
    n_0 = len(series_0)
    confidence_bound_0 = 1.96 / np.sqrt(n_0)
    ax4.axhline(y=confidence_bound_0, color='r', linestyle='--', alpha=0.5, label='95% Confidence')
    ax4.axhline(y=-confidence_bound_0, color='r', linestyle='--', alpha=0.5)
    ax4.legend()

    # Plot 5: Model comparison
    ax5 = axes[2, 0]
    models = ['Naive\nBaseline', 'AR-Fourier']
    atr_90_rmse = [results_atr_90['naive_baseline']['test_rmse'],
                    results_atr_90['ar_fourier']['best_metrics']['test_rmse']]
    atr_0_rmse = [results_atr_0['naive_baseline']['test_rmse'],
                   results_atr_0['ar_fourier']['best_metrics']['test_rmse']]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax5.bar(x - width/2, atr_90_rmse, width, label='ATR_90', color='blue', alpha=0.7)
    bars2 = ax5.bar(x + width/2, atr_0_rmse, width, label='ATR_0', color='green', alpha=0.7)

    ax5.set_ylabel('RMSE')
    ax5.set_title('Model Comparison - Test RMSE')
    ax5.set_xticks(x)
    ax5.set_xticklabels(models)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=8)

    # Plot 6: Improvement comparison
    ax6 = axes[2, 1]
    improvement_90 = (results_atr_90['naive_baseline']['test_rmse'] -
                      results_atr_90['ar_fourier']['best_metrics']['test_rmse']) / \
                     results_atr_90['naive_baseline']['test_rmse'] * 100
    improvement_0 = (results_atr_0['naive_baseline']['test_rmse'] -
                     results_atr_0['ar_fourier']['best_metrics']['test_rmse']) / \
                    results_atr_0['naive_baseline']['test_rmse'] * 100

    bars = ax6.bar(['ATR_90', 'ATR_0'], [improvement_90, improvement_0],
                   color=['blue', 'green'], alpha=0.7)
    ax6.set_ylabel('Improvement (%)')
    ax6.set_title('AR-Fourier Improvement over Naive Baseline')
    ax6.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('amd_analysis_plots.png', dpi=100)
    plt.show()
    print("Plots saved to 'amd_analysis_plots.png'")

def analyze_atr_target(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                        minutes_future: int) -> Dict:
    """
    Run complete analysis pipeline for a specific ATR target (0 or 90 minutes).
    Now receives pre-split data to ensure both atr_0 and atr_90 use the same splits.
    Note: CSV files are NOT saved here - they will be saved once in main() with both columns.
    """
    print("\n" + "="*60)
    print(f"Analyzing ATR_{minutes_future}")
    print("="*60)

    # Use the appropriate target column
    target_col = f'atr_{minutes_future}'

    # Verify the target column exists
    if target_col not in train_df.columns:
        raise ValueError(f"Column '{target_col}' not found in dataframe")

    # 2. Extract seasonality using ACF
    print("\n2. Implementing ACF to extract seasonality...")
    full_series = pd.concat([train_df[target_col], val_df[target_col], test_df[target_col]])
    acf_values, seasonal_period = extract_seasonality_acf(full_series)

    # 3. Extract trend slope
    print("\n3. Extracting trend slope...")
    slope, intercept = extract_trend_slope(train_df[target_col])

    # 4. Implement naive baseline
    print("\n4. Implementing Naive Baseline Model...")
    naive_results = implement_naive_baseline(
        train_df[target_col],
        val_df[target_col],
        test_df[target_col],
        horizon=1
    )

    # 5. Run Fourier analysis with K from 1 to 8
    print("\n5. Running Fourier analysis with K from 1 to 8...")
    print("Implementing AR-Fourier Model...")

    # Use detected seasonal period or default
    if seasonal_period is None or seasonal_period < 2:
        seasonal_period = 60  # Default to hourly pattern for minute data

    ar_fourier_results = implement_ar_fourier_model(
        train_df, val_df, test_df,
        target_col=target_col,
        period=seasonal_period,
        max_k=8,
        lag=1
    )

    # 6. Summary comparison
    print("\n" + "="*60)
    print(f"FINAL MODEL COMPARISON for ATR_{minutes_future}")
    print("="*60)

    print(f"\nNaive Baseline (Test Set):")
    print(f"  MAE:  {naive_results['test_mae']:.6f}")
    print(f"  MSE:  {naive_results['test_mse']:.6f}")
    print(f"  RMSE: {naive_results['test_rmse']:.6f}")

    best_metrics = ar_fourier_results['best_metrics']
    print(f"\nAR-Fourier Model (Best K={ar_fourier_results['best_k']}, Test Set):")
    print(f"  MAE:  {best_metrics['test_mae']:.6f}")
    print(f"  MSE:  {best_metrics['test_mse']:.6f}")
    print(f"  RMSE: {best_metrics['test_rmse']:.6f}")

    # Calculate improvement
    rmse_improvement = (naive_results['test_rmse'] - best_metrics['test_rmse']) / naive_results['test_rmse'] * 100
    print(f"\nImprovement over Naive Baseline: {rmse_improvement:.2f}%")

    # Return results for aggregation
    total_rows = len(train_df) + len(val_df) + len(test_df)
    return {
        'data_info': {
            'total_rows': total_rows,
            'target_column': target_col,
            'minutes_in_future': minutes_future
        },
        'split_info': {
            'train_ratio': TRAIN_RATIO,
            'val_ratio': VAL_RATIO,
            'test_ratio': TEST_RATIO,
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df)
        },
        'seasonality': {
            'detected_period': int(seasonal_period),
            'trend_slope': float(slope),
            'trend_intercept': float(intercept)
        },
        'naive_baseline': naive_results,
        'ar_fourier': {
            'best_k': int(ar_fourier_results['best_k']),
            'best_metrics': {k: float(v) for k, v in best_metrics.items() if k != 'K'}
        },
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df,
        'target_col': target_col
    }

def main():
    """
    Main function to run the complete AMD time series analysis pipeline for both ATR_0 and ATR_90.
    """
    print("="*60)
    print("AMD Time Series Analysis Pipeline")
    print("="*60)

    # 1. Load data
    print("\n1. Loading AMD dataset...")
    df = load_amd_data('AMD_20240102-090000_20251108-002700.csv')
    print(f"Loaded {len(df)} rows")
    print(f"Date range: {df['time_entry_ts'].min()} to {df['time_entry_ts'].max()}")

    # 2. Create both ATR columns at once to ensure they use the same data rows
    print("\n2. Creating both atr_0 and atr_90 columns...")
    df_with_atr = create_both_atr_columns(df, minutes_future=90)
    print(f"Both columns created. DataFrame now has {len(df_with_atr)} rows")

    # 3. Perform a single train-val-test split that will be used for BOTH analyses
    print(f"\n3. Performing {TRAIN_RATIO:.0%}-{VAL_RATIO:.0%}-{TEST_RATIO:.0%} split...")
    train_df, val_df, test_df = split_data(df_with_atr, TRAIN_RATIO, VAL_RATIO)
    print("Split complete. Both atr_0 and atr_90 will use these same splits.")

    # 4. Run analysis for ATR_0 (current ATR)
    print("\n" + "="*60)
    print("STARTING ANALYSIS FOR ATR_0 (Current ATR)")
    print("="*60)
    results_atr_0 = analyze_atr_target(train_df, val_df, test_df, minutes_future=0)

    # 5. Run analysis for ATR_90 (90 minutes in future)
    print("\n" + "="*60)
    print("STARTING ANALYSIS FOR ATR_90 (90 minutes in future)")
    print("="*60)
    results_atr_90 = analyze_atr_target(train_df, val_df, test_df, minutes_future=90)

    # 6. Save combined train/val/test CSV files (with both atr_0 and atr_90 columns)
    print("\n6. Saving combined train/val/test CSV files...")
    train_df.to_csv('amd_train.csv', index=False)
    val_df.to_csv('amd_val.csv', index=False)
    test_df.to_csv('amd_test.csv', index=False)
    print(f"  Saved amd_train.csv ({len(train_df)} rows)")
    print(f"  Saved amd_val.csv ({len(val_df)} rows)")
    print(f"  Saved amd_test.csv ({len(test_df)} rows)")
    print("  Each file contains both 'atr_0' and 'atr_90' columns")

    # 7. Generate plots for both
    print("\n7. Generating visualization plots...")
    plot_results_combined(results_atr_90, results_atr_0)

    # 8. Save combined results to JSON
    results_summary = {
        'atr_90': {
            'data_info': results_atr_90['data_info'],
            'split_info': results_atr_90['split_info'],
            'seasonality': results_atr_90['seasonality'],
            'naive_baseline': results_atr_90['naive_baseline'],
            'ar_fourier': results_atr_90['ar_fourier']
        },
        'atr_0': {
            'data_info': results_atr_0['data_info'],
            'split_info': results_atr_0['split_info'],
            'seasonality': results_atr_0['seasonality'],
            'naive_baseline': results_atr_0['naive_baseline'],
            'ar_fourier': results_atr_0['ar_fourier']
        },
        'comparison': {
            'atr_90_vs_atr_0': {
                'naive_rmse_difference': results_atr_90['naive_baseline']['test_rmse'] - results_atr_0['naive_baseline']['test_rmse'],
                'ar_fourier_rmse_difference': results_atr_90['ar_fourier']['best_metrics']['test_rmse'] - results_atr_0['ar_fourier']['best_metrics']['test_rmse'],
                'atr_90_improvement': (results_atr_90['naive_baseline']['test_rmse'] - results_atr_90['ar_fourier']['best_metrics']['test_rmse']) / results_atr_90['naive_baseline']['test_rmse'] * 100,
                'atr_0_improvement': (results_atr_0['naive_baseline']['test_rmse'] - results_atr_0['ar_fourier']['best_metrics']['test_rmse']) / results_atr_0['naive_baseline']['test_rmse'] * 100
            }
        }
    }

    with open('amd_analysis_results.json', 'w') as f:
        json.dump(results_summary, f, indent=4)
    print("\nResults saved to 'amd_analysis_results.json'")

    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)

if __name__ == "__main__":
    main()