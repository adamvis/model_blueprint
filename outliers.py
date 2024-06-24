import matplotlib.pyplot as plt
from hampel import hampel
import pandas as pd

def zscore_od(series, window, thresh=3, return_all=False):
    rolling_series = series.rolling(window=window, min_periods=1, center=True)
    avg = rolling_series.mean()
    std = rolling_series.std(ddof=0)
    z = series.sub(avg).div(std)   
    m = z.between(-thresh, thresh)
    
    if return_all:
        return z, avg, std, m
    return series.where(m, avg)
    
def plot_zscore_od(original_series, rolling_mean, median_series, ax=None):
    if ax is None:
        ax = plt.subplot()
    original_series.plot(label='data', ax=ax)
    rolling_mean.plot(label='mean', ax=ax)
    try:
        original_series.loc[~median_series].plot(label='outliers', marker='o', ls='')
        rolling_mean[~median_series].plot(label='replacement', marker='o', ls='')
    except ValueError:  #raised if `y` is empty.
        pass
    plt.legend()
    return ax

    return ax

def hampel_od(series, window, **kwargs):
    result = hampel(series, window_size=window, **kwargs)
#     result.filtered_data.index = series.index
#     result.outlier_indices = series.iloc[result.outlier_indices].index
#     result.medians = pd.Series(result.medians)
#     result.medians.index = series.index
#     result.median_absolute_deviations = pd.Series(result.median_absolute_deviations)
#     result.median_absolute_deviations.index = series.index
#     result.thresholds = pd.Series(result.thresholds)
#     result.thresholds.index = series.index
    return result
    
def plot_hampel_od(original_data, hampel_result):
    filtered_data = hampel_result.filtered_data
    medians = hampel_result.medians
    mad_values = hampel_result.median_absolute_deviations
    thresholds = hampel_result.thresholds
    outlier_indices = hampel_result.outlier_indices
    
    filtered_data.index = original_data.index
    medians = pd.Series(medians, index=original_data.index)
    mad_values = pd.Series(mad_values, index=original_data.index)
    thresholds = pd.Series(thresholds, index=original_data.index)
#     outlier_indices = original_data.iloc[outlier_indices].index.values
    
    plt.clf()
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    # Plot the original data with estimated standard deviations in the first subplot
    axes[0].plot(original_data, label='Original Data', color='b')
    axes[0].fill_between(original_data.index, medians + thresholds,
                         medians - thresholds, color='gray', alpha=0.5, label='Median +- Threshold')
    axes[0].set_xlabel('Data Point')
    axes[0].set_ylabel('Value')
    axes[0].set_title('Original Data with Bands representing Upper and Lower limits')

    for i in outlier_indices:
        axes[0].plot(i, original_data.iloc[i], 'ro', markersize=5)  # Mark as red

    axes[0].legend()

    # Plot the filtered data in the second subplot
    axes[1].plot(filtered_data, label='Filtered Data', color='g')
    axes[1].set_xlabel('Data Point')
    axes[1].set_ylabel('Value')
    axes[1].set_title('Filtered Data')
    axes[1].legend()

    # Adjust spacing between subplots
    plt.tight_layout()

    return fig
    
