import os
import pandas as pd
import scipy.stats as stats

# Load the dataset
file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_fin_perf.csv")
data = pd.read_csv(file_path)

# Filter out irrelevant index
data_filtered = data.drop(columns=['in_index6'])

# Separate the data for ALV.DE, TTE.PA, and other tickers
data_alv = data_filtered[data_filtered['ticker'] == 'ALV.DE']
data_tte = data_filtered[data_filtered['ticker'] == 'TTE.PA']
data_others = data_filtered[~data_filtered['ticker'].isin(['ALV.DE', 'TTE.PA'])]

# Function to perform t-tests
def perform_ttests(data1, data2):
    ttest_results = {}
    metrics_to_test = ['ROE (%)', 'Net Profit Margin (%)', 'Revenue Growth (%)']
    
    for metric in metrics_to_test:
        data1_metric = data1[metric].dropna()
        data2_metric = data2[metric].dropna()
        
        if len(data1_metric) > 1 and len(data2_metric) > 1:
            t_stat, p_value = stats.ttest_ind(data1_metric, data2_metric, equal_var=False)
            ttest_results[metric] = {'t_stat': t_stat, 'p_value': p_value}
        else:
            ttest_results[metric] = {'t_stat': 'insufficient data', 'p_value': 'insufficient data'}
    
    return ttest_results

# Perform t-tests for ALV.DE and TTE.PA against others
ttest_results_alv_vs_others = perform_ttests(data_alv, data_others)
ttest_results_tte_vs_others = perform_ttests(data_tte, data_others)

# Print the results
print("ALV.DE vs Others Results:", ttest_results_alv_vs_others)
print("TTE.PA vs Others Results:", ttest_results_tte_vs_others)
