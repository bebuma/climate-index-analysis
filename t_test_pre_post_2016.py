import os
import pandas as pd
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
absolute_path = os.path.abspath(__file__)
file_path = os.path.dirname(absolute_path) + "/data_fin_perf.csv"
data = pd.read_csv(file_path)

# Filter out irrelevant index
data_filtered = data.drop(columns=['in_index6'])

# Creating a binary variable for pre and post 2016
data_filtered['pre_post_2016'] = data_filtered['year'].apply(lambda x: 0 if x < 2016 else 1)

# Group data by ticker
grouped_data = data_filtered.groupby('ticker')

# Performing t-tests using statsmodels for each financial metric for each ticker
ttest_results_sm = {}

for ticker, group in grouped_data:
    print(ticker)
    print(group)
    ttest_results_sm[ticker] = {}
    for metric in ['ROE (%)', 'Net Profit Margin (%)', 'Revenue Growth (%)']:
        # Check if the group has enough data for the metric
        if group[metric].notnull().sum() > 1:
            # Regression model with the pre_post_2016 binary variable
            model = sm.OLS(group[metric], sm.add_constant(group['pre_post_2016'])).fit()
            
            # Extracting the t-statistic and p-value for the binary variable
            t_stat = model.tvalues[1]
            p_value = model.pvalues[1]
            
            # Storing the results in a dictionary
            ttest_results_sm[ticker][metric] = {'t_stat': t_stat, 'p_value': p_value}
        else:
            ttest_results_sm[ticker][metric] = {'t_stat': 'insufficient data', 'p_value': 'insufficient data'}

# Outputting the results
print(ttest_results_sm)

# Your result data
result_data = ttest_results_sm

# Convert the results to a DataFrame for easier processing
metrics = ['ROE (%)', 'Net Profit Margin (%)', 'Revenue Growth (%)']
df_results = pd.DataFrame(columns=['Ticker', 'Metric', 'T-Statistic', 'P-Value'])

for ticker, metrics_data in result_data.items():
    for metric, values in metrics_data.items():
        df_results = df_results.append({'Ticker': ticker, 'Metric': metric, 'T-Statistic': values['t_stat'], 'P-Value': values['p_value']}, ignore_index=True)

# Separate DataFrames for each metric
dfs = {metric: df_results[df_results['Metric'] == metric] for metric in metrics}

# Plotting
for metric, df in dfs.items():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(df['Ticker'], df['T-Statistic'], label='T-Statistic')
    ax.set_xlabel('Ticker')
    ax.set_ylabel('T-Statistic')
    ax.set_title(f'T-Statistic for {metric}')
    ax.tick_params(axis='x', rotation=90)
    plt.tight_layout()
    plt.show()

    # Print the table of results
    print(f"\nTable of results for {metric}:")
    print(df[['Ticker', 'T-Statistic', 'P-Value']].to_string(index=False))

