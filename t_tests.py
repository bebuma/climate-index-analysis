import os
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt


def load_data(file_name):
    """
    Load the dataset from a CSV file.

    Parameters:
    file_name (str): The name of the CSV file to load.

    Returns:
    pd.DataFrame: The loaded dataset.
    """
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    return pd.read_csv(file_path)


def filter_data(data):
    """
    Filter out irrelevant columns and create a binary variable for pre and post 2016.

    Parameters:
    data (pd.DataFrame): The original dataset.

    Returns:
    pd.DataFrame: The filtered dataset with a new binary variable.
    """
    data_filtered = data.drop(columns=["in_index6"])
    data_filtered["pre_post_2016"] = (data_filtered["year"] >= 2016).astype(int)
    return data_filtered


def perform_ttests(data1, data2, metrics):
    """
    Perform t-tests on specified metrics between two datasets.

    Parameters:
    data1 (pd.DataFrame): The first dataset.
    data2 (pd.DataFrame): The second dataset.
    metrics (list): List of metrics to perform t-tests on.

    Returns:
    dict: A dictionary with t-test results for each metric.
    """
    ttest_results = {}
    for metric in metrics:
        data1_metric = data1[metric].dropna()
        data2_metric = data2[metric].dropna()

        if len(data1_metric) > 1 and len(data2_metric) > 1:
            t_stat, p_value = stats.ttest_ind(
                data1_metric, data2_metric, equal_var=False
            )
            ttest_results[metric] = {"t_stat": t_stat, "p_value": p_value}
        else:
            ttest_results[metric] = {
                "t_stat": "insufficient data",
                "p_value": "insufficient data",
            }

    return ttest_results


def perform_t_tests_sm(grouped_data, metrics):
    """
    Perform t-tests using statsmodels for each financial metric for each ticker.

    Parameters:
    grouped_data (pd.DataFrameGroupBy): Grouped data by ticker.
    metrics (list): List of metrics to perform t-tests on.

    Returns:
    dict: A dictionary with t-test results for each ticker and metric.
    """
    ttest_results_sm = {}
    for ticker, group in grouped_data:
        ttest_results_sm[ticker] = {}
        for metric in metrics:
            if group[metric].notnull().sum() > 1:
                model = sm.OLS(
                    group[metric], sm.add_constant(group["pre_post_2016"])
                ).fit()
                t_stat = model.tvalues[1]
                p_value = model.pvalues[1]
                ttest_results_sm[ticker][metric] = {
                    "t_stat": t_stat,
                    "p_value": p_value,
                }
            else:
                ttest_results_sm[ticker][metric] = {
                    "t_stat": "insufficient data",
                    "p_value": "insufficient data",
                }
    return ttest_results_sm


def convert_results_to_dataframe(ttest_results_sm):
    """
    Convert the t-test results to a DataFrame for easier processing.

    Parameters:
    ttest_results_sm (dict): T-test results from statsmodels.

    Returns:
    pd.DataFrame: DataFrame containing the t-test results.
    """
    df_results = pd.DataFrame(columns=["Ticker", "Metric", "T-Statistic", "P-Value"])
    for ticker, metrics_data in ttest_results_sm.items():
        for metric, values in metrics_data.items():
            new_row = pd.DataFrame(
                [
                    {
                        "Ticker": ticker,
                        "Metric": metric,
                        "T-Statistic": values["t_stat"],
                        "P-Value": values["p_value"],
                    }
                ]
            )
            df_results = pd.concat([df_results, new_row], ignore_index=True)
    return df_results


def plot_results(dfs, output_folder):
    """
    Plot the t-test results for each metric.

    Parameters:
    dfs (dict): Dictionary of DataFrames for each metric.
    output_folder (str): Folder to save the plots and CSV files.
    """
    for metric, df in dfs.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(df["Ticker"], df["T-Statistic"], label="T-Statistic")
        ax.set_xlabel("Ticker")
        ax.set_ylabel("T-Statistic")
        ax.set_title(f"T-Statistic for {metric}")
        ax.tick_params(axis="x", rotation=90)
        plt.tight_layout()
        filename = f"{metric.replace(' ', '_').replace('%', 'pct')}_t_statistic"
        plt.savefig(os.path.join(output_folder, filename + ".png"))
        # plt.show()
        plt.close()
        print(f"\nTable of results for {metric}:")
        show_df = df[["Ticker", "Metric", "T-Statistic", "P-Value"]].copy()
        print(show_df)
        show_df.to_csv(os.path.join(output_folder, filename + ".csv"))


def main():
    """
    Main function to load data, filter it, perform t-tests, and print results.
    """
    try:
        # Define output folder
        absolute_path = os.path.abspath(__file__)
        output_folder = os.path.join(os.path.dirname(absolute_path), "output")
        os.makedirs(output_folder, exist_ok=True)

        # Load the dataset
        data = load_data("data_fin_perf.csv")

        # Filter the data
        data_filtered = filter_data(data)

        # Separate data based on tickers
        data_alv = data_filtered[data_filtered["ticker"] == "ALV.DE"]
        data_tte = data_filtered[data_filtered["ticker"] == "TTE.PA"]
        data_others = data_filtered[~data_filtered["ticker"].isin(["ALV.DE", "TTE.PA"])]

        # Metrics to test
        metrics = ["ROE (%)", "Net Profit Margin (%)", "Revenue Growth (%)"]

        # (1) Testing between Non-listing company & Listing company in general
        # Perform t-tests using scipy
        ttest_results_alv_vs_others = pd.DataFrame(
            perform_ttests(data_alv, data_others, metrics)
        )
        ttest_results_tte_vs_others = pd.DataFrame(
            perform_ttests(data_tte, data_others, metrics)
        )

        # Print the results from scipy t-tests
        ttest_results_alv_vs_others.to_csv(
            os.path.join(output_folder, "ALV.DE_vs_Others_Results.csv")
        )
        print("\nALV.DE vs Others Results:\n", ttest_results_alv_vs_others)
        ttest_results_tte_vs_others.to_csv(
            os.path.join(output_folder, "TTE.PA_vs_Others_Results.csv")
        )
        print("\nTTE.PA vs Others Results:\n", ttest_results_tte_vs_others)

        # (2) Testing between Pre & Post 2016
        # Perform t-tests using statsmodels
        grouped_data = data_filtered.groupby("ticker")
        ttest_results_sm = perform_t_tests_sm(grouped_data, metrics)

        # Convert results to DataFrame
        df_results = convert_results_to_dataframe(ttest_results_sm)

        # Plot the results
        dfs = {metric: df_results[df_results["Metric"] == metric] for metric in metrics}
        plot_results(dfs, output_folder)

    except FileNotFoundError:
        print("The specified file was not found.")
    except pd.errors.EmptyDataError:
        print("The file is empty.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
