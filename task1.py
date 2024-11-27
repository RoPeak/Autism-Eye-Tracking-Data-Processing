# Task 1 of PCHT coursework 2
#
# Task 1a:
# Performing data analysis and computing averages (with standard deviation) for Pupil Diameter (PD)
# and Fixed Duration (FD) for the Autism (ASD) and Typically Devoloping (TD) groups

import os
import pandas as pd
import numpy as np


# Function to traverse the directory, load CSVs, and combine into a single DataFrame
def load_data(base_dir):
    print("BEGIN: load_data()")
    print("Loading data...")
    data = []
    for group in ["ASD", "TD"]:
        group_path = os.path.join(base_dir, group)
        for participant in os.listdir(group_path):
            participant_path = os.path.join(group_path, participant)
            for emotion_file in os.listdir(participant_path):
                emotion = emotion_file.split(".")[0]
                file_path = os.path.join(participant_path, emotion_file)
                df = pd.read_csv(file_path)
                df['Participant'] = participant
                df['Group'] = group
                df['Emotion'] = emotion
                data.append(df)
    print("Data loaded.")
    print(
        f"Processed {len(data)} files. Combined data shape: {pd.concat(data, ignore_index=True).shape[0]} rows x {pd.concat(data, ignore_index=True).shape[1]} columns."
    )
    print("Would you like to view a truncated version of the data? (y/n)")
    ans = input(">>> ")
    if ans.lower() == "y":
        print(pd.concat(data, ignore_index=True).head())
    else:
        print("Data not shown.")
    print("END: load_data()\n")
    return pd.concat(data, ignore_index=True)


# Function to clean the data by removing missing values and invalid columns
def clean_data(data):
    print("BEGIN: clean_data()")
    initial_shape = data.shape
    print("Analysing data...")
    print("Removing missing values...")
    data = data.drop(columns=["Unnamed: 3", "Unnamed: 4"])
    print(f"\t- Dropped invalid columns.")
    data = data.dropna()
    print(f"\t- Dropped missing values.")

    # Negative number check
    negative_values = (data.select_dtypes(include=['number']) < 0).any().any()

    print("Would you like to view a summary of the cleaned data? (y/n)")
    ans = input(">>> ")
    if ans.lower() == "y":
        dropped_columns = initial_shape[1] - data.shape[1]
        dropped_rows = initial_shape[0] - data.shape[0]
        print(
            f"Cleaning summary:\n\t- {dropped_columns} invalid columns removed.\n\t- {dropped_rows} rows with missing values removed."
        )
        if negative_values:
            print("\t- Negative values found in the data.")
        else:
            print("\t- No negative values found in the data.")
        print(
            f"Original shape was {initial_shape[0]} rows x {initial_shape[1]} columns.",
            end=" ")
        print(f"New shape is {data.shape[0]} rows x {data.shape[1]} columns.")
    print("END: clean_data()\n")
    return data


# Function to smooth data using Simple Moving Average method
def SMA_application(series, window_size=5):
    return series.rolling(window=window_size, min_periods=1).mean()


# Function to apply SMA to PD and FD columns in data
def smooth_data(data):
    print("BEGIN: smooth_data()")
    for participant in data['Participant'].unique():
        participant_data = data[data['Participant'] == participant]
        data.loc[data['Participant'] == participant,
                 'PD'] = SMA_application(participant_data['PD'])
        data.loc[data['Participant'] == participant,
                 'FD'] = SMA_application(participant_data['FD'])
    print("END: smooth_data()")
    return data


# Function to calculate the average and standard deviation for the ASD and TD groups
def calculate_averages_by_group(data):
    print("BEGIN: calculate_averages()")
    print("Calculating averages...")
    averages = data.groupby(['Group']).agg(PD_avg=('PD', 'mean'),
                                           PD_std=('PD', 'std'),
                                           FD_avg=('FD', 'mean'),
                                           FD_std=('FD', 'std'))
    print("Would you like to view the averages table? (y/n)")
    ans = input(">>> ")
    if ans.lower() == "y":
        print("Round by how many decimal places?")
        decimal_places = int(input(">>> "))
        print(
            f"ASD FD average: {round(averages.loc['ASD', 'PD_avg'], decimal_places)} ± {round(averages.loc['ASD', 'PD_std'], decimal_places)}"
        )
        print(
            f"TD PD average: {round(averages.loc['TD', 'PD_avg'], decimal_places)} ± {round(averages.loc['TD', 'PD_std'], decimal_places)}"
        )
        print(
            f"ASD FD average: {round(averages.loc['ASD', 'FD_avg'], decimal_places)} ± {round(averages.loc['ASD', 'FD_std'], decimal_places)}"
        )
        print(
            f"TD FD average: {round(averages.loc['TD', 'FD_avg'], decimal_places)} ± {round(averages.loc['TD', 'FD_std'], decimal_places)}"
        )

    print("END: calculate_averages()\n")
    return averages


# Function to calculate the average and standard deviation
def compute_mean_std(df, col):
    return df[col].mean(), df[col].std()


# Function to calculate the average and standard deviation for the ASD and TD groups based on the emotion
def calculate_averages_by_group_emotion(data):
    print("BEGIN: calculate_averages_by_group_emotion()")
    print("Calculating averages...")
    results = {}
    for emotion in ["Angry", "Happy", "Neutral"]:
        emotion_data = data[data['Emotion'] == emotion]
        pd_mean, pd_std = compute_mean_std(emotion_data, 'PD')
        fd_mean, fd_std = compute_mean_std(emotion_data, 'FD')
        results[emotion] = {'PD': (pd_mean, pd_std), 'FD': (fd_mean, fd_std)}
    print("Would you like to view the averages by emotion? (y/n)")
    ans = input(">>> ")
    if ans.lower() == "y":
        print("Round by how many decimal places?")
        decimal_places = int(input(">>> "))
        for emotion, stats in results.items():
            print(f"Emotion: {emotion}")
            print(
                f"PD average: {round(stats['PD'][0], decimal_places)} ± {round(stats['PD'][1], decimal_places)}"
            )
            print(
                f"FD average: {round(stats['FD'][0], decimal_places)} ± {round(stats['FD'][1], decimal_places)}"
            )
    print("END: calculate_averages_by_group_emotion()\n")
    return results


# Function to calculate the average FD for the ASD and TD groups for face vs non-face regions
# ROI = 1: face, ROI = 2: non-face
def calculate_average_FD_by_ROI(data):
    print("BEGIN: calculate_average_FD_by_ROI()")
    print("Calculating averages...")
    FD_averages = data.groupby(['Group', 'ROI']).agg(FD_avg=('FD', 'mean'),
                                                     FD_std=('FD', 'std'))
    print("Would you like to view the FD averages table? (y/n)")
    ans = input(">>> ")
    if ans.lower() == "y":
        print("Round by how many decimal places?")
        decimal_places = int(input(">>> "))
        for group in ["ASD", "TD"]:
            for roi in [1, 2]:
                print(
                    f"{group} FD average for ROI {roi}: {round(FD_averages.loc[(group, roi), 'FD_avg'], decimal_places)} ± {round(FD_averages.loc[(group, roi), 'FD_std'], decimal_places)}"
                )
    print("END: calculate_average_FD_by_ROI()\n")
    return FD_averages


# Function to output all the calculated data to a CSV file
def output_data(data, cleaned, smoothed, averages, averages_emotions,
                fd_averages):
    print("BEGIN: output_data()")
    print("Outputting data...")

    output_dir = "processed_data"
    os.makedirs(output_dir, exist_ok=True)

    data.to_csv(os.path.join(output_dir, "data.csv"), index=False)
    print("Original data saved to 'data.csv'.")
    cleaned.to_csv(os.path.join(output_dir, "cleaned_data.csv"), index=False)
    print("Cleaned data saved to 'cleaned_data.csv'.")
    smoothed.to_csv(os.path.join(output_dir, "smoothed_data.csv"), index=False)
    print("Smoothed data saved to 'smoothed_data.csv'.")
    pd.DataFrame(averages).to_csv(os.path.join(output_dir,
                                               "averages_task_1a.csv"),
                                  index=False)
    print("Averages saved to 'averages_task_1a.csv'.")
    pd.DataFrame(averages_emotions).to_csv(os.path.join(
        output_dir, "averages_emotions_1b.csv"),
                                           index=False)
    print("Averages by emotion saved to 'averages_emotions_1b.csv'.")
    pd.DataFrame(fd_averages).to_csv(os.path.join(output_dir,
                                                  "averages_fd_by_ROI_1c.csv"),
                                     index=False)
    print("Averages by ROI saved to 'averages_fd_by_ROI_1c.csv'.")

    print("END: output_data()\n")
    return


# Main execution
if __name__ == "__main__":
    original_data = load_data("autism_data_4101")

    cleaned_data = clean_data(original_data)

    smoothed_data = smooth_data(cleaned_data)

    averages = calculate_averages_by_group(smoothed_data)

    averages_by_emotion = calculate_averages_by_group_emotion(smoothed_data)

    fd_averages = calculate_average_FD_by_ROI(smoothed_data)

    print("Would you like to save the processed data to CSV files? (y/n)")
    ans = input(">>> ")
    if ans.lower() == "y":
        output_data(original_data, cleaned_data, smoothed_data, averages,
                    averages_by_emotion, fd_averages)

    print("All done! Exiting...")
