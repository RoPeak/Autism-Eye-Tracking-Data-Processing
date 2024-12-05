# Task 1 of PCHT coursework 2
#
# Task 1a:
# Performing data analysis and computing averages (with standard deviation) for Pupil Diameter (PD)
# and Fixed Duration (FD) for the Autism (ASD) and Typically Devoloping (TD) groups

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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
    # Create a dictionary to store results
    results = {}

    # Iterate over each emotion
    for emotion in ["Angry", "Happy", "Neutral"]:
        results[emotion] = {}
        emotion_data = data[data['Emotion'] == emotion]

        # For each emotion, calculate statistics for ASD and TD separately
        for group in ["ASD", "TD"]:
            group_emotion_data = emotion_data[emotion_data['Group'] == group]
            pd_mean, pd_std = compute_mean_std(group_emotion_data, 'PD')
            fd_mean, fd_std = compute_mean_std(group_emotion_data, 'FD')

            # Store the results in the dictionary
            results[emotion][group] = {
                'PD': (pd_mean, pd_std),
                'FD': (fd_mean, fd_std)
            }

    # Display results if desired
    print("Would you like to view the averages by emotion and group? (y/n)")
    ans = input(">>> ")
    if ans.lower() == "y":
        print("Round by how many decimal places?")
        decimal_places = int(input(">>> "))
        for emotion, stats in results.items():
            print(f"Emotion: {emotion}")
            for group, metrics in stats.items():
                print(f"  Group: {group}")
                print(
                    f"    PD average: {round(metrics['PD'][0], decimal_places)} ± {round(metrics['PD'][1], decimal_places)}"
                )
                print(
                    f"    FD average: {round(metrics['FD'][0], decimal_places)} ± {round(metrics['FD'][1], decimal_places)}"
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


# Plotting function to visualise the average PD and FD of ASD vs TD groups
def plot_averages(averages):
    groups = ['ASD', 'TD']
    metrics = ['PD', 'FD']

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    for i, metric in enumerate(metrics):
        means = [averages.loc[group, f'{metric}_avg'] for group in groups]
        stds = [averages.loc[group, f'{metric}_std'] for group in groups]
        axs[i].bar(groups,
                   means,
                   yerr=stds,
                   capsize=5,
                   color=['blue', 'orange'])
        axs[i].set_title(f'{metric} Averages by Group')
        axs[i].set_ylabel(f'{metric} (mean ± std)')
        axs[i].set_xlabel('Group')

    plt.tight_layout()
    plt.show()


# Plotting function to visualise the trends in the smoothed data
def plot_smoothed_trends(data, group, metric):
    participants = data[data['Group'] == group]['Participant'].unique()
    plt.figure(figsize=(12, 6))
    for participant in participants:
        participant_data = data[data['Participant'] == participant]
        plt.plot(participant_data.index,
                 participant_data[metric],
                 label=f'Participant {participant}')
    plt.title(f'{metric} Trends for {group} Group')
    plt.xlabel('Index')
    plt.ylabel(metric)
    plt.legend()
    plt.show()


# Plotting function to visualise the average PD and FD of ASD vs TD groups by emotion
def plot_emotion_group_averages(results):
    # Flatten the `results` dictionary into a DataFrame for easier plotting
    data_for_plot = []
    for emotion, group_data in results.items():
        for group, metrics in group_data.items():
            data_for_plot.append({
                'Emotion': emotion,
                'Group': group,
                'Metric': 'PD',
                'Average': metrics['PD'][0],
                'StdDev': metrics['PD'][1]
            })
            data_for_plot.append({
                'Emotion': emotion,
                'Group': group,
                'Metric': 'FD',
                'Average': metrics['FD'][0],
                'StdDev': metrics['FD'][1]
            })

    # Convert to a pandas DataFrame
    plot_df = pd.DataFrame(data_for_plot)

    # Separate PD and FD into two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

    # Calculate dynamic upper bounds for y-axis
    pd_data = plot_df[plot_df['Metric'] == 'PD']
    fd_data = plot_df[plot_df['Metric'] == 'FD']
    pd_ymax = (pd_data['Average'] + pd_data['StdDev']).max()
    fd_ymax = (fd_data['Average'] + fd_data['StdDev']).max()

    # Plot for Pupil Diameter (PD)
    sns.barplot(
        data=pd_data,
        x='Emotion',
        y='Average',
        hue='Group',
        ax=axes[0],
        errorbar=None,  # Disable automatic error bars
        palette="coolwarm")
    axes[0].set_title("Average Pupil Diameter (PD) by Emotion and Group")
    axes[0].set_ylabel("Average PD")
    axes[0].set_xlabel("Emotion")
    axes[0].legend(title="Group")
    axes[0].set_ylim(0, pd_ymax * 1.1)  # Add 10% buffer

    # Add custom error bars for PD
    for container, group in zip(axes[0].containers, ["ASD", "TD"]):
        group_data = pd_data[pd_data["Group"] == group]
        for bar, (_, row) in zip(container, group_data.iterrows()):
            bar_x = bar.get_x() + bar.get_width() / 2
            axes[0].errorbar(
                bar_x,
                row["Average"],
                yerr=row["StdDev"],
                fmt="none",
                color="black",
                capsize=5,
                linewidth=1,
            )

    # Plot for Fixation Duration (FD)
    sns.barplot(
        data=fd_data,
        x='Emotion',
        y='Average',
        hue='Group',
        ax=axes[1],
        errorbar=None,  # Disable automatic error bars
        palette="viridis")
    axes[1].set_title("Average Fixation Duration (FD) by Emotion and Group")
    axes[1].set_ylabel("Average FD")
    axes[1].set_xlabel("Emotion")
    axes[1].legend(title="Group")
    axes[1].set_ylim(0, fd_ymax * 1.1)  # Add 10% buffer

    # Add custom error bars for FD
    for container, group in zip(axes[1].containers, ["ASD", "TD"]):
        group_data = fd_data[fd_data["Group"] == group]
        for bar, (_, row) in zip(container, group_data.iterrows()):
            bar_x = bar.get_x() + bar.get_width() / 2
            axes[1].errorbar(
                bar_x,
                row["Average"],
                yerr=row["StdDev"],
                fmt="none",
                color="black",
                capsize=5,
                linewidth=1,
            )

    plt.tight_layout()
    plt.show()


# Plotting function for average FD for ROI 1 and 2 for ASD vs TD groups
def plot_fd_by_roi(fd_averages):
    groups = ['ASD', 'TD']
    rois = [1, 2]
    x = range(len(groups))

    face_means = [fd_averages.loc[(group, 1), 'FD_avg'] for group in groups]
    nonface_means = [fd_averages.loc[(group, 2), 'FD_avg'] for group in groups]

    face_stds = [fd_averages.loc[(group, 1), 'FD_std'] for group in groups]
    nonface_stds = [fd_averages.loc[(group, 2), 'FD_std'] for group in groups]

    width = 0.4
    plt.bar(x,
            face_means,
            width,
            yerr=face_stds,
            label='Face',
            color='green',
            capsize=5)
    plt.bar([i + width for i in x],
            nonface_means,
            width,
            yerr=nonface_stds,
            label='Non-Face',
            color='red',
            capsize=5)

    plt.title('Fixation Duration by ROI and Group')
    plt.xticks([i + width / 2 for i in x], groups)
    plt.ylabel('Fixation Duration (mean ± std)')
    plt.legend()
    plt.show()


# Main execution
if __name__ == "__main__":
    original_data = load_data("autism_data_4101")

    cleaned_data = clean_data(original_data)

    smoothed_data = smooth_data(cleaned_data)

    averages = calculate_averages_by_group(smoothed_data)
    plot_averages(averages)

    averages_by_emotion = calculate_averages_by_group_emotion(smoothed_data)
    plot_emotion_group_averages(averages_by_emotion)

    fd_averages = calculate_average_FD_by_ROI(smoothed_data)
    plot_fd_by_roi(fd_averages)

    print("Would you like to save the processed data to CSV files? (y/n)")
    ans = input(">>> ")
    if ans.lower() == "y":
        output_data(original_data, cleaned_data, smoothed_data, averages,
                    averages_by_emotion, fd_averages)

    print("All done! Exiting...")
