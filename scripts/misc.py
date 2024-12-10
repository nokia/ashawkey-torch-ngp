# import os
# import re
# import pandas as pd
# import matplotlib.pyplot as plt

# metric = "PSNR"

# def extract_metric_value(file_path):
#     metric_pattern = re.compile(fr'{re.escape(metric)} = (\d+(\.\d+)?)')
#     metric_values = []

#     with open(file_path, 'r') as file:
#         for line in file:
#             match = metric_pattern.search(line)
#             if match:
#                 metric_values.append(float(match.group(1)))

#     return metric_values

# def process_log_files(scene, method):
#     runs_dir = f"runs/runs_iNGP_360_syn_B=2to18"
#     if method == "":
#         log_dir = f"{runs_dir}/trial_syn_{scene}"
#     else:
#         log_dir = f"{runs_dir}/trial_syn_{scene}_{method}"

#     metric_data = {metric: []}

#     for file_name in os.listdir(log_dir):
#         if file_name.startswith('log'):
#             file_path = os.path.join(log_dir, file_name)
#             metric_values = extract_metric_value(file_path)

#             for epoch, value in enumerate(metric_values, start=1):
#                 metric_data[metric].append(value)

#     return pd.DataFrame(metric_data)

# if __name__ == "__main__":
#     indoor_scenes = ["mic", "ficus", "chair", "hotdog", "materials", "drums", "ship", "lego"]
#     outdoor_scenes = []
#     methods = ['', 'hsm']
#     methods_data = {}

#     fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
#     axes = axes.flatten()

#     for i, scenes in enumerate([indoor_scenes, outdoor_scenes]):
#         ax = axes[i]
#         scene_label = '(a) Indoors' if i == 0 else '(b) Outdoors'

#         for j, method in enumerate(methods):
#             method_data = pd.DataFrame()

#             for scene in scenes:
#                 scene_data = process_log_files(scene, method)
#                 method_data = pd.concat([method_data, scene_data], axis=1, ignore_index=True)

#             method_data = method_data
#             methods_data[method] = method_data

    #         # Use different line styles and colors
    #         line_style = '--' if j == 1 else '-'
    #         line_color = 'b' if j == 0 else 'r'

    #         # Plot the data for each method
    #         ax.plot(range(1, len(method_data) + 1), method_data, label=method, linestyle=line_style, color=line_color)

    #     ax.set_xlabel('Training Time (Minutes)', fontsize=10)
    #     ax.set_ylabel(metric, fontsize=10)
    #     ax.set_title(scene_label, fontsize=12)

    #     # Add grid lines
    #     ax.grid(True, linestyle='--', alpha=0.7)

    #     # Increase tick label font size
    #     ax.tick_params(axis='both', which='major', labelsize=10)

    # ax.legend(["Baseline", "Hard Point Sampling (Ours)"], loc="lower right", fontsize=12)


    # print(methods_data)
    # plt.tight_layout()
    # plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt

# # Read the CSV file into a DataFrame
# df = pd.read_csv('~/Downloads/avg_train_loss.csv')

# # Extract the columns of interest
# sample_mining_memory = df['Group: hsm - rgb loss'].dropna()[1:16700]
# print(sample_mining_memory)
# baseline_memory = df['Group: Random - rgb loss'].dropna()[1:9800]

# # Plot the data
# plt.figure(figsize=(8, 5))  # Adjust size for publication
# plt.plot(sample_mining_memory, label='Hard Sample Mining (Ours)', c="green")
# plt.plot(baseline_memory, label='Baseline', c="red")
# plt.xlabel('Iterations', fontsize=12)  # Increase font size for better readability
# plt.ylabel('Training Loss', fontsize=12)  # Increase font size for better readability
# # plt.title('Memory Usage per Iteration', fontsize=14, fontweight='bold')  # Add title with increased font size and bold style
# plt.legend(fontsize=10)  # Increase font size for legend
# plt.grid(True)
# plt.tight_layout()  # Adjust layout for better fit
# plt.savefig('memory_usage_plot.png', dpi=300)  # Save the plot as a high-resolution image
# plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# # Read the CSV file into a DataFrame
# df = pd.read_csv('~/Downloads/avg_iter_mem_usage.csv')

# # Extract the columns of interest
# # sample_mining_memory = df['Group: hsm - rgb loss'].dropna()[1000:15400]
# # baseline_memory = df['Group: Random - rgb loss'].dropna()[1000:8850]

# sample_mining_memory = df['Group: sample_mining - Memory Usage (MB)'].dropna()[1:15400]
# baseline_memory = df['Group: baseline - Memory Usage (MB)'].dropna()[1:8850]

# # Plot the data
# fig, ax = plt.subplots(figsize=(8, 5))  # Adjust size for publication
# ax.plot(sample_mining_memory, label='Hard Sample Mining (Ours)', c="green")
# ax.plot(baseline_memory, label='Baseline', c="red")
# ax.set_xlabel('Iterations', fontsize=12)  # Increase font size for better readability
# ax.set_ylabel('Memory Usage (MB)', fontsize=12)  # Increase font size for better readability

# # Add a marker to indicate skipped values on the x-axis
# ax.set_xticks([256, 5000, 10000, 15000])
# ax.set_xticklabels(['256', '5k', '10k', '15k'])  # Customize tick labels

# # ax.set_title('Memory Usage per Iteration', fontsize=14, fontweight='bold')  # Add title with increased font size and bold style
# ax.legend(fontsize=10)  # Increase font size for legend
# ax.grid(True)
# plt.tight_layout()  # Adjust layout for better fit

# plt.savefig('avg_mem_usage.png', dpi=300)  # Save the plot as a high-resolution image
# plt.show()



import os
import re
import pandas as pd

metric = "PSNR"

def extract_metric_value(file_path):
    metric_pattern = re.compile(fr'{re.escape(metric)} = (\d+(\.\d+)?)')
    metric_values = []

    with open(file_path, 'r') as file:
        for line in file:
            match = metric_pattern.search(line)
            if match:
                metric_values.append(float(match.group(1)))

    return metric_values

def process_log_files(scene, method):
    runs_dir = f"runs/longruns_iNGP_360_B=2to20"
    if method == "":
        log_dir = f"{runs_dir}/trial_ngp_ANR_360_{scene}"
    else:
        log_dir = f"{runs_dir}/trial_ngp_ANR_{method}_360_{scene}"

    metric_data = {metric: []}

    for file_name in os.listdir(log_dir):
        if file_name.startswith('log'):
            file_path = os.path.join(log_dir, file_name)
            metric_values = extract_metric_value(file_path)

            for epoch, value in enumerate(metric_values, start=1):
                metric_data[metric].append(value)

    return pd.DataFrame(metric_data)

if __name__ == "__main__":
    # scenes = ["mic", "ficus", "chair", "hotdog", "materials", "drums", "ship", "lego"]
    scenes = ['bonsai', 'kitchen', 'room', 'counter', 'garden', 'bicycle', 'stump']
    methods = ['', 'OHPM']

    table_data = []
    mins = [9]

    for scene in scenes:
        row_data = []
        for method in methods:
            scene_data = process_log_files(scene, method)
            row_data.extend([scene_data.iloc[i][0] for i in mins]    )#[scene_data.iloc[0][0], scene_data.iloc[4][0]])

        print(row_data)
        row = [f"{row_data[len(row_data)//2+i]:.2f}~|~{row_data[i]:.2f}" for i in range(len(row_data)//2)]  # , f"{row_data[3]:.2f}~|~{row_data[2]:.2f}"]
        table_data.append(row)


    # Create DataFrame for table
    table_df = pd.DataFrame(table_data, columns=[f'{minute+1}~min' for minute in mins], index=scenes)

    # Print LaTeX table
    print(table_df.transpose().to_latex(index=True))