import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # Your data
    data = [
        [0.55, 0.62, 0.43, 0.49, 385, 445],
        [0.50, 0.50, 0.36, 0.39, 341, 493],
        [0.28, 0.35, 0.31, 0.38, 322, 397],
        [0.25, 0.34, 0.19, 0.25, 400, 305],
        [0.42, 0.56, 0.45, 0.53, 226, 296],
        [0.32, 0.32, 0.36, 0.39, 252, 365]
    ]

    # error_data = [
    #     [0.07, 0.08, 0.08, 0.08, 37, 28],
    #     [ 0.08, 0.08, 0.09, 0.10, 52, 5],
    #     [ 0.08, 0.08, 0.07, 0.08, 42, 44],
    #     [ 0.08, 0.09, 0.08, 0.07, 51, 52],
    #     [ 0.09, 0.09, 0.10, 0.11, 36, 43],
    #     [ 0.07, 0.07, 0.07, 0.08, 68, 85]
    # ]

    error_data = [
        [0.014, 0.014, 0.011, 0.011],
        [0.006, 0.006, 0.006, 0.006],
        [0.014, 0.014, 0.015, 0.015],
        [0.018, 0.018, 0.011, 0.011],
        [0.025, 0.025, 0.018, 0.018],
        [0.007, 0.007, 0.011, 0.011]
    ]

    # Extracting the first 4 columns for each row
    x_labels = ["Animals", "Trans.", "Fruits", "Furniture", "Various", "Vegetables"]
    bars_data = np.array(data)[:, :4]

    # Extracting matching numbers from columns 5 and 6 in 'data'
    matching_numbers = np.array(data)[:, [4, 5]]

    # Define colors for each column
    column_colors = ['lightblue', 'blue', 'lightgreen', 'green']

    # Plotting the bar graph with assigned colors, error bars, and matching numbers
    fig, ax = plt.subplots(figsize=(7.5, 5.3))
    plt.rcParams.update({'font.size': 10.5})


    bar_width = 0.2
    bar_positions = np.arange(len(x_labels))

    legend_labels = ['ImageNet Full', 'ImageNet Retained', 'EcoSet Full', 'EcoSet Retained']

    for i in range(4):
        ax.bar(bar_positions + i * bar_width, bars_data[:, i], yerr=[err[i] for err in error_data],
               width=bar_width, label=legend_labels[i], color=column_colors[i], capsize=5)


    # Adding matching numbers from columns 5 and 6 in 'data' above the error bars
    for i, (num1, num2) in enumerate(matching_numbers):
        ax.text(bar_positions[i] + 1.5 * bar_width, 0.65,
                f"{int(num1)}", ha='center', va='bottom', color='blue')

        ax.text(bar_positions[i] + 3.5 * bar_width, 0.65,
                f"{int(num2)}", ha='center', va='bottom', color='green')

    ax.set_xticks(bar_positions + 2.5 * bar_width)
    ax.set_xticklabels(x_labels, fontsize=11.5)
    # ax.set_xlabel('Dataset')
    ax.set_ylabel('Prediction (Pearson\'s $R^2$)', fontsize=15)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.set_ylim(0, 0.7)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)

    # ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))

    plt.tight_layout()
    plt.savefig('./figures/aim_1_40_poster.png')