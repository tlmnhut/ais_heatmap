import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import pathlib

from viz_heatmap import postprocess_img


def plot_aim1():
    # Your data
    # data = [
    #     [0.55, 0.62, 0.43, 0.49, 385, 445],
    #     [0.50, 0.50, 0.36, 0.39, 341, 493],
    #     [0.28, 0.35, 0.31, 0.38, 322, 397],
    #     [0.25, 0.34, 0.19, 0.25, 400, 305],
    #     [0.42, 0.56, 0.45, 0.53, 226, 296],
    #     [0.32, 0.32, 0.36, 0.39, 252, 365]
    # ]
    #
    # error_data = [
    #     [0.014, 0.014, 0.011, 0.011],
    #     [0.006, 0.006, 0.006, 0.006],
    #     [0.014, 0.014, 0.015, 0.015],
    #     [0.018, 0.018, 0.011, 0.011],
    #     [0.025, 0.025, 0.018, 0.018],
    #     [0.007, 0.007, 0.011, 0.011]
    # ]

    data = [
        [0.54, 0.79, 0.46, 0.71, 208, 376],
        [0.40, 0.52, 0.42, 0.47, 251, 421],
        [0.51, 0.60, 0.51, 0.61, 239, 366],
        [0.42, 0.56, 0.36, 0.48, 329, 303],
        [0.38, 0.61, 0.49, 0.64, 264, 168],
        [0.48, 0.61, 0.51, 0.64, 223, 205]
    ]

    error_data = [
        [0.041, 0.041, 0.040, 0.040],
        [0.019, 0.019, 0.009, 0.009],
        [0.016, 0.016, 0.015, 0.015],
        [0.022, 0.022, 0.019, 0.019],
        [0.037, 0.037, 0.025, 0.025],
        [0.021, 0.021, 0.022, 0.022]
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
        ax.text(bar_positions[i] + 1.5 * bar_width, 0.85,
                f"{int(num1)}", ha='center', va='bottom', color='blue')

        ax.text(bar_positions[i] + 3.5 * bar_width, 0.85,
                f"{int(num2)}", ha='center', va='bottom', color='green')

    ax.set_xticks(bar_positions + 2.5 * bar_width)
    ax.set_xticklabels(x_labels, fontsize=11.5)
    # ax.set_xlabel('Dataset')
    ax.set_ylabel('Prediction (Spearman\'s $\\rho$)', fontsize=15)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.set_ylim(0, 0.9)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)

    # ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))

    plt.tight_layout()
    plt.savefig('./figures/aim_1_40_spearman.png')


def resize_map_transalnet_ais(trained_dataset = 'ecoset'):
    dataset_names = ['animals', 'automobiles', 'fruits', 'furniture', 'various', 'vegetables']
    proxy_heatmap_all, predict_heatmap_all = [], []
    for dataset in dataset_names:
        proxy_heatmap_list = np.load(f"./data/peterson/transalnet_dense/sal_mat/{dataset}.npy") # here the transalnet heatmap
        my_heatmap_list = np.load(f'./res/heatmaps/{trained_dataset}/remove_fm/{dataset}.npy') # here the AIS heatmap
        img_list = sorted(list(pathlib.Path(f'./data/peterson/original/{dataset}/images').glob('*'))) # here the dir to the RGB images,
        # because each image may have different shapes (animals 300x300 the rest 500x500), and transalnet returns the shape 288x384,
        # our AIS heatmaps return the shape 14x14, we need the original image shape to scale the 2 heatmaps

        proxy_heatmap, predict_heatmap = [], []
        for idx in range(len(my_heatmap_list)):
            rgb_image = Image.open(img_list[idx])
            proxy_heatmap.append(postprocess_img(proxy_heatmap_list[idx], rgb_image.size))
            predict_heatmap.append(cv2.resize(my_heatmap_list[idx], rgb_image.size))
        proxy_heatmap_all.append(proxy_heatmap)
        predict_heatmap_all.append(predict_heatmap)
    proxy_heatmap_all = np.array(proxy_heatmap_all, dtype=object)
    predict_heatmap_all = np.array(predict_heatmap_all, dtype=object)
    np.save(f'./res/tmp/{trained_dataset}_transalnet_heatmap', proxy_heatmap_all)
    np.save(f'./res/tmp/{trained_dataset}_ais_heatmap', predict_heatmap_all)


# def predict_hsj(activations, hsim):
#     # dnn_rsm = np.corrcoef(activations)
#     dnn_rsm = cosine_similarity(activations)
#     dnn_rsm = upper_tri(dnn_rsm)
#     hsim = upper_tri(hsim)
#     # r2 = pearsonr(dnn_rsm, hsim)[0] ** 2
#     # r2 = cosine_similarity(dnn_rsm.reshape(1, -1), hsim.reshape(1, -1))[0, 0] ** 2
#     rho = spearmanr(dnn_rsm, hsim)[0]
#     return rho
#
#
# def corr_modify(penultimate_mat, hsim_mat):
#     hsim_mat = upper_tri(hsim_mat)
#     corr_list = []
#     for i in range(penultimate_mat.shape[1]):
#         # dnn_rsm = np.corrcoef(penultimate_mat[:, i, :])
#         dnn_rsm = cosine_similarity(penultimate_mat[:, i, :])
#         dnn_rsm = upper_tri(dnn_rsm)
#         # r2_list.append(pearsonr(dnn_rsm, hsim_mat)[0] ** 2)
#         # r2_list.append(cosine_similarity(dnn_rsm.reshape(1, -1), hsim_mat.reshape(1, -1))[0, 0] ** 2)
#         corr_list.append(spearmanr(dnn_rsm, hsim_mat)[0])
#     return np.array(corr_list)
#
#
# def analyze_cv(trained_dataset = 'ecoset', keep_or_remove='remove'):
#     dataset_names = ['animals', 'automobiles', 'fruits', 'furniture', 'various', 'vegetables']
#
#     fig, axs = plt.subplots(6, 5, figsize=(15, 12), sharex=True, sharey=True)
#     axs_flat = axs.flatten()
#
#     for dataset in dataset_names:
#         cv_r2 = np.load(f'./res/corr/{trained_dataset}/{keep_or_remove}_cv/cv_{dataset}_cosine_cosine.npy')
#         train_max, test_max, train_original, test_original, max_idx = [], [], [], [],[]
#         for fold in range(cv_r2.shape[0]):
#             train_max_idx = np.nanargmax(cv_r2[fold][0])
#             max_idx.append(train_max_idx)
#             train_max.append(cv_r2[fold][0][train_max_idx])
#             test_max.append(cv_r2[fold][1][train_max_idx])
#             # print(train_max_idx)
#             train_original.append(cv_r2[fold][0][-1])
#             test_original.append(cv_r2[fold][1][-1])
#
#             ax = axs[dataset_names.index(dataset), fold]
#             ax.plot(np.arange(cv_r2.shape[2]), cv_r2[fold][0], label='train')
#             ax.plot(np.arange(cv_r2.shape[2]), cv_r2[fold][1], label='test')
#             ax.set_title(f'{dataset}, fold {fold}')
#             ax.grid(True)
#             ax.axvline(train_max_idx, color='red', linestyle='--')
#
#         print(dataset, '-',
#               'train full', round(np.mean(train_original), 4), round(np.std(train_original), 4), '-',
#               'train max', round(np.mean(train_max), 4), round(np.std(train_max), 4), '-',
#               'test full', round(np.mean(test_original), 4), round(np.std(test_original), 4), '-',
#               'test max', round(np.mean(test_max), 4), round(np.std(test_max), 4), '-',
#               'max idx', round(np.mean(max_idx)), round(np.std(max_idx)))
#
#         axs_flat[0].legend()
#         plt.tight_layout()
#         plt.savefig(f'./figures/{trained_dataset}/cv_{keep_or_remove}_cosine_cosine.png')
