import numpy as np
from scipy.stats import spearmanr, pearsonr, entropy, ks_2samp
from scipy.special import softmax
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FuncFormatter
import cv2


def rank_baseline(trained_dataset):
    dataset_names = ['animals', 'automobiles', 'fruits', 'furniture', 'various', 'vegetables']
    for dataset in dataset_names:
        conv = np.load(f'./res/acts/{trained_dataset}/original/vgg16_peterson_{dataset}_last_conv.npy')
        sum_acts = np.sum(conv, axis=(2, 3))
        highest_sum_idx = np.argmax(sum_acts, axis=1)
        sum_non_zero = np.count_nonzero(conv, axis=(2, 3))
        highest_non_zero_idx = np.argmax(sum_non_zero, axis=1)

        scores = np.load(f'./res/scores/{trained_dataset}/keep/{dataset}_dnn_corr_modified.npy')
        sort_scores_lh = np.argsort(scores, axis=1)

        highest_sum_position = []
        highest_non_zero_position = []
        spearman_rank = []
        for r in range(sort_scores_lh.shape[0]):
            highest_sum_position.append(np.where(sort_scores_lh[r] == highest_sum_idx[r])[0][0])
            highest_non_zero_position.append(np.where(sort_scores_lh[r] == highest_non_zero_idx[r])[0][0])
            spearman_rank.append(spearmanr(sum_non_zero[r], scores[r])[0])

        highest_sum_position = np.array(highest_sum_position)
        highest_non_zero_position = np.array(highest_non_zero_position)
        spearman_rank = np.array(spearman_rank)

        print(dataset, 512 - highest_sum_position.min(), 512 - highest_sum_position.max(),
              np.round(512 - highest_sum_position.mean(), 2), np.round(np.sum(highest_sum_position == 511) / 120 * 100, 2))
        print(dataset, 512 - highest_non_zero_position.min(), 512 - highest_non_zero_position.max(),
              np.round(512 - highest_non_zero_position.mean(), 2),
              np.round(np.sum(highest_non_zero_position == 511) / 120 * 100, 2), np.mean(spearman_rank))


def get_stats():
    dataset_names = ['animals', 'automobiles', 'fruits', 'furniture', 'various', 'vegetables']
    perturbation_ecoset, perturbation_imagenet = [], []
    for dataset in dataset_names:
        perturbation_ecoset.append(np.load(f'./res/scores/ecoset/remove_fm/{dataset}_pert_scores.npy'))
        perturbation_imagenet.append(np.load(f'./res/scores/imagenet/remove_fm/{dataset}_pert_scores.npy'))
    perturbation_ecoset, perturbation_imagenet = np.array(perturbation_ecoset), np.array(perturbation_imagenet)
    perturbation_ecoset = perturbation_ecoset * (perturbation_ecoset > 0)
    perturbation_imagenet = perturbation_imagenet * (perturbation_imagenet > 0)

    # # Correlation among the perturbation scores
    # print(np.corrcoef(perturbation_ecoset.mean(axis=1)))
    # print(np.corrcoef(perturbation_imagenet.mean(axis=1)))

    # Log avg
    stat_ecoset, stat_imagenet = perturbation_ecoset.mean(axis=1), perturbation_imagenet.mean(axis=1)
    stat_ecoset, stat_imagenet = np.log(stat_ecoset), np.log(stat_imagenet)

    # stat_ecoset, stat_imagenet = entropy(perturbation_ecoset, axis=1), entropy(perturbation_imagenet, axis=1) # col
    # stat_ecoset = np.apply_along_axis(lambda x: entropy(x[x > 0]), axis=1, arr=np.maximum(perturbation_ecoset, 0))
    # stat_imagenet = np.apply_along_axis(lambda x: entropy(x[x > 0]), axis=1, arr=np.maximum(perturbation_imagenet, 0))
    # stat_ecoset = np.apply_along_axis(lambda x: entropy(x[x < 0]), axis=1, arr=np.minimum(perturbation_ecoset, 0))
    # stat_imagenet = np.apply_along_axis(lambda x: entropy(x[x < 0]), axis=1, arr=np.minimum(perturbation_imagenet, 0))
    # stat_ecoset, stat_imagenet = entropy(perturbation_ecoset + np.abs(np.min(perturbation_ecoset)) + 1e-9, axis=2),\
    #     entropy(perturbation_imagenet + np.abs(np.min(perturbation_imagenet)) + 1e-9, axis=2) # row
    # stat_ecoset, stat_imagenet = perturbation_ecoset.max(axis=1), perturbation_imagenet.max(axis=1)
    # stat_ecoset, stat_imagenet = np.log(stat_ecoset), np.log(stat_imagenet)
    # stat_ecoset, stat_imagenet = np.var(perturbation_ecoset, axis=1), np.var(perturbation_imagenet, axis=1)

    # # MAD
    # stat_ecoset = np.apply_along_axis(lambda x: np.mean(np.abs(x - np.mean(x))), axis=1, arr=perturbation_ecoset)
    # stat_imagenet = np.apply_along_axis(lambda x: np.mean(np.abs(x - np.mean(x))), axis=1, arr=perturbation_imagenet)

    stats_sig = []
    for i in range(len(dataset_names)):
        stat_test = ks_2samp(stat_ecoset[i], stat_imagenet[i])
        print(dataset_names[i], stat_test)
        if stat_test[1] < 0.05:
            stats_sig.append(' (*)')
        else:
            stats_sig.append('')

    # max_stat = np.max([stat_ecoset, stat_imagenet])
    # min_stat = np.min([stat_ecoset, stat_imagenet])
    # bins = np.linspace(min_stat, max_stat, 11)

    def custom_formatter(x, pos):
        return f"{round(x*10000)}"
    fig, axs = plt.subplots(1, 6, figsize=(18, 4), sharey=True)#, sharex=True, sharey=True)
    axs_flat = axs.flatten()
    for i in range(len(dataset_names)):
        max_stat = np.max([stat_ecoset[i], stat_imagenet[i]])
        min_stat = np.min([stat_ecoset[i], stat_imagenet[i]])
        bins = np.linspace(min_stat, max_stat, 11)
        hist_ecoset, _ = np.histogram(stat_ecoset[i], bins=bins)
        hist_imagenet, _ = np.histogram(stat_imagenet[i], bins=bins)
        # bin_centers = 0.5 * (bins[:-1] + bins[1:])
        axs_flat[i].plot(bins[1:], hist_ecoset, label='EcoSet', color='blue')
        axs_flat[i].plot(bins[1:], hist_imagenet, label='ImageNet', color='orange')
        axs_flat[i].set_xticks(bins[1:])
        axs_flat[i].set_xticklabels([round(j, 2) for j in bins[1:]], rotation=45, ha='center', fontsize=12)
        axs_flat[i].set_title(f"{dataset_names[i].replace('automobiles', 'transportation')}".capitalize() + stats_sig[i],
                              fontsize=17)
        axs_flat[i].grid(True)
        axs_flat[i].tick_params(axis='y', labelsize=12)
        # if i == 3 or i == 4 or i == 5:
        # axs_flat[i].set_xlabel('Entropy Column', fontsize=15)
        # axs_flat[i].set_xlabel('Variance Column', fontsize=15)
        if i == 0:
            axs_flat[i].set_ylabel('Number of feature maps', fontsize=15)
            # axs_flat[i].set_ylabel('Number of images', fontsize=15)
        # axs_flat[i].xaxis.set_major_formatter(FuncFormatter(custom_formatter))
    # fig.text(0.5, 0.015, 'Mean absolute deviation of rows', ha='center', va='center', fontsize=15)
    # fig.text(0.5, 0.023, 'Log average', ha='center', va='center', fontsize=15)
    axs_flat[0].legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(f'./figures/august2024/aim_2_log_avg.png')
    # plt.savefig(f'./figures/scores_entropy_c_only_positive.png')
    # plt.savefig(f'./figures/scores_entropy_r_only_positive.png')
    # plt.savefig(f'./figures/scores_var_c_keep_negative.png')
    # plt.savefig(f'./figures/august2024/aim_2_mad_c_keep_negative.png')


def heatmap_corr():
    dataset_names = ['animals', 'automobiles', 'fruits', 'furniture', 'various', 'vegetables']
    corr_all = []
    for dataset in dataset_names:
        resize_shape = (500, 500)
        if dataset == 'animals':
            resize_shape = (300, 300)
        heatmap_ecoset = np.load(f'./res/heatmaps/ecoset/remove_fm/{dataset}.npy')
        heatmap_imagenet = np.load(f'./res/heatmaps/imagenet/remove_fm/{dataset}.npy')
        corr = []
        for i in range(120):
            heatmap_ecoset_resize = cv2.resize(heatmap_ecoset[i], resize_shape)
            heatmap_imagenet_resize = cv2.resize(heatmap_imagenet[i], resize_shape)
            corr.append(pearsonr(heatmap_ecoset_resize.flatten(), heatmap_imagenet_resize.flatten())[0])
            # corr.append(pearsonr(heatmap_ecoset.flatten(), heatmap_imagenet.flatten())[0])
        corr_all.append(corr)
    corr_all = np.array(corr_all)

    # Calculate the range for the bins based on the entire dataset
    # min_val = np.min(corr_all)
    # max_val = np.max(corr_all)
    # print(min_val, max_val)
    # Define the bins for the histogram
    # bins = np.linspace(min_val, max_val, 11)  # 20 bins
    bins = np.arange(-0.4, 1.2, 0.2)
    # Plot histograms for each row on the same axes
    plt.figure(figsize=(6, 5))
    for i in range(6):
        # Compute the histogram
        hist, bin_edges = np.histogram(corr_all[i], bins=bins, density=True)
        cum_hist = np.cumsum(hist)  # Cumulative sum of the histogram
        cum_hist_percentage = cum_hist / cum_hist[-1] * 100  # Convert to percentage
        # Plot the histogram as a line graph
        plt.plot(bin_edges[:-1], cum_hist_percentage, label=dataset_names[i].replace('automobiles', 'transportation').capitalize(),
                 marker='')  # bin_edges[:-1] to align with hist counts

    # Create custom x-axis labels as ranges
    bin_ranges = [f'({bin_edges[j]:.1f}, {bin_edges[j + 1]:.1f}]' for j in range(len(bin_edges) - 1)]
    plt.xticks(ticks=bin_edges[:-1], labels=bin_ranges, rotation=45, ha='right')

    # Add labels and title
    plt.xlabel('Correlation', fontsize=15)
    plt.ylabel('Percentage of images', fontsize=15)
    # plt.title('Histograms of 6 Rows')
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    plt.legend(loc='best', fontsize=15)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'./figures/august2024/aim_2_hist_corr_heatmap.png')

    return corr_all


def analyze_corr_heatmap(corr_heatmap, last_fc_ecoset, last_fc_imagenet):
    softmax_ecoset = softmax(last_fc_ecoset, axis=1)
    softmax_imagenet = softmax(last_fc_imagenet, axis=1)
    entropy_ecoset = entropy(softmax_ecoset, axis=1)
    entropy_imagenet = entropy(softmax_imagenet, axis=1)

    sorted_idx = np.argsort(corr_heatmap)
    sorted_corr = corr_heatmap[sorted_idx]
    sorted_entropy_ecoset = entropy_ecoset[sorted_idx]
    sorted_entropy_imagenet = entropy_imagenet[sorted_idx]
    disagreement_entropy = np.max(np.vstack([sorted_entropy_ecoset, sorted_entropy_imagenet]), axis=0) # either one of the two

    # selected_idx = np.where(sorted_corr >= 0.4)[0]
    # corr_ecoset = pearsonr(sorted_corr[selected_idx], sorted_entropy_ecoset[selected_idx])[0]
    # corr_imagenet = pearsonr(sorted_corr[selected_idx], sorted_entropy_imagenet[selected_idx])[0]
    corr_disagreement = pearsonr(sorted_corr, disagreement_entropy)[0]

    return corr_disagreement


if __name__ == '__main__':
    # rank_baseline(trained_dataset='ecoset')
    get_stats()

    # corr_heatmap = heatmap_corr()
    # print(np.mean(corr_heatmap, axis=1), np.std(corr_heatmap, axis=1))

    # dataset_names = ['animals', 'automobiles', 'fruits', 'furniture', 'various', 'vegetables']
    # for i in range(len(dataset_names)):
    #     last_fc_ecoset = np.load(f'./res/acts/ecoset/original/vgg16_peterson_{dataset_names[i]}_last_fc.npy')
    #     last_fc_imagenet = np.load(f'./res/acts/imagenet/original/vgg16_peterson_{dataset_names[i]}_last_fc.npy')
    #     print(dataset_names[i], np.round(analyze_corr_heatmap(corr_heatmap[i], last_fc_ecoset, last_fc_imagenet), 2))

