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
        perturbation_ecoset.append(np.load(f'./res/scores/ecoset/remove_individual/{dataset}_keep_negative.npy'))
        perturbation_imagenet.append(np.load(f'./res/scores/imagenet/remove_individual/{dataset}_keep_negative.npy'))
    perturbation_ecoset, perturbation_imagenet = np.array(perturbation_ecoset), np.array(perturbation_imagenet)

    # stat_ecoset, stat_imagenet = perturbation_ecoset.mean(axis=1), perturbation_imagenet.mean(axis=1)
    # stat_ecoset, stat_imagenet = np.log(stat_ecoset), np.log(stat_imagenet)
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
    stat_ecoset = np.apply_along_axis(lambda x: np.mean(np.abs(x - np.mean(x))), axis=2, arr=perturbation_ecoset)
    stat_imagenet = np.apply_along_axis(lambda x: np.mean(np.abs(x - np.mean(x))), axis=2, arr=perturbation_imagenet)

    for i in range(len(dataset_names)):
        print(dataset_names[i], ks_2samp(stat_ecoset[i], stat_imagenet[i]))

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
        axs_flat[i].set_xticklabels([round(j, 4) for j in bins[1:]], rotation=45, ha='center', fontsize=12)
        axs_flat[i].set_title(f"{dataset_names[i].replace('automobiles', 'transportation')}".capitalize(),
                              fontsize=17)
        axs_flat[i].grid(True)
        axs_flat[i].tick_params(axis='y', labelsize=12)
        # if i == 3 or i == 4 or i == 5:
        # axs_flat[i].set_xlabel('Entropy Column', fontsize=15)
        # axs_flat[i].set_xlabel('Variance Column', fontsize=15)
        if i == 0:
            # axs_flat[i].set_ylabel('Number of feature maps', fontsize=15)
            axs_flat[i].set_ylabel('Number of images', fontsize=15)
        axs_flat[i].xaxis.set_major_formatter(FuncFormatter(custom_formatter))
    fig.text(0.5, 0.015, 'Mean absolute deviation of rows', ha='center', va='center', fontsize=15)
    # fig.text(0.5, 0.015, 'Log average', ha='center', va='center', fontsize=15)
    axs_flat[0].legend(fontsize=15)
    plt.tight_layout()
    # plt.savefig(f'./figures/scores_avg_keep_negative.png')
    # plt.savefig(f'./figures/scores_entropy_c_only_positive.png')
    # plt.savefig(f'./figures/scores_entropy_r_only_positive.png')
    # plt.savefig(f'./figures/scores_var_c_keep_negative.png')
    plt.savefig(f'./figures/scores_mad_r_keep_negative_e.png')


def heatmap_corr():
    dataset_names = ['animals', 'automobiles', 'fruits', 'furniture', 'various', 'vegetables']
    corr_all = []
    for dataset in dataset_names:
        resize_shape = (500, 500)
        if dataset == 'animals':
            resize_shape = (300, 300)
        heatmap_ecoset = np.load(f'./res/heatmaps/ecoset/remove_individual/{dataset}.npy')
        heatmap_imagenet = np.load(f'./res/heatmaps/imagenet/remove_individual/{dataset}.npy')
        corr = []
        for i in range(120):
            heatmap_ecoset_resize = cv2.resize(heatmap_ecoset[i], resize_shape)
            heatmap_imagenet_resize = cv2.resize(heatmap_imagenet[i], resize_shape)
            corr.append(pearsonr(heatmap_ecoset_resize.flatten(), heatmap_imagenet_resize.flatten())[0])
        corr_all.append(corr)
    corr_all = np.array(corr_all)
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
    disagreement_entropy = np.max(np.vstack([sorted_entropy_ecoset, sorted_entropy_imagenet]), axis=0)

    # selected_idx = np.where(sorted_corr >= 0.4)[0]
    # corr_ecoset = pearsonr(sorted_corr[selected_idx], sorted_entropy_ecoset[selected_idx])[0]
    # corr_imagenet = pearsonr(sorted_corr[selected_idx], sorted_entropy_imagenet[selected_idx])[0]
    corr_disagreement = pearsonr(sorted_corr, disagreement_entropy)[0]

    return corr_disagreement


if __name__ == '__main__':
    # rank_baseline(trained_dataset='ecoset')
    get_stats()

    # dataset_names = ['animals', 'automobiles', 'fruits', 'furniture', 'various', 'vegetables']
    # corr_heatmap = heatmap_corr()
    # for i in range(len(dataset_names)):
    #     last_fc_ecoset = np.load(f'./res/acts/ecoset/original/vgg16_peterson_{dataset_names[i]}_last_fc.npy')
    #     last_fc_imagenet = np.load(f'./res/acts/imagenet/original/vgg16_peterson_{dataset_names[i]}_last_fc.npy')
    #     print(dataset_names[i], np.round(analyze_corr_heatmap(corr_heatmap[i], last_fc_ecoset, last_fc_imagenet), 2))

