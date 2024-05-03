import os
import pathlib

import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr

from utils import pearsonr_2d, get_behavior_data, get_fmri_data


def corr_modified_vs_original(modified_mat, original_mat):
    """
    Computes Pearson correlation coefficients between each of 512 penultimate layers from the modified model (size 4096)
    and the untouched original penultimate layers matrix (shape 120 x 4096).
    Repeat the procedure for all 120 images.

    Args:
    - modified_mat (numpy.ndarray): A 3D numpy.ndarray shape 120 x 512 x 4096.
    - original_mat (numpy.ndarray): A 2D numpy.ndarray shape 120 x 4096.

    Returns:
    numpy.ndarray: A 3D numpy.ndarray (shape 120 x 512 x 120).
    """
    corr_scores_all_img = []
    for i in tqdm(range(original_mat.shape[0])):  # 120
        corr_scores_each_img = []
        for j in range(modified_mat.shape[1]):  # 512
            corr_scores = pearsonr_2d(modified_mat[i, j], original_mat)
            corr_scores_each_img.append(corr_scores)
        corr_scores_all_img.append(corr_scores_each_img)
    return np.array(corr_scores_all_img)


# def compute_perturbation(original_scores, modified_scores, baseline_scores):
#     """
#     Compute perturbation scores.
#
#     Args:
#     - original_scores (2D numpy.ndarray shape 144 x 144): Similarity matrix of original untouched model.
#     - modified_scores (3D numpy.ndarray shape 144 x 512 x 144): Correlation between penultimate layers of modified model
#     and original untouched model.
#     - baseline_scores (2D numpy.ndarray shape 144 x 144): Behavior or fMRI data.
#
#     Returns:
#     perturbation_all_img (2D numpy.ndarray shape 144 x 512): Perturbation scores for each feature map for each image.
#     """
#     perturbation_all_img = []
#     for i in range(baseline_scores.shape[0]):  # 144
#         #
#         corr_modified = pearsonr_2d(baseline_scores[i], 1 - modified_scores[i])
#         corr_modified = corr_modified ** 2
#         #
#         corr_original = pearsonr(baseline_scores[i], 1 - original_scores[i])[0] ** 2  # TODO perhaps we don't need it
#         #
#         perturbation_one_img = np.abs(corr_original - corr_modified)  # TODO remove_collective too
#         # perturbation_one_img = corr_original - corr_modified
#
#         perturbation_one_img = (perturbation_one_img - np.min(perturbation_one_img)) / (
#                 np.max(perturbation_one_img) - np.min(perturbation_one_img))  # TODO normalization
#         perturbation_all_img.append(perturbation_one_img)
#     return np.array(perturbation_all_img)


# this function is important
def compute_perturbation(second_oi_original, first_oi_modified, corr_supervised):
    """
    Compute perturbation scores.

    Args:
    - original_rsm (2D numpy.ndarray shape 120 x 120): Similarity matrix of original untouched model.
    - modified_rsm (3D numpy.ndarray shape 120 x 512 x 120): Correlation between penultimate layers of the modified model
    and the original untouched model.
    - supervised_rsm (2D numpy.ndarray shape 120 x 120): Human data.

    Returns:
    perturbation_all_img (2D numpy.ndarray shape 120 x 512): Perturbation scores for each feature map for each image.
    """
    perturbation_all_img = []
    for i in range(corr_supervised.shape[0]):  # 120 images
        # second_oi_modified = pearsonr_2d(corr_supervised[i], first_oi_modified[i])
        second_oi_modified = pearsonr_2d(np.delete(corr_supervised[i], i), np.delete(first_oi_modified[i], i, axis=1))
        # corr_modified = pearsonr_2d(original_rsm[i], modified_rsm[i])
        # corr_original = pearsonr(baseline_rsm[i], original_rsm[i])[0]

        # perturbation_one_img = corr_original - corr_modified
        # perturbation_one_img = 1 - corr_modified
        # perturbation_one_img = corr_modified
        # perturbation_one_img = - corr_modified
        # perturbation_one_img = second_oi_original[i] - second_oi_modified
        perturbation_one_img = second_oi_original[i] - second_oi_modified

        # perturbation_one_img = perturbation_one_img * (perturbation_one_img > 0) # only keep > 0 values
        # if np.sum(perturbation_one_img):
        #     perturbation_one_img = perturbation_one_img / np.sum(perturbation_one_img) # normalize

        # perturbation_one_img = (perturbation_one_img - np.min(perturbation_one_img)) / (
        #         np.max(perturbation_one_img) - np.min(perturbation_one_img))  # min max normalization
        perturbation_all_img.append(perturbation_one_img)

    return np.array(perturbation_all_img)


# if __name__ == '__main__':
#     # # Compute the correlations between removed-1 and original feature maps
#     # original_penultimate = np.load('./res/activations/original_penultimate.npy')
#     # remove1_penultimate = np.load('./res/activations/remove1_penultimate.npy')
#     # corr_remove1 = corr_modified_vs_original(modified_mat=remove1_penultimate,
#     #                                          original_mat=original_penultimate)
#     # np.save('./res/sim_mat/remove1_vs_original_penultimate', corr_remove1)
#
#     # # Compute perturbation scores
#
#     corr_original_penultimate = np.load('./res/sim_mat/original_penultimate.npy')
#     corr_remove1_vs_original = np.load('./res/sim_mat/remove1_vs_original_penultimate.npy')
#     fmri_rdm = get_behavior_data('set1')  # TODO change to fMRI FFA for faces, PPA for places, vTC
#     perturbation_no_square = compute_perturbation_only_positive_no_abs(original_scores=corr_original_penultimate,
#                                                                        modified_scores=corr_remove1_vs_original,
#                                                                        baseline_scores=fmri_rdm,
#                                                                        )
#     np.save('./res/scores/remove1_only_negative_no_abs.npy', perturbation_no_square)



if __name__ == '__main__':
    trained_dataset = 'imagenet'
    peterson_rsm = np.load("./data/peterson/hsim_peterson.npz")
    dataset_names = ['animals', 'automobiles', 'fruits', 'furniture', 'various', 'vegetables']
    for dataset in dataset_names:
        corr_human = peterson_rsm[dataset]

        original_penultimate = np.load(f'./res/acts/{trained_dataset}/original/vgg16_peterson_{dataset}_penultimate.npy')
        # remove_penultimate = np.load(f'./res/acts/{trained_dataset}/remove/vgg16_peterson_{dataset}_penultimate.npy')
        # # keep_penultimate = np.load(f'./res/acts/{trained_dataset}/keep/vgg16_peterson_{dataset}_penultimate.npy')
        # corr_modified = corr_modified_vs_original(modified_mat=remove_penultimate, original_mat=original_penultimate)
        # np.save(f'./res/corr/{trained_dataset}/remove_individual/{dataset}', corr_modified)

        # original_rsm = np.corrcoef(original_penultimate)
        # corr_baseline_individual = []
        # for i in range(original_rsm.shape[0]):
        #     corr_baseline_individual.append(pearsonr(np.delete(original_rsm[i], i), np.delete(corr_human[i], i))[0])
        #     # corr_baseline_individual.append(pearsonr(original_rsm[i], corr_human[i])[0])
        # np.save(f'./res/corr/{trained_dataset}/original_individual/{dataset}', np.array(corr_baseline_individual))

        corr_baseline_individual = np.load(f'./res/corr/{trained_dataset}/original_individual/{dataset}.npy')
        corr_remove_individual = np.load(f'./res/corr/{trained_dataset}/remove_individual/{dataset}.npy')
        perturbation_scores = compute_perturbation(second_oi_original=corr_baseline_individual,
                                                   first_oi_modified=corr_remove_individual,
                                                   corr_supervised=corr_human)
        np.save(f'./res/scores/{trained_dataset}/remove_individual/{dataset}_keep_negative', perturbation_scores)

        # all_images = sorted([path.name for path in pathlib.Path(f"./data/peterson/original/animals/images/").glob("*")])
        # selected_images = sorted([path.name for path in pathlib.Path(f"./data/peterson/sub_category/animals/reptile").glob("*")] )#+
        #                          #[path.name for path in pathlib.Path(f"./data/peterson/sub_category/animals/amphibian").glob("*")])
        # selected_idx = [all_images.index(path) for path in selected_images]
        # corr_original_penultimate_selected = corr_original_penultimate[selected_idx, :][:, selected_idx]
        # corr_keep_vs_original_selected = corr_keep_vs_original[selected_idx, :, :][:, :, selected_idx]
        # corr_human_selected = corr_human[selected_idx, :][:, selected_idx]
        # only_positive_no_abs = compute_perturbation_only_positive_no_abs(original_rsm=corr_original_penultimate_selected,
        #                                                                  modified_rsm=corr_keep_vs_original_selected,
        #                                                                  baseline_rsm=corr_human_selected)
        # np.save(f'./res/scores/{trained_dataset}/keep/selected_reptile_only_positive_no_abs', only_positive_no_abs)

