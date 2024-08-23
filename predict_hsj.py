import pathlib

import numpy as np
from scipy.stats import pearsonr, spearmanr, ttest_rel
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

from extract_acts import extract_acts_batch, extract_acts_from_last_conv
from model import VGG16Prune


def upper_tri(r):
    # Extract off-diagonal elements of each Matrix
    ioffdiag = np.triu_indices(r.shape[0], k=1)  # indices of off-diagonal elements
    r_offdiag = r[ioffdiag]
    return r_offdiag


def analyze_cv_2(trained_dataset='ecoset', keep_or_remove='remove'):
    dataset_names = ['animals', 'automobiles', 'fruits', 'furniture', 'various', 'vegetables']
    # test_original_111, test_max_111 = [], []
    for dataset in dataset_names:
        train_max_all, test_max_all, train_original_all, test_original_all, max_idx_all = [], [], [], [], []
        for seed in range(111, 999, 111): # 999
            cv_r2 = np.load(f'./res/corr/{trained_dataset}/{keep_or_remove}_cv/seed_{seed}/cv_{dataset}.npy')
            train_max, test_max, train_original, test_original, max_idx = [], [], [], [],[]
            for fold in range(cv_r2.shape[0]):
                train_max_idx = np.nanargmax(cv_r2[fold][0])
                max_idx.append(train_max_idx)
                train_max.append(cv_r2[fold][0][train_max_idx])
                test_max.append(cv_r2[fold][1][train_max_idx])
                # print(train_max_idx)
                train_original.append(cv_r2[fold][0][-1])
                test_original.append(cv_r2[fold][1][-1])
            train_max_all.append(train_max)
            test_max_all.append(test_max)
            train_original_all.append(train_original)
            test_original_all.append(test_original)
            max_idx_all.append(max_idx)
        train_max_all = np.array(train_max_all).flatten()
        test_max_all = np.array(test_max_all).flatten()
        train_original_all = np.array(train_original_all).flatten()
        test_original_all = np.array(test_original_all).flatten()
        max_idx_all = np.array(max_idx_all).flatten()

        # test_original_111.append(test_original_all)
        # test_max_111.append(test_max_all)
        # print(test_max_all.shape)
        print(dataset, '-',
              'train full', round(np.mean(train_original_all), 2), round(np.std(train_original_all), 2), '-',
              'train max', round(np.mean(train_max_all), 2), round(np.std(train_max_all), 2), '-',
              'test full', round(np.mean(test_original_all), 2), round(np.std(test_original_all), 2), '-',
              'test max', round(np.mean(test_max_all), 2), round(np.std(test_max_all), 2), '-',
              'max idx', round(np.mean(max_idx_all)), round(np.std(max_idx_all)), '-',
              'se', round(np.sqrt((np.sum((test_max_all - test_original_all) ** 2)) / (40* (40-1))), 3))
        print(dataset, ttest_rel(test_max_all, test_original_all, alternative='greater'))
    # return test_original_111, test_max_111



if __name__ == '__main__':
    trained_dataset = 'ecoset'
    hsim = np.load('./data/peterson/hsim_peterson.npz')
    dataset_names = hsim.files

    model = VGG16Prune(trained_dataset=trained_dataset)
    if torch.cuda.is_available():
        model.to('cuda')
    model.eval()

    n_fold = 5
    upper_triangle_size = (120**2 - 120) // 2
    # n_sample = 120
    # seed_value = 111
    for seed_value in [333, 444, 555, 666, 777, 888]:
        np.random.seed(seed_value)
        test_idx = np.arange(upper_triangle_size)
        np.random.shuffle(test_idx)
        test_idx = test_idx.reshape(n_fold, -1)
        train_idx = np.array([list(set(range(upper_triangle_size)) - set(chunk)) for chunk in test_idx])

        pathlib.Path(f'./res/corr/{trained_dataset}/original_cv/seed_{seed_value}').mkdir(parents=True, exist_ok=True)
        pathlib.Path(f'./res/corr/{trained_dataset}/remove_cv/seed_{seed_value}').mkdir(parents=True, exist_ok=True)

        # r2 = []
        for dataset in tqdm(dataset_names):
            hsim_flatten = upper_tri(hsim[dataset])
            penultimate_original = np.load(f'./res/acts/{trained_dataset}/original/vgg16_peterson_{dataset}_penultimate.npy')
            rsm_original_flatten = upper_tri(cosine_similarity(penultimate_original))
            penultimate_remove = np.load(f'./res/acts/{trained_dataset}/remove/vgg16_peterson_{dataset}_penultimate.npy')
            rsm_remove_flatten = [upper_tri(cosine_similarity(penultimate_remove[:, i, :])) for i in range(len(penultimate_remove[1]))]

            # find the ranking
            rho_fold_original, rho_fold_remove = [], []
            for fold in range(n_fold):
                rho_fold_original.append(spearmanr(rsm_original_flatten[train_idx[fold]], hsim_flatten[train_idx[fold]])[0])
                rho_fold_remove.append([spearmanr(rsm_remove_flatten[i][train_idx[fold]], hsim_flatten[train_idx[fold]])[0]
                                        for i in range(len(penultimate_remove[1]))])
            rho_fold_original = np.array(rho_fold_original)
            rho_fold_remove = np.array(rho_fold_remove)
            np.save(f'./res/corr/{trained_dataset}/original_cv/seed_{seed_value}/train_{dataset}', rho_fold_original)
            np.save(f'./res/corr/{trained_dataset}/remove_cv/seed_{seed_value}/train_{dataset}', rho_fold_remove)

            # SFS
            last_conv_acts = np.load(f'./res/acts/{trained_dataset}/original/vgg16_peterson_{dataset}_last_conv.npy')
            rho_fold_change = rho_fold_original[:, np.newaxis] - rho_fold_remove
            rho_fold_train_test = []
            for fold in range(n_fold):
                order = np.argsort(rho_fold_change[fold])[::-1]
                train_rho_list, test_rho_list = [], []
                for fm_idx in range(len(order)):
                    mask = np.zeros_like(rho_fold_change[fold])
                    mask[order[:fm_idx+1]] = 1
                    penultimate = extract_acts_from_last_conv(model=model,
                                                     last_conv=last_conv_acts,
                                                     batch_size=120,
                                                     mask=mask.astype(np.uint8))
                    rsm_penultimate_flatten = upper_tri(cosine_similarity(penultimate))
                    train_rho_list.append(spearmanr(rsm_penultimate_flatten[train_idx[fold]], hsim_flatten[train_idx[fold]])[0])
                    test_rho_list.append(spearmanr(rsm_penultimate_flatten[test_idx[fold]], hsim_flatten[test_idx[fold]])[0])
                rho_fold_train_test.append([train_rho_list, test_rho_list])
            np.save(f'./res/corr/{trained_dataset}/remove_cv/seed_{seed_value}/cv_{dataset}', np.array(rho_fold_train_test))

# penultimate_check = []
# for i in range(512):
#     mask = np.zeros(512)
#     mask[i] = 1
#     penultimate = extract_acts_from_last_conv(model=model,
#                                               last_conv=last_conv_acts,
#                                               batch_size=120,
#                                               mask=mask.astype(np.uint8))
#     penultimate_check.append(penultimate)

#     dnn_acts = np.load(f'./res/acts/{trained_dataset}/original/vgg16_peterson_{dataset}_penultimate.npy')
#     # r2 = []
#     # for i in range(512):
#     #     r2.append(predict_hsj(activations=dnn_acts[:, i, :], hsim=hsim[dataset]))
#     #     np.save(f'./res/corr/{trained_dataset}/remove_collective/{dataset}', np.array(r2))
#     r2.append(predict_hsj(activations=dnn_acts, hsim=hsim[dataset]))
# np.save(f'./res/corr/{trained_dataset}/original/all_datasets', np.array(r2))

# r2_fold_modified = np.load(f'./res/corr/{trained_dataset}/remove_cv/train_{dataset}.npy')
# r2_fold_original = np.load(f'./res/corr/{trained_dataset}/original_cv/train_{dataset}.npy')