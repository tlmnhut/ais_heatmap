import pathlib

import numpy as np
from scipy.stats import pearsonr, ttest_rel
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


def predict_hsj(activations, hsim):
    dnn_rsm = np.corrcoef(activations)
    # dnn_rsm = cosine_similarity(activations)
    dnn_rsm = upper_tri(dnn_rsm)
    hsim = upper_tri(hsim)
    r2 = pearsonr(dnn_rsm, hsim)[0] ** 2
    # r2 = cosine_similarity(dnn_rsm.reshape(1, -1), hsim.reshape(1, -1))[0, 0] ** 2
    return r2


def r2_modify(penultimate_mat, hsim_mat):
    hsim_mat = upper_tri(hsim_mat)
    r2_list = []
    for i in range(penultimate_mat.shape[1]):
        dnn_rsm = np.corrcoef(penultimate_mat[:, i, :])
        # dnn_rsm = cosine_similarity(penultimate_mat[:, i, :])
        dnn_rsm = upper_tri(dnn_rsm)
        r2_list.append(pearsonr(dnn_rsm, hsim_mat)[0] ** 2)
        # r2_list.append(cosine_similarity(dnn_rsm.reshape(1, -1), hsim_mat.reshape(1, -1))[0, 0] ** 2)
    return np.array(r2_list)


def analyze_cv(trained_dataset = 'ecoset', keep_or_remove='remove'):
    dataset_names = ['animals', 'automobiles', 'fruits', 'furniture', 'various', 'vegetables']

    fig, axs = plt.subplots(6, 5, figsize=(15, 12), sharex=True, sharey=True)
    axs_flat = axs.flatten()

    for dataset in dataset_names:
        cv_r2 = np.load(f'./res/corr/{trained_dataset}/{keep_or_remove}_cv/cv_{dataset}_cosine_cosine.npy')
        train_max, test_max, train_original, test_original, max_idx = [], [], [], [],[]
        for fold in range(cv_r2.shape[0]):
            train_max_idx = np.nanargmax(cv_r2[fold][0])
            max_idx.append(train_max_idx)
            train_max.append(cv_r2[fold][0][train_max_idx])
            test_max.append(cv_r2[fold][1][train_max_idx])
            # print(train_max_idx)
            train_original.append(cv_r2[fold][0][-1])
            test_original.append(cv_r2[fold][1][-1])

            ax = axs[dataset_names.index(dataset), fold]
            ax.plot(np.arange(cv_r2.shape[2]), cv_r2[fold][0], label='train')
            ax.plot(np.arange(cv_r2.shape[2]), cv_r2[fold][1], label='test')
            ax.set_title(f'{dataset}, fold {fold}')
            ax.grid(True)
            ax.axvline(train_max_idx, color='red', linestyle='--')

        print(dataset, '-',
              'train full', round(np.mean(train_original), 4), round(np.std(train_original), 4), '-',
              'train max', round(np.mean(train_max), 4), round(np.std(train_max), 4), '-',
              'test full', round(np.mean(test_original), 4), round(np.std(test_original), 4), '-',
              'test max', round(np.mean(test_max), 4), round(np.std(test_max), 4), '-',
              'max idx', round(np.mean(max_idx)), round(np.std(max_idx)))

        axs_flat[0].legend()
        plt.tight_layout()
        plt.savefig(f'./figures/{trained_dataset}/cv_{keep_or_remove}_cosine_cosine.png')


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
        # print(dataset, '-',
        #       'train full', round(np.mean(train_original_all), 2), round(np.std(train_original_all), 2), '-',
        #       'train max', round(np.mean(train_max_all), 2), round(np.std(train_max_all), 2), '-',
        #       'test full', round(np.mean(test_original_all), 2), round(np.std(test_original_all), 2), '-',
        #       'test max', round(np.mean(test_max_all), 2), round(np.std(test_max_all), 2), '-',
        #       'max idx', round(np.mean(max_idx_all)), round(np.std(max_idx_all)), '-',
        #       'se', round(np.sqrt((np.sum((test_max_all - test_original_all) ** 2)) / (40* (40-1))), 3))
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

    n_sample = 120
    # seed_value = 111
    for seed_value in [333, 444, 555, 666, 777, 888]:
        np.random.seed(seed_value)
        test_idx = np.arange(n_sample)
        np.random.shuffle(test_idx)
        test_idx = test_idx.reshape(5, -1)
        train_idx = np.array([list(set(range(n_sample)) - set(chunk)) for chunk in test_idx])

        pathlib.Path(f'./res/corr/{trained_dataset}/original_cv/seed_{seed_value}').mkdir(parents=True, exist_ok=True)
        pathlib.Path(f'./res/corr/{trained_dataset}/remove_cv/seed_{seed_value}').mkdir(parents=True, exist_ok=True)

        # r2 = []
        for dataset in tqdm(dataset_names):
            hsim_dataset = hsim[dataset]
            penultimate_remove = np.load(f'./res/acts/{trained_dataset}/remove/vgg16_peterson_{dataset}_penultimate.npy')
            penultimate_original = np.load(f'./res/acts/{trained_dataset}/original/vgg16_peterson_{dataset}_penultimate.npy')

            # find the ranking
            r2_fold_original, r2_fold_remove = [], []
            for fold in range(5):
                train_original = penultimate_original[train_idx[fold], :]
                test_original = penultimate_original[test_idx[fold], :]
                train_remove = penultimate_remove[train_idx[fold], :]
                test_remove = penultimate_remove[test_idx[fold], :]
                train_hsim = hsim_dataset[train_idx[fold]][:, train_idx[fold]]
                test_hsim = hsim_dataset[test_idx[fold]][:, test_idx[fold]]

                r2_fold_original.append(predict_hsj(activations=train_original, hsim=train_hsim))
                r2_fold_remove.append(r2_modify(penultimate_mat=train_remove, hsim_mat=train_hsim))
            r2_fold_original = np.array(r2_fold_original)
            r2_fold_remove = np.array(r2_fold_remove)
            np.save(f'./res/corr/{trained_dataset}/original_cv/seed_{seed_value}/train_{dataset}', r2_fold_original)
            np.save(f'./res/corr/{trained_dataset}/remove_cv/seed_{seed_value}/train_{dataset}', r2_fold_remove)

            # SFS
            last_conv_acts = np.load(f'./res/acts/{trained_dataset}/original/vgg16_peterson_{dataset}_last_conv.npy')
            r2_fold_change = r2_fold_original[:, np.newaxis] - r2_fold_remove
            r2_fold_train_test = []
            for fold in range(5):
                train_hsim = hsim_dataset[train_idx[fold]][:, train_idx[fold]]
                test_hsim = hsim_dataset[test_idx[fold]][:, test_idx[fold]]

                order = np.argsort(r2_fold_change[fold])[::-1]
                train_r2_list, test_r2_list = [], []
                for fm_idx in range(len(order)):
                    mask = np.zeros_like(r2_fold_change[fold])
                    mask[order[:fm_idx+1]] = 1
                    penultimate = extract_acts_from_last_conv(model=model,
                                                     last_conv=last_conv_acts,
                                                     batch_size=120,
                                                     mask=mask.astype(np.uint8))
                    train_penultimate = penultimate[train_idx[fold], :]
                    test_penultimate = penultimate[test_idx[fold], :]
                    train_r2_list.append(predict_hsj(activations=train_penultimate, hsim=train_hsim))
                    test_r2_list.append(predict_hsj(activations=test_penultimate, hsim=test_hsim))
                r2_fold_train_test.append([train_r2_list, test_r2_list])
            np.save(f'./res/corr/{trained_dataset}/remove_cv/seed_{seed_value}/cv_{dataset}', np.array(r2_fold_train_test))

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