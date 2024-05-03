import pathlib
import cv2
import numpy as np
from scipy.io import loadmat
from PIL import Image
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler


def get_behavior_data(dataset_name, trim=False):
    """
    Load the behavioral data.

    Args:
    - dataset_name (str): either 'set1' or 'set2'.
    - trim (bool): If True, returns the averaged behavioral data without restoring it to the 2D RDM.

    Returns:
    avg_rdm (array): Averaged 2D Representational Dissimilarity Matrix (RDM) of the behavioral data.
    If trim=True, returns a 1D array of the averaged dissimilarity values.
    """

    behavior_data = loadmat('./data/BEHAVIOR.mat', simplify_cells=True)
    behavior_data = behavior_data['BEHAVIOR'][dataset_name]['pairwisedistances']

    avg_data = np.mean(behavior_data, axis=0)

    if trim:
        return avg_data

    # restore the 2D RDM. First, fill dissimilarity values in the upper triangle part.
    # Then, copy the upper triangle part to the lower triangle part.
    chunk_list = []
    start_idx = 0
    avg_data = list(avg_data)
    for i in range(143, -1, -1):
        chunk = [0] * (144 - i) + avg_data[start_idx:start_idx + i]
        chunk_list.append(chunk)
        start_idx += i
    avg_rdm = np.array(chunk_list)
    avg_rdm += avg_rdm.T

    return avg_rdm


def get_fmri_data(dataset_name, brain_area, trim=False):
    """
    Load the fMRI data.

    Args:
    - dataset_name (str): either 'set1' or 'set2'.
    - brain_area (str): specify the name of a brain area to get data from, e.g. 'FFA', 'PPA', 'vTC', etc.
    - trim (bool): If True, returns the averaged fMRI data without restoring it to the 2D RDM.

    Returns:
    avg_rdm (array): Averaged 2D Representational Dissimilarity Matrix (RDM) of the fMRI data.
    If trim=True, returns a 1D array of the averaged dissimilarity values.
    """

    fmri_data = loadmat('./data/FMRI.mat', simplify_cells=True)
    fmri_data = fmri_data['FMRI'][brain_area][dataset_name]['pairwisedistances']

    avg_data = np.mean(fmri_data, axis=0)

    if trim:
        return avg_data

    # restore the 2D RDM. First, fill dissimilarity values in the upper triangle part.
    # Then, copy the upper triangle part to the lower triangle part.
    chunk_list = []
    start_idx = 0
    avg_data = list(avg_data)
    for i in range(143, -1, -1):
        chunk = [0] * (144 - i) + avg_data[start_idx:start_idx + i]
        chunk_list.append(chunk)
        start_idx += i
    avg_rdm = np.array(chunk_list)
    avg_rdm += avg_rdm.T

    return avg_rdm


def compute_ori_dnn_sim_mat():
    """
    Compute the similarity matrix between the penultimate layer activations of the original DNN
    model and saves it to a numpy file. The shape of the result matrix is 144 x 144.
    """

    penultimate = np.load('./res/activations/original_penultimate.npy')
    sim_mat = np.corrcoef(penultimate)
    np.save('./res/sim_mat/original_penultimate.npy', sim_mat)


def pearsonr_2d(x, y):
    # credit: https://stackoverflow.com/a/64660866
    """
    Computes pearson correlation coefficient where x is a 1D and y a 2D array.

    Returns:
    rho (1D array): Pearson correlation coefficient between x and each row of y.
    """
    # credit: https://stackoverflow.com/a/64660866

    upper = np.sum((x - np.mean(x)) * (y - np.mean(y, axis=1)[:, None]), axis=1)
    lower = np.sqrt(np.sum(np.power(x - np.mean(x), 2)) * np.sum(np.power(y - np.mean(y, axis=1)[:, None], 2), axis=1))
    pearson_r = upper / lower
    return pearson_r


def measure_overlap_heatmap():
    # heatmap = np.load('./data/peterson/transalnet_res/sal_mat/animals/stim-0007.npy')
    # heatmap = MinMaxScaler().fit_transform(heatmap)
    # data = []
    # for i in range(heatmap.shape[0]):
    #     for j in range(heatmap.shape[1]):
    #         data.append([i/heatmap.shape[0], j/heatmap.shape[1] , heatmap[i, j]])
    # data = np.array(data)
    # db = DBSCAN(eps=0.04, min_samples=10).fit(data)
    # labels = db.labels_
    # # Number of clusters in labels, ignoring noise if present.
    # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # n_noise_ = list(labels).count(-1)
    # print("Estimated number of clusters: %d" % n_clusters_)
    # print("Estimated number of noise points: %d" % n_noise_)
    # print(f"Silhouette Coefficient: {metrics.silhouette_score(data, labels):.3f}")

    dataset_names = ['animals', 'automobiles', 'fruits', 'furniture', 'various', 'vegetables']
    for dataset in dataset_names:
        heatmap_list = sorted(list(pathlib.Path(f"./data/peterson/transalnet_dense/sal_mat/{dataset}").glob("*")))
        pathlib.Path(f"./data/peterson/transalnet_dense/sal_mat_top_5/{dataset}").mkdir(parents=True, exist_ok=True)
        for path in heatmap_list:
            heatmap = np.load(str(path))
            filter_heatmap = heatmap * (heatmap > np.percentile(heatmap, 95))
            filter_img = Image.fromarray(filter_heatmap * 255).convert('RGB')

            filter_img.save(str(path).replace('sal_mat', 'sal_mat_top_5').replace('.npy', '.png'))


def corr_heatmap(trained_dataset):
    dataset_names = ['animals', 'automobiles', 'fruits', 'furniture', 'various', 'vegetables']
    r2_original = np.load(f'./res/corr/{trained_dataset}/original/all_datasets.npy')

    heatmap_scores = []
    for dataset in dataset_names:
        r2_remove = np.load(f'./res/corr/{trained_dataset}/remove_collective/{dataset}.npy')
        r2_perturbation = r2_original[dataset_names.index(dataset)] - r2_remove
        r2_perturbation = r2_perturbation * (r2_perturbation > 0)
        heatmap_scores.append(r2_perturbation / sum(r2_perturbation))
    heatmap_scores = np.array(heatmap_scores)
    return np.corrcoef(heatmap_scores)

