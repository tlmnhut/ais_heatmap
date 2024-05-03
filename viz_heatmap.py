import numpy as np
import cv2
from matplotlib import cm
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import pathlib
from torchvision.io import image
from torchvision.io.image import read_image, ImageReadMode
from torchvision.transforms.functional import to_pil_image
import os
import torchvision.transforms as T
import scipy.stats
from sklearn.metrics import confusion_matrix
import numpy.ma as ma
import matplotlib.pyplot as plt


def overlay_mask(img: Image.Image, mask: np.array, colormap: str = "jet_r", alpha: float = 0.7) -> Image.Image:
    # Credit:
    # Copyright (C) 2020-2023, François-Guillaume Fernandez.

    # This program is licensed under the Apache License 2.0.
    # See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.
    """Overlay a colormapped mask on a background image
    Args:
        img: background image
        mask: mask to be overlayed in grayscale
        colormap: colormap to be applied on the mask
        alpha: transparency of the background image
    Returns:
        overlayed image
    Raises:
        TypeError: when the arguments have invalid types
        ValueError: when the alpha argument has an incorrect value
    """

    # if not isinstance(img, Image.Image) or not isinstance(mask, Image.Image):
    #     raise TypeError("img and mask arguments need to be PIL.Image")

    if not isinstance(alpha, float) or alpha < 0 or alpha >= 1:
        raise ValueError("alpha argument is expected to be of type float between 0 and 1")

    cmap = cm.get_cmap(colormap)
    # Resize mask and apply colormap
    overlay = cv2.resize(mask, img.size)
    overlay = overlay/ overlay.max()
    overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)
    # Overlay the image with the mask
    overlayed_img = Image.fromarray((alpha * np.asarray(img) + (1 - alpha) * np.asarray(overlay)).astype(np.uint8))

    return overlayed_img


def compute_heatmap(feature_maps, scores):
    """
    Computes a weighted average of a set of feature maps using a set of corresponding scores.

    Args:
    - feature_maps (numpy.ndarray): A 4D array of feature maps (shape 144 x 512 x 14 x 14).
    - scores (numpy.ndarray): A 2D array of scores for each feature map (shape 144 x 512).

    Returns:
    numpy.ndarray: A 3D array representing the computed heatmap (shape 144 x 14 x 14).
    """
    return np.average(feature_maps, weights=scores, axis=0)


def compute_heatmap_normalized(feature_maps, scores):
    """
    Computes a weighted average of a set of feature maps using a set of corresponding scores.
    but normalizes within each 14x14 map before.
    Args:
    - feature_maps (numpy.ndarray): A 3D array of feature maps (shape 144 x 14 x 14).
    - scores (numpy.ndarray): A 2D array of scores for each feature map (shape 144 x 14 x 14).

    Returns:
    numpy.ndarray: A 2D array representing the computed heatmap (shape 14 x 14).
    """

    # Calculate the minimum and maximum values along the (1, 2) axes
    min_vals = np.min(feature_maps, axis=(1, 2), keepdims=True)
    max_vals = np.max(feature_maps, axis=(1, 2), keepdims=True)

    # Handle division by zero by replacing zero denominators with a small epsilon value
    eps = 1e-10
    denom = max_vals - min_vals
    denom[denom == 0] = eps

    # Normalize the feature maps within each 14x14 map
    normalized_feature_maps = (feature_maps - min_vals) / denom

    # Handle NaN values by replacing them with 0
    normalized_feature_maps = np.nan_to_num(normalized_feature_maps)

    # Compute the weighted average of the normalized feature maps using the scores
    weighted_average = np.average(normalized_feature_maps, weights=scores, axis=0)

    return weighted_average


def compute_correlation_activation_vs_perturbation(feature_maps, scores):
    activation_sum = np.sum(feature_maps, axis=(1, 2))  # produces 512 values
    activation_corr = scipy.stats.pearsonr(activation_sum, scores)[0]
    return activation_corr


def rank_of_feature_on_perturbation(feature_maps, scores):
    activation_sum = np.sum(feature_maps, axis=(1, 2))  # sum the activation per feature map
    # activation_sum=np.sum(feature_maps, axis=0)
    highest_activation = np.max(activation_sum)  # looks for the max
    highest_activation_indice = np.asarray(
        activation_sum == highest_activation).nonzero()  # extract the indice of the max activation
    # print(highest_activation_indice)
    ranked_scores = np.argsort(scores, axis=0)  # rank the scores
    rank_of_max_activation = ranked_scores[
        highest_activation_indice]  # looks for the indice in scores where max activation occurred
    rank_of_max_activation += 1  # start rank count at 1
    return rank_of_max_activation


def compute_heatmap_no_weights(feature_maps):
    return np.average(feature_maps, axis=0)


def visualize_heatmap(img_path_list, heatmap_collection, save_dir):
    """
    Generate and save a collection of heatmaps based on a set of input images and corresponding heatmap data.
    For each image, the function generates a heatmap by overlaying the heatmap data on top of the image.
    The resulting image is then saved in the specified output directory.

    Args:
    - img_path_list (list): A list of file paths to input images.
    - heatmap_collection (np.ndarray): A 3D (144 x 14 x 14) numpy.ndarray objects representing the heatmaps.
    - save_dir (str): The directory in which to save the generated heatmaps.
    """
    for i in range(len(img_path_list)):
        img_path = str(img_path_list[i])
        # img = read_image(img_path)
        img = Image.open(img_path).convert('RGB')
        # print(img)
        # heatmap = overlay_mask(img, Image.fromarray(heatmap_collection[i]), alpha=0.6, colormap='jet')
        heatmap = overlay_mask(img, heatmap_collection[i], alpha=0.6, colormap='jet')
        save_path = str(save_dir + '/' + img_path.split('/')[-1])
        heatmap.save(save_path)


def postprocess_img(pred, org_shape):
    # copy from transalnet
    pred = np.array(pred)
    shape_r = org_shape[0]
    shape_c = org_shape[1]
    predictions_shape = pred.shape

    rows_rate = shape_r / predictions_shape[0]
    cols_rate = shape_c / predictions_shape[1]

    if rows_rate > cols_rate:
        new_cols = (predictions_shape[1] * shape_r) // predictions_shape[0]
        pred = cv2.resize(pred, (new_cols, shape_r))
        img = pred[:, ((pred.shape[1] - shape_c) // 2):((pred.shape[1] - shape_c) // 2 + shape_c)]
    else:
        new_rows = (predictions_shape[0] * shape_c) // predictions_shape[1]
        pred = cv2.resize(pred, (shape_c, new_rows))
        img = pred[((pred.shape[0] - shape_r) // 2):((pred.shape[0] - shape_r) // 2 + shape_r), :]

    return img


def overlay_prediction_ground_truth(proxy_heatmap, predict_heatmap, rgb_image):
    proxy_heatmap = postprocess_img(proxy_heatmap, rgb_image.size)
    predict_heatmap = cv2.resize(predict_heatmap, rgb_image.size)

    contours = []
    relative_risk = []
    for percent in [5, 10, 15]:
        proxy_heatmap_filter = proxy_heatmap > np.percentile(proxy_heatmap, 100 - percent)
        predict_heatmap_filter = predict_heatmap > np.percentile(predict_heatmap, 100 - percent)
        tn, fp, fn, tp = confusion_matrix(proxy_heatmap_filter.flatten(), predict_heatmap_filter.flatten()).ravel()
        relative_risk.append((tp / (tp + fn)) / (fp / (fp + tn)))

        contour_proxy_heatmap = np.array(Image.fromarray(proxy_heatmap_filter).filter(ImageFilter.FIND_EDGES))
        contour_predict_heatmap = np.array(Image.fromarray(predict_heatmap_filter).filter(ImageFilter.FIND_EDGES))
        contours.append(contour_proxy_heatmap)
        contours.append(contour_predict_heatmap)

    draw = ImageDraw.Draw(rgb_image)
    colors = [(230, 115, 0), (0, 0, 255),
              (255, 153, 51), (0, 102, 255),
              (255, 204, 153), (51, 153, 255)]
    for idx in range(len(contours)):
        points = [(index[0], index[1]) for index in np.argwhere(contours[idx].T)]
        draw.point(points, fill=colors[idx])
    font = ImageFont.truetype("./figures/arial.ttf", size=int(rgb_image.size[0] / 10))
    draw.text((0, 0), ' '.join([str(np.round(i, 2)) for i in relative_risk]), font=font, fill=(0,255,0,255)) # green

    return rgb_image, relative_risk


def viz_grid_heatmap():
    trained_dataset = 'ecoset'
    dataset_names = ['animals', 'automobiles', 'fruits', 'furniture', 'various', 'vegetables']
    selected_img = [['stim-0793.png', 'stim-0976.png', 'stim-0007.png', 'stim-0906.png'],
                    ['elevator2.jpg', 'tractor3.jpg', 'raft4.jpg', 'truck6.jpg'],
                    ['ackee2.png', 'durian2.png', 'pineapple6.png', 'passionfruit6.png'],
                    ['cupboard-03.jpg', 'coffee-table-01.jpg', 'coffee-table-03.jpg', 'coffee-table-07.jpg'],
                    ['artificialobject8.jpg', 'artificialobject3.jpg', 'naturalobject23.jpg', 'humanbody5.jpg'],
                    ['artichoke1.jpg', 'cabbage2.jpg', 'pea2.jpg', 'chives3.jpg']
    ]
    fig, axs = plt.subplots(4, 6, figsize=(12, 8))
    for dataset in range(6):
        for img in range(4):
            heatmap_path = f'./figures/{trained_dataset}/remove_individual/{dataset_names[dataset]}/' +\
                           selected_img[dataset][img]
            heatmap_img = Image.open(heatmap_path)
            heatmap_img = heatmap_img.resize((300, 300))
            axs[img, dataset].imshow(heatmap_img)
            axs[img, dataset].axis('off')

            # overlap_path = f'./figures/{trained_dataset}/remove_individual/overlap_quant/{dataset_names[dataset]}/' +\
            #                selected_img[dataset][img]
            # overlap_img = Image.open(overlap_path)
            # overlap_img = overlap_img.resize((300, 300))
            # axs[img, dataset].imshow(overlap_img)
            # axs[img, dataset].axis('off')
        axs[0, dataset].set_title(f"{dataset_names[dataset].replace('automobiles', 'transportation')}".capitalize(),
                                  fontsize=13)
    plt.tight_layout()
    plt.savefig('./figures/grid_tmp.png')


def viz_grid_overlap():
    trained_dataset = 'ecoset'
    selected_img = [f'./figures/{trained_dataset}/remove_individual/overlap_quant/animals/stim-0793.png',
                    f'./figures/{trained_dataset}/remove_individual/overlap_quant/automobiles/elevator2.jpg',
                    f'./figures/{trained_dataset}/remove_individual/overlap_quant/animals/stim-0906.png',
                    f'./figures/{trained_dataset}/remove_individual/overlap_quant/automobiles/truck6.jpg',
    ]
    fig, axs = plt.subplots(1, 4, figsize=(12, 3))
    axs_flat = axs.flatten()
    for i in range(4):
        overlap_img = Image.open(selected_img[i])
        overlap_img = overlap_img.resize((300, 300))
        axs_flat[i].imshow(overlap_img)
        axs_flat[i].axis('off')
    plt.tight_layout()
    plt.savefig('./figures/grid_overlap.png')


if __name__ == '__main__':
    # trained_dataset = 'ecoset'
    # dataset_names = ['animals', 'automobiles', 'fruits', 'furniture', 'various', 'vegetables']
    # for dataset in dataset_names:
    #     feature_maps = np.load(f'./res/acts/{trained_dataset}/original/vgg16_peterson_{dataset}_last_conv.npy')
    #     scores = np.load(f'./res/scores/{trained_dataset}/keep/{dataset}_dnn_corr_modified.npy')
    #     # scores = feature_maps.sum(axis=-1).sum(axis=-1)
    #     # scores = np.count_nonzero(feature_maps, axis=(2, 3))
    #     # scores = scores / scores.sum(axis=1, keepdims=True)
    #     heatmap_collection = []
    #     # activation_corr_collection = []
    #     # activation_rank_collection = []
    #     for i in range(feature_maps.shape[0]):
    #         # activation_rank = rank_of_feature_on_perturbation(feature_map_collection[i], score_collection[i])
    #         # activation_rank_collection.append(activation_rank)
    #         heatmap = compute_heatmap(feature_maps=feature_maps[i], scores=scores[i])
    #         # activation_correlation = compute_correlation_activation_vs_perturbation(
    #         #      feature_maps=feature_map_collection[i], scores=score_collection[i])
    #         heatmap_collection.append(heatmap)
    #         # activation_corr_collection.append(activation_correlation)
    #         # np.save('./res/activation ranks/activation_ranks_{}'.format(dataset_label_names[label]), activation_rank_collection)
    #         # mean_activation_collection.append(np.mean(activation_rank_collection))
    #         # std_activation_collection.append(np.std(activation_rank_collection))
    #     heatmap_collection = np.array(heatmap_collection)
    #     np.save(f'./res/heatmaps/{trained_dataset}/keep/{dataset}_dnn_corr_modified', heatmap_collection)
    #
    #     # Overlay heatmaps on top of images and then save them
    #     image_path_list = sorted(list(pathlib.Path(f"./data/peterson/resize_224/{dataset}/images/").glob("*")))
    #     heatmap_collection = np.load(f'./res/heatmaps/{trained_dataset}/keep/{dataset}_dnn_corr_modified.npy')
    #     save_dir = f'./figures/{trained_dataset}/keep/dnn_corr_modified/{dataset}/'
    #     pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    #     visualize_heatmap(img_path_list=image_path_list,
    #                       heatmap_collection=heatmap_collection,
    #                       save_dir=save_dir)

        # image_path_list = sorted(list(pathlib.Path(f"./data/peterson/resize_224/{dataset}/images/").glob("*")))
        # heatmap_collection = []
        # for path in image_path_list:
        #     heatmap_collection.append(np.load(f'./data/peterson/transalnet_dense/sal_mat/{dataset}/' + path.name.split('.')[0] + '.npy'))
        # save_dir = f'./data/peterson/transalnet_dense/img_heatmap/{dataset}/'
        # pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
        # visualize_heatmap(img_path_list=image_path_list,
        #                   heatmap_collection=heatmap_collection,
        #                   save_dir=save_dir)


    # trained_dataset = 'ecoset'
    # dataset = 'animals'
    #
    # all_images = sorted([path.name for path in pathlib.Path(f"./data/peterson/original/animals/images/").glob("*")])
    # selected_images = sorted(
    #     [path.name for path in pathlib.Path(f"./data/peterson/sub_category/animals/reptile").glob("*")] )#+
    #     #[path.name for path in pathlib.Path(f"./data/peterson/sub_category/animals/amphibian").glob("*")])
    # selected_idx = [all_images.index(path) for path in selected_images]
    #
    # feature_maps = np.load(f'./res/acts/{trained_dataset}/original/vgg16_peterson_{dataset}_last_conv.npy')[selected_idx]
    # scores = np.load(f'./res/scores/{trained_dataset}/keep/selected_reptile_only_positive_no_abs.npy')
    # heatmap_collection = []
    # for i in range(feature_maps.shape[0]):
    #     heatmap = compute_heatmap(feature_maps=feature_maps[i], scores=scores[i])
    #     heatmap_collection.append(heatmap)
    # heatmap_collection = np.array(heatmap_collection)
    # np.save(f'./res/heatmaps/{trained_dataset}/keep/selected_reptile_only_positive_no_abs', heatmap_collection)
    #
    # # Overlay heatmaps on top of images and then save them
    # image_path_list = [sorted(list(pathlib.Path(f"./data/peterson/resize_224/{dataset}/images/").glob("*")))[i]
    #                    for i in selected_idx]
    # heatmap_collection = np.load(f'./res/heatmaps/{trained_dataset}/keep/selected_reptile_only_positive_no_abs.npy')
    # save_dir = f'./figures/{trained_dataset}/keep/selected/reptile/'
    # pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    # visualize_heatmap(img_path_list=image_path_list,
    #                   heatmap_collection=heatmap_collection,
    #                   save_dir=save_dir)


    # train_dataset = 'ecoset'
    # dataset_names = ['animals', 'automobiles', 'fruits', 'furniture', 'various', 'vegetables']
    # for dataset in dataset_names:
    #     proxy_heatmap_list = sorted(list(pathlib.Path(f"./data/peterson/transalnet_dense/sal_mat/{dataset}").glob("*")))
    #     my_heatmap_list = np.load(f'./res/heatmaps/{train_dataset}/keep/{dataset}_dnn_corr_modified.npy')
    #     img_list = sorted(list(pathlib.Path(f'./data/peterson/resize_224/{dataset}/images').glob('*')))
    #     pathlib.Path(f'./figures/{train_dataset}/keep/overlap_quant/{dataset}/').mkdir(parents=True, exist_ok=True)
    #
    #     rr_list = []
    #     for idx in range(len(proxy_heatmap_list)):
    #         proxy_heatmap = np.load(str(proxy_heatmap_list[idx]))
    #         my_heatmap = my_heatmap_list[idx]
    #         rgb_image = Image.open(img_list[idx])
    #         overlaid_img, relative_risk = overlay_prediction_ground_truth(proxy_heatmap=proxy_heatmap,
    #                                                        predict_heatmap=my_heatmap, rgb_image=rgb_image)
    #         rgb_image.save(f'./figures/{train_dataset}/keep/overlap_quant/{dataset}/' + img_list[idx].name)
    #         rr_list.append(relative_risk)
    #     rr_list = np.array(rr_list)
    #     print(dataset, rr_list.mean(axis=0), rr_list.std(axis=0))


    # trained_dataset = 'ecoset'
    # dataset_names = ['animals', 'automobiles', 'fruits', 'furniture', 'various', 'vegetables']
    # # r2_original = np.load(f'./res/corr/{trained_dataset}/original/all_datasets.npy')
    # rr_list_all = []
    # for dataset in dataset_names:
    #     # r2_remove = np.load(f'./res/corr/{trained_dataset}/remove/{dataset}.npy')
    #     # r2_perturbation = r2_original[dataset_names.index(dataset)] - r2_remove
    #     # r2_perturbation = r2_perturbation * (r2_perturbation > 0)
    #     # heatmap_scores = r2_perturbation / sum(r2_perturbation)
    #
    #     # feature_maps = np.load(f'./res/acts/{trained_dataset}/original/vgg16_peterson_{dataset}_last_conv.npy')
    #     # heatmap_scores = np.load(f'./res/scores/{trained_dataset}/remove_individual/{dataset}.npy')
    #     # heatmap_collection = []
    #     # for i in range(feature_maps.shape[0]):
    #     #     heatmap = compute_heatmap(feature_maps=feature_maps[i], scores=heatmap_scores[i]) # i on heatmap_scores or not
    #     #     heatmap_collection.append(heatmap)
    #     # np.save(f'./res/heatmaps/{trained_dataset}/remove_individual/{dataset}', np.array(heatmap_collection))
    #
    #     # image_path_list = sorted(list(pathlib.Path(f"./data/peterson/original/{dataset}/images/").glob("*")))
    #     # heatmap_collection = np.load(f'./res/heatmaps/{trained_dataset}/remove_individual/{dataset}.npy')
    #     # save_dir = f'./figures/{trained_dataset}/remove_individual/{dataset}/'
    #     # pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    #     # visualize_heatmap(img_path_list=image_path_list,
    #     #                   heatmap_collection=heatmap_collection,
    #     #                   save_dir=save_dir)
    #
    #     proxy_heatmap_list = np.load(f"./data/peterson/transalnet_dense/sal_mat/{dataset}.npy")
    #     my_heatmap_list = np.load(f'./res/heatmaps/{trained_dataset}/remove_individual/{dataset}.npy')
    #     img_list = sorted(list(pathlib.Path(f'./data/peterson/original/{dataset}/images').glob('*')))
    #     pathlib.Path(f'./figures/{trained_dataset}/remove_individual/overlap_quant/{dataset}/').mkdir(parents=True, exist_ok=True)
    #     rr_list = []
    #     for idx in range(len(my_heatmap_list)):
    #         proxy_heatmap = proxy_heatmap_list[idx]
    #         my_heatmap = my_heatmap_list[idx]
    #         rgb_image = Image.open(img_list[idx])
    #         overlaid_img, relative_risk = overlay_prediction_ground_truth(proxy_heatmap=proxy_heatmap,
    #                                                        predict_heatmap=my_heatmap, rgb_image=rgb_image)
    #         rgb_image.save(f'./figures/{trained_dataset}/remove_individual/overlap_quant/{dataset}/' + img_list[idx].name)
    #         rr_list.append(relative_risk)
    #     rr_list_all.append(rr_list)
    #     rr_list = np.array(rr_list)
    #     print(dataset, np.round(rr_list.mean(axis=0), 2), np.round(rr_list.std(axis=0), 2))
    # rr_list_all = np.vstack(np.array(rr_list_all))
    # print(np.round(rr_list_all.mean(axis=0), 2), np.round(rr_list_all.std(axis=0), 2))

    viz_grid_overlap()
    # viz_grid_heatmap()
