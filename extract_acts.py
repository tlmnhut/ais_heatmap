import pathlib
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data import TensorDataset, DataLoader

from model import VGG16, VGG16Remove1, VGG16Keep1, VGG16Prune


def extract_acts(model, img_path, mask=None):
    """
    Extract activations from many layers as instructed by the model definitions.

    Args:
    - model: A PyTorch model object used for processing the input image.
    - img_path: A string representing the path to the input image.

    Returns:
    - Activations from many layers after being forwarded as defined in the model.
    The output is returned as a PyTorch tensor object.
    """

    input_image = Image.open(img_path).convert('RGB')
    preprocess = transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    if mask is not None:
        mask = torch.from_numpy(mask)

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        if mask is not None:
            mask = mask.to('cuda')

    with torch.no_grad():
        if mask is not None:
            output = model(input_batch, mask)
        else:
            output = model(input_batch)

    return output


def extract_acts_batch(model, img_dir, batch_size=1, mask=None):
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_data = datasets.ImageFolder(root=img_dir, transform=preprocess)
    loader = DataLoader(img_data, batch_size=batch_size, shuffle=False)

    if mask is not None:
        mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            mask = mask.to('cuda')

    output = []
    for data in loader:
        input_batch, _ = data
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
        with torch.no_grad():
            if mask is not None:
                output.append(model(input_batch, mask)[0]) # only get penultimate
            else:
                output.append(model(input_batch)[0]) # only get penultimate

    return np.vstack(output)


def extract_acts_from_last_conv(model, last_conv, batch_size=1, mask=None):
    fake_labels = np.ones(last_conv.shape[0])
    penultimate_dataset = TensorDataset(torch.Tensor(last_conv), torch.Tensor(fake_labels))
    loader = DataLoader(penultimate_dataset, batch_size=batch_size, shuffle=False)

    if mask is not None:
        mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            mask = mask.to('cuda')

    output = []
    for data in loader:
        input_batch, _ = data
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
        with torch.no_grad():
            if mask is not None:
                output.append(model(input_batch, mask)[0]) # only get penultimate
            else:
                output.append(model(input_batch)[0]) # only get penultimate

    return np.vstack(output)


def predict_class(last_layer_scores):
    # Credit: https://pytorch.org/hub/pytorch_vision_vgg/
    """
    Print the top predicted categories with their corresponding probabilities based on the
    softmax scores of the last fully connected layer.

    Args:
    - last_layer_scores: A tensor of scores representing the last fully connected layer.

    Returns:
    - None. This function prints the top predicted categories with their corresponding probabilities.
    """
    probabilities = torch.nn.functional.softmax(last_layer_scores[0], dim=0)
    # Read the categories
    with open("./data/imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 1)  # 5
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())


if __name__ == '__main__':
    trained_dataset = 'imagenet'
    model_mode = 'keep'

    # model = VGG16(trained_dataset=trained_dataset)
    # model = VGG16Remove1(trained_dataset=trained_dataset)
    model = VGG16Keep1(trained_dataset=trained_dataset)
    # model = VGG16Prune(trained_dataset=trained_dataset)

    if torch.cuda.is_available():
        model.to('cuda')
    model.eval()

    dataset_names = ['animals', 'automobiles', 'fruits', 'furniture', 'various', 'vegetables']
    for dataset in dataset_names:
        image_path_list = sorted(list(pathlib.Path(f"./data/peterson/original/{dataset}/images").glob("*")))
        # scores = np.load(f'./res/scores/{trained_dataset}/keep/{dataset}_dnn_corr_modified.npy')
        last_conv_list, penultimate_list, last_fc_list = [], [], []
        for idx in tqdm(range(len(image_path_list))):
            # last_conv, penultimate, last_fc = extract_acts(model=model, img_path=image_path_list[idx])
            penultimate, last_fc = extract_acts(model=model, img_path=image_path_list[idx])
            # prune_mask = (scores[idx] > 0).astype(np.uint8)
            # penultimate, last_fc = extract_acts(model=model, img_path=image_path_list[idx], mask=prune_mask)
            # print(last_conv.shape, penultimate.shape, last_fc.shape)

            # last_conv_list.append(last_conv)
            penultimate_list.append(penultimate)
            last_fc_list.append(last_fc)
        # np.save(f'./res/acts/{trained_dataset}/{model_mode}/vgg16_peterson_{dataset}_last_conv.npy', np.array(last_conv_list))
        np.save(f'./res/acts/{trained_dataset}/{model_mode}/vgg16_peterson_{dataset}_penultimate.npy', np.array(penultimate_list))
        np.save(f'./res/acts/{trained_dataset}/{model_mode}/vgg16_peterson_{dataset}_last_fc.npy', np.array(last_fc_list))

