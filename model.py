import numpy as np
import torch
import torch.nn as nn
from torchvision import models


# Original untouched VGG-19
class VGG19Original(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        vgg19 = models.vgg19(pretrained=pretrained)
        # print(vgg19)
        self.features_0 = nn.ModuleList(vgg19.children())[0][:36]
        self.features_0 = nn.Sequential(*self.features_0)
        self.features_1 = nn.ModuleList(vgg19.children())[0][36:]
        self.features_1 = nn.Sequential(*self.features_1)
        self.avgpool = nn.ModuleList(vgg19.children())[1]
        self.classifier_0 = nn.ModuleList(vgg19.children())[2][:6]
        self.classifier_0 = nn.Sequential(*self.classifier_0)
        self.classifier_1 = nn.ModuleList(vgg19.children())[2][6:]
        self.classifier_1 = nn.Sequential(*self.classifier_1)

    def forward(self, input_img):
        # print(input_img)
        last_conv = self.features_0(input_img)
        # print(pre_last_conv, pre_last_conv.shape)
        last_max_pool = self.features_1(last_conv)
        avg_pool = self.avgpool(last_max_pool)
        flatten = torch.flatten(avg_pool, 1)
        penultimate = self.classifier_0(flatten)
        # print(penultimate, penultimate.shape)
        last_fc = self.classifier_1(penultimate)
        return last_conv, penultimate, last_fc


# Zero-out one feature map in the las conv layer
class VGG19Remove1(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        vgg19 = models.vgg19(pretrained=pretrained)
        # print(vgg19)
        self.features_0 = nn.ModuleList(vgg19.children())[0][:36]
        self.features_0 = nn.Sequential(*self.features_0)
        self.features_1 = nn.ModuleList(vgg19.children())[0][36:]
        self.features_1 = nn.Sequential(*self.features_1)
        self.avgpool = nn.ModuleList(vgg19.children())[1]
        self.classifier_0 = nn.ModuleList(vgg19.children())[2][:6]  # stop at penultimate layer
        self.classifier_0 = nn.Sequential(*self.classifier_0)
        self.classifier_1 = nn.ModuleList(vgg19.children())[2][6:]
        self.classifier_1 = nn.Sequential(*self.classifier_1)

    def forward(self, input_img):
        last_conv = self.features_0(input_img)
        # print(last_conv.shape)
        penultimate_list, last_fc_list = [], []
        for idx in range(last_conv.shape[1]):
            last_conv_copy = last_conv.clone()
            last_conv_copy[:, idx, :, :] = 0  # Zero-out here
            # print(pre_last_conv, pre_last_conv.shape)
            last_max_pool = self.features_1(last_conv_copy)
            avg_pool = self.avgpool(last_max_pool)
            flatten = torch.flatten(avg_pool, 1)
            penultimate = self.classifier_0(flatten)
            # print(penultimate, penultimate.shape)
            last_fc = self.classifier_1(penultimate)
            penultimate_list.append(penultimate[0].detach().cpu().numpy())
            last_fc_list.append(last_fc[0].detach().cpu().numpy())
        return np.array(penultimate_list), np.array(last_fc_list)


class VGG19Keep1(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        vgg19 = models.vgg19(pretrained=pretrained)
        # print(vgg19)
        self.features_0 = nn.ModuleList(vgg19.children())[0][:36]
        self.features_0 = nn.Sequential(*self.features_0)
        self.features_1 = nn.ModuleList(vgg19.children())[0][36:]
        self.features_1 = nn.Sequential(*self.features_1)
        self.avgpool = nn.ModuleList(vgg19.children())[1]
        self.classifier_0 = nn.ModuleList(vgg19.children())[2][:6]  # stop at penultimate layer
        self.classifier_0 = nn.Sequential(*self.classifier_0)
        self.classifier_1 = nn.ModuleList(vgg19.children())[2][6:]
        self.classifier_1 = nn.Sequential(*self.classifier_1)

    def forward(self, input_img):
        last_conv = self.features_0(input_img)
        # print(last_conv.shape)
        penultimate_list, last_fc_list = [], []
        for idx in range(last_conv.shape[1]):
            last_conv_copy = last_conv.clone()
            for j in range((last_conv.shape[1])):
                if j != idx:
                    last_conv_copy[:, j, :, :] = 0
            # print(pre_last_conv, pre_last_conv.shape)
            last_max_pool = self.features_1(last_conv_copy)
            avg_pool = self.avgpool(last_max_pool)
            flatten = torch.flatten(avg_pool, 1)
            penultimate = self.classifier_0(flatten)
            # print(penultimate, penultimate.shape)
            last_fc = self.classifier_1(penultimate)
            penultimate_list.append(penultimate[0].detach().cpu().numpy())
            last_fc_list.append(last_fc[0].detach().cpu().numpy())
        return np.array(penultimate_list), np.array(last_fc_list)


class VGG16(nn.Module):
    def __init__(self, trained_dataset='imagenet'):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        if trained_dataset.lower() == 'ecoset':
            vgg16.classifier[6] = nn.Linear(4096, 565, bias=True)
            path_to_state_dict = 'https://osf.io/z5uf3/download'
            vgg16.load_state_dict(torch.torch.hub.load_state_dict_from_url(path_to_state_dict))
            # print(trained_dataset)

        self.features_0 = nn.ModuleList(vgg16.children())[0][:30]
        self.features_0 = nn.Sequential(*self.features_0)
        self.features_1 = nn.ModuleList(vgg16.children())[0][30:]
        self.features_1 = nn.Sequential(*self.features_1)
        self.avgpool = nn.ModuleList(vgg16.children())[1]
        self.classifier_0 = nn.ModuleList(vgg16.children())[2][:6]  # stop at penultimate layer
        self.classifier_0 = nn.Sequential(*self.classifier_0)
        self.classifier_1 = nn.ModuleList(vgg16.children())[2][6:]
        self.classifier_1 = nn.Sequential(*self.classifier_1)

    def forward(self, input_img):
        # print(input_img)
        last_conv = self.features_0(input_img)
        # print(pre_last_conv, pre_last_conv.shape)
        last_max_pool = self.features_1(last_conv)
        avg_pool = self.avgpool(last_max_pool)
        flatten = torch.flatten(avg_pool, 1)
        penultimate = self.classifier_0(flatten)
        # print(penultimate, penultimate.shape)
        last_fc = self.classifier_1(penultimate)
        return last_conv[0].detach().cpu().numpy(), penultimate[0].detach().cpu().numpy(), last_fc[0].detach().cpu().numpy()


class VGG16Remove1(nn.Module):
    def __init__(self, trained_dataset='imagenet'):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        if trained_dataset.lower() == 'ecoset':
            vgg16.classifier[6] = nn.Linear(4096, 565, bias=True)
            path_to_state_dict = 'https://osf.io/z5uf3/download'
            vgg16.load_state_dict(torch.torch.hub.load_state_dict_from_url(path_to_state_dict))
            print(trained_dataset)

        self.features_0 = nn.ModuleList(vgg16.children())[0][:30]
        self.features_0 = nn.Sequential(*self.features_0)
        self.features_1 = nn.ModuleList(vgg16.children())[0][30:]
        self.features_1 = nn.Sequential(*self.features_1)
        self.avgpool = nn.ModuleList(vgg16.children())[1]
        self.classifier_0 = nn.ModuleList(vgg16.children())[2][:6]  # stop at penultimate layer
        self.classifier_0 = nn.Sequential(*self.classifier_0)
        self.classifier_1 = nn.ModuleList(vgg16.children())[2][6:]
        self.classifier_1 = nn.Sequential(*self.classifier_1)

    def forward(self, input_img):
        last_conv = self.features_0(input_img)
        # print(last_conv.shape)
        penultimate_list, last_fc_list = [], []
        for idx in range(last_conv.shape[1]):
            last_conv_copy = last_conv.clone()
            last_conv_copy[:, idx, :, :] = 0  # Zero-out here
            # print(pre_last_conv, pre_last_conv.shape)
            last_max_pool = self.features_1(last_conv_copy)
            avg_pool = self.avgpool(last_max_pool)
            flatten = torch.flatten(avg_pool, 1)
            penultimate = self.classifier_0(flatten)
            # print(penultimate, penultimate.shape)
            last_fc = self.classifier_1(penultimate)
            penultimate_list.append(penultimate[0].detach().cpu().numpy())
            last_fc_list.append(last_fc[0].detach().cpu().numpy())
        return np.array(penultimate_list), np.array(last_fc_list)


class VGG16Keep1(nn.Module):
    def __init__(self, trained_dataset='imagenet'):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        if trained_dataset.lower() == 'ecoset':
            vgg16.classifier[6] = nn.Linear(4096, 565, bias=True)
            path_to_state_dict = 'https://osf.io/z5uf3/download'
            vgg16.load_state_dict(torch.torch.hub.load_state_dict_from_url(path_to_state_dict))
            print(trained_dataset)

        self.features_0 = nn.ModuleList(vgg16.children())[0][:30]
        self.features_0 = nn.Sequential(*self.features_0)
        self.features_1 = nn.ModuleList(vgg16.children())[0][30:]
        self.features_1 = nn.Sequential(*self.features_1)
        self.avgpool = nn.ModuleList(vgg16.children())[1]
        self.classifier_0 = nn.ModuleList(vgg16.children())[2][:6]  # stop at penultimate layer
        self.classifier_0 = nn.Sequential(*self.classifier_0)
        self.classifier_1 = nn.ModuleList(vgg16.children())[2][6:]
        self.classifier_1 = nn.Sequential(*self.classifier_1)

    def forward(self, input_img):
        last_conv = self.features_0(input_img)
        # print(last_conv.shape)
        penultimate_list, last_fc_list = [], []
        for idx in range(last_conv.shape[1]):
            last_conv_copy = last_conv.clone()
            for j in range((last_conv.shape[1])):
                if j != idx:
                    last_conv_copy[:, j, :, :] = 0
            # print(pre_last_conv, pre_last_conv.shape)
            last_max_pool = self.features_1(last_conv_copy)
            avg_pool = self.avgpool(last_max_pool)
            flatten = torch.flatten(avg_pool, 1)
            penultimate = self.classifier_0(flatten)
            # print(penultimate, penultimate.shape)
            last_fc = self.classifier_1(penultimate)
            penultimate_list.append(penultimate[0].detach().cpu().numpy())
            last_fc_list.append(last_fc[0].detach().cpu().numpy())
        return np.array(penultimate_list), np.array(last_fc_list)


class VGG16Prune(nn.Module):
    def __init__(self, trained_dataset='imagenet'):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        if trained_dataset.lower() == 'ecoset':
            vgg16.classifier[6] = nn.Linear(4096, 565, bias=True)
            path_to_state_dict = 'https://osf.io/z5uf3/download'
            vgg16.load_state_dict(torch.torch.hub.load_state_dict_from_url(path_to_state_dict))
            print(trained_dataset)

        # self.features_0 = nn.ModuleList(vgg16.children())[0][:30]
        # self.features_0 = nn.Sequential(*self.features_0)
        self.features_1 = nn.ModuleList(vgg16.children())[0][30:]
        self.features_1 = nn.Sequential(*self.features_1)
        self.avgpool = nn.ModuleList(vgg16.children())[1]
        self.classifier_0 = nn.ModuleList(vgg16.children())[2][:6]  # stop at penultimate layer
        self.classifier_0 = nn.Sequential(*self.classifier_0)
        self.classifier_1 = nn.ModuleList(vgg16.children())[2][6:]
        self.classifier_1 = nn.Sequential(*self.classifier_1)

    def forward(self, input_img, mask):
        # last_conv = self.features_0(input_img)
        last_conv = input_img
        # print(last_conv.shape)
        last_conv_pruned = last_conv * mask[:, np.newaxis, np.newaxis]
        last_max_pool = self.features_1(last_conv_pruned)
        avg_pool = self.avgpool(last_max_pool)
        flatten = torch.flatten(avg_pool, 1)
        penultimate = self.classifier_0(flatten)
        # print(penultimate, penultimate.shape)
        last_fc = self.classifier_1(penultimate)
        penultimate = penultimate.detach().cpu().numpy()
        last_fc = last_fc.detach().cpu().numpy()
        return penultimate, last_fc
