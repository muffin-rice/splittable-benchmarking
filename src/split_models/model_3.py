'''This file contains the split versions (in ClientModel and ServerModel) of
configs/pascal_voc2012/supervised_compression/entropic_student/deeplabv3_splittable_resnet50-fp-beta0.16_from_deeplabv3_resnet50.yaml
This one uses the weights trained on ilsvrc for its backbone (the default path used by yoshi's load function)'''

import torch
from torch import nn
from torch.nn import functional
from copy import deepcopy
from collections import OrderedDict

# ==================================================== Client ====================================================

class ClientModel(nn.Module):
    '''client model for deeplabv3_splittable_resnet50-fp-beta0.16_from_deeplabv3_resnet50.yaml'''

    def __init__(self, student_model2):
        super(ClientModel, self).__init__()
        student_model = deepcopy(student_model2)
        self.encoder = student_model.backbone.bottleneck_layer.encoder
        self.training = student_model.training

        self.entropy_bottleneck = student_model.backbone.bottleneck_layer.entropy_bottleneck

    def encoder_forward(self, z):
        latent = self.encoder(z)
        latent_strings = self.entropy_bottleneck.compress(latent)

        return {'strings': [latent_strings], 'shape': latent.size()[-2:]}

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.encoder_forward(x)
        return features, input_shape

# ==================================================== SERVER ====================================================

class ServerModel(nn.Module):
    '''server model for deeplabv3_splittable_resnet50-fp-beta0.16_from_deeplabv3_resnet50.yaml'''
    def __init__(self, student_model2):
        super().__init__()
        student_model = deepcopy(student_model2)
        self.backbone = student_model.backbone

        self.entropy_bottleneck = self.backbone.bottleneck_layer.entropy_bottleneck
        self.decoder = self.backbone.bottleneck_layer.decoder
        self.backbone.bottleneck_layer = nn.Identity()

        self.classifier = student_model.classifier

        # self.aux_classifier = None
        self.aux_classifier = student_model.aux_classifier

    def backbone_forward(self, x):
        out = OrderedDict()
        for module_key, module in self.backbone.named_children():
            x = module(x)

            if module_key in self.backbone.return_layer_dict:
                out_name = self.backbone.return_layer_dict[module_key]
                out[out_name] = x

        return out

    def decode(self, strings, shape):
        latent_hat = self.entropy_bottleneck.decompress(strings[0], shape)
        return self.decoder(latent_hat)

    def forward(self, features, input_shape):
        '''image sizes '''
        features = torch.Tensor(self.decode(**features))
        features = self.backbone_forward(features)

        result = OrderedDict()
        x = features['out']
        x = self.classifier(x)
        x = functional.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result['out'] = x

        if self.aux_classifier is not None:
            x = features['aux']
            x = self.aux_classifier(x)
            x = functional.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result['aux'] = x
        return result