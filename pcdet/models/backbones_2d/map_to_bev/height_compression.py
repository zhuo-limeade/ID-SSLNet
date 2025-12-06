import torch.nn as nn
import torch
import time

from torch.utils.tensorboard import SummaryWriter


class HeightCompression_mod1(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.conv_mha = nn.Sequential(
            # nn.Conv2d(128, 256,
            #     kernel_size=3,padding=1,),
            # nn.BatchNorm2d(256, eps=1e-3, momentum=1e-2),
            # nn.ReLU(inplace=True),
            nn.Conv2d(256, 256,
                kernel_size=3,padding=1,),
            nn.BatchNorm2d(256, eps=1e-3, momentum=1e-2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256,
                kernel_size=3,padding=1,),
            nn.BatchNorm2d(256, eps=1e-3, momentum=1e-2),
            nn.ReLU(inplace=True)
        )
        # self.writer = SummaryWriter("/home/ken/program/tensorboard/feature/f1")
        # self.pointer = 0


    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        
        sp_mha = batch_dict['sp_mha']# [N, 42*3, 200, 176, ]引入3D特征
        time1 = time.time()
        sp_mha = self.conv_mha(sp_mha)
        batch_dict['time2'] = time.time() - time1
        # print(sp_mha.shape)
        # x = torch.sum(sp_mha, dim=1)
        # print(x.shape)
        # self.writer.add_image("train", x, self.pointer)
        # self.writer.close()
        # self.pointer +=1
        # exit()
        batch_dict['spatial_features'] = sp_mha
        batch_dict['spatial_features_stride'] = 256
        return batch_dict

class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        time1 = time.time()
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        batch_dict['time2'] = time.time() - time1

        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        return batch_dict

class HeightCompression_FV(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']

        encoded_spconv_tensor_fv = batch_dict['encoded_spconv_tensor_fv']# [n,64,5,45,176]
        spatial_features = encoded_spconv_tensor_fv.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * H, D, W)# [n,2880,5,176]

        batch_dict['spatial_features_fv'] = spatial_features
        batch_dict['spatial_features_stride_fv'] = batch_dict['encoded_spconv_tensor_stride_fv']
        # print(spatial_features.shape)
        return batch_dict

    
