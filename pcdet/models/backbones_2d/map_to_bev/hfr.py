import torch.nn as nn
import torch
from pcdet.utils.spconv_utils import replace_feature, spconv

from ..aspp import ASPPNeck


class HeightFeatureRefineNet(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.num_height = self.model_cfg.NUM_HEIGHT if self.model_cfg.NUM_HEIGHT else 10
        self.conv1d_sp = spconv.SparseSequential(
            spconv.SubMConv3d(128, 128, (3,1,1), padding=(1,0,0), bias=False, indice_key='subm_h21'),
            nn.ReLU(),
            spconv.SubMConv3d(128, 128, (3,1,1), padding=(1,0,0), bias=False, indice_key='subm_h22'),
            nn.ReLU(),
            spconv.SubMConv3d(128, 256, 1, padding=0, bias=False, indice_key='subm_h23'),
            nn.ReLU(),
        )
        
        self.linear = nn.Sequential(
            nn.Linear(self.num_height, self.num_height//2),
            nn.ReLU(),
            nn.Linear(self.num_height//2, self.num_height//4),
            nn.ReLU(),
            nn.Linear(self.num_height//4, 1),
            nn.Sigmoid(),
        )



        self.aspp = ASPPNeck(model_cfg, self.model_cfg.get('INPUT_CHANNELS'))

    def hfr_proc(self, x):
        x = self.conv1d_sp(x)
        x_conv1d = x.dense()
        x_conv1d = x_conv1d.permute(0,1,3,4,2)

        out = self.linear(x_conv1d).permute(0,1,4,2,3).squeeze(2)
        out = out.contiguous()
        return out

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
        spconv_4 = batch_dict['multi_scale_3d_features']['x_conv4']
        hfa_out = self.hfr_proc(spconv_4)
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        
        batch_dict['spatial_features'] = spatial_features.contiguous()
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        batch_dict = self.aspp(batch_dict)
        aspp_out = batch_dict['spatial_features_2d']
        aspp_out = aspp_out*hfa_out
        batch_dict['spatial_features_2d'] = aspp_out


        return batch_dict

