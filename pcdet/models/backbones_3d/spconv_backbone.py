from functools import partial

import torch
import torch.nn as nn
import torch.functional as F

import time

from ...utils.spconv_utils import replace_feature, spconv


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, bias=None, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        if bias is None:
            bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out


class VoxelBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] -> [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] -> [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] -> [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 176, 5] -> [200, 176, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }



    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        time1 = time.time()
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)
        
        x_conv1 = self.conv1(x)
        
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        batch_dict['time0'] = time.time() - time1
        time1 = time.time()
        
        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)
        batch_dict['time1'] = time.time() - time1
        
        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict
        
class VoxelBackBone8x_mod1(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        use_bias = self.model_cfg.get('USE_BIAS', None)
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, bias=use_bias, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] -> [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='subm2'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] -> [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='subm3'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] -> [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='subm4'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)

        self.conv_mha_3_2 = spconv.SparseSequential(
            # [200, 176, 5] -> [200, 176, 1]
            spconv.SparseConv3d(64, 128, (5, 3, 3), stride=(5, 1, 1), padding=(0,1,1),
                                bias=False, indice_key='spconv_mha_2'),
            norm_fn(128),
            nn.ReLU(),
        )

        # self.conv_out = spconv.SparseSequential(
        #     # [200, 176, 5] -> [200, 176, 2]
        #     spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
        #                         bias=False, indice_key='spconv_down2'),
        #     norm_fn(128),
        #     nn.ReLU(),
        # )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }



    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)
        
        x_conv1 = self.conv1(x)
        
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        x_mha3_2 = self.conv_mha_3_2(x_conv4)
        x_mha3_2 = x_mha3_2.dense().squeeze(2)
        batch_dict.update({'sp_mha':x_mha3_2})

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        # out = self.conv_out(x_conv4)
        
        batch_dict.update({
            # 'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict

class VoxelBackBone8x_mod2(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] -> [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] -> [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        # self.conv4 = spconv.SparseSequential(
        #     # [400, 352, 11] -> [200, 176, 5]
        #     block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        # )

        # self.conv_mha_3_2 = spconv.SparseSequential(
        #     # [200, 176, 5] -> [200, 176, 1]
        #     spconv.SparseConv3d(64, 128, (5, 3, 3), stride=(5, 1, 1), padding=(0,1,1),
        #                         bias=False, indice_key='spconv_mha_2'),
        #     norm_fn(128),
        #     nn.ReLU(),
        # )

        self.conv_mha_2 = spconv.SparseSequential(
            # [400, 352, 11] -> [200, 176, 1]
            spconv.SparseConv3d(64, 128, (11, 3, 3), stride=(11, 2, 2), padding=(0,1,1),
                                bias=False, indice_key='spconv_mha'),
            norm_fn(128),
            nn.ReLU(),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        # self.conv_out = spconv.SparseSequential(
        #     # [200, 176, 5] -> [200, 176, 2]
        #     spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
        #                         bias=False, indice_key='spconv_down2'),
        #     norm_fn(128),
        #     nn.ReLU(),
        # )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }



    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)
        
        x_conv1 = self.conv1(x)
        
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        # x_conv4 = self.conv4(x_conv3)

        x_mha3_2 = self.conv_mha_2(x_conv3)
        x_mha3_2 = x_mha3_2.dense().squeeze(2)
        batch_dict.update({'sp_mha':x_mha3_2})

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        # out = self.conv_out(x_conv4)
        
        # batch_dict.update({
        #     'encoded_spconv_tensor': out,
        #     'encoded_spconv_tensor_stride': 8
        # })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                # 'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict


class VoxelResBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        use_bias = self.model_cfg.get('USE_BIAS', None)
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, bias=use_bias, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, bias=use_bias, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, bias=use_bias, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, bias=use_bias, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, bias=use_bias, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, bias=use_bias, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, bias=use_bias, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128, 128, bias=use_bias, norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 128
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        # print('voxel_features', voxel_features.shape)
        # print('conv_input', x.spatial_shape)
        # print('x_conv1', x_conv1.spatial_shape)
        # print('x_conv1', x_conv2.spatial_shape)
        # print('x_conv1', x_conv3.spatial_shape)
        # print('x_conv1', x_conv4.spatial_shape)
        # print('x_conv1', out.spatial_shape)
        return batch_dict

class VoxelResBackBone8x_mod1(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        use_bias = self.model_cfg.get('USE_BIAS', None)
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, bias=use_bias, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, bias=use_bias, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, bias=use_bias, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, bias=use_bias, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, bias=use_bias, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, bias=use_bias, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, bias=use_bias, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128, 128, bias=use_bias, norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 176, 5] -> [200, 176, 2]
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )

        self.conv_mha_1 = spconv.SparseSequential(
            # [800, 704, 21] -> [200, 176, 1]
            spconv.SparseConv3d(32, 42, (21, 5, 5), stride=(21, 4, 4), padding=(0,1,1),
                                bias=False, indice_key='spconv_mha'),
            norm_fn(42),
            nn.ReLU(),
        )
        self.conv_mha_2 = spconv.SparseSequential(
            # [400, 352, 11] -> [200, 176, 1]
            spconv.SparseConv3d(64, 42, (11, 3, 3), stride=(11, 2, 2), padding=(0,1,1),
                                bias=False, indice_key='spconv_mha'),
            norm_fn(42),
            nn.ReLU(),
        )
        self.conv_mha_3 = spconv.SparseSequential(
            # [200, 176, 5] -> [200, 176, 1]
            spconv.SparseConv3d(128, 42, (5, 1, 1), stride=(5, 1, 1), padding=(0,0,0),
                                bias=False, indice_key='spconv_mha'),
            norm_fn(42),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 128
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        
        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)
        x_mha1 = self.conv_mha_1(x_conv2).dense().squeeze(2)# [4, 42, 200, 176]
        x_mha2 = self.conv_mha_2(x_conv3).dense().squeeze(2)
        x_mha3 = self.conv_mha_3(x_conv4).dense().squeeze(2)
        x_mha = torch.cat([x_mha1, x_mha2, x_mha3], dim=1)
        batch_dict.update({'sp_mha':x_mha})

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        # batch_dict.update({
        #     'multi_scale_3d_features': {
        #         'x_conv1': x_conv1,
        #         'x_conv2': x_conv2,
        #         'x_conv3': x_conv3,
        #         'x_conv4': x_conv4,
        #     }
        # })

        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        # print('voxel_features', voxel_features.shape)
        # print('conv_input', x.spatial_shape)
        # print('x_conv1', x_conv1.spatial_shape)
        # print('x_conv1', x_conv2.spatial_shape)
        # print('x_conv1', x_conv3.spatial_shape)
        # print('x_conv1', x_conv4.spatial_shape)
        # print('x_conv1', out.spatial_shape)
        return batch_dict

class VoxelResBackBone8x_mod2(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        use_bias = self.model_cfg.get('USE_BIAS', None)
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, bias=use_bias, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, bias=use_bias, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, bias=use_bias, norm_fn=norm_fn, indice_key='res2'),
            # SparseBasicBlock(32, 32, bias=use_bias, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, bias=use_bias, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, bias=use_bias, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, bias=use_bias, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128, 128, bias=use_bias, norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        # self.conv_out = spconv.SparseSequential(
        #     # [200, 150, 5] -> [200, 150, 2]
        #     spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
        #                         bias=False, indice_key='spconv_down2'),
        #     norm_fn(128),
        #     nn.ReLU(),
        # )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 128
        }
        # self.conv_out = spconv.SparseSequential(
        #     # [200, 176, 5] -> [200, 176, 2]
        #     spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
        #                         bias=False, indice_key='spconv_down2'),
        #     norm_fn(128),
        #     nn.ReLU(),
        # )

        # self.conv_mha_1 = spconv.SparseSequential(
        #     # [800, 704, 21] -> [200, 176, 1]
        #     spconv.SparseConv3d(32, 42, (21, 5, 5), stride=(21, 4, 4), padding=(0,1,1),
        #                         bias=False, indice_key='spconv_mha'),
        #     norm_fn(42),
        #     nn.Sigmoid(),
        # )

        # self.conv_mha_2 = spconv.SparseSequential(
        #     # [400, 352, 11] -> [200, 176, 1]
        #     spconv.SparseConv3d(64, 256, (11, 3, 3), stride=(11, 2, 2), padding=(0,1,1),
        #                         bias=False, indice_key='spconv_mha'),
        #     norm_fn(256),
        #     nn.ReLU(),
        # )
        # self.conv_mha_3_1 = spconv.SparseSequential(
        #     # [200, 176, 5] -> [200, 176, 1]
        #     spconv.SparseConv3d(128, 256, (3, 3, 3), stride=(1, 1, 1), padding=(1,1,1),
        #                         bias=False, indice_key='spconv_mha_1'),
        #     norm_fn(256),
        #     nn.ReLU(),
        # )
        self.conv_mha_3 = spconv.SparseSequential(
            # [200, 176, 5] -> [200, 176, 1]
            spconv.SparseConv3d(128, 256, (5, 3, 3), stride=(5, 1, 1), padding=(0,1,1),
                                bias=False, indice_key='spconv_mha_2'),
            norm_fn(256),
            nn.ReLU(),
        )

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        # time1 = time.time()
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # batch_dict.update( {'time0':time.time() - time1})
        # print(batch_dict['time0'])
        
        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        # out = self.conv_out(x_conv4)
        # x_mha1 = self.conv_mha_1(x_conv2).dense().squeeze(2)# [4, 42, 200, 176]
        # x_mha2 = self.conv_mha_2(x_conv3).dense().squeeze(2)
        # x_mha3_1 = self.conv_mha_3_1(x_conv4)
        # time1 = time.time()
        x_mha3_2 = self.conv_mha_3(x_conv4)
        x_mha3_2 = x_mha3_2.dense().squeeze(2)
        # batch_dict.update({'time1':time.time() - time1})
        # x_mha3_2  = x_mha3_2 + torch.sum(x_mha3_1.dense(), dim=2).squeeze(2)
        # print(x_ind.shape)
        # exit()
        # x_mha = torch.cat([x_mha1, x_mha2, x_mha3], dim=1)
        batch_dict.update({'sp_mha':x_mha3_2})

        batch_dict.update({
            # 'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        # batch_dict.update({
        #     'multi_scale_3d_features': {
        #         'x_conv1': x_conv1,
        #         'x_conv2': x_conv2,
        #         'x_conv3': x_conv3,
        #         'x_conv4': x_conv4,
        #     }
        # })

        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        # print('voxel_features', voxel_features.shape)
        # print('conv_input', x.spatial_shape)
        # print('x_conv1', x_conv1.spatial_shape)
        # print('x_conv1', x_conv2.spatial_shape)
        # print('x_conv1', x_conv3.spatial_shape)
        # print('x_conv1', x_conv4.spatial_shape)
        # print('x_conv1', out.spatial_shape)
        return batch_dict

class VoxelResBackBone8x_mod2_1(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        use_bias = self.model_cfg.get('USE_BIAS', None)
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, bias=use_bias, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, bias=use_bias, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, bias=use_bias, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, bias=use_bias, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, bias=use_bias, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, bias=use_bias, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, bias=use_bias, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128, 128, bias=use_bias, norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        # self.conv_out = spconv.SparseSequential(
        #     # [200, 176, 5] -> [200, 176, 2]
        #     spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(3, 1, 1), padding=last_pad,
        #                         bias=False, indice_key='spconv_down2'),
        #     norm_fn(128),
        #     nn.ReLU(),
        # )

        # self.conv_mha_1 = spconv.SparseSequential(
        #     # [800, 704, 21] -> [200, 176, 1]
        #     spconv.SparseConv3d(32, 42, (21, 5, 5), stride=(21, 4, 4), padding=(0,1,1),
        #                         bias=False, indice_key='spconv_mha'),
        #     norm_fn(42),
        #     nn.Sigmoid(),
        # )

        # self.conv_mha_2 = spconv.SparseSequential(
        #     # [400, 352, 11] -> [200, 176, 1]
        #     spconv.SparseConv3d(64, 256, (11, 3, 3), stride=(11, 2, 2), padding=(0,1,1),
        #                         bias=False, indice_key='spconv_mha'),
        #     norm_fn(256),
        #     nn.ReLU(),
        # )

        self.conv_mha_3_2 = spconv.SparseSequential(
            # [200, 176, 5] -> [200, 176, 1]
            spconv.SparseConv3d(128, 128, (5, 3, 3), stride=(5, 1, 1), padding=(0,1,1),
                                bias=False, indice_key='spconv_mha_2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 128
        }
        

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        
        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        # out = self.conv_out(x_conv4)
        # x_mha1 = self.conv_mha_1(x_conv2).dense().squeeze(2)# [4, 42, 200, 176]
        # x_mha2 = self.conv_mha_2(x_conv3).dense().squeeze(2)
        x_mha3_2 = self.conv_mha_3_2(x_conv4).dense().squeeze(2)
        # print(x_ind.shape)
        # exit()
        # x_mha = torch.cat([x_mha1, x_mha2, x_mha3], dim=1)
        batch_dict.update({'sp_mha':x_mha3_2})

        batch_dict.update({
            # 'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        # batch_dict.update({
        #     'multi_scale_3d_features': {
        #         'x_conv1': x_conv1,
        #         'x_conv2': x_conv2,
        #         'x_conv3': x_conv3,
        #         'x_conv4': x_conv4,
        #     }
        # })

        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        # print('voxel_features', voxel_features.shape)
        # print('conv_input', x.spatial_shape)
        # print('x_conv1', x_conv1.spatial_shape)
        # print('x_conv1', x_conv2.spatial_shape)
        # print('x_conv1', x_conv3.spatial_shape)
        # print('x_conv1', x_conv4.spatial_shape)
        # print('x_conv1', out.spatial_shape)
        return batch_dict

class VoxelResBackBone8x_mod(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        use_bias = self.model_cfg.get('USE_BIAS', None)
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, bias=use_bias, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, bias=use_bias, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, bias=use_bias, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, bias=use_bias, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, bias=use_bias, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, bias=use_bias, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, bias=use_bias, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128, 128, bias=use_bias, norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 176, 5] -> [200, 176, 2]
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )

        self.conv_mha_1 = spconv.SparseSequential(
            # [800, 704, 21] -> [200, 176, 1]
            spconv.SparseConv3d(32, 42, (21, 5, 5), stride=(21, 4, 4), padding=(0,1,1),
                                bias=False, indice_key='spconv_mha'),
            norm_fn(42),
            nn.ReLU(),
        )
        self.conv_mha_2 = spconv.SparseSequential(
            # [400, 352, 11] -> [200, 176, 1]
            spconv.SparseConv3d(64, 42, (11, 3, 3), stride=(11, 2, 2), padding=(0,1,1),
                                bias=False, indice_key='spconv_mha'),
            norm_fn(42),
            nn.ReLU(),
        )
        self.conv_mha_3 = spconv.SparseSequential(
            # [200, 176, 5] -> [200, 176, 1]
            spconv.SparseConv3d(128, 42, (5, 1, 1), stride=(5, 1, 1), padding=(0,0,0),
                                bias=False, indice_key='spconv_mha'),
            norm_fn(42),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 128
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        
        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)
        x_mha1 = self.conv_mha_1(x_conv2).dense().squeeze(2)# [4, 42, 200, 176]
        x_mha2 = self.conv_mha_2(x_conv3).dense().squeeze(2)
        x_mha3 = self.conv_mha_3(x_conv4).dense().squeeze(2)
        x_mha = torch.cat([x_mha1, x_mha2, x_mha3], dim=1)
        batch_dict.update({'sp_mha':x_mha})

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        # batch_dict.update({
        #     'multi_scale_3d_features': {
        #         'x_conv1': x_conv1,
        #         'x_conv2': x_conv2,
        #         'x_conv3': x_conv3,
        #         'x_conv4': x_conv4,
        #     }
        # })

        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        # print('voxel_features', voxel_features.shape)
        # print('conv_input', x.spatial_shape)
        # print('x_conv1', x_conv1.spatial_shape)
        # print('x_conv1', x_conv2.spatial_shape)
        # print('x_conv1', x_conv3.spatial_shape)
        # print('x_conv1', x_conv4.spatial_shape)
        # print('x_conv1', out.spatial_shape)
        return batch_dict

class VoxelResBackBone8x_FV(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        use_bias = self.model_cfg.get('USE_BIAS', None)
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            SparseBasicBlock(16, 16, bias=use_bias, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, bias=use_bias, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [800, 704, 21] -> [400, 352, 11]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, bias=use_bias, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, bias=use_bias, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [400, 352, 11] -> [200, 176, 5]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, bias=use_bias, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, bias=use_bias, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [200, 176, 5] -> [200, 176, 2]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, bias=use_bias, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128, 128, bias=use_bias, norm_fn=norm_fn, indice_key='res4'),
        )
        self.conv_fv1 = spconv.SparseSequential(
            # [400, 352, 11] -> [134, 176, 11]
            spconv.SubMConv3d(64, 64, 3, stride=1, padding=1,
                                bias=False, indice_key='spconv_fv_sm1'),
            norm_fn(64),
            nn.ReLU(),
            spconv.SparseConv3d(64, 64, 3, stride=(1, 3, 2), padding=1,
                                bias=False, indice_key='spconv_fv1'),
            norm_fn(64),
            nn.ReLU(),
        )
        self.conv_fv2 = spconv.SparseSequential(
            # [134, 176, 11] -> [45, 176, 11]
            spconv.SubMConv3d(64, 64, 3, stride=1, padding=1,
                                bias=False, indice_key='spconv_fv_sm2'),
            norm_fn(64),
            nn.ReLU(),
            spconv.SparseConv3d(64, 64, 3, stride=(1, 3, 1), padding=1,
                                bias=False, indice_key='spconv_fv2'),
            norm_fn(64),
            nn.ReLU(),
        )
        # self.conv_fv3 = spconv.SparseSequential(
        #     # [100, 176, 11] -> [50, 176, 11]
        #     spconv.SubMConv3d(64, 64, 3, stride=1, padding=1,
        #                         bias=False, indice_key='spconv_fv_sm3'),
        #     norm_fn(64),
        #     nn.ReLU(),
        #     spconv.SparseConv3d(64, 64, 3, stride=(1, 3, 1), padding=1,
        #                         bias=False, indice_key='spconv_fv3'),
        #     norm_fn(64),
        #     nn.ReLU(),
        # )
        # self.conv_fv4 = spconv.SparseConv3d(64, 64, 3, stride=(1, 2, 1), padding=1,
        #                         bias=False, indice_key='spconv_fv4')# [50, 176, 11] -> [25, 176, 11]
        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 176, 5] -> [200, 176, 2]
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 128
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)# [41, 1600, 1408]

        x_conv1 = self.conv1(x)# 41, 1600, 1408]
        x_conv2 = self.conv2(x_conv1)# [21, 800, 704]
        x_conv3 = self.conv3(x_conv2)# [11, 400, 352]
        x_conv4 = self.conv4(x_conv3)# [5, 200, 176]
        x_conv_fn1 = self.conv_fv1(x_conv3)
        x_conv_fn2 = self.conv_fv2(x_conv_fn1)
        # x_conv_fn3 = self.conv_fv3(x_conv_fn2)
        # x_conv_fn4 = self.conv_fv4(x_conv_fn3)
        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)# [200, 176, 2]
        # print('out', type(out))

        # print('voxel_features', voxel_features.shape)
        # print('conv_input', x.spatial_shape)
        # print('x_conv1', x_conv1.spatial_shape)
        # print('x_conv2', x_conv2.spatial_shape)
        # print('x_conv3', x_conv3.spatial_shape)
        # print('x_conv4', x_conv4.spatial_shape)
        # print('x_conv_fn1', x_conv_fn1.spatial_shape)
        # print('x_conv_fn2', x_conv_fn2.spatial_shape)
        # print('x_conv_fn3', x_conv_fn3.spatial_shape)
        # print('out', out.spatial_shape)
        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8,
            'encoded_spconv_tensor_fv': x_conv_fn2,
            'encoded_spconv_tensor_stride_fv': 9,
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        
        return batch_dict

class VoxelResBackBone8x_mod3(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        use_bias = self.model_cfg.get('USE_BIAS', None)
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, bias=use_bias, norm_fn=norm_fn, indice_key='res1'),
            # SparseBasicBlock(16, 16, bias=use_bias, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, bias=use_bias, norm_fn=norm_fn, indice_key='res2'),
            # SparseBasicBlock(32, 32, bias=use_bias, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, bias=use_bias, norm_fn=norm_fn, indice_key='res3'),
            # SparseBasicBlock(64, 64, bias=use_bias, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(64, 64, bias=use_bias, norm_fn=norm_fn, indice_key='res4'),
            # SparseBasicBlock(128, 128, bias=use_bias, norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 176, 5] -> [200, 176, 2]
            spconv.SparseConv3d(64, 64, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(64),
            nn.ReLU(),
        )

        self.conv_mha_3 = spconv.SparseSequential(
            # [200, 176, 5] -> [200, 176, 1]
            spconv.SparseConv3d(64, 42, (5, 1, 1), stride=(5, 1, 1), padding=(0,0,0),
                                bias=False, indice_key='spconv_mha'),
            norm_fn(42),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 128
        }
        

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        
        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)
        # x_mha1 = self.conv_mha_1(x_conv2).dense().squeeze(2)# [4, 42, 200, 176]
        # x_mha2 = self.conv_mha_2(x_conv3).dense().squeeze(2)
        x_mha3 = self.conv_mha_3(x_conv4).dense().squeeze(2)
        # x_mha = torch.cat([x_mha1, x_mha2, x_mha3], dim=1)
        batch_dict.update({'sp_mha':x_mha3})

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        # batch_dict.update({
        #     'multi_scale_3d_features': {
        #         'x_conv1': x_conv1,
        #         'x_conv2': x_conv2,
        #         'x_conv3': x_conv3,
        #         'x_conv4': x_conv4,
        #     }
        # })

        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        # print('voxel_features', voxel_features.shape)
        # print('conv_input', x.spatial_shape)
        # print('x_conv1', x_conv1.spatial_shape)
        # print('x_conv2', x_conv2.spatial_shape)
        # print('x_conv3', x_conv3.spatial_shape)
        # print('x_conv4', x_conv4.spatial_shape)
        # print('out', out.spatial_shape)
        return batch_dict

class VoxelResBackBone8x_mod22_7(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        use_bias = self.model_cfg.get('USE_BIAS', None)
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, bias=use_bias, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, bias=use_bias, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, bias=use_bias, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, bias=use_bias, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, bias=use_bias, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, bias=use_bias, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, bias=use_bias, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128, 128, bias=use_bias, norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 176, 5] -> [200, 176, 2]
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )

        self.conv_mha_3 = nn.MaxPool3d((5, 3, 3), stride=(5,1,1),padding=1)# [200, 176, 5] -> [200, 176, 1]
        self.mha3_sigmoid = nn.Sigmoid()

        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 128
        }
        

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        
        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)
        # x_mha1 = self.conv_mha_1(x_conv2).dense().squeeze(2)# [4, 42, 200, 176]
        # x_mha2 = self.conv_mha_2(x_conv3).dense().squeeze(2)
        x_mha3 = self.conv_mha_3(x_conv4.dense()).squeeze(2)
        x_mha3 = self.mha3_sigmoid(x_mha3)
        # print(x_mha3.shape)
        # x_mha = torch.cat([x_mha1, x_mha2, x_mha3], dim=1)
        batch_dict.update({'sp_mha':x_mha3})

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        # batch_dict.update({
        #     'multi_scale_3d_features': {
        #         'x_conv1': x_conv1,
        #         'x_conv2': x_conv2,
        #         'x_conv3': x_conv3,
        #         'x_conv4': x_conv4,
        #     }
        # })

        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        # print('voxel_features', voxel_features.shape)
        # print('conv_input', x.spatial_shape)
        # print('x_conv1', x_conv1.spatial_shape)
        # print('x_conv1', x_conv2.spatial_shape)
        # print('x_conv1', x_conv3.spatial_shape)
        # print('x_conv1', x_conv4.spatial_shape)
        # print('x_conv1', out.spatial_shape)
        return batch_dict