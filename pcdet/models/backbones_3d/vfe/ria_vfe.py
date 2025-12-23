import torch
import torch.nn as nn 

from .vfe_template import VFETemplate

class Reflection_Feature_Extractor(nn.Module):
    def __init__(self, channels, max_num_points) -> None:
        super().__init__()
        # input()
        self.rfe_sequence = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.Sigmoid(),
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.Sigmoid(),
            nn.Conv1d(channels, 1, 1),
        )
        self.linear_refine = nn.Sequential(
            nn.Linear(max_num_points, max_num_points),
            nn.Sigmoid(),
            nn.Linear(max_num_points, max_num_points),
            nn.Sigmoid(),
            nn.Linear(max_num_points, max_num_points),
        )
        
    def forward(self, input, batch_size):
        N, M, C = input.shape
        x = input.permute(0,2,1)# [N, C, M]
        x = self.rfe_sequence(x)# [N, C, M]->[N, 1, M]
        x = x.squeeze(1)# [N, 1, M]->[N, M]
        x = self.linear_refine(x)# [N, M]->[N, M]
        out = x.view(N,M,1)# [N, M]->[N, M, 1]
        return out



class RIA_VFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = 9
        self.rfe = Reflection_Feature_Extractor(4, 10)
        self.threshold = 0.1

    def get_output_feature_dim(self):
        return self.num_point_features
    
    def foreground_split(self, refined_voxel, voxel_features):
        refined_voxel = torch.sigmoid(refined_voxel)
        foreground_voxel = refined_voxel.clone()
        background_voxel = refined_voxel.clone()
        foreground_point = voxel_features.clone()
        foreground_voxel[refined_voxel<=self.threshold] = 0
        background_voxel[refined_voxel>self.threshold] = 0
        a = refined_voxel<=self.threshold
        a = torch.cat([a]*4, dim=-1)
        foreground_point[a==1] = 0
        foreground_point_num = refined_voxel>self.threshold
        background_point_num = refined_voxel<=self.threshold
        foreground_point_num = foreground_point_num.squeeze(dim=-1).sum(dim=1).unsqueeze(-1)
        background_point_num = background_point_num.squeeze(dim=-1).sum(dim=1).unsqueeze(-1)
        return torch.concat([voxel_features,foreground_point[...,:-1], foreground_voxel, background_voxel, ],dim=-1), foreground_point_num, background_point_num
        

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        # here use the mean_vfe module to substitute for the original pointnet extractor architecture
        voxel_features, voxel_num_points = batch_dict['voxels'], batch_dict['voxel_num_points']
        refined_feature = self.rfe(voxel_features, batch_dict['batch_size'])
        refined_voxel, foreground_point_num, background_point_num = self.foreground_split(refined_feature, voxel_features)
        # voxel_features:[N, M, C], voxel_num_points
        # 求每个voxel内 所有点的和
        # eg:SECOND  shape (Batch*16000, 5, 4) -> (Batch*16000, 4)
        points_mean = refined_voxel[:, :, :].sum(dim=1, keepdim=False)
        # 正则化项， 保证每个voxel中最少有一个点，防止除0
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
        # 求每个voxel内点坐标的平均值
        points_mean = points_mean / normalizer
        # print(points_mean[...,4:7].shape, normalizer.shape, foreground_point_num.shape)
        foreground_point_num = torch.clamp_min(foreground_point_num, min=1.0)
        background_point_num = torch.clamp_min(background_point_num, min=1.0)
        points_mean[...,4:7] = points_mean[...,4:7]*normalizer/foreground_point_num
        points_mean[...,-1:] = points_mean[...,-1:]*normalizer/background_point_num
        # 将处理好的voxel_feature信息重新加入batch_dict中
        batch_dict['voxel_features'] = points_mean.contiguous()

        return batch_dict
