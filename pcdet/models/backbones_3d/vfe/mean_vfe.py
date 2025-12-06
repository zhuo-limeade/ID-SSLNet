import torch

from .vfe_template import VFETemplate

import open3d

class MeanVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features

    def get_output_feature_dim(self):
        return self.num_point_features

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
        # 求每个voxel内 所有点的和
        # eg:SECOND  shape (Batch*16000, 5, 4) -> (Batch*16000, 4)
        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
        # 正则化项， 保证每个voxel中最少有一个点，防止除0
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
        # 求每个voxel内点坐标的平均值
        points_mean = points_mean / normalizer
        # 将处理好的voxel_feature信息重新加入batch_dict中
        batch_dict['voxel_features'] = points_mean.contiguous()
        # print('points_mean', points_mean.shape)
        # points = points_mean.cpu().numpy()
        # pts = open3d.geometry.PointCloud()
        # pts.points = open3d.utility.Vector3dVector(points[:, :3])
        # open3d.visualization.draw_geometries([pts])
        return batch_dict