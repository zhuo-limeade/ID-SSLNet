import torch
import numpy as np

class Global_Point_Offset2D():
    '''
        args:
            全局格点偏移能够实现整个图中所有特征的曼哈顿距离的加权和
    '''
    def __init__(self):
        pass
    
    '''
        args:
            用于生成曼哈顿型偏移模板
    '''
    def generate_offset_template(self, distance=0):
        template = torch.ones((distance*2+1, distance*2+1))# 获得R*R模板
        template_size = torch.tensor(template.shape)
        template = template.nonzero()-torch.div(template_size[:2], 2, rounding_mode='floor') # 获得相对模板中心距离
        template_index = torch.abs(template[:,0]) + torch.abs(template[:,1])# 获取每个位置相对中心距离
        template_index = template_index <= distance
        template = template * template_index.unsqueeze_(1)
        template_index = torch.any(template!=0, dim=1).nonzero().squeeze(1)
        template = template[template_index]# 获取全局偏移量
        return template, template_index

    '''
        args:
            用于曼哈顿型体素采样
    '''
    def global_offset(self, voxel, coord, weight=None, distance=0):
        if weight == None :
            weight = torch.ones(2*distance*(distance+1))
        template, template_index = self.generate_offset_template(distance)
        coord_temp = coord.clone()
        
        for index, offset_index in enumerate(template):
            coord_temp[:,2:] = coord_temp[:,2:] + offset_index# 将坐标进行全局偏移
            equal_coords = torch.eq(coord_temp, coord)
            
            all_equal_indices = torch.all(equal_coords, dim=1)# 检查偏移坐标与当前坐标相等
            equal_indices = torch.nonzero(all_equal_indices).squeeze()# 获取相等的坐标的索引

            equal_coords = coord_temp[equal_indices] - coord_temp[:,2:]
            match_index = torch.tensor([torch.all(coord == equal_coords[i] , dim=1).nonzero() for i in range(len(equal_coords))])

            voxel[match_index,:] += voxel[equal_indices,:]*weight
        return voxel

class Global_Point_Offset_Aggregation():
    '''
        args:
            全局格点偏移能够实现整个图中所有特征的曼哈顿距离的加权和
    '''
    def __init__(self):
        pass
    
    '''
        args:
            用于生成曼哈顿型偏移模板
    '''
    def generate_offset_template(self, distance=0):
        template = torch.ones((distance*2+1, distance*2+1))# 获得R*R模板
        template_size = torch.tensor(template.shape)
        template = template.nonzero()-torch.div(template_size[:2], 2, rounding_mode='floor') # 获得相对模板中心距离
        template_index = torch.abs(template[:,0]) + torch.abs(template[:,1])# 获取每个位置相对中心距离
        template_index = template_index <= distance
        template = template * template_index.unsqueeze_(1)
        template_index = torch.any(template!=0, dim=1).nonzero().squeeze(1)
        template = template[template_index]# 获取全局偏移量
        return template, template_index

    '''
        args:
            用于曼哈顿型体素采样
    '''
    def global_offset_aggregate(self, voxel, coord, weight=None, distance=0):
        if weight == None :
            weight = torch.ones(2*distance*(distance+1))
        template, template_index = self.generate_offset_template(distance)
        coord_temp = coord.clone()
        
        for index, offset_index in enumerate(template):
            coord_temp[:,2:] = coord_temp[:,2:] + offset_index# 将坐标进行全局偏移
            equal_coords = torch.eq(coord_temp, coord)
            
            all_equal_indices = torch.all(equal_coords, dim=1)# 检查偏移坐标与当前坐标相等
            equal_indices = torch.nonzero(all_equal_indices).squeeze()# 获取相等的坐标的索引

            equal_coords = coord_temp[equal_indices] - coord_temp[:,2:]
            match_index = torch.tensor([torch.all(coord == equal_coords[i] , dim=1).nonzero() for i in range(len(equal_coords))])

            voxel[match_index,:] += voxel[equal_indices,:]*weight
        return voxel



if __name__ == '__main__':
    # # eq_projection_demo()
    gpo = Global_Point_Offset2D()
    gpo.generate_offset_template(distance=1)

    # eq_projection_demo()
    # A = torch.tensor([[1.0, 2.0, 3.0],
    #               [4.0, 5.0, 6.0],
    #               [7.0, 8.0, 9.0]])
    # B = torch.tensor([[4.0, 5.0, 6.0],
    #                 [1.0, 2.0, 3.0]])
    # print(torch.tensor([torch.all(A == B[i] , dim=1).nonzero() for i in range(len(B))]).shape)
    pass
