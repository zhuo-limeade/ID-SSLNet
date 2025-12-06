import torch
import numpy as np
import torch.nn as nn
from torch.nn.functional import sigmoid
from ...ops.iou3d_nms import iou3d_nms_utils
import torch.functional as F
from torchvision.ops import RoIPool


from .anchor_head_template import AnchorHeadTemplate, AnchorHeadTemplate_FV


class AnchorHeadSingle_MD(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)
        self.cls_channel = self.num_anchors_per_location * self.num_class
        self.box_channel = self.num_anchors_per_location * self.box_coder.code_size
        self.dir_channel = self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS
        self.conv = nn.Conv2d(
            input_channels, 
            self.cls_channel+self.box_channel+self.dir_channel,
            kernel_size=1
        )
        self.se_blc72 = SE_MD_Block(72, ratio=8)
        # self.mhax = MultiHeadSelfAttentionClassifier(200, 8, 16, 200)
        # self.mhay = MultiHeadSelfAttentionClassifier(176, 8, 15, 176) 
        # self.mhax = MultiHeadSelfAttention(200, 8)
        # self.mhay = MultiHeadSelfAttention(176, 8)
        # self.conv_cls = nn.Conv2d(
        #     input_channels, self.num_anchors_per_location * self.num_class,
        #     kernel_size=1
        # )
        # self.conv_box = nn.Conv2d(
        #     input_channels, self.num_anchors_per_location * self.box_coder.code_size,
        #     kernel_size=1
        # )

        # if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
        #     self.conv_dir_cls = nn.Conv2d(
        #         input_channels,
        #         self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
        #         kernel_size=1
        #     )
        # else:
        #     self.conv_dir_cls = None
        self.init_weights()

        # self.se_blc18 = SE_MD_Block(18, ratio=1.5)
        # self.se_blc12 = SE_MD_Block(12, ratio=1.5)
        # self.se_blc42 = SE_MD_Block(42, ratio=3)


    def init_weights(self):
        pi = 0.01
        # nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        # nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

        nn.init.constant_(self.conv.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']
        preds = self.conv(spatial_features_2d)# [N, B*6, 200, 176, ]
        

        # qx = torch.mean(preds,dim=3)
        # qy = torch.mean(preds,dim=2)
        # # print(qx.shape, qy.shape)
        # qx = self.mhax(qx)
        # qy = self.mhay(qy)
        # q = torch.einsum('ijk,ijl->ijkl', qx, qy)
        # preds = q+preds
        preds = self.se_blc72(preds) + preds
        # print(preds.shape)
        preds = preds.permute(0, 2, 3, 1)  # [N, 200, 176, B*6]
        self.forward_ret_dict['cls_preds'] = preds[:,:,:,:self.cls_channel].contiguous()
        self.forward_ret_dict['box_preds'] = preds[:,:,:,self.cls_channel:self.cls_channel+self.box_channel].contiguous()
        self.forward_ret_dict['dir_cls_preds'] = preds[:,:,:,self.cls_channel+self.box_channel:self.cls_channel+self.box_channel+self.dir_channel].contiguous()
        # print(self.forward_ret_dict['cls_preds'].shape,self.forward_ret_dict['cls_preds'].shape,self.forward_ret_dict['cls_preds'].shape)

        # cls_preds = self.conv_cls(spatial_features_2d)
        # box_preds = self.conv_box(spatial_features_2d)

        # cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, 200, 176, C*6]
        # box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, 200, 176, B*6]
        # # 
        # # cls_preds = cls_preds + self.se_blc18(cls_preds)
        # # box_preds = box_preds + self.se_blc42(box_preds)

        # self.forward_ret_dict['cls_preds'] = cls_preds
        # self.forward_ret_dict['box_preds'] = box_preds


        # if self.conv_dir_cls is not None:
        #     dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
        #     dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
        #     self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
            
        #     # dir_cls_preds = dir_cls_preds + self.se_blc12(dir_cls_preds)
        # else:
        #     dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            # batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
            #     batch_size=data_dict['batch_size'],
            #     cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            # )
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=self.forward_ret_dict['cls_preds'], 
                box_preds=self.forward_ret_dict['box_preds'], 
                dir_cls_preds=self.forward_ret_dict['dir_cls_preds']
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict

class AnchorHeadSingle(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)
        self.cls_channel = self.num_anchors_per_location * self.num_class
        self.box_channel = self.num_anchors_per_location * self.box_coder.code_size
        self.dir_channel = self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS
        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, 200, 176, C*6]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, 200, 176, B*6]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict

class AnchorHeadSingle_FV(AnchorHeadTemplate_FV):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)
        self.cls_channel = self.num_anchors_per_location * self.num_class
        self.box_channel = self.num_anchors_per_location * self.box_coder.code_size
        self.dir_channel = self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS
        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()
        if self.model_cfg.TRAINING:
            self.roiasa = ROIAlignSpatialAttention_modi1()
        
        

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)
    
    # def gen_fv_gt_mask(self, data_dict, img_size=[11,176 ]):
    #     spatial_features_2d_fv = data_dict['spatial_features_2d_fv']
    #     batch_size = data_dict['batch_size']
    #     spatial_features_2d_fv = spatial_features_2d_fv.view(batch_size, 72, -1)

    
        
    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']# [4, 512, 200, 176]

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, 200, 176, C*6]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, 200, 176, B*6]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds


        preds_fv = data_dict['spatial_features_2d_fv'].permute(0, 2, 3, 1)
        
        self.forward_ret_dict['cls_preds_fv'] = preds_fv[:,:,:,:self.cls_channel].contiguous()
        self.forward_ret_dict['box_preds_fv'] = preds_fv[:,:,:,self.cls_channel:self.cls_channel+self.box_channel].contiguous()
        self.forward_ret_dict['dir_cls_preds_fv'] = preds_fv[:,:,:,self.cls_channel+self.box_channel:self.cls_channel+self.box_channel+self.dir_channel].contiguous()
        
        

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=self.forward_ret_dict['cls_preds'], 
                box_preds=self.forward_ret_dict['box_preds'], 
                dir_cls_preds=self.forward_ret_dict['dir_cls_preds']
            )
        data_dict['batch_cls_preds'] = batch_cls_preds
        data_dict['batch_box_preds'] = batch_box_preds
        data_dict['cls_preds_normalized'] = False
        anchor_labels, preds_box, total_classes = self.post_processing(data_dict)
        # print('len', preds_box[0].shape)
        targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
        self.forward_ret_dict.update(targets_dict)
        targets_dict_fv = self.assign_targets_fv(
                gt_boxes=data_dict['gt_boxes']
            )
        self.forward_ret_dict.update(targets_dict_fv)

        if self.training:
            
            l1_loss = self.roiasa(preds_box, data_dict, self.forward_ret_dict, is_train=self.training)
            # l1_loss = 0
            self.forward_ret_dict['l1_loss'] = l1_loss
            

        if not self.training or self.predict_boxes_when_training:
            data_dict['total_scores'] = total_classes
            # batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
            #     batch_size=data_dict['batch_size'],
            #     cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            # )
            # data_dict['batch_cls_preds'] = batch_cls_preds
            # data_dict['batch_box_preds'] = batch_box_preds
            # data_dict['cls_preds_normalized'] = False

            fv_cls_out, scores_list = self.roiasa(preds_box, data_dict, self.forward_ret_dict, is_train=self.training)
            data_dict['fv_cls_out'] = anchor_labels
            data_dict['preds_box'] = preds_box
            

            #**********生成pred_dict

        # gt信息在data_dict['gt_boxes']中,格式为[N,8],第二维最后一列为分类

        # forward_ret_dict存储在父类AnchorHeadTemplate_FV中,进行loss计算操作
        
        return data_dict


class SE_MD_Block(nn.Module):
    def __init__(self, inchannel, ratio=8):
        super(SE_MD_Block, self).__init__()
        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, int(inchannel / ratio), bias=False),  # 从 c -> c/r
            nn.LeakyReLU(negative_slope=0.1),
            # nn.ReLU(),
            nn.Linear(int(inchannel / ratio), inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        # 读取批数据图片数量及通道数
        b, c, h, w,  = x.size()
        x_redim = x.clone()
        # x_redim = x.permute(0,3,1,2)# 改变维度排列顺序
        # if x_redim[0,0,:,:].equal(x[0,:,:,0]):
        #     print("yes")
        # Fsq操作：经池化后输出b*c的矩阵
        y = self.gap(x_redim).view(b, c)
        # Fex操作：经全连接层输出（b，c，1，1）矩阵
        y = self.fc(y).view(b, c, 1, 1)
        # Fscale操作：将得到的权重乘以原来的特征图x
        x = x_redim * y.expand_as(x_redim)
        return x


class ROIAlignSpatialAttention(nn.Module):
    def __init__(self, window_size=(5,5)):
        super(ROIAlignSpatialAttention, self).__init__()
        # self.roi_pool = RoIPool((7,7), spatial_scale=1.0)
        self.pre_roi_pros = nn.Sequential(
        nn.Conv2d(
            72, 18, kernel_size=1),# (N, 256, H, W)->(N, 72, H, W)
        )
        self.window_size = window_size
        window_size = window_size[0]*window_size[1]
        self.mha = MultiHeadSelfAttentionClassifier(
                            embed_dim=window_size, 
                            num_heads=self.window_size[0],
                            hidden_dim=window_size*18,
                            num_classes=3)
        self.roi_fv_loss = nn.CrossEntropyLoss()
        # self.sigmoid = nn.Sigmoid()

    def gen_sample_grid(self, box, window_size=(7,4), grid_offsets=(0, 0), spatial_scale=2.5):
        box_es = self.gen_box(box)
        N = box.shape[0]
        win = window_size[0] * window_size[1]
        xg, yg, wg, lg, rg = torch.split(box_es, 1, dim=-1)
        print('xg', xg.shape)
        zg, hg = box[:,2].unsqueeze(1), box[:,5].unsqueeze(1)

        xg = xg.unsqueeze(-1).expand(N, *window_size) # 维度扩展为 [N,window_size]
        yg = yg.unsqueeze(-1).expand(N, *window_size) # 维度扩展为 [N,window_size]
        zg = zg.unsqueeze(-1).expand(N, *window_size) # 维度扩展为 [N,window_size]
        rg = rg.unsqueeze(-1).expand(N, *window_size) # 维度扩展为 [N,window_size]

        cosTheta = torch.cos(rg)
        sinTheta = torch.sin(rg)

        xx = torch.linspace(-.5, .5, window_size[0]).type_as(box_es).view(1, -1) * wg
        yy = torch.linspace(-.5, .5,  window_size[1]).type_as(box_es).view(1, -1) * lg
        zz = torch.linspace(-.5, .5,  window_size[1]).type_as(box_es).view(1, -1) * hg

        xx = xx.unsqueeze_(-1).expand(N, *window_size)
        yy = yy.unsqueeze_(1).expand(N, *window_size)
        zz = zz.unsqueeze_(1).expand(N, *window_size)

        x=(xx * cosTheta + yy * sinTheta + xg)
        # y=(yy * cosTheta - xx * sinTheta + yg)
        z=(zz + zg)


        x = (x.permute(1, 2, 0).contiguous() + grid_offsets[0]) * spatial_scale
        z = (z.permute(1, 2, 0).contiguous() + grid_offsets[1]) * spatial_scale

        return x.view(win, -1), z.view(win, -1)

    def bi_roi_gridsample(self, image, gt_cls, box, window_size=(5, 5), spatial_scale=2.5, is_train=True):
        bbox = self.gen_box(box)# 将box转为角点
        sampled_features = torch.zeros((1,18,*window_size)).cuda()
        sampled_gts = torch.zeros((1,6,*window_size)).cuda()
        for i ,coord in enumerate(bbox):
            grid_z = torch.linspace(coord[0].item(), coord[2].item(), window_size[0]).cuda()
            grid_x = torch.linspace(coord[1].item(), coord[3].item(), window_size[1]).cuda()
            grid_z, grid_x = torch.meshgrid((grid_z-1)/4, (grid_x-35.2)/70.4)# 规范化后融合
            # 将 grid_x 和 grid_y 调整为 (B, H, W, 2) 形状
            grid_x = grid_x.unsqueeze(0).unsqueeze(-1)
            grid_z = grid_z.unsqueeze(0).unsqueeze(-1)
            # 生成roi align的grid
            grid = torch.cat((grid_z, grid_x), dim=-1)
            # 对fv的预测进行roi提取
            sampled_pred = torch.nn.functional.grid_sample(image, grid, align_corners=True,mode='nearest')
            sampled_features = torch.cat([sampled_features,sampled_pred], dim=0)
            if is_train:
                # 对fv的gt进行roi提取
                sampled_gt = torch.nn.functional.grid_sample(gt_cls, grid, align_corners=True,mode='nearest')
                sampled_gts = torch.cat([sampled_gts,sampled_gt], dim=0)
        # 将sampled_features转为[N,18,window_size]
        sampled_features = sampled_features[1:,...].view(-1, 18, window_size[0]*window_size[1])
        # sampled_gts[N,6,window_size]
        
        if is_train:
            '''
            sampled_gts = sampled_gts[1:,...].view(-1, 6, window_size[0]*window_size[1])
            non_zero_mask = sampled_features != 0# 得到sampled_features中不为0的mask
            non_zero_mask_sum = torch.any(non_zero_mask!=False, dim=2)# 得到sampled_features中最后一维不为0的mask
            non_zero_roi_mask =  torch.any(non_zero_mask_sum!=False, dim=1)# 得到sampled_features中不为0的roi
            non_zero_mask_sum = non_zero_mask_sum[non_zero_roi_mask]
            sampled_features = sampled_features[non_zero_roi_mask]# 得到sampled_features中不为0的roi的特征
            sampled_gts = sampled_gts[non_zero_roi_mask]# 得到sampled_features中不为0的roi对应的gt
            non_zero_mask = non_zero_mask[non_zero_roi_mask]# 得到sampled_features中不为0的roi的mask
            sum_of_non_zero = torch.sum(sampled_features * non_zero_mask, dim=2) /\
                    torch.sum(non_zero_mask, dim=2, dtype=torch.float)# 将筛选后的特征不为0的部分求取均值，得到[N，18]
            sum_of_non_zero = sum_of_non_zero[non_zero_mask_sum]
            sampled_gts = torch.mean(sampled_gts, dim=2).view(-1)
            # 创建 Smooth L1 损失函数
            # smooth_l1_loss = self.smooth_l1(sum_of_non_zero, sampled_gts)
            one_hot_targets = torch.zeros(
                *list(sampled_gts.shape), 4
            ).cuda()
            one_hot_targets = one_hot_targets.scatter_(-1, sampled_gts.unsqueeze(dim=-1).long(), 1.0).view(-1)# 转换为独热编码
            # 对标签进行独热编码后进行
            one_hot_targets = one_hot_targets[..., 1:]# 除去dontcare的标签
            # sampled_gts = sampled_gts[1:,...].view(-1)
            # sum_of_non_zero = sampled_features.view(-1)
            # one_hot_targets = torch.zeros(*list(sampled_gts.shape), 3).cuda()
            # one_hot_targets = one_hot_targets.scatter_(-1, sampled_gts.unsqueeze(dim=-1).long(), 1.0).view(-1)# 转换为独热编码
            smooth_l1_loss = self.smooth_l1(sum_of_non_zero, one_hot_targets)
            return smooth_l1_loss
            '''

            sum_of_non_zero = sampled_features.permute(0,2,1).contiguous()
            sum_of_non_zero = sum_of_non_zero.view(-1, 3)
            # print('sum_of_non_zero', sum_of_non_zero.shape[0])
            sum_of_non_zero = torch.cat((torch.zeros(sum_of_non_zero.shape[0], 1).cuda(), sum_of_non_zero), dim=-1)
            # if torch.any(sum_of_non_zero!=0):
            #     print("yes")
            cared = sampled_gts>=0
            sampled_gts = sampled_gts * cared# 进行gt筛选
            sampled_gts = sampled_gts.view(-1, 6, window_size[0]*window_size[1]).permute(0,2,1).contiguous()
            sampled_gts = sampled_gts[1:,...].view(-1)# [N,6,5,5]
            sampled_gts = sampled_gts.to(torch.int64)
            # print('max', torch.max(sampled_gts))
            # sampled_gts = sampled_gts(-1, 6*window_size[0]*window_size[1])
            # one_hot_targets = torch.zeros(*list(sampled_gts.shape), 4).cuda()# [N, 6*25*25, 3+1]

            # one_hot_targets = one_hot_targets.scatter_(-1, sampled_gts.unsqueeze(dim=-1).long(), 1.0)# 转换为独热编码
            # one_hot_targets = one_hot_targets[...,1:].contiguous().view(-1)
            roi_fv_loss = self.roi_fv_loss(sum_of_non_zero, sampled_gts)
            return roi_fv_loss

        else:
            
            sampled_features = sampled_features.view(-1, window_size[0]*window_size[1]*6, 3)# 转为[N, 150, 3]
            # sampled_features_mask = sampled_features<0.1
            # sampled_features = sampled_features*sampled_features_mask*(-1) + sampled_features
            sampled_features = torch.sigmoid(sampled_features)# 将sampled_features转到0-1之间

            # sampled_features = torch.any(sampled_features>0.5, dim=2)
            sampled_scores = torch.argmax(sampled_features, dim=2)# ****对候选框提取特征进行求极大得到最终的预测结果
            # sampled_labels0 = torch.sum(sampled_labels==0,dim=-1).unsqueeze(-1)
            # sampled_labels1 = torch.sum(sampled_labels==1,dim=-1).unsqueeze(-1)
            # sampled_labels2 = torch.sum(sampled_labels==1,dim=-1).unsqueeze(-1)
            # sampled_features = torch.cat((sampled_labels0,sampled_labels1,sampled_labels2),dim=-1)
            # sampled_scores = torch.sum(sampled_scores, dim=-1)
            sampled_labels0 = torch.sum(sampled_scores==0,dim=-1).unsqueeze(-1)
            sampled_labels1 = torch.sum(sampled_scores==1,dim=-1).unsqueeze(-1)
            sampled_labels2 = torch.sum(sampled_scores==2,dim=-1).unsqueeze(-1)
            sampled_indices = torch.cat((sampled_labels0,sampled_labels1,sampled_labels2),dim=-1)
            _,sampled_indices = torch.max(sampled_indices,dim=-1)

            # sampled_features = torch.sum(sampled_features, dim=1)# 求和统计
            # sampled_features_indices = torch.argmax(sampled_features,dim=1)# 对onehot进行解码
            # non_zero_mask = sampled_features > 0.1*150
            # non_zero_mask_sum = torch.any(non_zero_mask!=False, dim=2)# [N,150]
            # sampled_features = sampled_features*non_zero_mask

            # non_zero_roi_mask =  torch.any(non_zero_mask!=False, dim=2)
            # non_zero_mask_sum = non_zero_mask_sum[non_zero_roi_mask]
            # sampled_features = sampled_features[non_zero_roi_mask]
            # non_zero_mask = non_zero_mask[non_zero_roi_mask]
            # sum_of_non_zero = torch.sum(sampled_features * non_zero_mask, dim=2)
            # sum_of_non_zero = sum_of_non_zero[non_zero_mask_sum] 
            # sampled_features = sampled_features.view(-1, 6, 3, window_size[0]*window_size[1])
            # sampled_features = torch.sigmoid(sampled_features)
            return sampled_indices
            # return torch.max(torch.sigmoid(sum_of_non_zero).view(-1,18), dim=-1)
        
    # 用于生成指定的box
    def gen_box(self, predict_loc):
        x0,y0,z0, x,y,z, theta = torch.split(predict_loc, 1, dim=1)
        cosTheta = torch.cos(theta)
        sinTheta = torch.sin(theta)
        x1 = x0-(y*cosTheta+x*sinTheta)/2
        x2 = x0+(y*cosTheta+x*sinTheta)/2
        z1 = z0-z/2
        z2 = z0+z/2
        return torch.cat([z1, x1, z2, x2], dim=1)
        

    def forward(self, rois, data_dict, forward_dict, is_train=True):
        # spatial_features_2d_fv = forward_dict['cls_preds_fv'].permute(0,3,1,2)
        spatial_features_2d_fv = data_dict['spatial_features_2d_fv']
        batch_size = data_dict['batch_size']
        spatial_features_2d_fv = self.pre_roi_pros(spatial_features_2d_fv)
        # gt信息在data_dict['gt_boxes']中,格式为[N,8],第二维最后一列为分类
        gt_class_fv = forward_dict['box_cls_labels_fv'].view(batch_size,6,11,176).unsqueeze(1).float()
        batch_l1_loss = torch.tensor(.0).cuda()
        sampled_features_indices_list = []
        for index in range(batch_size):
            N, _ = rois[index].shape
            gt_class_batch = gt_class_fv[index,:]
            # print('gt_class_batch',gt_class_batch.shape)
            # features = spatial_features_2d_fv[index,...]
            # print('rois[index]', rois[index].shape)
            # print('spatial_features_2d_fv[index]', spatial_features_2d_fv[index].shape)
            if is_train:
                smooth_l1_loss = self.bi_roi_gridsample(
                                spatial_features_2d_fv[index:index+1], 
                                gt_class_batch, 
                                rois[index],
                                is_train=is_train)
                batch_l1_loss+=smooth_l1_loss
            else:
                sampled_features_indices = self.bi_roi_gridsample(
                                spatial_features_2d_fv[index:index+1], 
                                gt_class_batch, 
                                rois[index]
                                ,is_train=is_train)
                sampled_features_indices_list.append(sampled_features_indices)

            # print(self.forward_ret_dict['box_preds'][index].shape)
            # a = a.squeeze(1).squeeze(2).permute(1,0)
            # a = torch.mean(a, dim=0)

            # print('a', a.shape)
            # a = self.roi_linears(a)# [N,8]
            # iou = iou3d_nms_utils.boxes_iou3d_gpu(data_dict['gt_boxes'][index,:,:7], a[:,:7])
            # output = F.roi_pool(input_features, rois, output_size=(200, 176))
            # roi_list = [row for row in rois[index][:]]
            # roi_out = self.roi_pool(x, roi_list)
            # x = self.conv(roi_out)
            
        # return x
        if is_train:
            return batch_l1_loss
        else:
            return sampled_features_indices_list

class ROIAlignSpatialAttention_modi1(nn.Module):
    def __init__(self, window_size=(5,5), channels=18):
        super(ROIAlignSpatialAttention_modi1, self).__init__()
        self.window_size = window_size
        self.channels=channels
        # self.roi_pool = RoIPool((7,7), spatial_scale=1.0)
        self.pre_roi_pros_fv = nn.Sequential(
        nn.Conv2d(
            128, 18, kernel_size=3, padding=1),# (N, 128, H, W)->(N, 18, H, W)
            nn.BatchNorm2d(18, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                18, 18, kernel_size=1),# (N, 256, H, W)->(N, 18, H, W)
        )
        self.pre_roi_pros_bev = nn.Sequential(
            nn.Conv2d(
                512, 18, kernel_size=3, padding=1),# (N, 256, H, W)->(N, 18, H, W)
            nn.BatchNorm2d(18, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                18, 18, kernel_size=1),# (N, 256, H, W)->(N, 18, H, W)
        )
        # self.fv_conv = nn.Conv2d(18, 18, kernel_size=1),# (N, 256, H, W)->(N, 18, H, W)
        # self.bev_conv = nn.Conv2d(18, 18, kernel_size=1),# (N, 256, H, W)->(N, 18, H, W)
        window_size = window_size[0]*window_size[1]
        self.MHA = MultiHeadSelfAttentionClassifier(
                            embed_dim=window_size, 
                            num_heads=self.window_size[0],
                            hidden_dim=window_size*18,
                            num_classes=3)
        self.roi_fv_loss = nn.CrossEntropyLoss()

    # 用于生成指定的box
    def gen_box(self, predict_loc):
        x0,y0,z0, x,y,z, theta = torch.split(predict_loc, 1, dim=1)
        cosTheta = torch.cos(theta)
        sinTheta = torch.sin(theta)
        x1 = x0-(y*cosTheta+x*sinTheta)/2
        x2 = x0+(y*cosTheta+x*sinTheta)/2
        y1=y0 - (y*sinTheta+x*cosTheta)/2
        y2=y0 + (y*sinTheta+x*cosTheta)/2
        z1 = z0-z/2
        z2 = z0+z/2
        return torch.cat([z1, x1, z2, x2], dim=1), torch.cat([y1, x1, y2, x2], dim=1)

    def gen_bi_roi_grids(self, bbox, window_size=(5, 5), is_bev=False, is_train=True):
        grids = torch.zeros((1, *window_size, 2)).cuda()
        for i ,coord in enumerate(bbox):
            grid_z = torch.linspace(coord[0].item(), coord[2].item(), window_size[0]).cuda()
            grid_x = torch.linspace(coord[1].item(), coord[3].item(), window_size[1]).cuda()
            if is_bev:
                grid_z, grid_x = torch.meshgrid(grid_z/40, (grid_x-35.2)/35.2)# 规范化后融合
            else:
                grid_z, grid_x = torch.meshgrid((grid_z+1)/2, (grid_x-35.2)/35.2)# 规范化后融合
            # 将 grid_x 和 grid_y 调整为 (B, H, W, 2) 形状
            grid_x = grid_x.unsqueeze(0).unsqueeze(-1)
            grid_z = grid_z.unsqueeze(0).unsqueeze(-1)
            # 生成roi align的grid
            grid = torch.cat((grid_z, grid_x), dim=-1)
            # 对fv的预测进行roi提取
            grids = torch.cat((grids, grid), dim=0)
        return grids[1:,...]
        
    def gen_sampled_features(self, grids, feature, channels, window_size=(5,5)):
        sampled_features = torch.zeros((1, channels, *window_size)).cuda()
        for grid in grids:
            # 对feature的预测进行roi提取
            sampled_feature = torch.nn.functional.grid_sample(feature, grid.unsqueeze(0), align_corners=True, mode='nearest')
            sampled_features = torch.cat([sampled_features, sampled_feature], dim=0)
        sampled_features = sampled_features[1:,...].view(-1, channels, window_size[0]*window_size[1])
        return sampled_features

    def get_roi_results(self, features_fv, features_bev, roi, gt=None, is_train=True):
        bbox_fv, bbox_bev = self.gen_box(roi)# 生成fv平面的roi角点box
        grids_fv = self.gen_bi_roi_grids(bbox_fv, is_train=is_train)
        grids_bev = self.gen_bi_roi_grids(bbox_bev, is_bev=True, is_train=is_train)
        sampled_features_fv = self.gen_sampled_features(grids_fv, features_fv, channels=self.channels)
        sampled_features_bev = self.gen_sampled_features(grids_bev, features_bev, channels=self.channels)
        # sampled_features_fv = self.fv_conv(sampled_features_fv)
        # sampled_features_bev = self.bev_conv(sampled_features_bev)
        mha = self.MHA(sampled_features_bev, sampled_features_fv)# [n, cls+1]
        # mha_max = torch.argmax(mha, dim=1)# [n, cls+1]
        if is_train:
            # 借鉴faster_rcnn生成gt_target的操作
            iou = iou3d_nms_utils.boxes_iou3d_gpu(roi, gt[:,:-1])
            iou_max, iou_max_indices = torch.max(iou, dim=1)
            '''
            #生成roi的iou掩膜
            roi_argmax_iou = torch.argmax(iou, dim=1)# [num_roi]
            # 生成gt的iou掩膜
            gt_argmax_iou = torch.argmax(iou, dim=0)# [num_gt]
            # 求得roi的ioumax
            roi_max_iou, roi_max_iou_indices  = torch.max(iou, dim=1)
            #对每个roi布置最匹配的gt
            for i in range(gt_argmax_iou.shape[0]):
                roi_argmax_iou[gt_argmax_iou[i]] = torch.tensor(i)
            
            # 筛选阈值大于0.3的roi
            roi_label = torch.ones((roi.shape[0],1), dtype=torch.int32)*-1
            roi_label = roi_label.cuda()
            roi_label[roi_max_iou<0.3] = 0
            roi_label[roi_max_iou>=0.3] = 1
            if gt_argmax_iou.shape[0]>0:
                roi_label[gt_argmax_iou] = 1
            '''
            gt_assigned = gt[iou_max_indices, -1]-1
            if torch.any(iou_max<0.1):
                gt_assigned *= iou_max>0.1
            # for i, indice in enumerate(iou_thres_indices):
            #     gt_assigned = torch.cat((gt_assigned, gt[indice, -1]), dim=0)
            # gt_assigned = 
            # one_hot_targets = torch.zeros(
            #     *list(gt_assigned.shape), 3+1
            # ).cuda()
            # one_hot_targets = one_hot_targets.scatter_(-1, gt_assigned.unsqueeze(dim=-1).long(), 1.0)# 转换为独热编码
            # one_hot_targets = one_hot_targets[..., 1:]# 除去dontcare的标签
            roi_fv_loss = self.roi_fv_loss(mha, gt_assigned.to(torch.int64))/mha.shape[0]*20
            return roi_fv_loss
        else:
            # print('mha', mha.shape)
            return torch.argmax(mha, dim=1)+1, torch.softmax(mha, dim=1)


    def forward(self, rois, data_dict, forward_dict, is_train=True):
        spatial_features_2d_fv = data_dict['spatial_features_2d_fv_extra']
        spatial_features_2d_bev = data_dict['spatial_features_2d']
        # spatial_features_2d_bev = forward_dict['cls_preds'].permute(0,3,1,2)
        
        batch_size = data_dict['batch_size']
        spatial_features_2d_fv = self.pre_roi_pros_fv(spatial_features_2d_fv)
        spatial_features_2d_bev = self.pre_roi_pros_bev(spatial_features_2d_bev)
        if is_train:
            # gt信息在data_dict['gt_boxes']中,格式为[N,8],第二维最后一列为分类
            gt_class_fv = data_dict['gt_boxes']
            batch_loss = torch.tensor(.0).cuda()
            for index in range(batch_size):
                N, L = rois[index].shape
                gt_class_batch = gt_class_fv[index, ...]
                # bbox = self.gen_box_fv(rois[index])# 将box转为角点
                _loss = self.get_roi_results(
                                spatial_features_2d_fv[index:index+1],
                                spatial_features_2d_bev[index:index+1],
                                rois[index],
                                gt_class_batch,
                                is_train=is_train)
                batch_loss+=_loss

            return batch_loss
        else:
            pres_list = []
            scores_list = []
            for index in range(batch_size):
                cls_pred, box_scores = self.get_roi_results(
                                spatial_features_2d_fv[index:index+1],
                                spatial_features_2d_bev[index:index+1],
                                rois[index],
                                is_train=is_train)
                pres_list.append(cls_pred)
                scores_list.append(box_scores)# box_scores太低
            return pres_list, scores_list

# 定义多头自注意力模块
class MultiHeadSelfAttention(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim] 
        key --- [N, T_k, key_dim]
        mask --- [N, T_k]
    output:
        out --- [N, T_q, num_units]
        scores -- [h, N, T_q, T_k]
    '''
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, feat1, feat2):
        # queries，keys，values的形状:
        # (batch_size，查询或者“键－值”对的个数，num_hiddens)
        # valid_lens　的形状:
        # (batch_size，)或(batch_size，查询的个数)
        # 经过变换后，输出的queries，keys，values　的形状:
        # (batch_size*num_heads，查询或者“键－值”对的个数，
        # num_hiddens/num_heads)
        assert feat1.shape == feat2.shape
        batch_size, seq_len, embed_dim = feat1.size()

        # 将输入向量拆分为多个头
        q = self.query(feat2).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(feat1).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(feat2).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力权重，点乘后除权
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float))
        attn_weights = torch.softmax(attn_weights, dim=-1)

        # 注意力加权求和
        attended_values = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        # 经过线性变换和残差连接
        feat1 = self.fc(attended_values) + feat1

        return feat1

class MultiHeadSelfAttentionClassifier(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, num_classes):
        super(MultiHeadSelfAttentionClassifier, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc2 = nn.Linear(hidden_dim//2, num_classes)
        self.conv = nn.Conv2d(36, 18, kernel_size=3, padding=1)

    def forward(self, bev, fv):
        x1 = self.attention(bev, fv).view(-1, 18, 5, 5)
        x2 = self.attention(fv, bev).view(-1, 18, 5, 5)
        x = torch.cat((x1, x2), dim=1)
        x = self.conv(x).view(-1, self.hidden_dim)
        # x = x.mean(dim=1)  # 对每个位置的向量求平均
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x