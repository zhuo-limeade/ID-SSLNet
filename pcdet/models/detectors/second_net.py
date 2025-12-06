from .detector3d_template import Detector3DTemplate
import time

class SECONDNet(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        # time1 = time.time()
        
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
        # batch_dict['time0'] = time.time() - time1

        if self.training:

            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            # time1 = time.time()
            
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            # batch_dict['time1'] = time.time() - time1
            return pred_dicts, recall_dicts
            # return batch_dict['pred_dicts'], batch_dict['recall_dict']

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict

class SECONDNet_FV(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:

            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            # pred_dicts, recall_dicts = self.post_processing(batch_dict)
            recall_dict = {}
            pred_dicts = []
            # print('len', len(batch_dict['preds_box']))
            for index, preds_box in enumerate(batch_dict['preds_box']):
                # print('preds_box',preds_box.shape)
                recall_dict = self.generate_recall_record(
                    box_preds=preds_box,
                    recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                    thresh_list=self.model_cfg.POST_PROCESSING.RECALL_THRESH_LIST
                )
                record_dict = {
                    'pred_boxes': preds_box,
                    'pred_scores': batch_dict['total_scores'][index],
                    'pred_labels': batch_dict['fv_cls_out'][index]
                }
                pred_dicts.append(record_dict)
            return pred_dicts, recall_dict
            # return batch_dict['pred_dicts'], batch_dict['recall_dict']

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss_fv1()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict

