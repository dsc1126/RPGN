import torch
import torch.nn as nn
import time
import numpy as np
import random
import os
import sys

from util.config import cfg
cfg.task = 'test'
from util.log import logger
import util.utils as utils
import util.eval as eval

from tensorboardX import SummaryWriter
# summary writer
global writer
writer = SummaryWriter(cfg.exp_path)

def init():
    global result_dir
    result_dir = os.path.join(cfg.exp_path, 'result', 'epoch{}_nmst{}_scoret{}_npointt{}'.format(cfg.test_epoch, cfg.TEST_NMS_THRESH, cfg.TEST_SCORE_THRESH, cfg.TEST_NPOINT_THRESH), cfg.split)
    backup_dir = os.path.join(result_dir, 'backup_files')
    os.makedirs(backup_dir, exist_ok=True)
    os.makedirs(os.path.join(result_dir, 'predicted_masks'), exist_ok=True)
    os.system('cp test_regional.py {}'.format(backup_dir))
    os.system('cp {} {}'.format(cfg.model_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.dataset_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.config, backup_dir))

    global semantic_label_idx
    semantic_label_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
    # ----------------------------------------------------------------------------
    CLASS_LABELS = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
    UNKNOWN_ID = -100
    N_CLASSES = len(CLASS_LABELS)
    # ----------------------------------------------------------------------------
    
    # logger.info(cfg)

    random.seed(cfg.test_seed)
    np.random.seed(cfg.test_seed)
    torch.manual_seed(cfg.test_seed)
    torch.cuda.manual_seed_all(cfg.test_seed)

# ----------------------------------------------------------------------------
def confusion_matrix(pred_ids, gt_ids):
    assert pred_ids.shape == gt_ids.shape, (pred_ids.shape, gt_ids.shape)
    idxs= gt_ids>=0
    return np.bincount(pred_ids[idxs]*20+gt_ids[idxs],minlength=400).reshape((20,20)).astype(np.ulonglong)

def get_iou(label_id, confusion):
    # true positives
    tp = np.longlong(confusion[label_id, label_id])
    # false positives
    fp = np.longlong(confusion[label_id, :].sum()) - tp
    # false negatives
    fn = np.longlong(confusion[:, label_id].sum()) - tp

    denom = (tp + fp + fn)
    if denom == 0:
        return float('nan')
    return (float(tp) / denom, tp, denom)

def get_tp_fp_fn(label_id, confusion):
    # true positives
    tp = np.longlong(confusion[label_id, label_id])
    # false positives
    fp = np.longlong(confusion[label_id, :].sum()) - tp
    # false negatives
    fn = np.longlong(confusion[:, label_id].sum()) - tp

    # denom = (tp + fp + fn)
    # if denom == 0:
    #     return float('nan')
    return (tp, fp, fn)

def evaluate(pred_ids,gt_ids,N_CLASSES):
    print('evaluating', gt_ids.size, 'points...')
    confusion=confusion_matrix(pred_ids,gt_ids)
    class_ious = {}
    class_tp_fp_fn = {}
    # class_fp = {}
    # class_fn = {}
    mean_iou = 0
    iou_0 = 0
    iou_1 = 0
    iou_2 = 0
    iou_3 = 0
    iou_4 = 0
    iou_5 = 0

    if N_CLASSES == 20:
        CLASS_LABELS = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
        #print("N_CLASSES is 20")
    elif N_CLASSES == 3:
        CLASS_LABELS = ['0', '1', '2']
        #print("N_CLASSES is 3")
    elif N_CLASSES == 2:
        CLASS_LABELS = ['0', '1']
    elif N_CLASSES == 4:
        CLASS_LABELS = ['0', '1', '2', '3'] #, '4']
    elif N_CLASSES == 6:
        CLASS_LABELS = ['0', '1', '2', '3', '4', '5']

    for i in range(N_CLASSES):
        label_name = CLASS_LABELS[i]
        class_ious[label_name] = get_iou(i, confusion)
        class_tp_fp_fn[label_name] = get_tp_fp_fn(i, confusion)
        class_isnan = np.isnan(class_ious[label_name])

        if class_isnan.any() != True:
            mean_iou+=class_ious[label_name][0]/N_CLASSES #20
            # Return iou for individual class
            if N_CLASSES == 3:
                if i == 0:
                    iou_0 = class_ious[label_name][0]
                elif i == 1:
                    iou_1 = class_ious[label_name][0]
                elif i == 2:
                    iou_2 = class_ious[label_name][0]
            if N_CLASSES == 6:
                if i == 0:
                    iou_0 = class_ious[label_name][0]
                elif i == 1:
                    iou_1 = class_ious[label_name][0]
                elif i == 2:
                    iou_2 = class_ious[label_name][0]
                elif i == 3:
                    iou_3 = class_ious[label_name][0]
                elif i == 4:
                    iou_4 = class_ious[label_name][0]
                elif i == 5:
                    iou_5 = class_ious[label_name][0]

        else:
            print("empty class id:", i)


    print('classes          IoU')
    print('----------------------------')
    for i in range(N_CLASSES):
        label_name = CLASS_LABELS[i]
        class_ious[label_name] = get_iou(i, confusion)
        class_isnan = np.isnan(class_ious[label_name])
        if class_isnan.any() != True: #if class_isnan is not True:            
            print('{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d})'.format(label_name, class_ious[label_name][0], class_ious[label_name][1], class_ious[label_name][2]))
    print('mean IOU', mean_iou)

    return mean_iou, iou_0, iou_1, iou_2, iou_3, iou_4, iou_5, class_tp_fp_fn
# ----------------------------------------------------------------------------

def test(model, model_fn, data_name, epoch):
    print(" ")
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')

    if cfg.dataset == 'scannetv2':
        if data_name == 'scannet':
            from data.scannetv2_inst_regional_yaml1 import Dataset
            dataset = Dataset(test=True)
            dataset.testLoader()
        else:
            print("Error: no data loader - " + data_name)
            exit(0)
    dataloader = dataset.test_data_loader

    # ######### Added for val set ####################
    # import data.scannetv2_inst_regional
    # dataset = data.scannetv2_inst_regional.Dataset()
    # dataset.valLoader()
    # dataloader = dataset.val_data_loader
    # ################################################

    with torch.no_grad():
        model = model.eval()
        start = time.time()

        matches = {}

        total_time = 0
        inference_time = 0

        for i, batch in enumerate(dataloader):
            N = batch['feats'].shape[0]
            test_scene_name = dataset.test_file_names[int(batch['id'][0])].split('/')[-1][:12] #[:13]

            start = time.time()

            start1 = time.time()
            preds = model_fn(batch, model, epoch)
            end1 = time.time() - start1

            ##### get predictions (#1 semantic_pred, pt_offsets; #2 scores, proposals_pred)
            semantic_scores = preds['semantic']  # (N, nClass=20) float32, cuda
            semantic_pred = semantic_scores.max(1)[1]  # (N) long, cuda

            regional_scores2 = preds['regional2']  # (N, nClass=20) float32, cuda
            regional_pred2 = regional_scores2.max(1)[1]  # (N) long, cuda

            pt_offsets = preds['pt_offsets']    # (N, 3), float32, cuda

            if (epoch > cfg.prepare_epochs):
                scores = preds['score']   # (nProposal, 1) float, cuda
                scores_pred = torch.sigmoid(scores.view(-1))

                proposals_idx, proposals_offset = preds['proposals']
                # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
                # proposals_offset: (nProposal + 1), int, cpu
                # print("proposals_offset:", proposals_offset)
                proposals_pred = torch.zeros((proposals_offset.shape[0] - 1, N), dtype=torch.int, device=scores_pred.device) # (nProposal, N), int, cuda
                proposals_pred[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1

                semantic_id = torch.tensor(semantic_label_idx, device=scores_pred.device)[semantic_pred[proposals_idx[:, 1][proposals_offset[:-1].long()].long()]] # (nProposal), long

                # ##### score threshold
                # score_mask = (scores_pred > cfg.TEST_SCORE_THRESH)
                # scores_pred = scores_pred[score_mask]
                # proposals_pred = proposals_pred[score_mask]
                # semantic_id = semantic_id[score_mask]

                ##### npoint threshold
                proposals_pointnum = proposals_pred.sum(1)
                npoint_mask = (proposals_pointnum > cfg.TEST_NPOINT_THRESH)
                scores_pred = scores_pred[npoint_mask]
                proposals_pred = proposals_pred[npoint_mask]
                semantic_id = semantic_id[npoint_mask]

                # ##### nms
                # if semantic_id.shape[0] == 0:
                #     pick_idxs = np.empty(0)
                # else:
                #     proposals_pred_f = proposals_pred.float()  # (nProposal, N), float, cuda
                #     intersection = torch.mm(proposals_pred_f, proposals_pred_f.t())  # (nProposal, nProposal), float, cuda
                #     proposals_pointnum = proposals_pred_f.sum(1)  # (nProposal), float, cuda
                #     proposals_pn_h = proposals_pointnum.unsqueeze(-1).repeat(1, proposals_pointnum.shape[0])
                #     proposals_pn_v = proposals_pointnum.unsqueeze(0).repeat(proposals_pointnum.shape[0], 1)
                #     cross_ious = intersection / (proposals_pn_h + proposals_pn_v - intersection)
                #     pick_idxs = non_max_suppression(cross_ious.cpu().numpy(), scores_pred.cpu().numpy(), cfg.TEST_NMS_THRESH)  # int, (nCluster, N)
                # clusters = proposals_pred[pick_idxs]
                # cluster_scores = scores_pred[pick_idxs]
                # cluster_semantic_id = semantic_id[pick_idxs]

                
                # #### use semantic to score
                # scores_pred = torch.zeros(proposals_pred.shape[0])
                # for i in range(proposals_pred.shape[0]):
                #     sem_pred = semantic_scores[proposals_pred[i]==1]
                #     sem_argmax = semantic_pred[proposals_pred[i]==1]
                #     # print("sem_pred:", sem_pred.shape, sem_pred)
                #     m = nn.Softmax(dim=1)
                #     sem_pred_softmax = m(sem_pred)
                #     # print("sem_pred_softmax:", sem_pred_softmax)
                #     sem_score = sem_pred_softmax[:,sem_argmax].mean().cpu().numpy()
                #     # print("sem_score:", sem_score)
                #     scores_pred[i] = torch.from_numpy(sem_score)

                #### no nms, direct output
                clusters = proposals_pred
                cluster_scores = scores_pred
                cluster_semantic_id = semantic_id
                
                nclusters = clusters.shape[0]

            end = time.time() - start
            start = time.time()

            logger.info("evaluate scene: {}/{}, point number: {}, processing time: {:.2f}s ".format(batch['id'][0] + 1, len(dataset.test_files), N, end))
            total_time += end
            inference_time += end1

            ##### prepare for evaluation
            if cfg.eval:
                pred_info = {}
                pred_info['conf'] = cluster_scores.cpu().numpy()
                pred_info['label_id'] = cluster_semantic_id.cpu().numpy()
                pred_info['mask'] = clusters.cpu().numpy()
                gt_file = os.path.join(cfg.data_root, cfg.dataset, cfg.split + '_gt', test_scene_name + '.txt')
                gt2pred, pred2gt = eval.assign_instances_for_scan(test_scene_name, pred_info, gt_file)
                matches[test_scene_name] = {}
                matches[test_scene_name]['gt'] = gt2pred
                matches[test_scene_name]['pred'] = pred2gt

        ##### evaluation
        if cfg.eval:
            ap_scores = eval.evaluate_matches(matches)
            avgs = eval.compute_averages(ap_scores)
            eval.print_results(avgs)

        print("total processing time:", total_time, " s")

def non_max_suppression(ious, scores, threshold):
    ixs = scores.argsort()[::-1]
    pick = []
    while len(ixs) > 0:
        i = ixs[0]
        pick.append(i)
        iou = ious[i, ixs[1:]]
        remove_ixs = np.where(iou > threshold)[0] + 1
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


if __name__ == '__main__':
    init()

    ##### get model version and data version
    exp_name = cfg.config.split('/')[-1][:-5]
    model_name = exp_name.split('_')[0]
    data_name = exp_name.split('_')[-1]

    ##### model
    # logger.info('=> creating model ...')
    # logger.info('Classes: {}'.format(cfg.classes))

    if model_name == 'rpgn':
        from model.rpgn.rpgn_regional_yaml1 import RPGN as Network
        from model.rpgn.rpgn_regional_yaml1 import model_fn_decorator
    else:
        print("Error: no model version " + model_name)
        exit(0)
    model = Network(cfg)

    use_cuda = torch.cuda.is_available()
    # logger.info('cuda available: {}'.format(use_cuda))
    assert use_cuda
    model = model.cuda()

    # logger.info(model)
    # logger.info('#classifier parameters (model): {}'.format(sum([x.nelement() for x in model.parameters()])))

    ##### model_fn (criterion)
    model_fn = model_fn_decorator(test=True)

    ##### load model
    utils.checkpoint_restore(model, cfg.exp_path, cfg.config.split('/')[-1][:-5], use_cuda, cfg.test_epoch, dist=False, f=cfg.pretrain)      # resume from the latest epoch, or specify the epoch to restore

    ##### evaluate
    test(model, model_fn, data_name, cfg.test_epoch)