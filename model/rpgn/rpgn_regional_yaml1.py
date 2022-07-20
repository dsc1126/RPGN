import torch
import torch.nn as nn
import spconv
from spconv.modules import SparseModule
import functools
from collections import OrderedDict
import sys
sys.path.append('../../')
import numpy as np

from lib.rpgn_ops.functions import rpgn_ops
from util import utils

import torch.nn.functional as F
import open3d as o3d
from torchviz import make_dot, make_dot_from_trace
import time
import pdb
import os

class ResidualBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(
                nn.Identity()
            )
        else:
            self.i_branch = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False)
            )

        self.conv_branch = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key),
            norm_fn(out_channels),
            nn.ReLU(),
            spconv.SubMConv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        )

    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features, input.indices, input.spatial_shape, input.batch_size)

        output = self.conv_branch(input)
        output.features += self.i_branch(identity).features

        return output


class VGGBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        self.conv_layers = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        )

    def forward(self, input):
        return self.conv_layers(input)

class VGGBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        self.conv_layers = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=5 , padding=2, bias=False, indice_key=indice_key)
        )

    def forward(self, input):
        return self.conv_layers(input)

class UBlock(nn.Module):
    def __init__(self, nPlanes, norm_fn, block_reps, block, indice_key_id=1):

        super().__init__()

        self.nPlanes = nPlanes

        blocks = {'block{}'.format(i): block(nPlanes[0], nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id)) for i in range(block_reps)}
        blocks = OrderedDict(blocks)
        self.blocks = spconv.SparseSequential(blocks)

        if len(nPlanes) > 1:
            self.conv = spconv.SparseSequential(
                norm_fn(nPlanes[0]),
                nn.ReLU(),
                spconv.SparseConv3d(nPlanes[0], nPlanes[1], kernel_size=2, stride=2, bias=False, indice_key='spconv{}'.format(indice_key_id), padding=0)
            )

            self.u = UBlock(nPlanes[1:], norm_fn, block_reps, block, indice_key_id=indice_key_id+1)

            self.deconv = spconv.SparseSequential(
                norm_fn(nPlanes[1]),
                nn.ReLU(),
                spconv.SparseInverseConv3d(nPlanes[1], nPlanes[0], kernel_size=2, bias=False, indice_key='spconv{}'.format(indice_key_id))
            )

            blocks_tail = {}
            for i in range(block_reps):
                blocks_tail['block{}'.format(i)] = block(nPlanes[0] * (2 - i), nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id))
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = spconv.SparseSequential(blocks_tail)

    def forward(self, input):
        output = self.blocks(input)
        identity = spconv.SparseConvTensor(output.features, output.indices, output.spatial_shape, output.batch_size)

        if len(self.nPlanes) > 1:
            output_decoder = self.conv(output)
            output_decoder = self.u(output_decoder)
            output_decoder = self.deconv(output_decoder)

            output.features = torch.cat((identity.features, output_decoder.features), dim=1)

            output = self.blocks_tail(output)

        return output

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        print("BCE:", BCE)
        print("dice_loss:", dice_loss)
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

class RPGN(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        input_c = cfg.input_channel
        m = cfg.m
        classes = cfg.classes
        block_reps = cfg.block_reps
        block_residual = cfg.block_residual

        self.cluster_radius = cfg.cluster_radius
        self.cluster_meanActive = cfg.cluster_meanActive
        self.cluster_shift_meanActive = cfg.cluster_shift_meanActive
        self.cluster_npoint_thre = cfg.cluster_npoint_thre

        self.score_scale = cfg.score_scale
        self.score_fullscale = cfg.score_fullscale
        self.mode = cfg.score_mode

        self.prepare_epochs = cfg.prepare_epochs

        self.pretrain_path = cfg.pretrain_path
        self.pretrain_module = cfg.pretrain_module
        self.fix_module = cfg.fix_module

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        if block_residual:
            block = ResidualBlock
        else:
            block = VGGBlock

        if cfg.use_coords:
            input_c += 3

        #### backbone
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(input_c, m, kernel_size=3, padding=1, bias=False, indice_key='subm1')
        )

        self.unet = UBlock([m, 2*m, 3*m, 4*m, 5*m, 6*m, 7*m], norm_fn, block_reps, block, indice_key_id=1)

        self.output_layer = spconv.SparseSequential(
            norm_fn(m),
            nn.ReLU()
        )

        # ### large kernel to increase field of view
        # self.large_view = VGGBlock2(m,m,norm_fn) #nn.Conv3d(16,16,25) #

        #### semantic segmentation + regional info prediction
        self.classes = classes
        self.regional_size = 3
        self.size_size = 6

        #### semantic
        self.semantic = nn.Sequential(
            nn.Linear(m, m, bias=True),
            norm_fn(m),
            nn.ReLU()
        )
        self.semantic_linear = nn.Linear(m, self.classes, bias=True)

        #### regional
        self.regional = nn.Sequential(
            nn.Linear(m, m, bias=True),
            norm_fn(m),
            nn.ReLU()
        )
        self.regional_linear1 = nn.Linear(m, self.regional_size, bias=True)
        self.regional_linear2 = nn.Linear(m, self.regional_size, bias=True)


        #### size
        self.size = nn.Sequential(
            nn.Linear(m, m, bias=True),
            norm_fn(m),
            nn.ReLU()
        )
        self.size_linear = nn.Linear(m, self.size_size, bias=True)


        #### offset
        self.offset = nn.Sequential(
            nn.Linear(m, m, bias=True),
            norm_fn(m),
            nn.ReLU()
        )
        self.offset_linear = nn.Linear(m, 3, bias=True)


        #### score branch
        self.score_unet = UBlock([m, 2*m], norm_fn, 2, block, indice_key_id=1)
        self.score_outputlayer = spconv.SparseSequential(
            norm_fn(m),
            nn.ReLU()
        )
        self.score_linear = nn.Linear(m, 1)

        self.apply(self.set_bn_init)


        #### fix parameter
        module_map = {'input_conv': self.input_conv, 'unet': self.unet, 'output_layer': self.output_layer,
                      'semantic': self.semantic, 'semantic_linear': self.semantic_linear, 'offset': self.offset, 'offset_linear': self.offset_linear, #'linear': self.linear, 'linear3': self.linear3
                      'regional': self.regional, 'regional_linear1': self.regional_linear1, 'regional_linear2': self.regional_linear2, 'size': self.size, 'size_linear': self.size_linear,
                      'score_unet': self.score_unet, 'score_outputlayer': self.score_outputlayer, 'score_linear': self.score_linear}

        for m in self.fix_module:
            mod = module_map[m]
            for param in mod.parameters():
                param.requires_grad = False

        #### load pretrain weights
        if self.pretrain_path is not None:
            pretrain_dict = torch.load(self.pretrain_path)
            for m in self.pretrain_module:
                print("Load pretrained " + m + ": %d/%d" % utils.load_model_param(module_map[m], pretrain_dict, prefix=m))


    @staticmethod
    def set_bn_init(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)


    def clusters_voxelization(self, clusters_idx, clusters_offset, feats, coords, fullscale, scale, mode):
        '''
        :param clusters_idx: (SumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N, cpu
        :param clusters_offset: (nCluster + 1), int, cpu
        :param feats: (N, C), float, cuda
        :param coords: (N, 3), float, cuda
        :return:
        '''
        c_idxs = clusters_idx[:, 1].cuda()
        clusters_feats = feats[c_idxs.long()]
        clusters_coords = coords[c_idxs.long()]

        clusters_coords_mean = rpgn_ops.sec_mean(clusters_coords, clusters_offset.cuda())  # (nCluster, 3), float

        clusters_coords_mean = torch.index_select(clusters_coords_mean, 0, clusters_idx[:, 0].cuda().long())  # (sumNPoint, 3), float
        clusters_coords -= clusters_coords_mean

        clusters_coords_min = rpgn_ops.sec_min(clusters_coords, clusters_offset.cuda())  # (nCluster, 3), float
        clusters_coords_max = rpgn_ops.sec_max(clusters_coords, clusters_offset.cuda())  # (nCluster, 3), float

        if clusters_idx.shape[0] >0:
            clusters_scale = 1 / ((clusters_coords_max - clusters_coords_min) / fullscale).max(1)[0] - 0.01  # (nCluster), float
            clusters_scale = torch.clamp(clusters_scale, min=None, max=scale)
        else:
            clusters_scale = 0
            print("clusters_idx.shape[0] <= 0")

        min_xyz = clusters_coords_min * clusters_scale.unsqueeze(-1)  # (nCluster, 3), float
        max_xyz = clusters_coords_max * clusters_scale.unsqueeze(-1)

        clusters_scale = torch.index_select(clusters_scale, 0, clusters_idx[:, 0].cuda().long())

        clusters_coords = clusters_coords * clusters_scale.unsqueeze(-1)

        range = max_xyz - min_xyz
        offset = - min_xyz + torch.clamp(fullscale - range - 0.001, min=0) * torch.rand(3).cuda() + torch.clamp(fullscale - range + 0.001, max=0) * torch.rand(3).cuda()
        offset = torch.index_select(offset, 0, clusters_idx[:, 0].cuda().long())
        clusters_coords += offset
        assert clusters_coords.shape.numel() == ((clusters_coords >= 0) * (clusters_coords < fullscale)).sum()

        clusters_coords = clusters_coords.long()
        clusters_coords = torch.cat([clusters_idx[:, 0].view(-1, 1).long(), clusters_coords.cpu()], 1)  # (sumNPoint, 1 + 3)

        out_coords, inp_map, out_map = rpgn_ops.voxelization_idx(clusters_coords, int(clusters_idx[-1, 0]) + 1, mode)
        # output_coords: M * (1 + 3) long
        # input_map: sumNPoint int
        # output_map: M * (maxActive + 1) int

        out_feats = rpgn_ops.voxelization(clusters_feats, out_map.cuda(), mode)  # (M, C), float, cuda

        spatial_shape = [fullscale] * 3
        voxelization_feats = spconv.SparseConvTensor(out_feats, out_coords.int().cuda(), spatial_shape, int(clusters_idx[-1, 0]) + 1)

        return voxelization_feats, inp_map


    def forward(self, input, input_map, coords, batch_idxs, batch_offsets, epoch, batch_id):
        '''
        :param input_map: (N), int, cuda
        :param coords: (N, 3), float, cuda
        :param batch_idxs: (N), int, cuda
        :param batch_offsets: (B + 1), int, cuda
        '''
        ret = {}

        output = self.input_conv(input)
        output = self.unet(output)
        output = self.output_layer(output)

        output_feats = output.features[input_map.long()]
        
        #### semantic
        semantic_scores = self.semantic_linear(output_feats)   # (N, nClass), float
        semantic_preds = semantic_scores.max(1)[1]    # (N), long
        

        #### regional
        regional_scores_feats = self.regional(output_feats)   # (N, nClass), float
        regional_scores = self.regional_linear1(regional_scores_feats)
        regional_scores2 = self.regional_linear2(regional_scores_feats)
        regional_preds = regional_scores.max(1)[1]    # (N), long
        regional_preds2 = regional_scores2.max(1)[1]    # (N), long

        #### size
        size_scores = self.size_linear(output_feats)
        size_preds = size_scores.max(1)[1]    # (N), long

        #### offset
        pt_offsets_feats = self.offset(output_feats)
        pt_offsets = self.offset_linear(pt_offsets_feats)   # (N, 3), float32
        pt_offsets_detached = pt_offsets[:,0:2]#.detach()

        ret['semantic_scores'] = semantic_scores
        ret['regional_scores'] = regional_scores
        ret['regional_scores2'] = regional_scores2
        ret['size_scores'] = size_scores
        ret['pt_offsets'] = pt_offsets

        if(epoch > self.prepare_epochs):
            #### get prooposal clusters
            object_idxs = torch.nonzero(semantic_preds > 1).view(-1) #ignore background prediction

            batch_idxs_ = batch_idxs[object_idxs]
            batch_offsets_ = utils.get_batch_offsets(batch_idxs_, input.batch_size)
            coords_ = coords[object_idxs]
            pt_offsets_ = pt_offsets[object_idxs]
            semantic_preds_ = semantic_preds[object_idxs]
            regional_preds_ = regional_preds[object_idxs]
            regional_preds_2 = regional_preds2[object_idxs]
            size_preds_ = size_preds[object_idxs]

            algorithm_selection = 2 ### 1 is original rpgn algorithm with 3 digits, 2 is new clustering algorithm with 11 digit
            if algorithm_selection == 1:
                coords_offsets_regional = coords_ + pt_offsets_
                coords_regional = coords_ 
            elif algorithm_selection == 2:
                coords_offsets_regional = torch.cat(((coords_ + pt_offsets_),regional_preds_2.unsqueeze(-1).float()),dim=1)
                coords_regional = torch.cat((coords_,regional_preds_2.unsqueeze(-1).float()),dim=1)

            ### concate with pt_offset
            if algorithm_selection == 2:
                pt_offsets_dist = pt_offsets_ # original offset distance
                pt_offsets_norm = torch.norm(pt_offsets_, p=2, dim=1)
                pt_offsets_cos = pt_offsets_ / (pt_offsets_norm.unsqueeze(-1) + 1e-8)
                
                # ### use size_preds_ as digit 11
                # coords_regional = torch.cat((coords_regional,pt_offsets_cos.float(), pt_offsets_dist.float(), size_preds_.unsqueeze(-1).float()),dim=1)
                # coords_offsets_regional = torch.cat((coords_offsets_regional,pt_offsets_cos.float(), pt_offsets_dist.float(), size_preds_.unsqueeze(-1).float()),dim=1)
                ### use semantic_preds_ as digit 11, for testing
                # print("size_preds_.unsqueeze(-1).float():", size_preds_.unsqueeze(-1).float().shape)
                # print("semantic_preds_.unsqueeze(-1).float():", semantic_preds_.unsqueeze(-1).float().shape)

                ### changed size to semantic for testing
                coords_regional = torch.cat((coords_regional,pt_offsets_cos.float(), pt_offsets_dist.float(), semantic_preds_.unsqueeze(-1).float()),dim=1)
                coords_offsets_regional = torch.cat((coords_offsets_regional,pt_offsets_cos.float(), pt_offsets_dist.float(), semantic_preds_.unsqueeze(-1).float()),dim=1)

            semantic_preds_cpu = semantic_preds[object_idxs].int().cpu()
            regional_preds_cpu = regional_preds[object_idxs].int().cpu()

            P_Q_set_selection = 1 ### 0 is P+Q, 1 is P only, 2 is Q only
            if P_Q_set_selection == 0:
                idx_shift, start_len_shift = rpgn_ops.ballquery_batch_p(coords_offsets_regional, batch_idxs_, batch_offsets_, self.cluster_radius, self.cluster_shift_meanActive)
                proposals_idx_shift, proposals_offset_shift = rpgn_ops.bfs_cluster(semantic_preds_cpu, regional_preds_cpu, idx_shift.cpu(), start_len_shift.cpu(), self.cluster_npoint_thre)
                proposals_idx_shift[:, 1] = object_idxs[proposals_idx_shift[:, 1].long()].int()
                # proposals_idx_shift: (sumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
                # proposals_offset_shift: (nProposal + 1), int

                idx, start_len = rpgn_ops.ballquery_batch_p(coords_regional, batch_idxs_, batch_offsets_, self.cluster_radius, self.cluster_meanActive)
                proposals_idx, proposals_offset = rpgn_ops.bfs_cluster(semantic_preds_cpu, regional_preds_cpu, idx.cpu(), start_len.cpu(), self.cluster_npoint_thre)
                proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()
                # proposals_idx: (sumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
                # proposals_offset: (nProposal + 1), int

                proposals_idx_shift[:, 0] += (proposals_offset.size(0) - 1)
                proposals_offset_shift += proposals_offset[-1]
                proposals_idx = torch.cat((proposals_idx, proposals_idx_shift), dim=0)
                proposals_offset = torch.cat((proposals_offset, proposals_offset_shift[1:]))

            elif P_Q_set_selection == 1:
                idx, start_len = rpgn_ops.ballquery_batch_p(coords_regional, batch_idxs_, batch_offsets_, self.cluster_radius, self.cluster_meanActive)
                proposals_idx, proposals_offset = rpgn_ops.bfs_cluster(semantic_preds_cpu, regional_preds_cpu, idx.cpu(), start_len.cpu(), self.cluster_npoint_thre)
                proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()
                # proposals_idx: (sumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
                # proposals_offset: (nProposal + 1), int

                # proposals_idx_shift[:, 0] += (proposals_offset.size(0) - 1)
                # proposals_offset_shift += proposals_offset[-1]
                proposals_idx = proposals_idx #proposals_idx_shift  #torch.cat((proposals_idx, proposals_idx_shift), dim=0)
                proposals_offset = proposals_offset #proposals_offset_shift[1:]  #torch.cat((proposals_offset, proposals_offset_shift[1:]))

            elif P_Q_set_selection == 2:
                idx_shift, start_len_shift = rpgn_ops.ballquery_batch_p(coords_offsets_regional, batch_idxs_, batch_offsets_, self.cluster_radius, self.cluster_shift_meanActive)
                proposals_idx_shift, proposals_offset_shift = rpgn_ops.bfs_cluster(semantic_preds_cpu, regional_preds_cpu, idx_shift.cpu(), start_len_shift.cpu(), self.cluster_npoint_thre)
                proposals_idx_shift[:, 1] = object_idxs[proposals_idx_shift[:, 1].long()].int()
                # proposals_idx_shift: (sumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
                # proposals_offset_shift: (nProposal + 1), int

                idx, start_len = rpgn_ops.ballquery_batch_p(coords_offsets_regional, batch_idxs_, batch_offsets_, self.cluster_radius, self.cluster_shift_meanActive)
                proposals_idx, proposals_offset = rpgn_ops.bfs_cluster(semantic_preds_cpu, regional_preds_cpu, idx.cpu(), start_len.cpu(), self.cluster_npoint_thre)
                proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()
                # proposals_idx: (sumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
                # proposals_offset: (nProposal + 1), int

                proposals_idx_shift[:, 0] += (proposals_offset.size(0) - 1)
                proposals_offset_shift += proposals_offset[-1]
                proposals_idx = torch.cat((proposals_idx, proposals_idx_shift), dim=0)
                proposals_offset = torch.cat((proposals_offset, proposals_offset_shift[1:]))

            elif P_Q_set_selection == 3: # New Algorithm original coords + direction
                coords_offsets_regional = torch.cat((coords_,regional_preds_2.unsqueeze(-1).float()),dim=1)
                coords_regional = torch.cat(((coords_ - pt_offsets_),regional_preds_2.unsqueeze(-1).float()),dim=1)

                coords_regional = torch.cat((coords_regional,pt_offsets_cos.float(), pt_offsets_dist.float(), size_preds_.unsqueeze(-1).float()),dim=1)
                coords_offsets_regional = torch.cat((coords_offsets_regional,pt_offsets_cos.float(), pt_offsets_dist.float(), size_preds_.unsqueeze(-1).float()),dim=1)

                idx, start_len = rpgn_ops.ballquery_batch_p(coords_regional, batch_idxs_, batch_offsets_, self.cluster_radius, self.cluster_meanActive)
                proposals_idx, proposals_offset = rpgn_ops.bfs_cluster(semantic_preds_cpu, regional_preds_cpu, idx.cpu(), start_len.cpu(), self.cluster_npoint_thre)
                proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()
                # proposals_idx: (sumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
                # proposals_offset: (nProposal + 1), int

                # proposals_idx_shift[:, 0] += (proposals_offset.size(0) - 1)
                # proposals_offset_shift += proposals_offset[-1]
                proposals_idx = proposals_idx #proposals_idx_shift  #torch.cat((proposals_idx, proposals_idx_shift), dim=0)
                proposals_offset = proposals_offset #proposals_offset_shift[1:]  #torch.cat((proposals_offset, proposals_offset_shift[1:]))

            elif P_Q_set_selection == 4: # New Algorithm dual set, need to use NMS
                coords_offsets_regional = torch.cat((coords_,regional_preds_2.unsqueeze(-1).float()),dim=1)
                coords_regional = torch.cat(((coords_ - pt_offsets_),regional_preds_2.unsqueeze(-1).float()),dim=1)

                coords_regional = torch.cat((coords_regional,pt_offsets_cos.float(), pt_offsets_dist.float(), size_preds_.unsqueeze(-1).float()),dim=1)
                coords_offsets_regional = torch.cat((coords_offsets_regional,pt_offsets_cos.float(), pt_offsets_dist.float(), size_preds_.unsqueeze(-1).float()),dim=1)

                idx_shift, start_len_shift = rpgn_ops.ballquery_batch_p(coords_offsets_regional, batch_idxs_, batch_offsets_, self.cluster_radius, self.cluster_shift_meanActive)
                proposals_idx_shift, proposals_offset_shift = rpgn_ops.bfs_cluster(semantic_preds_cpu, regional_preds_cpu, idx_shift.cpu(), start_len_shift.cpu(), self.cluster_npoint_thre)
                proposals_idx_shift[:, 1] = object_idxs[proposals_idx_shift[:, 1].long()].int()
                # proposals_idx_shift: (sumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
                # proposals_offset_shift: (nProposal + 1), int

                idx, start_len = rpgn_ops.ballquery_batch_p(coords_regional, batch_idxs_, batch_offsets_, self.cluster_radius, self.cluster_meanActive)
                proposals_idx, proposals_offset = rpgn_ops.bfs_cluster(semantic_preds_cpu, regional_preds_cpu, idx.cpu(), start_len.cpu(), self.cluster_npoint_thre)
                proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()
                # proposals_idx: (sumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
                # proposals_offset: (nProposal + 1), int

                proposals_idx_shift[:, 0] += (proposals_offset.size(0) - 1)
                proposals_offset_shift += proposals_offset[-1]
                proposals_idx = torch.cat((proposals_idx, proposals_idx_shift), dim=0)
                proposals_offset = torch.cat((proposals_offset, proposals_offset_shift[1:]))

            #### proposals voxelization again
            if proposals_idx.shape[0] > 0:
                input_feats, inp_map = self.clusters_voxelization(proposals_idx, proposals_offset, output_feats, coords, self.score_fullscale, self.score_scale, self.mode)

                #### score
                score = self.score_unet(input_feats)
                score = self.score_outputlayer(score)
                score_feats = score.features[inp_map.long()] # (sumNPoint, C)
                score_feats = rpgn_ops.roipool(score_feats, proposals_offset.cuda())  # (nProposal, C)
                scores = self.score_linear(score_feats)  # (nProposal, 1)

            else:
                scores = torch.zeros(1,2)

            ret['proposal_scores'] = (scores, proposals_idx, proposals_offset)

        return ret


def model_fn_decorator(test=False):
    #### config
    from util.config import cfg

    #### criterion
    semantic_criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label).cuda()
    regional_criterion = nn.CrossEntropyLoss().cuda() #nn.CrossEntropyLoss(ignore_index=0).cuda() #nn.CrossEntropyLoss().cuda()
    regional_criterion2 = nn.CrossEntropyLoss().cuda() #nn.CrossEntropyLoss(ignore_index=0).cuda() #nn.CrossEntropyLoss().cuda()
    size_criterion = nn.CrossEntropyLoss().cuda() #nn.CrossEntropyLoss(ignore_index=0).cuda() #nn.CrossEntropyLoss().cuda()
    direction_criterion = nn.CrossEntropyLoss().cuda() #nn.CrossEntropyLoss(ignore_index=0).cuda() #nn.BCELoss(reduction='mean').cuda() #nn.CrossEntropyLoss().cuda() #nn.CrossEntropyLoss(ignore_index=0).cuda()
    score_criterion = nn.BCELoss(reduction='none').cuda()

    def test_model_fn(batch, model, epoch):
        coords = batch['locs'].cuda()              # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
        voxel_coords = batch['voxel_locs'].cuda()  # (M, 1 + 3), long, cuda
        p2v_map = batch['p2v_map'].cuda()          # (N), int, cuda
        v2p_map = batch['v2p_map'].cuda()          # (M, 1 + maxActive), int, cuda

        coords_float = batch['locs_float'].cuda()  # (N, 3), float32, cuda
        feats = batch['feats'].cuda()              # (N, C), float32, cuda

        batch_offsets = batch['offsets'].cuda()    # (B + 1), int, cuda

        spatial_shape = batch['spatial_shape']

        batch_id = batch['id']

        if cfg.use_coords:
            feats = torch.cat((feats, coords_float), 1)
        voxel_feats = rpgn_ops.voxelization(feats, v2p_map, cfg.mode)  # (M, C), float, cuda

        input_ = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, cfg.batch_size)

        ret = model(input_, p2v_map, coords_float, coords[:, 0].int(), batch_offsets, epoch, batch_id)
        semantic_scores = ret['semantic_scores']  # (N, nClass) float32, cuda
        regional_scores = ret['regional_scores']  # (N, nClass) float32, cuda
        regional_scores2 = ret['regional_scores2']  # (N, nClass) float32, cuda
        size_scores = ret['size_scores']  # (N, nClass) float32, cuda
        # direction_scores = ret['direction_scores']  # (N, nClass) float32, cuda
        pt_offsets = ret['pt_offsets']            # (N, 3), float32, cuda
        if (epoch > cfg.prepare_epochs):
            scores, proposals_idx, proposals_offset = ret['proposal_scores']

        ##### preds
        with torch.no_grad():
            preds = {}
            preds['semantic'] = semantic_scores
            preds['regional'] = regional_scores
            preds['regional2'] = regional_scores2
            preds['size'] = size_scores
            # preds['direction'] = direction_scores
            preds['pt_offsets'] = pt_offsets
            if (epoch > cfg.prepare_epochs):
                preds['score'] = scores
                preds['proposals'] = (proposals_idx, proposals_offset)

        return preds


    def model_fn(batch, model, epoch):
        ##### prepare input and forward
        # batch {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
        # 'locs_float': locs_float, 'feats': feats, 'labels': labels, 'instance_labels': instance_labels,
        # 'instance_info': instance_infos, 'instance_pointnum': instance_pointnum,
        # 'id': tbl, 'offsets': batch_offsets, 'spatial_shape': spatial_shape}
        coords = batch['locs'].cuda()                          # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
        voxel_coords = batch['voxel_locs'].cuda()              # (M, 1 + 3), long, cuda
        p2v_map = batch['p2v_map'].cuda()                      # (N), int, cuda
        v2p_map = batch['v2p_map'].cuda()                      # (M, 1 + maxActive), int, cuda

        coords_float = batch['locs_float'].cuda()              # (N, 3), float32, cuda
        feats = batch['feats'].cuda()                          # (N, C), float32, cuda
        labels = batch['labels'].cuda()                        # (N), long, cuda
        instance_labels = batch['instance_labels'].cuda()      # (N), long, cuda, 0~total_nInst, -100
        regional_labels = batch['regional_labels'].cuda()
        regional_labels2 = batch['regional_labels2'].cuda()
        distance_labels = batch['distance_labels'].cuda()
        distance_labels2 = batch['distance_labels2'].cuda()
        direction_labels = batch['direction_labels'].cuda()
        size_labels = batch['size_labels'].cuda()

        instance_info = batch['instance_info'].cuda()          # (N, 9), float32, cuda, (meanxyz, minxyz, maxxyz)
        instance_pointnum = batch['instance_pointnum'].cuda()  # (total_nInst), int, cuda

        batch_offsets = batch['offsets'].cuda()                # (B + 1), int, cuda

        spatial_shape = batch['spatial_shape']

        if cfg.use_coords:
            feats = torch.cat((feats, coords_float), 1)
        voxel_feats = rpgn_ops.voxelization(feats, v2p_map, cfg.mode)  # (M, C), float, cuda

        input_ = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, cfg.batch_size)

        ret = model(input_, p2v_map, coords_float, coords[:, 0].int(), batch_offsets, epoch)
        # make_dot(ret, params=dict(list(model.named_parameters()))) #dict(model.named_parameters()))
        # with torch.onnx.set_training(model, False):
        #     trace, _ = torch.jit.get_trace_graph(model, args=(x,))
        # make_dot_from_trace(trace)
        semantic_scores = ret['semantic_scores'] # (N, nClass) float32, cuda
        regional_scores = ret['regional_scores']
        regional_scores2 = ret['regional_scores2']
        size_scores = ret['size_scores']
        # direction_scores = ret['direction_scores']
        pt_offsets = ret['pt_offsets']           # (N, 3), float32, cuda
        if(epoch > cfg.prepare_epochs):
            scores, proposals_idx, proposals_offset = ret['proposal_scores']
            # scores: (nProposal, 1) float, cuda
            # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
            # proposals_offset: (nProposal + 1), int, cpu

        loss_inp = {}
        loss_inp['semantic_scores'] = (semantic_scores, labels)
        loss_inp['regional_scores'] = (regional_scores, regional_labels)
        loss_inp['regional_scores2'] = (regional_scores2, regional_labels2)
        loss_inp['distance_labels'] = (distance_labels)
        loss_inp['distance_labels2'] = (distance_labels2)
        # loss_inp['direction_scores'] = (direction_scores, direction_labels)
        loss_inp['size_scores'] = (size_scores, size_labels)

        loss_inp['pt_offsets'] = (pt_offsets, coords_float, instance_info, instance_labels)
        if(epoch > cfg.prepare_epochs):
            loss_inp['proposal_scores'] = (scores, proposals_idx, proposals_offset, instance_pointnum)

        loss, loss_out, infos = loss_fn(loss_inp, epoch)

        ##### accuracy / visual_dict / meter_dict
        with torch.no_grad():
            preds = {}
            preds['semantic'] = semantic_scores
            preds['regional'] = regional_scores
            preds['regional2'] = regional_scores2
            preds['size'] = size_scores
            # preds['direction'] = direction_scores
            preds['pt_offsets'] = pt_offsets
            if(epoch > cfg.prepare_epochs):
                preds['score'] = scores
                preds['proposals'] = (proposals_idx, proposals_offset)

            visual_dict = {}
            visual_dict['loss'] = loss
            for k, v in loss_out.items():
                visual_dict[k] = v[0]

            meter_dict = {}
            meter_dict['loss'] = (loss.item(), coords.shape[0])
            for k, v in loss_out.items():
                meter_dict[k] = (float(v[0]), v[1])

        return loss, preds, visual_dict, meter_dict

    def direction_IoU_loss(input, target):
            smooth = 1.
            iflat = input.view(-1)
            tflat = target.view(-1)
            intersection = (iflat * tflat).sum()
            
            return 1 - ((intersection + smooth) / (iflat.sum() + tflat.sum() - intersection + smooth))

    ALPHA = 0.8 #0.5
    BETA = 0.2 #0.5

    def direction_TverskyLoss(inputs, targets, smooth=1, alpha=ALPHA, beta=BETA):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return 1 - Tversky
    
    def loss_fn(loss_inp, epoch):

        loss_out = {}
        infos = {}

        '''semantic loss'''
        semantic_scores, semantic_labels = loss_inp['semantic_scores']
        # semantic_scores: (N, nClass), float32, cuda
        # semantic_labels: (N), long, cuda

        semantic_loss = semantic_criterion(semantic_scores, semantic_labels)
        loss_out['semantic_loss'] = (semantic_loss, semantic_scores.shape[0])

        '''regional loss'''
        regional_scores, regional_labels = loss_inp['regional_scores']
        # regional_scores: (N, regional_size), float32, cuda
        # regional_labels: (N), long, cuda

        distance_labels = loss_inp['distance_labels']

        # ### label zero considers only foreground points
        # object_idxs = torch.nonzero(semantic_labels > 1).view(-1) #ignore background prediction
        # regional_scores = regional_scores[object_idxs]
        # regional_labels = regional_labels[object_idxs]
        regional_scores2, regional_labels2 = loss_inp['regional_scores2']
        distance_labels2 = loss_inp['distance_labels2']
        size_scores, size_labels = loss_inp['size_scores']

        ######################### Downsampling Background Points #####################
        ### for torch > 1.5
        regional_labels_zero = torch.nonzero(((regional_labels == 0)&(semantic_labels >= 2)), as_tuple=False)

        regional_labels_nonzero = torch.nonzero(regional_labels != 0, as_tuple=False) 
        regional_labels_one = torch.nonzero(regional_labels == 1, as_tuple=False) 
        regional_labels_two = torch.nonzero(regional_labels == 2, as_tuple=False)

        regional_labels_zero_mask = regional_labels_zero[:,0]
        regional_labels_nonzero_mask = regional_labels_nonzero[:,0]
        regional_labels_one_mask = regional_labels_one[:,0]
        regional_labels_two_mask = regional_labels_two[:,0]

        perm = torch.randperm(regional_labels_zero_mask.size(0))
        select_num = int(regional_labels_zero_mask.size(0) * 1.0)

        rand_idx = perm[:select_num]
        regional_labels_zero_mask_sampled = regional_labels_zero_mask
        regional_labels_two_mask_sampled = regional_labels_two_mask
        regional_labels_mask = torch.cat((regional_labels_zero_mask_sampled, regional_labels_one_mask, regional_labels_two_mask_sampled), 0)
        ##############################################################################

        regional_loss = regional_criterion(regional_scores[regional_labels_mask], regional_labels[regional_labels_mask].squeeze())
        loss_out['regional_loss'] = (regional_loss, regional_scores.shape[0])

        regional_loss2 = regional_criterion(regional_scores2[regional_labels_mask], regional_labels2[regional_labels_mask].squeeze())
        loss_out['regional_loss2'] = (regional_loss2, regional_scores2.shape[0])

        size_loss = size_criterion(size_scores, size_labels.squeeze())
        loss_out['size_loss'] = (size_loss, size_scores.shape[0])

        ################################## Dice loss ##################################
        def dice_loss(input, target):
            smooth = 1.
            iflat = input.view(-1)
            tflat = target.view(-1)
            intersection = (iflat * tflat).sum()
            
            return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


        def dice_BCE_loss(input, target):
            smooth = 1.
            iflat = input.view(-1)
            tflat = target.view(-1)
            intersection = (iflat * tflat).sum()
            dice_loss = 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))
            BCE = F.binary_cross_entropy(iflat, tflat.type(torch.cuda.FloatTensor), reduction='mean')
            Dice_BCE = BCE + dice_loss

            return Dice_BCE


        ALPHA = 0.8 #0.5
        BETA = 0.2 #0.5

        def TverskyLoss(inputs, targets, smooth=1, alpha=ALPHA, beta=BETA):
            
            #comment out if your model contains a sigmoid or equivalent activation layer
            #inputs = F.sigmoid(inputs)

            alpha=0.8
            beta=0.2       
            
            #flatten label and prediction tensors
            inputs = inputs.view(-1)
            targets = targets.view(-1)
            
            #True Positives, False Positives & False Negatives
            TP = (inputs * targets).sum()    
            FP = ((1-targets) * inputs).sum()
            FN = (targets * (1-inputs)).sum()
           
            Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
            
            return 1 - Tversky

        def TverskyLoss2(inputs, targets, smooth=1, alpha=ALPHA, beta=BETA):
            
            #comment out if your model contains a sigmoid or equivalent activation layer
            #inputs = F.sigmoid(inputs)

            alpha=0.8
            beta=0.2       
            
            #flatten label and prediction tensors
            inputs = inputs.view(-1)
            targets = targets.view(-1)
            
            #True Positives, False Positives & False Negatives
            TP = (inputs * targets).sum()    
            FP = ((1-targets) * inputs).sum()
            FN = (targets * (1-inputs)).sum()
           
            Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
            
            return 1 - Tversky

        def distance_loss(inputs, targets, gt_dist_map):
            smooth = 1.

            inputs = inputs.view(-1)
            targets = targets.view(-1)
            gt_dist_map = gt_dist_map.view(-1)  
            FP = ((1-targets) * inputs)#.sum()

            return ((gt_dist_map.mul(FP)).sum()+smooth) / (FP.sum()+smooth)

        sigmoid = nn.Sigmoid()
        regional_scores_two2 = regional_scores2[regional_labels_mask][:,2]

        ### fixed Tversky loss bug
        regional_label_two2 = regional_labels2[regional_labels_mask]
        regional_label_two2[regional_label_two2==1] = 0.0
        regional_label_two2[regional_label_two2==2] = 1.0
        TverskyLoss2_2 = TverskyLoss(sigmoid(regional_scores_two2), regional_label_two2)
        regional_scores_zero2 = regional_scores2[regional_labels_mask][:,0]

        regional_label_zero2 = regional_labels2[regional_labels_mask]
        regional_label_zero2[regional_label_zero2==0] = 3
        regional_label_zero2[regional_label_zero2==1] = 0
        regional_label_zero2[regional_label_zero2==2] = 0
        regional_label_zero2[regional_label_zero2==3] = 1
        TverskyLoss2_0 = TverskyLoss2(sigmoid(regional_scores_zero2), regional_label_zero2)

        TverskyLoss2 = (TverskyLoss2_2 + TverskyLoss2_0) / 4 #2
        loss_out['TverskyLoss2'] = (TverskyLoss2, semantic_scores.shape[0])
        distance_loss2 = distance_loss(sigmoid(regional_scores_two2), regional_label_two2, distance_labels2[regional_labels_mask,:])
        loss_out['distance_loss2'] = (distance_loss2, semantic_scores.shape[0])
       
        regional_scores_two = regional_scores[regional_labels_mask][:,2]
        regional_label_two = (regional_labels[regional_labels_mask].squeeze()) / 2 # from label 2 to value 1
     
        TverskyLoss = TverskyLoss(sigmoid(regional_scores_two), regional_label_two)
        loss_out['TverskyLoss'] = (TverskyLoss, semantic_scores.shape[0])
        distance_loss = distance_loss(sigmoid(regional_scores_two), regional_label_two, distance_labels[regional_labels_mask,:])
        loss_out['distance_loss'] = (distance_loss, semantic_scores.shape[0])

        '''offset loss'''
        pt_offsets, coords, instance_info, instance_labels = loss_inp['pt_offsets']
        # pt_offsets: (N, 3), float, cuda
        # coords: (N, 3), float32
        # instance_info: (N, 9), float32 tensor (meanxyz, minxyz, maxxyz)
        # instance_labels: (N), long

        gt_offsets = instance_info[:, 0:3] - coords   # (N, 3)
        pt_diff = pt_offsets - gt_offsets   # (N, 3)
        pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)   # (N)
        valid = (instance_labels != cfg.ignore_label).float()
        offset_norm_loss = torch.sum(pt_dist * valid) / (torch.sum(valid) + 1e-6)

        gt_offsets_norm = torch.norm(gt_offsets, p=2, dim=1)   # (N), float
        gt_offsets_ = gt_offsets / (gt_offsets_norm.unsqueeze(-1) + 1e-8)
        pt_offsets_norm = torch.norm(pt_offsets, p=2, dim=1)
        pt_offsets_ = pt_offsets / (pt_offsets_norm.unsqueeze(-1) + 1e-8)
        direction_diff = - (gt_offsets_ * pt_offsets_).sum(-1)   # (N)
        offset_dir_loss = torch.sum(direction_diff * valid) / (torch.sum(valid) + 1e-6)
        ### added a normalize factor of 1 to avoid negative value
        offset_dir_loss += 1

        loss_out['offset_norm_loss'] = (offset_norm_loss, valid.sum())
        loss_out['offset_dir_loss'] = (offset_dir_loss, valid.sum())

        if (epoch > cfg.prepare_epochs):
            '''score loss'''
            scores, proposals_idx, proposals_offset, instance_pointnum = loss_inp['proposal_scores']
            # scores: (nProposal, 1), float32
            # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
            # proposals_offset: (nProposal + 1), int, cpu
            # instance_pointnum: (total_nInst), int

            ious = rpgn_ops.get_iou(proposals_idx[:, 1].cuda(), proposals_offset.cuda(), instance_labels, instance_pointnum) # (nProposal, nInstance), float
            gt_ious, gt_instance_idxs = ious.max(1)  # (nProposal) float, long
            gt_scores = get_segmented_scores(gt_ious, cfg.fg_thresh, cfg.bg_thresh)

            score_loss = score_criterion(torch.sigmoid(scores.view(-1)), gt_scores)
            score_loss = score_loss.mean()

            loss_out['score_loss'] = (score_loss, gt_ious.shape[0])

        '''total loss'''
        loss = TverskyLoss2 + regional_loss2 + cfg.loss_weight[0] * semantic_loss + cfg.loss_weight[1] * offset_norm_loss + cfg.loss_weight[2] * offset_dir_loss #+ 0.1*size_loss
    
        if(epoch > cfg.prepare_epochs):
            loss += (cfg.loss_weight[3] * score_loss)

        return loss, loss_out, infos


    def get_segmented_scores(scores, fg_thresh=1.0, bg_thresh=0.0):
        '''
        :param scores: (N), float, 0~1
        :return: segmented_scores: (N), float 0~1, >fg_thresh: 1, <bg_thresh: 0, mid: linear
        '''
        fg_mask = scores > fg_thresh
        bg_mask = scores < bg_thresh
        interval_mask = (fg_mask == 0) & (bg_mask == 0)

        segmented_scores = (fg_mask > 0).float()
        k = 1 / (fg_thresh - bg_thresh)
        b = bg_thresh / (bg_thresh - fg_thresh)
        segmented_scores[interval_mask] = scores[interval_mask] * k + b

        return segmented_scores


    if test:
        fn = test_model_fn
    else:
        fn = model_fn
    return fn
