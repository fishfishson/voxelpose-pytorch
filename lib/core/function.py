from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os
import copy
import cv2

import torch
import numpy as np
from tqdm import tqdm

from utils.vis import *

logger = logging.getLogger(__name__)


def train_3d(config, model, optimizer, loader, epoch, output_dir, writer_dict, device=torch.device('cuda'), dtype=torch.float, finetune_backbone=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_2d = AverageMeter()
    losses_3d = AverageMeter()
    losses_cord = AverageMeter()

    model.train()

    if model.module.backbone is not None:
        if not finetune_backbone:
            model.module.backbone.eval()  # Comment out this line if you want to train 2D backbone jointly

    accumulation_steps = 4
    accu_loss_3d = 0

    end = time.time()
    for i, (inputs, targets_2d, weights_2d, targets_3d, meta, input_heatmap) in enumerate(loader):
        data_time.update(time.time() - end)

        if 'panoptic' in config.DATASET.TEST_DATASET or 'chi3d' in config.DATASET.TEST_DATASET:
            pred, heatmaps, grid_centers, loss_2d, loss_3d, loss_cord = model(views=inputs, meta=meta,
                                                                              targets_2d=targets_2d,
                                                                              weights_2d=weights_2d,
                                                                              targets_3d=targets_3d[0])
        elif 'campus' in config.DATASET.TEST_DATASET or 'shelf' in config.DATASET.TEST_DATASET:
            pred, heatmaps, grid_centers, loss_2d, loss_3d, loss_cord = model(meta=meta, targets_3d=targets_3d[0],
                                                                              input_heatmaps=input_heatmap)

        loss_2d = loss_2d.mean()
        loss_3d = loss_3d.mean()
        loss_cord = loss_cord.mean()

        losses_2d.update(loss_2d.item())
        losses_3d.update(loss_3d.item())
        losses_cord.update(loss_cord.item())
        loss = loss_2d + loss_3d + loss_cord
        losses.update(loss.item())

        loss = loss / accumulation_steps
        loss.backward()
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # if loss_cord > 0:
        #     optimizer.zero_grad()
        #     (loss_2d + loss_cord).backward()
        #     optimizer.step()

        # if accu_loss_3d > 0 and (i + 1) % accumulation_steps == 0:
        #     optimizer.zero_grad()
        #     accu_loss_3d.backward()
        #     optimizer.step()
        #     accu_loss_3d = 0.0
        # else:
        #     accu_loss_3d += loss_3d / accumulation_steps

        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            gpu_memory_usage = torch.cuda.memory_allocated(0) / 1024 / 1024
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed: {speed:.1f} samples/s\t' \
                  'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss: {loss.val:.6f} ({loss.avg:.6f})\t' \
                  'Loss_2d: {loss_2d.val:.7f} ({loss_2d.avg:.7f})\t' \
                  'Loss_3d: {loss_3d.val:.7f} ({loss_3d.avg:.7f})\t' \
                  'Loss_cord: {loss_cord.val:.6f} ({loss_cord.avg:.6f})\t' \
                  'Memory {memory:.1f}'.format(
                    epoch, i, len(loader), batch_time=batch_time,
                    speed=len(inputs) * inputs[0].size(0) / batch_time.val,
                    data_time=data_time, loss=losses, loss_2d=losses_2d, loss_3d=losses_3d,
                    loss_cord=losses_cord, memory=gpu_memory_usage)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss_3d', losses_3d.val, global_steps)
            writer.add_scalar('train_loss_cord', losses_cord.val, global_steps)
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            seqs = meta[0]['key']
            for b in range(len(seqs)):
                seq = seqs[b].split('_')
                imgs = []
                for k in range(len(inputs)):
                    hm_gt = save_batch_heatmaps_multi(inputs[k][b:b+1], targets_2d[k][b:b+1], None)
                    hm_pred = save_batch_heatmaps_multi(inputs[k][b:b+1], heatmaps[k][b:b+1], None)
                    img = np.vstack([hm_gt, hm_pred])
                    imgs.append(img)
                imgs = np.vstack(imgs)
                prefix = '{}/{:06}_{}_{}_{}'.format(
                    os.path.join(output_dir, 'train'), global_steps, seq[0], seq[1], seq[-1])
                basename = os.path.basename(prefix)
                dirname = os.path.dirname(prefix)
                dirname1 = os.path.join(dirname, 'heatmaps')
                if not os.path.exists(dirname1):
                    os.makedirs(dirname1)
                cv2.imwrite(os.path.join(dirname1, basename) + '.jpg', imgs)
            save_debug_3d_cubes(config, meta[0], grid_centers, prefix)
            save_debug_3d_images(config, meta[0], pred, prefix)


def validate_3d(config, model, loader, output_dir, epoch=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    model.eval()

    preds = []
    with torch.no_grad():
        end = time.time()
        for i, (inputs, targets_2d, weights_2d, targets_3d, meta, input_heatmap) in tqdm(enumerate(loader)):
            data_time.update(time.time() - end)
            if 'panoptic' in config.DATASET.TEST_DATASET or 'chi3d' in config.DATASET.TEST_DATASET:
                pred, heatmaps, grid_centers, _, _, _ = model(views=inputs, meta=meta, targets_2d=targets_2d,
                                                              weights_2d=weights_2d, targets_3d=targets_3d[0])
            elif 'campus' in config.DATASET.TEST_DATASET or 'shelf' in config.DATASET.TEST_DATASET:
                pred, heatmaps, grid_centers, _, _, _ = model(meta=meta, targets_3d=targets_3d[0],
                                                              input_heatmaps=input_heatmap)
            pred = pred.detach().cpu().numpy()
            for b in range(pred.shape[0]):
                preds.append(pred[b])

            batch_time.update(time.time() - end)
            end = time.time()
            if i % config.PRINT_FREQ == 0 or i == len(loader) - 1:
                gpu_memory_usage = torch.cuda.memory_allocated(0)
                msg = 'Test: [{0}/{1}]\t' \
                      'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed: {speed:.1f} samples/s\t' \
                      'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'Memory {memory:.1f}'.format(
                        i, len(loader), batch_time=batch_time,
                        speed=len(inputs) * inputs[0].size(0) / batch_time.val,
                        data_time=data_time, memory=gpu_memory_usage)
                logger.info(msg)

                seqs = meta[0]['key']
                for b in range(len(seqs)):
                    seq = seqs[b].split('_')
                    imgs = []
                    for k in range(len(inputs)):
                        hm_gt = save_batch_heatmaps_multi(inputs[k][b:b+1], targets_2d[k][b:b+1], None)
                        hm_pred = save_batch_heatmaps_multi(inputs[k][b:b+1], heatmaps[k][b:b+1], None)
                        img = np.vstack([hm_gt, hm_pred])
                        imgs.append(img)
                    imgs = np.vstack(imgs)
                    if epoch is not None:
                        prefix = '{}/{:03}_{}_{}_{}'.format(
                            os.path.join(output_dir, 'validation'), epoch, seq[0], seq[1], seq[-1])
                    else:
                        prefix = '{}/val_{}_{}_{}'.format(
                            os.path.join(output_dir, 'validation'), seq[0], seq[1], seq[-1])
                    basename = os.path.basename(prefix)
                    dirname = os.path.dirname(prefix)
                    dirname1 = os.path.join(dirname, 'heatmaps')
                    if not os.path.exists(dirname1):
                        os.makedirs(dirname1)
                    cv2.imwrite(os.path.join(dirname1, basename) + '.jpg', imgs)
                save_debug_3d_cubes(config, meta[0], grid_centers, prefix)
                save_debug_3d_images(config, meta[0], pred, prefix)
            
            if epoch is not None:
                pass
            else:
                key = meta[0]['key'][0].split('_')
                if 'Hug' in key[1]:
                    save_debug_3d_json(config, meta[0], pred, output_dir, vis=True)
                else:
                    save_debug_3d_json(config, meta[0], pred, output_dir, vis=False)
            
    metric = None
    if 'panoptic' in config.DATASET.TEST_DATASET or 'chi3d' in config.DATASET.TEST_DATASET:
        aps, _, mpjpe, recall = loader.dataset.evaluate(preds)
        msg = 'ap@25: {aps_25:.4f}\tap@50: {aps_50:.4f}\tap@75: {aps_75:.4f}\t' \
              'ap@100: {aps_100:.4f}\tap@125: {aps_125:.4f}\tap@150: {aps_150:.4f}\t' \
              'recall@500mm: {recall:.4f}\tmpjpe@500mm: {mpjpe:.3f}'.format(
                aps_25=aps[0], aps_50=aps[1], aps_75=aps[2], aps_100=aps[3],
                aps_125=aps[4], aps_150=aps[5], recall=recall, mpjpe=mpjpe
              )
        logger.info(msg)
        metric = np.mean(aps)
    elif 'campus' in config.DATASET.TEST_DATASET or 'shelf' in config.DATASET.TEST_DATASET:
        actor_pcp, avg_pcp, _, recall = loader.dataset.evaluate(preds)
        msg = '     | Actor 1 | Actor 2 | Actor 3 | Average | \n' \
              ' PCP |  {pcp_1:.2f}  |  {pcp_2:.2f}  |  {pcp_3:.2f}  |  {pcp_avg:.2f}  |\t Recall@500mm: {recall:.4f}'.format(
                pcp_1=actor_pcp[0]*100, pcp_2=actor_pcp[1]*100, pcp_3=actor_pcp[2]*100, pcp_avg=avg_pcp*100, recall=recall)
        logger.info(msg)
        metric = np.mean(avg_pcp)

    return metric


def speed_3d(config, model, loader, output_dir, epoch=None):
    full_time_metric = AverageMeter()
    pose_time_metric = AverageMeter()
    model.eval()

    with torch.no_grad():
        for i, (inputs, targets_2d, weights_2d, targets_3d, meta, input_heatmap) in tqdm(enumerate(loader)):
            if 'panoptic' in config.DATASET.TEST_DATASET or 'chi3d' in config.DATASET.TEST_DATASET:
                pred, full_time, pose_time = model(views=inputs, meta=meta, targets_2d=targets_2d,
                             weights_2d=weights_2d, targets_3d=targets_3d[0], test_time=True)
            elif 'campus' in config.DATASET.TEST_DATASET or 'shelf' in config.DATASET.TEST_DATASET:
                pred = model(meta=meta, targets_3d=targets_3d[0], input_heatmaps=input_heatmap, test_time=True)
            pred = pred.detach().cpu().numpy()
            
            full_time_metric.update(full_time)
            pose_time_metric.update(pose_time)
    
            print(full_time_metric.avg)
            print(pose_time_metric.avg)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
