import os
import shutil
import time
import pickle
import json
import copy
import collections
from terminaltables import AsciiTable

import numpy as np
import random
from copy import deepcopy

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from .lr_schedulers import LinearWarmupMultiStepLR, LinearWarmupCosineAnnealingLR
from .postprocessing import postprocess_results
from .detect_eval import compute_AP_AR
from ..modeling import MaskedConv1D, Scale, AffineDropPath, LayerNorm
from IPython import embed
from tqdm import tqdm

################################################################################
def calculate_IoU_batch2(i0, i1):
    union = (np.min(np.stack([i0[0], i1[0]], 0), 0), np.max(np.stack([i0[1], i1[1]], 0), 0))
    inter = (np.max(np.stack([i0[0], i1[0]], 0), 0), np.min(np.stack([i0[1], i1[1]], 0), 0))
    iou = 1.0 * (inter[1] - inter[0] + 1) / (union[1] - union[0] + 1)
    iou[union[1] - union[0] < -1e-5] = 0
    iou[iou < 0] = 0.0
    return iou

j_list = [70, 80, 90]
IOU_THRESHOLDS = [0.75, 0.85, 0.95]
AR_IOU_THRESHOLDS = [0.75, 0.85, 0.95]
print(f'AR_IOU_THRESHOLDS: {AR_IOU_THRESHOLDS}', flush=True)
MAX_DETECTIONS = [1, 5, 10]

# [nb, 2], [nb, 2]
def top_n_metric(preds, label):
    result = {}
    bsz = preds[0].shape[0]
    top_iou = []
    for pred in preds:
        iou = calculate_IoU_batch2((pred[:, 0], pred[:, 1]), (label[:, 0], label[:, 1]))
        top_iou.append(iou)
    iou = np.max(np.stack(top_iou, 1), 1)
    result['mIoU'] = np.mean(iou)
    # for i in range(1, 10, 1):
    #     result['IoU@0.{}'.format(i)] = 1.0 * np.sum(iou >= i / 10) / bsz
    for i in np.arange(0, 100, 1):
        result['IoU@{}'.format(i / 100)] = np.sum(iou >= i / 100) / bsz
    return result

def nms_temporal(x1,x2,s, overlap):
    pick = []
    assert len(x1)==len(s)
    assert len(x2)==len(s)
    if len(x1)==0:
        return pick
    #     union = map(operator.sub, x2, x1) # union = x2-x1
    union = list(map(operator.sub, x2, x1)) # union = x2-x1

    I = [i[0] for i in sorted(enumerate(s), key=lambda x:x[1])] # sort and get index

    while len(I)>0:
        i = I[-1]
        pick.append(i)
        xx1 = [max(x1[i],x1[j]) for j in I[:-1]]
        xx2 = [min(x2[i],x2[j]) for j in I[:-1]]
        inter = [max(0.0, k2-k1) for k1, k2 in zip(xx1, xx2)]
        o = [inter[u]/(union[i] + union[I[u]] - inter[u]) for u in range(len(I)-1)]
        I_new = []
        for j in range(len(o)):
            if o[j] <=overlap:
                I_new.append(I[j])
        I = I_new
    return pick

#     table = AsciiTable(display_list)
#     return table.table

def display_python_performance(x, data='CharadesSTA'):
    """
    """
    global IOU_THRESHOLDS, MAX_DETECTIONS

    ap_keys = [f"mAP@{t}".rstrip("0").rstrip(".") for t in IOU_THRESHOLDS] + ["mAP"]
    ar_keys = [f"AR@{k}" for k in MAX_DETECTIONS] + ["mAR"]

    # ===== AP 部分 =====
    ap_table_data = [["Type"] + ap_keys]
    ap_row = ["AP"]
    for k in ap_keys:
        v = getattr(x[k], "avg", x[k]) if hasattr(x[k], "avg") else x[k]
        ap_row.append(f"{v * 100:.2f}")
    ap_table_data.append(ap_row)

    ap_table = AsciiTable(ap_table_data)
    ap_str = ap_table.table

    # ===== AR 部分 =====
    ar_table_data = [["Type"] + ar_keys]
    ar_row = ["AR"]
    for k in ar_keys:
        v = getattr(x[k], "avg", x[k]) if hasattr(x[k], "avg") else x[k]
        ar_row.append(f"{v * 100:.2f}")
    ar_table_data.append(ar_row)

    ar_table = AsciiTable(ar_table_data)
    ar_str = ar_table.table

    final_output = f"{ap_str}\n{ar_str}"

    return final_output



def print_md_performance(x, data='CharadesSTA'):
    item = ''
    i_list = [1]
    # j_list = [7, 8, 9]

    for i in i_list:
        for j in j_list:
            key = 'IoU@{}'.format(j/100)
            item += '| {:.2f} '.format(x[key].avg * 100)
        item += '| {:.2f} |\n'.format(x[f'mIoU'].avg * 100)

    for i in i_list:
        for j in j_list:
            key = 'IoU@{}'.format(j/100)
            item += '&{:.2f} '.format(x[key].avg * 100)
        item += '&{:.2f} |'.format(x[f'mIoU'].avg * 100)
        if i_list[-1] != i:
            item += '\n'
    return item

def get_average_performance(x, data='CharadesSTA', i=1):
    # # j_list = [7, 8, 9]
    # average = 0.0
    # for j in j_list:
    #     key = 'IoU@{}'.format(j / 100)
    #     average += x[key].avg
    # average = average / len(j_list)
    #
    # # average = x['IoU@0.9'].avg
    # average *= 100.0
    # return average
    if isinstance(x['mAP'], AverageMeter):
        average = x['mAP'].avg
    else:
        average = x['mAP']
    average *= 100.0
    return average

def print_python_performance(x, data='CharadesSTA'):
    item = ''
    i_list = [1]
    # j_list = [7, 8, 9]

    item += '{}:  {:.2f}\n'.format('average_R1', get_average_performance(x, data))
    item += '{}:        {:.2f}\n'.format('mIoU', x['mIoU'].avg * 100)
    for i in i_list:
        for j in j_list:
            key = 'IoU@{}'.format(j/100)
            item += '{}: {:.2f}  '.format(key, x[key].avg * 100)
        # if i == 1:
        #     item += '\n'
    return  item

def merge_ResultSaveObj(ResultSaveObj_list):
    ret = ResultSaveObj()
    ret.merge(ResultSaveObj_list)
    return ret

class ResultSaveObj(object):
    def __init__(self):
        self.save_flag = False
        self.video_info = collections.defaultdict(list)

    def merge(self, ResultSaveObj_list):
        for itm in ResultSaveObj_list:
            for k, v in itm.video_info.items():
                # self.video_info[k] += copy.deepcopy(v)
                self.video_info[k] += v
            # self.video_info += copy.deepcopy(itm.video_info)

    def to_numpy(self, x):
        return x.detach().cpu().numpy()

    def add_batch(self, save_result):
        # if not self.save_flag:
        #     return
        for k, v in save_result.items():
            self.video_info[k].append(v)

    def eval(self):
        gt_time = self.video_info['gt_time']
        pred_time = self.video_info['pred_time']
        score = self.video_info['score']
        # embed()
        metric_dict_1 = compute_AP_AR(
            pred_time, gt_time, score,
            iou_thresholds_ap=np.array(IOU_THRESHOLDS), iou_thresholds_ar=np.array(AR_IOU_THRESHOLDS), ar_points=MAX_DETECTIONS
        )
        return metric_dict_1


def fix_random_seed(seed, include_cuda=True):
    rng_generator = torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if include_cuda:
        # training: disable cudnn benchmark to ensure the reproducibility
        cudnn.enabled = True
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # this is needed for CUDA >= 10.2
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
    else:
        cudnn.enabled = True
        cudnn.benchmark = True
    return rng_generator


# def save_checkpoint(state, is_best, file_folder,
def save_checkpoint(state, file_folder,
                    file_name='checkpoint.pth.tar'):
    pass

def print_model_params(model):
    for name, param in model.named_parameters():
        print(name, param.min().item(), param.max().item(), param.mean().item())
    return


def make_optimizer(model, optimizer_config):
    """create optimizer
    return a supported optimizer
    """
    # separate out all parameters that with / without weight decay
    # see https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L134
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d, MaskedConv1D)
    blacklist_weight_modules = (LayerNorm, torch.nn.GroupNorm)

    # loop over all modules / params
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            elif pn.endswith('scale') and isinstance(m, (Scale, AffineDropPath)):
                # corner case of our scale layer
                no_decay.add(fpn)
            elif pn.endswith('rel_pe'):
                # corner case for relative position encoding
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, \
        "parameters %s were not separated into either decay/no_decay set!" \
        % (str(param_dict.keys() - union_params), )

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": optimizer_config['weight_decay']},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]

    if optimizer_config["type"] == "SGD":
        optimizer = optim.SGD(
            optim_groups,
            lr=optimizer_config["learning_rate"],
            momentum=optimizer_config["momentum"]
        )
    elif optimizer_config["type"] == "AdamW":
        optimizer = optim.AdamW(
            optim_groups,
            lr=optimizer_config["learning_rate"]
        )
    else:
        raise TypeError("Unsupported optimizer!")

    return optimizer


def make_scheduler(
    optimizer,
    optimizer_config,
    num_iters_per_epoch,
    last_epoch=-1
):
    """create scheduler
    return a supported scheduler
    All scheduler returned by this function should step every iteration
    """
    if optimizer_config["warmup"]:
        max_epochs = optimizer_config["epochs"] + optimizer_config["warmup_epochs"]
        max_steps = max_epochs * num_iters_per_epoch

        # get warmup params
        warmup_epochs = optimizer_config["warmup_epochs"]
        warmup_steps = warmup_epochs * num_iters_per_epoch

        # with linear warmup: call our custom schedulers
        if optimizer_config["schedule_type"] == "cosine":
            # Cosine
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_steps,
                max_steps,
                last_epoch=last_epoch
            )

        elif optimizer_config["schedule_type"] == "multistep":
            # Multi step
            steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
            scheduler = LinearWarmupMultiStepLR(
                optimizer,
                warmup_steps,
                steps,
                gamma=optimizer_config["schedule_gamma"],
                last_epoch=last_epoch
            )
        else:
            raise TypeError("Unsupported scheduler!")

    else:
        max_epochs = optimizer_config["epochs"]
        max_steps = max_epochs * num_iters_per_epoch

        # without warmup: call default schedulers
        if optimizer_config["schedule_type"] == "cosine":
            # step per iteration
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                max_steps,
                last_epoch=last_epoch
            )

        elif optimizer_config["schedule_type"] == "multistep":
            # step every some epochs
            steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                steps,
                gamma=schedule_config["gamma"],
                last_epoch=last_epoch
            )
        else:
            raise TypeError("Unsupported scheduler!")

    return scheduler


class AverageMeter(object):
    """Computes and stores the average and current value.
    Used to compute dataset stats from mini-batches
    """
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = 0.0

    def initialize(self, val, n):
        self.val = val
        self.avg = val
        self.sum = val * n
        self.count = n
        self.initialized = True

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.999, device=None):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


################################################################################
def train_one_epoch(
    train_loader,
    model,
    optimizer,
    scheduler,
    curr_epoch,
    model_ema = None,
    clip_grad_l2norm = -1,
    tb_writer = None,
    print_freq = 20
):
    """Training the model for one epoch"""
    # set up meters
    batch_time = AverageMeter()
    losses_tracker = {}
    # number of iterations per epoch
    num_iters = len(train_loader)
    # switch to train mode
    model.train()

    # main training loop
    print()
    print('='*100)
    print("[Train]: Epoch {:d} started".format(curr_epoch))
    print('=' * 100)
    start = time.time()
    for iter_idx, video_list in enumerate(train_loader, 0):
        # zero out optim
        optimizer.zero_grad(set_to_none=True)
        # forward / backward the model
        losses = model(video_list)
        losses['final_loss'].backward()
        # gradient cliping (to stabilize training if necessary)
        if clip_grad_l2norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                clip_grad_l2norm
            )
        # step optimizer / scheduler
        optimizer.step()
        scheduler.step()

        if model_ema is not None:
            model_ema.update(model)

        # printing (only check the stats when necessary to avoid extra cost)
        if (iter_idx != 0) and (iter_idx % print_freq) == 0:
            # measure elapsed time (sync all kernels)
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            # track all losses
            for key, value in losses.items():
                # init meter if necessary
                if key not in losses_tracker:
                    losses_tracker[key] = AverageMeter()
                # update
                losses_tracker[key].update(value.item())

            # log to tensor board
            lr = scheduler.get_last_lr()[0]
            global_step = curr_epoch * num_iters + iter_idx
            if tb_writer is not None:
                # learning rate (after stepping)
                tb_writer.add_scalar(
                    'train/learning_rate',
                    lr,
                    global_step
                )
                # all losses
                tag_dict = {}
                for key, value in losses_tracker.items():
                    if key != "final_loss":
                        tag_dict[key] = value.val
                tb_writer.add_scalars(
                    'train/all_losses',
                    tag_dict,
                    global_step
                )
                # final loss
                tb_writer.add_scalar(
                    'train/final_loss',
                    losses_tracker['final_loss'].val,
                    global_step
                )

            # print to terminal
            block1 = 'Epoch: [{:03d}][{:05d}/{:05d}]'.format(
                curr_epoch, iter_idx, num_iters
            )
            block2 = 'Time {:.1f} ({:.1f})'.format(
                batch_time.val, batch_time.avg
            )
            block3 = 'Loss {:.3f} ({:.3f})'.format(
                losses_tracker['final_loss'].val,
                losses_tracker['final_loss'].avg
            )
            block4 = ''
            for key, value in losses_tracker.items():
                if key != "final_loss":
                    block4  += '\t{:s} {:.3f} ({:.3f})'.format(
                        key, value.val, value.avg
                    )

            print(f'{block1} {block2}\n{block3} {block4}\n')

    # finish up and print
    lr = scheduler.get_last_lr()[0]
    print("\n[Train]: Epoch {:d} finished with lr={:.8f}\n\n".format(curr_epoch, lr))
    return


def valid_one_epoch(
    val_loader,
    model,
    curr_epoch,
    ext_score_file = None,
    evaluator = None,
    output_file = None,
    tb_writer = None,
    print_freq = 20,
):
    """Test the model on the validation set"""
    # either evaluate the results or save the results
    # assert (evaluator is not None) or (output_file is not None)

    # set up meters
    batch_time = AverageMeter()
    # switch to evaluate mode
    model.eval()
    # dict for results (for our evaluation code)
    results = {
        'video-id': [],
        't-start' : [],
        't-end': [],
        'label': [],
        'score': []
    }

    # loop over validation set
    start = time.time()
    acc_metrics_logger = collections.defaultdict(lambda: AverageMeter())
    result_save_obj_dict = {}

    result_save_obj = ResultSaveObj()
    iter_list = enumerate(val_loader, 0)
    batch_time = AverageMeter()

    for iter_idx, video_list in iter_list:
        # forward the model (wo. grad)
        with torch.no_grad():
            output_with_pred_list = model(video_list)

            # unpack the results into ANet format
            local_weight_list = [0.0]
            output_with_pred_list = [output_with_pred_list]
            local_weight_id = -1
            for output, local_weight in zip(output_with_pred_list, local_weight_list):
                local_weight_id += 1
                if local_weight not in result_save_obj_dict:
                    result_save_obj = ResultSaveObj()
                    result_save_obj_dict[local_weight] = result_save_obj
                else:
                    result_save_obj = result_save_obj_dict[local_weight]
                num_vids = len(output)
                for vid_idx in range(num_vids):
                    if output[vid_idx]['segments'].shape[0] > 0:
                        results['video-id'].extend(
                            [output[vid_idx]['video_id']] *
                            output[vid_idx]['segments'].shape[0]
                        )
                        results['t-start'].append(output[vid_idx]['segments'][:, 0])
                        results['t-end'].append(output[vid_idx]['segments'][:, 1])
                        results['label'].append(output[vid_idx]['labels'])
                        results['score'].append(output[vid_idx]['scores'])

                        pred_add = output[vid_idx]['segments'].detach().cpu().numpy() * output[vid_idx]['duration'] / val_loader.dataset.num_frames
                        gt_add =  np.array(output[vid_idx]['gt_time'])
                        score_add = output[vid_idx]['scores'].detach().cpu().numpy()
                        save_dict = {}
                        save_dict['gt_time'] = gt_add
                        save_dict['pred_time'] = pred_add
                        save_dict['score'] = score_add
                        result_save_obj.add_batch(save_dict)
        if (iter_idx != 0) and iter_idx % (print_freq) == 0:
            # measure elapsed time (sync all kernels)
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            # print timing
            print('Test: [{0:05d}/{1:05d}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})'.format(
                  iter_idx, len(val_loader), batch_time=batch_time))

    # gather all stats and evaluate
    results['t-start'] = torch.cat(results['t-start']).numpy()
    results['t-end'] = torch.cat(results['t-end']).numpy()
    results['label'] = torch.cat(results['label']).numpy()
    results['score'] = torch.cat(results['score']).numpy()

    return None, acc_metrics_logger, result_save_obj_dict