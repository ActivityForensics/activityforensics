# python imports
import argparse
import os
import time
import datetime
import yaml
import json
from pprint import pprint

# torch imports
import torch

import torch.nn as nn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
# for visualization
# from torch.utils.tensorboard import SummaryWriter

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import (train_one_epoch, valid_one_epoch, ANETdetection,
                        save_checkpoint, make_optimizer, make_scheduler,
                        fix_random_seed, ModelEma, display_python_performance, get_average_performance, merge_ResultSaveObj)
import itertools
import collections
from IPython import embed

def load_json(filename):
    with open(filename, encoding='utf8') as fr:
        return json.load(fr)

from terminaltables import AsciiTable

################################################################################
def main(args):
    """main function that handles training / inference"""

    """1. setup parameters / folders"""
    # parse args
    args.start_epoch = 0
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    # pprint(cfg)

    # tensorboard writer
    tb_writer = None

    # fix the random seeds (this will fix everything)
    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    # re-scale learning rate / # workers based on number of GPUs
    cfg['opt']["learning_rate"] *= len(cfg['devices'])
    cfg['loader']['num_workers'] *= len(cfg['devices'])

    cfg['dataset']['max_seq_len'] = cfg['dataset']['num_frames']
    cfg['save_root'] = os.path.join('model_ckpt')
    """2. create dataset / dataloader"""
    train_dataset = make_dataset(
        cfg['dataset_name'], True, cfg['train_split_list'], **cfg['dataset']
    )
    # update cfg based on dataset attributes (fix to epic-kitchens)
    # train_db_vars = train_dataset.get_attributes()
    cfg['model']['train_cfg']['head_empty_cls'] = []

    # data loaders
    train_loader = make_data_loader(
        train_dataset, True, rng_generator, **cfg['loader'])
    """2. create dataset / dataloader"""
    val_dataset_list = []
    val_loader_list = []

    for val_split in cfg['val_split_list']:
        val_dataset = make_dataset(
            cfg['dataset_name'], False, val_split, **cfg['dataset']
        )
        val_loader = make_data_loader(
            val_dataset, False, None, 1, cfg['loader']['num_workers']
        )
        val_dataset_list.append(val_dataset)
        val_loader_list.append(val_loader)

    """3. create model, optimizer, and scheduler"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # model.load_state_dict(torch.load(os.path.join(cfg['save_root'], '001.pth')))
    # embed()
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])
    # optimizer
    optimizer = make_optimizer(model, cfg['opt'])
    # schedule
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    """4. Resume from model / Misc"""

    args.print_freq = 100
    det_eval, output_file = None, None
    """5. Test the model"""

    """4. training / validation loop"""
    print("\nStart training model {:s} ...".format(cfg['model_name']))

    # start training
    max_epochs = cfg['opt'].get(
        'early_stop_epochs',
        cfg['opt']['epochs'] + cfg['opt']['warmup_epochs']
    )

    model_ema = None
    new_best_per_split = None
    cfg['train_split'] = cfg['train_split_list'][0]
    cfg['test_split_list'] = cfg['val_split_list']
    for epoch in range(args.start_epoch, max_epochs):
        # train for one epoch
        args.print_freq = 50
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            model_ema = model_ema,
            clip_grad_l2norm = cfg['train_cfg']['clip_grad_l2norm'],
            tb_writer=tb_writer,
            print_freq=args.print_freq
        )

        if (epoch % cfg['opt']['valid_epoch'] != 0 or epoch < cfg['opt']['start_test_epoch']) and epoch != max_epochs - 1:
            continue
        args.print_freq = 2000
        print('=' * 100)
        print(f'[Test]: Epoch {epoch} started')
        print('=' * 100)
        split_results_dict = {tmp_k: [] for tmp_k in cfg['val_split_list']}
        split_results_obj_dict = {tmp_k: [] for tmp_k in cfg['val_split_list']}
        for val_split, val_loader in zip(cfg['val_split_list'], val_loader_list):
            split_output_file = output_file
            _, acc_results, result_save_obj_dict = valid_one_epoch(
                val_loader,
                model,
                -1,
                evaluator=det_eval,
                output_file=split_output_file,
                ext_score_file=cfg['test_cfg']['ext_score_file'],
                tb_writer=None,
                print_freq=args.print_freq,
            )


            # 计算平均性能指标
            for local_weight in result_save_obj_dict:

                val_results_obj = result_save_obj_dict[local_weight]
                split_results_obj_dict[val_split].append(val_results_obj)


        merge_keys = cfg['val_split_list']
        new_split_results_dict = collections.defaultdict(list)

        in_domain = [tmp_itm.replace('train', 'test') for tmp_itm in cfg['train_split_list'] if 'real' not in tmp_itm]
        out_domain_2 = [tmp_itm for tmp_itm in cfg['test_split_list'] if tmp_itm not in in_domain]

        domain_name_list = ['in_domain', 'out_domain']
        tqdm_list = [in_domain, out_domain_2]

        domain_name_id = -1
        start_add = len(split_results_obj_dict)
        for merge_combo in tqdm_list:
            domain_name_id += 1
            if len(merge_combo) <= 0:
                continue
            merge_key_name = "+".join(merge_combo)
            merge_result_list = []
            # embed()
            for merge_idx in range(len(split_results_obj_dict[merge_combo[0]])):
                merge_objs = [split_results_obj_dict[k][merge_idx] for k in merge_combo]
                merge_obj = merge_ResultSaveObj(merge_objs)
                merge_result_list.append(merge_obj)
            split_results_obj_dict[domain_name_list[domain_name_id]+f' ({merge_key_name})'] = merge_result_list


        tqdm_list = tqdm(split_results_obj_dict.items())
        start_id = -1
        for merge_k, merge_v_list in tqdm_list:
            start_id += 1
            if start_id < start_add and cfg['test_cfg']['skip_separate_flag']:
                continue
            for merge_v in merge_v_list:
                new_split_results_dict[merge_k].append(merge_v.eval())

        # embed()
        for test_split_key in new_split_results_dict:
            if 'in_domain ' in test_split_key:
                break
        assert 'in_domain ' in test_split_key
        for test_split_key_assist in new_split_results_dict:
            if 'out_domain ' in test_split_key_assist:
                break
        assert 'out_domain ' in test_split_key_assist

        if new_best_per_split is None:
            new_best_per_split = {
                val_split: {
                    "best_avg": float("-inf"),
                    "best_epoch": None,
                    "best_local_weight": None,
                    "best_results": None,
                }
                for val_split in new_split_results_dict
            }

        local_weight_list = list(result_save_obj_dict.keys())
        print('='*100)
        num_train_samples = len(train_dataset)
        print(f"Current Validation Results | Epoch {epoch} | Trained on {cfg['train_split_list']} ({num_train_samples} samples)")
        print('=' * 100)
        for merge_k, merge_v_list in new_split_results_dict.items():
            for merge_v, local_weight in zip(merge_v_list, local_weight_list):
                avg_perf = get_average_performance(merge_v)
                print(f"Results for {merge_k}: avg={avg_perf:.4f} | epoch {epoch} | local_weight {local_weight}")
                print(display_python_performance(merge_v))
                if avg_perf > new_best_per_split[merge_k]["best_avg"]:
                    new_best_per_split[merge_k]["best_avg"] = avg_perf
                    new_best_per_split[merge_k]["best_epoch"] = epoch
                    new_best_per_split[merge_k]["best_local_weight"] = local_weight
                    new_best_per_split[merge_k]["best_results"] = merge_v
                    # print(f"Update best results for {merge_k}: avg={avg_perf:.4f} | epoch {epoch} | local_weight {local_weight}")
                    print(f"Update best results")
                print()

        # print('='*100)
        num_train_samples = len(train_dataset)
        print('='*100)
        print(f"Best Validation Results | Epoch {epoch} | Trained on {cfg['train_split_list']} ({num_train_samples} samples)")
        print('='*100)
        for val_split in new_best_per_split:
            rec = new_best_per_split[val_split]
            print(f"Best for {val_split}:\nR1 = {rec['best_avg']:.4f}\nepoch {rec['best_epoch']} | local_weight {rec['best_local_weight']}")
            print(display_python_performance(rec["best_results"]))
            print()

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('--config', metavar='DIR',
                        help='path to a config file')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    parser.add_argument('-c', '--ckpt-freq', default=5, type=int,
                        help='checkpoint frequency (default: every 5 epochs)')
    parser.add_argument('--output', default='', type=str,
                        help='name of exp folder (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to a checkpoint (default: none)')
    parser.add_argument('--tag', default='baseline', type=str, help='experiment tag')
    args = parser.parse_args()

    main(args)
