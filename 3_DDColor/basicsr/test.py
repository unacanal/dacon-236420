import datetime
import logging
import math
import time
import torch
import warnings

warnings.filterwarnings("ignore")

from os import path as osp

from basicsr.data import build_dataloader, build_dataset
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import build_model
from basicsr.utils import (AvgTimer, MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str,
                           init_tb_logger, init_wandb_logger, make_exp_dirs, mkdir_and_rename, scandir)
from basicsr.utils.options import copy_opt_file, dict2str, parse_options


def init_tb_loggers(opt):
    # initialize wandb logger before tensorboard logger to allow proper sync
    if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project')
                                                     is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, ('should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join(opt['root_path'], 'tb_logger', opt['name']))
    return tb_logger


def create_val_dataloader(opt, logger):
    # create val dataloaders
    val_loaders = []
    for phase, dataset_opt in opt['datasets'].items():
        if phase.split('_')[0] == 'val':
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            logger.info(f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
            val_loaders.append(val_loader)
    return val_loaders


def test_pipeline(root_path):
    # parse options, set distributed setting, set random seed
    opt, args = parse_options(root_path, is_train=False)
    opt['root_path'] = root_path

    torch.backends.cudnn.benchmark = True

    # mkdir for experiments and logger
    make_exp_dirs(opt)
    
    # WARNING: should not use get_root_logger in the above codes
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # initialize wandb and tb loggers
    tb_logger = init_tb_loggers(opt)

    # create validation dataloaders
    val_loaders = create_val_dataloader(opt, logger)

    # create model
    model = build_model(opt)
    
    # validation
    logger.info('Begin validation...')
    for val_loader in val_loaders:
        model.validation(val_loader, current_iter=0, tb_logger=tb_logger, save_img=opt['val']['save_img'])

    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
