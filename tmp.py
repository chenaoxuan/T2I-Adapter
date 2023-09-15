import cv2
import torch
import os

from basicsr.utils import img2tensor, tensor2img, scandir, get_time_str, get_root_logger, get_env_info
from ldm.data.dataset_subject import dataset_replay, single_data
import argparse
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.modules.encoders.adapter import Adapter
from PIL import Image
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from basicsr.utils.options import copy_opt_file, dict2str
import logging
from dist_util import init_dist, master_only, get_bare_model, get_dist_info
from ldm.util import load_model_from_config


@master_only
def mkdir_and_rename(path):
    """mkdirs. If path exists, rename it with timestamp and create a new one.

    Args:
        path (str): Folder path.
    """
    if os.path.exists(path):
        new_name = path + '_archived_' + get_time_str()
        print(f'Path already exists. Rename it to {new_name}', flush=True)
        os.rename(path, new_name)
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(experiments_root, 'models'))
    os.makedirs(os.path.join(experiments_root, 'training_states'))
    os.makedirs(os.path.join(experiments_root, 'visualization'))


def load_resume_state(opt):
    resume_state_path = None
    if opt.auto_resume:
        state_path = os.path.join('experiments', opt.name, 'training_states')
        if os.path.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = os.path.join(state_path, f'{max(states):.0f}.state')
                opt.resume_state_path = resume_state_path
                resume_ad_path = os.path.join('experiments', opt.name, 'models', f'model_ad_{max(states):.0f}.pth')
                opt.resume_ad_path = resume_ad_path

    # else:
    #     if opt['path'].get('resume_state'):
    #         resume_state_path = opt['path']['resume_state']

    if resume_state_path is None:
        resume_state = None
        resume_ad = None
    else:
        device_id = torch.cuda.current_device()
        resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))
        resume_ad = torch.load(resume_ad_path, map_location=lambda storage, loc: storage.cuda(device_id))
        # check_resume(opt, resume_state['iter'])
    return resume_state, resume_ad


def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="the prompt to render"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="the prompt to render"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="the prompt to render"
    )
    parser.add_argument(
        "--use_shuffle",
        type=bool,
        default=True,
        help="the prompt to render"
    )
    parser.add_argument(
        "--dpm_solver",
        action='store_true',
        help="use dpm_solver sampling",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--auto_resume",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="ckp/sd-v1-4.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/train_subj.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--print_fq",
        type=int,
        default=5,
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    # parser.add_argument(
    #     "--gpus",
    #     default=[0, 1, 2, 3],
    #     help="gpu idx",
    # )
    parser.add_argument(
        '--local_rank',
        default=0,
        type=int,
        help='node rank for distributed training'
    )
    parser.add_argument(
        '--launcher',
        default='pytorch',
        type=str,
        help='node rank for distributed training'
    )
    parser.add_argument(
        "--distributed",
        action='store_true',
        help="distributed training on Linux",
    )
    parser.add_argument(
        "--gpu",
        action='store_true',
        help="training on gpu",
    )
    return parser.parse_args()


if __name__ == '__main__':

    opt = get_parse()
    config = OmegaConf.load(f"{opt.config}")
    opt.name = config['name']
    device = torch.device("cuda") if torch.cuda.is_available() and opt.gpu else torch.device("cpu")

    # distributed setting
    if opt.distributed:
        init_dist(opt.launcher)
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(opt.local_rank)

    # stable diffusion
    model = load_model_from_config(config, f"{opt.ckpt}")

    # to gpus
    if opt.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank)
    else:
        model = model.to(device)

    experiments_root = os.path.join('experiments', opt.name)

    # resume state
    resume_state, resume_ad = load_resume_state(opt)
    if resume_state is None:
        mkdir_and_rename(experiments_root)
        start_epoch = 0
        current_iter = 0
        start_data = 1
        # WARNING: should not use get_root_logger in the above codes, including the called functions
        # Otherwise the logger will not be properly initialized
        log_file = os.path.join(experiments_root, f"train_{opt.name}_{get_time_str()}.log")
        logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
        logger.info(get_env_info())
        logger.info(dict2str(config))
    else:
        # WARNING: should not use get_root_logger in the above codes, including the called functions
        # Otherwise the logger will not be properly initialized
        log_file = os.path.join(experiments_root, f"train_{opt.name}_{get_time_str()}.log")
        logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
        logger.info(get_env_info())
        logger.info(dict2str(config))
        resume_optimizers = resume_state['optimizers']
        # optimizer.load_state_dict(resume_optimizers)
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, " f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
        start_data = resume_state['data']

    # copy the yml file to the experiment root
    copy_opt_file(opt.config, experiments_root)
    save_data_dict = {}
    # training
    logger.info(f'Start training from data:{start_data}, epoch: {start_epoch}, iter: {current_iter}')
    for now_data in range(start_data, int(config.dataset.end_data) + 1):
        single_d = single_data(now_data)
        train_dataset = dataset_replay(
            root_path=config['dataset']['root_path'],
            now_task=str(now_data),
            iftrain=True,
            image_size=512
        )
        if opt.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None
        val_dataset = dataset_replay(
            root_path=config['dataset']['root_path'],
            now_task=str(now_data),
            iftrain=False,
            image_size=512
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.num_workers,
            pin_memory=True,
            sampler=train_sampler)
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=False)
        # optimizer
        params = list(model.model.diffusion_model.adapter.parameters())
        optimizer = torch.optim.AdamW(params, lr=config['training']['lr'])
        for epoch in range(start_epoch, opt.epochs):
            if opt.distributed:
                train_dataloader.sampler.set_epoch(epoch)
            # train saved data
            for pre_data_idx, pre_data in save_data_dict.items():
                for idx in len(train_dataloader):
                    t, noise, z, c = pre_data.get_simple_data()
                    current_iter += 1
                    optimizer.zero_grad()
                    model.zero_grad()
                    l_pixel, loss_dict = model(z, c=c, t=t, noise=noise, features_adapter=True, pre_data=True)
                    l_pixel.backward()
                    optimizer.step()
                    if (current_iter + 1) % opt.print_fq == 0:
                        logger.info(f"Data:{pre_data_idx}, Epoch:{epoch}")
                        for t in loss_dict:
                            loss_dict[t] = round(loss_dict[t].item(), 6)
                        logger.info(loss_dict)

            # train
            for _, data in enumerate(train_dataloader):
                current_iter += 1
                with torch.no_grad():
                    if opt.distributed:
                        c = model.module.get_learned_conditioning(data['sentence'])
                        z = model.module.encode_first_stage((data['im'] * 2 - 1.).to(device))
                        z = model.module.get_first_stage_encoding(z)
                    else:
                        c = model.get_learned_conditioning(data['sentence'])
                        z = model.encode_first_stage((data['im'] * 2 - 1.).to(device))
                        z = model.get_first_stage_encoding(z)

                optimizer.zero_grad()
                model.zero_grad()
                l_pixel, loss_dict, save_data = model(z, c=c, features_adapter=True)
                single_d.add_data(epoch, l_pixel.item(), save_data)
                l_pixel.backward()
                optimizer.step()

                if (current_iter + 1) % opt.print_fq == 0:
                    logger.info(f"Data:{now_data}, Epoch:{epoch}")
                    for t in loss_dict:
                        loss_dict[t] = round(loss_dict[t].item(), 6)
                    logger.info(loss_dict)
                    print(save_data)

                # save checkpoint
                if opt.distributed:
                    rank, _ = get_dist_info()
                else:
                    rank = 0
                # if (rank == 0) and ((epoch + 1) % config['training']['save_freq_epoch'] == 0):
                #     save_filename = f'model_ad_{epoch + 1}.pth'
                #     save_path = os.path.join(experiments_root, 'models', save_filename)
                #     save_dict = {}
                # model_ad_bare = get_bare_model(model_ad)
                # state_dict = model_ad_bare.state_dict()
                # for key, param in state_dict.items():
                #     if key.startswith('module.'):  # remove unnecessary 'module.'
                #         key = key[7:]
                #     save_dict[key] = param.cpu()
                # torch.save(save_dict, save_path)
                # # save state
                # state = {'data': now_data, 'epoch': epoch, 'iter': current_iter + 1,
                #          'optimizers': optimizer.state_dict()}
                # save_filename = f'{epoch + 1}.state'
                # save_path = os.path.join(experiments_root, 'training_states', save_filename)
                # torch.save(state, save_path)

            # val
            if opt.distributed:
                rank, _ = get_dist_info()
            else:
                rank = 0
            if rank == 0 and (epoch + 1) % config['training']['val_freq_epoch'] == 0:
                with torch.no_grad():
                    if opt.dpm_solver:
                        sampler = DPMSolverSampler(model)
                    elif opt.plms:
                        sampler = PLMSSampler(model)
                    else:
                        sampler = DDIMSampler(model)
                    for d_idx, data in enumerate(val_dataloader):
                        for v_idx in range(opt.n_samples):
                            c = model.get_learned_conditioning(data['sentence'])
                            shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                            samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                             conditioning=c,
                                                             batch_size=1,
                                                             shape=shape,
                                                             verbose=False,
                                                             unconditional_guidance_scale=opt.scale,
                                                             unconditional_conditioning=model.get_learned_conditioning(
                                                                 [""]),
                                                             eta=opt.ddim_eta,
                                                             x_T=None)
                            x_samples_ddim = model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                            for id_sample, x_sample in enumerate(x_samples_ddim):
                                x_sample = 255. * x_sample
                                img = x_sample.astype(np.uint8)
                                img = cv2.putText(img.copy(), data['sentence'][0], (10, 30),
                                                  cv2.FONT_HERSHEY_SIMPLEX,
                                                  0.5,
                                                  (0, 255, 0), 2)
                                cv2.imwrite(os.path.join(experiments_root, 'visualization',
                                                         'sample_d%03d_e%04d_v%02d_s%04d.png' % (
                                                             now_data, epoch, d_idx, v_idx)),
                                            img[:, :, ::-1])
        save_data_dict[now_data] = single_d
