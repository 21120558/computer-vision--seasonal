import os
import sys
import glob
import yaml
import random
import numpy as np
import cv2
import torch
import zipfile

def load_yaml(path):
    print(f"Loading configs from {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def setup_runtime(args):
    # Setup CUDA
    cuda_device_id = args.gpu
    if cuda_device_id is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device_id)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    # Setup random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ## Load config
    cfgs = {}
    if args.config is not None and os.path.isfile(args.config):
        cfgs = load_yaml(args.config)

    cfgs['config'] = args.config
    cfgs['seed'] = args.seed
    cfgs['num_workers'] = args.num_workers
    cfgs['device'] = 'cuda:0' if torch.cuda.is_available() and cuda_device_id is not None else 'cpu'

    print(f"Environment: GPU {cuda_device_id} seed {args.seed} number of workers {args.num_workers}")
    return cfgs

def mkdir(path):
    """Create directory PATH recursively if it does not exist."""
    os.makedirs(path, exist_ok=True)

def save_images(out_fold, imgs, prefix='', suffix='', sep_folder=True, ext='.png'):
    if sep_folder:
        out_fold = os.path.join(out_fold, suffix)
    mkdir(out_fold)
    prefix = prefix + '_' if prefix else ''
    suffix = '_' + suffix if suffix else ''
    offset = len(glob.glob(os.path.join(out_fold, prefix+'*'+suffix+ext))) +1

    imgs = imgs.transpose(0,2,3,1)
    for i, img in enumerate(imgs):
        if 'depth' in suffix:
            im_out = np.uint16(img[...,::-1]*65535.)
        else:
            im_out = np.uint8(img[...,::-1]*255.)
        cv2.imwrite(os.path.join(out_fold, prefix+'%05d'%(i+offset)+suffix+ext), im_out)


def save_images_fake(out_fold, img, prefix='', suffix='', sep_folder=True, ext='.png'):
    if sep_folder:
        out_fold = os.path.join(out_fold, suffix)
    mkdir(out_fold)
    img = img.transpose(1,2,0)
    im_out = np.uint8(img[...,::-1]*255.)
    cv2.imwrite(os.path.join(out_fold, "hahahha" +ext), im_out)

def clean_checkpoint(checkpoint_dir, keep_num=2):
    if keep_num > 0:
        names = list(sorted(
            glob.glob(os.path.join(checkpoint_dir, 'checkpoint*.pth'))
        ))
        if len(names) > keep_num:
            for name in names[:-keep_num]:
                print(f"Deleting obslete checkpoint file {name}")
                os.remove(name)

def save_scores(out_path, scores, header=''):
    print('Saving scores to %s' %out_path)
    np.savetxt(out_path, scores, fmt='%.8f', delimiter=',\t', header=header)

def save_videos(out_fold, imgs, prefix='', suffix='', sep_folder=True, ext='.mp4', cycle=False):
    if sep_folder:
        out_fold = os.path.join(out_fold, suffix)
    mkdir(out_fold)
    prefix = prefix + '_' if prefix else ''
    suffix = '_' + suffix if suffix else ''
    offset = len(glob.glob(os.path.join(out_fold, prefix+'*'+suffix+ext))) +1

    imgs = imgs.transpose(0,1,3,4,2)  # BxTxCxHxW -> BxTxHxWxC
    for i, fs in enumerate(imgs):
        if cycle:
            fs = np.concatenate([fs, fs[::-1]], 0)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vid = cv2.VideoWriter(os.path.join(out_fold, prefix+'%05d'%(i+offset)+suffix+ext), fourcc, 5, (fs.shape[2], fs.shape[1]))
        [vid.write(np.uint8(f[...,::-1]*255.)) for f in fs]
        vid.release()