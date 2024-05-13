import os
import math
import glob
import torch
import torch.nn as nn
import torchvision
from .renderer import Renderer
from seasonal.networks import ConfNet, EDDeconv, Encoder, PerceptualLoss
from . import utils

EPS = 1e-7

class Model():
    def __init__(self, configs):
        self.model_name = configs.get('model_name', self.__class__.__name__)
        self.image_size = configs.get('image_size', 64)
        self.min_depth = configs.get('min_depth', 0.9)
        self.max_depth = configs.get('max_depth', 1.1)
        self.border_depth = configs.get('border_depth', (0.7 * self.max_depth + 0.3 * self.min_depth))
        self.min_amb_light = configs.get('min_amb_light', 0.)
        self.max_amb_light = configs.get('max_amb_light', 1.)
        self.min_diff_light = configs.get('min_diff_light', 0.)
        self.max_diff_light = configs.get('max_diff_light', 1.)
        self.xyz_rotation_range = configs.get('xyz_rotation_range', 60)
        self.xy_translation_range = configs.get('xy_translation_range', 0.1)
        self.z_translation_range = configs.get('z_translation_range', 0.1)
        self.use_conf_map = configs.get('use_conf_map', True)
        self.lam_perc = configs.get('lam_perc', 1)
        self.lam_flip = configs.get('lam_flip', 0.5)
        self.lam_flip_start_epoch = configs.get('lam_flip_start_epoch', 0)
        self.lam_depth_sm = configs.get('lam_depth_sm', 0)
        self.lr = configs.get('lr', 1e-4)
        self.load_gt_depth = configs.get('load_gt_depth', True)
        self.renderer = Renderer(configs)
        self.other_param_names = ['PerceptualLoss']

        self.depth_rescaler = lambda d : (1 + d) / 2 * self.max_depth + (1 - d) / 2 * self.min_depth
        self.amb_light_rescaler = lambda x : (1 + x)/2 * self.max_amb_light + (1 - x)/2 * self.min_amb_light
        self.diff_light_rescaler = lambda x : (1 + x)/2 * self.max_diff_light + (1 - x)/2 * self.min_diff_light

        self.init_networks()
        self.init_optimizer()


    def init_networks(self):
        self.netDepth = EDDeconv(channels_in=3, channels_out=1, num_filters=64, zdim=256)
        self.netAlbedo = EDDeconv(channels_in=3, channels_out=3, num_filters=64, zdim=256)
        self.netLight = Encoder(channels_in=3, channels_out=4, num_filters=32)
        self.netView = Encoder(channels_in=3, channels_out=6, num_filters=32)
        self.netConf = ConfNet(channels_in=3, channels_out=2, num_filters=64, zdim=128)
        self.PerceptualLoss = PerceptualLoss(requires_grad=False)

        self.network_names = [k for k in vars(self) if 'net' in k]

    def init_optimizer(self):
        make_optimizer = lambda model: torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.lr,
            betas=(0.9, 0.999),
            weight_decay=5e-4
        )
        self.optimizer_names = []
        for net_name in self.network_names:
            optimizer = make_optimizer(getattr(self, net_name))
            optim_name = net_name.replace('net', 'optimizer')
            setattr(self, optim_name, optimizer)
            self.optimizer_names += [optim_name]
    
    def load_model_state(self, cp):
        for k in cp:
            if k and k in self.network_names:
                getattr(self, k).load_state_dict(cp[k])

    def load_optimizer_state(self, cp):
        for k in cp:
            if k and k in self.optimizer_names:
                getattr(self, k).load_state_dict(cp[k])

    def get_model_state(self):
        states = {}
        for net_name in self.network_names:
            states[net_name] = getattr(self, net_name).state_dict()
        return states
    
    def to_device(self, device):
        self.device = device
        for net_name in self.network_names:
            setattr(self, net_name, getattr(self, net_name).to(device))
        if self.other_param_names:
            for param_name in self.other_param_names:
                setattr(self, param_name, getattr(self, param_name).to(device))

    def set_train(self):
        for net_name in self.network_names:
            getattr(self, net_name).train()

    def set_eval(self):
        for net_name in self.network_names:
            getattr(self, net_name).eval()

    def get_optimizer_state(self):
        states = {}
        for optim_name in self.optimizer_names:
            states[optim_name] = getattr(self, optim_name).state_dict()
        return states

    def photometric_loss(self, im1, im2, mask=None, conf_sigma=None):
        loss = (im1 - im2).abs()
        if conf_sigma is not None:
            loss = loss *2**0.5 / (conf_sigma + EPS) + (conf_sigma + EPS).log()
        if mask is not None:
            mask = mask.expand_as(loss)
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.mean()
        return loss

    def backward(self):
        for optim_name in self.optimizer_names:
            getattr(self, optim_name).zero_grad()
        self.loss_total.backward()
        for optim_name in self.optimizer_names:
            getattr(self, optim_name).step()

    def predict_canonical_depth(self):
        self.canon_depth_raw = self.netDepth(self.input_im).squeeze(1)  # BxHxW
        self.canon_depth = self.canon_depth_raw - self.canon_depth_raw.view(self.b, -1).mean(1).view(self. b,1,1)
        self.canon_depth = self.canon_depth.tanh()
        self.canon_depth = self.depth_rescaler(self.canon_depth)

    def depth_smoothness(self):
        self.loss_depth_sm = ((self.canon_depth[:,:-1,:] - self.canon_depth[:,1:,:]) /(self.max_depth-self.min_depth)).abs().mean()
        self.loss_depth_sm += ((self.canon_depth[:,:,:-1] - self.canon_depth[:,:,1:]) /(self.max_depth-self.min_depth)).abs().mean()

    def clamp_border_depth(self):
        depth_border = torch.zeros(1, self.h, self.w - 4).to(self.input_im.device)
        depth_border = nn.functional.pad(depth_border, (2, 2), mode='constant', value=1)
        self.canon_depth = self.canon_depth*(1 - depth_border) + depth_border * self.border_depth
        self.canon_depth = torch.cat([self.canon_depth, self.canon_depth.flip(2)], 0)  # flip

    def predict_canonical_albedo(self):
        self.canon_albedo = self.netAlbedo(self.input_im)  # Bx3xHxW
        self.canon_albedo = torch.cat([self.canon_albedo, self.canon_albedo.flip(3)], 0)  # flip

    def predict_confidence_map(self):
        conf_sigma_l1, conf_sigma_percl = self.netConf(self.input_im)  # Bx2xHxW
        self.conf_sigma_l1 = conf_sigma_l1[:, :1]
        self.conf_sigma_l1_flip = conf_sigma_l1[:, 1:]
        self.conf_sigma_percl = conf_sigma_percl[:, :1]
        self.conf_sigma_percl_flip = conf_sigma_percl[:, 1:]

    def predict_lighting(self):
        canon_light = self.netLight(self.input_im).repeat(2, 1)  # Bx4
        self.canon_light_a = self.amb_light_rescaler(canon_light[:, :1])  # ambience term
        self.canon_light_b = self.diff_light_rescaler(canon_light[:, 1:2])  # diffuse term
        canon_light_dxy = canon_light[:,2:]
        self.canon_light_d = torch.cat([canon_light_dxy, torch.ones(self.b * 2,1).to(self.input_im.device)], 1)
        self.canon_light_d = self.canon_light_d / ((self.canon_light_d**2).sum(1, keepdim=True)) ** 0.5  # diffuse light direction

    def shading(self):
        self.canon_normal = self.renderer.get_normal_from_depth(self.canon_depth)
        self.canon_diffuse_shading = (self.canon_normal * self.canon_light_d.view(-1, 1, 1, 3)).sum(3).clamp(min=0).unsqueeze(1)
        canon_shading = self.canon_light_a.view(-1, 1, 1, 1) + self.canon_light_b.view(-1, 1, 1, 1) * self.canon_diffuse_shading
        self.canon_im = (self.canon_albedo / 2 + 0.5) * canon_shading * 2 - 1

    def predict_view(self):
        self.view = self.netView(self.input_im).repeat(2,1)
        self.view = torch.cat([
            self.view[:,:3] * math.pi/180 * self.xyz_rotation_range,
            self.view[:,3:5] * self.xy_translation_range,
            self.view[:,5:] * self.z_translation_range
        ], 1)

    def reconstruct_view(self):
        self.renderer.set_transform_matrices(self.view)
        self.recon_depth = self.renderer.warp_canon_depth(self.canon_depth)
        self.recon_normal = self.renderer.get_normal_from_depth(self.recon_depth)
        self.grid_2d_from_canon = self.renderer.get_inv_warped_2d_grid(self.recon_depth)
        self.recon_im = nn.functional.grid_sample(self.canon_im, self.grid_2d_from_canon, mode='bilinear', align_corners=True)

        fake = self.recon_depth.clone()
        current_dir = os.path.dirname(os.path.realpath(__file__))
        fake = ((fake[:1] - self.min_depth)/(self.max_depth-self.min_depth)).clamp(0, 1).detach().cpu().unsqueeze(1).numpy()
        utils.save_images(current_dir, fake, suffix='recon_image', sep_folder=True)

        margin = (self.max_depth - self.min_depth) / 2
        recon_im_mask = (self.recon_depth < self.max_depth + margin).float()  # invalid border pixels have been clamped at max_depth+margin
        self.recon_im_mask_both = recon_im_mask[:self.b] * recon_im_mask[self.b:]  # both original and flip reconstruction
        self.recon_im_mask_both = self.recon_im_mask_both.repeat(2, 1, 1).unsqueeze(1).detach()
        self.recon_im = self.recon_im * self.recon_im_mask_both

    def render_symmetry_axis(self):
        canon_sym_axis = torch.zeros(self.h, self.w).to(self.input_im.device)
        canon_sym_axis[:, self.w // 2 - 1:self.w // 2 + 1] = 1
        self.recon_sym_axis = nn.functional.grid_sample(canon_sym_axis.repeat(self.b * 2, 1, 1, 1), self.grid_2d_from_canon, mode='bilinear', align_corners=False)
        self.recon_sym_axis = self.recon_sym_axis * self.recon_im_mask_both
        green = torch.FloatTensor([-1, 1, -1]).to(self.input_im.device).view(1, 3, 1, 1)
        self.input_im_symline = (0.5 * self.recon_sym_axis) * green + (1 - 0.5 * self.recon_sym_axis) *self.input_im.repeat(2, 1, 1, 1)

    def cal_loss_function(self):
        self.loss_l1_im = self.photometric_loss(self.recon_im[:self.b], self.input_im, mask=self.recon_im_mask_both[:self.b], conf_sigma=self.conf_sigma_l1)
        self.loss_l1_im_flip = self.photometric_loss(self.recon_im[self.b:], self.input_im, mask=self.recon_im_mask_both[self.b:], conf_sigma=self.conf_sigma_l1_flip)
        self.loss_perc_im = self.PerceptualLoss(self.recon_im[:self.b], self.input_im, mask=self.recon_im_mask_both[:self.b], conf_sigma=self.conf_sigma_percl)
        self.loss_perc_im_flip = self.PerceptualLoss(self.recon_im[self.b:], self.input_im, mask=self.recon_im_mask_both[self.b:], conf_sigma=self.conf_sigma_percl_flip)
        lam_flip = 1 if self.trainer.current_epoch < self.lam_flip_start_epoch else self.lam_flip
        self.loss_total = self.loss_l1_im + lam_flip * self.loss_l1_im_flip + self.lam_perc*(self.loss_perc_im + lam_flip*self.loss_perc_im_flip) + self.lam_depth_sm*self.loss_depth_sm 

    def forward(self, input):
        self.input_im = input.to(self.device) * 2.0 - 1.0
        self.b, self.c, self.h, self.w = self.input_im.shape

        self.predict_canonical_depth()
        self.depth_smoothness()
        self.clamp_border_depth()

        self.predict_canonical_albedo()
        self.predict_confidence_map()
        self.predict_lighting()

        self.shading()
        self.predict_view()
        self.reconstruct_view()
        self.render_symmetry_axis()
        
        self.cal_loss_function()

        metrics = {'loss': self.loss_total}
        return metrics 

    def save_results(self, save_dir):
        b, c, h, w = self.input_im.shape

        with torch.no_grad():
            v0 = torch.FloatTensor([-0.1*math.pi/180*60,0,0,0,0,0]).to(self.input_im.device).repeat(b,1)
            canon_im_rotate = self.renderer.render_yaw(self.canon_im[:b], self.canon_depth[:b], v_before=v0, maxr=90, nsample=15)  # (B,T,C,H,W)
            canon_im_rotate = canon_im_rotate.clamp(-1,1).detach().cpu() /2+0.5
            canon_normal_rotate = self.renderer.render_yaw(self.canon_normal[:b].permute(0,3,1,2), self.canon_depth[:b], v_before=v0, maxr=90, nsample=15)  # (B,T,C,H,W)
            canon_normal_rotate = canon_normal_rotate.clamp(-1,1).detach().cpu() /2+0.5

        input_im = self.input_im[:b].detach().cpu().numpy() /2+0.5
        input_im_symline = self.input_im_symline.detach().cpu().numpy() /2.+0.5
        canon_albedo = self.canon_albedo[:b].detach().cpu().numpy() /2+0.5
        canon_im = self.canon_im[:b].clamp(-1,1).detach().cpu().numpy() /2+0.5
        recon_im = self.recon_im[:b].clamp(-1,1).detach().cpu().numpy() /2+0.5
        recon_im_flip = self.recon_im[b:].clamp(-1,1).detach().cpu().numpy() /2+0.5
        canon_depth = ((self.canon_depth[:b] -self.min_depth)/(self.max_depth-self.min_depth)).clamp(0,1).detach().cpu().unsqueeze(1).numpy()
        recon_depth = ((self.recon_depth[:b] -self.min_depth)/(self.max_depth-self.min_depth)).clamp(0,1).detach().cpu().unsqueeze(1).numpy()
        canon_diffuse_shading = self.canon_diffuse_shading[:b].detach().cpu().numpy()
        canon_normal = self.canon_normal[:b].permute(0,3,1,2).detach().cpu().numpy() /2+0.5
        recon_normal = self.recon_normal[:b].permute(0,3,1,2).detach().cpu().numpy() /2+0.5
        if self.use_conf_map:
            conf_map_l1 = 1/(1+self.conf_sigma_l1[:b].detach().cpu().numpy()+EPS)
            conf_map_l1_flip = 1/(1+self.conf_sigma_l1_flip[:b].detach().cpu().numpy()+EPS)
            conf_map_percl = 1/(1+self.conf_sigma_percl[:b].detach().cpu().numpy()+EPS)
            conf_map_percl_flip = 1/(1+self.conf_sigma_percl_flip[:b].detach().cpu().numpy()+EPS)
        canon_light = torch.cat([self.canon_light_a, self.canon_light_b, self.canon_light_d], 1)[:b].detach().cpu().numpy()
        view = self.view[:b].detach().cpu().numpy()

        canon_im_rotate_grid = [torchvision.utils.make_grid(img, nrow=int(math.ceil(b**0.5))) for img in torch.unbind(canon_im_rotate,1)]  # [(C,H,W)]*T
        canon_im_rotate_grid = torch.stack(canon_im_rotate_grid, 0).unsqueeze(0).numpy()  # (1,T,C,H,W)
        canon_normal_rotate_grid = [torchvision.utils.make_grid(img, nrow=int(math.ceil(b**0.5))) for img in torch.unbind(canon_normal_rotate,1)]  # [(C,H,W)]*T
        canon_normal_rotate_grid = torch.stack(canon_normal_rotate_grid, 0).unsqueeze(0).numpy()  # (1,T,C,H,W)

        sep_folder = True
        utils.save_images(save_dir, input_im, suffix='input_image', sep_folder=sep_folder)
        utils.save_images(save_dir, input_im_symline, suffix='input_image_symline', sep_folder=sep_folder)
        utils.save_images(save_dir, canon_albedo, suffix='canonical_albedo', sep_folder=sep_folder)
        utils.save_images(save_dir, canon_im, suffix='canonical_image', sep_folder=sep_folder)
        utils.save_images(save_dir, recon_im, suffix='recon_image', sep_folder=sep_folder)
        utils.save_images(save_dir, recon_im_flip, suffix='recon_image_flip', sep_folder=sep_folder)
        utils.save_images(save_dir, canon_depth, suffix='canonical_depth', sep_folder=sep_folder)
        utils.save_images(save_dir, recon_depth, suffix='recon_depth', sep_folder=sep_folder)
        utils.save_images(save_dir, canon_diffuse_shading, suffix='canonical_diffuse_shading', sep_folder=sep_folder)
        utils.save_images(save_dir, canon_normal, suffix='canonical_normal', sep_folder=sep_folder)
        utils.save_images(save_dir, recon_normal, suffix='recon_normal', sep_folder=sep_folder)
        if self.use_conf_map:
            utils.save_images(save_dir, conf_map_l1, suffix='conf_map_l1', sep_folder=sep_folder)
            utils.save_images(save_dir, conf_map_l1_flip, suffix='conf_map_l1_flip', sep_folder=sep_folder)
            utils.save_images(save_dir, conf_map_percl, suffix='conf_map_percl', sep_folder=sep_folder)
            utils.save_images(save_dir, conf_map_percl_flip, suffix='conf_map_percl_flip', sep_folder=sep_folder)
        # utils.save_txt(save_dir, canon_light, suffix='canonical_light', sep_folder=sep_folder)
        # utils.save_txt(save_dir, view, suffix='viewpoint', sep_folder=sep_folder)

        utils.save_videos(save_dir, canon_im_rotate_grid, suffix='image_video', sep_folder=sep_folder, cycle=True)
        utils.save_videos(save_dir, canon_normal_rotate_grid, suffix='normal_video', sep_folder=sep_folder, cycle=True)

    def save_scores(self, path):
        # save scores if gt is loaded
        if self.load_gt_depth:
            header = 'MAE_masked, \
                      MSE_masked, \
                      SIE_masked, \
                      NorErr_masked'
            mean = self.all_scores.mean(0)
            std = self.all_scores.std(0)
            header = header + '\nMean: ' + ',\t'.join(['%.8f'%x for x in mean])
            header = header + '\nStd: ' + ',\t'.join(['%.8f'%x for x in std])
            utils.save_scores(path, self.all_scores, header=header)


        