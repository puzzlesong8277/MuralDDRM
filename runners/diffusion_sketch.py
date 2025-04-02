from email.mime import image
import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data

from PIL import Image
from torchvision import transforms

from models.diffusion import Model
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path, download
from functions.denoising import efficient_generalized_steps
from color_encoder.correction import load_color_encoder,ColorCorrection,ColorCorrection_y0

import torchvision.utils as tvu

from guided_diffusion.unet import UNetModel
from guided_diffusion.script_util import create_model, create_classifier, classifier_defaults, args_to_dict
import random
from datasets.imagenet_subset import ImageDataset

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=torch.device("cuda")):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def sample(self):
        cls_fn = None
        if self.config.model.type == 'simple':    
            model = Model(self.config)
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            elif self.config.data.dataset == 'CelebA_HQ':
                name = 'celeba_hq'
            else:
                raise ValueError
            if name != 'celeba_hq':
                ckpt = get_ckpt_path(f"ema_{name}", prefix=self.args.exp)
                print("Loading checkpoint {}".format(ckpt))
            elif name == 'celeba_hq':
                #ckpt = '~/.cache/diffusion_models_converted/celeba_hq.ckpt'
                ckpt = os.path.join(self.args.exp, "logs/celeba/celeba_hq.ckpt")
                if not os.path.exists(ckpt):
                    download('https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt', ckpt)
            else:
                raise ValueError
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model = torch.nn.DataParallel(model)

        elif self.config.model.type == 'openai':
            config_dict = vars(self.config.model)
            model = create_model(**config_dict)
            if self.config.model.use_fp16:
                model.convert_to_fp16()
            if self.config.model.class_cond:
                ckpt = os.path.join(self.args.exp, 'logs/imagenet/%dx%d_diffusion.pt' % (self.config.data.image_size, self.config.data.image_size))
                if not os.path.exists(ckpt):
                    download('https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_diffusion_uncond.pt' % (self.config.data.image_size, self.config.data.image_size), ckpt)
            else:
                ckpt = os.path.join(self.args.exp, "logs/imagenet/256x256_diffusion_uncond.pt")
                if not os.path.exists(ckpt):
                    download('https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt', ckpt)
                
            
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model.eval()
            model = torch.nn.DataParallel(model)

            if self.config.model.class_cond:
                ckpt = os.path.join(self.args.exp, 'logs/imagenet/%dx%d_classifier.pt' % (self.config.data.image_size, self.config.data.image_size))
                if not os.path.exists(ckpt):
                    image_size = self.config.data.image_size
                    download('https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_classifier.pt' % image_size, ckpt)
                classifier = create_classifier(**args_to_dict(self.config.classifier, classifier_defaults().keys()))
                classifier.load_state_dict(torch.load(ckpt, map_location=self.device))
                classifier.to(self.device)
                if self.config.classifier.classifier_use_fp16:
                    classifier.convert_to_fp16()
                classifier.eval()
                classifier = torch.nn.DataParallel(classifier)

                import torch.nn.functional as F
                def cond_fn(x, t, y):
                    with torch.enable_grad():
                        x_in = x.detach().requires_grad_(True)
                        logits = classifier(x_in, t)
                        log_probs = F.log_softmax(logits, dim=-1)
                        selected = log_probs[range(len(logits)), y.view(-1)]
                        return torch.autograd.grad(selected.sum(), x_in)[0] * self.config.classifier.classifier_scale
                cls_fn = cond_fn

        self.sample_sequence(model, cls_fn)

    def sample_sequence(self, model, cls_fn=None, device=torch.device("cuda")):
        args, config = self.args, self.config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device
        device_count = torch.cuda.device_count()
        dataset,test_dataset = get_dataset(args, config)
        
        
        if args.subset_start >= 0 and args.subset_end > 0:
            assert args.subset_end > args.subset_start
            test_dataset = torch.utils.data.Subset(test_dataset, range(args.subset_start, args.subset_end))
        else:
            args.subset_start = 0
            args.subset_end = len(test_dataset)

        print(f'Dataset has size {len(test_dataset)}')    
        
        def seed_worker(worker_id):
            worker_seed = args.seed % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        def trans_sketch(sketch):
            noise = torch.randn_like(sketch)*sigma_0
            mask = (sketch <= 0).any(dim=1, keepdim = True)
            result = torch.where(mask, sketch, noise)
            return result

        def load_npy(txt_file):
            class_to_npy = {}
            class_to_name = {}
            with open(txt_file,'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        file_name = parts[0]
                        class_index = int(parts[1])
                        class_to_npy[class_index] = f'npy/mural_masks/{file_name}.npy'
                        class_to_name[class_index] = f'{file_name}.png'
            return class_to_npy ,class_to_name

        def trans_rgb(rgb_r):    
            rgb_g = rgb_r + 1
            rgb_b = rgb_g + 1
            rgb = torch.cat([rgb_r, rgb_g, rgb_b], dim=0)
            return rgb 

        def squeezed( sketch):
            squeezed_sketch=sketch.squeeze(0)
            squeezed_sketch_orig=squeezed_sketch[0].cpu().numpy()
            for i in range(squeezed_sketch_orig.shape[0]):
                for j in range(squeezed_sketch_orig.shape[1]):
                    if squeezed_sketch_orig[i,j]<=-0.97:
                        squeezed_sketch_orig[i,j]=2
                    else:
                        squeezed_sketch_orig[i,j]=0
            return squeezed_sketch_orig

        g = torch.Generator()
        g.manual_seed(args.seed)
        val_loader = data.DataLoader(
            test_dataset,
            batch_size=config.sampling.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )
        sketch_loader = data.DataLoader(
            dataset,
            batch_size=config.sampling.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )

        deg = args.deg
        H_funcs = None

        txt_file = 'exp/txt/murals_test_1.txt'
        class_to_npy,class_to_name = load_npy(txt_file)

        args.sigma_0 = 2 * args.sigma_0 #to account for scaling to [-1,1]
        sigma_0 = args.sigma_0
        
        print(f'Start from {args.subset_start}')
        idx_init = args.subset_start
        idx_so_far = args.subset_start
        avg_psnr = 0.0
        lism = 0.1
        pbar = tqdm.tqdm(val_loader)
        sketchbar = tqdm.tqdm(sketch_loader)


        for x_orig , classes in pbar: 
            x_orig = data_transform(self.config, x_orig)
            x_orig = x_orig.to(self.device)
            from functions.svd_replacement import Inpainting 
            if deg == 'inp_mural':   
                for sketch_orig, s_classes in sketchbar: 
                    if s_classes == classes:
                        sketch_orig = sketch_orig.to(self.device)
                        sketch_orig = data_transform(self.config, sketch_orig)
                        squeezed_sketch_orig=squeezed(sketch_orig)
                    
                    else:
                        continue
                    sketch = torch.from_numpy(squeezed_sketch_orig).to(self.device).reshape(-1)
                    npy_class = classes.item()
                    mask_npy=np.load(class_to_npy[npy_class])
                    mural_name = class_to_name[npy_class]
                    mask = torch.from_numpy(mask_npy).to(self.device).reshape(-1) 

                    miss=mask+sketch 
                    missing_r = torch.nonzero(miss==0).long().reshape(-1) * 3
                    missline_r = torch.nonzero(miss==2).long().reshape(-1) * 3
                    missing = trans_rgb(missing_r)
                    missline = trans_rgb(missline_r)
                    H_funcs = Inpainting(config.data.channels, config.data.image_size, missing, missline, self.device)
                    sketch_orig = sketch_orig.to(self.device) * lism 

                    for i in range(len(x_orig)):
                        tvu.save_image(
                            inverse_data_transform(config, x_orig[i]), os.path.join(self.args.image_folder +'/degradation/', mural_name)
                        ) 

                    Color_model = load_color_encoder( 'exp/logs/color/color_encoder_6.pth', device)
                    image_folder = self.args.image_folder
                    with torch.no_grad():
                        ColorCorrection(image_folder, Color_model, mural_name, device)

                    img_path = os.path.join(f'{image_folder}/y_color_correction/{mural_name}')
                    y_orig = Image.open(img_path).convert("RGB")

                    resize_transform = transforms.Compose([
                    transforms.Resize((x_orig.shape[2], x_orig.shape[3])), 
                    transforms.ToTensor(), 
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
                    ])

                    y_orig = resize_transform(y_orig)

                    y_orig = y_orig.unsqueeze(0) 
                    y_orig = y_orig.to(device)
                    y_0 = H_funcs.H(y_orig,sketch_orig)


                    y_0 = y_0 + sigma_0 * torch.randn_like(y_0)
                    pinv_y_0 = H_funcs.H_pinv(y_0).view(y_0.shape[0], config.data.channels, self.config.data.image_size, self.config.data.image_size)

                    pinv_y_0 += H_funcs.H_pinv(H_funcs.H(torch.ones_like(pinv_y_0),torch.zeros_like(pinv_y_0))).reshape(*pinv_y_0.shape) - 1

                    for i in range(len(pinv_y_0)):
                        tvu.save_image(
                            inverse_data_transform(config, pinv_y_0[i]), os.path.join(self.args.image_folder +'/y0_mask', mural_name)
                        )

                    x = torch.randn(
                        y_0.shape[0],
                        config.data.channels,
                        config.data.image_size,
                        config.data.image_size,
                        device=self.device,) 

                    # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
                    with torch.no_grad():
                        
                        x, _ = self.sample_image(x, model, missing , H_funcs, y_0,  sketch_orig, mural_name , sigma_0=sigma_0, last=False, cls_fn=cls_fn, classes=classes)

                    x = [inverse_data_transform(config, y) for y in x]

                    
                    
                    for i in [-1]: 
                        for j in range(x[i].size(0)):
                            tvu.save_image(
                                x[i][j], os.path.join(self.args.image_folder+'/restoration', mural_name)
                            )

    def sample_image(self, x, model,missing, H_funcs, y_0, sketch, mural_name, sigma_0, last=True, cls_fn=None, classes=None):
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)
        
        x = efficient_generalized_steps(self,x, seq, model, self.betas, missing, H_funcs, y_0, sketch, mural_name, sigma_0, \
            etaB=self.args.etaB, etaA=self.args.eta, etaC=self.args.eta, cls_fn=cls_fn, classes=classes)
        if last:
            x = x[0][-1]
        return x
