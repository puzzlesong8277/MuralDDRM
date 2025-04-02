from fileinput import filename
import os
from tokenize import group
import natsort
import torch
import torchvision
import natsort
import torch.nn as nn
from torchvision import transforms, models
from torchvision import transforms
from color_encoder.utils_color_encoder import *
from PIL import Image



def load_color_encoder( color_encoder_path, device):
    color_encoder = models.resnet34(pretrained=False)
    color_encoder.fc = nn.Linear(in_features=512, out_features=6, bias=False)
    color_encoder = color_encoder.to(device)
    encoder_checkpoint = torch.load(color_encoder_path, map_location=device)
    color_encoder.load_state_dict(encoder_checkpoint['model_state_dict'])
    color_encoder.eval()
    return color_encoder


def ColorCorrection(folder, model_function, filename , device):
    test_folder = os.path.join(folder+'/degradation')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(512)])
    newpath = os.path.join(folder+'/y_color_correction')
    
    image_path = os.path.join(test_folder, filename)

    if os.path.exists(image_path):
        img_scan = transform(Image.open(image_path)).to(device)

        output = model_function(img_scan[None, ...].to(device))
        [r_mean, g_mean, b_mean, r_std, g_std, b_std] = output[0].cpu().numpy()

        mean_pred = r_mean, g_mean, b_mean
        std_pred = r_std, g_std, b_std

        image_scan, mean_scan, std_scan = load_and_preprocess_image(image_path)

        pred_shift = apply_color_shift(image_scan.copy(), mean_pred, std_pred, mean_scan, std_scan)

        pred_shift = transform(pred_shift)

        torchvision.utils.save_image(pred_shift, f'{newpath}/{filename}')

def ColorCorrection_y0(img_tensors, folder, model_function, device ):
    transform = transforms.Compose([transforms.Resize(512)]) 
    newpath = os.path.join(folder + '/y_color_correction')

    os.makedirs(newpath, exist_ok=True)

    for i, img_tensor in enumerate(img_tensors):
        img_tensor = img_tensor.to(device) 
        output = model_function(img_tensor[None, ...].to(device)) 
        [r_mean, g_mean, b_mean, r_std, g_std, b_std] = output[0].cpu().detach().numpy()

        mean_pred = r_mean, g_mean, b_mean
        std_pred = r_std, g_std, b_std

        mean_scan = img_tensor.mean(dim=[1, 2]) 
        std_scan = img_tensor.std(dim=[1, 2])

        pred_shift = apply_color_shift(img_tensor.clone(), mean_pred, std_pred, mean_scan.cpu().numpy(), std_scan.cpu().numpy())

        torchvision.utils.save_image(pred_shift, f'{newpath}/corrected_image_{i}.png')

def mean_filter(image,kernel_size , device):
    kernel = torch.ones((1,1,kernel_size,kernel_size),dtype = torch.float32) / (kernel_size * kernel_size)
    kernel = kernel.repeat(image.size(1),1,1,1).to(device)
    output_tensor = torch.nn.functional.conv2d(image, kernel, padding= kernel_size//2 , groups=image.size(1)).to(device)
    return output_tensor

def add_y0combine(xt,missing, filename, t, device):
    d_degree = missing.size(0)/(xt.shape[2]*xt.shape[3])
    n = (xt.shape[2]/8)-d_degree*((xt.shape[2]/8)-1)
    n = round(n)
    if n % 2 ==0:
        n+=1
    n=max(1,min(n,7))

    if t>500: 
        img_path = os.path.join(filename)
        image = Image.open(img_path).convert("RGB")

        base_scale= 0.1
        scale = min(base_scale * (1-t/1000),base_scale)

        resize_transform = transforms.Compose([
            transforms.Resize((xt.shape[2], xt.shape[3])), 
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
            ])

        image_resized = resize_transform(image)

        image_resized = image_resized.unsqueeze(0)  

        image_resized = image_resized.to(device)

        image_resized = mean_filter(image_resized,n,device=xt.device) .to(device)
        xt = xt.to(device)
        xt = mean_filter(xt,n,device=xt.device).to(device)

        result =  xt + scale*image_resized - scale*xt

    else:
        result =  xt




    return result