
import torch
import piq
import csv
from PIL import Image

import os
import argparse

from torchvision import datasets, transforms


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_list1, args = None, transform=None):
        self.image_list1 = image_list1
        self.transform = transform
        self.args = args

    def __len__(self):
        return len(self.image_list1)

    def __getitem__(self, idx):
        x_path = self.image_list1[idx]
#         y_path = self.image_list2[idx]
        y_path = x_path.replace(self.args.base,self.args.folder)
    
        x = Image.open(x_path).convert('RGB')
        y = Image.open(y_path).convert('RGB')
        
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        return x,y


parser = argparse.ArgumentParser("Triplane Gaussian Splatting")
parser.add_argument("--base", default='baseline_omni3D_sub_aligned', help="path to config file")
parser.add_argument("--folder", default='baseline_gso_sub_aligned', help="path to output folder")
args, extras = parser.parse_known_args()

args.folder1 = args.base

base=[]

for sub_folder1 in os.listdir(args.base):
    sub_folder1 = os.path.join(args.base,sub_folder1)
    for file1 in os.listdir(sub_folder1):
        file1 = os.path.join(sub_folder1,file1)
        base.append(file1)


print(f'load {args.base} {len(base)} done!')

# 定义数据转换
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
])


# 创建自定义数据集
custom_dataset = CustomDataset(sorted(base), args = args, transform=transform)

# 创建数据加载器
data_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=20, num_workers=8, shuffle=False)

import lpips
# Initialize the LPIPS model
loss_fn = lpips.LPIPS(net='vgg').to('cuda').eval()
lpips_score = loss_fn(x,y).mean().item()

total_psnr = 0
total_ssim = 0
total_lpips = 0
total_batches = 0

# Calculate the LPIPS distance between the two tensors
# lpips_distance = loss_fn(A, B).mean()

from tqdm import tqdm
# 现在您可以使用 data_loader 来迭代加载图像数据
for images in tqdm(data_loader):
    try:

        # 处理每个批次的图像数据
        x,y = images
        x = x.cuda()
        y = y.cuda()
        with torch.no_grad():
            # Calculate PSNR
            psnr = piq.psnr(x, y, data_range=1.)
            # Calculate SSIM
            ssim_score = piq.SSIMLoss(data_range=1.)(x, y)
            # Calculate LPIPS
            lpips_score = loss_fn(x,y).mean().item()

        total_psnr += psnr.item()
        total_ssim += ssim_score.item()
        total_lpips += lpips_score
        total_batches += 1
    except:
        print('error')
        continue

#     print(f'PSNR: {psnr}, SSIM: {ssim_score}, LPIPS: {lpips_score}')
average_psnr = total_psnr / total_batches
average_ssim = total_ssim / total_batches
average_lpips = total_lpips / total_batches

print(f'Average PSNR: {average_psnr}, Average SSIM: {average_ssim}, Average LPIPS: {average_lpips}')

with open('results.txt', mode='a+') as txt_file:
    txt_file.write(str([args.base, args.folder, average_psnr, average_ssim, average_lpips]) +'\n')
    