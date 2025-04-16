
import torch
from torchvision import datasets, transforms

import os
import argparse
from tqdm import tqdm

import trimesh
from chamferdist import ChamferDistance



class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, base, args = None):
        self.base = base
        self.args = args

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x_path = self.base[idx]
#         y_path = self.image_list2[idx]
        y_path = x_path.replace(self.args.base,self.args.folder)
        try:
            x_mesh = trimesh.load_mesh(x_path)
            y_mesh = trimesh.load_mesh(y_path)
            
            x_vertices = torch.tensor(x_mesh.vertices, dtype=torch.float)
            y_vertices = torch.tensor(y_mesh.vertices, dtype=torch.float)
            return x_vertices,y_vertices
        
        except:
            print(self.base[idx])
            return self[idx+1]
        
    
    

parser = argparse.ArgumentParser("Triplane Gaussian Splatting")
parser.add_argument("--base", default='baseline_gso_3dgs', help="path to config file")
parser.add_argument("--folder", default='adv_visual_gso_sub_aligned_eps_2.0', help="path to output folder")
args, extras = parser.parse_known_args()


base= []

for file in os.listdir(args.base):
    file = os.path.join(args.base,file)
    base.append(file)
print(f'load {args.base} {len(base)} done!')



# 创建自定义数据集
custom_dataset = CustomDataset(sorted(base), args = args)

# 创建数据加载器
data_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=16, num_workers=8, shuffle=False)


chamferDist = ChamferDistance()


total_cd = 0
total_batches = 0


for x,y in tqdm(data_loader):
    
    x_vertices,y_vertices = x.cuda(), y.cuda()
    
    with torch.no_grad():
        # Calculate CD
        cd = chamferDist(x_vertices,y_vertices).mean().item()

        total_cd += cd
    total_batches += 1



average_cd = total_cd / total_batches

print(f'Average CD: {average_cd}')

with open('results_point_cloud.txt', mode='a+') as txt_file:
    txt_file.write(str([args.base, args.folder, average_cd]) +'\n')
    