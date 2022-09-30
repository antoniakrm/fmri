import torch
import os
import argparse
from torch import nn
import numpy as np

parser = argparse.ArgumentParser(description='Unsupervised training')
parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
parser.add_argument("--cuda", type=bool, default=True, help="Run on GPU")
params = parser.parse_args()

np.random.seed(params.seed)
torch.manual_seed(params.seed)
if params.cuda:
    torch.cuda.manual_seed(params.seed)

dir_root = os.path.expanduser(f'~/Dir/projects/summer_intern/data/unsup_best_mapping/Random_seed_{params.seed}')
if not os.path.exists(dir_root):
    os.mkdir(dir_root)

emb_dim = 2048
# Resnet152, OPT30B-Reduced-2048
default_mapping = nn.Linear(emb_dim, emb_dim, bias=False)

torch.save(default_mapping.weight.data.cpu().numpy(), os.path.join(dir_root,f'random_init_{params.seed}.pth'))
optim_mapping = nn.Linear(emb_dim, emb_dim, bias=False)
optim_mapping_shift = nn.Linear(emb_dim, emb_dim, bias=False)

# default_mapping.weight.data.copy_(torch.diag(torch.ones(emb_dim))) 
# distance from optimal map 63.962074
# print(default_mapping.weight.data.cpu().numpy())

# optim_path = os.path.join(dir_root, f'Random_seed_{params.seed}.pth')
# optim_path seed is 28, since seed28 is the highest one.
optim_path = os.path.expanduser('~/Dir/projects/summer_intern/data/unsup_best_mapping/offset_mapping_0.0.pth')
assert os.path.isfile(optim_path)
to_reload = torch.from_numpy(torch.load(optim_path))
optim_mapping.weight.data.copy_(to_reload.type_as(optim_mapping.weight.data))

dist_origin = np.linalg.norm(default_mapping.weight.data.cpu().numpy() - optim_mapping.weight.data.cpu().numpy())
# print(dist_origin)
# for shift in [0.01, 0.02, 0.04, 0.06, 0.08, 0.1]:
with open(dir_root+'/absolute_dist.txt', 'w+') as dis_w:
    dis_w.write(f'Random_seed:{params.seed} and the original distance: {dist_origin}\n')
    for shift in np.arange(0.1, 1, 0.1):
        for i in range(emb_dim):
            for j in range(emb_dim):
                optim_mapping_shift.weight.data[i][j] = optim_mapping.weight.data[i][j] - \
                    (optim_mapping.weight.data[i][j] - default_mapping.weight.data[i][j]) * shift

        dist_offset = np.linalg.norm(default_mapping.weight.data.cpu().numpy() - optim_mapping_shift.weight.data.cpu().numpy())
        dis_w.write(f'{dist_offset}\n')
        torch.save(optim_mapping_shift.weight.data.cpu().numpy(), f'{dir_root}/offset_mapping_{round(shift,1)}.pth')
    
    dis_w.write('\n')
    dis_w.close()
# print(dist_origin)
# print(dist_offset)