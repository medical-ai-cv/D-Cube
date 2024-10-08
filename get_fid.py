from cleanfid import fid
import torch

category = ['class_1', 'class_2', 'class_3', 'class_4']

for c in category:
    x_path = '/synthetic dataset path in_here/{}'.format(c)
    y_path = '/original dataset path in_here/{}'.format(c)

    score = fid.compute_fid(x_path, y_path, num_workers=4, device=torch.device("cuda"))
    print('{} FID: {}'.format(c, score))




