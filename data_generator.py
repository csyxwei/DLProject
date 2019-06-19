import glob
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch

patch_size, stride = 64, 10
aug_times = 1
scales = [1, 0.9, 0.8, 0.7]

class DenoisingDataset(Dataset):
    def __init__(self, xs, sigma):
        super(DenoisingDataset, self).__init__()
        self.xs = xs
        self.sigma = sigma

    def __getitem__(self, index):
        batch_x = self.xs[index]
        noise = torch.randn(batch_x.size()).mul_(self.sigma / 255.0)
        batch_y = batch_x + noise
        return batch_y, batch_x

    def __len__(self):
        return self.xs.size(0)


def data_aug(img, mode=0):
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def gen_patches(file_name):
    img = cv2.imread(file_name, 0)
    h, w = img.shape
    patches = []
    for s in scales:
        h_scaled, w_scaled = int(h * s), int(w * s)
        img_scaled = cv2.resize(img, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)
        for i in range(0, h_scaled - patch_size + 1, stride):
            for j in range(0, w_scaled - patch_size + 1, stride):
                x = img_scaled[i:i + patch_size, j:j + patch_size]
                for k in range(aug_times):
                    x_aug = data_aug(x, mode=np.random.randint(0, 8))
                    patches.append(x_aug)
    return patches


def data_generator(data_dir='data/Train400', batch_size=128, verbose=False):
    file_list = glob.glob(data_dir + '/*.png')
    data = []
    for i, file in enumerate(file_list):
        patches = gen_patches(file)
        for patch in patches:
            data.append(patch)
        if verbose:
            print('{}/{} is done ^_^'.format(i + 1, len(file_list)))
    data = np.array(data, dtype='uint8')
    data = np.expand_dims(data, axis=3)
    discard_n = len(data) - len(data) // batch_size * batch_size
    data = np.delete(data, range(discard_n), axis=0)
    return data

if __name__ == '__main__':

    data = data_generator(data_dir='data/Train400')