import torch
import os, time
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import data_generator as dg
import numpy as np
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from data_generator import DenoisingDataset
from option import get_parser
from model import DnCNN
from skimage.io import imread
from unet import UNet
from util import show, save_result
from skimage.measure import compare_psnr, compare_ssim

os.environ['CUDA_VISIBLE_DEVICES'] = '3'



# print('xxxxxxxxxxx')

class sum_squared_error(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        return F.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)


def train(model, opt):
    save_dir = os.path.join('models', opt.model + '_' + 'sigma' + str(opt.sigma))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    opt.cuda = True
    torch.set_num_threads(4)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.2)
    criterion = nn.MSELoss()
    # criterion = sum_squared_error()
    if opt.cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    data = dg.data_generator(data_dir=opt.train_data, batch_size=opt.batch_size)
    data = data.astype('float32') / 255.0
    data = torch.from_numpy(data.transpose(0, 3, 1, 2))
    train_dataset = DenoisingDataset(data, opt.sigma)

    for epoch in range(opt.epochs):
        model.train()
        scheduler.step(epoch)
        epoch_loss = 0.0
        start_time = time.time()
        train_loader = DataLoader(dataset=train_dataset,
                                  drop_last=True,
                                  batch_size=opt.batch_size,
                                  shuffle=True)

        for i, (batch_y, batch_x) in enumerate(train_loader):
            optimizer.zero_grad()
            if opt.cuda:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            loss = criterion(model(batch_y), batch_x)
            epoch_loss + loss.item()
            loss.backward()
            optimizer.step()
            if (i + 1) % 10 == 0:
                print('Train Epoch: %4d [%4d / %4d] loss = %2.4f' % (
                    epoch + 1, i + 1, data.size(0) // opt.batch_size, loss.item() / opt.batch_size))
        validate(model, opt, epoch)
        t = time.time() - start_time
        print('epcoh = %4d , loss = %4.4f , time = %4.2f s' % (epoch + 1, epoch_loss / (i + 1), t))
        torch.save(model, os.path.join(save_dir, 'lastest.pth'))

def validate(model, opt, epoch):
    model.eval()
    if not os.path.exists(opt.result_dir):
        os.makedirs(opt.result_dir)
    for set_cur in opt.set_names:
        if not os.path.exists(os.path.join(opt.result_dir, set_cur, str(opt.sigma))):
            os.makedirs(os.path.join(opt.result_dir, set_cur, str(opt.sigma)))
        psnrs = []
        ssims = []

        for im in os.listdir(os.path.join(opt.test_data, set_cur)):
            if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):
                x = np.array(imread(os.path.join(opt.test_data, set_cur, im)), dtype='float32') / 255.0
                np.random.seed(0)
                y = x + np.random.normal(0, opt.sigma / 255.0, x.shape)
                y = y.astype('float32')
                y_ = torch.from_numpy(y).view(1, -1, y.shape[0], y.shape[1])
                torch.cuda.synchronize()
                start_time = time.time()
                y_ = y_.cuda()
                x_ = model(y_)  # inference
                x_ = x_.view(y.shape[0], y.shape[1])
                x_ = x_.cpu()
                x_ = x_.detach().numpy().astype(np.float32)
                torch.cuda.synchronize()
                elapsed_time = time.time() - start_time
                print('%10s : %10s : %2.4f second' % (set_cur, im, elapsed_time))

                psnr_x_ = compare_psnr(x, x_)
                ssim_x_ = compare_ssim(x, x_)
                if opt.save_result:
                    name, ext = os.path.splitext(im)
                    # show(np.hstack((x, y, x_)))  # show the image

                    save_result(np.hstack((x, y, x_)), path=os.path.join(opt.result_dir, set_cur, str(opt.sigma),
                                                                         name + '_dncnn_' + str(
                                                                             epoch) + ext))  # save the denoised image
                psnrs.append(psnr_x_)
                ssims.append(ssim_x_)

        psnr_avg = np.mean(psnrs)
        ssim_avg = np.mean(ssims)
        psnrs.append(psnr_avg)
        ssims.append(ssim_avg)
        if opt.save_result:
            save_result(np.hstack((psnrs, ssims)), path=os.path.join(opt.result_dir, set_cur, 'results.txt'))
        print('Datset: {0:10s} \n  PSNR = {1:2.2f}dB, SSIM = {2:1.4f}'.format(set_cur, psnr_avg, ssim_avg))


if __name__ == '__main__':
    model = UNet(1, 1)

    opt = get_parser().parse_args()
    train(model, opt)