import os
import time
import torch
import numpy as np
from option import get_test_parser
from skimage.io import imread, imsave
from skimage.measure import compare_psnr, compare_ssim
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    opt = get_test_parser().parse_args()
    if not os.path.exists(os.path.join(opt.model_dir, opt.model_name)):
        print('Model file not found')
        exit()
    else:
        model = torch.load(os.path.join(opt.model_dir, opt.model_name))
        print('load trained model')

    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
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
                start_time = time.time()
                y_ = y_.cuda()
                x_ = model(y_)  # inference
                x_ = x_.view(y.shape[0], y.shape[1])
                x_ = x_.cpu()
                x_ = x_.detach().numpy().astype(np.float32)
                elapsed_time = time.time() - start_time
                print('%10s : %10s : %2.4f second' % (set_cur, im, elapsed_time))
                psnr_x_ = compare_psnr(x, x_)
                ssim_x_ = compare_ssim(x, x_)
                name, ext = os.path.splitext(im)
                path = os.path.join(opt.result_dir, set_cur, str(opt.sigma), name + ext)
                imsave(path, np.clip(np.hstack((x, y, x_)), 0, 1))
                psnrs.append(psnr_x_)
                ssims.append(ssim_x_)
        psnr_avg = np.mean(psnrs)
        ssim_avg = np.mean(ssims)
        psnrs.append(psnr_avg)
        ssims.append(ssim_avg)
        print('Datset: {0:10s} \n  PSNR = {1:2.2f}dB, SSIM = {2:1.4f}'.format(set_cur, psnr_avg, ssim_avg))
