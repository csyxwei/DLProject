import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='DnCNN', type=str, help='choose a type of model')
    parser.add_argument('--batch_size', default=110, type=int, help='batch size')
    parser.add_argument('--train_data', default='data/Train400', type=str, help='path of train data')
    parser.add_argument('--test_data', default='data/Test', type=str, help='directory of test dataset')
    parser.add_argument('--set_names', default=['Set12'], help='directory of test dataset')
    parser.add_argument('--sigma', default=50, type=int, help='noise level')
    parser.add_argument('--model_dir', default='models', type=str, help='directory of test dataset')
    parser.add_argument('--epochs', default=50, type=int, help='number of train epoches')
    parser.add_argument('--gpu', default='1', type=int, help='gpus')
    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
    parser.add_argument('--result_dir', default='results', type=str, help='directory of test dataset')
    parser.add_argument('--save_result', default=1, type=int, help='save the denoised image, 1 or 0')
    return parser
