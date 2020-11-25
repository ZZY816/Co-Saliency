from PIL import Image
from dataset import get_loader
import torch
from torchvision import transforms
from util import save_tensor_img
from tqdm import tqdm
from torch import nn
import os
import argparse


def main(args):
    dataset_names = args.testset.split('+')
    for idataset in dataset_names:
        if idataset == 'CoCA':
            test_img_path = '../Dataset/CoCA/image/'
            test_gt_path = '../Dataset/CoCA/binary/'
            saved_root = os.path.join(args.save_root, 'CoCA')
        elif idataset == 'CoSOD3k':
            test_img_path = '../Dataset/CoSOD3k/Image/'
            test_gt_path = '../Dataset/CoSOD3k/GroundTruth/'
            saved_root = os.path.join(args.save_root, 'CoSOD3k')
        elif idataset == 'CoSal2015':
            test_img_path = '../Dataset/CoSal2015/Image/'
            test_gt_path = '../Dataset/CoSal2015/GT/'
            saved_root = os.path.join(args.save_root, 'CoSal2015')
        elif idataset == 'iCoseg':
            test_img_path = '../Dataset/iCoseg/Image/'
            test_gt_path = '../Dataset/iCoseg/GroundTruth/'
            saved_root = os.path.join(args.save_root, 'iCoseg')
        elif idataset == 'MSRC':
            test_img_path = '../Dataset/MSRC/Image/'
            test_gt_path = '../Dataset/MSRC/GroundTruth/'
            saved_root = os.path.join(args.save_root, 'MSRC')
        else:
            print('Unkonwn test dataset')
            print(args.dataset)

        test_loader = get_loader(test_img_path,
                                 test_gt_path,
                                 args.size,
                                 1,
                                 istrain=False,
                                 shuffle=False,
                                 num_workers=8,
                                 pin=True)

        # Init model
        device = torch.device("cuda")
        exec('from models import ' + args.model)
        model = eval(args.model + '()')
        model = model.to(device)
        ginet_dict = torch.load(os.path.join(args.param_root,
                                             'gicd_ginet.pth'))
        model.to(device)
        model.ginet.load_state_dict(ginet_dict)

        model.eval()
        model.set_mode('test')

        tensor2pil = transforms.ToPILImage()

        for batch in tqdm(test_loader):
            inputs = batch[0].to(device)
            gts = batch[1].to(device)
            subpaths = batch[2]
            ori_sizes = batch[3]

            scaled_preds = model(inputs)

            os.makedirs(os.path.join(saved_root, subpaths[0][0].split('/')[0]),
                        exist_ok=True)
            num = len(scaled_preds)
            for inum in range(num):
                subpath = subpaths[inum][0]
                ori_size = (ori_sizes[inum][0].item(),
                            ori_sizes[inum][1].item())
                res = nn.functional.interpolate(scaled_preds[inum][-1],
                                                size=ori_size,
                                                mode='bilinear',
                                                align_corners=True)
                save_tensor_img(res, os.path.join(saved_root, subpath))


if __name__ == '__main__':
    # Parameter from command line
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', default='GICD', type=str)
    parser.add_argument(
        '--testset',
        default='CoCA',
        type=str,
        help="Options: 'CoCA','CoSal2015','CoSOD3k','iCoseg','MSRC'")
    parser.add_argument('--size', default=224, type=int, help='input size')
    parser.add_argument('--param_root',
                        default='paras',
                        type=str,
                        help='model folder')
    parser.add_argument('--save_root',
                        default='../SalMaps/pred',
                        type=str,
                        help='Output folder')
    args = parser.parse_args()

    main(args)
