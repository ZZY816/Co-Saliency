from util import Logger, AverageMeter, save_checkpoint, save_tensor_img, set_seed
set_seed(1996)

import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from matplotlib import pyplot as plt
import time
import argparse
from tqdm import tqdm
from dataset import get_loader
from criterion import Eval
from models.RPNet import Mynet
from loss import DSLoss_IoU_noCAM
import shutil
import torchvision.utils as vutils

# Parameter from command line
parser = argparse.ArgumentParser(description='')

parser.add_argument('--bs', '--batch_size', default=1, type=int)
parser.add_argument('--lr',
                    '--learning_rate',
                    default=1e-4,
                    type=float,
                    help='Initial learning rate')
parser.add_argument('--resume',
                    default=None,
                    type=str,
                    help='path to latest checkpoint')
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--start_epoch',
                    default=0,
                    type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--trainset',
                    default='DUTS_class',
                    type=str,
                    help="Options: 'Jigsaw2_DUTS', 'DUTS_class'")
parser.add_argument('--valset',
                    default='CoSal15',
                    type=str,
                    help="Options: 'CoSal15', 'CoCA'")
parser.add_argument('--size', default=224, type=int, help='input size')
import datetime
timenow = datetime.datetime.now()
now = timenow.strftime('%Y:%m:%d_%H:%M:%S' )
parser.add_argument('--tmp', default='/home/nku/New_Co-Sal/tmp'+'_'+now, help='Temporary folder')
parser.add_argument("--use_tensorboard", default=True)
parser.add_argument("--jigsaw", default=True)

args = parser.parse_args()

# Init TensorboardX
if args.use_tensorboard:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(args.tmp)

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')



train_img_path = './Dataset/DUTS_class/img/'
train_gt_path = './Dataset/DUTS_class/gt/'
train_sal_path = './Dataset/DUTS_class/sal/'
train_loader = get_loader(train_img_path,
                          train_gt_path,
                           train_sal_path,
                              args.size,
                              args.bs,
                              max_num=12,
                              istrain=True,
                              jigsaw=args.jigsaw,
                              shuffle=False,
                              num_workers=4,
                              pin=True)


val_img_path = './Dataset/CoSal2015/img/'
val_gt_path = './Dataset/CoSal2015/gt/'
val_sal_path = './Dataset/CoSal2015/sal/'
val_loader = get_loader(val_img_path,
                            val_gt_path,
                            val_sal_path,
                            args.size,
                            1,
                            istrain=False,
                            jigsaw=args.jigsaw,
                            shuffle=False,
                            num_workers=4,
                            pin=True)



os.makedirs(args.tmp, exist_ok=True)

logger = Logger(os.path.join(args.tmp, "log.txt"))

device = torch.device("cuda")

model = Mynet()
model = model.to(device)

'''backbone_params = list(map(id, model.ginet.backbone.parameters()))
base_params = filter(lambda p: id(p) not in backbone_params,
                     model.ginet.parameters())'''

# Setting optimizer
optimizer = optim.Adam(params=model.parameters(), lr=args.lr, betas=[0.9, 0.99])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# log model and optimizer pars
logger.info("Model details:")
logger.info(model)
logger.info("Optimizer details:")
logger.info(optimizer)
logger.info("Scheduler details:")
logger.info(scheduler)
logger.info("Other hyperparameters:")
logger.info(args)

# Setting Loss

dsloss = DSLoss_IoU_noCAM()


def main():
    val_mae_record = []
    val_Sm_record = []

    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    print(args.epochs)
    for epoch in range(args.start_epoch, args.epochs):
        train_loss = train(epoch)
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'scheduler': scheduler.state_dict(),
            },
            path=args.tmp)
        # Save checkpoint
        if epoch%1 == 0 :
            [val_mae, val_Sm] = validate(epoch)
            val_mae_record.append(val_mae)
            val_Sm_record.append(val_Sm)



    mysal_dict = model.state_dict()
    torch.save(mysal_dict, os.path.join(args.tmp, 'rpnet.pth'))

    # Show in tensorboard
    if args.use_tensorboard:
        writer.add_scalar('Loss/total', train_loss, epoch)

        writer.add_scalar('Metric/MAE', val_mae, epoch)
        writer.add_scalar('Metric/Sm', val_Sm, epoch)


def train(epoch):
    loss_log = AverageMeter()

    # Switch to train mode
    model.train()

    for batch_idx, batch in enumerate(train_loader):
        inputs = batch[0].to(device)
        gts = batch[1].to(device)
        sals = batch[2].to(device)

        out_list, similarity_list = model(inputs, sals)
        loss = 0.
        loss_show = 0.
        for out in out_list:
            #print(type(out))
            loss = loss + dsloss(out, gts)
            loss_show = loss_show + dsloss(out, gts)/len(out)


        loss_log.update(loss_show/len(out_list), args.bs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 20 == 0:
            # NOTE: Top2Down; [0] is the grobal slamap and [5] is the final output
            logger.info('Epoch[{0}/{1}] Iter[{2}/{3}]  '
                        'Train Loss: {loss.val:.3f} ({loss.avg:.3f})  '.format(
                            epoch,
                            args.epochs,
                            batch_idx,
                            len(train_loader),
                            loss=loss_log,
                        ))
    scheduler.step()
    logger.info('@==Final== Epoch[{0}/{1}]  '
                'Train Loss: {loss.avg:.3f}  '.format(epoch,
                                                      args.epochs,
                                                      loss=loss_log))

    return loss_log.avg


def validate(epoch):

    # Switch to evaluate mode
    model.eval()
    #model.load_state_dict(torch.load(args.tmp+'/checkpoint.pth')['state_dict'])

    saved_root = os.path.join(args.tmp, 'Salmaps')
    saved_root_of_6 = os.path.join(args.tmp, 'Salmaps_of_6')
    similarity_root_of_6 = os.path.join(args.tmp, 'Similarity_of_6')
    all_root =  os.path.join(args.tmp, 'all')
    all_root_similarity = os.path.join(args.tmp, 'all_similarity')
    # make dir for saving results
    os.makedirs(saved_root, exist_ok=True)
    os.makedirs(saved_root_of_6, exist_ok=True)
    os.makedirs(similarity_root_of_6, exist_ok=True)
    os.makedirs(all_root, exist_ok=True)
    os.makedirs(all_root_similarity, exist_ok=True)
    for batch in tqdm(val_loader):
        inputs = batch[0].to(device)
        gts = batch[1].to(device)
        sals = batch[2].to(device)
        subpaths = batch[3]
        ori_sizes = batch[4]
        with torch.no_grad():
            scaled_preds, similarity_list = model(inputs, sals)

        os.makedirs(os.path.join(saved_root, subpaths[0][0].split('/')[0]),
                    exist_ok=True)
        save_6_list = []
        for i in range(len(scaled_preds)):
            save_6 = os.path.join(saved_root_of_6,  'salmaps_' + str(i))
            save_6_list.append(save_6)
            save_similarity = os.path.join(similarity_root_of_6,  'similarity_' + str(i))
            os.makedirs(os.path.join(save_6, subpaths[0][0].split('/')[0]),
                        exist_ok=True)
            os.makedirs(os.path.join(save_similarity, subpaths[0][0].split('/')[0]),
                        exist_ok=True)

            num = len(subpaths)
            for inum in range(num):
                subpath = subpaths[inum][0]
                #print(subpath)
                ori_size = (ori_sizes[inum][0].item(), ori_sizes[inum][1].item())
                res = nn.functional.interpolate(scaled_preds[i][inum],
                                            size=ori_size,
                                            mode='bilinear',
                                            align_corners=True)
                res_similiarity = nn.functional.interpolate(similarity_list[i][inum],
                                                size=ori_size,
                                                mode='bilinear',
                                                align_corners=True)
                if i == len(scaled_preds)-1:
                    save_tensor_img(res, os.path.join(saved_root, subpath))

                save_tensor_img(res, os.path.join(save_6, subpath))
                os.makedirs(os.path.join(all_root, subpath.split('/')[0]  ), exist_ok=True)
                shutil.copy(os.path.join(save_6, subpath), os.path.join(all_root, subpath).split('.')[0] + '_' + str(i) + '.png')
                save_tensor_img(res_similiarity, os.path.join(save_similarity, subpath))
                os.makedirs(os.path.join(all_root_similarity, subpath.split('/')[0]  ), exist_ok=True)
                shutil.copy( os.path.join(save_similarity, subpath), os.path.join(all_root_similarity, subpath).split('.')[0] + '_' + str(i) + '.png')

    for save in save_6_list:
        evaler = Eval(pred_root=save, label_root=val_gt_path)
        mae = evaler.eval_mae()
        Sm = evaler.eval_Smeasure()


        logger.info('@==Final== Epoch[{0}/{1}]  '
                'MAE: {mae:.3f}  '
                'Sm: {Sm:.3f}'.format(epoch, args.epochs, mae=mae, Sm=Sm))

    return mae, Sm


if __name__ == '__main__':
    main()
