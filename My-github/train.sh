CUDA_VISIBLE_DEVICES=0 python train.py --model GICD --loss DSLoss_IoU_noCAM --trainset DUTS_class --valset CoSal15 --size 224 --tmp /home/nku/COD/GICD_Share/tmp --lr 1e-4 --bs 1 --epochs 100 --use_tensorboard --jigsaw

