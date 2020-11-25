<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h3 align="center">Gradient-Induced Co-Saliency Detection</h3>

  <p align="center">
    Zhao Zhang*, Wenda Jin*, Jun Xu, Ming-Ming Cheng
    <br />
    <a href="http://zhaozhang.net/coca.html"><strong>⭐ Project Home »</strong></a>
    <br />
    <a href="https://arxiv.org/abs/2004.13364" target="_black">[PDF]</a>
    <a href="#" target="_black">[Code]</a>
    <a href="https://www.bilibili.com/video/BV1y5411a7Rq/" target="_black">[Short Video]</a>
    <a href="https://www.bilibili.com/video/BV1bi4y137c6" target="_black">[Long Video]</a>
    <a href="http://zhaozhang.net/papers/20_GICD/slides.pdf" target="_black">[Slides]</a>
    <a href="http://zhaozhang.net/papers/20_GICD/translation.pdf" target="_black">[中译版]</a>
    <a href="./papers/20_GICD/bibtex.txt" target="_black">[bib]</a>
    <br />
    <br />
  </p>
</p>


***
The training code of the ECCV 2020 paper 
[Gradient-Induced Co-Saliency Detection](https://arxiv.org/abs/2004.13364).

More details can be found at our [project home.](http://zhaozhang.net/coca.html)



## Prerequisites
* PyTorch = 1.3.1
* tqdm


<!-- USAGE EXAMPLES -->
## Usage
### Prepare data
#### Option 1
1. Download classified DUTS [DUTS_Class (78cb) (220M)](https://pan.baidu.com/s/154MqlqkQ0IoyRs92rwOX9g)
2. Enable the `--jigsaw` in `train.sh` to online jigsaw
#### Option 2
1. Download the jigsaw dataset directly. It's so big [Baidu pan (kbhg) (6.3G)](https://pan.baidu.com/s/1cfwGehc9Cq_bWqxEdkllSQ)
2. Disable the `--jigsaw` in `train.sh`

### Train
1. Put the vgg-16 pretrained model in to `./models`. [Baidu Pan: (6l5l)]( https://pan.baidu.com/s/1oXlUzCtkRS_yKpILhXHQeg)
2. Configure the `train.sh`
```shell
--tmp ./tmp/GICD_run1 (Root folder of log, model, and tensorboard.)
--trainset DUTS_class/Jigsaw2_DUTS
--jigsaw (Depend on dataset)
```
2. Run by
```
sh train.sh
```

### Test
1. Configure the input root and the output root in `test.sh`

``` 
--param_path ./tmp/GICD_run1 (tmp dir path)
--save_root your_output_root
```

2. Run by
```
sh test.sh
```
## Prediction results
The co-saliency maps of GICD can be found at our [project home.](http://zhaozhang.net/coca.html)

## Citation
If you find this work is useful for your research, please cite our paper:
```
@inproceedings{zhang2020gicd,
 title={Gradient-Induced Co-Saliency Detection},
 author={Zhang, Zhao and Jin, Wenda and Xu, Jun and Cheng, Ming-Ming},
 booktitle={European Conference on Computer Vision (ECCV)},
 year={2020}
}
```

## Contact
If you have any questions, feel free to contact me via `zzhang🥳mail😲nankai😲edu😲cn`