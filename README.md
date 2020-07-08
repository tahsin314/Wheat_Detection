## Wheat Detection Competition

My scripts for the [Global Wheat Detection](https://www.kaggle.com/c/global-wheat-detection/) competition.

### Tasks
- Add `IoU` metric and `IoU` loss.
- Fix `Mosaic Augmentation`.

### Might be useful
- [End to End Object Detection with Transformers:DETR](https://www.kaggle.com/tanulsingh077/end-to-end-object-detection-with-transformers-detr)
#### For now I'm follwing [this](https://www.kaggle.com/shonenkov/training-efficientdet) wonderful kernel.

### Instructions
- Download `timm_efficientdet_pytorch` from [here](https://www.kaggle.com/shonenkov/timm-efficientdet-pytorch)
- Download `OmegaConf` from [here](https://www.kaggle.com/shonenkov/omegaconf).
- Run `git clone https://github.com/tahsin314/Wheat_Detection`
- Enter the `Wheat_Detection` folder and run `conda env create -f environment.yml`
- Run `conda activate wheat`
- Download Competition [data](https://www.kaggle.com/c/global-wheat-detection/data) and extract.
- In `config.py` edit `timm_efficientdet_path`, `omegaconf_path`, `data_dir` according to your paths. you can also change `batch_size`, `sz`(Image size), `accum_step` etc. parameters. 
- Run `train.py` 