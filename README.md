# DeepConvolutionalForestParameterRegression

# Datasets
Processed Sentinel-1 scenes for the Liwale AOI in Tanzania and the three Norwegian AOIs are openly available, and can be downloaded from ed [here](https://drive.google.com/drive/folders/1ZyxcArRPV6FfXzZsCIwSgECJCrx-AYxf).

The ALS-derived AGB prediction maps for the Tanzanian dataset are available on request from the corresponding author of _"Mapping and estimating forest area and aboveground biomass in miombo woodlands in Tanzania using data from airborne laser scanning, TanDEM-X, RapidEye, and global forest maps: A comparison of estimated precision"_, see [here](https://www.sciencedirect.com/science/article/abs/pii/S0034425716300062) for more information.  

The ALS-derived SV prediction maps for the Norwegian datasets are available on request from the corresponding author of _"Comparing the accuracies of forest attributes predicted from airborne laser scanning and digital aerial photogrammetry in operational forest inventories"_, see [here](https://www.sciencedirect.com/science/article/pii/S0034425719301178) for more information. 

The ground reference AGB data are part of the Tanzanian National Forest Inventory program. They are confidential and, therefore, not publicly available. The ground reference SV data from Norway are private property and, therefore, not publicly available. Thus are the masks representing the ground reference target data, and the coverage of the ALS-derived SV prediction maps for the Norwegian datasets not publicly available.

## Dataset storage paths: 
The model assumes that there exists paired input and target images for training and validation. I.e. input image patches for **training** are assumed to be stored as follows: `python/datasets/Tanzania/A/train`. Target image patches are assumed to be stored as follows: `python/datasets/Tanzania/B/train`.
In the paths above, replace `train` with e.g. `val` to achieve the corresponding paths to **validation** datasets.
**Note!** The test phase only requires input data, thus the path to the test data can be specified as follows: `python/datasets/Tanzania/A/test`. 

## Tanzanian models: 
The Tanzanian models were trained using a false-colour (RGB) Sentinel-1 image patches, i.e. each having the shape 64x64x3. In total, the number of target channels were two, stacked as follows `[ALS-derived prediction maps, ground reference target mask]`. See [here](https://arxiv.org/abs/2306.11103) for the paper and reference to the input and target data. 

## Norwegian models: 
The Norwegian models were trained using 9-channel Sentinel-1 image patches, i.e. each having the shape 64x64x9. In total, the number of target channels were three, stacked as follows `[ALS-derived prediction maps, timber volume mask, ground reference target mask]`. See [here](https://arxiv.org/abs/2306.11103) for the paper and reference to the input and target data.


# Source code
The source code for the paper _"Forest Parameter Prediction by Multiobjective Deep Learning of Regression Models Trained with Pseudo-Target Imputation"_, see [here](https://arxiv.org/abs/2306.11103) for the paper, can be found in the `python` directory.

## Pretraining: 
To pretrain e.g. a Tanzanian Pixel-aware Regression U-Net model run e.g. 

`python3 python/train_agb.py --alpha 0 --batch_size 2 --beta1 0.7 --checkpoints_dir ./checkpoints_unet/tanzanian_agb_train --convert False --crop_size 64 --dataset_mode rs_agb --enc_depth 5 --encoder_name resnet34 --epoch 50 --epoch_count 51 --extension .tif --gamma 0 --gan_mode lsgan --impute_w 400 --input_nc 3 --input_shp 64 --l1_l2 l1 --l_type pixel --lr 0.0001 --name pretrain_pixel_model --netD basic --netG customUnet --niter 300  --niter_decay 0 --no_flip --norm none --phase train --preprocess none --pretrainweights None --relu True --save_epoch_freq 50 --spec_loss fft --stage 1 --target_nc 1 --train_input_path python/datasets/Tanzania/A/train --train_target_path python/datasets/Tanzania/B/train --use_wandb True --wandb_name pretrain_pixel_model --wandb_project Tanzania_pretrain_agb_pixel`

Replace `train_agb.py` with `train_volume.py` to train the stem volume model and change e.g. `input_nc` to 9


## Fine-tuning: 
To fine-tune e.g. the Tanzanian Pixel-aware Regression U-Net model on a combination of a pixel- GAN- and FFT-loss run e.g. 

`python3 python/train_agb.py --alpha 0.01 --batch_size 2 --beta1 0.7 --checkpoints_dir ./checkpoints_unet/tanzanian_agb_train --convert False --crop_size 64 --continue_train_finetune --dataset_mode rs_agb --enc_depth 5 --encoder_name resnet34 --epoch 150 --epoch_count 151 --extension .tif --gamma 1e-7 --gan_mode lsgan --impute_w 500 --input_nc 3 --input_shp 64 --l1_l2 l1 --l_type prepixel_gan_spec --lr 0.0001 --name pretrain_pixel_model --name_finetune finetune_pixel_gan_spec --netD pixel --netG customUnet --niter 500  --niter_decay 0 --no_flip --norm instance --phase train --preprocess none --pretrainweights None --relu True --save_epoch_freq 50 --spec_loss_name fft --stage 2 --target_nc 1 --train_input_path python/datasets/Tanzania/A/train --train_target_path python/datasets/Tanzania/B/train --use_wandb True --wandb_name finetune_pixel_gan_spec --wandb_project Tanzania_finetune_agb_pixel`

`--epoch 150` `--epoch_count 151` represents from which epoch the fine-tuning start (epoch 150) and how the fine-tuned checkpoint models should be indexed, i.e. starting from epoch 151 and increasing.

## Model test phase: 
To test e.g the pretrained Tanzanian Pixel-aware Regression U-Net mode above run e.g 

`python3 python/test.py --alpha 0 --batch_size 1 --checkpoints_dir ./checkpoints_unet/tanzanian_agb_train --convert False --crop_size 64 --dataset_mode remotesensing --enc_depth 5 --encoder_name resnet34 --epoch 50 --extension .tiff --gpu_ids -1 --input_nc 3 --input_shp 64 --name pretrain_pixel_model --netG customUnet --no_flip --norm none --norm_mean 0.485 0.456 0.406 --norm_std 0.229 0.224 0.225 --num_test 507 --phase test --preprocess none --pretrainweights None --results_dir ./results/Tanzania/A_for_AGB --stage 1 --target_nc 1 --test_input_path python/datasets/Tanzania/test/ --use_wandb False`

# Citation
If you use this code please cite 
`@misc{björk2023forest,
      title={Forest Parameter Prediction by Multiobjective Deep Learning of Regression Models Trained with Pseudo-Target Imputation}, 
      author={Sara Björk and Stian N. Anfinsen and Michael Kampffmeyer and Erik Næsset and Terje Gobakken and Lennart Noordermeer},
      year={2023},
      eprint={2306.11103},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}`

The code-basis is inspired by Isola. et al.'s implementation for their Pix2pix model. See [here](https://github.com/phillipi/pix2pix) for references of how to cite their work.  