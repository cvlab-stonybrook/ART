# ART
Official repository for **Attention in Referral Transformer** aka **ART** proposed in **"Look Hear: Gaze Prediction for Speech-directed Human Attention"** by Sounak Mondal, Seoyoung Ahn, Zhibo Yang, Niranjan Balasubramanian, Dimitris Samaras, Gregory Zelinsky, and Minh Hoai. 

🎉 Our work has been accepted to **ECCV 2024**!

📚 For download links and details of our **RefCOCO-Gaze** dataset, please visit our dedicated [dataset repository](https://github.com/cvlab-stonybrook/refcoco-gaze). 

📜 Find the preprint on [arXiv](https://arxiv.org/pdf/2407.19605).


# Installation

```bash
conda env create -n art --file art_env_export.yml
conda activate art
```
# Items to download

Download the contents of this [Google Drive folder](https://drive.google.com/drive/folders/1dguTIvidQh9wyuuhFQyy2xmtZoO0tTQp?usp=sharing) to the root directory.

```
-- ./data
    -- images_512X320                             # images for training on gaze prediction
    -- images_gaze.zip                            # COCO images for pre-training
    -- refcocogaze_train_tf_512X320_6.json        # RefCOCO-Gaze training set pre-processed to aid teacher-forcing
    -- refcocogaze_val_tf_512X320_6.json          # RefCOCO-Gaze validation set pre-processed to aid teacher-forcing
    -- refcocogaze_test_correct_512X320.json      # RefCOCO-Gaze test scanpaths
    -- clusters_refcocogaze.npy
    -- catDict.pkl
    -- refcocogaze_test_synced.pkl
-- ./checkpoints                                  
    -- art_checkpoint.pkg                         # checkpoint for inference on test data
    -- vgcore_checkpoint.pkg                      # pre-trained checkpoint to train on gaze prediction
-- ./logs
    -- pretraining
    -- training 
```

Extract ```images_gaze.zip``` in the ```data``` subdirectory.  

# Scripts

To pre-train ART, execute the following

```CUDA_VISIBLE_DEVICES=<gpu-id(s)> python3 pretrain.py```

To train ART, execute the following

```CUDA_VISIBLE_DEVICES=<gpu-id(s)> python3 train.py --pretrained_vgcore --vgcore_model_checkpoint=<pretrained_checkpoint>```

To run inference of ART on the test set of RefCOCO-Gaze, and evaluate it on the scanpath and saliency metrics, execute the following

```CUDA_VISIBLE_DEVICES=<gpu-id(s)> python3 inference.py --checkpoint=<checkpoint_location>```


# Citation

If you use either the ART model or RefCOCO-Gaze dataset, please cite as follows:
```
@InProceedings{Mondal_2024_ECCV,
author = {Mondal, Sounak and Ahn, Seoyoung and Yang, Zhibo and Balasubramanian, Niranjan and Samaras, Dimitris and Zelinsky, Gregory and Hoai, Minh},
title = {Look Hear: Gaze Prediction for Speech-directed Human Attention},
booktitle = {European Conference on Computer Vision (ECCV)},
year = {2024}
}
```
