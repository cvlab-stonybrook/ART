# ART
Official repository for **Attention in Referral Transformer** aka **ART** proposed in **"Look Hear: Gaze Prediction for Speech-directed Human Attention"** by Sounak Mondal, Seoyoung Ahn, Zhibo Yang, Niranjan Balasubramanian, Dimitris Samaras, Gregory Zelinsky, and Minh Hoai. 

🎉 Our work has been accepted to **ECCV 2024**!

📚 For download links and details of our **RefCOCO-Gaze** dataset, please visit our dedicated [dataset repository](https://github.com/cvlab-stonybrook/refcoco-gaze). 

📜 Find the preprint on [arXiv](https://arxiv.org/pdf/2407.19605).

📨 Contact **Sounak Mondal** at ```somondal@cs.stonybrook.edu``` for any queries.

For computer systems to effectively interact with humans using spoken language, they need to understand how the words being generated affect the users' moment-by-moment attention. Our study focuses on the incremental prediction of attention as a person is seeing an image and hearing a referring expression defining the object in the scene that should be fixated by gaze. To predict the gaze scanpaths in this incremental object referral task, we developed the Attention in Referral Transformer model or ART, which predicts the human fixations spurred by each word in a referring expression. ART uses a multimodal transformer encoder to jointly learn gaze behavior and its underlying grounding tasks, and an autoregressive transformer decoder to predict, for each word, a variable number of fixations based on fixation history. To train ART, we created RefCOCO-Gaze, a large-scale dataset of 19,738 human gaze scanpaths, corresponding to 2,094 unique image-expression pairs, from 220 participants performing our referral task. In our quantitative and qualitative analyses, ART not only outperforms existing methods in scanpath prediction, but also appears to capture several human attention patterns, such as waiting, scanning, and verification.
 
# Installation

```bash
conda env create -n art --file art_env_export.yml
conda activate art
```
Disclaimer: We observed that CUDA versions affect reproducibility of the pre-training process. Hence, we strongly recommend using CUDA version 11.x for reproducing our work - versions 11.3 and 11.4 have worked for us. 

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

Acknowledgement - A portion of our code is adapted from the [repository](https://github.com/djiajunustc/TransVG) for [TransVG](https://openaccess.thecvf.com/content/ICCV2021/papers/Deng_TransVG_End-to-End_Visual_Grounding_With_Transformers_ICCV_2021_paper.pdf) model. We would like to thank the authors Deng et al. for open-sourcing their code. 

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
