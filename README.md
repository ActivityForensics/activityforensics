# [CVPR 2026] ActivityForensics: A Comprehensive Benchmark for Localizing Manipulated Activity in Videos
Peijun Bao,   Anwei Luo,   Gang Pan,   Alex C. Kot,   Xudong Jiang

- [[Project Page]](https://activityforensics.github.io/)
[[Code]](https://github.com/ActivityForensics/activityforensics) 
- [[Paper]](https://activityforensics.github.io/activityforensics/files/ActivityForensics_CVPR2026.pdf) 
[[Supp]](https://activityforensics.github.io/activityforensics/files/ActivityForensics_Supp_CVPR2026.pdf)

- [[Dataset]](https://entuedu-my.sharepoint.com/:f:/g/personal/peijun_bao_staff_main_ntu_edu_sg/IgB-nNOfNgG6QaqwV2CPjQY9AdDht4epuR3G-IhOK7Ihbbg) (password: ActivityForensics)

- If this dataset is useful for your work, we would appreciate it if you could cite our paper.
```bibtex
@inproceedings{bao2026activityforensics,
    title={ActivityForensics: A Comprehensive Benchmark for Localizing Manipulated Activity in Videos},
    author={Bao, Peijun and Luo, Anwei and Pan, Gang and Kot, Alex C. and Jiang, Xudong},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2026}
}
```

## News 
- 🔥 **[2026.03.29]** We release the extracted features and annotations.
- 🔥 **[2026.03.27]** We open-sourced the code.
- 🔥 **[2026.03.01]** Our paper has been accepted to CVPR 2026



## Dataset Preparation

- You can download the dataset at [OneDrive](https://entuedu-my.sharepoint.com/:f:/g/personal/peijun_bao_staff_main_ntu_edu_sg/IgB-nNOfNgG6QaqwV2CPjQY9AdDht4epuR3G-IhOK7Ihbbg)  (password: ActivityForensics).
- We provide pre-extracted features for all videos to enable faster and more convenient experimentation. To run the code, you only need to download the extracted features, raw videos are not required.
```
data/
  ├── feat/    # Pre-extracted video features
  └── annot/   # Annotations
```
## Build and install dependencies

```bash
cd libs/utils
python setup.py build
python setup.py install
cd ../../
```

---

## Training

Run the following command to start training:

```bash
python train.py --config configs/tadiff.yaml
```

---





## Motivation: In which video are the activities AI-manipulated?

| [![Video A](https://activityforensics.github.io/activityforensics/fig/02_samples/0UK3H_fast.jpg)](https://activityforensics.github.io/activityforensics/fig/02_samples/0UK3H_crop.mp4) | ![arrow](https://activityforensics.github.io/activityforensics/fig/02_samples/main_code.png) | [![Video B](https://activityforensics.github.io/activityforensics/fig/02_samples/0UK3H_fast.jpg)](https://activityforensics.github.io/activityforensics/fig/02_samples/0UK3H+2.10=8.20=charades@test_delete@0UK3H@639@wan_crop.mp4) |
|:---------------------------:|:---------------------------:|:---------------------------:|
|   Original Video | (video can be played in new tab) | Manipulated Video |



## Beyond Appearance: The First Benchmark for Activity-Level Forensics

| ![](https://activityforensics.github.io/activityforensics/fig/01_motivation/1.png) | ![](https://activityforensics.github.io/activityforensics/fig/01_motivation/2.png) |
|:---------------------------:|:---------------------------:|
| Appearance-level forgery | Activity-level forgery |


<p align="center">
  <img src="https://activityforensics.github.io/activityforensics/fig/01_motivation/table.png" width="60%">
</p>



## Grounding-Assisted Dataset Construction

![](https://activityforensics.github.io/activityforensics/fig/03_generation/main.png)

<p align="center"><em>Overview of grounding-assisted dataset generation pipeline.</em></p>


## Dataset and Code License

This code and dataset are for **research purposes only** and **non-commercial use only**.  

For more details on dataset policy, and license terms, please refer to [DATA_POLICY.md](DATA_POLICY.md).

## Acknowledgement

Our code is based on [ActionFormer](https://github.com/happyharrycn/actionformer_release).  We thank the authors for their excellent work.

We also thank the authors of several open-source video generation and editing methods:
- [Wan](https://github.com/Wan-Video/Wan2.1)  
- [SciFi](https://github.com/GVCLab/Sci-Fi)  
- [FCVG](https://github.com/Tian-one/FCVG)
- [LTX](https://github.com/Lightricks/LTX-Video)  
- [VACE](https://github.com/ali-vilab/VACE)

## Contact

You are welcome to send pull requests or share some ideas with me.  
Peijun Bao (peijun001@e.ntu.edu.sg)  




