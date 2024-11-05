# Geometry-Cloak
## This is the official implementation of Geometry Cloak: Preventing TGS-based 3D Reconstruction from Copyrighted Images (NeurIPS 2024) [[Arxiv]](https://arxiv.org/abs/2410.22705)

# Usage

We implement our Geometry Cloak on [Triplane Meets Gaussian Splatting: Fast and Generalizable Single-View 3D Reconstruction with Transformers](https://arxiv.org/abs/2312.09147). 

To set the running environment, please refer to [TGS](https://github.com/VAST-AI-Research/TriplaneGaussian).

The codes are provided in test_geometry_cloak.ipynb

If you find our paper useful for your work please cite:
```
@inproceedings{song2024geometry,
  author    = {Song, Qi and Luo, Ziyuan and Cheung, Ka Chun and See, Simon and Wan, Renjie},
  title     = {Geometry Cloak: Preventing TGS-based 3D Reconstruction from Copyrighted Images},
  booktitle   = {NeurIPS},
  year      = {2024},
}
```

# Acknowledgement

* Credits to [Hadi Salman](https://hadisalman.com/) and [Zixin Zou](https://github.com/zouzx) for the amazing [Photoguard](https://github.com/MadryLab/photoguard) and [TriplaneGaussian](https://github.com/VAST-AI-Research/TriplaneGaussian):
  
  ```
  @inproceedings{zou2024triplane,
  title={Triplane meets gaussian splatting: Fast and generalizable single-view 3d reconstruction with transformers},
  author={Zou, Zi-Xin and Yu, Zhipeng and Guo, Yuan-Chen and Li, Yangguang and Liang, Ding and Cao, Yan-Pei and Zhang, Song-Hai},
  booktitle={CVPR},
  pages={10324--10335},
  year={2024}}

  @inproceedings{salman2023raising,
  title={Raising the cost of malicious AI-powered image editing},
  author={Salman, Hadi and Khaddaj, Alaa and Leclerc, Guillaume and Ilyas, Andrew and M{\k{a}}dry, Aleksander},
  booktitle={ICML},
  year={2023}}

    ```


