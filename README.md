## PCLDet
This is Pytorch implementation for "PCLDet: Prototypical Contrastive Learning for Fine-Grained Object Detection in Remote Sensing Images"

## Overview
The overall framework for PCLDet.
![image](https://github.com/G-Naughty/PCLDet/assets/47738176/d6ff4688-58eb-4c3f-9268-46f1faf33654)


## Running the code
1. Run tools/train.py to train a new model.
2. Set "--config" and "--work-dir" as your path.
For example:
 In tools/train.py, set "--config" as config/GGM/con_redet_re50_refpn_1x_fair1m.py
All PCLDet config files are placed in the path "config/GGM" and start with "con_".
3. Run tools/test.py to test the model.

## Citation

If you find this project useful in your research, please consider cite:

```bibtex
@article{ouyang2023pcldet,
  title={PCLDet: Prototypical Contrastive Learning for Fine-grained Object Detection in Remote Sensing Images},
  author={Ouyang, Lihan and Guo, Guangmiao and Fang, Leyuan and Ghamisi, Pedram and Yue, Jun},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2023},
  publisher={IEEE}
}
```
