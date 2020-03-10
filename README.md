# Early-Exit-On-Videos
The base models that we are using are from this git:
https://github.com/kenshohara/3D-ResNets-PyTorch
## Training EE model on hmdb51 dataset:
```
python3 main.py --dataset hmdb51 --video_path <datapat>  --pretrain_path resnext-101-kinetics-hmdb51_split1.pth --annotation_path <annotation_path>  --model ee_resnext --model_depth 101 --batch_size 64 --learning_rate 0.1 --weight_decay 0.00001 --earlyexit_lossweight 0.9  --earlyexit_threshold 0.8 --result_path results/ee_resnext_exit1&
```
## Training EE model on ucf101 dataset:
```
python3 main.py --dataset ucf101 --video_path <datapat>  --pretrain_path resnext-101-kinetics-hmdb51_split1.pth --annotation_path <annotation_path>  --model ee_resnext --model_depth 101 --batch_size 64 --learning_rate 0.1 --weight_decay 0.00001 --earlyexit_lossweight 0.9  --earlyexit_threshold 0.8 --result_path results/ee_resnext_exit1 --n_classes 101&
```
