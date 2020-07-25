# LO-GAN

Tensorflow implementation of the paper Retrospective Correction of Motion Artifacts in MRI using Generative Adversarial Network with Local Optimization.

## How to run

### Prerequisites
- tensorflow==1.9
- tqdm==4.36.1


Download trained weights from [Baidu Drive](https://pan.baidu.com/s/1Ah__t93W91NR6ueXeHSNzQ), access code：vsn9. 

Put the weights into 
```bash
./model/dataset_name(ADNI/ABID)/
```
To test model, you should run:
```bash
python3 main.py --pre_trained_model dataset_name
```
or
```bash
./test.sh
```
