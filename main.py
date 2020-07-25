import tensorflow as tf
from Deblur_Net import Deblur_Net
from mode import *
import argparse
import os
import sys


parser = argparse.ArgumentParser()

def str2bool(v):
    return v.lower() in ('true')

## Model specification
parser.add_argument("--channel", type = int, default = 3)
parser.add_argument("--n_feats", type = int, default = 64)
parser.add_argument("--num_of_down_scale", type = int, default = 2)
parser.add_argument("--gen_resblocks", type = int, default = 9)
parser.add_argument("--discrim_blocks", type = int, default = 4)

## Data specification 
parser.add_argument("--test_Sharp_path", type = str, default = "sharp")
parser.add_argument("--test_Blur_path", type = str, default = "blur")
parser.add_argument("--vgg_path", type = str, default = "./vgg19/vgg19.npy")
parser.add_argument("--patch_size", type = int, default = 256)
parser.add_argument("--loss_components", type = str, default = './experiment/validation')
parser.add_argument("--result_path", type = str, default = "./result")
parser.add_argument("--model_path", type = str, default = "model")

## Optimization
parser.add_argument("--batch_size", type = int, default = 1)
parser.add_argument("--max_epoch", type = int, default = 250)
parser.add_argument("--learning_rate", type = float, default = 1e-4)
parser.add_argument("--decay_step", type = int, default = 125)
parser.add_argument("--test_with_train", type = str2bool, default = True)
parser.add_argument("--save_test_result", type = str2bool, default = False)

## Training or test specification
parser.add_argument("--mode", type = str, default = "test_only")
parser.add_argument("--critic_updates", type = int, default = 1)
parser.add_argument("--augmentation", type = str2bool, default = False)
parser.add_argument("--load_X", type = int, default = 256)
parser.add_argument("--load_Y", type = int, default = 256)
parser.add_argument("--fine_tuning", type = str2bool, default = True)
parser.add_argument("--log_freq", type = int, default = 1)
parser.add_argument("--model_save_freq", type = int, default = 20)
parser.add_argument("--test_batch", type = int, default = 1)
parser.add_argument("--pre_trained_model", type = str, default = "ADNI")
parser.add_argument("--chop_forward", type = str2bool, default = False)
parser.add_argument("--chop_size", type = int, default = 8e4)
parser.add_argument("--chop_shave", type = int, default = 16)
parser.add_argument("--init_epoch", type = int, default = 0)
parser.add_argument("--gpu", type = str, default = "0")


args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

args.test_Sharp_path = os.path.join("./"+args.pre_trained_model, args.test_Sharp_path)
args.test_Blur_path = os.path.join("./"+args.pre_trained_model, args.test_Blur_path)
args.pre_trained_model = os.path.join("./model", args.pre_trained_model, "DeblurGAN-250")

if not os.path.exists(args.result_path):
    os.makedirs(args.result_path)
if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)


model = Deblur_Net(args)
model.build_graph()

print("Build model!")

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep = 1)

test_only(args, model, sess, saver)
    

