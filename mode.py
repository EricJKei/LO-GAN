import os
import tensorflow as tf
from PIL import Image
import numpy as np
import time
# import util
from skimage.measure import compare_ssim as ssim
from tqdm import tqdm


def test_only(args, model, sess, saver):

    saver.restore(sess,args.pre_trained_model)
    print("saved model is loaded for test only!")
    print("model path is %s"%args.pre_trained_model)
    
    blur_img_name = sorted(os.listdir(args.test_Blur_path))

    sess.run(model.data_loader.init_op['val_init'])
    import time
    start_time = time.time()
    for i in range(len(blur_img_name)):
        output = sess.run(model.output)
        output = Image.fromarray(output[0,:,:,0])
        split_name = blur_img_name[i].split('.')
        output.save(os.path.join(args.result_path, '%s_sharp.png'%(''.join(map(str, split_name[:-1])))))
    print(time.time()-start_time)
