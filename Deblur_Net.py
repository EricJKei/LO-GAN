import tensorflow as tf
from tensorflow.contrib.layers import flatten
from ops import *
import numpy as np
from data_loader import dataloader

class Deblur_Net():
    
    def __init__(self, args):
        
        self.data_loader = dataloader(args)
        self.channel = args.channel
        self.n_feats = args.n_feats
        self.mode = args.mode
        self.batch_size = args.batch_size      
        self.num_of_down_scale = args.num_of_down_scale
        self.gen_resblocks = args.gen_resblocks
        self.discrim_blocks = args.discrim_blocks
        self.vgg_path = args.vgg_path
        
        self.learning_rate = args.learning_rate
        self.decay_step = args.decay_step
        
    def down_scaling_feature(self, name, x, n_feats):
        print("down:", n_feats*2)
        x = Conv(name = name + 'conv', x = x, filter_size = 3, in_filters = n_feats, out_filters = n_feats * 2, strides = 2, padding = 'SAME')
        x = instance_norm(name = name + 'instance_norm', x = x, dim = n_feats * 2)
        x = tf.nn.relu(x)
        
        return x
    
    def up_scaling_feature(self, name, x, n_feats):
        print(n_feats//2)
        x = Conv_transpose(name = name + 'deconv', x = x, filter_size = 3, in_filters = n_feats, out_filters = n_feats // 2, fraction = 2, padding = 'SAME')
        x = instance_norm(name = name + 'instance_norm', x = x, dim = n_feats // 2)
        x = tf.nn.relu(x)

        return x
    
    def res_block(self, name, x, n_feats):
        
        _res = x

        x = tf.pad(x, [[0,0],[1,1],[1,1],[0,0]], mode = 'REFLECT')
        x = Conv(name = name + 'conv1', x = x, filter_size = 3, in_filters = n_feats, out_filters = n_feats, strides = 1, padding = 'VALID')
        x = instance_norm(name = name + 'instance_norm1', x = x, dim = n_feats)
        x = tf.nn.relu(x)
        
        x = tf.pad(x, [[0,0],[1,1],[1,1],[0,0]], mode = 'REFLECT')
        x = Conv(name = name + 'conv2', x = x, filter_size = 3, in_filters = n_feats, out_filters = n_feats, strides = 1, padding = 'VALID')
        x = instance_norm(name = name + 'instance_norm2', x = x, dim = n_feats)

        x = x + _res
        
        return x
    
    def generator(self, x, reuse = False, name = 'generator'):
        
        with tf.variable_scope(name_or_scope = name, reuse = reuse):
            _res = x

            x = tf.pad(x, [[0,0],[3,3],[3,3],[0,0]], mode = 'REFLECT')
            x = Conv(name = 'conv1', x = x, filter_size = 7, in_filters = self.channel, out_filters = self.n_feats, strides = 1, padding = 'VALID')
            x = instance_norm(name = 'inst_norm1', x = x, dim = self.n_feats)
            x = tf.nn.relu(x)

            '''
            for i in range(self.num_of_down_scale):
                x = self.down_scaling_feature(name = 'down_%02d'%i, x = x, n_feats = self.n_feats * (i + 1))
            for i in range(self.gen_resblocks):
                x = self.res_block(name = 'res_%02d'%i, x = x, n_feats = self.n_feats * (2 ** self.num_of_down_scale))
            for i in range(self.num_of_down_scale):
                x = self.up_scaling_feature(name = 'up_%02d'%i, x = x, n_feats = self.n_feats * (2 ** (self.num_of_down_scale - i)))
            '''

            tmp_x = []
            for i in range(self.num_of_down_scale):
                x = self.down_scaling_feature(name = 'down_%02d'%i, x = x, n_feats = self.n_feats * (i + 1))
                tmp_x.append(tf.identity(x))
            for i in range(self.gen_resblocks):
                x = self.res_block(name = 'res_%02d'%i, x = x, n_feats = self.n_feats * (2 ** self.num_of_down_scale))
            for i in range(self.num_of_down_scale):
                cal_x = tf.concat([x, tmp_x[1-i]], axis=-1)
                in_filters = int(cal_x.shape[-1])
                x = self.up_scaling_feature(name = 'up_%02d'%i, x = cal_x, n_feats = in_filters)

            x = tf.pad(x, [[0,0],[3,3],[3,3],[0,0]], mode = 'REFLECT')
            #x = Conv(name = 'conv_last', x = x, filter_size = 7, in_filters = self.n_feats, out_filters = self.channel, strides = 1, padding = 'VALID')
            x = Conv(name = 'conv_last', x = x, filter_size = 7, in_filters = int(x.shape[-1]), out_filters = self.channel, strides = 1, padding = 'VALID')            
            x = tf.nn.tanh(x)
            x = x + _res
            x = tf.clip_by_value(x, -1.0, 1.0)

            return x

    def discriminator(self, x, reuse = False, name = 'discriminator'):
        
        with tf.variable_scope(name_or_scope = name, reuse = reuse):
            x = Conv(name = 'conv1', x = x, filter_size = 4, in_filters = self.channel, out_filters = self.n_feats, strides = 2, padding = "SAME")
            x = instance_norm(name = 'inst_norm1', x = x, dim = self.n_feats)
            x = tf.nn.leaky_relu(x)
            
            prev = 1
            n = 1
            
            for i in range(self.discrim_blocks):
                prev = n
                n = min(2 ** (i+1), 16)

                x = Conv(name = 'conv%02d'%i, x = x, filter_size = 4, in_filters = self.n_feats * prev, out_filters = self.n_feats * n, strides = 2, padding = "SAME")
                x = instance_norm(name = 'instance_norm%02d'%i, x = x, dim = self.n_feats * n)
                x = tf.nn.leaky_relu(x)
                
            prev = n
            n = min(2**self.discrim_blocks, 16)

            x = Conv(name = 'conv_d1', x = x, filter_size = 4, in_filters = self.n_feats * prev, out_filters = self.n_feats * n*2, strides = 2, padding = "SAME")
            x = instance_norm(name = 'instance_norm_d1', x = x, dim = self.n_feats * n)
            x = tf.nn.leaky_relu(x)
            
            x = Conv(name = 'conv_d2', x = x, filter_size = 4, in_filters = self.n_feats * n, out_filters = 2048, strides = 1, padding = "SAME")
            x = flatten(x)

            return x
    
        
    def build_graph(self):

        self.data_loader.build_loader()

        x = self.data_loader.next_batch[0]
        label = self.data_loader.next_batch[1]

        x = (2.0 * x / 255.0) - 1.0
        label = (2.0 * label / 255.0) - 1.0
        
        self.gene_img = self.generator(x, reuse = False)

        self.output = (self.gene_img + 1.0) * 255.0 / 2.0
        self.output = tf.round(self.output)
        self.output = tf.cast(self.output, tf.uint8)
