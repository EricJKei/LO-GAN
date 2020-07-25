import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
import math


def Conv(name, x, filter_size, in_filters, out_filters, strides, padding):

    with tf.variable_scope(name):
        n= filter_size * filter_size * out_filters
        kernel = tf.get_variable('filter', [filter_size, filter_size, in_filters, out_filters],tf.float32, initializer=tf.random_normal_initializer(stddev = 0.01))
        bias = tf.get_variable('bias',[out_filters],tf.float32, initializer = tf.zeros_initializer())
        
        return tf.nn.conv2d(x, kernel, [1,strides,strides,1], padding = padding) + bias
    


def Conv_transpose(name, x, filter_size, in_filters, out_filters, fraction = 2, padding = "SAME"):
    
    with tf.variable_scope(name):
        n = filter_size * filter_size * out_filters
        kernel = tf.get_variable('filter', [filter_size, filter_size, out_filters, in_filters], tf.float32, initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)) )
        size = tf.shape(x)
        output_shape = tf.stack([size[0], size[1] * fraction, size[2] * fraction, out_filters])
        x = tf.nn.conv2d_transpose(x, kernel, output_shape, [1, fraction, fraction, 1], padding)
        
        return x
        
def instance_norm(name, x, dim, affine = False, BN_decay = 0.999, BN_epsilon = 1e-3):

    mean, variance = tf.nn.moments(x, axes = [1, 2])
    x = (x - mean) / ((variance + BN_epsilon) ** 0.5)

    if affine :
        beta = tf.get_variable(name = name + "beta", shape = dim, dtype = tf.float32,
                               initializer = tf.constant_initializer(0.0, tf.float32))
        gamma = tf.get_variable(name + "gamma", dim, tf.float32,
                                initializer = tf.constant_initializer(1.0, tf.float32))
        x = gamma * x + beta

    return x

def Dense(input, in_units, out_units, name):
    W = tf.get_variable(shape=[in_units, out_units], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(),
                        name=name+'_weights')
    bias = tf.get_variable(shape = [out_units], dtype=tf.float32, name=name+'_bias')
    return tf.matmul(input, W) + bias


# 实现Batch Normalization
# def instance_norm(name, x, dim, is_training = True, BN_decay=0.9,BN_epsilon=1e-5):
#
#     # 获取输入维度并判断是否匹配卷积层(4)或者全连接层(2)
#     shape = x.shape
#     assert len(shape) in [2,4]
#
#     param_shape = shape[-1]
#     with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
#         # 声明BN中唯一需要学习的两个参数，y=gamma*x+beta
#         gamma = tf.get_variable('gamma',param_shape,initializer=tf.constant_initializer(1))
#         beta  = tf.get_variable('beat', param_shape,initializer=tf.constant_initializer(0))
#
#         # 计算当前整个batch的均值与方差
#         axes = list(range(len(shape)-1))
#         batch_mean, batch_var = tf.nn.moments(x,axes,name='moments')
#
#         # 采用滑动平均更新均值与方差
#         ema = tf.train.ExponentialMovingAverage(BN_decay)
#
#         def mean_var_with_update():
#             ema_apply_op = ema.apply([batch_mean,batch_var])
#             with tf.control_dependencies([ema_apply_op]):
#                 return tf.identity(batch_mean), tf.identity(batch_var)
#
#         # 训练时，更新均值与方差，测试时使用之前最后一次保存的均值与方差
#         mean, var = tf.cond(tf.equal(is_training,True),mean_var_with_update,
#                 lambda:(ema.average(batch_mean),ema.average(batch_var)))
#
#         # 最后执行batch normalization
#         return tf.nn.batch_normalization(x,mean,var,beta,gamma,BN_epsilon)
