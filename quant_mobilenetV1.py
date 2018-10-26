import tensorflow as tf
import json
import numpy as np


def quant_conv(i, w, b, padding,
                  stride,
                  i_z, w_z, r_z,
                  m):
 
#    m = round(m,7)
#    print(m)
#    m = tf.cast(m, tf.float32)
    
    temp = tf.nn.conv2d(i-i_z, w-w_z, [1, stride, stride, 1], padding) + b
    
    temp = tf.rint(temp)

    temp = tf.multiply(temp, m)
    
    temp = tf.rint(temp)
    
    temp = temp + r_z
    
    temp = tf.clip_by_value(temp, 0, 255)
    
    temp = tf.rint(temp)   
    
    return temp



def quant_depthwise_conv(i, w, b, padding,
                         stride,
                         i_z, w_z, r_z,
                         m):

#    m = round(m,7)
#    print(m)
#    m = tf.cast(m, tf.float32)

    
    temp = tf.nn.depthwise_conv2d(i-i_z, w-w_z, [1, stride, stride, 1], padding) + b
    
    temp = tf.rint(temp)
    
    temp = tf.multiply(temp, m)

    temp = tf.rint(temp)
   
    temp = temp + r_z
    
    temp = tf.clip_by_value(temp, 0, 255)
    
    temp = tf.rint(temp)   
    
    return temp



def get_weight(json_file): 
    
    with open (json_file, 'r', encoding='utf-8') as f:
        ops = json.loads(json.load(f))
        
    mobilenet_v1_w_index = ['MobilenetV1/MobilenetV1/Conv2d_0/weights_quant/FakeQuantWithMinMaxVars',
                      'MobilenetV1/MobilenetV1/Conv2d_1_depthwise/weights_quant/FakeQuantWithMinMaxVars',
                      'MobilenetV1/MobilenetV1/Conv2d_1_pointwise/weights_quant/FakeQuantWithMinMaxVars',
                      'MobilenetV1/MobilenetV1/Conv2d_2_depthwise/weights_quant/FakeQuantWithMinMaxVars',
                      'MobilenetV1/MobilenetV1/Conv2d_2_pointwise/weights_quant/FakeQuantWithMinMaxVars',
                      'MobilenetV1/MobilenetV1/Conv2d_3_depthwise/weights_quant/FakeQuantWithMinMaxVars',
                      'MobilenetV1/MobilenetV1/Conv2d_3_pointwise/weights_quant/FakeQuantWithMinMaxVars',
                      'MobilenetV1/MobilenetV1/Conv2d_4_depthwise/weights_quant/FakeQuantWithMinMaxVars',
                      'MobilenetV1/MobilenetV1/Conv2d_4_pointwise/weights_quant/FakeQuantWithMinMaxVars',
                      'MobilenetV1/MobilenetV1/Conv2d_5_depthwise/weights_quant/FakeQuantWithMinMaxVars',
                      'MobilenetV1/MobilenetV1/Conv2d_5_pointwise/weights_quant/FakeQuantWithMinMaxVars',
                      'MobilenetV1/MobilenetV1/Conv2d_6_depthwise/weights_quant/FakeQuantWithMinMaxVars',
                      'MobilenetV1/MobilenetV1/Conv2d_6_pointwise/weights_quant/FakeQuantWithMinMaxVars',
                      'MobilenetV1/MobilenetV1/Conv2d_7_depthwise/weights_quant/FakeQuantWithMinMaxVars',
                      'MobilenetV1/MobilenetV1/Conv2d_7_pointwise/weights_quant/FakeQuantWithMinMaxVars',
                      'MobilenetV1/MobilenetV1/Conv2d_8_depthwise/weights_quant/FakeQuantWithMinMaxVars',
                      'MobilenetV1/MobilenetV1/Conv2d_8_pointwise/weights_quant/FakeQuantWithMinMaxVars',
                      'MobilenetV1/MobilenetV1/Conv2d_9_depthwise/weights_quant/FakeQuantWithMinMaxVars',
                      'MobilenetV1/MobilenetV1/Conv2d_9_pointwise/weights_quant/FakeQuantWithMinMaxVars',
                      'MobilenetV1/MobilenetV1/Conv2d_10_depthwise/weights_quant/FakeQuantWithMinMaxVars',
                      'MobilenetV1/MobilenetV1/Conv2d_10_pointwise/weights_quant/FakeQuantWithMinMaxVars',
                      'MobilenetV1/MobilenetV1/Conv2d_11_depthwise/weights_quant/FakeQuantWithMinMaxVars',
                      'MobilenetV1/MobilenetV1/Conv2d_11_pointwise/weights_quant/FakeQuantWithMinMaxVars',
                      'MobilenetV1/MobilenetV1/Conv2d_12_depthwise/weights_quant/FakeQuantWithMinMaxVars',
                      'MobilenetV1/MobilenetV1/Conv2d_12_pointwise/weights_quant/FakeQuantWithMinMaxVars',
                      'MobilenetV1/MobilenetV1/Conv2d_13_depthwise/weights_quant/FakeQuantWithMinMaxVars',
                      'MobilenetV1/MobilenetV1/Conv2d_13_pointwise/weights_quant/FakeQuantWithMinMaxVars',
                      'MobilenetV1/Logits/Conv2d_1c_1x1/weights_quant/FakeQuantWithMinMaxVars'
                      ]
    
    
    mobilenet_v1_b_index = ['MobilenetV1/MobilenetV1/Conv2d_0/Conv2D_Fold_bias',
                      'MobilenetV1/MobilenetV1/Conv2d_1_depthwise/depthwise_Fold_bias',
                      'MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Conv2D_Fold_bias',
                      'MobilenetV1/MobilenetV1/Conv2d_2_depthwise/depthwise_Fold_bias',
                      'MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Conv2D_Fold_bias',
                      'MobilenetV1/MobilenetV1/Conv2d_3_depthwise/depthwise_Fold_bias',
                      'MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Conv2D_Fold_bias',
                      'MobilenetV1/MobilenetV1/Conv2d_4_depthwise/depthwise_Fold_bias',
                      'MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Conv2D_Fold_bias',
                      'MobilenetV1/MobilenetV1/Conv2d_5_depthwise/depthwise_Fold_bias',
                      'MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Conv2D_Fold_bias',
                      'MobilenetV1/MobilenetV1/Conv2d_6_depthwise/depthwise_Fold_bias',
                      'MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Conv2D_Fold_bias',
                      'MobilenetV1/MobilenetV1/Conv2d_7_depthwise/depthwise_Fold_bias',
                      'MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Conv2D_Fold_bias',
                      'MobilenetV1/MobilenetV1/Conv2d_8_depthwise/depthwise_Fold_bias',
                      'MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Conv2D_Fold_bias',
                      'MobilenetV1/MobilenetV1/Conv2d_9_depthwise/depthwise_Fold_bias',
                      'MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Conv2D_Fold_bias',
                      'MobilenetV1/MobilenetV1/Conv2d_10_depthwise/depthwise_Fold_bias',
                      'MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Conv2D_Fold_bias',
                      'MobilenetV1/MobilenetV1/Conv2d_11_depthwise/depthwise_Fold_bias',
                      'MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Conv2D_Fold_bias',
                      'MobilenetV1/MobilenetV1/Conv2d_12_depthwise/depthwise_Fold_bias',
                      'MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Conv2D_Fold_bias',
                      'MobilenetV1/MobilenetV1/Conv2d_13_depthwise/depthwise_Fold_bias',
                      'MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Conv2D_Fold_bias',                                   
                      'MobilenetV1/Logits/Conv2d_1c_1x1/Conv2D_bias'
                      ]  
    
    
    mobilenet_v1_r_index =['MobilenetV1/MobilenetV1/Conv2d_0/Relu6',
                           'MobilenetV1/MobilenetV1/Conv2d_1_depthwise/Relu6',
                           'MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Relu6',
                           'MobilenetV1/MobilenetV1/Conv2d_2_depthwise/Relu6',
                           'MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Relu6',
                           'MobilenetV1/MobilenetV1/Conv2d_3_depthwise/Relu6',
                           'MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Relu6',
                           'MobilenetV1/MobilenetV1/Conv2d_4_depthwise/Relu6',
                           'MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Relu6',
                           'MobilenetV1/MobilenetV1/Conv2d_5_depthwise/Relu6',
                           'MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Relu6',
                           'MobilenetV1/MobilenetV1/Conv2d_6_depthwise/Relu6',
                           'MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Relu6',
                           'MobilenetV1/MobilenetV1/Conv2d_7_depthwise/Relu6',
                           'MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Relu6',
                           'MobilenetV1/MobilenetV1/Conv2d_8_depthwise/Relu6',
                           'MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Relu6',
                           'MobilenetV1/MobilenetV1/Conv2d_9_depthwise/Relu6',
                           'MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Relu6',
                           'MobilenetV1/MobilenetV1/Conv2d_10_depthwise/Relu6',
                           'MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Relu6',
                           'MobilenetV1/MobilenetV1/Conv2d_11_depthwise/Relu6',
                           'MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Relu6',
                           'MobilenetV1/MobilenetV1/Conv2d_12_depthwise/Relu6',
                           'MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Relu6',
                           'MobilenetV1/MobilenetV1/Conv2d_13_depthwise/Relu6',
                           'MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu6',
                           'MobilenetV1/Logits/Conv2d_1c_1x1/BiasAdd'                                              
                           ]
    
    w_array = []
    b_array = []
    w_z = []
    b_s = []
    r_s = []
    #m = []
    #b_z = []
    
    stride = [2,1,1,2,1,1,1,2,1,1,1,2,1,
              1,1,1,1,1,1,1,1,1,1,
              2,1,1,1,1]
    
    
    for i in range(len(mobilenet_v1_w_index)):
        
        temp = ops[mobilenet_v1_w_index[i]]['numpy_array']
        w_array.append(np.array(temp, dtype=np.float32).transpose([1,2,3,0]))
            
        temp = ops[mobilenet_v1_w_index[i]]['quantization'][1]
        w_z.append(temp)
    
        
        temp = ops[mobilenet_v1_b_index[i]]['numpy_array']
        b_array.append(np.array(temp, dtype=np.float32))
        
        b__s = ops[mobilenet_v1_b_index[i]]['quantization'][0]
    #    b__s = round(b__s,7)
        b_s.append(b__s)
        r__s = ops[mobilenet_v1_r_index[i]]['quantization'][0]
    #    r__s = round(r__s,7)    
        r_s.append(r__s) 
        
        
    b_s = np.array(b_s, np.float32)    
    r_s = np.array(r_s, np.float32)    
    m =   np.array(b_s/r_s, np.float32)
    
    i_z = ops['input']['quantization'][1]
    o_z = ops['MobilenetV1/Logits/Conv2d_1c_1x1/BiasAdd']['quantization'][1]
        
    return w_array, b_array, stride, w_z, m, i_z, o_z  
        
 


def quant_mobilenetV1_base(inputs,
                          w_array, b_array, stride,
                          w_z, m,
                          i_z, o_z):

#    i_s = 0.007874015718698502
#    i_z = 128
    
    # Conv2d_0 
    net0 = quant_conv(inputs, w_array[0], b_array[0], 'SAME',
                      stride[0],
                      i_z, w_z[0], 0,
                      m[0])
    
    
    # Conv2d_1
    net1_ = quant_depthwise_conv(net0, w_array[1], b_array[1], 'SAME',
                                  stride[1],
                                  0, w_z[1], 0,
                                  m[1])
        
    net1 = quant_conv(net1_, w_array[2], b_array[2], 'SAME',
                      stride[2],
                      0, w_z[2], 0,
                      m[2])
    
    # Conv2d_2
    net2_ = quant_depthwise_conv(net1, w_array[3], b_array[3], 'SAME',
                                  stride[3],
                                  0, w_z[3], 0,
                                  m[3])
        
    net2 = quant_conv(net2_, w_array[4], b_array[4], 'SAME',
                      stride[4],
                      0, w_z[4], 0,
                      m[4])
    
    # Conv2d_3
    net3_ = quant_depthwise_conv(net2, w_array[5], b_array[5], 'SAME',
                                  stride[5],
                                  0, w_z[5], 0,
                                  m[5])
        
    net3 = quant_conv(net3_, w_array[6], b_array[6], 'SAME',
                      stride[6],
                      0, w_z[6], 0,
                      m[6])
    
    
    # Conv2d_4
    net4_ = quant_depthwise_conv(net3, w_array[7], b_array[7], 'SAME',
                                  stride[7],
                                  0, w_z[7], 0,
                                  m[7])
        
    net4 = quant_conv(net4_, w_array[8], b_array[8], 'SAME',
                      stride[8],
                      0, w_z[8], 0,
                      m[8])
    
    
    
    # Conv2d_5
    net5_ = quant_depthwise_conv(net4, w_array[9], b_array[9], 'SAME',
                                  stride[9],
                                  0, w_z[9], 0,
                                  m[9])
        
    net5 = quant_conv(net5_, w_array[10], b_array[10], 'SAME',
                      stride[10],
                      0, w_z[10], 0,
                      m[10])
    
    
    
    # Conv2d_6
    net6_ = quant_depthwise_conv(net5, w_array[11], b_array[11], 'SAME',
                                  stride[11],
                                  0, w_z[11], 0,
                                  m[11])
        
    net6 = quant_conv(net6_, w_array[12], b_array[12], 'SAME',
                      stride[12],
                      0, w_z[12], 0,
                      m[12])
    
    
    # Conv2d_7
    net7_ = quant_depthwise_conv(net6, w_array[13], b_array[13], 'SAME',
                                  stride[13],
                                  0, w_z[13], 0,
                                  m[13])
        
    net7 = quant_conv(net7_, w_array[14], b_array[14], 'SAME',
                      stride[14],
                      0, w_z[14], 0,
                      m[14])
    
    
    # Conv2d_8
    net8_ = quant_depthwise_conv(net7, w_array[15], b_array[15], 'SAME',
                                  stride[15],
                                  0, w_z[15], 0,
                                  m[15])
        
    net8 = quant_conv(net8_, w_array[16], b_array[16], 'SAME',
                      stride[16],
                      0, w_z[16], 0,
                      m[16])
    
    
    # Conv2d_9
    net9_ = quant_depthwise_conv(net8, w_array[17], b_array[17], 'SAME',
                                  stride[17],
                                  0, w_z[17], 0,
                                  m[17])
        
    net9 = quant_conv(net9_, w_array[18], b_array[18], 'SAME',
                      stride[18],
                      0, w_z[18], 0,
                      m[18])
     
    
    # Conv2d_10
    net10_ = quant_depthwise_conv(net9, w_array[19], b_array[19], 'SAME',
                                  stride[19],
                                  0, w_z[19], 0,
                                  m[19])
        
    net10 = quant_conv(net10_, w_array[20], b_array[20], 'SAME',
                      stride[20],
                      0, w_z[20], 0,
                      m[20])
    
    
    # Conv2d_11
    net11_ = quant_depthwise_conv(net10, w_array[21], b_array[21], 'SAME',
                                  stride[21],
                                  0, w_z[21], 0,
                                  m[21])
        
    net11 = quant_conv(net11_, w_array[22], b_array[22], 'SAME',
                      stride[22],
                      0, w_z[22], 0,
                      m[22])
    
    
    
    # Conv2d_12
    net12_ = quant_depthwise_conv(net11, w_array[23], b_array[23], 'SAME',
                                  stride[23],
                                  0, w_z[23], 0,
                                  m[23])
        
    net12 = quant_conv(net12_, w_array[24], b_array[24], 'SAME',
                      stride[24],
                      0, w_z[24], 0,
                      m[24])
    
    
    # Conv2d_13
    net13_ = quant_depthwise_conv(net12, w_array[25], b_array[25], 'SAME',
                                  stride[25],
                                  0, w_z[25], 0,
                                  m[25])
        
    net13 = quant_conv(net13_, w_array[26], b_array[26], 'SAME',
                      stride[26],
                      0, w_z[26], 0,
                      m[26])
    
    
    
    # avg pool
    pool = tf.nn.avg_pool(net13, [1,7,7,1], [1,2,2,1], 'VALID')
    pool = tf.rint(pool)

    
    # logit    
    logit = quant_conv(pool, w_array[27], b_array[27], 'SAME',
                      stride[27],
                      0, w_z[27], o_z,
                      m[27])

    
    logit = tf.squeeze(logit, [1, 2])

    return logit



def quant_mobilenetV1(inputs, json_file):

    w_array, b_array, stride, w_z, m, i_z, o_z  = get_weight(json_file)

    logit = quant_mobilenetV1_base(inputs,
                                  w_array, b_array, stride,
                                  w_z, m,
                                  i_z, o_z)
    
    return logit

