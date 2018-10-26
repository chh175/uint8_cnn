import tensorflow as tf
import json
import numpy as np
from PIL import Image




#json_file = 'mobilenetv2_flower_quant_8_26.json'
#
#with open (json_file, 'r', encoding='utf-8') as f:
#    ops = json.loads(json.load(f))


# 普通卷积
def quant_conv(i, w, b, padding,
                  stride,
                  i_z, w_z, r_z,
                  m):

    '''
    i:input，输入，为uint8
    w：weight，权重，为uint8
    padding:是否边界填充
    stride：步长
    i_z：input_zero，输入为0时浮点值所对应的整型数值，为uint8
    w_z: weight_zero，权重为0时浮点值所对应的整型数值，为uint8 
    r_z：result_zero，输出为0时浮点值所对应的整型数值，为uint8
    m：缩小因子，将卷积之后的结果乘m，为float32

        
    输出为uint8    
    
    '''    
    
    
#    m = round(m,7)
#    print(m)
#    m = tf.cast(m, tf.float32)
    
    temp = tf.nn.conv2d(i-i_z, w-w_z, [1, stride, stride, 1], padding) + b
    
#    temp = tf.rint(temp)

#   将卷积后的结果乘以一个缩小因子m
    temp = tf.multiply(temp, m)


#   四舍五入为整型    
    temp = tf.rint(temp)
    
    temp = temp + r_z

#   将结果钳位到0~255区间    
    temp = tf.clip_by_value(temp, 0, 255)
    
#    temp = tf.rint(temp)   
    
    return temp


# 深度卷积
def quant_depthwise_conv(i, w, b, padding,
                         stride,
                         i_z, w_z, r_z,
                         m):

    '''
    i:input，输入，为uint8
    w：weight，权重，为uint8
    padding:是否边界填充
    stride：步长
    i_z：input_zero，输入为0时浮点值所对应的整型数值，为uint8
    w_z: weight_zero，权重为0时浮点值所对应的整型数值，为uint8 
    r_z：result_zero，输出为0时浮点值所对应的整型数值，为uint8
    m：缩小因子，将卷积之后的结果乘m，为float32
        
    输出为uint8    
    
    '''    
    
    
#    m = round(m,7)
#    print(m)
#    m = tf.cast(m, tf.float32)

    
    temp = tf.nn.depthwise_conv2d(i-i_z, w-w_z, [1, stride, stride, 1], padding) + b
    
#    temp = tf.rint(temp)
    
    temp = tf.multiply(temp, m)

    temp = tf.rint(temp)
   
    temp = temp + r_z
    
    temp = tf.clip_by_value(temp, 0, 255)
    
#    temp = tf.rint(temp)   
    
    return temp



# 量化的残差加法
def quant_add(i1, i1_s, i1_z,
              i2, i2_s, i2_z,
              i3_s, i3_z):

    '''
    i1:input_1，输入1，为uint8
    i1_s:input_1_scale，输入1对应的缩放因子，为float32
    i1_z:input_1_zero，输入1对应的零点，为uint8
        
    i2:input_2，输入2，为uint8
    i2_s:input_2_scale，输入2对应的缩放因子，为float32
    i2_z:input_2_zero，输入2对应的零点，为uint8
    
    i3_s:input_3_scale，输出3对应的缩放因子，为float32
    i3_z:input_3_zero，输出3对应的零点，为uint8
            

    输出为uint8    
    '''    

    
    r_1 = (i1-i1_z)*i1_s/i3_s
#    r_1 = tf.rint(r_1)
    
    
    r_2 = (i2-i2_z)*i2_s/i3_s 
#    r_2 = tf.rint(r_2)
    
    temp = r_1 + r_2
    temp = tf.rint(temp)
    
    temp = temp + i3_z
    
    temp = tf.clip_by_value(temp, 0, 255)
#    temp = tf.rint(temp)
    
    return temp
    
  

def get_weight(json_file): 
    '''
    解析json_file文件，获取相关参数
    
    返回值：
    w_array:卷积weight数组，为uint8
    b_array:卷积bias数组，为int32
    stride:卷积步长数组
    w_z：卷积weight数组所对应的零点数组，为uint8
    r_z:每一层输出所对应的零点值数组，为uint8
    r_s：每一层输出所对应的scale，为float32
    m:每一层的缩小因子，为float32
    add_s：残差加法所对应的scale，为float32
    add_z:残差加法所对应的zero，为uint8
    i_z:输入所对应的zero，为uint8
    logit_s:logit输出所对应的scale，为float32
    logit_z:logit输出所对应的zero，为uint8
    
    
    '''

    with open (json_file, 'r', encoding='utf-8') as f:
        ops = json.loads(json.load(f))
    
    
    w_index = \
    ['MobilenetV2/Conv/weights_quant/FakeQuantWithMinMaxVars',
     
     'MobilenetV2/expanded_conv/depthwise/weights_quant/FakeQuantWithMinMaxVars',
     'MobilenetV2/expanded_conv/project/weights_quant/FakeQuantWithMinMaxVars',
     
     'MobilenetV2/expanded_conv_1/expand/weights_quant/FakeQuantWithMinMaxVars',
     'MobilenetV2/expanded_conv_1/depthwise/weights_quant/FakeQuantWithMinMaxVars',
     'MobilenetV2/expanded_conv_1/project/weights_quant/FakeQuantWithMinMaxVars',
     
     'MobilenetV2/expanded_conv_2/expand/weights_quant/FakeQuantWithMinMaxVars',
     'MobilenetV2/expanded_conv_2/depthwise/weights_quant/FakeQuantWithMinMaxVars',
     'MobilenetV2/expanded_conv_2/project/weights_quant/FakeQuantWithMinMaxVars',
     
     'MobilenetV2/expanded_conv_3/expand/weights_quant/FakeQuantWithMinMaxVars',
     'MobilenetV2/expanded_conv_3/depthwise/weights_quant/FakeQuantWithMinMaxVars',
     'MobilenetV2/expanded_conv_3/project/weights_quant/FakeQuantWithMinMaxVars',
     
     'MobilenetV2/expanded_conv_4/expand/weights_quant/FakeQuantWithMinMaxVars',
     'MobilenetV2/expanded_conv_4/depthwise/weights_quant/FakeQuantWithMinMaxVars',
     'MobilenetV2/expanded_conv_4/project/weights_quant/FakeQuantWithMinMaxVars',
     
     'MobilenetV2/expanded_conv_5/expand/weights_quant/FakeQuantWithMinMaxVars',
     'MobilenetV2/expanded_conv_5/depthwise/weights_quant/FakeQuantWithMinMaxVars',
     'MobilenetV2/expanded_conv_5/project/weights_quant/FakeQuantWithMinMaxVars',
     
     'MobilenetV2/expanded_conv_6/expand/weights_quant/FakeQuantWithMinMaxVars',
     'MobilenetV2/expanded_conv_6/depthwise/weights_quant/FakeQuantWithMinMaxVars',
     'MobilenetV2/expanded_conv_6/project/weights_quant/FakeQuantWithMinMaxVars',
     
     'MobilenetV2/expanded_conv_7/expand/weights_quant/FakeQuantWithMinMaxVars',
     'MobilenetV2/expanded_conv_7/depthwise/weights_quant/FakeQuantWithMinMaxVars',
     'MobilenetV2/expanded_conv_7/project/weights_quant/FakeQuantWithMinMaxVars',
     
     'MobilenetV2/expanded_conv_8/expand/weights_quant/FakeQuantWithMinMaxVars',
     'MobilenetV2/expanded_conv_8/depthwise/weights_quant/FakeQuantWithMinMaxVars',
     'MobilenetV2/expanded_conv_8/project/weights_quant/FakeQuantWithMinMaxVars',
     
     'MobilenetV2/expanded_conv_9/expand/weights_quant/FakeQuantWithMinMaxVars',
     'MobilenetV2/expanded_conv_9/depthwise/weights_quant/FakeQuantWithMinMaxVars',
     'MobilenetV2/expanded_conv_9/project/weights_quant/FakeQuantWithMinMaxVars',
     
     'MobilenetV2/expanded_conv_10/expand/weights_quant/FakeQuantWithMinMaxVars',
     'MobilenetV2/expanded_conv_10/depthwise/weights_quant/FakeQuantWithMinMaxVars',
     'MobilenetV2/expanded_conv_10/project/weights_quant/FakeQuantWithMinMaxVars',
     
     'MobilenetV2/expanded_conv_11/expand/weights_quant/FakeQuantWithMinMaxVars',
     'MobilenetV2/expanded_conv_11/depthwise/weights_quant/FakeQuantWithMinMaxVars',
     'MobilenetV2/expanded_conv_11/project/weights_quant/FakeQuantWithMinMaxVars',
     
     'MobilenetV2/expanded_conv_12/expand/weights_quant/FakeQuantWithMinMaxVars',
     'MobilenetV2/expanded_conv_12/depthwise/weights_quant/FakeQuantWithMinMaxVars',
     'MobilenetV2/expanded_conv_12/project/weights_quant/FakeQuantWithMinMaxVars',
     
     'MobilenetV2/expanded_conv_13/expand/weights_quant/FakeQuantWithMinMaxVars',
     'MobilenetV2/expanded_conv_13/depthwise/weights_quant/FakeQuantWithMinMaxVars',
     'MobilenetV2/expanded_conv_13/project/weights_quant/FakeQuantWithMinMaxVars',
     
     'MobilenetV2/expanded_conv_14/expand/weights_quant/FakeQuantWithMinMaxVars',
     'MobilenetV2/expanded_conv_14/depthwise/weights_quant/FakeQuantWithMinMaxVars',
     'MobilenetV2/expanded_conv_14/project/weights_quant/FakeQuantWithMinMaxVars',
     
     'MobilenetV2/expanded_conv_15/expand/weights_quant/FakeQuantWithMinMaxVars',
     'MobilenetV2/expanded_conv_15/depthwise/weights_quant/FakeQuantWithMinMaxVars',
     'MobilenetV2/expanded_conv_15/project/weights_quant/FakeQuantWithMinMaxVars',
     
     'MobilenetV2/expanded_conv_16/expand/weights_quant/FakeQuantWithMinMaxVars',
     'MobilenetV2/expanded_conv_16/depthwise/weights_quant/FakeQuantWithMinMaxVars',
     'MobilenetV2/expanded_conv_16/project/weights_quant/FakeQuantWithMinMaxVars',
     
     'MobilenetV2/Conv_1/weights_quant/FakeQuantWithMinMaxVars',
     
     'MobilenetV2/Logits/Conv2d_1c_1x1/weights_quant/FakeQuantWithMinMaxVars',
     ]
    
    
    b_index = \
    ['MobilenetV2/Conv/Conv2D_Fold_bias',
     
     'MobilenetV2/expanded_conv/depthwise/depthwise_Fold_bias',
     'MobilenetV2/expanded_conv/project/Conv2D_Fold_bias',
     
     'MobilenetV2/expanded_conv_1/expand/Conv2D_Fold_bias',
     'MobilenetV2/expanded_conv_1/depthwise/depthwise_Fold_bias',
     'MobilenetV2/expanded_conv_1/project/Conv2D_Fold_bias',
     
     'MobilenetV2/expanded_conv_2/expand/Conv2D_Fold_bias',
     'MobilenetV2/expanded_conv_2/depthwise/depthwise_Fold_bias',
     'MobilenetV2/expanded_conv_2/project/Conv2D_Fold_bias',
     
     'MobilenetV2/expanded_conv_3/expand/Conv2D_Fold_bias',
     'MobilenetV2/expanded_conv_3/depthwise/depthwise_Fold_bias',
     'MobilenetV2/expanded_conv_3/project/Conv2D_Fold_bias',
     
     'MobilenetV2/expanded_conv_4/expand/Conv2D_Fold_bias',
     'MobilenetV2/expanded_conv_4/depthwise/depthwise_Fold_bias',
     'MobilenetV2/expanded_conv_4/project/Conv2D_Fold_bias',
     
     'MobilenetV2/expanded_conv_5/expand/Conv2D_Fold_bias',
     'MobilenetV2/expanded_conv_5/depthwise/depthwise_Fold_bias',
     'MobilenetV2/expanded_conv_5/project/Conv2D_Fold_bias',
     
     'MobilenetV2/expanded_conv_6/expand/Conv2D_Fold_bias',
     'MobilenetV2/expanded_conv_6/depthwise/depthwise_Fold_bias',
     'MobilenetV2/expanded_conv_6/project/Conv2D_Fold_bias',
     
     'MobilenetV2/expanded_conv_7/expand/Conv2D_Fold_bias',
     'MobilenetV2/expanded_conv_7/depthwise/depthwise_Fold_bias',
     'MobilenetV2/expanded_conv_7/project/Conv2D_Fold_bias',
     
     'MobilenetV2/expanded_conv_8/expand/Conv2D_Fold_bias',
     'MobilenetV2/expanded_conv_8/depthwise/depthwise_Fold_bias',
     'MobilenetV2/expanded_conv_8/project/Conv2D_Fold_bias',
     
     'MobilenetV2/expanded_conv_9/expand/Conv2D_Fold_bias',
     'MobilenetV2/expanded_conv_9/depthwise/depthwise_Fold_bias',
     'MobilenetV2/expanded_conv_9/project/Conv2D_Fold_bias',
     
     'MobilenetV2/expanded_conv_10/expand/Conv2D_Fold_bias',
     'MobilenetV2/expanded_conv_10/depthwise/depthwise_Fold_bias',
     'MobilenetV2/expanded_conv_10/project/Conv2D_Fold_bias',
     
     'MobilenetV2/expanded_conv_11/expand/Conv2D_Fold_bias',
     'MobilenetV2/expanded_conv_11/depthwise/depthwise_Fold_bias',
     'MobilenetV2/expanded_conv_11/project/Conv2D_Fold_bias',
     
     'MobilenetV2/expanded_conv_12/expand/Conv2D_Fold_bias',
     'MobilenetV2/expanded_conv_12/depthwise/depthwise_Fold_bias',
     'MobilenetV2/expanded_conv_12/project/Conv2D_Fold_bias',
     
     'MobilenetV2/expanded_conv_13/expand/Conv2D_Fold_bias',
     'MobilenetV2/expanded_conv_13/depthwise/depthwise_Fold_bias',
     'MobilenetV2/expanded_conv_13/project/Conv2D_Fold_bias',
     
     'MobilenetV2/expanded_conv_14/expand/Conv2D_Fold_bias',
     'MobilenetV2/expanded_conv_14/depthwise/depthwise_Fold_bias',
     'MobilenetV2/expanded_conv_14/project/Conv2D_Fold_bias',
     
     'MobilenetV2/expanded_conv_15/expand/Conv2D_Fold_bias',
     'MobilenetV2/expanded_conv_15/depthwise/depthwise_Fold_bias',
     'MobilenetV2/expanded_conv_15/project/Conv2D_Fold_bias',
     
     'MobilenetV2/expanded_conv_16/expand/Conv2D_Fold_bias',
     'MobilenetV2/expanded_conv_16/depthwise/depthwise_Fold_bias',
     'MobilenetV2/expanded_conv_16/project/Conv2D_Fold_bias',
      
     'MobilenetV2/Conv_1/Conv2D_Fold_bias',
     
     'MobilenetV2/Logits/Conv2d_1c_1x1/Conv2D_bias',
     ]
    
    
    r_index = \
    ['MobilenetV2/Conv/Relu6',
     
     'MobilenetV2/expanded_conv/depthwise/Relu6',
     'MobilenetV2/expanded_conv/project/add_fold',
     
     'MobilenetV2/expanded_conv_1/expand/Relu6',
     'MobilenetV2/expanded_conv_1/depthwise/Relu6',
     'MobilenetV2/expanded_conv_1/project/add_fold',
     
     'MobilenetV2/expanded_conv_2/expand/Relu6',
     'MobilenetV2/expanded_conv_2/depthwise/Relu6',
     'MobilenetV2/expanded_conv_2/project/add_fold',
     
     'MobilenetV2/expanded_conv_3/expand/Relu6',
     'MobilenetV2/expanded_conv_3/depthwise/Relu6',
     'MobilenetV2/expanded_conv_3/project/add_fold',
     
     'MobilenetV2/expanded_conv_4/expand/Relu6',
     'MobilenetV2/expanded_conv_4/depthwise/Relu6',
     'MobilenetV2/expanded_conv_4/project/add_fold',
     
     'MobilenetV2/expanded_conv_5/expand/Relu6',
     'MobilenetV2/expanded_conv_5/depthwise/Relu6',
     'MobilenetV2/expanded_conv_5/project/add_fold',
     
     'MobilenetV2/expanded_conv_6/expand/Relu6',
     'MobilenetV2/expanded_conv_6/depthwise/Relu6',
     'MobilenetV2/expanded_conv_6/project/add_fold',
     
     'MobilenetV2/expanded_conv_7/expand/Relu6',
     'MobilenetV2/expanded_conv_7/depthwise/Relu6',
     'MobilenetV2/expanded_conv_7/project/add_fold',
     
     'MobilenetV2/expanded_conv_8/expand/Relu6',
     'MobilenetV2/expanded_conv_8/depthwise/Relu6',
     'MobilenetV2/expanded_conv_8/project/add_fold',
     
     'MobilenetV2/expanded_conv_9/expand/Relu6',
     'MobilenetV2/expanded_conv_9/depthwise/Relu6',
     'MobilenetV2/expanded_conv_9/project/add_fold',
     
     'MobilenetV2/expanded_conv_10/expand/Relu6',
     'MobilenetV2/expanded_conv_10/depthwise/Relu6',
     'MobilenetV2/expanded_conv_10/project/add_fold',
     
     'MobilenetV2/expanded_conv_11/expand/Relu6',
     'MobilenetV2/expanded_conv_11/depthwise/Relu6',
     'MobilenetV2/expanded_conv_11/project/add_fold',
     
     'MobilenetV2/expanded_conv_12/expand/Relu6',
     'MobilenetV2/expanded_conv_12/depthwise/Relu6',
     'MobilenetV2/expanded_conv_12/project/add_fold',
     
     'MobilenetV2/expanded_conv_13/expand/Relu6',
     'MobilenetV2/expanded_conv_13/depthwise/Relu6',
     'MobilenetV2/expanded_conv_13/project/add_fold',
     
     'MobilenetV2/expanded_conv_14/expand/Relu6',
     'MobilenetV2/expanded_conv_14/depthwise/Relu6',
     'MobilenetV2/expanded_conv_14/project/add_fold',
     
     'MobilenetV2/expanded_conv_15/expand/Relu6',
     'MobilenetV2/expanded_conv_15/depthwise/Relu6',
     'MobilenetV2/expanded_conv_15/project/add_fold',
     
     'MobilenetV2/expanded_conv_16/expand/Relu6',
     'MobilenetV2/expanded_conv_16/depthwise/Relu6',
     'MobilenetV2/expanded_conv_16/project/add_fold',
     
     'MobilenetV2/Conv_1/Relu6',
     
     'MobilenetV2/Logits/Conv2d_1c_1x1/BiasAdd',
      
     ]
    
    quant_add_index = \
    ['MobilenetV2/expanded_conv_2/add',
     'MobilenetV2/expanded_conv_4/add',
     'MobilenetV2/expanded_conv_5/add',
     'MobilenetV2/expanded_conv_7/add',
     'MobilenetV2/expanded_conv_8/add',
     'MobilenetV2/expanded_conv_9/add',
     'MobilenetV2/expanded_conv_11/add',
     'MobilenetV2/expanded_conv_12/add',
     'MobilenetV2/expanded_conv_14/add',
     'MobilenetV2/expanded_conv_15/add',     
     ]
    
    stride = [2,
              1,1,
              1,2,1,
              1,1,1,
              1,2,1,
              1,1,1,
              1,1,1,
              1,2,1,
              1,1,1,
              1,1,1,
              1,1,1,
              1,1,1,
              1,1,1,
              1,1,1,
              1,2,1,
              1,1,1,
              1,1,1,
              1,1,1,
              1,
              1,]
    
        
    w_array = []
    b_array = []
    w_z = []
    r_z = []
    b_s = []
    r_s = []
    add_s = []
    add_z = []
    
    
    for i in range(len(quant_add_index)):
        add_s.append(ops[quant_add_index[i]]['quantization'][0])
        add_z.append(ops[quant_add_index[i]]['quantization'][1])
       
    
    
    for i in range(len(w_index)):
        
        temp = ops[w_index[i]]['numpy_array']
        w_array.append(np.array(temp, dtype=np.float32).transpose([1,2,3,0]))
            
        temp = ops[w_index[i]]['quantization'][1]
        w_z.append(temp)
    
        temp = ops[r_index[i]]['quantization'][1]
        r_z.append(temp)
    
    
        
        temp = ops[b_index[i]]['numpy_array']
        b_array.append(np.array(temp, dtype=np.float32))
        
        b__s = ops[b_index[i]]['quantization'][0]
        b_s.append(b__s)
        r__s = ops[r_index[i]]['quantization'][0]
        r_s.append(r__s) 
    
    
    b_s = np.array(b_s, np.float32)    
    r_s = np.array(r_s, np.float32)    
    m =   np.array(b_s/r_s, np.float32)
    
    i_z = ops['input']['quantization'][1]
    
    logit_s = ops['MobilenetV2/Logits/Conv2d_1c_1x1/BiasAdd']['quantization'][0]
    logit_z = ops['MobilenetV2/Logits/Conv2d_1c_1x1/BiasAdd']['quantization'][1]

    
    return  w_array, b_array, stride, w_z, r_z, r_s, m, add_s, add_z, i_z, logit_s, logit_z 



def quant_mobilenetV2_base(inputs,
                          w_array, b_array, stride,
                          w_z, r_z,
                          r_s, m,
                          add_s, add_z,
                          i_z):

    '''
    inputs: (None, 224, 224, 3)，为uint8
    w_array: weight数组，为uint8
    b_array：bias数组，为uint8
    stride：步长数组
    w_z：weight_zero数组，为uint8
    r_z: 每一层输出层的zero数组，为uint8
    r_s：每一层输出层对应的scale数组，为float
    m: 每一层最后的缩小因子数组，为float
    add_s：残差连接对应的缩放因子数组，为float
    add_z：残差连接对应的零点数组，为uint8
    i_z：输入对应的零点，为uint8
        
    '''    
    
    

    #Conv
    layer1 = quant_conv(inputs, w_array[0], b_array[0], 'SAME',
                      stride[0],
                      i_z, w_z[0], r_z[0],
                      m[0])
    
    #expanded_conv
    layer2_ = quant_depthwise_conv(layer1, w_array[1], b_array[1], 'SAME',
                      stride[1],
                      r_z[0], w_z[1], r_z[1],
                      m[1])
    
    layer2 = quant_conv(layer2_, w_array[2], b_array[2], 'SAME',
                      stride[2],
                      r_z[1], w_z[2], r_z[2],
                      m[2])
    
    
    #expanded_conv_1
    layer3_ = quant_conv(layer2, w_array[3], b_array[3], 'SAME',
                      stride[3],
                      r_z[2], w_z[3], r_z[3],
                      m[3])
    
    layer3__ = quant_depthwise_conv(layer3_, w_array[4], b_array[4], 'SAME',
                      stride[4],
                      r_z[3], w_z[4], r_z[4],
                      m[4])
    
    layer3 = quant_conv(layer3__, w_array[5], b_array[5], 'SAME',
                      stride[5],
                      r_z[4], w_z[5], r_z[5],
                      m[5])
    
    
    #expanded_conv_2
    layer4_ = quant_conv(layer3, w_array[6], b_array[6], 'SAME',
                      stride[6],
                      r_z[5], w_z[6], r_z[6],
                      m[6])
    
    layer4__ = quant_depthwise_conv(layer4_, w_array[7], b_array[7], 'SAME',
                      stride[7],
                      r_z[6], w_z[7], r_z[7],
                      m[7])
    
    layer4___ = quant_conv(layer4__, w_array[8], b_array[8], 'SAME',
                      stride[8],
                      r_z[7], w_z[8], r_z[8],
                      m[8])
    
    layer4 = quant_add(layer3, r_s[5], r_z[5],
                      layer4___, r_s[8], r_z[8],
                      add_s[0], add_z[0])
    
        
    #expanded_conv_3
    layer5_ = quant_conv(layer4, w_array[9], b_array[9], 'SAME',
                      stride[9],
                      add_z[0], w_z[9], r_z[9],
                      m[9])
    
    layer5__ = quant_depthwise_conv(layer5_, w_array[10], b_array[10], 'SAME',
                      stride[10],
                      r_z[9], w_z[10], r_z[10],
                      m[10])
    
    layer5 = quant_conv(layer5__, w_array[11], b_array[11], 'SAME',
                      stride[11],
                      r_z[10], w_z[11], r_z[11],
                      m[11])
    
    
    #expanded_conv_4
    layer6_ = quant_conv(layer5, w_array[12], b_array[12], 'SAME',
                      stride[12],
                      r_z[11], w_z[12], r_z[12],
                      m[12])
    
    layer6__ = quant_depthwise_conv(layer6_, w_array[13], b_array[13], 'SAME',
                      stride[13],
                      r_z[12], w_z[13], r_z[13],
                      m[13])
    
    layer6___ = quant_conv(layer6__, w_array[14], b_array[14], 'SAME',
                      stride[14],
                      r_z[13], w_z[14], r_z[14],
                      m[14])
    
    layer6 = quant_add(layer5, r_s[11], r_z[11],
                      layer6___, r_s[14], r_z[14],
                      add_s[1], add_z[1])
    
    
    #add_test = np.load('layer_6.npy').astype(np.float32) 
    
    #expanded_conv_5
    layer7_ = quant_conv(layer6, w_array[15], b_array[15], 'SAME',
                      stride[15],
                      add_z[1], w_z[15], r_z[15],
                      m[15])
    
    layer7__ = quant_depthwise_conv(layer7_, w_array[16], b_array[16], 'SAME',
                      stride[16],
                      r_z[15], w_z[16], r_z[16],
                      m[16])
    
    layer7___ = quant_conv(layer7__, w_array[17], b_array[17], 'SAME',
                      stride[17],
                      r_z[16], w_z[17], r_z[17],
                      m[17])
    
    layer7 = quant_add(layer6, add_s[1], add_z[1],
                      layer7___, r_s[17], r_z[17],
                      add_s[2], add_z[2])
    
    
    
    #add_test = np.load('layer_7.npy').astype(np.float32) 
    #expanded_conv_6
    layer8_ = quant_conv(layer7, w_array[18], b_array[18], 'SAME',
                      stride[18],
                      add_z[2], w_z[18], r_z[18],
                      m[18])
    
    layer8__ = quant_depthwise_conv(layer8_, w_array[19], b_array[19], 'SAME',
                      stride[19],
                      r_z[18], w_z[19], r_z[19],
                      m[19])
    
    layer8 = quant_conv(layer8__, w_array[20], b_array[20], 'SAME',
                      stride[20],
                      r_z[19], w_z[20], r_z[20],
                      m[20])
    
    
    #add_test = np.load('layer_7.npy').astype(np.float32) 
    #expanded_conv_7
    layer9_ = quant_conv(layer8, w_array[21], b_array[21], 'SAME',
                      stride[21],
                      r_z[20], w_z[21], r_z[21],
                      m[21])
    
    layer9__ = quant_depthwise_conv(layer9_, w_array[22], b_array[22], 'SAME',
                      stride[22],
                      r_z[21], w_z[22], r_z[22],
                      m[22])
    
    layer9___ = quant_conv(layer9__, w_array[23], b_array[23], 'SAME',
                      stride[23],
                      r_z[22], w_z[23], r_z[23],
                      m[23])
    
    layer9 = quant_add(layer8, r_s[20], r_z[20],
                      layer9___, r_s[23], r_z[23],
                      add_s[3], add_z[3])
    
    
    
    #expanded_conv_8
    layer10_ = quant_conv(layer9, w_array[24], b_array[24], 'SAME',
                      stride[24],
                      add_z[3], w_z[24], r_z[24],
                      m[24])
    
    layer10__ = quant_depthwise_conv(layer10_, w_array[25], b_array[25], 'SAME',
                      stride[25],
                      r_z[24], w_z[25], r_z[25],
                      m[25])
    
    layer10___ = quant_conv(layer10__, w_array[26], b_array[26], 'SAME',
                      stride[26],
                      r_z[25], w_z[26], r_z[26],
                      m[26])
    
    layer10 = quant_add(layer9, add_s[3], add_z[3],
                      layer10___, r_s[26], r_z[26],
                      add_s[4], add_z[4])
    
    
    
    #add_test = np.load('layer_10.npy').astype(np.float32) 
    
    #expanded_conv_9
    layer11_ = quant_conv(layer10, w_array[27], b_array[27], 'SAME',
                      stride[27],
                      add_z[4], w_z[27], r_z[27],
                      m[27])
    
    layer11__ = quant_depthwise_conv(layer11_, w_array[28], b_array[28], 'SAME',
                      stride[28],
                      r_z[27], w_z[28], r_z[28],
                      m[28])
    
    layer11___ = quant_conv(layer11__, w_array[29], b_array[29], 'SAME',
                      stride[29],
                      r_z[28], w_z[29], r_z[29],
                      m[29])
    
    layer11 = quant_add(layer10, add_s[4], add_z[4],
                      layer11___, r_s[29], r_z[29],
                      add_s[5], add_z[5])
    
    
    
    #expanded_conv_10
    layer12_ = quant_conv(layer11, w_array[30], b_array[30], 'SAME',
                      stride[30],
                      add_z[5], w_z[30], r_z[30],
                      m[30])
    
    layer12__ = quant_depthwise_conv(layer12_, w_array[31], b_array[31], 'SAME',
                      stride[31],
                      r_z[30], w_z[31], r_z[31],
                      m[31])
    
    layer12 = quant_conv(layer12__, w_array[32], b_array[32], 'SAME',
                      stride[32],
                      r_z[31], w_z[32], r_z[32],
                      m[32])
    
    
    #add_test = np.load('layer_12.npy').astype(np.float32) 
    
    #expanded_conv_11
    layer13_ = quant_conv(layer12, w_array[33], b_array[33], 'SAME',
                      stride[33],
                      r_z[32], w_z[33], r_z[33],
                      m[33])
    
    layer13__ = quant_depthwise_conv(layer13_, w_array[34], b_array[34], 'SAME',
                      stride[34],
                      r_z[33], w_z[34], r_z[34],
                      m[34])
    
    layer13___ = quant_conv(layer13__, w_array[35], b_array[35], 'SAME',
                      stride[35],
                      r_z[34], w_z[35], r_z[35],
                      m[35])
    
    layer13 = quant_add(layer12, r_s[32], r_z[32],
                      layer13___, r_s[35], r_z[35],
                      add_s[6], add_z[6])
    
    
    #expanded_conv_12
    layer14_ = quant_conv(layer13, w_array[36], b_array[36], 'SAME',
                      stride[36],
                      add_z[6], w_z[36], r_z[36],
                      m[36])
    
    layer14__ = quant_depthwise_conv(layer14_, w_array[37], b_array[37], 'SAME',
                      stride[37],
                      r_z[36], w_z[37], r_z[37],
                      m[37])
    
    layer14___ = quant_conv(layer14__, w_array[38], b_array[38], 'SAME',
                      stride[38],
                      r_z[37], w_z[38], r_z[38],
                      m[38])
    
    layer14 = quant_add(layer13, add_s[6], add_z[6],
                      layer14___, r_s[38], r_z[38],
                      add_s[7], add_z[7])
    
    
    
    #expanded_conv_13
    layer15_ = quant_conv(layer14, w_array[39], b_array[39], 'SAME',
                      stride[39],
                      add_z[7], w_z[39], r_z[39],
                      m[39])
    
    layer15__ = quant_depthwise_conv(layer15_, w_array[40], b_array[40], 'SAME',
                      stride[40],
                      r_z[39], w_z[40], r_z[40],
                      m[40])
    
    layer15 = quant_conv(layer15__, w_array[41], b_array[41], 'SAME',
                      stride[41],
                      r_z[40], w_z[41], r_z[41],
                      m[41])
    
    
    #add_test = np.load('layer_15.npy').astype(np.float32) 
    
    #expanded_conv_14
    layer16_ = quant_conv(layer15, w_array[42], b_array[42], 'SAME',
                      stride[42],
                      r_z[41], w_z[42], r_z[42],
                      m[42])
    
    layer16__ = quant_depthwise_conv(layer16_, w_array[43], b_array[43], 'SAME',
                      stride[43],
                      r_z[42], w_z[43], r_z[43],
                      m[43])
    
    layer16___ = quant_conv(layer16__, w_array[44], b_array[44], 'SAME',
                      stride[44],
                      r_z[43], w_z[44], r_z[44],
                      m[44])
    
    layer16 = quant_add(layer15, r_s[41], r_z[41],
                      layer16___, r_s[44], r_z[44],
                      add_s[8], add_z[8])
    
    
    #expanded_conv_15
    layer17_ = quant_conv(layer16, w_array[45], b_array[45], 'SAME',
                      stride[45],
                      add_z[8], w_z[45], r_z[45],
                      m[45])
    
    layer17__ = quant_depthwise_conv(layer17_, w_array[46], b_array[46], 'SAME',
                      stride[46],
                      r_z[45], w_z[46], r_z[46],
                      m[46])
    
    layer17___ = quant_conv(layer17__, w_array[47], b_array[47], 'SAME',
                      stride[47],
                      r_z[46], w_z[47], r_z[47],
                      m[47])
    
    layer17 = quant_add(layer16, add_s[8], add_z[8],
                      layer17___, r_s[47], r_z[47],
                      add_s[9], add_z[9])
    
    
    
    #expanded_conv_16
    
    #add_test = np.load('layer_17.npy').astype(np.float32) 
    
    layer18_ = quant_conv(layer17, w_array[48], b_array[48], 'SAME',
                      stride[48],
                      add_z[9], w_z[48], r_z[48],
                      m[48])
    
    layer18__ = quant_depthwise_conv(layer18_, w_array[49], b_array[49], 'SAME',
                      stride[49],
                      r_z[48], w_z[49], r_z[49],
                      m[49])
    
    layer18 = quant_conv(layer18__, w_array[50], b_array[50], 'SAME',
                      stride[50],
                      r_z[49], w_z[50], r_z[50],
                      m[50])
    
    
    #Conv_1
    #add_test = np.load('layer_18.npy').astype(np.float32) 
    
    layer19 = quant_conv(layer18, w_array[51], b_array[51], 'SAME',
                      stride[51],
                      r_z[50], w_z[51], r_z[51],
                      m[51])
    
    
    # avg pool
    pool = tf.nn.avg_pool(layer19, [1,7,7,1], [1,1,1,1], 'VALID')
    pool = tf.rint(pool)
    
    
    # logit
    logit = quant_conv(pool, w_array[52], b_array[52], 'SAME',
                      stride[52],
                      r_z[51], w_z[52], r_z[52],
                      m[52])
    
    logit = tf.squeeze(logit, [1, 2])
    
    return logit




def quant_mobilenetV2(inputs, json_file):
    '''
    inputs: (None,224,224,3)
    json_file: 将tflite里的权重序列化成相应的json文件
    
    用法：
    inputs = tf.placeholder(tf.float32, [None, 224, 224, 3], name='input')
    json_file = 'mobilenetv2_flower_quant_8_26.json'
    logit = quant_mobilenetV2(inputs, json_file)
    with tf.Session() as sess:
        r = sess.run(logit, feed_dict={inputs:img})

    
    '''

    w_array, b_array, stride, w_z, r_z, r_s, m, add_s, add_z, i_z, \
    logit_s, logit_z = get_weight(json_file)


    logit = quant_mobilenetV2_base(inputs,
                                  w_array, b_array, stride,
                                  w_z, r_z,
                                  r_s, m,
                                  add_s, add_z,
                                  i_z)

    
#   将量化值映射回浮点值 ，根据公式 r = s*(Q - Z)
    logit_r = logit_s*(logit - logit_z)
    
    prediction = tf.nn.softmax(logit_r)
    
    return logit, prediction




