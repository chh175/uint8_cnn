from PIL import Image
import numpy as np
import tensorflow as tf
import json
from quant_mobilenetV2 import quant_mobilenetV2



np.set_printoptions(suppress=True)

# PIL读图方式
def read_image(path):
    img = Image.open(path).resize((224,224))
    img = np.expand_dims(img, 0).astype(np.float32)
    
    return img


label = []
with open('labels.txt','r') as f:
    for line in f:
        label.append(line.strip())

inputs = tf.placeholder(tf.float32, [None, 224, 224, 3], name='input')
json_file = 'mobilenetv2_flower_quant_8_26.json'

logit, prediction = quant_mobilenetV2(inputs, json_file)



while(1):

    img_dir = input('please input image name:')
    img = read_image(img_dir)

    with tf.Session() as sess:
        [my_logit, my_prediction] = sess.run([logit[0], prediction[0]], feed_dict={inputs:img})   


    top5_index = np.argsort(my_prediction)[::-1][0:5]
    
    
    for i in top5_index:
        print(label[i], my_prediction[i])