#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import caffe
import cv2

caffe_root = "/home/minzhao/caffe-master/"
net_file = "deploy.prototxt"
model = "bvlc_googlenet.caffemodel"

caffe.set_mode_gpu()
caffe.set_device(0)

net = caffe.Net(net_file, model, caffe.TEST)
image_mean = np.load(caffe_root + "python/caffe/imagenet/ilsvrc_2012_mean.npy").mean(1).mean(1)


transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1)) # height*width*channel to c*h*w
transformer.set_mean('data', image_mean)
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2, 1, 0))# RGB to BGR

def pre(transformed_image,name=None):
    net.blobs['data'].data[...] = transformed_image

    output = net.forward()
    output_prob = output['prob'][0] # 第一张图片的类别概率
    top_k = output_prob.argsort()[::-1]
    print 'New print ###########'
    for pred in top_k[:1]:    
        print 'The predicted class is : ', pred # argmax()函数是求取矩阵中最大元素的索引
        print 'confidence is : ', output_prob[pred]
        label_file = caffe_root + "data/ilsvrc12/synset_words.txt"
        labels = np.loadtxt(label_file, str, delimiter='\t')
        print 'The label is : ', labels[pred]

    if name == 'no':
        return

    if name != None:
        f = plt.figure()
        plt.imshow(transformer.deprocess('data',transformed_image))
        plt.title(name)
        f.show()
    else:
        f = plt.figure()
        plt.imshow(transformer.deprocess('data',transformed_image))
        plt.title('Origin')
        f.show()


def predict(imURL,name=None):
    image = caffe.io.load_image(imURL)
    transformed_image = transformer.preprocess('data', image)
    pre(transformed_image)
    if name == 'make disclass':
        probs = np.zeros_like(net.blobs['prob'].data)
        # 111 is the class to get grad
        probs[0][111] = 1
        gradient = net.backward(prob=probs)['data'][0]

        f = plt.figure()

        # see the noise in pic
        plt.imshow(transformer.deprocess('data', gradient / np.percentile(gradient, 98))*255)
        plt.title('Noise')
        f.show()

        pre(transformed_image+0.5*np.sign(gradient),'After noise')
    else:
        for x in xrange(1,9):
            probs = np.zeros_like(net.blobs['prob'].data)
            # class 281 is cat
            probs[0][281] = 1
            gradient = net.backward(prob=probs)['data'][0]
            transformed_image += 0.5*np.sign(gradient)
            pre(transformed_image,'no')
        f = plt.figure()
        plt.imshow(transformer.deprocess('data', transformed_image))
        plt.title('Cat')
        f.show()
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    predict(imURL='/home/minzhao/pd.jpg',name='make disclass')
    # predict(imURL='/home/minzhao/dog.jpg',name='change to cat')
    raw_input()