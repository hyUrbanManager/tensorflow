
# coding: utf-8

# # tensorflow MNIST 全连接网络案例


###1.导入必要的库
###2.定义全连接网络函数
###3.初始化网络参数
###4.训练模型



#step1:required libararies
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('./MNIST_data',one_hot=True)



## step2:establish Fully Connected layer function
### inputs:输入,W(Weight):权值
### b(bias):偏置, acti(activation function):激活函数
def fc_layer(inputs,W,b,acti):
    if acti==None:
        return tf.matmul(inputs,W)+b
    else:
        return acti(tf.matmul(inputs,W)+b)




#step3:initialize parameters
#####数值变量#####
batch_size=128 ###训练块的大小
input_shape=28*28 ###特征数目(像素数目)
output_shape=10 ###one_hot(独热)编码
h1_nodes=500 ###隐含层的神经元个数
####一系列参数####
###输入和输出######
x=tf.placeholder(tf.float32,
                 [batch_size,
                  input_shape],
                name='x-inputs')
y_=tf.placeholder(tf.float32,
                 [batch_size,
                  output_shape],
                name='y-inputs')

#######权值和偏置###############

###隐含层的权值和偏置####
W1=tf.Variable(tf.random_normal([input_shape,h1_nodes],
                                stddev=0.1),name='W1')
b1=tf.Variable(tf.zeros([h1_nodes]),name='b1')

###输出层的权值和偏置####
W2=tf.Variable(tf.random_normal([h1_nodes,output_shape],
                                stddev=0.1),name='W2')
b2=tf.Variable(tf.zeros([output_shape]),name='b2')

###前向传播(Forward Propagation)输出
output1=fc_layer(x,W1,b1,tf.nn.relu) ##隐含层输出
outputs=fc_layer(output1,W2,b2,None) ##输出层输出

###定义损耗函数(交叉熵函数)####
cross_entropy=tf.nn.softmax_cross_entropy_with_logits_v2(logits=outputs,
                                                         labels=y_)
loss=tf.reduce_mean(cross_entropy)

###选择优化器(optimizer)和learing rate=0.01,SGD
train_step=tf.train.GradientDescentOptimizer(0.01).minimize(loss)

###预测正确的样本个数和准确率
correct_predictions=tf.equal(tf.argmax(outputs,1),tf.argmax(y_,1))

accuracy=tf.reduce_mean(tf.cast(correct_predictions,tf.float32))


###step 4: train model
import time ##计时器
####ops###
epochs=50 ###训练次数
num_batchs=int(mnist.train.num_examples/batch_size)
start_time=time.time()
with tf.Session() as sess:
    ###初始化变量###
    sess.run(tf.global_variables_initializer())
    for i in range(epochs):
        for _ in range(num_batchs):
            ###对于训练样本##
            ##取训练块操作
            xs,ys=mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:xs,y_:ys})
            cost,acc=sess.run([loss,accuracy],
                              feed_dict={x:xs,y_:ys})
            ##取测试块操作
            xt,yt=mnist.test.next_batch(batch_size)
            cost_t,acc_t=sess.run([loss,accuracy],
                              feed_dict={x:xt,y_:yt})
        if((i+1)%10==0):
            print('训练集上,经过%d次训练,cost=%.6f,acc=%.6f'%(i+1,cost,acc))
            print('测试集上,经过%d次训练,cost=%.6f,acc=%.6f'%(i+1,cost_t,acc_t))
            
    end_time=time.time()
    print('优化完成!程序运行%.2f秒'%(end_time-start_time))

