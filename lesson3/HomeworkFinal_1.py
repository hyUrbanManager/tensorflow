# 数据可视化
import os
import matplotlib.image as img
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.model_selection import train_test_split
from skimage.color import rgb2gray
from skimage.transform import resize

# 获取根目录
# os.chdir('./flowers')
data_path = 'G:\\ai\\Homework2\\flowers-recognition\\flowers'
os.chdir(data_path)
root_path = os.getcwd()

# 定义用于存储训练、测试数据的列表
train_img = [] ; test_img = []
train_cla = [] ; test_cla = []

num_count = 0
num_classes = 5

# 图片数据集扩充
#def ImageExpend (imgx, imgy):
#    train_xl = [] ; train_yl = []
#    
#    train_xl.extend(imgx)
#    train_xl.extend([ndimage.rotate(xx, 90) for xx in imgx])
#    train_xl.extend([ndimage.rotate(xx, 180) for xx in imgx])
#    train_xl.extend([ndimage.rotate(xx, 270) for xx in imgx])
#    
#    train_yl.extend(imgy)
#    train_yl.extend(imgy)
#    train_yl.extend(imgy)
#    train_yl.extend(imgy)
#    return train_xl, train_yl

# 遍历根目录下的子文件夹
for folder in os.listdir(root_path):
    img_path = os.path.join(root_path, folder)
    img_l = [] ; clas_l = []
    # 遍历文件夹里的图片
    for images in os.listdir(img_path):
        img_data = img.imread(os.path.join(img_path, images))
        img_l.append(img_data)
        clas_l.extend([num_count])
    num_count += 1
        
    # 将图片灰度化并压缩成32*32
    img_32 = [resize(rgb2gray(xx),(32,32)) for xx in img_l]
    # 分割出80%的训练数据和20%的测试数据
    xtr, xte, ytr, yte = train_test_split(img_32, clas_l, test_size = 0.2, random_state = 1)
#    trxl, tryl = ImageExpend(xtr, ytr)
    train_img.extend(xtr)
    train_cla.extend(ytr)
    test_img.extend(xte)
    test_cla.extend(yte)
#    if num_count == 1:
#        break

# 打乱数据的顺序
import numpy as np
from sklearn.utils import shuffle
def shuffle_data(data, data_class):
    num_images = len(data)
    data = np.array(data)
    data_class = np.array(data_class)
    indx = np.arange(0,num_images)
    indx = shuffle(indx)
    data_shuffle = data[indx]
    class_shuffle = data_class[indx]
    return data_shuffle, class_shuffle

# 分批获取数据
def getSmall_data(inputs, labels, batch_size):
    i = 0
    while True:
        small_data = inputs[i:(batch_size + i)]
        small_label = labels[i:(batch_size + i)]
        i += batch_size
        yield small_data, small_label

# 将训练集与测试集打乱
x_train, y_train = shuffle_data(train_img, train_cla)
x_test, y_test = shuffle_data(test_img, test_cla)

# 将标签数据转换成独热编码：按五个类别进行分类
import keras
Y_train = keras.utils.to_categorical(np.array(y_train), num_classes)
Y_test = keras.utils.to_categorical(np.array(y_test), num_classes)

# 声明卷积层的变量并实现向前传播
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

# 卷积与池化滤波器
def conv2d(x,w):
    return tf.nn.conv2d(x, w, strides = [1,1,1,1], padding='SAME')
def max_pool(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding='SAME')

# 定义神经网络相关参数
batch_size = 256
batch_num = int(len(train_img) / batch_size)

# 建立输入占位符
x_ = tf.placeholder(tf.float32, [None, 32, 32])
y_ = tf.placeholder(tf.float32, [None, num_classes])

x_image = tf.reshape(x_, [-1, 32, 32, 1])

# 第一层卷积
w_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)

# 第一层池化
h_pool1 = max_pool(h_conv1)

# 第二层卷积
w_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)

# 第二层池化
h_pool2 = max_pool(h_conv2)

# 将特征图进行展开
h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])

# 全连接层
w_fc1 = weight_variable([8 * 8 * 64, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

# 减少过拟合
#keep_prob = tf.placeholder("float")
#h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层
w_fc2 = weight_variable([1024, num_classes])
b_fc2 = bias_variable([num_classes])
y_conv = tf.nn.softmax(tf.matmul(h_fc1, w_fc2) + b_fc2)

# 损失函数
#loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = y_conv,labels = y_))
#train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 使用向前传播计算准确率
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
#        x_train, y_train = shuffle_data(train_img, train_cla)
#        Y_train = keras.utils.to_categorical(np.array(y_train), num_classes)
        train_data = getSmall_data(x_train, Y_train, batch_size)
        for j in range(batch_num):
            x, y = next(train_data)
            sess.run(train_step, feed_dict = {x_ : x, y_ : y})
        if (i+1) % 10 == 0:
            train_accuracy = sess.run(accuracy, feed_dict = {x_ : x, y_ : y})
            print('step:{}, train:{:.5f}'.format(i+1, train_accuracy))
    test_accuracy = sess.run(accuracy, feed_dict = {x_ : x_test, y_ : Y_test})
    print('test:{:.5f}'.format(test_accuracy))

#plt.figure(figsize=(10,10))
#for i in range(25):
#    plt.subplot(5,5,(i+1))
#    plt.imshow(trxl[i + 610])
##    plt.title(y_train[i])
#    plt.axis('off')
#plt.show()

