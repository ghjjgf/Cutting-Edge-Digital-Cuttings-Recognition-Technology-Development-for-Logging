# 该包提供类似关系型数据库（Oracle）中以表格形式处理数据的功能，更加灵活
import pandas as pd
import tensorflow as tf
import numpy as np
import datetime as dt
import cv2
import os
from collections import Counter
#tqdm是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息，用户只需要封装任意的迭代器
from tqdm import tqdm
# ----------------部分参数设置----------------------
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# 训练数据集地址
train_img_url = 'data/rock_sample/1/' # input
# 将所有的图片resize成512*512
w = 512
h = 512
c = 3
learning_rate=0.0001
# GPU内存限制数
gpu_memory = 5120
# 每次执行图片数
batch_size = 3
# 网络fit迭代次数
num_epochs = 100
#-------------------设置显存占用率-----------------
gpus = tf.config.experimental.list_physical_devices('GPU')
# 对需要进行限制的GPU进行设置
tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_memory)])
# 查看GPU是否可用
print(tf.test.is_gpu_available())
data_path = 'data/rock_label_1.csv' # input
# 读取类标数据
data_df = pd.read_csv(data_path,encoding='gbk')
# 对label进行Encoder编码
from sklearn.preprocessing import LabelEncoder
label_data = pd.DataFrame(columns=['label'])
le_credit_level = LabelEncoder().fit(data_df['样本类别'])
label_data['label'] = le_credit_level.transform(data_df['样本类别'])
label_counts = len(list(label_data['label'].value_counts()))
label_data['label'].value_counts()

# 定义查看图片函数
def img_show(img):
    cv2.namedWindow("Image") 
    cv2.imshow("Image", img) 
    cv2.waitKey (0) 
    cv2.destroyAllWindows()
# 查看图片
img1 = cv2.imread('data/rock_sample/1/12-1.bmp') # input
img1 = cv2.resize(img1, (w, h))
# img_show(img1)

#数据预处理
#图片加载
# 读取图片+数据处理函数
def read_img(path,label_pd):
    print("数据集地址："+path)
    imgs = []
    labels = []
    for root, dirs, files in os.walk(path):
        for file in tqdm(files):
            # print(path+'/'+file+'/'+folder)
            # 读取的图片
            # img = io.imread(os.path.join(root,file))
            img = cv2.imread(os.path.join(root, file))
            # skimage.transform.resize(image, output_shape)改变图片的尺寸
            img = cv2.resize(img, (w, h))
            # 将读取的图片数据加载到imgs[]列表中
            imgs.append(img)
    labels = list(label_pd['label'])
    # 将读取的图片和labels信息，转化为numpy结构的ndarr(N维数组对象（矩阵）)数据信息
    return imgs,labels
# 调用读取图片的函数，得到图片和labels的数据集
data1, label1 = read_img(train_img_url,label_data)

# 将图像进行随机翻转，裁剪
def img_cut(imgs,label):
    imgs_out = []
    label_out = []
    for i in tqdm(range(len(imgs))):
        # 添加原图
        imgs_out.append(imgs[i])
        label_out.append(label[i])
        # 随机翻转
        for e in range(-1,2):
            # 1:水平翻转,0:垂直翻转,-1:水平垂直翻转
            f_img = cv2.flip(imgs[i], e)
            # 添加翻转图像
            imgs_out.append(f_img)
            # 添加类标
            label_out.append(label[i])
            # 裁剪翻转后图片
            # 生成裁剪随机数：70%-90%大小
            rd_num = np.random.uniform(0.7, 0.9)
            # 生成随机裁剪长宽
            rd_w = int(w * rd_num)
            rd_h = int(h * rd_num)
            # 进行裁剪
            crop_img = tf.image.random_crop(imgs[i],[rd_w,rd_h,c]).numpy()
            # 重新调整大小
            re_img = cv2.resize(crop_img, (w, h))
            # 添加裁剪图像
            imgs_out.append(re_img)
            # 添加类标
            label_out.append(label[i])
        # 原图随机裁剪，执行3次
        for f in range(3):
            # 生成裁剪随机数：50%-80%大小
            rd_num = np.random.uniform(0.5, 0.8)
            # 生成随机裁剪长宽
            rd_w = int(w * rd_num)
            rd_h = int(h * rd_num)
            # 进行裁剪
            crop_img = tf.image.random_crop(imgs[i],[rd_w,rd_h,c]).numpy()
            # 重新调整大小
            re_img = cv2.resize(crop_img, (w, h))
            # 添加裁剪图像
            imgs_out.append(re_img)
            # 添加类标
            label_out.append(label[i])
    return imgs_out,label_out
# 执行函数
data2,label2 = img_cut(data1,label1)
# 将图像数据转换为numpy数组
data_sample,label_sample = np.asarray(data2, np.float32), np.asarray(label2, np.int32)

# 打乱顺序
# 读取data矩阵的第一维数（图片的个数）
num_example = data_sample.shape[0]
# 产生一个num_example范围，步长为1的序列
arr = np.arange(num_example)
# 调用函数，打乱顺序
np.random.shuffle(arr)
# 按照打乱的顺序，重新排序
data= data_sample[arr]
label= label_sample[arr]

#恒等模块——identity_block
def identity_block(X,f,filters,stage,block):
    """
    三层的恒等残差块
    param :
        X -- 输入的张量，维度为（m, n_H_prev, n_W_prev, n_C_prev）
        f -- 整数，指定主路径的中间 CONV 窗口的形状
        filters -- python整数列表，定义主路径的CONV层中的过滤器数目
        stage -- 整数，用于命名层，取决于它们在网络中的位置
        block --字符串/字符，用于命名层，取决于它们在网络中的位置    
    return:    
        X -- 三层的恒等残差块的输出，维度为：(n_H, n_W, n_C)    
    """
    #定义基本的名字
    conv_name_base = "res"+str(stage)+block+"_branch"
    bn_name_base = "bn"+str(stage)+block+"_branch"
    #过滤器
    F1,F2,F3=filters
    #保存输入值，后将输入值返回主路径
    X_shortcut = X
    
    #主路径第一部分
    X = layers.Conv2D(filters=F1,kernel_size=(1,1),strides=(1,1),padding="valid",
               name=conv_name_base+"2a",kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis=3,name=bn_name_base+"2a")(X)
    X = layers.Activation("relu")(X)
    
    #主路径第二部分
    X = layers.Conv2D(filters=F2,kernel_size=(f,f),strides=(1,1),padding="same",
               name=conv_name_base+"2b",kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis=3,name=bn_name_base+"2b")(X)
    X = layers.Activation("relu")(X)
    
    #主路径第三部分
    X = layers.Conv2D(filters=F3,kernel_size=(1,1),strides=(1,1),padding="valid",
               name=conv_name_base+"2c",kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis=3,name=bn_name_base+"2c")(X)

    # 主路径最后部分,为主路径添加shortcut并通过relu激活
    X = layers.Add()([X,X_shortcut])
    X = layers.Activation("relu")(X)
    return X

#卷积残差块——convolutional_block
def convolutional_block(X,f,filters,stage,block,s=2):
    """    
    param :
    X -- 输入的张量，维度为（m, n_H_prev, n_W_prev, n_C_prev）
    f -- 整数，指定主路径的中间 CONV 窗口的形状（过滤器大小，ResNet中f=3）
    filters -- python整数列表，定义主路径的CONV层中过滤器的数目
    stage -- 整数，用于命名层，取决于它们在网络中的位置
    block --字符串/字符，用于命名层，取决于它们在网络中的位置
    s -- 整数，指定使用的步幅
    return:
    X -- 卷积残差块的输出，维度为：(n_H, n_W, n_C)
    """
    #定义基本名字
    conv_name_base = "res"+str(stage)+block+"_branch"
    bn_name_base = "bn"+str(stage)+block+"_branch"
    #过滤器
    F1,F2,F3=filters
    #保存输入值，后将输入值返回主路径
    X_shortcut = X
    
    # 主路径第一部分
    X = layers.Conv2D(filters=F1, kernel_size=(1, 1), strides=(s,s), padding="valid",
               name=conv_name_base + "2a", kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis=3, name=bn_name_base + "2a")(X)
    X = layers.Activation("relu")(X)
    
    # 主路径第二部分
    X = layers.Conv2D(filters=F2, kernel_size=(f, f), strides=(1,1), padding="same",
               name=conv_name_base + "2b", kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis=3, name=bn_name_base + "2b")(X)
    X = layers.Activation("relu")(X)
    
    # 主路径第三部分
    X = layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding="valid",
               name=conv_name_base + "2c", kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis=3, name=bn_name_base + "2c")(X)
    
    #shortcut路径
    X_shortcut = layers.Conv2D(filters=F3,kernel_size=(1,1),strides=(s,s),padding="valid",
                        name=conv_name_base+"1",kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = layers.BatchNormalization(axis=3,name=bn_name_base+"1")(X_shortcut)

    # 主路径最后部分,为主路径添加shortcut并通过relu激活
    X = layers.Add()([X, X_shortcut])
    X = layers.Activation("relu")(X)
    
    return X

#50层ResNet模型构建
def ResNet50(input_shape = (w,h,c),classes = label_counts):
    """ 
    构建50层的ResNet,结构为：
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER
    
    param : 
        input_shape -- 数据集图片的维度
        classes -- 整数，分类的数目
    return:
        model -- Keras中的模型实例
    """
    #将输入定义为维度大小为 input_shape的张量
    X_input = layers.Input(input_shape)
    # Zero-Padding
    X = layers.ZeroPadding2D((3,3))(X_input)
    # Stage 1
    X = layers.Conv2D(64,kernel_size=(7,7),strides=(2,2),name="conv1",kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis=3,name="bn_conv1")(X)
    X = layers.Activation("relu")(X)
    X = layers.MaxPooling2D(pool_size=(3,3),strides=(2,2))(X)
    # Stage 2
    X = convolutional_block(X,f=3,filters=[64,64,256],stage=2,block="a",s=1)
    X = identity_block(X,f=3,filters=[64,64,256],stage=2,block="b")
    X = identity_block(X,f=3,filters=[64,64,256],stage=2,block="c")
    #Stage 3
    X = convolutional_block(X,f=3,filters=[128,128,512],stage=3,block="a",s=2)
    for i in range(7):
        X = identity_block(X,f=3,filters=[128,128,512],stage=3,block="b"+str(i))
    # Stage 4 
    X = convolutional_block(X,f=3,filters=[256,256,1024],stage=4,block="a",s=2)
    for i in range(35):
        X = identity_block(X,f=3,filters=[256,256,1024],stage=4,block="b"+str(i))
    #Stage 5 
    X = convolutional_block(X,f=3,filters=[512,512,2048],stage=5,block="a",s=2)
    X = identity_block(X,f=3,filters=[256,256,2048],stage=5,block="b")
    X = identity_block(X,f=3,filters=[256,256,2048],stage=5,block="c")
    #最后阶段
    #平均池化 
    X = layers.AveragePooling2D(pool_size=(2,2))(X)
    #输出层
    X = layers.Flatten()(X)
    #展平
    X = layers.Dense(classes,activation="softmax",name="fc"+str(classes),kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
    #创建模型
    model = keras.models.Model(inputs=X_input,outputs=X,name="ResNet50")
    return model

#运行构建的模型图
ResNet_model = ResNet50(input_shape=(w,h,c),classes=label_counts)
#编译模型来配置学习过程
ResNet_model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
'''
第一轮训练100个epoch
'''
history_1 = ResNet_model.fit(data,label, epochs = num_epochs,batch_size = batch_size,validation_split=0.1)
#保存模型
ResNet_model.save('models_save/Resnet_model_0322_100.h5')
'''
第二轮训练100个epoch
'''
history_2 = ResNet_model.fit(data,label, epochs = num_epochs,batch_size = batch_size,validation_split=0.1)
#保存模型
ResNet_model.save('models_save/Resnet_model_0322_200.h5')
'''
第三轮训练100个epoch
'''
history_3 = ResNet_model.fit(data,label, epochs = num_epochs,batch_size = batch_size,validation_split=0.1)
#保存模型
ResNet_model.save('models_save/Resnet_model_0322_300.h5')
'''
第四轮训练100个epoch
'''
history_4 = ResNet_model.fit(data,label, epochs = num_epochs,batch_size = batch_size,validation_split=0.1)
#保存模型
ResNet_model.save('models_save/Resnet_model_0322_400.h5')

# 获取预测数据
data_pre,label_pre = np.asarray(data1, np.float32), np.asarray(label1, np.int32)
# 这里读取epoch100次的模型，也可以自行修改读取其他轮数模型
save_model = keras.models.load_model('models_save/Resnet_model_0322_100.h5')
#做出预测
predictions = save_model.predict(data_pre)

#转换为普通数组
y_pred = [np.argmax(x)for x in predictions]
# 评估模型
from sklearn.metrics import precision_score
# 最终模型准确率
pre_score = precision_score(label_pre, y_pred, average="micro")
print('最终模型准确率为：'+str(float(pre_score)))



