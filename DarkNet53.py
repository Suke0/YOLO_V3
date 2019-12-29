# -*- coding: utf-8 -*-
# @Time    : 2019/12/12 0:08
# @Author  : Suke0
# @Email   : 652434288@qq.com
# @File    : DarkNet53.py
# @Software: PyCharm
import tensorflow as tf

class DarkNet53(tf.keras.Model):
    def __init__(self):
        super(DarkNet53,self).__init__()
        self.layers_1 = Conv2D_BN_LeakyReLU(0,32,3,1)#输入416，输出416

        # 第一次下采样
        self.layers_2a = tf.keras.layers.ZeroPadding2D(((1,0),(1,0))) #输入416，输出417
        self.layers_2 = Conv2D_BN_LeakyReLU(1,64,3,2) #下采样,输入417，输出(417 - 3) / 2 + 1 = 208

        # resBlock_1---------------------------start------------------------输入208
        self.layers_3 = Conv2D_BN_LeakyReLU(2,32,1,1)
        self.layers_4 = Conv2D_BN_LeakyReLU(3,64,3,1)
        self.layers_4a = tf.keras.layers.Add()
        # resBlock_1---------------------------end--------------------------输出208

        # 第二次下采样
        self.layers_5a = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))#输入208，输出209
        self.layers_5 = Conv2D_BN_LeakyReLU(4,128, 3, 2)#下采样，输入,209，输出(209 - 3) / 2 + 1 = 104

        # resBlock_2---------------------------start------------------------输入104
        self.layers_6 = Conv2D_BN_LeakyReLU(5,64, 1, 1)
        self.layers_7 = Conv2D_BN_LeakyReLU(6,128, 3, 1)
        self.layers_7a = tf.keras.layers.Add()

        self.layers_8 = Conv2D_BN_LeakyReLU(7,64, 1, 1)
        self.layers_9 = Conv2D_BN_LeakyReLU(8,128, 3, 1)
        self.layers_9a = tf.keras.layers.Add()
        # resBlock_2---------------------------end------------------------输出104

        # 第三次下采样
        self.layers_10a = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))#输入104，输出105
        self.layers_10 = Conv2D_BN_LeakyReLU(9,256, 3, 2) # 下采样，输入105，输出(105 - 3) / 2 + 1 = 52

        # resBlock_8---------------------------start------------------------输入52
        self.layers_11 = Conv2D_BN_LeakyReLU(10,128, 1, 1)
        self.layers_12 = Conv2D_BN_LeakyReLU(11,256, 3, 1)
        self.layers_12a = tf.keras.layers.Add()

        self.layers_13 = Conv2D_BN_LeakyReLU(12,128, 1, 1)
        self.layers_14 = Conv2D_BN_LeakyReLU(13,256, 3, 1)
        self.layers_14a = tf.keras.layers.Add()

        self.layers_15 = Conv2D_BN_LeakyReLU(14,128, 1, 1)
        self.layers_16 = Conv2D_BN_LeakyReLU(15,256, 3, 1)
        self.layers_16a = tf.keras.layers.Add()

        self.layers_17 = Conv2D_BN_LeakyReLU(16,128, 1, 1)
        self.layers_18 = Conv2D_BN_LeakyReLU(17,256, 3, 1)
        self.layers_18a = tf.keras.layers.Add()

        self.layers_19 = Conv2D_BN_LeakyReLU(18,128, 1, 1)
        self.layers_20 = Conv2D_BN_LeakyReLU(19,256, 3, 1)
        self.layers_20a = tf.keras.layers.Add()

        self.layers_21 = Conv2D_BN_LeakyReLU(20,128, 1, 1)
        self.layers_22 = Conv2D_BN_LeakyReLU(21,256, 3, 1)
        self.layers_22a = tf.keras.layers.Add()

        self.layers_23 = Conv2D_BN_LeakyReLU(22,128, 1, 1)
        self.layers_24 = Conv2D_BN_LeakyReLU(23,256, 3, 1)
        self.layers_24a = tf.keras.layers.Add()

        self.layers_25 = Conv2D_BN_LeakyReLU(24,128, 1, 1)
        self.layers_26 = Conv2D_BN_LeakyReLU(25,256, 3, 1)
        self.layers_26a = tf.keras.layers.Add()
        # resBlock_8---------------------------end------------------------输出52

        #第四次下采样
        self.layers_27a = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))#输入52，输出53
        self.layers_27 = Conv2D_BN_LeakyReLU(26,512, 3, 2) # 下采样#输入53，输出(53 - 3) / 2 + 1 = 26

        # resBlock_8---------------------------start------------------------输入26
        self.layers_28 = Conv2D_BN_LeakyReLU(27,256, 1, 1)
        self.layers_29 = Conv2D_BN_LeakyReLU(28,512, 3, 1)
        self.layers_29a = tf.keras.layers.Add()

        self.layers_30 = Conv2D_BN_LeakyReLU(29,256, 1, 1)
        self.layers_31 = Conv2D_BN_LeakyReLU(30,512, 3, 1)
        self.layers_31a = tf.keras.layers.Add()

        self.layers_32 = Conv2D_BN_LeakyReLU(31,256, 1, 1)
        self.layers_33 = Conv2D_BN_LeakyReLU(32,512, 3, 1)
        self.layers_33a = tf.keras.layers.Add()

        self.layers_34 = Conv2D_BN_LeakyReLU(33,256, 1, 1)
        self.layers_35 = Conv2D_BN_LeakyReLU(34,512, 3, 1)
        self.layers_35a = tf.keras.layers.Add()

        self.layers_36 = Conv2D_BN_LeakyReLU(35,256, 1, 1)
        self.layers_37 = Conv2D_BN_LeakyReLU(36,512, 3, 1)
        self.layers_37a = tf.keras.layers.Add()

        self.layers_38 = Conv2D_BN_LeakyReLU(37,256, 1, 1)
        self.layers_39 = Conv2D_BN_LeakyReLU(38,512, 3, 1)
        self.layers_39a = tf.keras.layers.Add()

        self.layers_40 = Conv2D_BN_LeakyReLU(39,256, 1, 1)
        self.layers_41 = Conv2D_BN_LeakyReLU(40,512, 3, 1)
        self.layers_41a = tf.keras.layers.Add()

        self.layers_42 = Conv2D_BN_LeakyReLU(41,256, 1, 1)
        self.layers_43 = Conv2D_BN_LeakyReLU(42,512, 3, 1)
        self.layers_43a = tf.keras.layers.Add()
        # resBlock_8---------------------------end------------------------输出26

        #第五次下采样
        self.layers_44a = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))#输入26，输出27
        self.layers_44 = Conv2D_BN_LeakyReLU(43,1024, 3, 2) # 下采样#输入26，输出(27 - 3) / 2 + 1 = 13

        # resBlock_4---------------------------start------------------------输入13
        self.layers_45 = Conv2D_BN_LeakyReLU(44,512, 1, 1)
        self.layers_46 = Conv2D_BN_LeakyReLU(45,1024, 3, 1)
        self.layers_46a = tf.keras.layers.Add()

        self.layers_47 = Conv2D_BN_LeakyReLU(46,512, 1, 1)
        self.layers_48 = Conv2D_BN_LeakyReLU(47,1024, 3, 1)
        self.layers_48a = tf.keras.layers.Add()

        self.layers_49 = Conv2D_BN_LeakyReLU(48,512, 1, 1)
        self.layers_50 = Conv2D_BN_LeakyReLU(49,1024, 3, 1)
        self.layers_50a = tf.keras.layers.Add()

        self.layers_51 = Conv2D_BN_LeakyReLU(50,512, 1, 1)
        self.layers_52 = Conv2D_BN_LeakyReLU(51,1024, 3, 1)#
        self.layers_52a = tf.keras.layers.Add()
        # resBlock_4---------------------------end------------------------输出13
        pass

    def call(self,input_tensor,training = False):#(batch_size,416,416,3)
        x = self.layers_1(input_tensor,training)  # 输入(batch_size,416,416,3)，输出(batch_size,416,416,32)

        # 第一次下采样
        x = self.layers_2a(x)  # 输入(batch_size,416,416,32)，输出(batch_size,417,417,32)
        x = self.layers_2(x,training)  # 下采样,输入(batch_size,417,417,32) ，输出(417 - 3) / 2 + 1 = 208  (batch_size,208,208,64)

        # resBlock_1---------------------------start------------------------输入208
        x_in = self.layers_3(x,training) # 输入(batch_size,208,208,64)，输出(batch_size,208,208,32)
        x_in = self.layers_4(x_in,training) # 输入(batch_size,208,208,32)，输出(batch_size,208,208,64)
        x = self.layers_4a([x,x_in]) # 输出(batch_size,208,208,64)
        # resBlock_1---------------------------end--------------------------输出208

        # 第二次下采样
        x = self.layers_5a(x)  # 输入208，输出209
        x = self.layers_5(x,training) # 下采样，输入,209，输出(209 - 3) / 2 + 1 = 104

        # resBlock_2---------------------------start------------------------输入104
        x_in = self.layers_6(x,training)
        x_in = self.layers_7(x_in,training)
        x = self.layers_7a([x,x_in])

        x_in = self.layers_8(x,training)
        x_in = self.layers_9(x_in,training)
        x = self.layers_9a([x,x_in])
        # resBlock_2---------------------------end------------------------输出104

        # 第三次下采样
        x = self.layers_10a(x)  # 输入104，输出105
        x = self.layers_10(x,training)  # 下采样，输入105，输出(105 - 3) / 2 + 1 = 52

        # resBlock_8---------------------------start------------------------输入52
        x_in = self.layers_11(x,training)
        x_in = self.layers_12(x_in,training)
        x = self.layers_12a([x,x_in])

        x_in = self.layers_13(x,training)
        x_in = self.layers_14(x_in,training)
        x = self.layers_14a([x,x_in])

        x_in = self.layers_15(x,training)
        x_in = self.layers_16(x_in,training)
        x = self.layers_16a([x,x_in])

        x_in = self.layers_17(x,training)
        x_in = self.layers_18(x_in,training)
        x = self.layers_18a([x,x_in])

        x_in = self.layers_19(x,training)
        x_in = self.layers_20(x_in,training)
        x = self.layers_20a([x,x_in])

        x_in = self.layers_21(x,training)
        x_in = self.layers_22(x_in,training)
        x = self.layers_22a([x,x_in])

        x_in = self.layers_23(x,training)
        x_in = self.layers_24(x_in,training)
        x = self.layers_24a([x,x_in])

        x_in = self.layers_25(x,training)
        x_in = self.layers_26(x_in,training)
        x = self.layers_26a([x,x_in])
        result1 = x  #(batch_size,52,52,256)

        # resBlock_8---------------------------end------------------------输出52

        # 第四次下采样
        x = self.layers_27a(x)  # 输入52，输出53
        x = self.layers_27(x,training)  # 下采样#输入53，输出(53 - 3) / 2 + 1 = 26

        # resBlock_8---------------------------start------------------------输入26
        x_in = self.layers_28(x,training)
        x_in = self.layers_29(x_in,training)
        x = self.layers_29a([x,x_in])

        x_in = self.layers_30(x,training)
        x_in = self.layers_31(x_in,training)
        x = self.layers_31a([x,x_in])

        x_in = self.layers_32(x,training)
        x_in = self.layers_33(x_in,training)
        x = self.layers_33a([x,x_in])

        x_in = self.layers_34(x,training)
        x_in = self.layers_35(x_in,training)
        x = self.layers_35a([x,x_in])

        x_in = self.layers_36(x,training)
        x_in = self.layers_37(x_in,training)
        x = self.layers_37a([x,x_in])

        x_in = self.layers_38(x,training)
        x_in = self.layers_39(x_in,training)
        x = self.layers_39a([x,x_in])

        x_in = self.layers_40(x,training)
        x_in = self.layers_41(x_in,training)
        x = self.layers_41a([x,x_in])

        x_in = self.layers_42(x,training)
        x_in = self.layers_43(x_in,training)
        x = self.layers_43a([x,x_in])
        result2 = x #(batch_size,26,26,512)

        # resBlock_8---------------------------end------------------------输出26

        # 第五次下采样
        x = self.layers_44a(x)  # 输入26，输出27
        x = self.layers_44(x,training)  # 下采样#输入26，输出(27 - 3) / 2 + 1 = 13

        # resBlock_4---------------------------start------------------------输入13
        x_in = self.layers_45(x,training)
        x_in = self.layers_46(x_in,training)
        x = self.layers_46a([x,x_in])

        x_in = self.layers_47(x,training)
        x_in = self.layers_48(x_in,training)
        x = self.layers_48a([x,x_in])

        x_in = self.layers_49(x,training)
        x_in = self.layers_50(x_in,training)
        x = self.layers_50a([x,x_in])

        x_in = self.layers_51(x,training)
        x_in = self.layers_52(x_in,training)
        x = self.layers_52a([x,x_in])

        result3 = x #(batch_size,13,13,1024)
        # resBlock_4---------------------------end------------------------输出13
        return result1,result2,result3
        pass
    pass


class Conv2D_BN_LeakyReLU(tf.keras.Model):
    def __init__(self,layer_idx,filters,kernel_size,strides=(1, 1)):
        super(Conv2D_BN_LeakyReLU,self).__init__()
        if strides == 2:
            padding_value = 'valid'
        else:
            padding_value = 'same'
            pass
        layer_name = "layer_{}".format(str(layer_idx))
        self.layer_1a = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding=padding_value,use_bias=False,name=layer_name)
        self.layer_1b = tf.keras.layers.BatchNormalization(renorm_momentum=0.9,momentum=0.9,epsilon=1e-05,name=layer_name)
        self.layer_1c = tf.keras.layers.LeakyReLU(alpha=0.1)
        pass


    def call(self, input_tensor, training=False):
        x = self.layer_1a(input_tensor)
        x = self.layer_1b(x, training = training)
        x = self.layer_1c(x)
        return x
        pass
    pass

if __name__=='__main__':
    import numpy as np

    imgs = np.random.randn(1, 416, 416, 3).astype(np.float32)
    input_tensor = tf.constant(imgs)
    print(123)
    #4D tensor with shape: (samples, channels, rows, cols)
    net = DarkNet53()
    result = net(input_tensor)
    print(result[0].shape,result[1].shape,result[2].shape)
    pass