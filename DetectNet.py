#-- coding: utf-8 --

import tensorflow as tf

class DetectNet(tf.keras.Model):
    def __init__(self,n_classes = 80):
        super(DetectNet,self).__init__()
        n_features = 3 * (n_classes + 1 + 4)
        self.ouput_stage_1a_conv5 = FiveTimes_Conv2D_BN_LeakyReLU([512, 1024, 512, 1024, 512],
                                                                       [75, 76, 77, 78, 79])
        self.ouput_stage_1b_conv2 = Conv2D_BN_LeakyReLU_Conv2D([1024, n_features], [80, 81])

        self.ouput_stage_2a_upsampling = Conv2D_BN_LeakyReLU_UpSampling2D([256], [84])
        self.ouput_stage_2b_conv5 = FiveTimes_Conv2D_BN_LeakyReLU([256, 512, 256, 512, 256], [87, 88, 89, 90, 91])
        self.ouput_stage_2c_conv2 = Conv2D_BN_LeakyReLU_Conv2D([512, n_features], [92, 93])

        self.ouput_stage_3a_upsampling = Conv2D_BN_LeakyReLU_UpSampling2D([128], [96])
        self.ouput_stage_3b_conv5 = FiveTimes_Conv2D_BN_LeakyReLU([128, 256, 128, 256, 128],
                                                                       [99, 100, 101, 102, 103])
        self.ouput_stage_3c_conv2 = Conv2D_BN_LeakyReLU_Conv2D([256, n_features], [104, 105])

        pass

    def call(self,res1,res2,res3,training=False):
        # 13*13
        x =self.ouput_stage_1a_conv5(res3,training=False)
        ouput_1 = self.ouput_stage_1b_conv2(x,training=False)

        # 26 * 26
        x = self.ouput_stage_2a_upsampling(x,training=False)
        x = tf.keras.layers.concatenate([x,res2])
        x = self.ouput_stage_2b_conv5(x,training=False)
        ouput_2 = self.ouput_stage_2c_conv2(x,training=False)

        # 52 * 52
        x = self.ouput_stage_3a_upsampling(x,training=False)
        x = tf.keras.layers.concatenate([x, res1])
        x = self.ouput_stage_3b_conv5(x,training=False)
        ouput_3 = self.ouput_stage_3c_conv2(x,training=False)

        return ouput_1,ouput_2,ouput_3
        pass
    pass

class Conv2D_BN_LeakyReLU_UpSampling2D(tf.keras.Model):
    def __init__(self,filters,layer_idx):
        super(Conv2D_BN_LeakyReLU_UpSampling2D,self).__init__()
        self.layer_1a = tf.keras.layers.Conv2D(filters[0], 1, 1,padding='same',use_bias=False,name="layer_{}".format(str(layer_idx[0])))
        self.layer_1b = tf.keras.layers.BatchNormalization(renorm_momentum=0.9, momentum=0.9, epsilon=1e-05, name="layer_{}".format(str(layer_idx[0])))
        self.layer_1c = tf.keras.layers.LeakyReLU(alpha=0.1)

        self.layer_2 = tf.keras.layers.UpSampling2D(2)
        pass


    def call(self, input_tensor, training=False):
        x = self.layer_1a(input_tensor)
        x = self.layer_1b(x, training = training)
        x = self.layer_1c(x)
        x = self.layer_2(x)
        return x
        pass
    pass


class Conv2D_BN_LeakyReLU_Conv2D(tf.keras.Model):
    def __init__(self,filters,layer_idx):
        super(Conv2D_BN_LeakyReLU_Conv2D,self).__init__()
        self.layer_1a = tf.keras.layers.Conv2D(filters[0], 3, 1,padding='same',use_bias=False, name="layer_{}".format(str(layer_idx[0])))
        self.layer_1b = tf.keras.layers.BatchNormalization(renorm_momentum=0.9,momentum=0.9,epsilon=1e-05,name="layer_{}".format(str(layer_idx[0])))
        self.layer_1c = tf.keras.layers.LeakyReLU(alpha=0.1)

        self.layer_2 = tf.keras.layers.Conv2D(filters[1], 1, 1,padding='same',name="layer_{}".format(str(layer_idx[1])))
        pass


    def call(self,input_tensor,training = False):
        x = self.layer_1a(input_tensor)
        x = self.layer_1b(x,training = training)
        x = self.layer_1c(x)
        x = self.layer_2(x)
        return x
        pass
    pass

class FiveTimes_Conv2D_BN_LeakyReLU(tf.keras.Model):
    def __init__(self,filters,layer_idx):
        super(FiveTimes_Conv2D_BN_LeakyReLU,self).__init__()
        self.layer_1a = tf.keras.layers.Conv2D(filters[0], 1, 1, padding='same', use_bias=False, name="layer_{}".format(str(layer_idx[0])))
        self.layer_1b = tf.keras.layers.BatchNormalization(renorm_momentum=0.9,momentum=0.9,epsilon=1e-05,name="layer_{}".format(str(layer_idx[0])))
        self.layer_1c = tf.keras.layers.LeakyReLU(alpha=0.1)

        self.layer_2a = tf.keras.layers.Conv2D(filters[1], 3, 1, padding='same', use_bias=False, name="layer_{}".format(str(layer_idx[1])))
        self.layer_2b = tf.keras.layers.BatchNormalization(renorm_momentum=0.9,momentum=0.9,epsilon=1e-05, name="layer_{}".format(str(layer_idx[1])))
        self.layer_2c = tf.keras.layers.LeakyReLU(alpha=0.1)

        self.layer_3a = tf.keras.layers.Conv2D(filters[2], 1, 1, padding='same', use_bias=False, name="layer_{}".format(str(layer_idx[2])))
        self.layer_3b = tf.keras.layers.BatchNormalization(renorm_momentum=0.9,momentum=0.9,epsilon=1e-05, name="layer_{}".format(str(layer_idx[2])))
        self.layer_3c = tf.keras.layers.LeakyReLU(alpha=0.1)

        self.layer_4a = tf.keras.layers.Conv2D(filters[3], 3, 1, padding='same', use_bias=False, name="layer_{}".format(str(layer_idx[3])))
        self.layer_4b = tf.keras.layers.BatchNormalization(renorm_momentum=0.9,momentum=0.9,epsilon=1e-05, name="layer_{}".format(str(layer_idx[3])))
        self.layer_4c = tf.keras.layers.LeakyReLU(alpha=0.1)

        self.layer_5a = tf.keras.layers.Conv2D(filters[4], 1, 1, padding='same', use_bias=False, name="layer_{}".format(str(layer_idx[4])))
        self.layer_5b = tf.keras.layers.BatchNormalization(renorm_momentum=0.9,momentum=0.9,epsilon=1e-05, name="layer_{}".format(str(layer_idx[4])))
        self.layer_5c = tf.keras.layers.LeakyReLU(alpha=0.1)
        pass


    def call(self,input_tensor,training = False):
        x = self.layer_1a(input_tensor)
        x = self.layer_1b(x, training = training)
        x = self.layer_1c(x)

        x = self.layer_2a(x)
        x = self.layer_2b(x, training = training)
        x = self.layer_2c(x)

        x = self.layer_3a(x)
        x = self.layer_3b(x, training = training)
        x = self.layer_3c(x)

        x = self.layer_4a(x)
        x = self.layer_4b(x, training = training)
        x = self.layer_4c(x)

        x = self.layer_5a(x)
        x = self.layer_5b(x, training = training)
        x = self.layer_5c(x)
        return x
        pass

    pass

if __name__ =="__main__":
    import numpy as np
    res1 = tf.constant(np.random.randn(1,52,52,256).astype(np.float32))
    res2 = tf.constant(np.random.randn(1, 26, 26, 512).astype(np.float32))
    res3 = tf.constant(np.random.randn(1, 13, 13, 1024).astype(np.float32))

    net = DetectNet()
    result = net(res1,res2,res3,training=True)
    print(result[0].shape,result[1].shape,result[2].shape)
    pass