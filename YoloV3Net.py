# -*- coding: utf-8 -*-
# @Time    : 2019/12/11 21:28
# @Author  : Suke0
# @Email   : 652434288@qq.com
# @File    : YoloV3Net.py
# @Software: PyCharm

from DarkNet53 import DarkNet53
from DetectNet import DetectNet
import tensorflow as tf
import cv2
from PIL import ImageDraw,Image
from weights import *


class YoloV3Net(tf.keras.Model):
    def __init__(self,n_classes):
        super(YoloV3Net,self).__init__()
        self.num_layers = 110
        self.n_classes = n_classes
        self.bodyNet = DarkNet53()
        self.detectNet = DetectNet(n_classes)
        pass

    def load_darknet_params(self, weights_file, skip_detect_layer=False):
        weight_reader = WeightReader(weights_file)
        weight_reader.load_weights(self, skip_detect_layer)
        pass

    def get_variables(self, layer_idx, suffix=None):
        if suffix:
            if suffix == "bias":
                find_name = "layer_{}/{}".format(layer_idx, suffix)
            else:
                find_name = "layer_{}_1/{}".format(layer_idx, suffix)
        else:
            find_name = "layer_{}/kernel".format(layer_idx)
        variables = []
        for v in self.variables:
            if find_name in v.name:
                variables.append(v)
                return variables

    def predict(self,input_array):
        output1, output2, output3 = self.call(tf.constant(input_array.astype(np.float32)))
        return output1,output2,output3
        pass

    def call(self,input_tensor,training = False):#(batch_size,416,416,3)
        res1,res2,res3 = self.bodyNet(input_tensor,training)
        output1,output2,output3 = self.detectNet(res1, res2, res3,training)
        # ouput_1.shape=(batch_size,13,13,3*(4+1+n_classes)),ouput_2.shape=(batch_size,26,26,3*(4+1+n_classes)),ouput_3.shape=(batch_size,52,52,3*(4+1+n_classes))
        return output1, output2, output3
        pass

    def detect(self, image, anchors, net_size=(416,416)):
        image_h, image_w, _ = image.shape
        preprocess_img = cv2.resize(image / 255., net_size)
        new_image = np.expand_dims(preprocess_img, axis=0)  # (1,416,416,3)
        # predict
        # [(batch_size, 13, 13, 3*(4+1+n_classes)), (batch_size,26,26,3*(4+1+n_classes)), (batch_size,52,52,3*(4+1+n_classes))]
        ys = self.predict(new_image)
        anchors = np.array(anchors).reshape(3, 6)
        results = []
        for i in range(len(ys)):
            # ys[i][0].shape = (13, 13, 3*(4+1+n_classes))
            res_tensor = self.output_handled(ys[i], anchors[3 - (i + 1)], net_size)
            results.append(res_tensor) # 三个尺度的形状分别为：[1, 507（13*13*3）, 5+c]、[1, 2028, 5+c]、[1, 8112, 5+c]
            pass
        detections = tf.concat([results[0], results[1], results[2]], axis=1)
        detected_boxes = self.detections_boxes(detections)
        conf_threshold = 0.5  # 置信度阈值
        iou_threshold = 0.4  # 重叠区域阈值
        filtered_boxes = self.non_max_suppression(detected_boxes, confidence_threshold=conf_threshold,
                                                  iou_threshold=iou_threshold)

        return filtered_boxes
        pass


    def output_handled(self, predictions, anchors, img_size, obj_thresh=0.3):
        anchors =anchors.reshape(3, 2)
        num_anchors = len(anchors)  # 候选框个数

        shape = predictions.get_shape().as_list()
        print("shape", shape)  # 三个尺度的形状分别为：[1, 13, 13, 3*(5+c)]、[1, 26, 26, 3*(5+c)]、[1, 52, 52, 3*(5+c)]
        grid_size = shape[1:3]  # 去 NHWC中的HW
        dim = grid_size[0] * grid_size[1]  # 每个格子所包含的像素
        bbox_attrs = 5 + self.n_classes

        predictions = tf.reshape(predictions, [-1, num_anchors * dim, bbox_attrs])  # 把h和w展开成dim

        stride = (img_size[0] // grid_size[0], img_size[1] // grid_size[1])  # 缩放参数 32（416/13）


        anchors = [(a[0] / stride[0], a[1] / stride[1]) for a in anchors]  # 将候选框的尺寸同比例缩小

        # 将包含边框的单元属性拆分
        box_centers, box_sizes, confidence, classes = tf.split(predictions, [2, 2, 1, self.n_classes], axis=-1)

        box_centers = tf.nn.sigmoid(box_centers)
        confidence = tf.nn.sigmoid(confidence)

        grid_x = tf.range(grid_size[0], dtype=tf.float32)  # 定义网格索引0,1,2...n
        grid_y = tf.range(grid_size[1], dtype=tf.float32)  # 定义网格索引0,1,2,...m
        a, b = tf.meshgrid(grid_x, grid_y)  # 生成网格矩阵 a0，a1.。。an（共M行）  ， b0，b0，。。。b0（共n个），第二行为b1

        x_offset = tf.reshape(a, (-1, 1))  # 展开 一共dim个
        y_offset = tf.reshape(b, (-1, 1))

        x_y_offset = tf.concat([x_offset, y_offset], axis=-1)  # 连接----[dim,2]
        x_y_offset = tf.reshape(tf.tile(x_y_offset, [1, num_anchors]), [1, -1, 2])  # 按候选框的个数复制xy（【1，n】代表第0维一次，第1维n次）

        box_centers = box_centers + x_y_offset  # box_centers为0-1，x_y为具体网格的索引，相加后，就是真实位置(0.1+4=4.1，第4个网格里0.1的偏移)
        box_centers = box_centers * stride  # 真实尺寸像素点

        anchors = tf.tile(anchors, [dim, 1])
        anchors = tf.cast(tf.expand_dims(anchors,0),tf.float32)
        box_sizes = tf.exp(box_sizes) * anchors  # 计算边长：hw
        box_sizes = box_sizes * stride  # 真实边长

        detections = tf.concat([box_centers, box_sizes, confidence], axis=-1)
        classes = tf.nn.sigmoid(classes)

        predictions = tf.concat([detections, classes], axis=-1)  # 将转化后的结果合起来

        print(predictions.get_shape())  # 三个尺度的形状分别为：[1, 507（13*13*3）, 5+c]、[1, 2028, 5+c]、[1, 8112, 5+c]

        return predictions  # 返回预测值

    def load_pre_trained(self, weights_file):
        var_list = self.variables
        with open(weights_file, "rb") as fp:
            _ = np.fromfile(fp, dtype=np.int32, count=5)  # 跳过前5个int32
            weights = np.fromfile(fp, dtype=np.float32)
        ptr = 0
        i = 0
        # assign_ops = []
        while i < len(var_list) - 1:
            var1 = var_list[i]
            var2 = var_list[i + 1]
            # 找到卷积项
            if 'kernel' in var1.name.split('/')[-1]:
                # 找到BN参数项
                if 'gamma' in var2.name.split('/')[-1]:
                    # 加载批量归一化参数
                    gamma, beta, mean, var = var_list[i + 1:i + 5]
                    batch_norm_vars = [beta, gamma, mean, var]
                    for var in batch_norm_vars:
                        shape = var.shape.as_list()
                        num_params = np.prod(shape)
                        var_weights = weights[ptr:ptr + num_params].reshape(shape)
                        ptr += num_params
                        var.assign(var_weights)
                    i += 4  # 已经加载了4个变量，指针移动4
                elif 'bias' in var2.name.split('/')[-1]:
                    bias = var2
                    bias_shape = bias.shape.as_list()
                    bias_params = np.prod(bias_shape)
                    bias_weights = weights[ptr:ptr + bias_params].reshape(bias_shape)
                    ptr += bias_params
                    #print(bias_weights)
                    bias.assign(bias_weights)
                    i += 1  # 移动指针

                shape = var1.shape.as_list()
                num_params = np.prod(shape)
                # 加载权重
                var_weights = weights[ptr:ptr + num_params].reshape((shape[3], shape[2], shape[0], shape[1]))
                var_weights = np.transpose(var_weights, (2, 3, 1, 0))
                ptr += num_params
                var1.assign(var_weights)
                i += 1
        pass


    #定义函数：将中心点、高、宽坐标 转化为[x0, y0, x1, y1]坐标形式
    def detections_boxes(self,detections):
        center_x, center_y, width, height, attrs = tf.split(detections, [1, 1, 1, 1, -1], axis=-1)
        w2 = width / 2
        h2 = height / 2
        x0 = np.maximum(np.minimum(center_x - w2,416),0)
        y0 = np.maximum(np.minimum(center_y - h2,416),0)
        x1 = np.maximum(np.minimum(center_x + w2,416),0)
        y1 = np.maximum(np.minimum(center_y + h2,416),0)

        boxes = tf.concat([x0, y0, x1, y1], axis=-1)
        detections = tf.concat([boxes, attrs], axis=-1)
        return detections

    #定义函数计算两个框的内部重叠情况（IOU）box1，box2为左上、右下的坐标[x0, y0, x1, x2]
    def iou(self,box1, box2):

        b1_x0, b1_y0, b1_x1, b1_y1 = box1
        b2_x0, b2_y0, b2_x1, b2_y1 = box2

        int_x0 = max(b1_x0, b2_x0)
        int_y0 = max(b1_y0, b2_y0)
        int_x1 = min(b1_x1, b2_x1)
        int_y1 = min(b1_y1, b2_y1)

        int_area = (int_x1 - int_x0) * (int_y1 - int_y0)

        b1_area = (b1_x1 - b1_x0) * (b1_y1 - b1_y0)
        b2_area = (b2_x1 - b2_x0) * (b2_y1 - b2_y0)

        #分母加个1e-05，避免除数为 0
        iou = int_area / (b1_area + b2_area - int_area + 1e-05)
        return iou


    #使用NMS方法，对结果去重
    def non_max_suppression(self,predictions_with_boxes, confidence_threshold, iou_threshold=0.4):

        conf_mask = np.expand_dims((predictions_with_boxes[:, :, 4] > confidence_threshold), -1)
        predictions = predictions_with_boxes * conf_mask
        predictions = predictions.numpy()
        result = {}
        for i, image_pred in enumerate(predictions):
            shape = image_pred.shape
            print("shape1",shape)
            non_zero_idxs = np.nonzero(image_pred)

            idx = list(set(non_zero_idxs[0]))
            #idx = non_zero_idxs[0]
            image_pred = image_pred[idx]
            print("shape2",image_pred.shape)
            image_pred = image_pred.reshape(-1, shape[-1])

            bbox_attrs = image_pred[:, :5]
            classes = image_pred[:, 5:]
            classes = np.argmax(classes, axis=-1)

            unique_classes = list(set(classes.reshape(-1)))

            for cls in unique_classes:
                cls_mask = classes == cls
                cls_boxes = bbox_attrs[np.nonzero(cls_mask)]
                cls_boxes = cls_boxes[cls_boxes[:, -1].argsort()[::-1]]
                cls_scores = cls_boxes[:, -1]
                cls_boxes = cls_boxes[:, :-1]

                while len(cls_boxes) > 0:
                    box = cls_boxes[0]
                    score = cls_scores[0]
                    if not cls in result:
                        result[cls] = []
                    result[cls].append((box, score))
                    cls_boxes = cls_boxes[1:]
                    ious = np.array([self.iou(box, x) for x in cls_boxes])
                    iou_mask = ious < iou_threshold
                    cls_boxes = cls_boxes[np.nonzero(iou_mask)]
                    cls_scores = cls_scores[np.nonzero(iou_mask)]

        return result


    def convert_to_original_size(self, box, size, original_size):
        ratio = original_size / size
        box = box.reshape(2, 2) * ratio
        return list(box.reshape(-1))


    # 将级别结果显示在图片上
    def draw_boxes(self, i, boxes, img_file, cls_names, detection_size):
        img = Image.open(img_file)
        draw = ImageDraw.Draw(img)

        for cls, bboxs in boxes.items():
            color = tuple(np.random.randint(0, 256, 3))
            for box, score in bboxs:
                box = self.convert_to_original_size(box, np.array(detection_size), np.array(img.size))
                draw.rectangle(box, outline=color)
                draw.text(box[:2], '{} {:.2f}%'.format(cls_names[cls], score * 100), fill=color)
                print('{} {:.2f}%'.format(cls_names[cls], score * 100), box[:2])
        img.save(f"output_img{i}.jpg")
        img.show()
        pass

    pass


# if __name__=='__main__':
#     import numpy as np
#
#     imgs = np.random.randn(3, 416, 416, 3).astype(np.float32)
#     input_tensor = tf.constant(imgs)
#     print(123)
#     #4D tensor with shape: (samples, channels, rows, cols)
#     net = YoloV3Net(training=True)
#     result = net(input_tensor)
#     print(result[0].shape,result[1].shape,result[2].shape)
#
#     for v in net.variables:
#         if('yolov3_net/detect_net/sequential_59/batch_normalization_71/gamma:0' == v.name):
#             print(v)
#     pass