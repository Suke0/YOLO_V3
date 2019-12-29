#-- coding: utf-8 --
import os
import tensorflow as tf
import glob
from tqdm import tqdm
import cv2
import numpy as np
from BatchGenerator import BatchGenerator
from YoloV3Net import YoloV3Net
from YoloV3Loss import *


# 定义分类
# LABELS = [
#     "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
#     "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
#     "pottedplant", "sheep", "sofa", "train", "tvmonitor"
# ]

# LABELS = [
#         'person','bicycle','car','motorbike','aeroplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter',
#         'bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase',
#         'frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle',
#         'wine glass','cup', 'fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut',
#         'cake','chair','sofa','pottedplant','bed','diningtable','toilet','tvmonitor','laptop','mouse','remote','keyboard','cell phone',
#         'microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
#         ]
LABELS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
# 定义coco锚点候选框
COCO_ANCHORS = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]

# 获取当前目录
PROJECT_ROOT = os.path.dirname(__file__)
save_dir = "./model"
weights_dir = "./weights"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    pass
save_fname = os.path.join(save_dir, "yolov3_weights.h5")
weights_fname = os.path.join(weights_dir, "yolov3.weights")
img_size = 416

# 定义样本路径
train_ann_dir = os.path.join(PROJECT_ROOT, "voc_train_data", "ann", "*.xml")
train_img_dir = os.path.join(PROJECT_ROOT, "voc_train_data", "img")

ann_dir = os.path.join(PROJECT_ROOT, "data", "ann", "*.xml")
img_dir = os.path.join(PROJECT_ROOT, "data", "img")

val_ann_dir = os.path.join(PROJECT_ROOT, "voc_val_data", "ann", "*.xml")
val_img_dir = os.path.join(PROJECT_ROOT, "voc_val_data", "img")

test_ann_dir = os.path.join(PROJECT_ROOT, "voc_test_data", "ann", "*.xml")
test_img_dir = os.path.join(PROJECT_ROOT, "voc_test_data", "img")

batch_size = 8
# 获取该路径下的xml
train_ann_fnames = glob.glob(train_ann_dir)
ann_fnames = glob.glob(ann_dir)
val_ann_fnames = glob.glob(val_ann_dir)
test_ann_fnames = glob.glob(test_ann_dir)


# 定义训练参数
learning_rate = 1e-4
num_epoches = 100


# 循环整个数据集，进行模型训练
def loop_train(model, optimizer,generator=None, xml_fnames=None, img_dir=None):
    # 制作数据集
    if not generator:
        generator = BatchGenerator(xml_fnames,
                                   img_dir,
                                   net_size=img_size,
                                   anchors=COCO_ANCHORS,
                                   batch_size=batch_size,
                                   label_names=LABELS,
                                   jitter=False
                                   )
    # one epoch
    n_steps = generator.steps_per_epoch
    for step in tqdm(range(n_steps)):  # 按批次循环获取数据，并计算训练
        xs, ys_1, ys_2, ys_3 = generator.next_batch()
        xs = tf.convert_to_tensor(xs)
        ys_1 = tf.convert_to_tensor(ys_1)
        ys_2 = tf.convert_to_tensor(ys_2)
        ys_3 = tf.convert_to_tensor(ys_3)
        ys = [ys_1, ys_2, ys_3]
        with tf.GradientTape() as tape:
            logits = model(xs,training=True)
            loss_value = loss_fn(ys, logits, anchors=COCO_ANCHORS, image_size=[img_size, img_size])
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # if step % 100 == 0:
        #     print('Training loss (for one batch) at step %s: %s' % (step, float(loss_value)))
            #print('Seen so far: %s samples' % ((step + 1) * 8))
        # model.train_on_batch(xs, ys)
        # print(model.variables[-3])
        # print(model.variables[-4])
        pass

    pass


#循环整个数据集，进行loss值验证
def loop_validation(model,generator=None, xml_fnames=None, img_dir=None):
    if not generator:
        generator = BatchGenerator(xml_fnames,
                                   img_dir,
                                   net_size=img_size,
                                   anchors=COCO_ANCHORS,
                                   batch_size=batch_size,
                                   label_names=LABELS,
                                   jitter=False
                                   )
    n_steps = generator.steps_per_epoch
    loss_value = 0
    for _ in range(n_steps): #按批次循环获取数据，并计算loss
        xs, yolo_1, yolo_2, yolo_3 = generator.next_batch()
        xs=tf.convert_to_tensor(xs)
        yolo_1=tf.convert_to_tensor(yolo_1)
        yolo_2=tf.convert_to_tensor(yolo_2)
        yolo_3=tf.convert_to_tensor(yolo_3)
        ys = [yolo_1, yolo_2, yolo_3]
        ys_ = model(xs,training=True)
        loss_value += loss_fn(ys, ys_,anchors=COCO_ANCHORS,
            image_size=[img_size, img_size] )
    loss_value /= generator.steps_per_epoch
    return loss_value
    pass



def main():
    yolo_v3 = YoloV3Net(n_classes=len(LABELS))
    yolo_v3.build((1, 416, 416, 3))
    totolCount = 0
    generator = BatchGenerator(ann_fnames,
                               img_dir,
                               net_size=img_size,
                               anchors=COCO_ANCHORS,
                               batch_size=batch_size,
                               label_names=LABELS,
                               jitter=False
                               )
    for v in yolo_v3.variables:
        print(v.name + "_____" + str(v.shape))
        totolCount += np.prod(v.shape)
        pass
    print(f'totolCount:{totolCount}')
    yolo_v3.load_darknet_params(weights_fname, skip_detect_layer=True)   # 加载预训练模型
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    history = []
    # 按批次循环获取数据，并计算训练

    for step in range(num_epoches):
        loop_train(yolo_v3, optimizer,generator)
        #yolo_v3.save_weights(save_fname)
        loss_value = loop_validation(yolo_v3,generator)  # 验证
        print("{}-th loss = {}".format(step, loss_value))

        #收集loss
        history.append(loss_value)
        if loss_value == min(history):  # 只有loss创新低时再保存模型
            print("update weight {}".format(loss_value))

            yolo_v3.save_weights(save_fname)
        pass

    pass
################################################################
#使用模型

def detect():
    yolo_v3 = YoloV3Net(n_classes=len(LABELS))
    IMAGE_FOLDER = os.path.join(PROJECT_ROOT,  "data", "test","*.png")
    img_fnames = glob.glob(IMAGE_FOLDER)

    #将训练好的模型载入
    yolo_v3.build((1,416,416,3))
    yolo_v3.load_weights(save_fname)

    for i, img_name in enumerate(img_fnames):  # 依次传入模型
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        filtered_boxes = yolo_v3.detect(img, COCO_ANCHORS, (img_size, img_size))
        yolo_v3.draw_boxes(i, filtered_boxes, img_name, LABELS, (img_size, img_size))
    pass

def load_pre_model():

    yolo_v3 = YoloV3Net(n_classes=len(LABELS))
    yolo_v3.build((1, 416, 416, 3))
    totolCount = 0

    for v in yolo_v3.variables:
        print(v.name + "_____" + str(v.shape))
        totolCount += np.prod(v.shape)
        # print(v)
        pass
    print(f'totolCount:{totolCount}')
    yolo_v3.load_pre_trained(weights_fname)

    IMAGE_FOLDER = os.path.join(PROJECT_ROOT, "coco_test", "*.jpg")
    img_fnames = glob.glob(IMAGE_FOLDER)

    for i, img_name in enumerate(img_fnames):  # 依次传入模型
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        filtered_boxes = yolo_v3.detect(img, COCO_ANCHORS, (img_size, img_size))
        yolo_v3.draw_boxes(i, filtered_boxes, img_name, LABELS,  (img_size, img_size))
    pass

if __name__ =="__main__":
    #main()
    detect()
    #load_pre_model()
    pass