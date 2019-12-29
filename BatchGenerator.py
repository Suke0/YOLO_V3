#-- coding: utf-8 --
import numpy as np
import cv2
import os
from xml.etree.ElementTree import parse
from random import shuffle

DOWNSAMPLE_RATIO = 32


class BatchGenerator():
    def __init__(self,ann_fnames,img_dir,label_names,batch_size,anchors,net_size=416,jitter=True,shuffle=True):
        self.ann_fnames= ann_fnames
        self.img_dir = img_dir
        self.batch_size = batch_size
        # anchors为[0,0,w,h]，共9个，前三个在13*13的featuremap上预测，中间三个在26*26上预测，后三个在52*52上预测
        self.anchors = create_anchor_boxes(anchors)
        self.jitter = jitter
        self.shuffle = shuffle
        self.label_names = label_names
        self.net_size = net_size
        self.steps_per_epoch = int(len(ann_fnames)/batch_size)
        self.epoch = 0
        self.end_epoch = False
        self.index = 0
        pass

    def next_batch(self):
        xs,ys_1,ys_2,ys_3 = [],[],[],[]
        for _ in range(self.batch_size):
            x,y1,y2,y3 = self._get()
            xs.append(x)
            ys_1.append(y1)
            ys_2.append(y2)
            ys_3.append(y3)
            pass
        if self.end_epoch == True:
            if self.shuffle:
                shuffle(self.ann_fnames)
            pass
            self.end_epoch = False
            self.epoch += 1
            pass
        return np.array(xs).astype(np.float32),np.array(ys_1).astype(np.float32),np.array(ys_2).astype(np.float32),np.array(ys_3).astype(np.float32)
        pass

    def _get(self):
        net_size = self.net_size
        #解析标注文件
        fname, boxes, coded_labels = parse_annotation(self.ann_fnames[self.index],self.img_dir,self.label_names)
        #读取图片，并按照设置修改图片尺寸

        image = cv2.imread(fname) #返回（高度，宽度，通道数）的元组
        boxes_ = np.copy(boxes)
        if self.jitter:#是否要增强数据
            #image, boxes_ = make_jitter_on_image(image,boxes_)
            pass
        image, boxes_ = resize_image(image,boxes_,net_size,net_size) #对原始图片进行缩放，并且重新计算True box的位置坐标
        #boxes_为[x1,y1,x2,y2]
        #生成3种尺度的格子
        #list_ys[shape为(13,13,3,45)的array,shape为(26,26,3,45)的array,shape为(52,52,3,45)的array]
        list_ys = create_empty_xy(net_size,len(self.label_names))
        for original_box, label in zip(boxes_,coded_labels):#original_box为[x1,y1,x2,y2]在416*416图片上的坐标
            #在anchors中，找到与其面积区域最匹配的候选框max_anchor，对应的尺度索引，该尺度下的第几个锚点
            max_anchor,scale_index,box_index = find_match_anchor(original_box,self.anchors)#anchors为[x,y,w,h]
            #计算在对应尺度上的中心的坐标和对应候选框的长宽缩放比例
            #通过scale_index找到预测original_box真实框的featuremap大小
            coded_box = encode_box(list_ys[scale_index],original_box,max_anchor,net_size,net_size)
            #coded_box=[featuremap上的x，featuremap上的y，t_w,t_h]
            assign_box(list_ys[scale_index],box_index,coded_box,label)
            pass
        self.index += 1
        if self.index == len(self.ann_fnames):
            self.index = 0
            self.end_epoch = True
            pass
        return image/255.,list_ys[2],list_ys[1],list_ys[0]
        pass



    pass


class PascalVocXmlParser(object):
    def __init__(self):
        pass

    def get_fname(self, annotation_file):
        root = self._root_tag(annotation_file)
        return root.find("filename").text

    def get_width(self, annotation_file):
        tree = self._tree(annotation_file)
        for elem in tree.iter():
            if 'width' in elem.tag:
                return int(elem.text)

    def get_height(self, annotation_file):
        tree = self._tree(annotation_file)
        for elem in tree.iter():
            if 'height' in elem.tag:
                return int(elem.text)

    def get_labels(self, annotation_file):
        root = self._root_tag(annotation_file)
        labels = []
        obj_tags = root.findall("object")
        for t in obj_tags:
            labels.append(t.find("name").text)
        return labels

    def get_boxes(self, annotation_file):
        root = self._root_tag(annotation_file)
        bbs = []
        obj_tags = root.findall("object")
        for t in obj_tags:
            box_tag = t.find("bndbox")
            x1 = box_tag.find("xmin").text
            y1 = box_tag.find("ymin").text
            x2 = box_tag.find("xmax").text
            y2 = box_tag.find("ymax").text
            box = np.array([int(float(x1)), int(float(y1)), int(float(x2)), int(float(y2))])
            bbs.append(box)
        bbs = np.array(bbs)
        return bbs

    def _root_tag(self, fname):
        tree = parse(fname)
        root = tree.getroot()
        return root

    def _tree(self, fname):
        tree = parse(fname)
        return tree


class Annotation(object):
    def __init__(self, filename):
        self.fname = filename
        self.labels = []
        self.coded_labels = []
        self.boxes = None
        pass

    def add_object(self, x1, y1, x2, y2, name, code):
        self.labels.append(name)
        self.coded_labels.append(code)

        if self.boxes is None:
            self.boxes = np.array([x1, y1, x2, y2]).reshape(-1, 4)
        else:
            box = np.array([x1, y1, x2, y2]).reshape(-1, 4)
            self.boxes = np.concatenate([self.boxes, box])
        pass

    pass

def create_anchor_boxes(anchors):
    boxes = []
    n_boxes = int(len(anchors)/2)
    for i in range(n_boxes):
        boxes.append(np.array([0,0,anchors[2*i],anchors[2*i+1]]))
        pass
    return np.array(boxes)
    pass


def parse_annotation(ann_fname, img_dir, labels_naming=[]):
    parser = PascalVocXmlParser()
    fname = parser.get_fname(ann_fname)

    annotation = Annotation(os.path.join(img_dir, fname))

    labels = parser.get_labels(ann_fname)
    boxes = parser.get_boxes(ann_fname)

    for label, box in zip(labels, boxes):
        x1, y1, x2, y2 = box
        if label in labels_naming:
            annotation.add_object(x1, y1, x2, y2, name=label, code=labels_naming.index(label))
    return annotation.fname, annotation.boxes, annotation.coded_labels

def resize_image(image,boxes,img_w,img_h):#对原始图片进行缩放，并且重新计算True box的位置坐标
    h,w,_ = image.shape #原图片的真实高宽
    # resize the image to standard size
    image = cv2.resize(image,(img_h,img_w))#原图片缩放后的高宽
    image = image[:,:,::-1] #cv2把图片读取后是把图片读成BGR形式的,img[：，：，：：-1]的作用就是实现BGR到RGB通道的转换
    # fix object's position and size
    new_boxes = []
    for box in boxes:
        x1,y1,x2,y2 = box
        x1 = int(x1 * float(img_w) / w)
        x1 = max(min(x1,img_w),0)
        x2 = int(x2 * float(img_w) / w)
        x2 = max(min(x2,img_w),0)

        y1 = int(y1 * float(img_h) / h)
        y1 = max(min(y1, img_h), 0)
        y2 = int(y2 * float(img_h) / h)
        y2 = max(min(y2, img_h), 0)
        new_boxes.append([x1,y1,x2,y2])
    return image, np.array(new_boxes)
    pass

#初始化标签
def create_empty_xy(net_size,n_classes,n_boxes=3):
    #获得最小矩阵格子
    base_grid_h, base_grid_w = net_size//DOWNSAMPLE_RATIO, net_size//DOWNSAMPLE_RATIO
    #初始化三种不同尺寸的矩阵，用于存放标签
    ys_1 = np.zeros((1 * base_grid_h, 1 * base_grid_w, n_boxes, 4 + 1 + n_classes))
    ys_2 = np.zeros((2 * base_grid_h, 2 * base_grid_w, n_boxes, 4 + 1 + n_classes))
    ys_3 = np.zeros((4 * base_grid_h, 4 * base_grid_w, n_boxes, 4 + 1 + n_classes))
    list_ys = [ys_3,ys_2,ys_1]
    return list_ys
    pass

#找到与物体尺寸最接近的候选框
def find_match_anchor(box, anchor_boxes):
    x1,y1,x2,y2 = box
    box_ = np.array([0,0,x2-x1,y2-y1])
    max_index = find_match_box(box_,anchor_boxes)#box_,anchor_boxes均为[x,y,w,h]
    max_anchor = anchor_boxes[max_index]
    scale_index = max_index // 3
    box_index = max_index % 3
    return max_anchor,scale_index,box_index
    pass



def encode_box(yolo_res,original_box,anchor_box,net_w,net_h):
    x1,y1,x2,y2 = original_box #original_box为[x1,y1,x2,y2]在416*416图片上的坐标
    _,_,anchor_w,anchor_h = anchor_box
    #取出格子在高和宽方向上的个数
    grid_h, grid_w = yolo_res.shape[:2]

    # 根据输入图片到当前矩阵的缩放比例，计算当前矩阵中，物体的中心点坐标
    center_x = 0.5 * (x1+x2)
    center_x = center_x / float(net_w) * grid_w
    center_y = 0.5 * (y1+y2)
    center_y = center_y / float(net_h) * grid_h

    # 计算物体相对于候选框的尺寸缩放值
    w = np.log(max((x2-x1),1) / float(anchor_w)) #t_w
    h = np.log(max((y2-y1),1) / float(anchor_h)) #t_h
    box = [center_x,center_y ,w ,h]#将中心点和缩放值打包返回
    return box
    pass

#将具体的值放到标签矩阵里。作为真正的标签
def assign_box(yolo_res, box_index, box, label):
    center_x, center_y, _, _ = box
    #向下取整，得到的就是格子的索引
    grid_x = int(np.floor(center_x))
    grid_y = int(np.floor(center_y))

    #填入所计算的数值，作为标签
    #yolo_res.shape=(13, 13, 3, 45)
    yolo_res[grid_y,grid_x,box_index] = 0.
    yolo_res[grid_y,grid_x,box_index,0:4] = box
    yolo_res[grid_y,grid_x,box_index,4] = 1.
    yolo_res[grid_y,grid_x,box_index,5 + label] = 1.
    pass


def iou_fun(box1, box2):

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


def find_match_box(centroid_box, centroid_boxes):#找到与图片中标注的真实框，iou最大的anchor框
    match_index = -1
    max_iou = -1
    #box=[0,0,w,h]
    for i, box in enumerate(centroid_boxes):
        iou = iou_fun(centroid_box, box)

        if max_iou < iou:
            match_index = i
            max_iou = iou
    return match_index
