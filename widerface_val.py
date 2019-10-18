import argparse
import sys
import os
import cv2
import time
from utils import writeResToTxt,readTestTxt

#**********************************自定义参数变量*****************************************
parser = argparse.ArgumentParser(description='detect_video')
parser.add_argument('--net_type', default="mb_tiny_RFB_fd", type=str,help='The network architecture ,optional:1. mb_tiny_RFB_fd (higher precision) or 2.mb_tiny_fd (faster)')
parser.add_argument('--input_size', default=320, type=int,help='define network input size,default optional value 128/160/320/480/640/1280')
# parser.add_argument('--trained_model', default='./weights/WIDERFace_DSFD_RES152.pth',type=str, help='Trained state_dict file path to open')
parser.add_argument('--threshold', default=0.6, type=float,help='score threshold')
parser.add_argument('--candidate_size', default=1000, type=int,help='nms candidate size')
parser.add_argument('--device', default="cuda:0", type=str,help='cuda:0 or cpu')
parser.add_argument('--label_path', default='../models/voc-model-labels.txt', type=str,help='label path')
parser.add_argument('--save_folder', default='./WIDERFace_mb_tiny_RFB_fd_320/', type=str,help='Dir to save results')
parser.add_argument('--widerface_root', default='/home/datalab/ex_disk/work/shengdan/dataset/Face_Dataset/detect/WIDER/WIDER_val/images/', help='Location of WIDERFACE root directory')
args = parser.parse_args()

#**********************************自定义模型加载*****************************************
from ../vision.ssd.config.fd_config import define_img_size
input_img_size = args.input_size
define_img_size(input_img_size)  # must put define_img_size() before 'import create_mb_tiny_fd, create_mb_tiny_fd_predictor'
from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
def loadUltraModel(args):
    label_path = args.label_path
    class_names = [name.strip() for name in open(label_path).readlines()]
    num_classes = len(class_names)

    net_type = args.net_type
    device = args.device
    candidate_size = args.candidate_size
    threshold = args.threshold

    if net_type == 'mb_tiny_fd':
        model_path = "../models/pretrained/Mb_Tiny_FD_train_input_320.pth"
        net = create_mb_tiny_fd(len(class_names), is_test=True, device=device)
        predictor = create_mb_tiny_fd_predictor(net, candidate_size=candidate_size, device=device)
    elif net_type == 'mb_tiny_RFB_fd':
        model_path = "../models/pretrained/Mb_Tiny_RFB_FD_train_input_320.pth"
        # model_path = "models/pretrained/Mb_Tiny_RFB_FD_train_input_640.pth"
        net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=device)
        predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=candidate_size, device=device)
    else:
        print("The net type is wrong!")
        sys.exit(1)
    
    net.load(model_path)

    return predictor

#**********************************自定义模型预测*****************************************
def ultraDetectFace(args,orig_image,predictor):
    if orig_image is None:
        print("input image error!")
        return

    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

    start = time.time()
    boxes, labels, probs = predictor.predict(image, args.candidate_size / 2, args.threshold)
    interval = time.time()-start
    print('Time: {:.6f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
    return boxes.numpy(),probs.numpy()

#**********************************自定义主函数*****************************************
if __name__ == "__main__":
    accept = readTestTxt('./data/wider_face_add_lm_10_10/ImageSets/Main/test.txt')

    predictor = loadUltraModel(args)
    num = 0
    widerface_root = args.widerface_root
    for root, dirs, files in os.walk(widerface_root, topdown=False):
        for name in files:
            imgPath = os.path.join(root,name)
            if imgPath.replace(widerface_root,'').replace('/','_').replace('.jpg','') in accept:
                orig_image = cv2.imread(imgPath)

                boxes,probs = ultraDetectFace(args,orig_image,predictor)
                writeResToTxt(args,boxes,probs,imgPath)
                num +=1
                print('Dealed :',num)

