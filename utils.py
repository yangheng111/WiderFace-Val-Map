import os
import cv2

#写入 wider 标签格式txt文件
def writeResToTxt(args,boxes,probs,imgPath):
    '''write result into txt.
    Args:
        args: Custom parameter.
        boxes: Model prediction box results，type: numpy.
        probs: Model prediction box‘s confidence results，type: numpy.

    Returns:
        Generate the file.
    '''
    save_folder = args.save_folder
    widerface_root = args.widerface_root
    save_txt = imgPath.replace(widerface_root,save_folder).replace('.jpg','.txt')
    save_dir = save_txt.replace(os.path.basename(save_txt),'')

    if not os.path.exists (save_dir):
        os.makedirs(save_dir)

    f = open(save_txt,'w')
    f.write(os.path.basename(imgPath)+'\n')
    f.write(str(len(boxes)) + '\n')

    for i in range(len(boxes)):
        box = boxes[i, :]
        label = probs[i]
        line = str(box[0])+' '+str(box[1]) +' '+str(box[2]-box[0])+' '+str(box[3]-box[1])+' ' +str(label)+' ' +'\n'
        f.write(line)

def readTestTxt(txtPath):
    '''get names.
    Args:
        txtPath: txt path.

    Returns:
        the all need names.
    '''
    f = open(txtPath)
    flines = f.readlines()
    names = [] 
    for line in flines:
        names.append(line.strip())
    return names