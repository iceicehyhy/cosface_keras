import os
import PIL.Image as Image
import numpy as np

train_list = '/home/iceicehyhy/Dataset/CASIA/CASIA_FIRST_10/pairs_train.txt'
root_dir = '/home/iceicehyhy/Dataset/CASIA/CASIA_FIRST_10'

def PIL_loader(path):
    try:
        img = Image.open(path).convert('RGB')
    except IOError:
        print('Cannot load image ' + path)
    else:
        return img


def default_reader(fileList):
    imgList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            imgPath, label = line.strip().split(' ')
            imgList.append((imgPath, int(label)))
    return imgList


def read_data_from_list():
    imgList = default_reader(train_list)
    for c_b in range (len(imgList)):
        img_p, label_ = imgList[c_b]
        img_ = np.asarray(PIL_loader(os.path.join(root_dir, img_p)))
        img_ = np.expand_dims(img_, axis= 0)
        label_ = np.expand_dims(label_, axis=0)
        if c_b == 0:
            batch_x = img_
            batch_y = label_
        else:
            batch_x = np.concatenate((batch_x, img_), axis= 0)
            batch_y = np.concatenate((batch_y, label_), axis= 0)
    
    index = np.arange(len(imgList))
    np.random.shuffle(index)
    cutoff_index = int(0.1 * len(imgList))
    train_index = index[cutoff_index:]
    val_index = index[:cutoff_index]
    return (batch_x[train_index], batch_y[train_index]), (batch_x[val_index], batch_y[val_index])