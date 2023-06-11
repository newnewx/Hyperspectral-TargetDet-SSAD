import numpy as np
import torch
import torch.utils.data as data
import os
from PIL import Image
import random
import imgaug.augmenters as iaa
# import data_loader.edge_utils as edge_utils

def is_image_file(filename): 
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def trans_to_tensor(pic): 
    if isinstance(pic, np.ndarray):
        img = torch.from_numpy(pic.transpose((2, 0, 1)))  # transpose and reshape is defferent
        return img.float().div(255)
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode) 
    img = img.view(pic.size[1], pic.size[0], nchannel)
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img

        #img = np.zeros(imgt.shape)
        # img_new[img_trans == 14] = 0
        #img[imgt == 103] = 255

def data_augment(img1, img2, flip=1, ROTATE_90=1, ROTATE_180=1, ROTATE_270=1, add_noise=1,medianblur=1):
    n = flip + ROTATE_90 + ROTATE_180 + ROTATE_270 + add_noise+medianblur
    a = random.random()
    if flip == 1:
        img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
        img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
    if ROTATE_90 == 1:
        img1 = img1.transpose(Image.ROTATE_90)
        img2 = img2.transpose(Image.ROTATE_90)
    if ROTATE_180 == 1:
        img1 = img1.transpose(Image.ROTATE_180)
        img2 = img2.transpose(Image.ROTATE_180)
    if ROTATE_270 == 1:
        img1 = img1.transpose(Image.ROTATE_270)
        img2 = img2.transpose(Image.ROTATE_270)
    if add_noise == 1:
        img1 = np.array(img1,dtype=np.uint8)
        aug = iaa.AdditiveGaussianNoise(scale=(0.01, 0.1*255))
        img1 = aug.augment_image(img1)
        img1 = Image.fromarray(img1.astype(np.uint8))
        img2 = img2
    if medianblur == 1:
        img1 = np.array(img1,dtype=np.uint8)
        aug = iaa.MedianBlur(k=(3,5))
        img1 = aug.augment_image(img1)
        img1 = Image.fromarray(img1.astype(np.uint8))
        img2 = img2
    return img1,img2
#inherit the class Dataset
#and implement 2 magic functions: __getitem__ and __len__ for python
#obj.__getitem__(index) == obj[index]
#len(obj)=obj.__len__ 
class train_dataset(data.Dataset):
    def __init__(self, data_path='', size_w=128, size_h=128, flip=1):
        super(train_dataset, self).__init__()
        self.list = [x for x in os.listdir(data_path + '/imgs/') if is_image_file(x)]
        self.data_path = data_path
        self.size_w = size_w
        self.size_h = size_h
        self.flip = flip
        
    def __getitem__(self, index):
        initial_path = os.path.join(self.data_path + '/imgs/', self.list[index])
        #semantic_path = os.path.join(self.data_path + '/masks/', self.list[index][:-4]+'_instance_color_RGB'+self.list[index][-4:])
        semantic_path = os.path.join(self.data_path + '/masks/', self.list[index])
        assert os.path.exists(semantic_path)
        try:
            initial_image = Image.open(initial_path).convert('RGB')
            # semantic_image = Image.open(semantic_path).point(lambda i: i * 80).convert('RGB')
            semantic_image = Image.open(semantic_path)                                             # 修改4.22

        except OSError:
            return None, None, None

        # label_n = np.array(semantic_image)
        # label = np.zeros(label_n.shape) 
        # label[label_n==103] = 255

        # semantic_image = Image.fromarray(semantic_image)

        initial_image = initial_image.resize((self.size_w, self.size_h), Image.BILINEAR)           # (1024, 1024)
        semantic_image = semantic_image.resize((self.size_w, self.size_h), Image.BILINEAR)

        # op_num = 6
        # index = np.random.randint(0,2,op_num)
        # if index.all() == 0:
        #     t = np.random.randint(0,6)
        #     index[t] = 1
        # flip, ROTATE_90, ROTATE_180, ROTATE_270, add_noise,medianblur = index
        initial_image,semantic_image=data_augment(initial_image,semantic_image)#,flip=flip, ROTATE_90=ROTATE_90, ROTATE_180=ROTATE_180, ROTATE_270=ROTATE_270, add_noise=add_noise,medianblur=medianblur)  #4.18添加

        label_n_2 = np.array(semantic_image)
        thr = (label_n_2.max() + label_n_2.min())/2
        label_resize = np.uint8(label_n_2 > thr)*255
        semantic_image = Image.fromarray(label_resize)

        ### normalize to [0,1] ###
        #semantic_image = semantic_image+1/2
        #semantic_image = np.float(semantic_image)
        #semantic_image = (semantic_image-semantic_image.min())/(semantic_image.max()-semantic_image.min())

        if self.flip == 1:
            a = random.random()
            if a < 1 / 3:
                initial_image = initial_image.transpose(Image.FLIP_LEFT_RIGHT)
                semantic_image = semantic_image.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                if a < 2 / 3:
                    initial_image = initial_image.transpose(Image.ROTATE_90)
                    semantic_image = semantic_image.transpose(Image.ROTATE_90)

        initial_image = trans_to_tensor(initial_image)
        initial_image = initial_image.mul_(2).add_(-1)  # -1 - 1
        semantic_image = trans_to_tensor(semantic_image)
        # semantic_image = semantic_image.mul_(2).add_(-1)                      # 修改4.22
        
        # _edgemap = semantic_image.numpy()                                  #shape修改
        # _edgemap = edge_utils.mask_to_onehot(_edgemap, 1)

        # _edgemap = edge_utils.onehot_to_binary_edges(_edgemap, 1, 1)  # ((_edgemap, 2, 1))

        # edgemap = torch.from_numpy(_edgemap).float()

        # return initial_image, semantic_image, edgemap, self.list[index]
        return initial_image, semantic_image, self.list[index]

    def __len__(self):
        return len(self.list)

if __name__ == '__main__':
    train_datatset_ = train_dataset('./data/train', 128, 128, 0)
    print('len_train_dataset:%d' % len(train_datatset_))
    # print('len_train_dataset:{}'.format(len(train_datatset_)))

    initial_image, semantic_image, name = train_datatset_[0]#.__getitem__(index=100)
    train_loader = torch.utils.data.DataLoader(dataset=train_datatset_, batch_size=8, shuffle=True,
                                               num_workers=0)
    train_batch_loader = iter(train_loader)
    img, masks, num = next(train_batch_loader)
    print(name)