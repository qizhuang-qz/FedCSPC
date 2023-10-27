import torch.utils.data as data
from PIL import Image
import numpy as np
import torchvision
from torchvision.datasets import MNIST, EMNIST, CIFAR10, CIFAR100, SVHN, FashionMNIST, ImageFolder, DatasetFolder, utils
from torch.utils.data import Dataset
import os
import os.path
import logging
import sys
import torch
import io
import scipy.io as matio
import ipdb

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def default_loader(image_path):
    return Image.open(image_path).convert('RGB')    
    
    
class MNIST_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download


        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        mnist_dataobj = MNIST(self.root, self.train, self.transform, self.target_transform, self.download)

        if torchvision.__version__ == '0.2.1':
            if self.train:
                data, target = mnist_dataobj.train_data, np.array(mnist_dataobj.train_labels)
            else:
                data, target = mnist_dataobj.test_data, np.array(mnist_dataobj.test_labels)
        else:
            data = mnist_dataobj.data
            target = np.array(mnist_dataobj.targets)


        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0
                   
            

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        img = img.reshape(28,28,1).cpu().numpy()


        if self.transform is not None:
            img = self.transform(img)


        return img, target

    def __len__(self):
        return len(self.data)




class FashionMNIST_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        fmnist_dataobj = FashionMNIST(self.root, self.train, self.transform, self.target_transform, self.download)

        if torchvision.__version__ == '0.2.1':
            if self.train:
                data, target = fmnist_dataobj.train_data, np.array(fmnist_dataobj.train_labels)
            else:
                data, target = fmnist_dataobj.test_data, np.array(fmnist_dataobj.test_labels)
        else:
            data = fmnist_dataobj.data
            target = np.array(fmnist_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class CIFAR10_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)

        if torchvision.__version__ == '0.2.1':
            if self.train:
                data, target = cifar_dataobj.train_data, np.array(cifar_dataobj.train_labels)
            else:
                data, target = cifar_dataobj.test_data, np.array(cifar_dataobj.test_labels)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)
#         ipdb.set_trace()
        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]
        # img = Image.fromarray(img)
        # print("cifar10 img:", img)
        # print("cifar10 target:", target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class CIFAR100_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        cifar_dataobj = CIFAR100(self.root, self.train, self.transform, self.target_transform, self.download)

        if torchvision.__version__ == '0.2.1':
            if self.train:
                data, target = cifar_dataobj.train_data, np.array(cifar_dataobj.train_labels)
            else:
                data, target = cifar_dataobj.test_data, np.array(cifar_dataobj.test_labels)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]
        img = Image.fromarray(img)
        # print("cifar10 img:", img)
        # print("cifar10 target:", target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

class TinyImageNet_load(Dataset):
    def __init__(self, root, dataidxs=None, train=True, transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")
        self.dataidxs = dataidxs
        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        
        if self.dataidxs is not None:
            self.samples = self.images[dataidxs]
        else:
            self.samples = self.images

#         print('samples.shape', self.samples.shape)
        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images;

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
#         print(self.tgt_idx_to_class)
    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(val_image_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item)
        self.images = np.array(self.images)
#         print('dataset.shape', self.images.shape)
    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]
    

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        img_path, tgt = self.samples[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        tgt = int(tgt)
        return sample, tgt     


class ImageFolder_custom(DatasetFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples)

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)

        
class Food101_truncated(torch.utils.data.Dataset):
    def __init__(self, dataidxs=None, transform=None, loader=default_loader, mode = None):

        image_path = '/Food101_Image/images/'
        data_path = '/Food101_Text/'
        if mode == 'train':
            with io.open(data_path + 'train_images.txt', encoding='utf-8') as file:
                path_to_images = file.read().split('\n')[:-1] #list-len:68175
            with io.open(data_path + 'train_labels.txt', encoding='utf-8') as file:
                labels = file.read().split('\n')[:-1] #list-len:68175

        elif mode == 'test':

            with io.open(data_path + 'test_images.txt', encoding='utf-8') as file:
                path_to_images = file.read().split('\n')[:-1] #list-len:25250
            with io.open(data_path + 'test_labels.txt', encoding='utf-8') as file:
                labels = file.read().split('\n')[:-1] #list-len:25250

        elif mode == 'val':
            with io.open(data_path + 'val_images.txt', encoding='utf-8') as file:
                path_to_images = file.read().split('\n')[:-1]
            with io.open(data_path + 'val_labels.txt', encoding='utf-8') as file:
                labels = file.read().split('\n')[:-1]

        else:
            assert 1<0, 'Please fill mode with any of train/val/test to facilitate dataset creation'

        #import ipdb; ipdb.set_trace()
        if mode == 'train' and dataidxs != None:
#             print('xxxxxxxxx', path_to_images)
            self.image_path = image_path
#             ipdb.set_trace()
            self.path_to_images = np.array(path_to_images)[dataidxs]
            self.labels = np.array(labels, dtype=int)[dataidxs]
            print('mode:', mode, 'len(path_to_images):', len(self.path_to_images))
            
        else:
            self.image_path = image_path
            self.path_to_images = path_to_images
            self.labels = np.array(labels, dtype=int)
            
        self.transform = transform
        self.loader = loader
        self.mode = mode
        

    def __getitem__(self, index):
        # get image matrix and transform to tensor
        path = self.path_to_images[index]
        img = self.loader(self.image_path + path + '.jpg')

        if self.transform is not None:
            img = self.transform(img)
        
        # get label
        label = self.labels[index]
            
        return img, label


    def __len__(self):
        return len(self.path_to_images)    

    
class Vireo172_truncated(torch.utils.data.Dataset):
    def __init__(self, dataidxs=None, transform=None, loader=default_loader, mode = None):

        image_path = '/Vireo172_Image/ready_chinese_food/'
        data_path =  '/Vireo172_Text/SplitAndIngreLabel/'
        
        if mode == 'train':
            
            with io.open(data_path + 'TR.txt', encoding='utf-8') as file:
                path_to_images = file.read().split('\n')[:-1]
            labels = matio.loadmat(data_path + 'train_label.mat')['train_label'][0]

        elif mode == 'test':
            
            with io.open(data_path + 'TE.txt', encoding='utf-8') as file:
                path_to_images = file.read().split('\n')[:-1]
            labels = matio.loadmat(data_path + 'test_label.mat')['test_label'][0]

        elif mode == 'val':

            with io.open(data_path + 'VAL.txt', encoding='utf-8') as file:
                path_to_images = file.read().split('\n')[:-1]
            labels = matio.loadmat(data_path + 'val_label.mat')['validation_label'][0]

        else:
            assert 1<0, 'Please fill mode with any of train/val/test to facilitate dataset creation'

        #import ipdb; ipdb.set_trace()
      
        if mode == 'train' and dataidxs != None:
#             print('xxxxxxxxx', path_to_images)
            self.image_path = image_path
            self.path_to_images = np.array(path_to_images)[dataidxs]
            self.labels = np.array(labels, dtype=int)[dataidxs]-1
            print('mode:', mode, 'len(path_to_images):', len(self.path_to_images))
            
        else:
            self.image_path = image_path
            self.path_to_images = path_to_images
            self.labels = np.array(labels, dtype=int)-1
        

        self.transform = transform
        self.loader = loader

        
    def __getitem__(self, index):
        # get image matrix and transform to tensor
        path = self.path_to_images[index]

        img = self.loader(self.image_path + path)

        if self.transform is not None:
            img = self.transform(img)
        
        # get label
        label = self.labels[index]

        #change vireo labels from 1-indexed to 0-indexed values

            
        return [img, label]

    def __len__(self):
        return len(self.path_to_images)        
        
class Data_for_label_cluster(torch.utils.data.Dataset):
    def __init__(self, data, labels, dataset):
        self.vectors = data
        self.classIDs = labels
        self.dataset = dataset

    def __getitem__(self, index):
        # get hidden vector
        vector = self.vectors[index]
        classID = self.classIDs[index]

    #         ipdb.set_trace()
        if self.dataset == 'cifar10':
            label_indicator = np.zeros([10], dtype=np.float32)
        elif self.dataset == 'cifar100':
            label_indicator = np.zeros([100], dtype=np.float32)
        label_indicator[int(classID)] = 1
    #         print(label_indicator)
        return [vector, label_indicator, index]

    def __len__(self):
        return len(self.classIDs)       
    
    
class Data_for_Retraining(torch.utils.data.Dataset):
    def __init__(self, data, labels, clientids):
        self.vectors = data
        self.classIDs = labels
        self.clientids = clientids

    def __getitem__(self, index):
        # get hidden vector
        vector = self.vectors[index]
        classID = self.classIDs[index]
        in_client_inx = np.where(self.clientids == self.clientids[index])[0]
        posi_data_index = self.__posi_sample__(index, in_client_inx)
        nega_data_index = self.__nega_sample__(index, in_client_inx)
        
        posi_vector = self.vectors[posi_data_index]
        nega_vector = self.vectors[nega_data_index]
        
        return vector, posi_vector, nega_vector, classID
        

    def __posi_sample__(self, index, in_client_inx, k=1):   
        target = self.classIDs[index]
        posi_idx = np.where(self.classIDs == target)[0]        
        posi_inx_intersection = np.intersect1d(posi_idx, in_client_inx)
        
        if len(posi_inx_intersection) >= k:
            posi_data_index = np.random.choice(posi_inx_intersection, k)
        else:
            posi_data_index = [index]
        
        return posi_data_index

    def __nega_sample__(self, index, in_client_inx, k=1):   
        target = self.classIDs[index]
        nega_idx = np.where(self.classIDs != target)[0]      
        nega_inx_intersection = np.intersect1d(nega_idx, in_client_inx)
        
        if len(nega_inx_intersection) >= k:
            nega_data_index = np.random.choice(nega_inx_intersection, k)
        else:
            nega_data_index = [index]
        
        return nega_data_index


    def __len__(self):
        return len(self.classIDs)        

    
    
class Data_for_Retraining_final(torch.utils.data.Dataset):
    def __init__(self, data, labels, clientids):
        self.vectors = data
        self.classIDs = labels
        self.clientids = clientids

    def __getitem__(self, index):
        # get hidden vector
        vector = self.vectors[index]
        classID = self.classIDs[index]
        in_client_inx = np.where(self.clientids == self.clientids[index])[0]
        posi_data_index = self.__posi_sample__(index, in_client_inx)
        nega_data_index = self.__nega_sample__(index, in_client_inx)
        
        posi_vector = self.vectors[posi_data_index]
        nega_vector = self.vectors[nega_data_index]
        
        return torch.tensor(vector), torch.tensor(posi_vector), torch.tensor(nega_vector), torch.tensor([classID])
        

    def __posi_sample__(self, index, in_client_inx, k=1):   
        target = self.classIDs[index]
        posi_idx = np.where(self.classIDs == target)[0]    
        posi_inx_intersection = np.intersect1d(posi_idx, in_client_inx)
        
        if len(posi_idx) >= k:
            posi_data_index = np.random.choice(posi_inx_intersection, k)
        else:
            posi_data_index = [index]
        
        return posi_data_index

    def __nega_sample__(self, index, in_client_inx, k=1):   
        target = self.classIDs[index]
        nega_idx = np.where(self.classIDs != target)[0]   
        nega_inx_intersection = np.intersect1d(nega_idx, in_client_inx)
        
        if len(nega_idx) >= k:
            nega_data_index = np.random.choice(nega_inx_intersection, k)
        else:
            nega_data_index = [index]
        
        return nega_data_index


    def __len__(self):
        return len(self.classIDs)        
        
    
    
    
    
    
    
    