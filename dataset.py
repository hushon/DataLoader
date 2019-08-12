import os
from glob import glob

import numpy as np

import utils
import data_getter

def scan_subdirectory(top_dir, exts):
    '''
    Scan sub-directories for files with matching extensions.

    Args:
        top_dir (str): top-level directory to search subdirectory.
        exts (list): a list containing file extensions.

    Returns:
        A dict object containing subfolder name as keys, 
        and list of file paths as values.

    Example usage: 
        Example directory tree: 
            .
            ├── my_dataset
                ├── classA
                |   ├── 0001.png
                |   ├── 0002.png
                |   └── ....
                └── classB
                    ├── 0001.png
                    ├── 0002.png
                    └── ...
        
        Then the function returns below dictionary.

        >>> scan_folder('./my_dataset', exts=['.png']    

        {
            'classA': ['./my_dataset/classA/0001.png', './my_dataset/classA/0002.png', ...], 
            'classB': ['./my_dataset/classB/0001.png', './my_dataset/classB/0002.png', ...]
        }
    '''
    def find_files(directory, exts):
        '''find files having exts in the directory'''
        return [path for path in sorted(glob(os.path.join(directory, '*'))) if os.path.splitext(path)[1] in exts]

    subfolders = [folder for folder in sorted(glob(os.path.join(top_dir, '*'))) if os.path.isdir(folder)]

    return {os.path.basename(folder): find_files(folder, exts) for folder in subfolders}

class ImageDataset:
    '''
    Dataset class provides methods that read, preprocess, and augment dataset.
    Example usage:
        trainset = Dataset().read_images_and_preprocess()
        trainset.random_fliplr().random_contrast().shuffle()
        traindata = trainset.dataset # returns dataset as is
        traindata = trainset.batch(N) # returns batched dataset

        testset = Dataset().read_images()
        testdata = testset.dataset
        testdata = testset.batch(N)
    '''
    def __init__(self):
        self._dataset = {}
        self._file_exts = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']

    @property
    def dataset(self):
        return self._dataset

    @property
    def classnames(self):
        return self._dataset.keys()

    @property
    def filenames(self):
        return self._filenames

    def from_directory(self, data_dir):
        '''
        Scan data_dir and get list of files contained in its sub-directories.
        Images are classified into the names of sub-directory.
        
        Args:
            data_dir (str): directory to look for sub-directories. each sub-directory will contain image files.
            shape (tuple): a tuple of 2 integers, (width, height) of images.
        '''
        dataset = self._dataset
        file_exts = self._file_exts

        dataset = scan_subdirectory(data_dir, file_exts)

        self._dataset = dataset

        self._filenames = {key: [os.path.basename(x) for x in dataset[key]] for key in dataset}

        self.map(utils.load_img)

        return self

    def map(self, func):
        '''
        apply function to all items in dataset dictionary
        '''
        dataset = self._dataset

        dataset = {key: [list(map(func, dataset[key])) for key in dataset]}

        self._dataset = dataset
        return self

    def read_trainset(self, data_dir, resize):
        '''
        Load and preprocess images from directory.
        Args: 
            data_dir (str): Directory of dataset. The directory must contain folders named 'A/' and 'B/'
            resize (tuple of ints): The images are resized by this parameter. e.g., resize=(512, 512)
        '''
        def preprocess_img(img, resize):
            img = utils.resize_img(img, size=resize)
            img = np.array(img)
            img = img[..., np.newaxis] if img.ndim == 2 else img
            assert img.ndim == 3
            return img

        A_path = os.path.join(data_dir, 'A', '*.png')
        B_path = os.path.join(data_dir, 'B', '*.png')

        A_files = sorted(glob(A_path))
        B_files = sorted(glob(B_path))

        assert len(A_files) > 0, 'directory cannot be empty: {}'.format(A_path)
        assert len(B_files) > 0, 'directory cannot be empty: {}'.format(B_path)

        A_names = [os.path.splitext(os.path.basename(file))[0] for file in A_files]
        B_names = [os.path.splitext(os.path.basename(file))[0] for file in B_files]

        assert A_names == B_names, 'filenames must match'

        A_imgs = [preprocess_img(utils.load_img(file), resize=resize) for file in A_files]
        B_imgs = [preprocess_img(utils.load_img(file), resize=resize) for file in B_files]

        assert [np.array(x).shape for x in A_imgs] == [np.array(x).shape for x in B_imgs], 'image shapes must match'
        
        dataset = {'name': A_names, 'feature': A_imgs, 'label': B_imgs}
        print(" [*] Loaded %d images from %s." % (len(dataset['name']), data_dir))

        self._dataset.update(dataset)
        return self

    def read_testset(self, data_dir, resize):

        def preprocess_img(img, resize):
            img = utils.resize_img(img, size=resize)
            img = np.array(img)
            img = img[..., np.newaxis] if img.ndim == 2 else img
            assert img.ndim == 3
            return img

        data_dir = os.path.join(data_dir, '*.png')

        files = sorted(glob(data_dir))

        assert len(files) > 0, 'directory cannot be empty: {}'.format(files)

        names = [os.path.splitext(os.path.basename(file))[0] for file in files]
        imgs = [preprocess_img(utils.load_img(file), resize=resize) for file in files]

        dataset = {'name': names, 'feature': imgs, 'label': imgs}
        print(" [*] Loaded %d images from %s." % (len(dataset['name']), data_dir))

        self._dataset.update(dataset)
        return self

    def shuffle(self):
        '''shuffle order of dataset'''
        dataset = self._dataset

        names = dataset['name']
        features = dataset['feature']
        labels = dataset['label']
        names, features, labels = utils.shuffle_lists(names, features, labels)

        self._dataset.update({'name': names, 'feature': features, 'label': labels})
        return self

    def map(self, func, key):
        '''
        apply function to data pointed by key
        '''
        dataset = self._dataset
        
        value = map(func, dataset[key])
        
        self._dataset.update({key: value})
        return self

    def random_fliplr(self, p=0.5):
        '''
        apply horizontal flip to features and labels with probability p
        '''
        assert 0<=p<=1, 'p must be a value between 0 and 1'

        dataset = self._dataset

        features = []
        labels = []
        for feature, label in zip(dataset['feature'], dataset['label']):
            if np.random.random() < p:
                feature = np.fliplr(feature)
                label = np.fliplr(label)
            features.append(feature)
            labels.append(label)

        self._dataset.update({'feature': features, 'label': labels})
        return self

    def random_contrast(self, lower=0.9, upper=1.1):
        '''
        apply contrast perturbations to features.
        x <= (x - mean)*factor + mean
        Args:
            lower (float): lower limit of random factor
            upper (float): upper limit of random factor
        '''
        def adjust_contrast(x, factor): 
            m = np.mean(x)
            return (x-m)*factor+m

        def apply_random(x, lower, upper): 
            factor = np.random.uniform(lower, upper)
            return adjust_contrast(x, factor)

        assert upper>=lower and lower>=0, 'upper must be greater or equal than lower, and lower must be non-negative.'

        dataset = self._dataset

        features = dataset['feature']
        features = [apply_random(x, lower, upper) for x in features]

        self._dataset.update({'feature': features})
        return self

    def batch(self, n, drop_remainder=False):
        '''returns batched dataset
        Args:
            n (int): number of items in a chunk
        Returns:
            a dict object containing batches.
            e.g., when trainset.dataset returns below
            {'feature': [192, 364, 339, 187, 244, 419], 'label': [1, 2, 3, 4, 5, 6]}
            trainset.batch(2) will return below
            {'feature': [[192, 364], [339, 187], [244, 419]], 'label': [[1, 2], [3, 4], [5, 6]]}'''
        dataset = self._dataset
        
        batch = {}
        for key in dataset:
            batch.update({key: list(utils.ichunks(dataset[key], n, drop_remainder))})

        return batch