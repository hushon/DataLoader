import cv2

def load_img(path):
    return cv2.imread(path)

def save_img(image, image_path):
    return cv2.imwrite(image_path, image)

def resize_img(img, size=(256,256)):
    return cv2.resize(img, dsize=size, interpolation=cv2.INTER_AREA)

def ichunks(x, n, drop_remainder=False):
    '''Creates n-sized chunks from a list-like object. Used to make minibatch of training dataset.
    Args:
        x (list, tuple, or any iterable type): input list-like object
        n (int): size of each chunk
        drop_remainder (bool): if True, drop the final chunk if its size is smaller then N
    Returns: 
        A generator that yields each chunk
    Example:
        >>> items = ichunks([1, 2, 3, 4, 5, 6], 2)
        >>> for x in items: print(x)
            [1, 2]
            [3, 4]
            [5, 6]
    '''
    assert n > 0, 'argument n must be greater than zero.'
    
    x = iter(x)
    while True:
        try: 
            chunk = []
            for i in range(n):
                chunk.append(next(x))
        except StopIteration:
            if drop_remainder:
                break
            else:
                if len(chunk)>0:
                    yield chunk
                    break
                else:
                    break
        else:
            yield chunk

def shuffle_lists(*lists):
    '''take lists and shuffle while sharing list order. lists should be of equal length.'''
    def shuffle(x, seed):
        random.Random(seed).shuffle(x)
        return x
    seed = random.randint(0,99)
    return tuple([shuffle(x, seed) for x in lists])

def split_dataset(x, n):
    '''
    split a list into n approximately-equal lengths of subsets.
    used to split dataset for K-fold cross-validation.
    Args:
        x: input list object
        n (int): number of chunks
    Returns:
        a list of n chunks, each with approximately equal size.
    '''
    k, m = divmod(len(x), n)
    return [x[i*k + min(i, m) : (i+1)*k + min(i+1, m)] for i in range(n)]