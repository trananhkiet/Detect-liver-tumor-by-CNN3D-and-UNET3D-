import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
TRANING_MEAN = 0.4
MIN_BOUND = -1000.0
MAX_BOUND = 400.0

def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image
def npy_generator(image_file_paths,mask_file_paths, use_normalize = False, batch_size=4, n_outputs = 1, subtract_mean = None,reshape = True, n_class = 1):
    image_file_paths.sort()
    mask_file_paths.sort()
    while(True):
        images_paths, masks_paths = shuffle(image_file_paths, mask_file_paths)
        for file_number in range(0,len(image_file_paths)):
            image_file = np.load(images_paths[file_number], mmap_mode = 'r')
            mask_file = np.load(masks_paths[file_number], mmap_mode = 'r')
            data_size = len(image_file)
            indices = np.arange(data_size)
            np.random.shuffle(indices)
            for i in range(0, data_size, batch_size):
                batch_indices = indices[i:min(i + batch_size, data_size)]
                images = image_file[batch_indices]
                masks = mask_file[batch_indices]
                # normalize and centering
                if use_normalize:
                    images = normalize(images)
                if subtract_mean!= None:
                    images -= subtract_mean
                #reshape
                if reshape:
                    images = np.reshape(images, images.shape + (1,))
                    if n_class == 1:
                        masks = np.reshape(masks, masks.shape + (1,))
                    else:
                        masks = to_categorical(masks,n_class)
                if n_outputs > 1:
                    output_mask = []
                    for n in range(0,n_outputs):
                        output_mask.append(masks)
                    yield (images, output_mask)
                else:
                    yield (images, masks)