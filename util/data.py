import time
import skimage.io as io
import numpy as np
import os
import scipy.ndimage
import random
import SimpleITK as sitk
import pandas as pd
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from util.display import *
#from display import *
MIN_BOUND = -150.0
MAX_BOUND = 200.0


def normalize(image):
    '''
    normalize image to 0 -> 1
    :param image: input image
    :return: image array
    '''
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image


def resample(volume, volume_spacing, target_spacing=(1, 1, 1)):
    '''
    resample 3D volume
    :param volume: input volume
    :param volume_spacing:
    :param target_spacing:
    :return:
    '''
    resize_factor0 = np.array(volume_spacing)/np.array(target_spacing)
    resize_factorz = volume_spacing[2]/target_spacing[2]
    resize_factorx = volume_spacing[0]/target_spacing[0]
    resize_factory = volume_spacing[1]/target_spacing[1]
    new_volume = scipy.ndimage.interpolation.zoom(volume, (resize_factorz,resize_factorx,resize_factory), mode='nearest')
    new_spacingx = volume.shape[2]/new_volume.shape[2]*volume_spacing[0]
    new_spacingy = volume.shape[1]/new_volume.shape[1]*volume_spacing[1]
    new_spacingz = volume.shape[0]/new_volume.shape[0]*volume_spacing[2]
    return new_volume , (new_spacingx,new_spacingy,new_spacingz)


def random_rotate_and_flip_90_with_mask(image, mask):
    '''
    random rotate 90, 180, 270 and flip
    :param image: input image
    :param mask: input image
    :return: image, mask
    '''
    do_flip = random.uniform(0, 1)

    if do_flip > 0.5:
        rt_image = np.flip(image, axis=1)
        rt_mask = np.flip(mask, axis=1)
    else:
        rt_image = image
        rt_mask = mask
    k_rot90 = random.randint(0, 3)

    if k_rot90 != 0:
        rt_image = np.rot90(rt_image, k_rot90, axes=(1, 2))
        rt_mask = np.rot90(rt_mask, k_rot90, axes=(1, 2))

    return rt_image, rt_mask


def random_crop_with_mask(image, mask, target_size):
    '''
    random crop
    :param image: input image
    :param mask: input image
    :param target_size: output size
    :return: croped image
    '''
    if image.shape[0] > target_size[0]:
        dim_0_offset = target_size[0]
        sample_dim_0 = np.random.randint(0, image.shape[0] - target_size[0])
    else:
        dim_0_offset = image.shape[0]
        sample_dim_0 = 0
    if image.shape[1] > target_size[1]:
        dim_1_offset = target_size[1]
        sample_dim_1 = np.random.randint(0, image.shape[1] - target_size[1])
    else:
        dim_1_offset = image.shape[1]
        sample_dim_1 = 0
    if image.shape[2] > target_size[2]:
        dim_2_offset = target_size[2]
        sample_dim_2 = np.random.randint(0, image.shape[2] - target_size[2])
    else:
        dim_2_offset = image.shape[2]
        sample_dim_2 = 0

    rt_image = image[sample_dim_0:(sample_dim_0 + dim_0_offset), sample_dim_1:(sample_dim_1 + dim_1_offset),
               sample_dim_2:(sample_dim_2 + dim_2_offset)]
    rt_mask = mask[sample_dim_0:(sample_dim_0 + dim_0_offset), sample_dim_1:(sample_dim_1 + dim_1_offset),
              sample_dim_2:(sample_dim_2 + dim_2_offset)]
    # zero padding if need
    rt_image = np.pad(rt_image, pad_width=(
        (0, target_size[0] - dim_0_offset), (0, target_size[1] - dim_1_offset), (0, target_size[2] - dim_2_offset)),
                      mode='constant', constant_values=0)
    rt_mask = np.pad(rt_mask, pad_width=(
        (0, target_size[0] - dim_0_offset), (0, target_size[1] - dim_1_offset), (0, target_size[2] - dim_2_offset)),
                     mode='constant', constant_values=0)

    return rt_image, rt_mask


def to1D(x, y, z, size):
    '''
    convert index 3D to 1D
    :param x:
    :param y:
    :param z:
    :param size: (maxX, maxY, maxZ)
    :return: 1D index
    '''
    return [z + size[2] * (y + size[1] * x)]


def to3D(idx, size):
    '''
    convert index 1D to 3D
    :param idx: 1D ndex
    :param size: (maxX, maxY, maxZ)
    :return: 3D ndex (x, y, z)
    '''
    z = idx % size[2]
    idx = idx // size[2]
    y = idx % size[1]
    x = idx // size[1]
    return x, y, z


def finding_axis_crop_size(cur_size, max_size, factor):
    n = np.int(np.ceil(cur_size/max_size))
    k = np.int(np.ceil(cur_size/(factor*n)))
    return factor * k


def finding_crop_size(current_size, max_crop_size, factor):
    crop_size_x = finding_axis_crop_size(current_size[0], max_crop_size[0], factor)
    crop_size_y = finding_axis_crop_size(current_size[1], max_crop_size[1], factor)
    crop_size_z = finding_axis_crop_size(current_size[2], max_crop_size[2], factor)
    return crop_size_x, crop_size_y, crop_size_z


def padding_scan(image, fragment_size=(192, 192, 64)):
    '''
    padding image to new min dynamic size = X * fragment_size >= image size, X is integer vector(x0, x1, x2),
    then zero padding
    :param image:  input image
    :param fragment_size:
    :return:  new size image
    '''
    factor = image.shape / np.array(fragment_size, dtype=float)
    factor = np.ceil(factor)
    factor = np.uint8(factor)
    new_size = factor * fragment_size
    offset = new_size - image.shape

    rt_image = np.pad(image, pad_width=(
        (0, offset[0]), (0, offset[1]), (0, offset[2])),
                      mode='constant', constant_values=0)
    return rt_image


def crop_scan(image, fragment_size=(192, 192, 64)):
    '''
    crop the image to list of many fragment image with target size. Must call padding_scan first
    :param image: input image
    :param fragment_size:
    :return: list (array) of fragment image, number of fragment per dimension
    '''
    factor = image.shape // np.array(fragment_size)
    total = factor[0] * factor[1] * factor[2]
    imgs = np.ndarray((total, fragment_size[0], fragment_size[1], fragment_size[2]), dtype=image.dtype)
    for i in range(0, factor[0]):
        for j in range(0, factor[1]):
            for k in range(0, factor[2]):
                img = image[i * fragment_size[0]:(i + 1) * fragment_size[0],
                      j * fragment_size[1]:(j + 1) * fragment_size[1],
                      k * fragment_size[2]:(k + 1) * fragment_size[2]]
                img_index = to1D(i, j, k, factor)
                imgs[img_index] = img
    return imgs, factor

# Function to distort image
def elastic_transform_v2(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    
def rebuild_scan(images, factor, fragment_size):
    '''
    rebuild image from many fragment and crop to original size
    :param images: input list (array) of fragment image)
    :param factor: number of fragment per dimension
    :param fragment_size:
    :return: output image array
    '''
    image = np.zeros(factor * fragment_size, dtype=images.dtype)
    for index in range(0, len(images)):
        i, j, k = to3D(index, factor)
        image[i * fragment_size[0]:(i + 1) * fragment_size[0], j * fragment_size[1]:(j + 1) * fragment_size[1],
        k * fragment_size[2]:(k + 1) * fragment_size[2]] = images[index]
    return image
def crop_to_original_size(image,original_size):
    image = image[0:original_size[0], 0:original_size[1], 0:original_size[2]]
    return image

def save_volume(volume, origin, spacing, filename):
    itkimage = sitk.GetImageFromArray(volume, isVector=False)
    itkimage.SetSpacing(spacing)
    itkimage.SetOrigin(origin)
    print('Save file: ',filename)
    sitk.WriteImage(itkimage, filename, True)

def get_box(scr_arr, min_threshold, max_threshold):
    x1 = y1 = z1 = 0;
    x2 = scr_arr.shape[0]
    y2 = scr_arr.shape[1]
    z2 = scr_arr.shape[2]
    for i in range(0,scr_arr.shape[0]):
        if (scr_arr[i,:,:]>=min_threshold).any() and (scr_arr[i,:,:]<=max_threshold).any():
            x1 = i
            break
    for i in range(0,scr_arr.shape[1]):
        if (scr_arr[:,i,:]>=min_threshold).any() and (scr_arr[:,i,:]<=max_threshold).any():
            y1 = i
            break
    for i in range(0,scr_arr.shape[2]):
        if (scr_arr[:,:,i]>=min_threshold).any() and (scr_arr[:,:,i]<=max_threshold).any():
            z1 = i
            break
    for i in range(0,scr_arr.shape[0]):
        k = scr_arr.shape[0] - i - 1
        if (scr_arr[k,:,:]>=min_threshold).any() and (scr_arr[k,:,:]<=max_threshold).any():
            x2 = k + 1
            break
    for i in range(0,scr_arr.shape[1]):
        k = scr_arr.shape[1] - i - 1
        if (scr_arr[:,k,:]>=min_threshold).any() and (scr_arr[:,k,:]<=max_threshold).any():
            y2 = k + 1
            break
    for i in range(0,scr_arr.shape[2]):
        k = scr_arr.shape[2] - i - 1
        if (scr_arr[:,:,k]>=min_threshold).any() and (scr_arr[:,:,k]<=max_threshold).any():
            z2 = k + 1
            break
    return x1, x2, y1, y2, z1, z2
    
def preprocess(image_arr, mask_arr,sample_spacing,use_normalize = True,AD_filter_Conductance = 1.0,resample_spacing = None,verbose = 0):
    image_to_process = np.array(image_arr, dtype = np.float32)
    mask_to_process = np.array(mask_arr, dtype = np.float32)
    t = time.monotonic()
    if resample_spacing != None:
        if verbose:
            print('Sample shape: ', image_to_process.shape)
            print('Sample spacing: ', sample_spacing)
        image_to_process, scan_spacing = resample(image_to_process,sample_spacing,resample_spacing)
        mask_to_process, mask_spacing = resample(mask_to_process,sample_spacing,resample_spacing)
        assert scan_spacing==mask_spacing, 'Resample error'
        sample_spacing = scan_spacing
        if verbose: 
            print('Resample :', time.monotonic() - t, ' s')
            t = time.monotonic()
            print('Sample new shape: ', image_to_process.shape)
            print('Sample new spacing: ', sample_spacing)
    #normalize
    if use_normalize:
        image_to_process = normalize(image_to_process)
        if verbose: 
            print('Normalize :', time.monotonic() - t, ' s')
            t = time.monotonic()
    #Anisotropic Diffusion Filter
    if AD_filter_Conductance > 0.:
        image_itk2 = sitk.GetImageFromArray(image_to_process, isVector=False)
        AD_filter = sitk.CurvatureAnisotropicDiffusionImageFilter()
        AD_filter.SetTimeStep(0.0625)
        AD_filter.SetNumberOfIterations(10)
        AD_filter.SetConductanceParameter(AD_filter_Conductance)
        image_itk2 = AD_filter.Execute(image_itk2)
        image_to_process = sitk.GetArrayFromImage(image_itk2)
        if verbose: 
            print('Anisotropic Diffusion Filter :', time.monotonic() - t, ' s')
            t = time.monotonic()
    mask_to_process[mask_to_process<0.5] = 0
    mask_to_process[mask_to_process>0.5] = 1
    mask_to_process = mask_to_process.astype(np.ubyte)
    return image_to_process, mask_to_process, sample_spacing

def preprocess_offline(image_paths, mask_paths,offset = 0,limit = 0,save_dir = None,use_normalize = True,AD_filter_Conductance = 1.0,resample_spacing = None,verbose = 0):
    n_samples = len(image_paths)
    assert offset < n_samples, 'offset must be smaller n_samples'
    if limit == 0 or limit>n_samples-offset:
        limit = n_samples-offset
    for i in range(0, limit):
        j = i + offset
        t = time.monotonic()
        if verbose:
            print('Sample Scan:',image_paths[j])
            print('Sample Ground Truth :',image_paths[j])
        image_itk = sitk.ReadImage(image_paths[j])
        mask_itk = sitk.ReadImage(mask_paths[j])
        sample_spacing = image_itk.GetSpacing()
        image_arr = sitk.GetArrayFromImage(image_itk)

        mask_arr = sitk.GetArrayFromImage(mask_itk)
        # set label for tumor
        mask_arr[mask_arr == 1] = 0
        mask_arr[mask_arr == 2] = 1
        if mask_arr.max() == 1 or mask_arr.max() == 0:
            print("Set label for tumor successful...")

        if verbose: 
            print('Loading scan : ', time.monotonic() - t, ' s')
            t = time.monotonic()
        image_arr , mask_arr, sample_spacing= preprocess(image_arr,mask_arr,sample_spacing,use_normalize,AD_filter_Conductance,resample_spacing,verbose)
        if verbose: 
            print('Preprocess :', time.monotonic() - t, ' s')
            t = time.monotonic()
        if save_dir!=None:
            scan_name = os.path.join(save_dir,'volume-'+str(j+1).zfill(3)+'.nii.gz')
            save_volume(image_arr,image_itk.GetOrigin(),sample_spacing,scan_name)
            gt_name = os.path.join(save_dir,'segmentation-'+str(j+1).zfill(3)+'.nii.gz')
            save_volume(mask_arr,mask_itk.GetOrigin(),sample_spacing,gt_name)
    
def generate_data(image_paths, mask_paths, save_images_npy, save_masks_npy, n_sample,target_size=(192, 192, 64), sample_per_scan=20,AD_filter_Conductance = 2.0,elastic_prob = 0.0, rotate_prob = 0.8 ,max_rotate_angle_x = 0.0,max_rotate_angle_y = 0.0, max_rotate_angle_z = 0.0,zoom_prob = 0.8,zoom_factor = 0.0, print_log = False):
    assert n_sample<=len(image_paths), 'the number of selected samples must be smaller than the number of scan avaiable'
    indices = np.arange(len(image_paths))
    np.random.shuffle(indices)
    indices = indices[0:n_sample]
    image_paths_selected = [image_paths[i] for i in indices]
    mask_paths_selected = [mask_paths[i] for i in indices]
    print('Indices: ', indices)
    print('Scans selected: ',image_paths_selected)
    print('Ground truth labels selected: ',mask_paths_selected)
    create_npy(image_paths_selected, mask_paths_selected, save_images_npy, save_masks_npy, target_size=target_size, sample_per_scan=sample_per_scan,AD_filter_Conductance = AD_filter_Conductance,elastic_prob = elastic_prob, rotate_prob = rotate_prob ,max_rotate_angle_x = max_rotate_angle_x,max_rotate_angle_y = max_rotate_angle_y, max_rotate_angle_z = max_rotate_angle_z,zoom_prob = zoom_prob,zoom_factor = zoom_factor, print_log = print_log)
    
def select_number(rate):
    # Normalize data set picking rate
    rate = np.array(rate)
    rate = rate / np.sum(rate)
    success_prob = random.uniform(0, 1)
    cur_prob = 0
    selected_number = 0
    for i in range(0,len(rate)):
        cur_prob = cur_prob+rate[i]
        if cur_prob>success_prob:
            selected_number=i
            break
    return selected_number
    


def data_augmentation(data_sets,datasets_rate,save_dir, part_number ,n_sample, sample_per_scan, crop_size=(192, 192, 64), elastic_prob = 0.0, rotate_prob = 0.8 ,max_rotate_angle_x = 0.0,max_rotate_angle_y = 0.0, max_rotate_angle_z = 0.0, print_log = False,condition_crop=0.9):
    print("generate data....6")
    assert len(data_sets)==len(datasets_rate), 'Missing data set selecting rate'
    total_time = time.monotonic()
    total = n_sample * sample_per_scan
    avg_time = 0
    scans = np.ndarray((total, crop_size[0], crop_size[1], crop_size[2]), dtype=np.float32)
    gts = np.ndarray((total, crop_size[0], crop_size[1], crop_size[2]), dtype=np.uint8)
    for i in range(0,n_sample):
        t = time.monotonic()
        # data set select
        data_set = data_sets[select_number(datasets_rate)]
        assert len(data_set)==2, 'Missing data, must have scan and ground truth in each dataset'
        scan_set = data_set[0]
        gt_set = data_set[1]
        assert len(scan_set)==len(gt_set), 'Missing data in scan set or ground truth set'
        # scan select
        index = random.randint(0, len(scan_set)-1)
        scan_path = scan_set[index]
        gt_path = gt_set[index]
        scan_itk = sitk.ReadImage(scan_path)
        gt_itk = sitk.ReadImage(gt_path)
        # Convert to array
        scan_arr = sitk.GetArrayFromImage(scan_itk)
        gt_arr = sitk.GetArrayFromImage(gt_itk)
        #move axis from (z,x,y) -> (x,y,z)
        scan_arr = np.moveaxis(scan_arr,0,-1)
        gt_arr = np.moveaxis(gt_arr,0,-1)
        print('Scan shape: ', scan_arr.shape)
        x1,x2,y1,y2,z1,z2 = get_box(scan_arr,0.7,1.0)
        scan_arr = scan_arr[x1:x2,y1:y2,z1:z2]
        gt_arr = gt_arr[x1:x2,y1:y2,z1:z2]
        print('Crop shape: ', scan_arr.shape)
        if print_log:
            print('Selected scan: ',scan_path)
            print('Selected gt: ',gt_path)
            print('Load scan and ground truth: ',time.monotonic()-t,' s')
            t = time.monotonic()
        for j in range(0,sample_per_scan):
            sample_time_init = time.monotonic()
            #make a copy
            scan_aug = np.array(scan_arr,dtype = 'float32')
            gt_aug = np.array(gt_arr,dtype = 'float32')
            have_tumor= True if gt_aug.max()==1 else False
            #Random rotation x y
            if random.uniform(0, 1) < rotate_prob and max_rotate_angle_x > 0.:
                rotate_angle = random.uniform(-max_rotate_angle_x, max_rotate_angle_x)
                scan_aug = scipy.ndimage.interpolation.rotate(scan_aug,rotate_angle,(1,2),reshape = False,mode = 'nearest')
                gt_aug = scipy.ndimage.interpolation.rotate(gt_aug,rotate_angle,(1,2),reshape = False,mode = 'nearest')
                if print_log:
                    print('Random rotation x: ',time.monotonic()-t,' s')
                    t = time.monotonic()
            if random.uniform(0, 1) < rotate_prob and max_rotate_angle_y > 0.:
                rotate_angle = random.uniform(-max_rotate_angle_y, max_rotate_angle_y)
                scan_aug = scipy.ndimage.interpolation.rotate(scan_aug,rotate_angle,(0,2),reshape = False,mode = 'nearest')
                gt_aug = scipy.ndimage.interpolation.rotate(gt_aug,rotate_angle,(0,2),reshape = False,mode = 'nearest')
                if print_log:
                    print('Random rotation y: ',time.monotonic()-t,' s')
                    t = time.monotonic()
            # Random crop z
            size_input_XY = np.max([np.min([scan_aug.shape[0],scan_aug.shape[1]]),crop_size[2]]).astype(int)
            while True and have_tumor:
                crop_scan, crop_mask = random_crop_with_mask(scan_aug, gt_aug, (size_input_XY,size_input_XY,crop_size[2]))
                if crop_mask.max() >condition_crop or random.uniform(0,1) < 0.01 :
                    scan_aug = crop_scan
                    gt_aug = crop_mask
                    break
            #Random rotation z
            if random.uniform(0, 1) < rotate_prob and max_rotate_angle_z > 0.:
                rotate_angle = random.uniform(-max_rotate_angle_z, max_rotate_angle_z)
                scan_aug = scipy.ndimage.interpolation.rotate(scan_aug,rotate_angle,(0,1),reshape = True,mode = 'nearest')
                gt_aug = scipy.ndimage.interpolation.rotate(gt_aug,rotate_angle,(0,1),reshape = True,mode = 'nearest')
                if print_log:
                    print('Random rotation z: ',time.monotonic()-t,' s')
                    t = time.monotonic()
            # Elastic Transform
            if random.uniform(0, 1) < elastic_prob:
                merge_volume = np.concatenate((scan_aug, gt_aug), axis=2)
                alpha_factor = 2
                sigma_factor = 0.08
                alpha_affine_factor = 0.08
                if print_log: 
                    print('Elastic Transform Prepare: ',time.monotonic()-t,' s')
                    t = time.monotonic()
                merge_result = elastic_transform_v2(image = merge_volume, alpha = merge_volume.shape[1] * alpha_factor, sigma = merge_volume.shape[1] * sigma_factor, alpha_affine = merge_volume.shape[1] * alpha_affine_factor)
                scan_aug = merge_result[...,:scan_aug.shape[2]]
                gt_aug = merge_result[...,-gt_aug.shape[2]:]
                if print_log: 
                    print('Elastic Transform Perform: ',time.monotonic()-t,' s')
                    t = time.monotonic()
            print("max: ",gt_aug.max())
            # Random crop x y
            have_tumor = True if gt_aug.max() > 0.5 else False
            while True and have_tumor:
                crop_scan, crop_mask = random_crop_with_mask(scan_aug, gt_aug, crop_size)
                if crop_mask.max() > condition_crop or random.uniform(0, 1) < 0.01 :
                    scan_aug = crop_scan
                    gt_aug = crop_mask
                    break
            scan_aug, gt_aug = random_crop_with_mask(scan_aug, gt_aug, crop_size)
            # Add to save array
            save_index = i*sample_per_scan+j
            scan_aug[scan_aug<0.] = 0.
            scan_aug[scan_aug>1.] = 1.
            gt_aug[gt_aug>=0.5] = 1
            gt_aug[gt_aug<0.5] = 0
            scans[save_index] = scan_aug
            gts[save_index] = gt_aug
            
            sample_process_time = time.monotonic()-sample_time_init
            print('Sample generate time: ',sample_process_time,' s')
            sample_left = total-save_index-1
            avg_time =((avg_time*save_index)+sample_process_time)/(save_index+1)
            time_left = avg_time*sample_left
            print('Elapsed time: ',time.strftime('%H:%M:%S', time.gmtime(time.monotonic()-total_time)))
            print('Estimate Remaining time: ',time.strftime('%H:%M:%S', time.gmtime(time_left)))
            print('-------------------------------------------Finish: ', save_index+1, '/ ', total,'-------------------------------------------')
    scan_file = os.path.join(save_dir,'scan-n'+str(total).zfill(4)+'-s'+str(crop_size[0])+'_'+str(crop_size[1])+'_'+str(crop_size[2])+'.part'+str(part_number).zfill(3)+'.npy')
    gt_file = os.path.join(save_dir,'gt-n'+str(total).zfill(4)+'-s'+str(crop_size[0])+'_'+str(crop_size[1])+'_'+str(crop_size[2])+'.part'+str(part_number).zfill(3)+'.npy')
    print('Saving scan: ',scan_file)
    np.save(scan_file, scans)
    print('Saving ground truth: ',gt_file)
    np.save(gt_file, gts)
    print('Done!!!')
