import numpy as np
from util.display import *
from util.data import *
import SimpleITK as sitk
from sklearn.cluster import DBSCAN
from util.data import *
from util.display import *
from model import cnn
import time
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from skimage import measure
from scipy import spatial
import sys
import os
import gc


def compute_VOE(y_true, y_pred, smooth=1e-8):
    IOU = (np.sum(np.logical_and(y_pred, y_true)) + smooth) / (np.sum(np.logical_or(y_pred, y_true)) + smooth)
    return 100 * (1 - IOU)

def compute_DICE(y_true, y_pred, smooth=1e-8):
    itersection = np.sum(np.logical_and(y_pred, y_true))
    union = np.sum(np.logical_or(y_pred, y_true))
    DICE = (2*itersection + smooth) / (union + itersection + smooth)
    return 100 * DICE
    

def get_surface_vert(volume_label,spacing):
    object_voxels = get_location(volume_label,1)
    surface_voxels = []
    for object_voxel in object_voxels:
        if object_voxel[0]-1>=0:
            x1 = object_voxel[0]-1
        else: x1 = 0
        if object_voxel[0]+2<=volume_label.shape[0]:
            x2 = object_voxel[0]+2
        else: x2 = volume_label.shape[0]
        
        if object_voxel[1]-1>=0:
            y1 = object_voxel[1]-1
        else: y1 = 0
        if object_voxel[1]+2<=volume_label.shape[1]:
            y2 = object_voxel[1]+2
        else: y2 = volume_label.shape[1]
        
        if object_voxel[2]-1>=0:
            z1 = object_voxel[2]-1
        else: z1 = 0
        if object_voxel[2]+2<=volume_label.shape[2]:
            z2 = object_voxel[2]+2
        else: z2 = volume_label.shape[2]
        non_object_voxels_count = np.sum(1-volume_label[x1:x2,y1:y2,z1:z2])
        if  non_object_voxels_count>0:
            surface_voxels.append(object_voxel)
    surface_vert = np.array(surface_voxels) * np.array(spacing)
    return surface_vert
    
def compute_VD(y_true, y_pred, smooth=1e-8):
    segm = np.sum(y_pred, dtype = np.float)
    gt = np.sum(y_true, dtype = np.float)
    vd = ((segm-gt)+smooth)/(gt+smooth)
    return 100 * vd

def compute_AvgD_RMSD_MaxD(gt_surface_vert, segm_surface_vert):
    #compute distance form gt to segm
    # build kd tree
    n_gt_surface = len(gt_surface_vert)
    n_segm_surface = len(segm_surface_vert)
    tree_segm = spatial.KDTree(segm_surface_vert,leafsize=30)
    distance_gt_to_segm,_ = tree_segm.query(gt_surface_vert)
    tree_gt = spatial.KDTree(gt_surface_vert,leafsize=30)
    distance_segm_to_gt,_= tree_gt.query(segm_surface_vert)
    avgD = 1/(n_gt_surface+n_segm_surface)*(np.sum(distance_gt_to_segm)+np.sum(distance_segm_to_gt))
    RMSD = np.sqrt(1/(n_gt_surface+n_segm_surface))*np.sqrt(np.sum(np.square(distance_gt_to_segm))+np.sum(np.square(distance_segm_to_gt)))
    maxD = np.max([np.max(distance_gt_to_segm),np.max(distance_segm_to_gt)])
    return avgD, RMSD, maxD

def compute_score(ref_metric,result_metrics):
    voe_score = 100-result_metrics[0]/ref_metric[0]*25
    vd_score = 100-np.abs(result_metrics[1])/ref_metric[1]*25
    avgd_score = 100-result_metrics[2]/ref_metric[2]*25
    rms_score = 100-result_metrics[3]/ref_metric[3]*25
    maxd_score = 100-result_metrics[4]/ref_metric[4]*25
    Avg_Score = (voe_score+vd_score+avgd_score+rms_score+maxd_score)/5.
    return Avg_Score
    

def compute_score_voe_vd(ref_metric,result_metrics):
    voe_score = 100-result_metrics[0]/ref_metric[0]*25
    vd_score = 100-np.abs(result_metrics[1])/ref_metric[1]*25
    Avg_Score = (voe_score+vd_score)/2.
    return Avg_Score
    
    
def get_location(arr,label):
    locate_arr = np.where(arr==label)
    x = locate_arr[0]
    y = locate_arr[1]
    z = locate_arr[2]
    rt_list = []
    for i in range(0,len(x)):
        rt_list.append((x[i],y[i],z[i]))
    return rt_list

    
def flood_hole_filling_algorithm(scr_arr,background_label):
    blobs_labels, num_label = measure.label(scr_arr, connectivity = 2, background=background_label, return_num = True)
    max_label = 0
    count = 0
    for i in range(1, num_label+1):
        label_count = np.count_nonzero(blobs_labels == i)
        if label_count>count:
            max_label = i
            count = label_count
    blobs_labels[blobs_labels!=max_label]=0
    blobs_labels[blobs_labels==max_label]=1
    return np.array(blobs_labels,dtype=np.uint8)
    
def flood_hole_filling(scr_arr):
    post_process_arr = flood_hole_filling_algorithm(scr_arr,background_label = 0)
    post_process_arr = flood_hole_filling_algorithm(post_process_arr,background_label = 1)
    return -post_process_arr + 1

def binary_median_image_filter(predicted_label, radius):
    image_itk = sitk.GetImageFromArray(predicted_label, isVector=False)
    Binary_Median_filter = sitk.BinaryMedianImageFilter()
    Binary_Median_filter.SetBackgroundValue(0)
    Binary_Median_filter.SetForegroundValue(1)
    Binary_Median_filter.SetRadius(radius)
    image_itk = Binary_Median_filter.Execute(image_itk)
    predicted_label2 = sitk.GetArrayFromImage(image_itk)
    return predicted_label2
    

def print3D(verts, faces, z_angle = 60):
    min_ax = np.min(verts,axis=0)
    max_ax = np.max(verts,axis=0)

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d', alpha = 0.6)

    #`verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces])
    mesh.set_alpha(0.9)
    mesh.set_edgecolor('w')
    mesh.set_facecolor('#ff0000')
    ax.add_collection3d(mesh)
    ax.set_xlabel("z-axis")
    ax.set_ylabel("x-axis")
    ax.set_zlabel("y-axis")
    ax.set_xlim(min_ax[0]-20, max_ax[0]+20)
    ax.set_ylim(min_ax[1]-20, max_ax[1]+20)
    ax.set_zlim(min_ax[2]-20, max_ax[2]+20)
    ax.view_init(azim=z_angle)
    plt.show()

def additional_scan_padding(scan, ratio, fragment_size=(512, 512, 64)):
    scan_shape = scan.shape
    x = (0,0)
    y = (0,0)
    z1 = int(fragment_size[2]*ratio)
    z2 = int(fragment_size[2]*(1-ratio))
    z = (z1,z2)
    additional_scan = np.pad(scan, pad_width=(x,y,z),mode='constant', constant_values=0)
    return  additional_scan
def crop_additional_scan_predict(scan, ratio, fragment_size=(512, 512, 64)):
    scan_shape = scan.shape
    z1 = int(fragment_size[2]*ratio)
    z2 = scan_shape[2] - int(fragment_size[2]*(1-ratio))
    rt_scan = scan[:,:,z1:z2]
    return rt_scan

def scan_predict(scan_to_process, model, batch_size, fragment_size, norm_mul_factor_arr = None, do_mul_factor = False, verbose = 0):
    crop_scans, factor = crop_scan(scan_to_process, fragment_size)
    crop_scans = np.reshape(crop_scans, crop_scans.shape + (1,))
    output = model.predict(crop_scans, batch_size = batch_size,verbose = verbose)[:,:,:,:,0]
    if do_mul_factor:
        output = output * norm_mul_factor_arr
    return rebuild_scan(output, factor, fragment_size)

def norm_factors(arr, number):
    size = len(arr)
    x2_arr = np.concatenate((arr,arr))
    norm = np.array(arr)
    f = np.int(size/number)
    for i in range(1,number):
        norm += x2_arr[f*i:f*i+size]
    return norm
def CNN_predict(input_scan_arr, model, target_size, batch_size, mul_factor_arr ,predict_time = 16, verbose = 0):
    gc.collect()
    original_size = input_scan_arr.shape
    scan_arr = padding_scan(input_scan_arr, target_size)
    #additional predictions
    if original_size[2]> target_size[2] and predict_time >= 2:
        ratios = np.arange(1,predict_time, dtype = float)
        ratios = ratios / predict_time
        norm_arr = norm_factors(mul_factor_arr,predict_time)
        norm_mul_factor_arr = mul_factor_arr / norm_arr
        #first scan
        predicted_output = scan_predict(scan_to_process = scan_arr, model = model, batch_size = batch_size, fragment_size = target_size, norm_mul_factor_arr = norm_mul_factor_arr,do_mul_factor= True, verbose = verbose)
        #additional scan
        for ratio in ratios:
            gc.collect()
            predicted_output_2 = scan_predict(scan_to_process = additional_scan_padding(scan_arr, ratio ,fragment_size=target_size), model = model, batch_size = batch_size, fragment_size = target_size, norm_mul_factor_arr = norm_mul_factor_arr,do_mul_factor= True, verbose = verbose)
            predicted_output += predicted_output_2[:,:,int(target_size[2]*ratio):predicted_output_2.shape[2] - int(target_size[2]*(1-ratio))]
            del predicted_output_2
    else:
        predicted_output = scan_predict(scan_to_process = scan_arr, model = model, batch_size = batch_size, fragment_size = target_size, norm_mul_factor_arr = None,do_mul_factor= False, verbose = verbose)
    predicted_label = crop_to_original_size(predicted_output,original_size)
    return predicted_label

def sigmoid_to_softmax(predicted_label):
    soft_max_label = np.ndarray(predicted_label.shape + (2,), dtype=predicted_label.dtype)
    soft_max_label[:,:,:,0] = 1. - predicted_label
    soft_max_label[:,:,:,1] = predicted_label
    return soft_max_label
    
    
def save_itk(image, origin, spacing, filename):
    image = image.astype(np.ubyte)
    itkimage = sitk.GetImageFromArray(image, isVector=False)
    itkimage.SetSpacing(spacing)
    itkimage.SetOrigin(origin)
    print('Save file: ',filename)
    sitk.WriteImage(itkimage, filename)
    
    
def evaluate_model_CNN(evaluate_X, evaluate_Y, checkpoint_path, mul_factor_arr, max_size = (512,512,128), batch_size = 1,z_limit = None,z_offset = None,crop_liver = False, resample_spacing = None,normalize_scan = False,predict_time = 16, use_AD_filter = True, filter_Conductance = 2.0, verbose = 0,use_flood_hole_filling = True, ref_metrics = [6.4,4.7,1.,1.8,19], dislay3D = False,save_dir = None):
    result = []
    total_time = time.monotonic()
    sys.setrecursionlimit(100000)

    for i in range(0,len(evaluate_X)):
        total_sample_time = time.monotonic()
        if verbose: print('-'*120)
        if verbose: print('Sample: ',evaluate_X[i])
        t = time.monotonic()
        image_itk = sitk.ReadImage(evaluate_X[i])
        img_itk_origin = image_itk.GetOrigin()
        if verbose: 
            print('Loading scan : ', time.monotonic() - t, ' s')
            t = time.monotonic()
        spacing = image_itk.GetSpacing()
        scan_arr = sitk.GetArrayFromImage(image_itk)
        scan_arr = np.moveaxis(scan_arr,0,-1)
        original_size = np.array(scan_arr.shape)
        if verbose: 
            print('Scan size: ', original_size)
            t = time.monotonic()

        if crop_liver:
            x1,x2,y1,y2,z1,z2 = get_box(scan_arr,1,400)
            if z_offset:
                z1 = z1 + z_offset
            if z_limit:
                z2 = np.min([z1 + z_limit,scan_arr.shape[2]]).astype(int)
            scan_arr = scan_arr[x1:x2,y1:y2,z1:z2]
            if verbose: 
                print('Crop shape: ', scan_arr.shape)
        pre_resample_size = np.array(scan_arr.shape)
        if resample_spacing != None:
            if verbose:
                print('Sample spacing: ', spacing)
            zoom_factor = np.array(spacing)/np.array(resample_spacing)
            scan_arr = scipy.ndimage.interpolation.zoom(scan_arr,zoom_factor,mode='nearest')
            if verbose: 
                print('Re size :', time.monotonic() - t, ' s')
                t = time.monotonic()
                print('Sample new shape: ', scan_arr.shape)
        if normalize_scan:
            scan_arr = normalize(scan_arr)
        #Anisotropic Diffusion Filter
        if use_AD_filter:
            image_itk2 = sitk.GetImageFromArray(scan_arr, isVector=False)
            AD_filter = sitk.CurvatureAnisotropicDiffusionImageFilter()
            AD_filter.SetTimeStep(0.0625)
            AD_filter.SetNumberOfIterations(10)
            AD_filter.SetConductanceParameter(filter_Conductance)
            image_itk2 = AD_filter.Execute(image_itk2)
            scan_arr = sitk.GetArrayFromImage(image_itk2)
            if verbose: 
                print('Anisotropic Diffusion Filter: ',time.monotonic()-t)
                t = time.monotonic()
        input_size = finding_crop_size(scan_arr.shape,max_size,16)
        target_size = [input_size[0],input_size[1],max_size[2]]
        if verbose:
            print('Input size: ',target_size)
            t = time.monotonic()
        model = cnn.CNN(size=target_size,base_n_filter = 16, output_activation_name='sigmoid', is_training = False,checkpoint_path = checkpoint_path)
        if verbose:
            print('Loading model : ', time.monotonic() - t, ' s')
            t = time.monotonic()
        predicted_label = CNN_predict(scan_arr.astype(np.float32), model, target_size, batch_size, mul_factor_arr = mul_factor_arr ,verbose = verbose, predict_time=predict_time)
        del model
        if verbose: 
            print('Predict total time: ', time.monotonic() - t, ' s')
            t = time.monotonic()
        if resample_spacing != None:
            zoom_factor = np.array(pre_resample_size)/np.array(predicted_label.shape)
            predicted_label = scipy.ndimage.interpolation.zoom(predicted_label,zoom_factor,mode='nearest')
            if verbose: 
                print('Resize: ', time.monotonic() - t, ' s')
                t = time.monotonic()
        predicted_label = np.round(predicted_label).astype(np.uint8)
        if use_flood_hole_filling:
            predicted_label = flood_hole_filling(predicted_label)
            if verbose:
                print('Flood hole filling: ', time.monotonic() - t, ' s')
                t = time.monotonic()
        if crop_liver:
            predicted_label = np.pad(predicted_label,((x1,original_size[0]-x2),(y1,original_size[1]-y2),(z1,original_size[2]-z2)),mode = 'constant')

        if verbose: 
            print('Output Size: ',predicted_label.shape)
        if save_dir:
            file_name = os.path.join(save_dir,''.join(os.path.splitext(os.path.basename(evaluate_X[i]))).replace('volume','segmentation'))
            save_itk(np.moveaxis(predicted_label,-1,0),img_itk_origin,spacing,file_name)
        if evaluate_Y!=None:
            assert len(evaluate_X) == len(evaluate_Y),'Path Error'
            ground_truth_itk = sitk.ReadImage(evaluate_Y[i])
            ground_truth = sitk.GetArrayFromImage(ground_truth_itk)
            ground_truth = np.moveaxis(ground_truth,0, -1)
            if verbose: 
                print('Loading ground_truth : ', time.monotonic() - t, ' s')
                t = time.monotonic()
            sample_voe = compute_VOE(ground_truth,predicted_label)
            sample_vd = compute_VD(ground_truth,predicted_label)
            gt_surface_vert = get_surface_vert(ground_truth,spacing)
            segm_surface_vert = get_surface_vert(predicted_label,spacing)
            sample_avgD, sample_RMSD, sample_maxD = compute_AvgD_RMSD_MaxD(gt_surface_vert,segm_surface_vert)
            sample_dice_score = compute_DICE(ground_truth,predicted_label)
            sample_result = [sample_voe,sample_vd,sample_avgD, sample_RMSD, sample_maxD,sample_dice_score]
            score = compute_score(ref_metrics,sample_result)
            if verbose: print('Sample: ','VOE :',sample_result[0],' %   ,','VD :',sample_result[1],' %   ,','AvgD :',sample_result[2],' mm   ,','RMSD :',sample_result[3],' mm   ,','MaxD :',sample_result[4],' mm   ,','Score :', score,'Dice score :',sample_result[5],' s')
            sample_result.append(score)
            result.append(sample_result)
            if verbose: print('Metrics compute : ', time.monotonic() - t, ' s')
            if dislay3D:
                gt_surface_vert, gt_faces,_,_ = measure.marching_cubes_lewiner(ground_truth,level = 0.,gradient_direction = 'ascent' ,spacing = spacing)
                print('Ground truth 3D: ')
                print3D(gt_surface_vert, gt_faces,z_angle = 180)
        if dislay3D:
            print('Segmentation result 3D: ')
            segm_surface_vert, segm_faces,_,_ = measure.marching_cubes_lewiner(predicted_label,level = 0.,gradient_direction = 'ascent' ,spacing = spacing)
            print3D(segm_surface_vert, segm_faces,z_angle = 60)
            print3D(segm_surface_vert, segm_faces,z_angle = 180)
            print3D(segm_surface_vert, segm_faces,z_angle = 300)
            
    if evaluate_Y!=None:
        result = np.array(result)
        result_pd = pd.DataFrame({'VOE (%)':result[:,0],'VD (%)':result[:,1],'AvgD (mm)':result[:,2],'RMSD (mm)':result[:,3],'MaxD (mm)':result[:,4],'Dice score (%)':result[:,5],'Score':result[:,6]})
        file_name = [os.path.splitext(os.path.basename(x))[0] for x in evaluate_X]
        result_pd = result_pd.set_index([file_name])
        result_pd = result_pd.append(result_pd.agg(['mean','std']))
        if verbose: print('Total time : ', time.monotonic() - total_time, ' s')
        return result_pd[result_pd.columns[::-1]]
    
def evaluate_model_UNET3D(evaluate_X, evaluate_Y, checkpoint_path, mul_factor_arr, max_size = (512,512,128), batch_size = 1,z_limit = None,z_offset = None,crop_liver = False, binary_median_image_filter_radius = 0, resample_spacing = None,normalize_scan = False,predict_time = 16, use_AD_filter = True, filter_Conductance = 2.0, verbose = 0,use_flood_hole_filling = True, ref_metrics = [6.4,4.7,1.,1.8,19], dislay3D = False,save_dir = None, save_set = 'LITS2017'):
    result = []
    total_time = time.monotonic()
    sys.setrecursionlimit(100000)

    for i in range(0,len(evaluate_X)):
        total_sample_time = time.monotonic()
        if verbose: print('-'*120)
        if verbose: print('Sample: ',evaluate_X[i])
        t = time.monotonic()
        image_itk = sitk.ReadImage(evaluate_X[i])
        img_itk_origin = image_itk.GetOrigin()
        if verbose: 
            print('Loading scan : ', time.monotonic() - t, ' s')
            t = time.monotonic()
        spacing = image_itk.GetSpacing()
        scan_arr = sitk.GetArrayFromImage(image_itk)
        scan_arr = np.moveaxis(scan_arr,0,-1)
        original_size = np.array(scan_arr.shape)
        if verbose: 
            print('Scan size: ', original_size)
            t = time.monotonic()

        if crop_liver:
            x1,x2,y1,y2,z1,z2 = get_box(scan_arr,1,400)
            if z_offset:
                z1 = z1 + z_offset
            if z_limit:
                z2 = np.min([z1 + z_limit,scan_arr.shape[2]]).astype(int)
            scan_arr = scan_arr[x1:x2,y1:y2,z1:z2]
            if verbose: 
                print('Crop shape: ', scan_arr.shape)
        pre_resample_size = np.array(scan_arr.shape)
        if resample_spacing != None:
            if verbose:
                print('Sample spacing: ', spacing)
            zoom_factor = np.array(spacing)/np.array(resample_spacing)
            scan_arr = scipy.ndimage.interpolation.zoom(scan_arr,zoom_factor,mode='nearest')
            if verbose: 
                print('Re size :', time.monotonic() - t, ' s')
                t = time.monotonic()
                print('Sample new shape: ', scan_arr.shape)
        if normalize_scan:
            scan_arr = normalize(scan_arr)
        #Anisotropic Diffusion Filter
        if use_AD_filter:
            image_itk2 = sitk.GetImageFromArray(scan_arr, isVector=False)
            AD_filter = sitk.CurvatureAnisotropicDiffusionImageFilter()
            AD_filter.SetTimeStep(0.0625)
            AD_filter.SetNumberOfIterations(10)
            AD_filter.SetConductanceParameter(filter_Conductance)
            image_itk2 = AD_filter.Execute(image_itk2)
            scan_arr = sitk.GetArrayFromImage(image_itk2)
            if verbose: 
                print('Anisotropic Diffusion Filter: ',time.monotonic()-t)
                t = time.monotonic()
        input_size = finding_crop_size(scan_arr.shape,max_size,16)
        target_size = [input_size[0],input_size[1],max_size[2]]
        if verbose:
            print('Input size: ',target_size)
            t = time.monotonic()
        del  image_itk
        model = cnn.Unet3D_release(size=target_size,base_n_filter = 16, output_activation_name='sigmoid', Unet3D_weights_path = checkpoint_path)
        if verbose:
            print('Loading model : ', time.monotonic() - t, ' s')
            t = time.monotonic()
        predicted_label = CNN_predict(scan_arr.astype(np.float32), model, target_size, batch_size, mul_factor_arr = mul_factor_arr ,verbose = verbose, predict_time=predict_time)
        del model
        del scan_arr
        if verbose: 
            print('Predict total time: ', time.monotonic() - t, ' s')
            t = time.monotonic()
        if resample_spacing != None:
            zoom_factor = np.array(pre_resample_size)/np.array(predicted_label.shape)
            predicted_label = scipy.ndimage.interpolation.zoom(predicted_label,zoom_factor,mode='nearest')
            if verbose: 
                print('Resize: ', time.monotonic() - t, ' s')
                t = time.monotonic()
        predicted_label = np.round(predicted_label).astype(np.uint8)
        if binary_median_image_filter_radius>=1:
            predicted_label = binary_median_image_filter(predicted_label,binary_median_image_filter_radius)
        if use_flood_hole_filling:
            predicted_label = flood_hole_filling(predicted_label)
            if verbose:
                print('Flood hole filling: ', time.monotonic() - t, ' s')
                t = time.monotonic()
        if crop_liver:
            predicted_label = np.pad(predicted_label,((x1,original_size[0]-x2),(y1,original_size[1]-y2),(z1,original_size[2]-z2)),mode = 'constant')

        if verbose: 
            print('Output Size: ',predicted_label.shape)
        if save_dir:
            if save_set == 'SLIVER07':
                file_name = os.path.join(save_dir,''.join(os.path.splitext(os.path.basename(evaluate_X[i]))).replace('orig','seg'))
            else:
                file_name = os.path.join(save_dir,''.join(os.path.splitext(os.path.basename(evaluate_X[i]))).replace('volume','segmentation'))
            save_itk(np.moveaxis(predicted_label,-1,0),img_itk_origin,spacing,file_name)
            
        if dislay3D:
            print('Segmentation result 3D: ')
            segm_surface_vert, segm_faces,_,_ = measure.marching_cubes_lewiner(predicted_label,level = 0.,gradient_direction = 'ascent' ,spacing = spacing)
            print3D(segm_surface_vert, segm_faces,z_angle = 60)
            print3D(segm_surface_vert, segm_faces,z_angle = 180)
            print3D(segm_surface_vert, segm_faces,z_angle = 300)
            
        if evaluate_Y!=None:
            assert len(evaluate_X) == len(evaluate_Y),'Path Error'
            ground_truth_itk = sitk.ReadImage(evaluate_Y[i])
            ground_truth = sitk.GetArrayFromImage(ground_truth_itk)
            ground_truth = np.moveaxis(ground_truth,0, -1)
            ground_truth[ground_truth<0.5] = 0
            ground_truth[ground_truth>=0.5] = 1
            ground_truth = ground_truth.astype(np.uint8)
            if verbose: 
                print('Loading ground_truth : ', time.monotonic() - t, ' s')
                t = time.monotonic()
            sample_voe = compute_VOE(ground_truth,predicted_label)
            sample_vd = compute_VD(ground_truth,predicted_label)
            gt_surface_vert = get_surface_vert(ground_truth,spacing)
            segm_surface_vert = get_surface_vert(predicted_label,spacing)
            sample_avgD, sample_RMSD, sample_maxD = compute_AvgD_RMSD_MaxD(gt_surface_vert,segm_surface_vert)
            sample_dice_score = compute_DICE(ground_truth,predicted_label)
            sample_result = [sample_voe,sample_vd,sample_avgD, sample_RMSD, sample_maxD,sample_dice_score]
            score = compute_score(ref_metrics,sample_result)
            if verbose: print('Sample: ','VOE :',sample_result[0],' %   ,','VD :',sample_result[1],' %   ,','AvgD :',sample_result[2],' mm   ,','RMSD :',sample_result[3],' mm   ,','MaxD :',sample_result[4],' mm   ,','Score :', score,'Dice score :',sample_result[5],' s')
            sample_result.append(score)
            result.append(sample_result)
            if verbose: print('Metrics compute : ', time.monotonic() - t, ' s')
            if dislay3D:
                gt_surface_vert, gt_faces,_,_ = measure.marching_cubes_lewiner(ground_truth,level = 0.,gradient_direction = 'ascent' ,spacing = spacing)
                print('Ground truth 3D: ')
                print3D(gt_surface_vert, gt_faces,z_angle = 60)
                print3D(gt_surface_vert, gt_faces,z_angle = 180)
                print3D(gt_surface_vert, gt_faces,z_angle = 300)
            
    if evaluate_Y!=None:
        result = np.array(result)
        result_pd = pd.DataFrame({'VOE (%)':result[:,0],'VD (%)':result[:,1],'AvgD (mm)':result[:,2],'RMSD (mm)':result[:,3],'MaxD (mm)':result[:,4],'Dice score (%)':result[:,5],'Score':result[:,6]})
        file_name = [os.path.splitext(os.path.basename(x))[0] for x in evaluate_X]
        result_pd = result_pd.set_index([file_name])
        result_pd = result_pd.append(result_pd.agg(['mean','std']))
        if verbose: print('Total time : ', time.monotonic() - total_time, ' s')
        return result_pd[result_pd.columns[::-1]]