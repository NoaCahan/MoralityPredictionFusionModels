import numpy as np
import pandas as pd
import pydicom as dicom
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import SimpleITK as sitk
import os.path
import matplotlib.pyplot as plt
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.ndimage
import random
import warnings
import torch
from collections import OrderedDict

def _load(checkpoint_path, use_cuda):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        device = torch.device('cpu')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        #checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    return checkpoint

def load_model(model, checkpoint_path, use_cuda):

    if os.path.exists(checkpoint_path):
        state_dict = _load(checkpoint_path, use_cuda)
        start_epoch = state_dict['epoch']
        best_prec1 = state_dict['best_prec1']
        if use_cuda:
            device = torch.device("cuda")
            
            #model.load_state_dict(state_dict['state_dict'])
            model.load_state_dict(state_dict['state_dict'], strict=False)
            model.to(device)
        else:
            new_state_dict = OrderedDict()
            for k, v in state_dict['state_dict'].items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
                # load params
            #model.load_state_dict(new_state_dict)
            model.load_state_dict(new_state_dict, strict=False)
        return model, start_epoch, best_prec1
    else:
        print("No checkpoint found!")
        return None

def save_results_to_csv(file, X, results):
    submisstion = pd.DataFrame()
    submisstion['accession_number'] = X['accession_number']
    submisstion['Severity'] = results
    submisstion.to_csv('results.csv', index=False)

def sort_label(X):
    # Choose label and replace True with 1 and False 0
    y = X.T_mortality30d
    y = y.astype(int)
    X = X.drop('T_mortality30d', axis=1)
    return X, y

def save_updated_image(img_arr, path, spacing):
    itk_scaled_img = sitk.GetImageFromArray(img_arr, isVector=False)
    itk_scaled_img.SetSpacing(spacing)
    #itk_scaled_img.SetOrigin(origin)
    sitk.WriteImage(itk_scaled_img, path)

def dicom_load_scan(path):

    # Two Sorting options: 'InstanceNumber', 'SliceLocation'
    attr = {}
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    # For missing ImagePositionPatient
    #slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    slices.sort(key = lambda x: float(x.SliceLocation))

    slices2 = []
    prev = -1000000
    # remove redundant slices
    for slice in slices:
        # For missing ImagePositionPatient
        #cur = slice.ImagePositionPatient[2]
        cur = slice.SliceLocation
        if cur == prev:
            continue
        prev = cur
        slices2.append(slice)
    slices = slices2

    for i in range(len(slices)-1):
        try:
            slice_thickness = np.abs(slices[i].ImagePositionPatient[2] - slices[i+1].ImagePositionPatient[2])
        except:
            slice_thickness = np.abs(slices[i].SliceLocation - slices[i+1].SliceLocation)
        if slice_thickness != 0:
            break

    #print('Patient:{} Slice: {}'.format(os.path.basename(path), slice_thickness))

    assert slice_thickness != 0

    for s in slices:
        s.SliceThickness = slice_thickness

    x, y = slices[0].PixelSpacing
    attr['Spacing'] = (x, y, slice_thickness)
    #attr['Origin'] = slices[0].ImagePositionPatient

    return (slices, attr)

def dicom_get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0#-1024

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)

def dicom_convert_folder(src, dst):
    for scandir in os.listdir(src):
        image, attr = dicom_convert_ct(src, scandir)
        save_updated_image(image, os.path.join(dst, scandir + '.mhd'), attr['Origin'], attr['Spacing'])

def dicom_convert_ct(src, scandir):
    slices, attr = dicom_load_scan(os.path.join(src, scandir))
    image = dicom_get_pixels_hu(slices)
    return image, attr

def load_mhd_image(img_file):
    itk_img = sitk.ReadImage(img_file)
    img = sitk.GetArrayFromImage(itk_img)
    return img
    
def view_results(input_mhd, output_mhd, indices, number_of_slices, mhd = True):
    if mhd:
        input = load_mhd_image(input_mhd).astype(np.float32)
        results = load_mhd_image(output_mhd).astype(np.float32)
    else:
        input = input_mhd
        if indices is not None :
            z_start, z_end, x_start, x_end, y_start, y_end = indices
            results = np.full(input.shape, -1024)
            results[ z_start:z_end, x_start:x_end, y_start:y_end] = output_mhd
        else:
            results = output_mhd
    z, y, x = np.shape(input)
    indices = random.sample(range(z),number_of_slices)
    f, plots = plt.subplots(number_of_slices, 2, figsize=(50, 50))
    k = 0
    for i in indices:
        plots[k, 0].axis('off')
        plots[k, 0].imshow(input[i], cmap=plt.cm.gray)
        plots[k, 1].axis('off')
        plots[k, 1].imshow(results[i], cmap=plt.cm.gray)
        k += 1
    plt.show()

def normalize(image, min_bound, max_bound):
    image = (image - min_bound) / (max_bound - min_bound)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

def crop_centered(image):
    offset_z = round( image.shape[0] / 20 )
    offset_x = round( image.shape[1] / 20 )
    offset_y = round( image.shape[2] / 20 )
    image = image[offset_z:image.shape[0]-offset_z,offset_x:image.shape[1]-offset_x, offset_y:image.shape[2]-offset_y]
    return image

def truncate(image, bounds=None ):#min_bound, max_bound):
    if bounds is not None:
        min_bound, max_bound = bounds
    image[image < min_bound] = min_bound
    image[image > max_bound] = max_bound
    return image

def mean_std(img):
    mu = np.mean(img)
    var = np.var(img)
    return mu, var

def resample_volume(img, attr, vox_spacing):

    (x_space, y_space, z_space) = attr['Spacing']
    spacing_old = list(map(float,(z_space, y_space, x_space)))

    (z_axis, y_axis, x_axis) = np.shape(img)
    spacing_new = (vox_spacing, vox_spacing, vox_spacing)
    
    #print('img: {} old spacing: {} new spacing: {}'.format(np.shape(img), spacing_old, spacing_new))
    resize_factor = np.array(spacing_old) / spacing_new 
    new_shape = np.round(np.shape(img) * resize_factor)
    real_resize_factor = new_shape / np.shape(img)
    img_rescaled = scipy.ndimage.interpolation.zoom(img, real_resize_factor, mode='nearest').astype(np.int16)
    return img_rescaled

# Resize volume to a given size using crop / pad / resample
def resize_volume(img, output_size):

    real_resize_factor = np.array(output_size) / np.shape(img)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img_rescaled = scipy.ndimage.interpolation.zoom(img, real_resize_factor, mode='nearest').astype(np.int16)

    return img_rescaled
	
def debug_img(img):
    plt.hist(img.flatten(), bins=80, color='c')
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.show()

def plot_3d(image, threshold=-300):
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    #p = p[:,:,::-1] # Maybe remove this

    verts, faces = measure.marching_cubes_classic(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70) # alpha=0.1)
    face_color = [0.45, 0.45, 0.75] # [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()

def pad_volume(volume, blocksize):

    blocks = np.ceil(np.array(volume.shape)/blocksize).astype(int)
    # Take maximum blocks required for training data
    maxblocks = np.array(target_split)
    new_shape = maxblocks*blocksize # (256 512 512)
    
    if np.array_equal(maxblocks, blocks):
        offset_before = ((new_shape-volume.shape)//2).astype(int)
        offset_after = ((new_shape-volume.shape)-offset_before).astype(int)
        padded_volume = np.pad(volume,[[offset_before[0],offset_after[0]],[offset_before[1],offset_after[1]],[offset_before[2],offset_after[2]]],mode='edge')
    else:
        offset = ((volume.shape - new_shape)//2).astype(int)       
        padded_volume = volume[offset[0]: (offset[0] + new_shape[0]),offset[1]:(offset[1] + new_shape[1]),offset[2]:(offset[2] + new_shape[2])]
    return padded_volume