"""
    Mask R-CNN
    Display and Visualization Functions.
    
    Copyright (c) 2017 Matterport, Inc.
    Licensed under the MIT License (see LICENSE for details)
    Written by Waleed Abdulla
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import sys
import logging
import random
import itertools
import colorsys
import IPython.display
import cv2
import skimage.io
#import mlperf
from   skimage.morphology.selem   import star 
import numpy                      as     np
from   skimage.measure            import find_contours
import matplotlib.pyplot          as     plt
from   matplotlib                 import patches,  lines
from   matplotlib.patches         import Polygon
from   numpy.ma.core              import masked
from   matplotlib.path            import Path
from   numpy.polynomial.chebyshev import chebtrim
""" ************************************************************************************************** """
# Root directory of the project
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)               # To find local version of the library

from mrcnn          import utils

sys.path.append('..//..//DEAP_C3D')
sys.path.append('..//..//DEAP_Face')

from Configurations  import Config_Face
from CommonFunctions import CommonFunctions
from Configurations  import Config_VGG

CommonFunctions      = CommonFunctions() 
Config_Face          = Config_Face()
Config_VGG           = Config_VGG()
""" ************************************************************************************************** """
def main_process (image, r, class_names, destination_dir, ImageSize, index, save = False , visualize=False):
    
    """ Extract coordinates of mask """
    x , y         = Step1 (image, r, class_names)
   
    """ Extract mask """
    mask,image    = Step2 (image,x,y)

    """ Extract masked image """
    MaskedImg     = Step3 (image, mask)
        
    """ Extract trimmed image """
    image, mask, Trimmed, MaskedImg = Step4 (x , y , image , mask , MaskedImg)

    print ('Trimmed Shape After Masking    ' , np.array (Trimmed).shape)
    print ('destination dirdestination_dir ' , destination_dir )

    """ Detect face region from trimmed image """
    if Config_Face.ApplyOpenCVAfterMask :
        
        """ Apply openCV """
        Trimmed          = Step5 (Trimmed , ImageSize)
        
        if np.array(Trimmed).shape[0] == 0: #NO face detected in current image index
            
            if index != 0: #If not first image                                
                idx                   = destination_dir.rfind("\\") 
                destination_dir_first = destination_dir[0 : int(idx)+1]
    
                currentImageIndex     = destination_dir [idx+1 : len(destination_dir)] 
                LastImageIndex        = int(currentImageIndex) - 1
                new_Destination_dir   = destination_dir_first + str(LastImageIndex) + '.npy'
                print ('New_Destination_Dir   ' , new_Destination_dir )
                
                from pathlib import Path
                new_Destination_dir       = Path ( new_Destination_dir )
                print ('new_Destination_dir' , new_Destination_dir )
                
                if new_Destination_dir.exists():
                    Trimmed               = np.load (new_Destination_dir)
            
                    """ Post-Processing Face + Mask Image """
                    Trimmed = Step6  ( Trimmed , ImageSize )
                    print ('Trimmed Shape After Post-processing' , np.array(Trimmed).shape)
                    
                    if Config_Face.ApplyDCTAfterMask:
                        
                        """ Apply DCT After Mask """
                        Trimmed = Step7 ( Trimmed )
                    
                    """ Visualize and Save mask, masked image """
                    Step8 (Trimmed, destination_dir, visualize, save)
                    print ('Trimmed Shape ' , np.array(Trimmed).shape)
                 
            else: #If current index is First Image
                """ """
                print ('******************* First index and no face detected so no face is saved ******************* ')
                 
        else: #If Face is detected
            Trimmed = Step6 (Trimmed, Config_Face.ImageSize)
            
            if Config_Face.ApplyDCTAfterMask :
                """ Apply DCT After Mask """
                Trimmed = Step7 (Trimmed)
             
            """ Visualize and Save mask, masked image """
            Step8 (Trimmed, destination_dir, visualize, save)
            print ('Trimmed Shape ' , np.array(Trimmed).shape)
           
def Step1 (image, r, class_names):

    """ 1- Get Mask Contour """
    
    contour     = get_mask_contour (image, r['rois'], r['masks'], class_names, r['class_ids'], r['scores'])

    """ 2- Round contour Values """
    new_contour = updateContour (contour)
   
    """ 3- Extract x and y of points separately """
    x,y         = getContourCoordinates (new_contour)  
    
    return x,y
   
def Step2 (original_img,x,y):
    """ 
        Get Mask
    """
    xycrop         = np.vstack((x, y)).T
    original_image = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    original_image = original_image.T              
    
    nr, nc       = original_image.shape
    ygrid, xgrid = np.mgrid[:nr, :nc]
    xypix        = np.vstack((xgrid.ravel(), ygrid.ravel())).T
    
    # construct a Path from the vertices
    pth          = Path (xycrop, closed=False)
    mask         = pth.contains_points(xypix)
    mask         = mask.reshape(original_image.shape) #Values of Mask: True, False 
    
    return mask, original_image;

def Step3 (original_image, mask):
    
    """ Extract masked image """
    
    masked          = np.zeros((original_image.shape[0], original_image.shape[1]))    
    
    for i in range (mask.shape[0]):
        for j in range (mask.shape[1]):
            if mask[i][j] == True:
                masked[i][j] = original_image [i][j]
            else:
                masked[i][j] = 255

    return masked

def Step4 (x,y , original_image, mask, masked):
    
    """ Extract trimmed image """
    
    xmin, xmax   = int(x.min()), int(np.ceil(x.max()))
    ymin, ymax   = int(y.min()), int(np.ceil(y.max()))
    trimmed      = masked [ymin:ymax, xmin:xmax]
    
    masked       = masked [ymin:ymax, xmin:xmax]
    original_image= original_image.T
    mask          = mask.T
    #masked        = masked.T
    trimmed       = trimmed.T
    masked        = masked.T
        
    return original_image, mask, trimmed, masked
 
def Step5 (image,ImageSize):
    
    """ Apply openCV """
    
    image        = np.array (image, dtype='uint8')  
    face_cascade = cv2.CascadeClassifier (Config_Face.face_detector)    
    faces        = face_cascade.detectMultiScale ( image )
    faceROI      = []     
    
    for (x,y,w,h) in faces:
        
        faceROI = image [y : y + h, x : x + w]
        #faceROI = cv2.resize (faceROI, (ImageSize,ImageSize))
        
    return faceROI 

def Step6(image, ImageSize):
    
    """ Post-Processing Face + Mask Image """
    # Trimmed        = tf.image.flip_left_right(Trimmed)                
    # Means          = tf.reshape (tf.constant(Config_VGG.VGG_MEAN), [1, 1])
    # Trimmed        = Trimmed - 128#Config_VGG.VGG_MEAN 
    # Image          = normalized_image(image)
    # Image          = resize_Image(image, size)
    
    print ('Resizing Face Image ... ')
    image = cv2.resize (image, (ImageSize,ImageSize))
    print ('New Face Size ' , image.shape)
    
    print ('Normalizing Face Image ... ')
    cv2.normalize (image, image, 0, 255, cv2.NORM_MINMAX)
    
    #print ("De-Noising Face Image ...")
    #image = cv2.fastNlMeansDenoising (image, None, 10, 10, 7, 21)
    #image = denoise2(image)
    #image = cv2.blur(image, (1, 1))
    
    return image  

def Step7(Trimmed):
    
    """ Apply DCT After Mask """
    print ('Trimmed Shape for DCT Before ' , np.array(Trimmed).shape)
    
    from scipy.fftpack import dct
    Trimmed = dct(np.rot90(dct(Trimmed),3))
    
    print ('Trimmed Shape for DCT After' , np.array(Trimmed).shape)
    
    return Trimmed

def Step8 (trimmed, destination_dir, visualize, save):

    """ Visualize and Save mask, masked image """    
    if visualize:
        visualizemaskedImage ( trimmed , visualize)

    if save:
        saveMask (trimmed, destination_dir)

""" ************************************************************************************************** """                       
def visualizemaskedImage(image, visualize):
    
    """ """
    figsize   = (Config_Face.ImageSize, Config_Face.ImageSize)
    _, axx    = plt.subplots(1, figsize=figsize )
    
    height, width = image.shape[:2]
    axx.set_ylim(height + 10, -10)
    axx.set_xlim(-10, width + 10)

    axx.imshow(image.astype(np.uint8), cmap=plt.cm.gray)
    
    if visualize:
        plt.show()  

def saveMask (trimmed, path):

        np.save (path , np.array(trimmed))

""" ************************************************************************************************** """               
def normalized_image(image):
    
    # Rescale from [0, 255] to [0, 2]
    #image = tf.multiply (image, 1. / 127.5)
    
    # Rescale to [-1, 1]
    mlperf.logger.log (key=mlperf.tags.INPUT_MEAN_SUBTRACTION, value=[1.0] * 3)
    
    return tf.subtract (image, 1.0)

def resize_Image(image, size):
    
    from PIL         import Image
    from resizeimage import resizeimage
    
    with open(image, 'r+b') as f:
        with Image.open(f) as image:
            cover = resizeimage.resize_cover(image, size)
            cover.save('test-image-cover.jpeg', image.format)
            
    return image

def denoise(img, weight=0.1, eps=1e-3, num_iter_max=200):
    
    """
        Perform total-variation de-noising on a gray-scale image.
         
        Parameters
        ----------
        img : array
            2-D input data to be de-noised.
        weight : float, optional
            Denoising weight. The greater `weight`, the more
            de-noising (at the expense of fidelity to `img`).
        eps : float, optional
            Relative difference of the value of the cost
            function that determines the stop criterion.
            The algorithm stops when:
                (E_(n-1) - E_n) < eps * E_0
        num_iter_max : int, optional
            Maximal number of iterations used for the
            optimization.
     
        Returns
        -------
        out : array
            De-noised array of floats.
         
        Notes
        -----
        Rudin, Osher and Fatemi algorithm.
    """
    
    u  = np.zeros_like(img)
    px = np.zeros_like(img)
    py = np.zeros_like(img)
     
    nm = np.prod(img.shape[:2])
    tau= 0.125
     
    i = 0
    while i < num_iter_max:
        u_old = u
    
    # x and y components of u's gradient
    ux = np.roll(u, -1, axis=1) - u
    uy = np.roll(u, -1, axis=0) - u

    # update the dual variable
    px_new   = px + (tau / weight) * ux
    py_new   = py + (tau / weight) * uy

    norm_new = np.maximum (1, np.sqrt(px_new **2 + py_new ** 2))
    px       = px_new / norm_new
    py       = py_new / norm_new

    # calculate divergence
    rx       = np.roll(px, 1, axis=1)
    ry       = np.roll(py, 1, axis=0)
    div_p    = (px - rx) + (py - ry)
     
    # update image
    u = img + weight * div_p
    
    # calculate error
    error = np.linalg.norm(u - u_old) / np.sqrt(nm)

    if i == 0:
            err_init = error
            err_prev = error
    else:
            # break if error small enough
            if np.abs(err_prev - error) < eps * err_init:
                """ """
                #break
            else:
                e_prev = error
                 
    # don't forget to update iterator
    i += 1

    return u

def denoise2 (image):
    
    from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,denoise_wavelet, estimate_sigma)
    from skimage             import data, img_as_float
    from skimage.util        import random_noise
    
    #original = img_as_float (data.chelsea()[100:250, 50:300])
    sigma    = 0 #.00155
    noisy    = random_noise (image, var=0)#sigma**2)
    
    #fig, ax  = plt.subplots (nrows=2, ncols=4, figsize=(8, 5),sharex=True, sharey=True)
    #plt.gray()
    #plt.show()
    
    return noisy

""" ************************************************************************************************** """               
def saveMaskOld (trimmed, path, save=False, visualize=False):
                        
    #fig = plt.figure(figsize=(2.9 , 2.9) , frameon=False)
    fig = plt.figure(figsize=(Config_Face.ImageSizeRatio , Config_Face.ImageSizeRatio) , frameon=False)

    #ax = plt.Axes(fig, [0., 0., 1., 1.])
    #ax.set_axis_off()
    #fig.add_axes(ax)
    #ax.imshow (trimmed, cmap=plt.cm.gray)
    
    if save:        
        #fig.savefig ( path )
        #buf   = fig2data ( fig )
        #print ('buf.shape buf.shape buf.shape buf.shape', buf.shape)
        #print ('buf.shape buf.shape buf.shape buf.shape', path)
        
        trimmed = cv2.resize (trimmed, (Config_Face.ImageSize,Config_Face.ImageSize))
        print ('Trimmed shapeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee ' , trimmed.shape)
        np.save (path , trimmed)
        
        #cv2.imwrite(path ,buf)
        """ """
       
def fun():

    if Config_Face.ApplyOpenCVAndMask:
     
        #backtorgb = np.array((trimmed.shape[0], trimmed.shape[1]) )
        #cv2.cvtColor ( trimmed , backtorgb,cv2.COLOR_GRAY2BGR )
         
        saveMask (trimmed, destination_dir, save, visualize)
         
        print ('destination_dirdestination_dirdestination_dirdestination_dir ', destination_dir)
        #faces  = CommonFunctions_Face.extractFace_Final (destination_dir , Config_Face.ImageSizeRatio*100)
        length = len (faces)
        print ('length of faces : ' , length)
                                 
        if length > 0:                                        
            LastFaceRegion = faces[0:]
          
        if visualize:
            display (np.array(faces[0:]))
         
        if save:
            imageName     = imageName [0 : len(imageName) - 4] + Config_Face.faceExtension
            faceImagePath = fullFacesPath + imageName          # -5 to remove old extension which was .tiff 
                
def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    print ('buffere size ' , buf.shape)
    buf.shape = ( w, h )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf

def get_mask_contour(image, boxes, masks, class_names, class_ids,
                      scores   =None,     title="",
                      figsize  =(16, 16),    ax=None,
                      show_mask=False, show_bbox=False,
                      colors   =None,  captions=None):

    # Number of instances
    N             = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # Generate random colors
    face_vert  = np.array((1,2))       
    for i in range(N):

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue

        # Mask 
        if class_ids[i] == 1: #Detected as person
            mask = masks [:, :, i]
    
            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask             = np.zeros( (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours                = find_contours(padded_mask, 0.5)            
            
            for verts in contours:
                #Get the contour with larger size (face)
                #print ('Hereeeeeeeeeeeeee ')
                #print ('verts length ', len (verts))
                #print ('face length  ', len (face_vert))
                
                if len(verts) > len(face_vert):
                    face_vert = verts

    return face_vert
    
def updateContour(contour):
    """ 
        Old contour has float indices
    """
    new_contour = []
    
    for index in range(contour.shape[0]):
        """ """
        pixel = np.array(contour[index])        
        x     = int(round(pixel[0]))
        y     = int(round(pixel[1]))
        new_contour.append([x,  y])
        
    return new_contour
 
def getContourCoordinates(new_contour):
    
    x = []
    y = []
    for i in range ( len (new_contour) ):
        x.append(new_contour[i][0])
        y.append(new_contour[i][1])
    #print ('x ' , x)
    #print ('y ' , y)
    
    x = np.array(x)
    y = np.array(y)

    return x,y 
                        
""" ************************************************************************************************** """    
def visualizeMaskedTrimmed(masked_image, mask, masked, trimmed):
    """ """    
    fig, ax = plt.subplots(2, 2)
    ax[0,0].imshow(masked_image, cmap=plt.cm.gray)
    ax[0,0].set_title('original')
    ax[0,1].imshow(mask, cmap=plt.cm.gray)
    ax[0,1].set_title('mask')
    ax[1,0].imshow(masked, cmap=plt.cm.gray)
    ax[1,0].set_title('masked original')
    ax[1,1].imshow(trimmed, cmap=plt.cm.gray)
    ax[1,1].set_title('trimmed original')
    plt.show()
    fig.savefig('foo.png')

def visualizemaskedAndContour(Masked, contour, axx=None):
    
    """ """
    visualize = False
    if not axx:
        figsize   = (20, 20)
        _, axx    = plt.subplots(1, figsize=figsize)
        visualize = True
    
    height, width = Masked.shape[:2]
    axx.set_ylim(height + 10, -10)
    axx.set_xlim(-10, width + 10)
    
    verts = np.fliplr(contour) - 1
    p     = Polygon(verts, facecolor="none", edgecolor='red')
    axx.add_patch(p)
    
    axx.imshow(Masked.astype(np.uint8))
    if visualize:
        plt.show()  
                   
""" ******************************************************************************** """    
def apply_mask(image, mask, color, alpha=0.5):
    """
        Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def random_colors(N, bright=True):
    """
        Generate random colors.
        To get visually distinct colors, generate them in HSV space then
        convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def display_persons(image, boxes, masks, class_names, class_ids,
                      scores   =None,     title="",
                      figsize  =(16, 16),    ax=None,
                      show_mask=False, show_bbox=False,
                      colors   =None,  captions=None):

    # Number of instances
    N             = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    _, ax     = plt.subplots(1, figsize=figsize)

    # Generate random colors
    colors        = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    
    for i in range(N):
        color    = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue

        # Mask    
        if class_ids[i] == 1:
            mask = masks [:, :, i]
    
            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask             = np.zeros( (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours                = find_contours(padded_mask, 0.5)
            face_vert               = np.array((1,1))
            
            for verts in contours:
                #Get the contour with larger size (face)
                if verts.shape[0] > face_vert.shape[0]:
                    face_vert = verts
                    
                # Subtract the padding and flip (y, x) to (x, y)    
                verts = np.fliplr(verts) - 1
                p     = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
                    
    ax.imshow(masked_image.astype(np.uint8))
    auto_show = True
    if auto_show:
         plt.show()

    return masked_image, face_vert

def display_persons_original(image, boxes, masks, class_names, class_ids,
                              scores   =None,     title="",
                              figsize  =(16, 16),    ax=None,
                              show_mask=False, show_bbox=False,
                              colors   =None,  captions=None):

    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show     = False
    if not ax:
        _, ax     = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        
        # Mask    
        if class_ids[i] == 1:
            mask        = masks [:, :, i]
           
            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros( (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours    = find_contours(padded_mask, 0.5)
           
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p     = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
                    
    ax.imshow(masked_image)#.astype(np.uint8))
    if auto_show:
         plt.show()

    print ('masked_image shape ', np.array(masked_image).shape)

    return masked_image

""" ************************************************************************************************** """
def display_instances(image, boxes, masks, class_names, class_ids,
                      scores   =None,     title="",
                      figsize  =(16, 16),    ax=None,
                      show_mask=True, show_bbox=True,
                      colors   =None,  captions=None):
    """
        boxes      : [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
        masks      : [height, width, num_instances]
        class_ids  : [num_instances]
        class_names: list of class names of the dataset
        scores     : (optional) confidence scores for each box
        title      : (optional) Figure title
        show_mask, show_bbox: To show masks and bounding boxes or not
        figsize    : (optional) the size of the image
        colors     : (optional) An array or colors to use with each object
        captions   : (optional) A list of strings to use as captions for each object
    """

    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show     = False
    if not ax:
        _, ax     = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id= class_ids[i]
            score   = scores[i] if scores is not None else None
            label   = class_names[class_id]
            x       = random.randint(x1, (x1 + x2) // 2)
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
            
        ax.text(x1, y1 + 8, caption,color='w', size=11, backgroundcolor="none")

        # Mask
        mask        = masks [:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros( (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours    = find_contours(padded_mask, 0.5)
       
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p     = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
                    
    ax.imshow(masked_image.astype(np.uint8))
    if auto_show:
        plt.show()
         