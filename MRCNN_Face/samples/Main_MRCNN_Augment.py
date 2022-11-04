from datetime import datetime
start = datetime.now()
print ('Start at : ', start)
""" ************************************************************************************************** """
import os
import sys
import random
import math
import skimage.io
import matplotlib
import cv2
import numpy                  as np
import matplotlib.pyplot      as plt
from skimage.morphology.selem import star 
""" ************************************************************************************************** """
#sys.path.append ('../DEAP_Face')
sys.path.append ('../../DEAP_Face_Augment')

from Configurations         import Config_Face
from CommonFunctions        import CommonFunctions
from Configurations         import Augment

CommonFunctions      = CommonFunctions () 
Config_Face          = Config_Face ()
Augment              = Augment ()

from MRCNN_Config import Config
Config             = Config()
sys.path.append ('../mrcnn')

sys.path.append(Config.ROOT_DIR)     # To find local version of the library
from   mrcnn       import utils
import mrcnn.model as     modellib
from   mrcnn       import visualize
sys.path.append ( Config.COCO_DIR )  # To find local version
import coco
""" ************************************************************************************************** """
class masking(object):
                               
    def main_process ( self , user_index, DEST_DIR2):
        
        COCO_MODEL_PATH = Config.COCO_MODEL_PATH
        MODEL_DIR       = Config.MODEL_DIR
        ROOT_DIR        = Config.ROOT_DIR
        SRC_DIR         = Config.IMAGE_DIR
        DEST_DIR        = Config.MASKS_DIR
        
        conf            = self.display_Config ( COCO_MODEL_PATH )
        model           = self.create_Model   ( MODEL_DIR , COCO_MODEL_PATH , conf )
         
        #SRC_DIR2        = Config_Face.DEAP_frames
        SRC_DIR2        = Augment.DEAP_Augment
        ImageSize       = Config_Face.ImageSize
        
        CommonFunctions.makeDirectory ( DEST_DIR2 )
        print ("SRC_DIR2 : " , SRC_DIR2  )
        print ("DEST_DIR2: " , DEST_DIR2 )
        
        self.process_Mask2 ( user_index , SRC_DIR2 , DEST_DIR2 , model, Config.class_names, ImageSize)     

    def process_Mask2 (self, userIndex , IMAGE_DIR , MASKS_DIR, model , class_names, ImageSize ):
        
        users             = os.listdir ( IMAGE_DIR )
        CommonFunctions.sort_nicely( users )
        print ("IMAGE_DIR: " , IMAGE_DIR )
        
        for userId , user in enumerate ( users ):
            
            if userId == userIndex :
                
                faces_file_path  = IMAGE_DIR   + user                
                trials           = os.listdir ( faces_file_path )
                CommonFunctions.sort_nicely ( trials )                
                userData         = []                      
                
                for trialId , trial in enumerate ( trials ):
                    
                        #if trialId > 22 :   
                        
                        images_path       = faces_file_path  + '\\' + trial                
                        image_names       = os.listdir ( images_path )
                        CommonFunctions.sort_nicely ( image_names )      
                        
                        for imageId , image_name in enumerate ( image_names ): 
                            
                                    #if imageId >  15 :   
                                   
                                    augment_path       = images_path  + '\\' + image_name 
                                    print ("Augment_path: " , augment_path )               
                                    augment_names      = os.listdir ( augment_path )
                                    CommonFunctions.sort_nicely ( augment_names )      
                                    
                                    for augmentId , augment_name in enumerate ( augment_names ): 
                                            #if augmentId == 0 :
                                                                                    
                                            #print ("New Path : " , images_path  +"\\" + str( image_name ) + "\\" + str (augment_name)   )
                                            #print ()
                                            
                                            path  = images_path + "\\" + str(image_name) + "\\" + str (augment_name)
                                            #image = np.load ( path )
                                            print ("image shape " , path ) #Image.shape :  (576, 720, 3)
                                            
                                            image           = skimage.io.imread (images_path + "\\" + str(image_name) + "\\" + str (augment_name) )
                                            print ("image shape " , image.shape)
                                            
                                            # Run Detection
                                            results         = model.detect ( [image] , verbose = 1)
                                            print ("image shape " , image.shape)
                                            
                                            """ ************************************************************************************************** """
                                            # Visualize results
                                            r               = results[0] #Detect only one object
                                            
                                            """ Extract Mask """
                                            #visualize.display_instances(image, r['rois'], r['masks'], class_names, r['class_ids'], r['scores'])
                                            destination_dir = MASKS_DIR + user + '\\' + trial + '\\' + image_name + "\\"
                                            CommonFunctions.makeDirectory ( destination_dir )
                                            
                                            destination_dir = destination_dir + str (augmentId + 1) 
                                            visualize.main_process ( image , r , class_names , destination_dir , ImageSize, augmentId, save = True , visualize = False ) 
                                                               
    def display_Config (self, COCO_MODEL_PATH ):
        
        class InferenceConfig ( coco.CocoConfig ):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT      = 1
            IMAGES_PER_GPU = 1
            
        """ ************************************************************************************************** """
        # Download COCO trained weights from Releases if needed
        if not os.path.exists( COCO_MODEL_PATH ):
            utils.download_trained_weights( COCO_MODEL_PATH )
        
        config     = InferenceConfig()
        #config.display()
    
        return config
    
    def create_Model   (self , MODEL_DIR , COCO_MODEL_PATH, config):
        
        # Create model object in inference mode.
        model      = modellib.MaskRCNN (mode = "inference", model_dir = MODEL_DIR, config = config)
        
        # Load weights trained on MS-COCO
        model.load_weights (COCO_MODEL_PATH , by_name = True )
        
        # COCO Class names
        # Index of the class in the list is its ID. 
        #For example, to get ID of the teddy bear class, use: class_names.index('teddy bear')
        
        return model
                    
    """ ************************************************************************************************** """
if __name__  == "__main__":
    
    masking    = masking()
    nusers     = 22
    
    for userIndex in range ( nusers ):
        
        if userIndex == 11 : 

            print ("Current user is ", str( userIndex + 1 ) )
                   
            masking.main_process ( userIndex ,  Config_Face.DEAP_masks )         
    
    from datetime import datetime
    print ()
    Duration = datetime.now() - start 
    print ('Duration is : ', Duration)               
    