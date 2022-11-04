import os

class Config():
    """ """
    # Directory to save logs and trained model
    # Root directory of the project
    ROOT_DIR        = os.path.abspath("../")
    MODEL_DIR       = os.path.join(ROOT_DIR, "logs")
    COCO_DIR        = os.path.join(ROOT_DIR, "samples/coco/" )
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5") # Local path to trained weights file
    IMAGE_DIR       = os.path.join(ROOT_DIR, "images")            # Directory of images to run detection on
    MASKS_DIR       = os.path.join(ROOT_DIR, "Masks\\")            # Directory of images to run detection on
    
    class_names     = ['BG'          , 'person'      , 'bicycle'      , 'car'        , 'motorcycle'  , 'airplane'  , #0-5
                       'bus'         , 'train'       , 'truck'        , 'boat'       , 'traffic light','pizza'     , #6-11
                       'fire hydrant', 'stop sign'   , 'parking meter', 'bench'      , 'bird'        , 'elephant'  , #12-17
                       'cat'         , 'dog'         , 'horse'        , 'sheep'      , 'cow'         , 'bear'      , 
                       'zebra'       , 'giraffe'     , 'backpack'     , 'umbrella'   , 'handbag'     , 'tie'       ,
                       'suitcase'    , 'frisbee'     , 'skis'         , 'snowboard'  , 'sports ball' , 'scissors'  ,
                       'kite'        , 'baseball bat', 'baseball glove','skateboard' , 'apple'       , 'remote'    ,
                       'surfboard'   , 'tennis racket', 'bottle'      , 'wine glass' , 'cup'         ,' bed'       ,
                       'fork'        , 'knife'       , 'spoon'        , 'bowl'       , 'banana'      , 'toothbrush',
                       'sandwich'    , 'orange'      , 'broccoli'     , 'carrot'     , 'hot dog'     , 'hair drier',
                       'donut'       , 'cake'        , 'chair'        , 'couch'      , 'potted plant', 'teddy bear', #66
                       'dining table', 'toilet'      , 'tv'           , 'laptop'     , 'mouse'       , 'toaster'   ,
                       'keyboard'    , 'cell phone'  , 'microwave'    , 'oven'       , 'clock'       , 'vase'      , #78
                        'sink'       , 'refrigerator', 'book']
        