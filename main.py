import numpy as np
from PIL import Image
import tensorflow as tf
# import cv2

def getPrediction(filename):
    
    
    
    #Load model 
    # saved_model_path = './model_x81_dcs65.h5'
    saved_model_path = './model_x1_1.h5'
    # another_strategy = tf.distribute.MirroredStrategy()
    # with another_strategy.scope():
    #     load_options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
    #options=load_options
    Brain_Model = tf.keras.models.load_model(saved_model_path ,  custom_objects={ 'accuracy' : tf.keras.metrics.MeanIoU(num_classes=4)},compile = False)
                                                    
                                                

    

    


    SIZE = 128 #Resize to same size as training images
    img_path = 'static/images/'+filename
    img = np.asarray(Image.open(img_path).resize((SIZE,SIZE)))
#     img = cv2.resize(img , (SIZE , SIZE))
    img1 = img[:,:,0]
    img2 = img[:,:,1]
    img = np.dstack([img1 , img2])
    img = np.expand_dims(img, axis=0)  #Get it tready as input to the network       
    img = img/255.      #Scale pixel values
    if img.shape == (1,128,128,2):
        
        pred = Brain_Model.predict(img) #Predict 
    else:
        pred = np.zeros((128,128,2))
    return pred
   
   
            
            
     
    

