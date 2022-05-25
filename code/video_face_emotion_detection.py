



import cv2
import numpy as np
from keras.prepocessing import image
from keras.models import model_from_json
from keras.preprocessing.image import Tokenizer
import face_recognition

#capture the video from default camera
webcam_video_stream=cv2.VideoCapture('modi.mp4')


#load the model and load the weights
face_exp_model=model_from_json(open("dataset/facial_expression_model_structure.json","r").read())
face_exp_model.load_weights('dataset/facial_expression_model_weights.h5')
#declare the emotions type
emotions_label=('angrry','disgust','fear','happy','sad','surprise','neutral')



#initilise the array location to hold all face location in aframe
all_face_locations=[]

while True:
    #get the current frame from the video stream as an image
    ret,current_frame=webcam_video_stream.read()
    #resize the current frame to 1/4 size to process faster
    current_frame_small=cv2.resize(current_frame,(0,0),fx=0.25,fy=0.25)
    #detect all face in the images
    #arguments are image,no of time to upsample,model
    all_face_locations=face_recognition.face_locations(current_frame_small,model='hog')
    
    #loop through the face location
    for i, current_face_location in enumerate(all_face_locations):
        #splitting the tuple to get four positon valus of cuurent face
        top_pos,right_pos,buttom_pos,l_pos=current_face_location
        top_pos=top_pos*4
        right_pos=right_pos*4
        buttom_pos=buttom_pos*4
        l_pos=l_pos*4
        #printing the location of cuurent face
        print("found face {} at top: {},right:{},buttom:{},left:{}".format(i+1, top_pos,right_pos,buttom_pos,l_pos))
        #slicing the current face from amin image
        current_face_image=current_frame[top_pos:buttom_pos,l_pos:right_pos]
       
       
        
        
        
         #draw rectangle around the face detection
        cv2.rectangle(current_frame, (l_pos,top_pos), (right_pos,buttom_pos), (0,0,255),2)
        
        #preprocessing input,convert it to an image like as the data in dataset 
        #convert to grey scale
        current_face_image=cv2.resize(current_face_image, (48,48))
        #convert the PIL image into a numpy array
        img_pixels=image.img_to_array(current_face_image)
        #exxpand the shape of an array into single row multiple column
        img_pixels=np.expand_dims(img_pixels, axis=0)
        #pixels are in range of [0,255] .normalize all pixels in scale of [0,1]
        img_pixels/=255
        
        
        
        #do prediction using model ,get  the prediction valuses for all 7 expressions
        exp_predictions=face_exp_model.predict(img_pixels)
        #find max index prediction value (0,7)
        max_index=np.argmax(exp_predictions[0])
        #get correspoding level from emotion level
        emotion_label=emotions_label[max_index]
        
        #display the name as text in the image
        font=cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame, emotion_label, (l_pos,buttom_pos), font, 0.5, (255,255,255),1)
        
        
        
        #showing the current face with rectangle drawn
        cv2.imshow("WebCame Video", current_frame)
    
    if cv2.waitKey(1) & 0xff==ord('q'):
        break
    
webcam_video_stream.release()
cv2.destroyAllWindows()
    
    
    
    
    
    
    
    
    
    
    
    
    
    