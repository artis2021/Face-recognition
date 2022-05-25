



import cv2
import face_recognition

original_image=cv2.imread('trump-modi.jpg')
# all u can check from below given pic
#original_image=cv2.imread('trump-modi-unknown.jpg')

#load the sample images and get 128 face embedding from there
modi_image=face_recognition.load_image_file('modi.jpg')
modi_face_encodings=face_recognition.face_encodings(modi_image)[0]

trump_image=face_recognition.load_image_file('trump.jpg')
trump_face_encodings=face_recognition.face_encodings(trump_image)[0]

#save the encodings and the correspoding lebels in seperate array in the same order
known_face_encodings=[modi_face_encodings,trump_face_encodings]
known_face_names=["Narendra Modi","Donald Trump"]

#load the unknown image to recognize face in it
image_to_recognize=face_rocognition.load_image_file('trump-modi.jpg')
#image_to_recognize=face_recognition.load_image_file('trump-modi-unknown.jpg')

#detect all face into the image
#arguments are image,no of times upsample model
all_face_locations=face_recognition.face_locations(image_to_recognize,model="hog")
#detect face encoding for all face detected
all_face_encodings=face_rocognition.face_encodings(image_to_recognize,all_face_locations)





#cv2.imshow("test", image_to_detect)


print("there are {} no of faces in given image".format(len(all_face_locations)))

#looping throught he face locations and face embeding
for current_face_location, current_face_encoding in zip(all_face_locations,all_face_encodings):
    #splitting the tuple to get four positon valus of cuurent face
    top_pos,right_pos,buttom_pos,l_pos = current_face_location
    #printing the location of cuurent face
   # print("found face {} at top: {},right:{},buttom:{},left:{}".format(i+1, top_pos,right_pos,buttom_pos,l_pos))
    #find all matches and get the list of matches
    all_matches=face_recognition.compare_faces(known_face_encodings, current_face_encoding)
    #string to hold the label
    name_of_person='Unknown face'
    #check if the all_matches have at least one item
    #if yes,get the index no of face that is located in the first index of all matches
    if True in all_matches:
        first_match_index=all_matches.index(True)
        name_of_person=known_face_names[first_match_index]
        
     #draw rectangle around the face detection
    cv2.rectangle(original_image, (l_pos,top_pos), (right_pos,buttom_pos), (255,0,0),2)
    
    #display the name as text in the image
    font=cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(original_image, name_of_person, (l_pos,buttom_pos), font, 0.5, (255,255,255),1)  
    
    #display the image
    cv2.imshow("Face Identified", original_image)
    
    
    
    
    
    
    
    
    
    
    
    
    current_face_image=image_to_detect[top_pos:buttom_pos,l_pos:right_pos]
    cv2.imshow("face no "+str(i+1),current_face_image)
    #cv2.waitKey(0)         //cheeck in case if it work by using or removing
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    