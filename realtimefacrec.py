#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 07:38:26 2022

@author: xorserd
"""

import cv2
import face_recognition

webcam_video_stream = cv2.VideoCapture(0)



russ_image=face_recognition.load_image_file('images/rd.png')
russ_face_encodings= face_recognition.face_encodings(russ_image)[0]

bit_image=face_recognition.load_image_file('images/bit.jpg')
bit_face_encodings= face_recognition.face_encodings(bit_image)[0]

known_face_encodings = [russ_face_encodings,bit_face_encodings]
known_face_names = ["Russell Damali","Bithiah Obbo"]

#initiliaze array to collect all face locations in the frame
all_face_locations=[]
all_face_encodings = []
all_face_names = []

#loop through every frame in the video  
while True:
    #get current frame
    ret, current_frame = webcam_video_stream.read()
    current_frame_small = cv2.resize(current_frame,(0,0),fx=0.25, fy=0.25)
    all_face_locations = face_recognition.face_locations(current_frame_small, number_of_times_to_upsample=2,model='hog')
    
    all_face_encodings = face_recognition.face_encodings(current_frame_small, all_face_locations)
    all_face_names = []
    

    #loop through each face location and face encornings found in the unknown image

    for current_face_location, current_face_encoding in zip(all_face_locations, all_face_encodings):
        #split tuple
        top_pos,right_pos,left_pos,bottom_pos = current_face_location  
        top_pos = top_pos*4
        right_pos= right_pos*4
        bottom_pos=bottom_pos*4
        left_pos=left_pos*4
        all_matches = face_recognition.compare_faces(known_face_encodings,current_face_encoding)
        name_of_person = 'Unknown face'
        
        if True in all_matches:
            first_march_index = all_matches.index(True)
            name_of_person = known_face_names[first_march_index]
        #draw rectabel
        cv2.rectangle(current_frame, (left_pos,top_pos),(right_pos,bottom_pos),(255,255,255),2)
        
        #display the name as the text in the image
        font = cv2.FONT_HERSHEY_DUPLEX 
        cv2.putText((current_frame), name_of_person, (left_pos,bottom_pos), font, 0.5, (0,0,0),1)
        #display the image
        cv2.imshow("Faces Identified", current_frame)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #looping through face locations
    for current_face_location, current_face_encoding in zip(all_face_locations, all_face_encodings):
        #split tuple
        top_pos,right_pos,left_pos,bottom_pos = current_face_location 
        #correct the co-ordinate location to the change while resizing the 1/4 size inside the loop
        top_pos = top_pos*4
        right_pos= right_pos*4
        bottom_pos=bottom_pos*4
        left_pos=left_pos*4
        
        cv2.rectangle(current_frame, (left_pos,top_pos),(right_pos,bottom_pos),(0,0,255),2)
    cv2.imshow("Webcam Video", current_frame)
    
    
    
    
    
    
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
webcam_video_stream.release()
cv2.destroyAllWindows()