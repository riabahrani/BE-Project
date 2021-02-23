from flask import Flask, render_template, request
import json
from flask_cors import CORS
import numpy as np
import cv2 as cv                         # Library for image processing
from math import floor
import keyboard


frame_width=1366
frame_height=768
ID = 0
offset=0
app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/shirt.html')
def plot():
    return render_template('shirt.html')
@app.route('/pant.html')
def ploty():
    return render_template('pant.html')
@app.route('/predict', methods=['GET','POST'])
def predict():
    
    shirtno = int(request.form["shirt"])
    # pantno = int(request.form["pant"])

    cv.waitKey(1)
    cap=cv.VideoCapture(0)
    ih=shirtno
    # i=pantno
    while True:
        imgarr=["shirt1.png",'shirt2.png','shirt51.jpg','shirt6.png']
        

        #ih=input("Enter the shirt number you want to try")
        imgshirt = cv.imread(imgarr[ih-1],1) #original img in bgr
        if ih==3:
            shirtgray = cv.cvtColor(imgshirt,cv.COLOR_BGR2GRAY) #grayscale conversion
            ret, orig_masks_inv = cv.threshold(shirtgray,200 , 255, cv.THRESH_BINARY) #there may be some issues with image threshold...depending on the color/contrast of image
            orig_masks = cv.bitwise_not(orig_masks_inv)

        else:
            shirtgray = cv.cvtColor(imgshirt,cv.COLOR_BGR2GRAY) #grayscale conversion
            ret, orig_masks = cv.threshold(shirtgray,0 , 255, cv.THRESH_BINARY) #there may be some issues with image threshold...depending on the color/contrast of image
            orig_masks_inv = cv.bitwise_not(orig_masks)
        origshirtHeight, origshirtWidth = imgshirt.shape[:2]
        
        # imgarr=["pant7.jpg",'pant21.png']
        # #i=input("Enter the pant number you want to try")
        # imgpant = cv.imread(imgarr[i-1],1)
        # imgpant=imgpant[:,:,0:3]#original img in bgr
        # pantgray = cv.cvtColor(imgpant,cv.COLOR_BGR2GRAY) #grayscale conversion
        # if i==1:
        #     ret, orig_mask = cv.threshold(pantgray,100 , 255, cv.THRESH_BINARY) #there may be some issues with image threshold...depending on the color/contrast of image
        #     orig_mask_inv = cv.bitwise_not(orig_mask)
        # else:
        #     ret, orig_mask = cv.threshold(pantgray,50 , 255, cv.THRESH_BINARY)
        #     orig_mask_inv = cv.bitwise_not(orig_mask)
        # origpantHeight, origpantWidth = imgpant.shape[:2]

        face_cascade=cv.CascadeClassifier('haarcascade_frontalface_default.xml')

        ret,img=cap.read()
       
        img_w = img.shape[0]
        img_h = img.shape[1]
        # img_w = int(width*0.75)
        # img_h = int(height*0.75)
        # img = cv.resize(img[:,:,0:3],(1000,1000), interpolation = cv.INTER_AREA)
        cv.namedWindow("img",cv.WINDOW_NORMAL)
        # cv.setWindowProperty('img',cv.WND_PROP_FULLSCREEN,cv.cv.CV_WINDOW_FULLSCREEN)
        cv.resizeWindow("img", frame_width, frame_height)
        gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

        for (x, y, w, h) in faces:
                    cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 1)

                    face_w = w
                    face_h = h
                    face_x1 = x
                    face_x2 = face_x1 + face_w
                    face_y1 = y
                    face_y2 = face_y1 + face_h

                # set the shirt size in relation to tracked face
                    shirtWidth = int(2.9 * face_w+ offset)
                    shirtHeight = int((shirtWidth * origshirtHeight / origshirtWidth)+offset/3)
                    cv.putText(img,(str(shirtWidth)+" "+str(shirtHeight)),(x+w,y+h),cv.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)

                    shirt_x1 = face_x2 - int(face_w / 2) - int(shirtWidth / 2)                             # setting shirt centered wrt recognized face
                    shirt_x2 = shirt_x1 + shirtWidth
                    shirt_y1 = face_y2 + 5                                                       # some padding between face and upper shirt. Depends on the shirt img
                    shirt_y2 = shirt_y1 + shirtHeight


                     # Check for clipping
                    if shirt_x1 < 0:
                        shirt_x1 = 0
                    if shirt_y1 < 0:
                        shirt_y1 = 0
                    if shirt_x2 > img_w:
                        shirt_x2 = img_w
                    if shirt_y2 > img_h:
                        shirt_y2 = img_h

                    shirtWidth = shirt_x2 - shirt_x1
                    shirtHeight = shirt_y2 - shirt_y1
                    if shirtWidth < 0 or shirtHeight < 0:
                        continue

                # Re-size the original image and the masks to the shirt sizes
                    shirt = cv.resize(imgshirt, (shirtWidth, shirtHeight),interpolation=cv.INTER_AREA)
                    mask = cv.resize(orig_masks, (shirtWidth, shirtHeight), interpolation=cv.INTER_AREA)
                    mask_inv = cv.resize(orig_masks_inv, (shirtWidth, shirtHeight), interpolation=cv.INTER_AREA)
                
                # take ROI for shirt from background equal to size of shirt image
                    roi = img[shirt_y1:shirt_y2, shirt_x1:shirt_x2]

                # roi_bg contains the original image only where the shirt is not
                # in the region that is the size of the shirt.
                    roi_bg = cv.bitwise_and(roi, roi, mask=mask_inv)
                    roi_fg = cv.bitwise_and(shirt, shirt, mask=mask)
                    dst = cv.add(roi_bg, roi_fg)
                    img[shirt_y1:shirt_y2, shirt_x1:shirt_x2] = dst


                    # kernel = np.ones((5, 5), np.float32) / 25
                    # imgshirtt = cv2.filter2D(dst, -1, kernel)

                    # if face_y1 + shirtHeight +face_h< frame_height:
                    #     #cv2.putText(frame, "press 'n' key for next item and 'p' for previous item", (x, y),cv2.FONT_HERSHEY_COMPLEX, .8, (255, 255, 255),1)
                    #     img[shirt_y1:shirt_y2, shirt_x1:shirt_x2] = dst

                    # else:
                    #     text = 'Too close to Screen'
                    #     #cv2.putText(frame, "press 'n'  key for next item and 'p' for previous item", (x-200, y-200),cv2.FONT_HERSHEY_COMPLEX, .8, (255, 255, 255), 1)
                    #     cv.putText(img, text, (int(face_x1-face_w/4.3), int(face_y1)), cv.FONT_HERSHEY_COMPLEX, 1,(0, 0, 250), 1)

                    # if keyboard.is_pressed('m' or 'M'):
                    #     ID= 0

                    # if keyboard.is_pressed('W' or 'w'):
                    #     ID= 1

                    # if keyboard.is_pressed('i'):
                    #     if offset>100:
                    #         print("THIS IS THE MAX SIZE AVAILABLE")
                    #     else:
                    #         offset+=50
                    #         print('+ pressed')

                    # if keyboard.is_pressed('d'):
                    #     if offset <0:
                    #         print("THIS IS THE MIN SIZE AVAILABLE")
                    #     else:
                    #         offset -= 50
                    #         print('- pressed')






           
            # # Re-size the original image and the masks to the shirt sizes
            # shirt = cv.resize(imgshirt, (shirtWidth,shirtHeight), interpolation = cv.INTER_AREA) #resize all,the masks you made,the originla image,everything
            # mask = cv.resize(orig_masks, (shirtWidth,shirtHeight), interpolation = cv.INTER_AREA)
            # masks_inv = cv.resize(orig_masks_inv, (shirtWidth,shirtHeight), interpolation = cv.INTER_AREA)
            # # take ROI for shirt from background equal to size of shirt image
            # rois = img[y1s:y2s, x1s:x2s]
            #     # roi_bg contains the original image only where the shirt is not
            #     # in the region that is the size of the shirt.
            # num=rois
            # roi_bgs = cv.bitwise_and(rois,num,mask = masks_inv)
            # # roi_fg contains the image of the shirt only where the shirt is
            # roi_fgs = cv.bitwise_and(shirt,shirt,mask = mask)
            # # join the roi_bg and roi_fg
            # dsts = cv.add(roi_bgs,roi_fgs)
            # img[y1s:y2s, x1s:x2s] = dsts # place the joined image, saved to dst back over the original image
            # #print "blurring"
            
            # break
        cv.imshow("img",img)
        #cv.setMouseCallback('img',change_dress)
        if cv.waitKey(100) == ord('q'):
            break

    cap.release()                           # Destroys the cap object
    cv.destroyAllWindows()                 # Destroys all the windows created by imshow

    return render_template('index.html')
if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True,port=5000)


 
# '000078_1.jpg',' 000182_1.jpg',' 000259_1.jpg',' 000260_1.jpg','000333_1.jpg','000333_1.jpg','000483_1.jpg','000496_1.jpg','001425_1.jpg','001432_1.jpg','001471_1.jpg','019358_1.jpg'
 			     
			     
                         
                         
                           
                           
                           
                          
                           
                           

					