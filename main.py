from skimage.measure import compare_ssim
import numpy as np
import imutils
import cv2
import pandas as pd

huarray=[]

def writefile(data,path):
    df=pd.DataFrame(np.array(data,dtype="object"),columns=['Label','h[0]','h[1]','h[2]','h[3]','h[4]','h[5]','h[6]'])
    with open(path,'w+') as f:
        df.to_csv(f,mode='a',header=False)

def videoplay():
    cap=cv2.VideoCapture(videofile)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def calculatehumoments(image1,image2,label):
    lst=[]
    MEIarray=list(cv2.HuMoments(cv2.moments(image1)).flatten())
    MHIarray=list(cv2.HuMoments(cv2.moments(image2)).flatten())
    lst.append(label)
    for i in MEIarray:
        lst.append(i)
    huarray.append(lst)

def createMEIsandMHIs(i,j,k):
    cap=cv2.VideoCapture('input/PS7A%dP%dT%d.mp4'%(i,j,k))
    firstFrame=None
    width,height=cap.get(3),cap.get(4)
    image1 = np.zeros((int(height), int(width)), np.uint8)
    image2 = np.zeros((int(height), int(width)), np.uint8)
    ctr=1
    while True:
        ret,frame=cap.read()
        if frame is None:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if firstFrame is None:
            firstFrame = gray
            continue
        frameDelta = cv2.absdiff(firstFrame, gray)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cv2.imshow('frame',thresh)
        image1=cv2.add(image1,thresh)
        image2=cv2.addWeighted(image2,1,thresh,ctr/1000,0)
        ctr+=1
        if cv2.waitKey(1)& 0xFF == ord('q'):
            break
    cv2.imwrite("output/MEI%d%d%d.jpg"%(i,j,k),image1)
    cv2.imwrite("output/MHI%d%d%d.jpg"%(i,j,k),image2)
    calculatehumoments(image1,image2,i)
    cap.release()
    cv2.destroyAllWindows()

# for i in range(3):
#     for j in range(3):
#         for k in range(3):
#             createMEIsandMHIs(i+1,j+1,k+1)

for j in range(3):
    for k in range(3):
        createMEIsandMHIs(j+1,5,k+1)

    writefile(huarray,"output/mei.csv")
