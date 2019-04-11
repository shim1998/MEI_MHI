from skimage.measure import compare_ssim
import numpy as np
import imutils
import cv2
import pandas as pd

huarray=[]

def image_resize(path, width = None, height = None, inter = cv2.INTER_AREA):
    image=cv2.imread(path,0)
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def writefile(data,path,column):
    df=pd.DataFrame(np.array(data,dtype="object"),columns=column)
    with open(path,'w+') as f:
        df.to_csv(f,mode='w',header=False)

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

pixels=[]
for i in range(3):
    for j in range(13):
        for k in range(3):
            createMEIsandMHIs(i+1,j+1,k+1)
            image1=image_resize("output/MEI%d%d%d.jpg"%(i+1,j+1,k+1),height=120,width=90)
            cv2.imwrite("data/MEIr%d%d%d.jpg"%(i+1,j+1,k+1),image1)
            print(image1.shape)
            pixels.append([i]+list(image1.flatten()))
            image2=image_resize("output/MHI%d%d%d.jpg"%(i+1,j+1,k+1),height=120,width=90)
            cv2.imwrite("data/MHIr%d%d%d.jpg"%(i+1,j+1,k+1),image2)
writefile(pixels,"data/dataset.csv",['Labels']+[0 for i in range(6030)])