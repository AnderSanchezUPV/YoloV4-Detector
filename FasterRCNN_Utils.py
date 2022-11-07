from keyword import iskeyword
import numpy as np

def xywhToX1Y1X2Y2(Boxes:np.array)-> np.array:
    Boxes[:,2]=Boxes[:,0]+Boxes[:,2]-1
    Boxes[:,3]=Boxes[:,1]+Boxes[:,3]-1
    return Boxes  

def X1Y1X2Y2Toxywh(Boxes:np.array)-> np.array:
    Boxes[:,2]=Boxes[:,2]-Boxes[:,0]+1
    Boxes[:,3]=Boxes[:,3]-Boxes[:,1]+1
    return Boxes

def clippBBox(Rois:np.array,imageSize:np.array)-> np.array:
    clippedRois=Rois.astype(np.float64)

    x1=clippedRois[:,0]
    y1=clippedRois[:,1]

    x2= clippedRois[:,0] + clippedRois[:,2] -1
    y2= clippedRois[:,1] + clippedRois[:,3] -1

    x1=np.where(x1<1,1,x1)
    y1=np.where(y1<1,1,y1)

    x2=np.where(x2>imageSize[1],imageSize[1],x2)
    y2=np.where(y2>imageSize[0],imageSize[0],y2)

    clippedRois=np.array([x1, y1, x2-x1+1, y2-y1+1]).T
    return clippedRois


def selectStrongestBbox(Bboxes,scores,labels=None,Threshold=0.5,DivByUnion=False,N=2000):
    isKept=np.full((Bboxes.shape[0]), True)

    # Bboxes Corners and areas
    area=Bboxes[:,2]*Bboxes[:,3]
    x1 = Bboxes[:, 0]  # x coordinate of the top-left corner
    x2 = Bboxes[:, 0]+Bboxes[:,2]  # y coordinate of the top-left corner
    y1 = Bboxes[:, 1]  # x coordinate of the bottom-right corner
    y2 = Bboxes[:, 1]+Bboxes[:,3] # y coordinate of the bottom-right corner

    # For each bbox i, suppress all surrounded bbox j where j>i and overlap
    # ratio is larger than overlapThreshold
    boxCount=0
    numOfBbox = Bboxes.shape[0]
    currentBox = 0
    for i in range(numOfBbox):
        currentBox=i
        status, boxCount = iDetermineLoopStatusTopK(N,i,boxCount,isKept)
        if status == 1:
            continue
        elif status == 0:
            break
        else:
            for j in range(i+1,numOfBbox):
                if not(isKept[j]):
                    continue
                width=np.minimum(x2[i],x2[j])-np.maximum(x1[i],x1[j])
                if width <= 0:
                    continue
                height=np.minimum(y2[i],y2[j])-np.maximum(y1[i],y1[j])
                if height <=0:
                    continue
                areaOfIntersect = width * height

                if DivByUnion:
                    overlapRatio = areaOfIntersect/(area[i]+area[j]-areaOfIntersect)
                else:
                    overlapRatio = areaOfIntersect/np.minimum(area[i],area[j])

                if overlapRatio > Threshold:
                    isKept[j] = False      
                

    # When the number of strongest boxes is reached, set the remainder to
    # false.
    isKept[currentBox+1:isKept.shape[0]]= False; 

    selectBboxes=Bboxes[isKept]
    selectedScores= scores[isKept]

    if labels.any() != None: 
        selectedlabels=labels[isKept]
    else:
        selectedlabels=[]


    return selectBboxes, selectedScores, selectedlabels

def iDetermineLoopStatusTopK(N,i,boxCount,isKept):
    if isKept[i]:
        if boxCount < N:
            boxCount = boxCount + 1
        if boxCount == N:
            status = 0
        else:
            status = 2     
    
    else:
        status=1

    return status, boxCount