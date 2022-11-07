import Matlab_utils as Mu
import cv2
import onnxruntime as ort
import numpy as np
import scipy.io as scipio
import os

from FasterRCNN_Utils import selectStrongestBbox
import time


def Initialize_YoloV4(Model_path):
    
    session = ort.InferenceSession(Model_path,providers=["CUDAExecutionProvider"])
    AnchorBoxes=np.array([[50,50],[55,55]])

    return session,AnchorBoxes

def drawBoxes(img:np.array,Bboxes:np.array,scores:np.array,labels:np.array)->np.array:
    for i in range(len(Bboxes)):
        start_point=tuple(Bboxes[i,0:2].astype(int))
        end_point=tuple((Bboxes[i,0:2]+Bboxes[i,2:4]).astype(int))
        text_point=tuple((Bboxes[i,0:2]-np.array([0,25])).astype(int)) 
        text='{:}:{:.2}'.format(labels[i][0][0],scores[i])         
        img=cv2.rectangle(img,start_point,end_point,(255,0,0),2)
        img = cv2.putText(img, text, text_point,cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255,0,0), 1, cv2.LINE_AA)
    return img

def rescale(inMatrix)-> np.array: # rescale between 0 and 1
    inmin=np.min(inMatrix)
    inmax=np.max(inMatrix)
    l=0 #lower bound
    u=1 #upper bound

    outMatrix=l+(inMatrix-inmin)/(inmax-inmin)*(u-l)
    #Mu.save2matlab(outMatrix,'pyScaledMatrix')
    return outMatrix

def preProccesYoloV4(img:np.array)->np.array:
    img=img.astype(np.single)
    img=rescale(img)
    return img

def Gen_Features(Tensor1:np.matrix,Tensor2:np.matrix,anchorBoxes:np.matrix)->np.matrix:
    # Procces Tensors    
    Features1= ProccesTensor(Tensor1)
    Features2= ProccesTensor(Tensor2)

    # Merge Predictions
    Features=np.vstack((Features1,Features2))    
    
    return Features

def ProccesTensor(inTensor:np.matrix):
    # Init Tensor
    ProccessedTensor=np.empty((inTensor.shape[0],8),dtype=object)
    # Properties
    inTensor=OnnxTensor2Matrix(inTensor) #HxWx32 matrix
    numChannelsPred=inTensor.shape[2]
    numAnchors=2
    numPredElemsPerAnchors = numChannelsPred/numAnchors
    channelsPredIdx = np.arange(1,numChannelsPred)
    predictionIdx=np.ones((numAnchors*5),dtype=int)

    # X Positions
    startIdx = 0
    endIdx = numChannelsPred
    stride = numPredElemsPerAnchors
    ind=np.arange(startIdx,endIdx,stride).astype(int)
    ProccessedTensor[0,1]=sigmoid(inTensor[:,:,ind])
    predictionIdx=np.hstack((predictionIdx,ind))

    # Y Positions
    startIdx = 1
    endIdx = numChannelsPred
    stride = numPredElemsPerAnchors
    ind=np.arange(startIdx,endIdx,stride).astype(int)
    ProccessedTensor[0,2]=sigmoid(inTensor[:,:,ind])
    predictionIdx=np.hstack((predictionIdx,ind))

    # Width
    startIdx = 2
    endIdx = numChannelsPred
    stride = numPredElemsPerAnchors
    ind=np.arange(startIdx,endIdx,stride).astype(int)
    ProccessedTensor[0,3]=np.exp(inTensor[:,:,ind])
    ProccessedTensor[0,6]=inTensor[:,:,ind]
    predictionIdx=np.hstack((predictionIdx,ind))

    # Height
    startIdx = 3
    endIdx = numChannelsPred
    stride = numPredElemsPerAnchors
    ind=np.arange(startIdx,endIdx,stride).astype(int)
    ProccessedTensor[0,4]=np.exp(inTensor[:,:,ind])
    ProccessedTensor[0,7]=inTensor[:,:,ind]
    predictionIdx=np.hstack((predictionIdx,ind))
    # Score
    startIdx = 4
    endIdx = numChannelsPred
    stride = numPredElemsPerAnchors
    ind=np.arange(startIdx,endIdx,stride).astype(int)
    ProccessedTensor[0,0]=sigmoid(inTensor[:,:,ind])
    predictionIdx=np.hstack((predictionIdx,ind))

    # Class #
    classidx=np.setdiff1d(channelsPredIdx,predictionIdx)
    ProccessedTensor[0,5]=sigmoid(inTensor[:,:,classidx])

    # Expand
    #ProccessedTensor=np.hstack((ProccessedTensor,ProccessedTensor[:,3:5]))

    # activation f(x)

    
    return ProccessedTensor

def sigmoid(inMat:np.matrix):
    outMat=1/(1 + np.exp(-inMat))
    return outMat

def anchorBoxesGen(anchorBoxes,subBoxes,Model_Params):
    tiledAnchors=np.empty(subBoxes.shape,dtype=np.matrix)
    NetworkInputSize=Model_Params['NetworkInputSize'][0][0]
    n=1
    for i in range(subBoxes.shape[0]):
        h,w,_=subBoxes[i,1].shape
        lin1=np.arange(0,h)
        lin2=np.arange(0,w)
        lin3=np.arange(0,anchorBoxes.shape[0])
        lin4=np.arange(0,n)
        outBox=np.meshgrid(lin2,lin1,lin3,lin4)
        tiledAnchors[i,0]=outBox[0]
        tiledAnchors[i,1]=outBox[1]

        outanchor=np.meshgrid(lin2,lin1,anchorBoxes[:,0],lin4)

        tiledAnchors[i,2]=outanchor[2]
        tiledAnchors[i,3]=outanchor[2]
    
    for i in range(subBoxes.shape[0]):
        h,w,_=subBoxes[i,1].shape
        tiledAnchors[i,0]=(np.squeeze(tiledAnchors[i,0])+subBoxes[i,0])/w
        tiledAnchors[i,1]=(np.squeeze(tiledAnchors[i,1])+subBoxes[i,1])/h
        tiledAnchors[i,2]=np.multiply(np.squeeze(tiledAnchors[i,2]),subBoxes[i,2])/NetworkInputSize[0,1]
        tiledAnchors[i,3]=np.multiply(np.squeeze(tiledAnchors[i,3]),subBoxes[i,3])/NetworkInputSize[0,0]



    return tiledAnchors

def reshapePredictions(Predictions):
    reshPredictions=np.empty(Predictions.shape,dtype=object)
    x,y =Predictions.shape
    for ii in range(x):
        for jj in range(y):
            pred=Predictions[ii,jj]
            h,w,c = pred.shape 
            _p=np.reshape(pred,(h*w*c,1),order='F')
            reshPredictions[ii,jj]=_p
            
    return reshPredictions

def reshapeClassesMat(classesMat,classes):
    reshclassesMats=np.empty(classesMat.shape,dtype=object)
    x =classesMat.shape[0]
    numClasses=classes.size
    n=1
    for ii in range(x):        
        pred=classesMat[ii,0]
        h,w,c = pred.shape
        numanchors = int(c/numClasses)
        _p=np.reshape(pred,(h*w,numClasses,numanchors,n),order='F')
        _p=np.swapaxes(_p,1,2)
        h,w,c,_ = _p.shape
        _p
        _p=np.reshape(_p,(h*w,c,n),order='F')
        reshclassesMats[ii,0]=_p


    return reshclassesMats

def cell2mat(detections):
    Matdetections=np.zeros((12352,16),dtype=np.float32)
    x,y=detections.shape
    idx=[0,2496,9856+2496]
    idy=[0,1,2,3,4,5,16]
    for ii in range(x):
        for jj in range(y):
            if jj!=5:          
                Matdetections[idx[ii]:idx[ii+1],idy[jj]:idy[jj+1]]=detections[ii,jj]
            elif jj==5:
                Matdetections[idx[ii]:idx[ii+1],idy[jj]:idy[jj+1]]=np.squeeze(detections[ii,jj])
    return Matdetections

def postproccesYoloV4(Features:np.matrix,anchorBoxes,Model_Params)->tuple((np.array,np.array,np.array)):
    classes=Model_Params['Classes'][0][0]
    subBoxes=Features[:,1:5]
    subBoxes=anchorBoxesGen(anchorBoxes,subBoxes,Model_Params)
    Features[:,1:5]=subBoxes
    # Apply following post processing steps to filter the detections:
    # * Filter detections based on threshold.
    # * Convert bboxes from spatial to pixel dimension.

    # Combine the prediction from different heads.
    detections=np.empty((2,6),dtype=object)
    Predictions=Features[:,0:5]
    Predictions=reshapePredictions(Predictions)

    classesMat=Features[:,5:6]
    classesMat=reshapeClassesMat(classesMat,classes)

    detections=np.hstack((Predictions,classesMat))
    detections[[0,1]]=detections[[1,0]] # solo depuracion. invertir orden de filas para equiparar a matlab. 
    # Detections 0 to 6 
    # remap to 12352x16
    
    detections=cell2mat(detections)

    classprob=np.amax(detections[:,5:16],axis=1)
    classIdx=np.argmax(detections[:,5:16],axis=1)

    detections[:,0]=np.multiply(detections[:,0],classprob)
    detections[:,5]=classIdx

    Threshold=Model_Params['Threshold'][0][0][0]
    ThresholdMask=np.greater_equal(detections[:,0],Threshold)
    detections=detections[ThresholdMask]

    Bboxes,scores,labels=postproccesDetections(detections,classes,Model_Params)
    

    return Bboxes,scores,labels

def postproccesDetections(detections,classes,Model_Params):

    scorePred=detections[:,0]
    bboxestmp=detections[:,1:5]
    classpred=detections[:,5]

    inputImageSIze=np.array([512,612])
    scale=np.array([612,512,612,512])
    bboxestmp=np.multiply(bboxestmp,scale)

    # Convert x and y position of detections from centre to top-left.
    # Resize boxes to image size.   
    bboxPred = ConvertCenterToTopLeft(bboxestmp)

    bboxPred,scorePred,classpred=BoxFilterFcn(bboxPred,scorePred,classpred,Model_Params)

    #Apply NMS
    Bboxes,scores,labels = selectStrongestBbox(bboxPred,scorePred,classpred)
    
    #Limit width
    detectionsWd= np.minimum(Bboxes[:,0]+Bboxes[:,2],inputImageSIze[1])
    Bboxes[:,2]=detectionsWd-Bboxes[:,0]

    #Limit Height
    detectionsHt= np.minimum(Bboxes[:,1]+Bboxes[:,3],inputImageSIze[0])
    Bboxes[:,3]=detectionsHt-Bboxes[:,1]


    Bboxes[np.less(Bboxes,1)]=1

    # Cast classesId to classNames
    labels=classes[labels.astype(int)]
    
    return Bboxes,scores,labels

def BoxFilterFcn(Rois:np.array,scores:np.array,labels,params)->np.array:
    MinSize=params['MinSize'][0][0][0]
    MaxSize=params['MaxSize'][0][0][0]

    Rois,scores,labels =filterSmallBBoxes(MinSize, Rois, scores,labels)
    Rois,scores,labels =filterLargeBBoxes(MaxSize, Rois, scores,labels)

    return Rois, scores,labels

def filterSmallBBoxes(MinSize:np.array, Rois:np.array, scores:np.array,labels):
    subRois=Rois[:,[3,2]]
    
    temptooSmall=np.less(subRois,MinSize)
    tooSmall=np.any(temptooSmall,axis=1)

    Rois=np.delete(Rois,tooSmall,axis=0)
    scores=np.delete(scores,tooSmall,axis=0)
    labels=np.delete(labels,tooSmall,axis=0)
    return Rois, scores, labels

def filterLargeBBoxes(MaxSize:np.array, Rois:np.array, scores:np.array,labels):
    subRois=Rois[:,[3,2]]
    
    temptooBig=np.greater(subRois,MaxSize)
    tooBig=np.any(temptooBig,axis=1)

    Rois=np.delete(Rois,tooBig,axis=0)
    scores=np.delete(scores,tooBig,axis=0)
    labels=np.delete(labels,tooBig,axis=0)
    return Rois, scores, labels

def ConvertCenterToTopLeft(bbox):
    bbox[:,0]=bbox[:,0]-bbox[:,2]/2+0.5
    bbox[:,1]=bbox[:,1]-bbox[:,3]/2+0.5
    return bbox

def cv2OnnxTensor(img):
    img=np.swapaxes(img,0,2)
    img=np.swapaxes(img,1,2)
    img=np.expand_dims(img, axis=0)
    img=np.float32(img)
    return img

def OnnxTensor2Matrix(Mat:np.matrix):
    Mat=np.squeeze(Mat) # remove batch dimension
    Mat=np.swapaxes(Mat,0,2)
    Mat=np.swapaxes(Mat,0,1)
    
    #Mu.save2matlab(Mat,'PyMat')
    return Mat

def Detect(img:np.matrix,session,anchorBoxes:np.matrix,Model_Params):
     
    # Preporcess Image
    img=preProccesYoloV4(img)
    img=cv2OnnxTensor(img)
    out1,out2 =session.run(list(map(lambda output: output.name, session.get_outputs())), {session.get_inputs()[0].name: img})
    
    # Generate Features
    Features=Gen_Features(out1,out2,anchorBoxes)
    # Post Process Output
    Bboxes,scores,labels=postproccesYoloV4(Features,anchorBoxes,Model_Params)
       
    #return Bboxes.astype(int), scores, labels
    return Bboxes, scores, labels

def Get_params(Path:str,Filename:str):
    _container=scipio.loadmat(Path)
    params=_container[Filename]
    return params

def single_Test(img:np.matrix,Model_path:str,Params_path:str):

    session,AnchorBoxes=Initialize_YoloV4(Model_path)
    Model_params=Get_params(Params_path,'yoloV4Params')
    start=time.perf_counter
    Bboxes, scores, labels = Detect(img,session,AnchorBoxes,Model_params)
    elapsed=time.perf_counter-start
    print("Inferencia: {:.2} ms".format(elapsed))

    img=drawBoxes(img,Bboxes,scores,labels)

    cv2.imshow('Test',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def time_Test(img:np.matrix,Model_path:str,Params_path:str):
    session,AnchorBoxes=Initialize_YoloV4(Model_path)
    Model_params=Get_params(Params_path,'yoloV4Params')

    timeArray=np.zeros(1000)
    for i in range (1000):
        start=time.time()
        Bboxes, scores, labels = Detect(img,session,AnchorBoxes,Model_params)
        elapsed=time.time()-start
        timeArray[i]=elapsed
    
    meanTime=np.mean(timeArray[1:len(timeArray.T)])
    maxTime=np.max(timeArray[1:len(timeArray.T)])
    medianTIme=np.median(timeArray[1:len(timeArray.T)])
    #print(1000*timeArray[1:len(timeArray)])
    print('Tiempo Medio:{:.3f} ms'.format(meanTime*1000))
    print('Tiempo Maximo:{:.3f} ms'.format(maxTime*1000))
    print('Tiempo Mediano:{:.3f} ms'.format(medianTIme*1000))

def DataStoreTest(DSpath:str,Model_path:str,Params_path:str):

    imgDS=os.listdir(DSpath)
    session,AnchorBoxes=Initialize_YoloV4(Model_path)
    Model_params=Get_params(Params_path,'yoloV4Params')

    for imgFile in imgDS:
        img=cv2.imread(os.path.join(DSpath,imgFile))

        start=time.perf_counter()
        Bboxes, scores, labels = Detect(img,session,AnchorBoxes,Model_params)
        elapsed=time.perf_counter()-start
        print("Inferencia: {:.4} ms".format(elapsed*1000))

        img=drawBoxes(img,Bboxes,scores,labels)

        #time.sleep(0.15)

        cv2.imshow('Test',img)
        if cv2.waitKey(1) & 0xFF == 32:
            cv2.destroyAllWindows()
            break

    cv2.destroyAllWindows()

   
    

def main(): # Proccesing image example
    # Load image
    #img=cv2.imread(r'C:\Users\dani\Documents\CoMAr\Custom Object Detector\Test images\RosBag_img_00004625.bmp')
  
    Model_path=r'Model\YoloV4_Aruco.onnx'

    Params_path=r'Model\YoloV4params.mat'

    DSpath=r'C:\Users\CoMAr\Documents\CoMAr Images\BBDD 10_06_2022 Imagenes'

    #single_Test(img,Model_path,Params_path)
    
    #time_Test(img,Model_path,Params_path)
    
    DataStoreTest(DSpath,Model_path,Params_path)



   

    #img=drawBoxes(img,Bboxes,scores,labels)

  

if __name__ == "__main__":
    main()
