import numpy as np
from scipy.io import savemat
import os

def PyTensor2MatTensor(PyTensor:np.array): 
    _Tensor=np.swapaxes(PyTensor,2,1)
    _Tensor=np.swapaxes(_Tensor,3,2)
    _Tensor=np.squeeze(_Tensor)
    matTensor=np.ascontiguousarray(_Tensor)
    return matTensor

def save2matlab(matrix:np.array,name:str):
    folder_path=r'D:\CoMAr Data\CoMAr BBox esay\CustomYoloV4\Results'
    full_name=name+'.mat'
    full_path=os.path.join(folder_path,full_name)
    mdic = {name: matrix}
    savemat(full_path, mdic)

def reshapeBoxDeltas(inBoxes:np.array)-> np.array:
    # in shape (H,W,Anchors,4 )
    # reshape to (HxWxAnchors,4 )
    H=inBoxes.shape[0]
    W=inBoxes.shape[1]
    Anch=inBoxes.shape[2]
    elements=H*W*Anch
    reshapedBoxes=np.zeros([elements,4])
    cont=0
    for ii in range(Anch):
        for jj in range(W):
            for kk in range(H):
                reshapedBoxes[cont,:]=inBoxes[kk,jj,ii,:]
                cont=cont+1

    return reshapedBoxes


def reshapeClsScores(inClsScores:np.array,numAnchors:np.int0)->np.array:
    # in shape (H,W,4 )
    # reshape to (HxWxAnchors,4 )
    ClsScores=inClsScores[:,:,0:numAnchors]
    H=ClsScores.shape[0]
    W=ClsScores.shape[1]
    Anch=ClsScores.shape[2]
    elements=H*W*Anch
    reshapedScores=np.zeros([elements])
    cont=0
    for ii in range(Anch):
        for jj in range(W):
            for kk in range(H):
                reshapedScores[cont]=ClsScores[kk,jj,ii]
                cont=cont+1


    return reshapedScores



