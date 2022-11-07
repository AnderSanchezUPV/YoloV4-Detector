import Detector as Dt
from pypylon import pylon
import cv2
import numpy as np 
import time 

def init_basler_cam():
                            ####        Conexion camera GIGE        ####
    # conecting to the first available camera
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

    # Grabing Continusely (video) with minimal delay
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
    converter = pylon.ImageFormatConverter()

    # converting to opencv bgr format
    # converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputPixelFormat = pylon.PixelType_Mono8
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    return camera, converter

def basler_camera_test(Model_path,Params_path):

    session,AnchorBoxes=Dt.Initialize_YoloV4(Model_path)
    Model_params=Dt.Get_params(Params_path,'yoloV4Params')

    camera, converter = init_basler_cam()
    while camera.IsGrabbing():
        try:
            grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

            if grabResult.GrabSucceeded():
                # Access the image data
                image = converter.Convert(grabResult)
                img = image.GetArray()

                # Repeat greyScaleimg on 3 channels
                imgRGB=np.dstack((img,img,img))

                # Procces with YoloV4
                start=time.perf_counter()
                Bboxes, scores, labels = Dt.Detect(imgRGB,session,AnchorBoxes,Model_params)
                elapsed=time.perf_counter()-start
                print("Tiempo Inferencia: {:.4} ms".format(elapsed*1000))

                # Draw result 
                img=Dt.drawBoxes(img,Bboxes,scores,labels)

                cv2.imshow('Tool Camera', img)
                if cv2.waitKey(1) & 0xFF == 32:
                    camera.StopGrabbing()
                    cv2.destroyAllWindows()
                    break
        except:
            print('Error en loop principal')
            camera.StopGrabbing()
            cv2.destroyAllWindows()        
            break

    return None



def main(): # Proccesing image example

    Model_path=r'Model\YoloV4_Aruco.onnx'

    Params_path=r'Model\YoloV4params.mat'
    basler_camera_test(Model_path,Params_path)

if __name__ == "__main__":
    main()
