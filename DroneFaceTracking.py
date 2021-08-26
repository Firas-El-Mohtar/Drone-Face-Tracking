# we should make a class called Faceclass
# with attributes same as the Facedetect classs

#with processing method that changes the bgr to rgb , resieze and returns the img2,imgh
# with proccess method similar to the process method in Facedetect
# with draw method similar to that in drawing_utils
# with our own drawing method that returns the list of landmark coordinates


import cv2 as cv
import mediapipe as mp
import time
import numpy as np


class Faceclass:

    def __init__(self, min_detection_confidence=0.5):

        self.min_detection_confidence = min_detection_confidence

        self.mpFaceDetection = mp.solutions.face_detection
        #even object can be an attribute
        self.faceDetection = self.mpFaceDetection.FaceDetection(
            self.min_detection_confidence)
        self.mpDraw = mp.solutions.drawing_utils

    #method to change to rgb
    def processing(self, img):
        #change to rgb
        imgrgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        return imgrgb

    #method to resize
    def myresize(self, img, resizefactor):
        #resieze
        #get h,w,d

        himg, wimg, dimg = img.shape
        h, w = int(himg*resizefactor), int(wimg*resizefactor)
        # pay attention ! here width before height
        frame = cv.resize(img, (w, h))
        return frame, h, w

    #method to process the image
    def myprocess(self, img):
        imgrgb = self.processing(img)
        result = self.faceDetection.process(imgrgb)
        return result

    #method to draw automatic
    def draw(self, frame, result):
        if result.detections:
            for detection in result.detections:
                #draw using
                self.mpDraw.draw_detection(frame, detection)

    #method to draw myway
    def mydraw(self, frame, result, himg, wimg):

        if result.detections:
            for detection in result.detections:
                #bounding box object
                bbo = detection.location_data.relative_bounding_box
                xmin, ymin, width, height = bbo.xmin, bbo.ymin, bbo.width, bbo.height
                #print(xmin,ymin,width,height)
                #pixel x min ...
                #int to remove decimal points
                pxmin, pymin, pwidth, pheight = int(
                    xmin*wimg), int(ymin*himg), int(width*wimg), int(height*himg)
                #bounding box list
                bblist = [pxmin, pymin, pwidth, pheight]
                center = [int(pxmin+pwidth//2), int(pymin + pheight//2)]
                cv.rectangle(frame, (pxmin, pymin), (pxmin+pwidth,
                                                     pymin+pheight), (0, 0, 255), thickness=3)
                canva = np.zeros_like(frame)

                cv.rectangle(canva, (pxmin, pymin), (pxmin+pwidth,
                                                     pymin+pheight), (255, 0, 0), thickness=-1)

                frame = cv.addWeighted(frame, 0.8, canva, 0.5, 1)

                #cv.circle(frame,center,5,(0,0,255),thickness=-1)
                cv.line(frame, (int(pxmin+pwidth//2)-4, int(pymin + pheight//2)),
                        (int(pxmin+pwidth//2)+4, int(pymin + pheight//2)), (0, 0, 255), thickness=1)
                cv.line(frame, (int(pxmin+pwidth//2), int(pymin + pheight//2)-4),
                        (int(pxmin+pwidth//2), int(pymin + pheight//2)+4), (0, 0, 255), thickness=1)

                return center, frame

    def follow(self, center, imgh, imgw):
        lr, fb, ud, y = 0, 0, 0, 0
        tolerance = 30
        linearspeed = 15
        #calculate horizantal error
        errorh = center[0] - imgw//2
        # ----- imgc ---- facecenterme ---> + --> go to my right
        # ----- facecenterme ---- imgcent ---> -v ---> go to my left
        if errorh < 0 and abs(errorh) > tolerance:
            #go to my left
            lr, fb, ud, y = linearspeed, 0, 0, 0
            print("go to my left")

        elif errorh > 0 and abs(errorh) > tolerance:
            #go to my rigth
            lr, fb, ud, y = -linearspeed, 0, 0, 0
            print("go to my right")
        else:
            #stay stable
            print("stay stable")
            lr, fb, ud, y = 0, 0, 0, 0

        return lr, fb, ud, y


if __name__ == "__main__":

    #video
    cap = cv.VideoCapture(0)
    #object form our class
    detect = Faceclass()

    while True:
        #read frame
        success, frame = cap.read()
        #change size
        frame, himg, wimg = detect.myresize(frame, 0.5)

        #process
        result = detect.myprocess(frame)
        #draw
        #detect.draw(frame,result)
        #mydraw
        if result.detections:
            facecenter, frame = detect.mydraw(frame, result, himg, wimg)

        cv.imshow("frame", frame)

        cv.waitKey(1)
