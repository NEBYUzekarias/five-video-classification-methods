from data import DataSet

import time
import cv2
import pickle
# import the necessary packages
from imutils.video import FileVideoStream
from imutils.video import FPS
import argparse
import imutils

 
import grpc
import humanaction_pb2
import humanaction_pb2_grpc

import numpy as np

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-v", "--video", required=True,
#     help="path to input video file")
# args = vars(ap.parse_args())


# start the file video stream thread and allow the buffer to
# start to fill



# start the FPS timer
fps = FPS().start()



def generate_image(video):
    fvs = FileVideoStream(video).start()
    time.sleep(1.0)
    
    while fvs.more():
       
        frame = fvs.read()

           
        # cv2.imshow('my webcam', frame)   
        
        frame = cv2.resize(frame,(299,299), interpolation = cv2.INTER_CUBIC)
        frame = np.array(frame)
        frame =pickle.dumps(frame)
        yield humanaction_pb2.Chunk(Content=frame)
        
        # time.sleep(0.035)
        cv2.waitKey(1)
        fps.update()

def guide_record_route(stub , video):

    
    route_summary = stub.Classify(generate_image(video))
    for response in route_summary:
        print("-------------- RecordRoute --------------")

        print("%s: %s" % ( response.class1 , response.message1))
        print("%s: %s" % ( response.class2 , response.message2))
        print("%s: %s" % ( response.class3 , response.message3))
        print("%s: %s" % ( response.class4 , response.message4))
        print("%s: %s" % ( response.class5 , response.message5))
        print("%s: %s" % ( response.class6 , response.message6))
        print("%s: %s" % ( response.class7 , response.message7))
        print("%s: %s" % ( response.class8 , response.message8))
        print("%s: %s" % ( response.class9 , response.message9))
        print("%s: %s" % ( response.class10 , response.message10))
      
   




def run(video):

    with grpc.insecure_channel('localhost:50054') as channel:
        stub = humanaction_pb2_grpc.HumanActionStub(channel)
        
        print("-------------- RecordRoute --------------")
        guide_record_route(stub , video)
        

if __name__ == '__main__':
    run()
