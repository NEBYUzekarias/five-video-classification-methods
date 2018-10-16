from data import DataSet

import time
import cv2
import pickle

 
import grpc
import humanaction_pb2
import humanaction_pb2_grpc

import numpy as np
vs = cv2.VideoCapture(-1)
time.sleep(2)

def generate_image():
    
    
    while True:
       
        ret_val,frame = vs.read()
        if ret_val == True:

           
            cv2.imshow('my webcam', frame)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        else:
            break
        frame = cv2.resize(frame,(299,299), interpolation = cv2.INTER_CUBIC)
        frame = np.array(frame)
        frame =pickle.dumps(frame)
        yield humanaction_pb2.Chunk(Content=frame)

def guide_record_route(stub):

    
    route_summary = stub.Classify(generate_image())
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
      
   




def run():

    with grpc.insecure_channel('localhost:50055') as channel:
        stub = humanaction_pb2_grpc.HumanActionStub(channel)
        
        print("-------------- RecordRoute --------------")
        guide_record_route(stub)
        

if __name__ == '__main__':
    run()
