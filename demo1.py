from data import DataSet

import time
import cv2
 



from extractor import Extractor
from keras.models import Model, load_model
from keras.layers import Input
import numpy as np




def show_webcam(mirror=False):  

    
    # initialize the video stream and pointer to output video file, then
    # allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    writer = None
    #
    saved_model = 'data/checkpoints/lstm-features.037-0.131.h5'
    vs = cv2.VideoCapture(-1)
    time.sleep(2)
    # Set defaults.
    seq_length = 40
    class_limit = 10  # Number of classes to extract. Can be 1-101 or None for all.
    data = DataSet(seq_length=seq_length, class_limit=class_limit)
    # get the model.

    modelE = Extractor()
    model = load_model(saved_model)
    # loop over frames from the video file stream
    
    while True:
        # grab the frame from the threaded video stream
    
        first =""
        v1 =""

        sequence = []
        for i in range (0,40):
            
            ret_val,frame = vs.read()
            if ret_val == True:
                if mirror: 
                    frame = cv2.flip(frame, 1)
                width = np.size(frame, 1)
                height = np.size(frame, 0)
                x = width/2
                y = height/2
                cv2.imshow('my webcam', frame)
                cv2.putText(frame, first + v1, (x,y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,0,0), thickness=1)
                frame = cv2.resize(frame,(299,299), interpolation = cv2.INTER_CUBIC)
                if cv2.waitKey(1) == 27: 
                    break  # esc to quit
            else:
                break
            features = modelE.extract(frame)

            sequence.append(features)

               
            
        # Predict!
        print( np.shape(sequence))
        prediction = model.predict(np.expand_dims(sequence, axis=0))
        print(prediction)
        sorted_lps = data.print_class_from_prediction(np.squeeze(prediction, axis=0))
        for i, class_prediction in enumerate(sorted_lps):
            if i > 10 - 1 or class_prediction[1] == 0.0:
                break
            print("%s: %.2f" % (class_prediction[0], class_prediction[1]))
            first = class_prediction[0]
            v1 = class_prediction[1]





def main():
    show_webcam(mirror=True)


if __name__ == '__main__':
    main()