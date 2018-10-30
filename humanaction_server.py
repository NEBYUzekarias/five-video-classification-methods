import grpc
from data import DataSet

import humanaction_pb2
import humanaction_pb2_grpc

from concurrent import futures


# from data import DataSet

import time
# import cv2
from extractor import Extractor
from keras.models import Model, load_model
from keras.layers import Input
import numpy as np
import pickle


_ONE_DAY_IN_SECONDS = 60 * 60 * 24




class HumanActionServicer(humanaction_pb2_grpc.HumanActionServicer):

    def Classify(self, request_iterator, context):
      saved_model = 'data/checkpoints/lstm-features.037-0.131.h5'

      point_count = 0
      seq_length = 40
      class_limit = 10  # Number of classes to extract. Can be 1-101 or None for all.
      data = DataSet(seq_length=seq_length, class_limit=class_limit)
      modelE = Extractor()
      
      model = load_model(saved_model)
      sequence = []
      for Chunk in request_iterator:
            byt = Chunk.Content
            byt = pickle.loads(byt)
            features = modelE.extract(byt)

            sequence.append(features)
            point_count += 1
            if point_count == 40:
              print( np.shape(sequence))
              prediction = model.predict(np.expand_dims(sequence, axis=0))
              print(prediction)
              message= []
              classs = []
              sorted_lps = data.print_class_from_prediction(np.squeeze(prediction, axis=0))
              if sorted_lps is not None:
                  for i, class_prediction in enumerate(sorted_lps):
                      if i > 10 - 1 or class_prediction[1] == 0.0:
                          break
                      print("%s: %.2f" % (class_prediction[0], class_prediction[1]))
                      message.append(class_prediction[1])
                      classs.append(class_prediction[0])
                      first = class_prediction[0]
                      v1 = class_prediction[1]
      
              yield humanaction_pb2.label(message1 = message[0],
                                          message2 = message[1],
                                          message3 = message[2],
                                          message4 = message[3],
                                          message5 = message[4],
                                          message6 = message[5],
                                          message7 = message[6],
                                          message8 = message[7],
                                          message9 = message[8],
                                          message10 = message[9],
                                          class1 = classs[0],
                                          class2 = classs[1],
                                          class3 = classs[2],
                                          class4 = classs[3],
                                          class5 = classs[4],
                                          class6 = classs[5],
                                          class7 = classs[6],
                                          class8 = classs[7],
                                          class9 = classs[8],
                                          class10 = classs[9]
                                          )
              sequence = []
              point_count =0






def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    humanaction_pb2_grpc.add_HumanActionServicer_to_server(
        HumanActionServicer(), server)
    server.add_insecure_port('[::]:50054')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()
