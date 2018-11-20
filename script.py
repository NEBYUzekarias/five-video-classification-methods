import humanaction_client
import argparse
from threading import Thread
import time 



ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
    help="path to input video file")
args = vars(ap.parse_args())



def imshoww():
	
	print("[INFO] starting video file thread...")
	fvs = FileVideoStream(args["video"]).start()
	time.sleep(1.0)
	
	# start the FPS timer
	fps = FPS().start()
	
	# loop over frames from the video file stream
	while fvs.more():
		# grab the frame from the threaded video file stream, resize
		# it, and convert it to grayscale (while still retaining 3
		# channels)
		frame = fvs.read()
		frame = imutils.resize(frame, width=450)
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		frame = np.dstack([frame, frame, frame])
	
		# display the size of the queue on the frame
		cv2.putText(frame, "Queue Size: {}".format(fvs.Q.qsize()),
			(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)	
	
		# show the frame and update the FPS counter
		cv2.imshow("Frame", frame)
		time.sleep(0.035)
		cv2.waitKey(1)
		fps.update()
	
	# stop the timer and display FPS information
	fps.stop()
	print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
	
	# do a bit of cleanup
	cv2.destroyAllWindows()
	fvs.stop()

thread = Thread(target = humanaction_client.run(args["video"]))
thread.daemon = True
thread.start()
Thread(target = imshoww).start()
