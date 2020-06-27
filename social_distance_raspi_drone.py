# USAGE
# python social_distance_raspi_drone.py --model MobileNetSSD_deploy.caffemodel --prototxt MobileNetSSD_deploy.prototxt --ip <IP ADDRESS> --port 8000

# import the necessary packages
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from imutils.video import FPS
from flask import render_template
from detection import CentroidTracker
from scipy.spatial import distance as dist
from itertools import combinations
from collections import OrderedDict
import datetime
import numpy as np
import threading
import argparse
import imutils
import time
import cv2
import os
import math

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
outputFrame = None
lock = threading.Lock()
CLASSES = ["background", "plane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "table",
	"dog", "horse", "motorcycle", "person", "plant", "sheep",
	"sofa", "train", "screen"]

tracker = CentroidTracker(maxDisappeared=40, maxDistance=50)

# initialize a flask object
app = Flask(__name__)

# initialize the video stream and allow the camera sensor to
# warmup
print("[INFO] starting video stream...")
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
fps = FPS().start()
app = Flask(__name__)

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")
	
def non_max_suppression_fast(boxes, overlapThresh):
    try:
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")
    except Exception as e:
        print("Exception occurred in non_max_suppression : {}".format(e))

def social_distancing(frameCount):
	# grab global references to the video stream, output frame, and
	# lock variables
	global vs, outputFrame, lock

	# initialize the motion detector and the total number of frames
	# read thus far
	total_frames = 0

	# loop over frames from the video stream
	while True:
		frame = vs.read()
		frame = imutils.resize(frame, width=900)
		frame = cv2.flip(frame, 0)
		total_frames += 1

		(H, W) = frame.shape[:2]

		blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)

		net.setInput(blob)
		person_detections = net.forward()
		rects = []
		for i in np.arange(0, person_detections.shape[2]):
			confidence = person_detections[0, 0, i, 2]
			if confidence > 0.5:
				idx = int(person_detections[0, 0, i, 1])

				if CLASSES[idx] != "person":
					continue

				person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
				(startX, startY, endX, endY) = person_box.astype("int")
				rects.append(person_box)

		boundingboxes = np.array(rects)
		boundingboxes = boundingboxes.astype(int)
		rects = non_max_suppression_fast(boundingboxes, 0.3)
		centroid_dict = dict()
		objects = tracker.update(rects)
		for (objectId, bbox) in objects.items():
			x1, y1, x2, y2 = bbox
			x1 = int(x1)
			y1 = int(y1)
			x2 = int(x2)
			y2 = int(y2)
			cX = int((x1 + x2) / 2.0)
			cY = int((y1 + y2) / 2.0)


			centroid_dict[objectId] = (cX, cY, x1, y1, x2, y2)

			# text = "ID: {}".format(objectId)
			# cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

		red_zone_list = []
		for (id1, p1), (id2, p2) in combinations(centroid_dict.items(), 2):
			dx, dy = p1[0] - p2[0], p1[1] - p2[1]
			distance = math.sqrt(dx * dx + dy * dy)
			if distance < 75.0:
				if id1 not in red_zone_list:
					red_zone_list.append(id1)
				if id2 not in red_zone_list:
					red_zone_list.append(id2)

		for id, box in centroid_dict.items():
			if id in red_zone_list:
				cv2.rectangle(frame, (box[2], box[3]), (box[4], box[5]), (0, 0, 255), 2)
			else:
				cv2.rectangle(frame, (box[2], box[3]), (box[4], box[5]), (0, 255, 0), 2)





		# acquire the lock, set the output frame, and release the
		# lock
		with lock:
			outputFrame = frame.copy()
		
def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock

	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue

			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

			# ensure the frame was successfully encoded
			if not flag:
				continue

		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

	# check to see if this is the main thread of execution
if __name__ == '__main__':
	# construct the argument parser and parse command line arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--prototxt", required=True,
		help="path to Caffe 'deploy' prototxt file")
	ap.add_argument("-m", "--model", required=True,
		help="path to Caffe pre-trained model")
	ap.add_argument("-i", "--ip", type=str, required=True,
		help="ip address of the device")
	ap.add_argument("-o", "--port", type=int, required=True,
		help="ephemeral port number of the server (1024 to 65535)")
	ap.add_argument("-u", "--movidius", type=bool, default=0,
		help="boolean indicating if the Movidius should be used")
	ap.add_argument("-f", "--frame-count", type=int, default=32,
		help="# of frames used to construct the background model")
	args = vars(ap.parse_args())

	net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

	# start a thread that will perform motion detection
	t = threading.Thread(target=social_distancing, args=(
		args["frame_count"],))
	t.daemon = True
	t.start()

	# start the flask app
	app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=False)
	fps.update()
	fps.stop()
	print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
	# release the video stream pointer
	vs.stop()
