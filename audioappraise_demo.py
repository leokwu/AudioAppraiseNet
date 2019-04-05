# USAGE
# python3.6 audioappraise_demo.py --model audioappraise.model --le le.pickle --picture f0.png

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os



def inference(config):
	startX = 20
	startY = 20
	endX = 80
	endY = 80
	# load the audioappraise  model and label encoder from disk
	print("[INFO] loading audioappraise ...")
	model = load_model(config.model)
	le = pickle.loads(open(config.le, "rb").read())

	# same manner as our training data
	frame = cv2.imread(config.picture)
	f0_frame = cv2.resize(frame, (32, 32))
	f0_frame = f0_frame.astype("float") / 255.0
	f0_frame = img_to_array(f0_frame)
	f0_frame = np.expand_dims(f0_frame, axis=0)

	# model to determine if the f0 is "good" or "bad"
	preds = model.predict(f0_frame)[0]
	j = np.argmax(preds)
	label = le.classes_[j]

	# draw the label and bounding box on the frame
	label = "{}: {:.4f}".f0_frame(label, preds[j])
	cv2.putText(frame, label, (startX, startY - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

	# show the output frame and wait for a key press
	cv2.imshow("Picture", frame)
	key = cv2.waitKey(1) & 0xFF

	# do a bit of cleanup
	cv2.destroyAllWindows()

if __name__ == '__main__':
	# construct the argument parse and parse the arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("-m", "--model", type=str, required=True,
					help="path to trained model")
	parser.add_argument("-l", "--le", type=str, required=True,
					help="path to label encoder")
	parser.add_argument("-p", "--picture", type=str, required=True,
						help="path to f0 picture")
	config = parser.parse_args()
	inference(config)
