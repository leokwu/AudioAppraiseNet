# USAGE
# python3.6 inference_audioappraise.py --model audioappraise.model --le le.pickle --picture f0.png

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import pickle
import cv2
import datetime


def inference(config):
	startX = 20
	startY = 20
	# load the audioappraise  model and label encoder from disk
	print("[INFO] loading audioappraise ...")
	model = load_model(config.model)
	le = pickle.loads(open(config.le, "rb").read())

	# same manner as our training data
	frame = cv2.imread(config.picture)
	f0_frame = cv2.resize(frame, (224, 224))
	f0_frame = f0_frame.astype("float") / 255.0
	f0_frame = img_to_array(f0_frame)
	f0_frame = np.expand_dims(f0_frame, axis=0)

	# model to determine if the f0 is "good" or "bad"
	start_time = datetime.datetime.now()
	preds = model.predict(f0_frame)[0]
	end_time = datetime.datetime.now()
	print("\npredict time ms: ", (start_time - end_time).microseconds/1000)
	print("\npreds: ", model.predict(f0_frame)[0])
	j = np.argmax(preds)
	print("\nj: ", j)
	label = le.classes_[j]
	print("\nlabel: ", label)

	# draw the label and probability on the frame
	label = "{}: {:.4f}".format(label, preds[j])
	cv2.putText(frame, label, (startX, startY),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

	# show the output frame and wait for a key press
	cv2.imshow("Picture", frame)
	while True:
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break
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
