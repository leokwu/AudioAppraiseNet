# USAGE
# python3.6 batch_inference_classify.py --model audioappraise.model --le le.pickle --classify classify/f0 --select classify/wav --output output/good

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import pickle
import cv2
import datetime
from imutils import paths
import os

def inference(config):
	# load the audioappraise  model and label encoder from disk
	print("[INFO] loading audioappraise ...")
	model = load_model(config.model)
	le = pickle.loads(open(config.le, "rb").read())

	imagePaths = list(paths.list_images(config.classify))
	# print(imagePaths)
	for imagePath in imagePaths:
		# same manner as our training data

		frame = cv2.imread(imagePath)
		f0_frame = cv2.resize(frame, (128, 128))
		f0_frame = f0_frame.astype("float") / 255.0
		f0_frame = img_to_array(f0_frame)
		f0_frame = np.expand_dims(f0_frame, axis=0)

		# model to determine if the f0 is "good" or "bad"
		start_time = datetime.datetime.now()
		preds = model.predict(f0_frame)[0]
		end_time = datetime.datetime.now()
		# print("\npredict time ms: ", (start_time - end_time).microseconds/1000)
		# print("\npreds: ", model.predict(f0_frame)[0])
		j = np.argmax(preds)
		# print("\nj: ", j)
		label = le.classes_[j]
		# print("\nlabel: ", label)
		if label == 'bad':
			# print(imagePath)
			prename = imagePath.split(os.path.sep)[-1].split('.')[0]
			# print(prename)
			fullname = f'{config.select}/{prename}.wav'
			print(fullname)
			cmd = f'cp {fullname} {config.output}'
			os.system(cmd)

if __name__ == '__main__':
	# construct the argument parse and parse the arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("-m", "--model", type=str, required=True,
					help="path to trained model")
	parser.add_argument("-l", "--le", type=str, required=True,
					help="path to label encoder")
	parser.add_argument("-c", "--classify", type=str, required=True,
						help="path to classify")
	parser.add_argument("-s", "--select", type=str, required=True,
						help="path to be select")
	parser.add_argument("-o", "--output", type=str, required=True,
						help="path save good wav")
	# parser.add_argument("-p", "--picture", type=str, required=True,
	# 					help="path to f0 picture")
	config = parser.parse_args()
	inference(config)
