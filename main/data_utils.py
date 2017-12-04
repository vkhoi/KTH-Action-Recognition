import imageio
import numpy as np
import os
import pickle

from PIL import Image
from scipy.misc.pilutil import imresize

CATEGORIES = ["boxing", "handclapping", "handwaving", "jogging", "running", 
	"walking"]

# Dataset are divided according to the instruction at:
# http://www.nada.kth.se/cvap/actions/00sequences.txt
TRAIN_PEOPLE_ID = [11, 12, 13, 14, 15, 16, 17, 18]
DEV_PEOPLE_ID = [19, 20, 21, 23, 24, 25, 1, 4]
TEST_PEOPLE_ID = [22, 2, 3, 5, 6, 7, 8, 9, 10]

def make_raw_dataset(dataset="train"):
	X = []
	y = []

	if dataset == "train":
		ID = TRAIN_PEOPLE_ID
	elif dataset == "dev":
		ID = DEV_PEOPLE_ID
	else:
		ID = TEST_PEOPLE_ID

	for category in CATEGORIES:
		# Get all files in current category's folder.
		folder_path = os.path.join("..", "dataset", category)
		filenames = sorted(os.listdir(folder_path))

		for filename in filenames:
			filepath = os.path.join("..", "dataset", category, filename)

			# Get id of person in this video.
			person_id = int(filename.split("_")[0][6:])
			if person_id not in ID:
				continue

			vid = imageio.get_reader(filepath, "ffmpeg")

			# Add each frame to correct list.
			for frame_idx, frame in enumerate(vid):
				# Convert to grayscale.
				frame = Image.fromarray(np.array(frame))
				frame = frame.convert("L")
				frame = np.array(frame.getdata(), dtype=np.uint8).reshape((120, 160))
				frame = imresize(frame, (60, 80))

				X.append(frame)
				y.append(category)

	pickle.dump(X, open("data/X_%s.p" % dataset, "wb"))
	pickle.dump(y, open("data/y_%s.p" % dataset, "wb"))

def optical_flow():
	pass