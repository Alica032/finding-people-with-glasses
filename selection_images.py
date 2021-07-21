import os
import cv2
from shutil import copyfile

from detection_people_in_glasses import find
from settings import SCRAPER_FOLDER, RESULTS_FOLDER


if __name__ == '__main__':
	for name in os.listdir(SCRAPER_FOLDER):
		try:
			path = os.path.join(SCRAPER_FOLDER, name)
			img = cv2.imread(path)
			property_img = find(img)
			if len(property_img) != 1:
				continue

			w_face, h_face, judge = property_img[0]
			if w_face >= 256 and h_face>=256 and judge:
				copyfile(path, os.path.join(RESULTS_FOLDER, name))
		except Exception:
			pass