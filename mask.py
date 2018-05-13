import os, cv2

data_folder_path='./data/'
l=os.listdir(data_folder_path)
for sub_folders in l:
	files=os.listdir(data_folder_path+sub_folders+'/')
	for file in files:
		img = cv2.imread(data_folder_path+sub_folders+'/'+file)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.GaussianBlur(img, (7,7), 3)
		img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
		ret, new = cv2.threshold(img, 25, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
		cv2.imwrite('data_folder_path+sub_folders+'/'+file',new)