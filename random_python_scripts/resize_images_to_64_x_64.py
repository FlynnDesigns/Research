import cv2 
import os 
dir_images = 'A:\\Research\\Training_data\\GAN\\run_0\\images_fid_and_is\\10K_training_images\\'

for file in os.listdir(dir_images):
    raw_image = cv2.imread(dir_images + file)
    final_image = cv2.resize(raw_image, dsize=(64,64),  interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(dir_images + file, final_image)
