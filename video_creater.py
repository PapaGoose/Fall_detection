import cv2
import os

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('videos/1.mp4', fourcc, 30.0, (640, 480))

files = os.listdir('images_for_video/1')

for image in files:
    img = cv2.imread('images_for_video/1/'+image)
    img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
    out.write(img)
    # cv2.imshow('fr', image)

out.release()
cv2.destroyAllWindows()