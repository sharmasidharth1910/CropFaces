import cv2
import matplotlib.pyplot as plt

#Detect multiple faces from the img provided and crop all the faces and save them as seperate files.
img = plt.imread("./group-photo.jpg")

#Find the faces in the provided image
#Use the haarcascade ml model to do so
detector = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
all_faces = detector.detectMultiScale(img, 1.3, 5)

#Cropping and saving all the faces found in the provided image
for idx, face in enumerate(all_faces, 1):
    x, y, w, h = face
    crop_face = img[y:y+h, x:x+w, :]
    plt.imsave(f'./image{idx}.jpg', crop_face)
    
print(str(len(all_faces)) + " faces found in the image and saved to the storage.")