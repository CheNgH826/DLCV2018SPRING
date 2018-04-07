import cv2
import scipy.misc

img = scipy.misc.imread('Suburb/image_0029.jpg')
# surf = cv2.SURF(400)
surf = cv2.xfeatures2d.SURF_create(6000)
kp, des = surf.detectAndCompute(img, None)
    
print(len(kp))
print(des.shape)

# img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
# img2 = cv2.drawKeypoints(img,kp,None)
# scipy.misc.imsave('img_keypoint.jpg', img2)