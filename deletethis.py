
# Starter code (run this first)
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from scipy.spatial.transform import Rotation as R
from math import pi ,sin, cos
import math


rotation_degrees = 5
rotation_radians = np.radians(rotation_degrees)
rotation_axis = np.array([0, 0, 1])

rotation_vector = rotation_radians * rotation_axis
rotation = R.from_rotvec(rotation_vector)

def load_image(filepath):
    """Loads an image into a numpy array.
    Note: image will have 3 color channels [r, g, b]."""
    img = np.float32(Image.open(filepath))
    #img = ImageOps.grayscale(img)
    return cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
    #return (np.asarray(img).astype('uint8')/255)[:, :, :3]

depth_map = cv.imread('./world_rotate/depthmap.png',0)
print(depth_map.shape)

height, width = depth_map.shape

pt_cloud = []
x_center = 250
y_center = 250
foc_l = .05
X_s = []
Z_s = []
Y_s = []
for u in range(0,height):
    for v in range(0,width):
        Y = depth_map[u][v]
                
        Z = (v - x_center) * depth_map[u][v] / (depth_map[u][v] / foc_l)
        X = (u - y_center) * depth_map[u][v] / (depth_map[u][v] / foc_l)
            
        
        Z_s.append(Z)
        X_s.append(X)
        Y_s.append(Y)

    

fig2 = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X_s,Y_s,Z_s,c=Z_s,cmap='Greens')
Y_s = np.multiply(Y_s,-1)
ax.scatter3D(X_s,Y_s,Z_s,c=Z_s, cmap='Blues')
plt.show()
