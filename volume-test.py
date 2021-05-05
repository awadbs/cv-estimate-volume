
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from matplotlib import cm
from colorspacious import cspace_converter
from collections import OrderedDict

import copy
import open3d as o3d
import pyvista as pv


def load_image(filepath):
    """Loads an image into a numpy array.
    Note: image will have 3 color channels [r, g, b]."""
    img = np.float32(Image.open(filepath))
    #img = ImageOps.grayscale(img)
    return cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype('uint8')


sift = cv.SIFT_create() # sift detector uses difference of gaussian for feature detection

# get feature matches between two images using SIFT
def get_point_matches(img1, img2):
    """Returns matches as array: (feature track, image, coord)"""

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=10)
    search_params = dict(checks=100)
    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.95*n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
        
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)
    # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]
    
    pt1 = np.append(pts1[0],1)
    pt2 = np.append(pts2[0],1)
    
    return np.stack((pts1, pts2), axis=1), pts1, pts2, F

# combine feature matches
def combine_matches(matches_a, matches_b):
    """Assumes that the 0'th image is the same between them."""
    combined_matches = []
    for ii in range(matches_a.shape[0]):
        ma0 = matches_a[ii, 0]
        # Find the match in b
        mi = np.where((matches_b[:, 0] == ma0).all(axis=1))[0]

        # If a match is found, add to the array
        if mi.size > 0:
            ma = matches_a[ii]
            mb = matches_b[int(mi[0])]
            combined_matches.append(np.concatenate(
                (ma, mb[1:]), axis=0))

    return np.array(combined_matches)


# In[4]:


def visualize_matches(img_a, img_b, matches, ax=None):
    #if ax is None:
    #    fig = plt.figure(figsize=(20,20))
    #    ax = plt.gca()
    
    sa = img_a.shape
    sb = img_b.shape
    sp = 40
    off = sa[1]+sp
    
    merged_imgs = np.zeros(
        (max(sa[0], sb[0]), sa[1]+sb[1]+sp),
        dtype=np.float)
    merged_imgs[0:sa[0], 0:sa[1]] = img_a
    merged_imgs[0:sb[0], sa[1]+sp:] = img_b
    #ax.imshow(merged_imgs)
    
    #for m in matches:
    #    ax.plot([m[0][0], m[1][0]+off], [m[0][1], m[1][1]], 'r', alpha=0.5)

def get_match_colors(image_c, combined_matches):
    colors = []
    nm = combined_matches.shape[0]
    for mi in range(nm):
        m = combined_matches[mi, 0, :]
        colors.append(image_c[m[1]-1:m[1]+2,
                              m[0]-1:m[0]+2].sum(axis=0).sum(axis=0)/9)
    
    return colors

def rectify_two(test1,test2,pts1,pts2, F):
    h1,w1 = test1.shape
    h2,w2 = test2.shape
    _, H1, H2 = cv.stereoRectifyUncalibrated(np.float32(pts1), np.float32(pts2), F, imgSize=(w1,h1))
    
    img1_rectified = cv.warpPerspective(test1, H1, (w1,h1))
    img2_rectified = cv.warpPerspective(test2, H2, (w2,h2))
    
    #fig, axes = plt.subplots(1,2, figsize=(15, 10))
    #axes[0].imshow(img1_rectified, cmap="gray")
    #axes[1].imshow(img2_rectified, cmap='gray')
    #axes[0].axhline(400)
    #axes[1].axhline(400)
    #axes[0].axhline(600)
    #axes[1].axhline(600)
    return img1_rectified, img2_rectified

def set_points(three_images_arr, i, depth_map):
    if len(three_images_arr) != 3:
        raise ValueError('Array must have 3 images')
    test1 = cv.imread(three_images_arr[0],0)
    test2 = cv.imread(three_images_arr[1],0)
    test3 = cv.imread(three_images_arr[2],0)
    matches1, pts1,pts2,F = get_point_matches(test1, test2)
    matches2, pts1_1,pts2_1,F_1 = get_point_matches(test1, test3)
    
    #visualize_matches(test1, test2, matches1, ax=None)
    #visualize_matches(test1, test3, matches2, ax=None)
    
    img1_rectified, img2_rectified = rectify_two(test1,test2,pts1,pts2, F)
    img1_rectified, img3_rectified = rectify_two(test1,test3,pts1_1,pts2_1, F_1)
    
    combined_matches = combine_matches(np.array(matches1), np.array(matches2)) 
    colors = get_match_colors(load_image(three_images_arr[0]), combined_matches)

    return combined_matches, colors, img1_rectified, img2_rectified, img3_rectified 

def get_depth(img1_rectified, img2_rectified):
    # CALCULATE DISPARITY (DEPTH MAP)
        block_size = 11
        min_disp = -128
        max_disp = 128
        num_disp = max_disp - min_disp
        uniquenessRatio = 5
        speckleWindowSize = 0
        speckleRange = 2
        disp12MaxDiff = 0

        stereo = cv.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            uniquenessRatio=uniquenessRatio,
            speckleWindowSize=speckleWindowSize,
            P1=8 * 1 * block_size * block_size,
            P2=32 * 1 * block_size * block_size,
        )

        disp = stereo.compute(np.uint8(img1_rectified), np.uint8(img2_rectified)).astype(np.float32)
        disp = cv.normalize(disp, disp, alpha=255, beta=0, norm_type=cv.NORM_MINMAX)
        disp = np.uint8(disp)
        #fig = plt.figure()
        #plt.imshow(disp)
        
def get_depth_2(img1_rectified, img2_rectified):
    stereoSGBM = cv.StereoSGBM_create(minDisparity=0, 
                                   numDisparities = 64,
                                   blockSize = 3,
                                   P1 = 8*1*3*3, 
                                   P2 = 32*1*3*3
                                  )
    dispSGBM = stereoSGBM.compute(img1_rectified, img2_rectified).astype(np.float32) / 16
    #fig1 = plt.figure()
    #plt.imshow(dispSGBM, 'gray')
    #plt.colorbar()
    
    
    stereoBM = cv.StereoBM_create(numDisparities=16, blockSize=15)
    dispBM = stereoBM.compute(img1_rectified, img2_rectified)

    #fig2 = plt.figure()
    #plt.imshow(dispBM, 'gray')
    #plt.colorbar()


#######################################################################################

#For this example we assume that all views have the same depth map. For other non-uniform solids we would use a separate depth map for each view.
depth_map = cv.imread('./world_rotate/depth_map_world.png')
depth_map = cv.cvtColor(depth_map, cv.COLOR_BGR2GRAY)


ulti_x = []
ulti_y = []
ulti_z = []

three_images =  [['./world_rotate/trans_01.png','./world_rotate/trans_02.png','./world_rotate/trans_03.png'],
                ['./world_rotate/trans_03.png','./world_rotate/trans_04.png','./world_rotate/trans_05.png'],
                 ['./world_rotate/trans_05.png','./world_rotate/trans_06.png','./world_rotate/trans_07.png'],
                 ['./world_rotate/trans_07.png','./world_rotate/trans_08.png','./world_rotate/trans_09.png'],
                 ['./world_rotate/trans_09.png','./world_rotate/trans_10.png','./world_rotate/trans_11.png'],
                 ['./world_rotate/trans_11.png','./world_rotate/trans_12.png','./world_rotate/trans_13.png'],
                 ['./world_rotate/trans_13.png','./world_rotate/trans_14.png','./world_rotate/trans_15.png'],
                 ['./world_rotate/trans_15.png','./world_rotate/trans_16.png','./world_rotate/trans_17.png'],
                 ['./world_rotate/trans_17.png','./world_rotate/trans_18.png','./world_rotate/trans_19.png'],
                 ['./world_rotate/trans_19.png','./world_rotate/trans_20.png','./world_rotate/trans_21.png'],
                 ['./world_rotate/trans_21.png','./world_rotate/trans_22.png','./world_rotate/trans_23.png'],
                 ['./world_rotate/trans_23.png','./world_rotate/trans_24.png','./world_rotate/trans_25.png']]

fig100 = plt.figure()
ax = plt.axes(projection='3d')
cmap_op = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn','viridis', 'plasma', 'inferno', 'magma', 'cividis']
for cmap_option, three_images_coords in enumerate(three_images):
    X_matches, colors, img1_rectified, img2_rectified, img3_rectified = set_points(three_images_coords, 0, depth_map)

    
    #get_depth(img1_rectified, img2_rectified)
    #get_depth_2(img1_rectified, img2_rectified)
    #get_depth_2(img2_rectified, img3_rectified)


    #We attempted to reproject the feature matches into the world view without success!
    #We blame the difference in coordinate systems between Blender and Python. 
    #Also the bpy library would not load for us.


    #Camera measurements in Blender
    #distance from lens to surface of sphere = 9.5104
    #distance from principal point to surface of sphere = 10.6558
    #15 degrees in radians is 0.261799

    #translation when rotating 15 degrees around the z-axis
    #x = 9.5144*cos(0.261799) = 9.1863419329
    #y = 9.5144*sin(0.261799) = 2.46250435877

    #9.51066800099 = baseline

    b = 35.9 
    f = 188.9 

    # Camera matrix
    K = [[f, 0, img1_rectified.shape[0]/2],
         [0, f, img1_rectified.shape[1]/2],
         [0, 0, 1]]
    R = np.eye(3)
    ta = [[0],[0],[1]]
    Pa = K @ np.concatenate((R, ta), axis=1)

    R = [[np.cos(15),-np.sin(15),0],
         [np.sin(15),np.cos(15),0],
         [0,0,1]]     
    tb = [[34.720032502], [9.3071030883], [0]] 
    Pb = K @ np.concatenate((R, tb), axis=1)


    def solve_point_triangulation(proj_points, proj_matrices):
        # First we build the matrix
        D = np.zeros((2*len(proj_points), 4), dtype=float)
        for ii, (p, P) in enumerate(zip(proj_points, proj_matrices)):
            D[2*ii + 0] = p[1] * P[2] - P[1]
            D[2*ii + 1] = P[0] - p[0] * P[2]

        # Now, solve
        u, s, vh = np.linalg.svd(D, full_matrices=False)
        X = vh[np.argmin(s)]
        return X/X[3]

    pa = X_matches[:,0].tolist()
    pb = X_matches[:,1].tolist()

    X0_rec = np.empty([len(pa),3])
    for pta,ptb in zip(pa,pb):
        pt = solve_point_triangulation([pta,ptb], [Pa, Pb])[:-1]
        X0_rec = np.append(X0_rec,np.array([pt]),axis=0)

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(X0_rec[:, 0], X0_rec[:, 1], X0_rec[:, 2])
    #plt.show()


    from skimage import filters

    src = cv.imread('./world_rotate/depthmap.png',0)
    #fig2 = plt.figure()
    #plt.imshow(src)

    test1 = cv.imread(three_images_coords[0],0)
    width = test1.shape[0]
    height = test1.shape[1]
    dim = (width, height)

    # resize image
    src = cv.resize(src, dim, interpolation = cv.INTER_AREA)
    height, width = src.shape

    pt_cloud = []
    x_center = src.shape[0]
    y_center = src.shape[1]
    foc_l = 188.9763779528

    im_1_pts = X_matches[:,0]
    im_2_pts = X_matches[:,1]
    im_3_pts = X_matches[:,2]

    ims = [im_1_pts, im_2_pts, im_3_pts]

    img_pts = []
    for imj in ims:
        X_s = []
        Z_s = []
        Y_s = []

        u_coords = imj[:,0]
        v_coords = imj[:,1]
        for u,v in zip(u_coords, v_coords):
                Y = src[u][v]*foc_l

                Z = (v - x_center) * src[u][v] / (src[u][v] / foc_l)
                X = (u - y_center) * src[u][v] / (src[u][v] / foc_l)

                Z_s.append(Z)
                X_s.append(X)
                Y_s.append(Y)
        img_pts.append(list(zip(X_s, Y_s, Z_s)))


    from scipy.spatial.transform import Rotation as R

    for i in range(0,len(img_pts)):
        
        angle = -15*((cmap_option*2) + i)
        rotation_degrees = angle
        rotation_radians = np.radians(rotation_degrees)
        rotation_axis = np.array([0, 0, 1])
        rotation_vector = rotation_radians * rotation_axis
        rotation = R.from_rotvec(rotation_vector)

        for j,pt in enumerate(img_pts[i]):
            if(cmap_option == 0 and j == 0):
                continue
            img_pts[i][j] = rotation.apply(pt)

    im_1_pts = img_pts[0]
    im_2_pts = img_pts[1]
    im_3_pts = img_pts[2]


    X_s, Y_s, Z_s = [a_tuple[0] for a_tuple in im_1_pts], [a_tuple[1] for a_tuple in im_1_pts], [a_tuple[2] for a_tuple in im_1_pts]
    X_s_1, Y_s_1, Z_s_1 = [a_tuple[0] for a_tuple in im_2_pts], [a_tuple[1] for a_tuple in im_2_pts], [a_tuple[2] for a_tuple in im_2_pts]
    X_s_2, Y_s_2, Z_s_2 = [a_tuple[0] for a_tuple in im_3_pts], [a_tuple[1] for a_tuple in im_3_pts], [a_tuple[2] for a_tuple in im_3_pts]
    #Map out the point cloud here.
    if(cmap_option==0):
        ax.scatter3D(X_s,Y_s,Z_s,c=Z_s, cmap=cmap_op[cmap_option])
    ax.scatter3D(X_s_1,Y_s_1,Z_s_1, cmap=cmap_op[cmap_option + 1])
    ax.scatter3D(X_s_2,Y_s_2,Z_s_2, cmap=cmap_op[cmap_option + 2])
    for x,x1,x2 in zip(X_s,X_s_1,X_s_2):
        
        ulti_x.append(x)
        ulti_x.append(x1)
        ulti_x.append(x2)
    
    for y,y1,y2 in zip(Y_s,Y_s_1,Y_s_2):
        
        ulti_y.append(y)
        ulti_y.append(y1)
        ulti_y.append(y2)

    for z,z1,z2 in zip(Z_s,Z_s_1,Z_s_2):

        ulti_z.append(z)
        ulti_z.append(z1)
        ulti_z.append(z2)
plt.show()
plt.close(fig100)

ulti_pts = np.array(list(zip(ulti_x,ulti_y,ulti_z)))


import matplotlib.tri as mtri
from scipy.spatial import Delaunay

tri = Delaunay(ulti_pts) # points: np.array() of 3d points 

fig55 = plt.figure()
ax = fig55.add_subplot(1, 1, 1, projection='3d')
ax.plot_trisurf(ulti_pts[:,0], ulti_pts[:,1], ulti_pts[:,2], triangles=tri.simplices, cmap=plt.cm.Spectral)
plt.show()
plt.close(fig55)

from scipy.spatial import ConvexHull
import mpl_toolkits.mplot3d as a3
import matplotlib as mpl
import scipy as sp

hull = ConvexHull(ulti_pts)
indices = hull.simplices
faces = ulti_pts[indices]

print(' Hull volume: ', hull.volume)

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')

#ax.dist = 30
#ax.azim = -140

#ax.set_xlabel('x')
#ax.set_ylabel('y')
#ax.set_zlabel('z')
#ax.set_xlim3d([-750, 750])
#ax.set_ylim3d([-750, 750])
#ax.set_zlim3d([-750, 750])

#for f in faces:
#    f = np.array(f)
#    f[:,2] = f[:,2] - 600
#    face = a3.art3d.Poly3DCollection([f.tolist()])
#    face.set_color(mpl.colors.rgb2hex(sp.rand(3)))
#    face.set_edgecolor('k')
#    face.set_alpha(0.5)
#    ax.add_collection3d(face)

#plt.show()
#plt.close(fig)

from operator import itemgetter

def triangleVol(p1, p2, p3):
    pts = [p1,p2,p3]
    p1 = max(pts, key=itemgetter(2))
    new_pts = []
    for i in pts:
        if i[0] != p1[0] or i[1] != p1[1] or i[2] != p1[2]:
            new_pts.append(i)
    pts = new_pts
    N = np.linalg.norm(np.cross((pts[0] - p1), (pts[1] - p1)));
    PV = np.linalg.norm([0,0,0] - p1);
    if(np.dot( PV, N ) > 0.0 ):
        p2 = pts[0]
        p3 = pts[1]
    else:
        p2 = pts[1]
        p3 = pts[0]
    
    v321 = p3[0]*p2[1]*p1[2];
    v231 = p2[0]*p3[1]*p1[2];
    v312 = p3[0]*p1[1]*p2[2];
    v132 = p1[0]*p3[1]*p2[2];
    v213 = p2[0]*p1[1]*p3[2];
    v123 = p1[0]*p2[1]*p3[2];
    return (1.0/6.0)*(-v321 + v231 + v312 - v132 - v213 + v123);


def meshVol(faces):
    vols = []
    for t in faces:
        vols.append(triangleVol(t[0], t[1], t[2]));
    return abs(sum(vols));

calc_vol = meshVol(faces)
print("Calculated volume is :", calc_vol)

print("Difference in volume: ", abs(hull.volume - calc_vol))





