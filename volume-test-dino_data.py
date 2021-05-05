#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Starter code (run this first)
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from matplotlib import cm
from colorspacious import cspace_converter
from collections import OrderedDict
import os
cmaps = OrderedDict()


# In[2]:


def load_image(filepath):
    """Loads an image into a numpy array.
    Note: image will have 3 color channels [r, g, b]."""
    img = np.float32(Image.open(filepath))
    #img = ImageOps.grayscale(img)
    return cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
    #return (np.asarray(img).astype('uint8')/255)[:, :, :3]


# In[3]:


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
    
    #print("Epipolar constraint is ", epi_cnst)
    
    #pt1 = np.append(pts1[0],1)
    #pt2 = np.append(pts2[0],1)
    #epi_cnst = np.transpose(pt2)@F@pt1
    
    #f = 0.05
    #f = 188.98
    #K1 = [[f, 0, img1.shape[0]/2],
    # [0, f, img1.shape[1]/2],
    # [0, 0, 1]]
    
    #K2 = [[f, 0, img1.shape[0]/2 + 34.72],
    # [0, f, img1.shape[1]/2 + 9.31],
    # [0, 0, 1]]
    
    #tb = [[34.720032502], [9.3071030883], [0]] 
    
    #E_mat = np.transpose(K2)@F@K1
    #u, s, vh = np.linalg.svd(E_mat, full_matrices=False)
    #newE = u@np.eye(3)@np.transpose(vh)
    #u, s, vh = np.linalg.svd(newE, full_matrices=False)
    
    #W = np.eye(3)
    
    #R1 = u@W@np.transpose(vh)
    #R2 = u@np.transpose(W)@np.transpose(vh)
    #print("u is ", u)
    
    #u = np.array(u)
    #t1 = u[:,-1]
    #t2 = -u[:,-1]
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
    if ax is None:
        fig = plt.figure(figsize=(20,20))
        ax = plt.gca()
    
    sa = img_a.shape
    sb = img_b.shape
    sp = 40
    off = sa[1]+sp
    
    merged_imgs = np.zeros(
        (max(sa[0], sb[0]), sa[1]+sb[1]+sp),
        dtype=np.float)
    merged_imgs[0:sa[0], 0:sa[1]] = img_a
    merged_imgs[0:sb[0], sa[1]+sp:] = img_b
    ax.imshow(merged_imgs)
    
    for m in matches:
        ax.plot([m[0][0], m[1][0]+off], [m[0][1], m[1][1]], 'r', alpha=0.5)

def get_match_colors(image_c, combined_matches):
    colors = []
    nm = combined_matches.shape[0]
    for mi in range(nm):
        m = combined_matches[mi, 0, :]
        colors.append(image_c[m[1]-1:m[1]+2,
                              m[0]-1:m[0]+2].sum(axis=0).sum(axis=0)/9)
    
    return colors


# In[5]:


def rectify_two(test1,test2,pts1,pts2, F):
    png = "1.png"
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


# In[6]:


def set_points(three_images_arr):
    if len(three_images_arr) != 3:
        raise ValueError('Array must have 3 images')
     
    test1 = load_image(three_images_arr[0])
    test1 = cv.rotate(test1, cv.ROTATE_90_COUNTERCLOCKWISE)
    test1 = test1[:,:,0]
    test2 = load_image(three_images_arr[1])
    test2 = cv.rotate(test2, cv.ROTATE_90_COUNTERCLOCKWISE)
    test2 = test2[:,:,0]
    test3 = load_image(three_images_arr[2])
    test3 = cv.rotate(test3, cv.ROTATE_90_COUNTERCLOCKWISE)
    test3 = test3[:,:,0]
    #plt.imshow(test2[:, :, 0])


    #test1 = cv.imread(three_images_arr[0],0)
    #test2 = cv.imread(three_images_arr[1],0)
    #test3 = cv.imread(three_images_arr[2],0)

    matches1, pts1, pts2, F = get_point_matches(test1, test2)
    matches2, pts1_1, pts2_1, F_1 = get_point_matches(test1, test3)
    
    #visualize_matches(test1, test2, matches1, ax=None)
    #visualize_matches(test1, test3, matches2, ax=None)
    
    img1_rectified, img2_rectified = rectify_two(test1,test2,pts1,pts2, F)
    img1_rectified, img3_rectified = rectify_two(test1,test3,pts1_1,pts2_1, F_1)
    
    combined_matches = combine_matches(np.array(matches1), np.array(matches2)) 
    colors = get_match_colors(load_image(three_images_arr[0]), combined_matches)

    return combined_matches, colors, img1_rectified, img2_rectified, img3_rectified



def get_depth_2(img1_rectified, img2_rectified):
        stereoSGBM = cv.StereoSGBM_create(minDisparity=0, 
                                       numDisparities = 64,
                                       blockSize = 3,
                                       P1 = 8*1*3*3, 
                                       P2 = 32*1*3*3
                                      )
        dispSGBM = stereoSGBM.compute(img1_rectified, img2_rectified).astype(np.float32) / 16
        fig1 = plt.figure()
        plt.imshow(dispSGBM, 'gray')
        plt.colorbar()

        stereoBM = cv.StereoBM_create(numDisparities=16, blockSize=15)
        dispBM = stereoBM.compute(img1_rectified, img2_rectified)

        fig2 = plt.figure()
        plt.imshow(dispBM, 'gray')
        plt.colorbar()
        
        

def get_depth(img1_rectified, img2_rectified):
    # CALCULATE DISPARITY (DEPTH MAP)
        # Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
        block_size = 11
        min_disp = -128
        max_disp = 128
        # Maximum disparity minus minimum disparity. The value is always greater than zero.
        # In the current implementation, this parameter must be divisible by 16.
        num_disp = max_disp - min_disp
        # Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct.
        # Normally, a value within the 5-15 range is good enough
        uniquenessRatio = 5
        # Maximum size of smooth disparity regions to consider their noise speckles and invalidate.
        # Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
        speckleWindowSize = 0
        # Maximum disparity variation within each connected component.
        # If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16.
        # Normally, 1 or 2 is good enough.
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

        b = 9.51066800099
        f = 0.05

        #fig = plt.figure()
        #plt.imshow(disp)
    #get_depth(img1_rectified, img2_rectified)
    

#test1 = load_image('./world_rotate/trans_01.png')
#fig1 = plt.figure()
#plt.imshow(test1[:, :, 0])

#fig2 = plt.figure()
#test2 = load_image('./world_rotate/trans_02.png')
#plt.imshow(test2[:, :, 0])

#fig3 = plt.figure()
#test3 = load_image('./world_rotate/trans_03.png')
#plt.imshow(test3[:, :, 0])

#depth_map = cv.imread('./world_rotate/depth_map_world.png')
#depth_map = cv.cvtColor(depth_map, cv.COLOR_BGR2GRAY)

#fig1 = plt.figure()
#plt.imshow(depth_map)

#depth_map_nl = cv.imread('./world_rotate/depth_map_nolight.png')
#depth_map_nl = cv.cvtColor(depth_map_nl, cv.COLOR_BGR2GRAY)

#fig1 = plt.figure()
#plt.imshow(depth_map_nl)

file1 = open('./dinoSparseRing/dinoSparseRing/dinoSR_par.txt', 'r')
Lines = file1.readlines()
file1.close()

my_dict = {}
three_images = []
K_mats = []
R_mats = []
t_mats = []

count = 0
# Strips the newline character
triple_set = []
K_triple = []
R_triple = []
t_triple = []
############

for i,line in enumerate(Lines[2:]):
    line = line.split(' ')
    line = [i.strip() for i in line]
    triple_set.append(line[0])
    line = line[1:]
    line = [float(i) for i in line]
    
    K_mat = [[line[0], line[1],line[2]], 
              [line[3], line[4],line[5]],
              [line[6], line[7], line[8]]]
    K_triple.append(K_mat)
    
    R_mat = [[line[9], line[10],line[11]], 
              [line[12], line[13],line[14]],
              [line[15], line[16], line[17]]]
    R_triple.append(R_mat)
    
    t_mat = [[line[18]],[line[19]],[line[20]]]
    t_triple.append(t_mat)
    
    i = i + 1
    if i%3 == 0:
        three_images.append(triple_set)
        K_mats.append(K_triple)
        R_mats.append(R_triple)
        t_mats.append(t_triple)
        triple_set = []
        K_triple = []
        R_triple = []
        t_triple = []
    
    
#"imgname.png k11 k12 k13 k21 k22 k23 k31 k32 k33 r11 r12 r13 r21 r22 r23 r31 r32 r33 t1 t2 t3"

fig100 = plt.figure()
ax = plt.axes(projection='3d')
cmap_op = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn','viridis', 'plasma', 'inferno', 'magma', 'cividis']

os.chdir('./dinoSparseRing/dinoSparseRing')

for cmap_option, three_images_coords in enumerate(three_images):
    X_matches, colors, img1_rectified, img2_rectified, img3_rectified = set_points(three_images_coords)

    #dispSGBM_1 = get_depth_2(img1_rectified, img2_rectified)
    #dispSGBM_2 = get_depth_2(img2_rectified, img3_rectified)


    # In[12]:


    #Camera measurements in Blender
    #distance from lens to surface of sphere = 9.5104
    #distance from principal point to surface of sphere = 10.6558
    #15 degrees in radians is 0.261799

    #translation when rotating 15 degrees around the z-axis
    #x = 9.5144*cos(0.261799) = 9.1863419329
    #y = 9.5144*sin(0.261799) = 2.46250435877

    #9.51066800099 = baseline

    #b = 35.9 
    #f = 188.9 

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

    # Camera matrix
    #K = [[f, 0, img1_rectified.shape[0]/2],
    #     [0, f, img1_rectified.shape[1]/2],
    #     [0, 0, 1]]
    #R = np.eye(3)
    #ta = [[0],[0],[1]]
    #Pa = K @ np.concatenate((R, ta), axis=1)

    #R = [[np.cos(15),-np.sin(15),0],
    #     [np.sin(15),np.cos(15),0],
    #     [0,0,1]]     
    #tb = [[34.720032502], [9.3071030883], [0]] 
    #Pb = K @ np.concatenate((R, tb), axis=1)

    K = K_mats[cmap_option][0]
    K_1 = K_mats[cmap_option][1]
    K_2 = K_mats[cmap_option][2]
    
    R = R_mats[cmap_option][0]
    R_1 = R_mats[cmap_option][1]
    R_2 = R_mats[cmap_option][2]
    
    t = t_mats[cmap_option][0]
    t_1 = t_mats[cmap_option][1]
    t_2 = t_mats[cmap_option][2]
    
    print("K is ", K)
    print("R is ", R)
    print("t is ", t)
    
    Pa = K @ np.concatenate((R, t), axis=1)
    Pb = K_1 @ np.concatenate((R_1, t_1), axis=1)
    Pc = K_2 @ np.concatenate((R_2, t_2), axis=1)
    
    pa = X_matches[:,0].tolist()
    pb = X_matches[:,1].tolist()
    pc = X_matches[:,2].tolist()
    

    X0_rec = np.empty([len(pa)+len(pc),3])
    em = 0
    for pta, ptb in zip(pa,pb):
        pt = solve_point_triangulation([pta,ptb], [Pa, Pb])[:-1]
        X0_rec[em] = np.array(pt)
        em = em + 1
        
    for ptb, ptc in zip(pa,pc):
        pt = solve_point_triangulation([pta,ptc], [Pa, Pc])[:-1]
        X0_rec[em] = np.array(pt)
        em = em + 1
        
        


    # In[13]:


    
    ax.scatter3D(X0_rec[:, 0], X0_rec[:, 1], X0_rec[:, 2], cmap=cmap_op[cmap_option])
plt.show()


    







