#!/usr/bin/env python
# coding: utf-8

# # Goal: Estimate Volume of an Object
# 
# This is programming project aims to estimate the volume of an object using stereo vision. The procedure is described on page 67 of the document below, as well as this write up. 
# Some parts of the code is recycled from previous projects.
# 
# https://carlos-hernandez.org/papers/fnt_mvs_2015.pdf
# 
# "First, implementing stereo rectification is a great idea and builds off of the work we did in class, so it's a great target. I don't think Blender will generate radial distortion, 
# so I don't expect that you'll need to model that effect (though I could be wrong).
# 
# 
# 
# Second, the NCC depth map technique we impemented in class, so (while interesting) is not going to count for much in your "implementation" section.
# 
# 
# 
# If your images are coming from Blender, you should not need to compute any segmentation: you can make the background transparent and use that to mask you image. 
# To compute the volume of the object, you will probably want to look into one of the multi-view stereo applications that I discussed in class and try implementing that, 
# since the described procedure is not generally capable of giving you volume (you need to combine the 2D projections somehow, which it does not seem that you are doing).
# 
# 
# 
# The LK optical flow algorithm seems like a good alternative plan; so pick whichever you would like. There are also a few open source datasets for optical flow if you'd like 
# to use those: https://vision.middlebury.edu/flow/data/"

# In[1]:


# Starter code (run this first)
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps


# In[2]:


def load_image(filepath):
    """Loads an image into a numpy array.
    Note: image will have 3 color channels [r, g, b]."""
    img = np.float32(Image.open(filepath))
    #img = ImageOps.grayscale(img)
    return cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
    #return (np.asarray(img).astype('uint8')/255)[:, :, :3]


# ### 1.1 Initial Feature Matching
# We need to detect blob and corner features
# in each image using the Difference-of-Gaussian (DoG) and Harris operators. (p.68)

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
        if m.distance < 0.85*n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
        
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)
    # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]
    
    return np.stack((pts1, pts2), axis=1), pts1,pts2,F
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


# ### 1.2 Model Fitting
# RANSAC-based methods randomly sample the 
# matches to generate candidate models and select the model with 
# the maximum number of consenting matches, which also 
# defines the number of remaining models to test

# In[4]:


## RANSAC Solution
def solve_homography(matches, base_img, img_example, rounds=100, sigma=5, s=4):  
# You will be computing this yourself using your implementation
# of the `solve_homography` function.
#    H_computed = [
#        [ 1.21083264e+00, -8.97707425e-03,  1.11129596e+00],
#        [ 3.15219064e-03,  1.22310850e+00, -1.67420761e+00],
#        [ 1.96243539e-05,  6.50992714e-05,  1.00000000e+00]]

    A = []
    num_rows = np.arange(len(matches))

    
    for i in num_rows:
        A.append([matches[i][0][0], matches[i][0][1], 1, 0, 0, 0, (-1*matches[i][1][0]*matches[i][0][0]), (-1*matches[i][1][0]*matches[i][0][1]), (-1*matches[i][1][1])])
        A.append([0, 0, 0, matches[i][0][0], matches[i][0][1], 1, (-1*matches[i][1][1]*matches[i][0][0]), (-1*matches[i][1][1]*matches[i][0][1]), (-1*matches[i][1][1])])
        
    num_rows = np.arange(len(A))
    A = np.stack(A, axis=0)
    [U, S, Vt] = np.linalg.svd(A)
    homography = Vt[-1].reshape(3, 3)
    
    #visualize_computed_transform(base_img, img_example, homography, matches)
    return homography

def solve_homography_ransac(matches, img_base, img_transformed, rounds=100, sigma=5, s=4):
    num_inliers = 0
    best_inliers = []
    best_H = []
    
    def get_inliers(matches, H, dist=sigma, chsq_thresh=5.99):
        orig_points = np.ones((matches.shape[0], 3))
        orig_points = np.stack(
            (matches[:,0,0].flatten(), matches[:,0,1].flatten(), np.ones(matches.shape[0])),
            axis=-1)
 
        trans_points = np.matmul(H, orig_points.T).T
        trans_points[:, 0] = trans_points[:, 0] / trans_points[:, 2]
        trans_points[:, 1] = trans_points[:, 1] / trans_points[:, 2]
        target_points = np.stack(
            (matches[:,1,0].flatten(), matches[:,1,1].flatten(), np.ones(matches.shape[0])),
            axis=-1)
        rsq = (
            np.abs(trans_points[:, 0] - target_points[:, 0]) ** 2 +
            np.abs(trans_points[:, 1] - target_points[:, 1]) ** 2
        )
        return matches[rsq <= chsq_thresh * (dist**2)]

    for _ in range(rounds):
        ps = np.random.choice(np.arange(matches.shape[0]), size=s)
        ms = matches[ps]
        H = solve_homography(ms, img_base, img_transformed)
        inliers = get_inliers(matches, H, sigma)
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_H = H.copy()

    best_H = solve_homography(best_inliers, img_base, img_transformed)
    best_inliers = get_inliers(matches, best_H, sigma)

    return best_H


# In[5]:


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

def correct_affine_ambiguity(M, S):
    Atilde = M
    Xtilde = S
    m = Atilde.shape[0]//2
    Am = np.zeros((3*m, 9))
    bm = np.zeros(3*m)
    for mi in range(m):
        Atl = Atilde[2*mi:2*mi+2]
        for ri in range(3):
            for ci in range(3):
                Am[3*mi+0, 3*ri + ci] = Atl[0, ri] * Atl[0, ci]
                bm[3*mi+0] = 1
                Am[3*mi+1, 3*ri + ci] = Atl[1, ri] * Atl[1, ci]
                bm[3*mi+1] = 1
                Am[3*mi+2, 3*ri + ci] = Atl[0, ri] * Atl[1, ci]
                bm[3*mi+2] = 0

    CCT = np.reshape(np.linalg.lstsq(Am, bm, rcond=None)[0], (3, 3))
        
    # Enforce a Positive Semi-Definite Matrix
    # (usually not necessary)
    C = 0.5*(CCT + CCT.T)
    w, v = np.linalg.eig(C)
    CCT = v @ np.diag(np.abs(w)) @ np.linalg.inv(v)

    C = np.linalg.cholesky(CCT)
    
    A = Atilde @ C
    X = np.linalg.inv(C) @ Xtilde
    return A, X


# In[6]:


def transform_image(image, tmat):
    import cv2
    return cv2.warpPerspective(
        image, 
        np.array(tmat).astype(float), 
        dsize=(image.shape[1], image.shape[0]))

def visualize_computed_transform(image_base, image_transformed, H, matches):
    fig = plt.figure(figsize=(8, 8), dpi=150)
    tmat = np.linalg.inv(H)
    image_rec = transform_image(image_transformed, tmat)
    
    # Plotting
    ax = plt.subplot(2, 1, 1)
    visualize_matches(image_base, image_transformed, matches, ax)
    plt.title('Base Images (with matches)')
    ax = plt.subplot(2, 2, 3)
    plt.imshow(image_rec, vmin=0, vmax=1)
    plt.title('Reconstructed Image')
    ax = plt.subplot(2, 2, 4)
    plt.imshow(image_base - image_rec, vmin=-0.3, vmax=0.3, cmap='PiYG')
    plt.title('Reconstruction Difference')


# In[7]:


test1 = load_image('./world_rotate/trans_01.png')
fig1 = plt.figure()
plt.imshow(test1[:, :, 0])

fig2 = plt.figure()
test2 = load_image('./world_rotate/trans_02.png')
plt.imshow(test2[:, :, 0])

fig3 = plt.figure()
test3 = load_image('./world_rotate/trans_03.png')
plt.imshow(test3[:, :, 0])

depth_map = cv.imread('./world_rotate/depth_map_world.png')
depth_map = cv.cvtColor(depth_map, cv.COLOR_BGR2GRAY)

fig1 = plt.figure()
plt.imshow(depth_map)


# In[8]:


depth_map[400,1000]


# In[9]:


def rectify_two(test1,test2,pts1,pts2,F, path1,path2):
    png = "1.png"
    h1,w1 = test1.shape
    h2,w2 = test2.shape
    _,H1, H2 = cv.stereoRectifyUncalibrated(
        np.float32(pts1), np.float32(pts2), F, imgSize=(w1,h1)
    )
    
    img1_rectified = cv.warpPerspective(test1, H1, (w1,h1))
    img2_rectified = cv.warpPerspective(test2, H2, (w2,h2))
    
    
    return img1_rectified, img2_rectified

def get_depth(img1_rectified, img2_rectified):
    #stereo = cv.StereoBM_create(numDisparities=16*10, blockSize=15)
    
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
    speckleWindowSize = 200
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
        speckleRange=speckleRange,
        disp12MaxDiff=disp12MaxDiff,
        P1=8 * 1 * block_size * block_size,
        P2=32 * 1 * block_size * block_size,
    )
    
    disp = stereo.compute(np.uint8(img1_rectified), np.uint8(img2_rectified)).astype(np.float32)
    disp = cv.normalize(disp, disp, alpha=255, beta=0, norm_type=cv.NORM_MINMAX)
    disp = np.uint8(disp)

    b = 9.51066800099
    f = 0.05
    
    fig = plt.figure()
    plt.imshow(disp);#plt.colorbar();plt.clim(0,400)#;plt.show()

    #plot depth by using disparity focal length `C1[0,0]` from stereo calibration and `T[0]` the distance between cameras

    fig2 = plt.figure()
    plt.imshow(b*f/(disp),cmap='hot');plt.clim(-0,500);plt.colorbar();plt.show()
    return disp
#read in set of 3 images, and return the points for that image.

def align_points(X, i):
    #T = np.asmatrix([[10.25235012], [2.74710893], [0]])
    #R_x = np.asmatrix([[1,0,0],[0,-np.cos(63.6),np.sin(63.6)],[0,-np.sin(63.6),-np.cos(63.6)]])
    R_t = X @ np.asmatrix([[-np.cos(15),np.sin(15),0],[-np.sin(15),-np.cos(15),0],[0,0,1]]).transpose()
    for x in range(0, i):
        #X = X - T
        X = R_t @ X
    return X

def set_points(three_images_arr, i, depth_map):
    if len(three_images_arr) != 3:
        raise ValueError('Array must have 3 images')
    test1 = cv.imread(three_images_arr[0],0)
    test2 = cv.imread(three_images_arr[1],0)
    test3 = cv.imread(three_images_arr[2],0)
    matches1, pts1,pts2,F = get_point_matches(test1, test2)
    matches2, pts1_1,pts2_1,F_1 = get_point_matches(test1, test3)
    
    img1_rectified, img2_rectified = rectify_two(test1,test2,pts1,pts2,F,"rect1","rect2")
    img2_rectified, img3_rectified = rectify_two(test2,test3,pts1_1,pts2_1,F_1, "rect3","rect4")

    disp1 = get_depth(img1_rectified, img2_rectified)
    disp2 = get_depth(img2_rectified, img3_rectified)
    disp3 = get_depth(img1_rectified, img3_rectified)
    
    combined_matches = combine_matches(np.array(matches1), np.array(matches2)) 
    colors = get_match_colors(load_image(three_images_arr[0]), combined_matches)
    centers = np.mean(combined_matches, axis=0)
    matches_norm = combined_matches - centers #centered points for each image
    return combined_matches, colors


def reproject_image_points(A, X, centers):  
    #print(f'X matrix: {X}')
    mtx = A @ X
    new_matrix = np.zeros((mtx.shape[1],int(mtx.shape[0]/2),2))


    for j in range(mtx.shape[1]):
        new_matrix[j][0] = [mtx[0][j]+centers[0][0],mtx[1][j]+centers[0][1]]
        new_matrix[j][1] = [mtx[2][j]+centers[1][0],mtx[3][j]+centers[1][1]]
        new_matrix[j][2] = [mtx[4][j]+centers[2][0],mtx[5][j]+centers[2][1]]

            #new_matrix[i][j] = [mtx[]]

    return new_matrix

three_images =  ['./world_rotate/trans_01.png','./world_rotate/trans_02.png','./world_rotate/trans_03.png']
#three_images1 =  ['./world_rotate/trans_03.png','./world_rotate/trans_04.png','./world_rotate/trans_05.png']
#three_images2 =  ['./world_rotate/trans_05.png','./world_rotate/trans_06.png','./world_rotate/trans_07.png']

#three_images3 =  ['./world_rotate/trans_07.png','./world_rotate/trans_08.png','./world_rotate/trans_09.png']
#three_images4 =  ['./world_rotate/trans_09.png','./world_rotate/trans_10.png','./world_rotate/trans_11.png']
#three_images5 =  ['./world_rotate/trans_11.png','./world_rotate/trans_12.png','./world_rotate/trans_13.png']

#three_images6 =  ['./world_rotate/trans_13.png','./world_rotate/trans_14.png','./world_rotate/trans_15.png']
#three_images7 =  ['./world_rotate/trans_15.png','./world_rotate/trans_16.png','./world_rotate/trans_17.png']
#three_images8 =  ['./world_rotate/trans_17.png','./world_rotate/trans_18.png','./world_rotate/trans_19.png']

#three_images9 =  ['./world_rotate/trans_19.png','./world_rotate/trans_20.png','./world_rotate/trans_21.png']
#three_images10 =  ['./world_rotate/trans_22.png','./world_rotate/trans_23.png','./world_rotate/trans_24.png']
#three_images11 =  ['./world_rotate/trans_24.png','./world_rotate/trans_25.png','./world_rotate/trans_26.png']


X, colors = set_points(three_images, 0, depth_map)

from scipy.spatial.transform import Rotation as R

npad = ((0, 0), (0, 0), (0, 1))
X_new = np.pad(X, pad_width=npad, mode='constant', constant_values=0)

rotation_degrees = -15
rotation_radians = np.radians(rotation_degrees)
rotation_axis = np.array([0, 0, 1])

rotation_vector = rotation_radians * rotation_axis
rotation = R.from_rotvec(rotation_vector)

for i,point in enumerate(X[:,0]):
    X_new[i,0,2] = depth_map[X_new[i,0,1],X_new[i,0,0]]
    X_new[i,1,2] = depth_map[X_new[i,1,1],X_new[i,1,0]]
    X_new[i,2,2] = depth_map[X_new[i,2,1],X_new[i,2,0]]
    
    X_new[i,1,:] = rotation.apply(X_new[i,1,:])
    X_new[i,2,:] = rotation.apply(X_new[i,2,:])
    X_new[i,2,:] = rotation.apply(X_new[i,2,:])

X = X_new
#X1, colors1 = set_points(three_images1, 2, depth_map)
#X2, colors2 = set_points(three_images2, 4, depth_map)

#X3, colors3 = set_points(three_images3, 6)
#X4, colors4 = set_points(three_images4, 8)
#X5, colors5 = set_points(three_images5, 10)

#X6, colors6 = set_points(three_images6, 12)
#X7, colors7 = set_points(three_images7, 14)
#X8, colors8 = set_points(three_images8, 16)

#X9, colors9 = set_points(three_images9, 18)
#X10, colors10 = set_points(three_images10, 20)
#X11, colors11 = set_points(three_images11, 22)

#X = np.concatenate((X,X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11), axis=1)
#colors = np.concatenate((colors,colors1,colors2,colors3,colors4,colors5,colors6,colors7,colors8,colors9,colors10,colors11))

#X = np.concatenate((X,X1,X2), axis=0)
#colors = np.concatenate((colors,colors1,colors2))

np.save("points.npy", X)
np.save("colors.npy", colors)


# In[10]:


import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

X = np.load("points.npy")
colors = np.load("colors.npy")
colors = np.true_divide(colors,255) # convert values to between 0 and 1
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:,0], X[:,1], X[:,1], c=colors)
ax.set_xlim3d([-750, 750])
ax.set_ylim3d([-750, 750])
ax.set_zlim3d([-750, 750])
plt.show()


# In[11]:


#visualize_matches(three_images[0][:, :, 0], test2[:, :, 0], X)
#h_graph = solve_homography_ransac(matches1, test1[:,:,0], test2[:,:,0], rounds=100, sigma=5, s=4)

def visualize_computed_transform(image_base, image_transformed, H, matches):
    fig = plt.figure(figsize=(8, 8), dpi=150)
    tmat = np.linalg.inv(H)
    image_rec = transform_image(image_transformed, tmat)
    
    # Plotting
    ax = plt.subplot(2, 1, 1)
    visualize_matches(image_base, image_transformed, matches, ax)
    plt.title('Base Images (with matches)')
    ax = plt.subplot(2, 2, 3)
    plt.imshow(image_rec, vmin=0, vmax=1)
    plt.title('Reconstructed Image')
    ax = plt.subplot(2, 2, 4)
    plt.imshow(image_base - image_rec, vmin=-0.3, vmax=0.3, cmap='PiYG')
    plt.title('Reconstruction Difference')


# In[12]:


#attempt at depth map function from https://bit.ly/3nueicu for cost function
#and depth map algo from https://bit.ly/3dYh8DE
def cost_depth(ref_image, x, y, d):
    #disparity is defined as x_l - x_r
    #cost curve is set of cost values for all allowable disparities
    # for the pixel

    #c(x,y,d) contains cost values for each possible match
    #within the defined disparity range from a corresponding pixels
    return
    #c_1(x,y) and c_2(x,y) indicate minimum and second minimum values
    #of cost curve
    
def c_mean(ref_image, x, y):
    #average of costs within the defined disparity range
    #costs within this margin will have negative influence on 
    #confidence value, where as costs outside, do not have
    #any influence
    return

def conf_depth(ref_image,x,y,d):
    #x and y are the constraints of the disparity range
    d_diff = max(d) - min(d)
    
    d_sum = 0
    for i in range(0, len(d-1)):
        d_delta = abs(d[i] - ref_image[x,y])
        d_num = d_diff/float(3)
        d_num = min((d_delta - 1), w)
        d_num = max(w, 0)**2
         
        c_delta = cost_depth(ref_image,x,y,d) - min(cost(ref_image,x,y,d))
        d_denom = max(np.divide(c_delta - c_mean(ref_image,x,y),3),1)
        
        d_sum = d_sum + (d_num/d_denom)
        
    return 1/float(d_sum)
        

def simple_depth_map(ref_image, neighboring_images, depth_range):
    ret_values = np.zeros((ref_image.shape,3))
    for i in range(rows):
        for j in range(cols):
            p = ref_image[i,j]
            C_best = float('-inf')
            for d in depth_range:
                C_pd = np.correlate(p, d, 'full')
                if C_pd > C_best:
                    C_best = d
                    d_best = d
                    f_best = conf_depth(ref_image,i,j,d)
                    
    ret_values[i] = (p, d_best, f_best)             
                    
            


# In[ ]:





# In[13]:


def penalty(f):
    return

lambda_reg = 1.0
#lambda_reg is a 
#regularization parameter controls the degree of spatial smoothness

def cost_vol(depth_map, label):
    E = np.sum(penalty(f)) + lambda_reg*np.sum(1-1)

