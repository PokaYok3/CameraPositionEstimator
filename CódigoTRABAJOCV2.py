#####################################################################################
#
# MRGCV Unizar - Computer vision - COURSE ASSIGMENT
#
# Title: Estimation of a camera pose using old photos.
#
# Date:
#
#####################################################################################
#
# Authors: Jaime Ángel Ezpeleta Sobrevía
#
# Version:
#
#####################################################################################

'''Imports and functions'''
import time
import collections
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import glob
import scipy.linalg as scAlg
import cv2
import scipy as sc
import scipy.optimize as scOptim
def svd(mat, norm=False) -> np.array:
    """
    Function to perform SVD of a given matrix.
    Last column of vector v is the desired solution.

    :param mat: matrix we want to perform SVD to
    norm: if False, it does not normalize the vector.
    :return:
    It returns the 3D point obtained by a match of two 2D points.
    """
    u, s, vh = np.linalg.svd(mat)
    v = vh.T
    pt = v[:, -1]
    if norm:
        pt = pt / pt[-1]
    # point_3d /= point_3d[3] This also works
    return pt
def homographyPts(x1, x2) ->np.array:
    """
    Function that builds matrices for later svd. Homography case.
    :param x1: first point of the 2D match
    :param x2: second point of the 2D match
    :return:
    A matrix, composed by the stack of equations of both points.
    We stack both points' equations to get the system of equations (2m x 4), where m = num of points of the match.
    """
    line1 = np.array([x1[0][0], x1[1][0], 1, 0, 0, 0, -x2[0][0] * x1[0][0], -x2[0][0] * x1[1][0], -x2[0][0]])
    line2 = np.array([0, 0, 0, x1[0][0], x1[1][0], 1, -x2[1][0] * x1[0][0], -x2[1][0] * x1[1][0], -x2[1][0]])
    A = np.vstack((line1, line2))
    return A
def indexMatrixToMatchesList(matchesList):
    """
     -input:
         matchesList: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
     -output:
        dMatchesList: list of n DMatch object
     """
    dMatchesList = []
    for row in matchesList:
        dMatchesList.append(cv2.DMatch(_queryIdx=row[0], _trainIdx=row[1], _distance=row[2]))
    return dMatchesList

def matchesListToIndexMatrix(dMatchesList):
    """
     -input:
         dMatchesList: list of n DMatch object
     -output:
        matchesList: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
     """
    matchesList = []
    for k in range(len(dMatchesList)):
        matchesList.append([int(dMatchesList[k].queryIdx), int(dMatchesList[k].trainIdx), dMatchesList[k].distance])
    return matchesList


def matchWith2NDRR(desc1, desc2, distRatio, minDist):
    """
    Nearest Neighbours Matching algorithm checking the Distance Ratio.
    A match is accepted only if its distance is less than distRatio times
    the distance to the second match.
    -input:
        desc1: descriptors from image 1 nDesc x 128
        desc2: descriptors from image 2 nDesc x 128
        distRatio:
    -output:
       matches: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
    """
    matches = []
    matches2=[]
    nDesc1 = desc1.shape[0]

    for kDesc1 in range(nDesc1): #Capturo cada uno de los descriptores de la imagen 1.
        dist = np.sqrt(np.sum((desc2 - desc1[kDesc1, :]) ** 2, axis=1)) #Distancia de descriptor 1 a descriptor 2
        indexSort = np.argsort(dist) #Ordeno distancia de menor a mayor
        if (dist[indexSort[0]] < minDist and dist[indexSort[0]]>dist[indexSort[1]]*distRatio): #Fuerzo a que si dist más cercana es menro que threshold y d1>d2*distratio.
            matches.append([kDesc1, indexSort[0], dist[indexSort[0]]])#Uno match a lista.




    return matches,matches2
def npztox1x2(npz):
    npz = np.load(npz)
    npz.files
    a = npz['keypoints0']
    b = npz['keypoints1']
    matches = npz['matches']

    # print(matches.shape)
    cont = 0
    print(matches.shape)
    for i in range(355):
        if (matches[i] != -1): cont += 1
    print(cont)

    x1 = np.empty((3, cont))
    x2 = np.empty((3, cont))
    cont = 0
    for i in range(355):
        print(i)
        if (matches[i] != -1):
            x1[0, cont] = a[i][0]
            x1[1, cont] = a[i][1]
            x1[2, cont] = 1
            x2[0, cont] = b[matches[i]][0]
            x2[1, cont] = b[matches[i]][1]
            x2[2, cont] = 1
            cont += 1
    print(x1)
    print(x2)
    return x1,x2
def npztox1x2bis(npz):
    npz = np.load(npz)

    npz.files
    a = npz['keypoints0']
    b = npz['keypoints1']
    matches = npz['matches']
    c = npz['match_confidence']
    vectSG = []

    for i in range(matches.shape[0]):
        if (matches[i] != -1):
            vectSG.append([i, matches[i], 1 / c[i]])

    return vectSG,a,b
def calibrationFunction():
    # parameters of the camera calibration pattern
    pattern_num_rows = 9
    pattern_num_cols = 6
    pattern_size = (pattern_num_rows, pattern_num_cols)

    # mobile phone cameras can have a very high resolution.
    # It can be reduced to reduce the computing overhead
    image_downsize_factor = 1

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((pattern_num_rows * pattern_num_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_num_rows, 0:pattern_num_cols].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    images = glob.glob('C:/Users/jaime/Desktop/MASTER/Computer Vision/TRABAJO/mobilePhoneCameraCalibration - copia/calib_*.jpg')
    # cv.namedWindow('img', cv.WINDOW_NORMAL)  # Create window with freedom of dimensions
    # cv.resizeWindow('img', 800, 600)
    for fname in images:
        img = cv2.imread(fname)
        img_rows = img.shape[1]
        img_cols = img.shape[0]
        new_img_size = (int(img_rows / image_downsize_factor), int(img_cols / image_downsize_factor))
        img = cv2.resize(img, new_img_size, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        print('Processing caliration image:', fname)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, pattern_size, corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)
    # See PyCharm help at https://www.jetbrains.com/help/pycharm/

    # initial_distortion = np.zeros((1, 5))
    # initial_K = np.eye(3)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None,
                                                      flags=(cv2.CALIB_ZERO_TANGENT_DIST))

    # # reprojection error for the calibration images
    # mean_error = 0
    # for i in range(len(objpoints)):
    #     imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    #     error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    #     mean_error += error
    # print("total error: {}".format(mean_error / len(objpoints)))

    # print('The calibartion matrix is')
    # print(mtx)
    # print('The radial distortion parameters are')
    # print(dist)
    return mtx
def disassemble_T(Tmat) -> np.array:
    """
       Disassembles the a SE(3) matrix into the rotation matrix and translation vector.
       Source: Class documentation, modifications at our own.
       - input:
           Tmat matrix of camera motion with respect to a reference.
       - output:
           rotation and translation.
    """
    rotation = Tmat[0:3, 0:3]
    translation = Tmat[0:3, 3]
    return rotation, translation
def ensamble_T(rotation, translation) -> np.array:
    """
    Ensamble the a SE(3) matrix with the rotation matrix and translation vector.
    - input:
        rotation: rotation (3x3 matrix)
        translation: translation (3x array) -> this thing that happens with arrays in python
    - output: returns the full T matrix of camera motion compounded by the inputs.
    """
    Tmat = np.zeros((4, 4), dtype=np.float32)
    Tmat[0:3, 0:3] = rotation
    Tmat[0:3, 3] = translation
    Tmat[3, 3] = 1
    return Tmat
def rightCrosssP(x1, x2, Pmat1, Pmat2) -> np.array:
    """
    Function that builds matrices for later svd.
    :param x1: first point of the 2D match
    :param x2: second point of the 2D match
    :return:
    A matrix, composed by the stack of equations of right cross product of both points.
    We stack both points' equations to get the system of equations (2m x 4), where m = num of points of the match.
    """
    A1 = np.empty([2, 4])
    A2 = np.empty([2, 4])
    for i in range(A1.shape[0]):
        for j in range(A1.shape[1]):
            A1[i][j] = x1[i][0] * Pmat1[2][j] - Pmat1[i][j]  # We do not multiply by w because w = 1.
            A2[i][j] = x2[i][0] * Pmat2[2][j] - Pmat2[i][j]  # We do not multiply by w because w = 1.
    A = np.vstack((A1, A2))
    return A

def getMax(xwa, xwb, xwc, xwd):
    """
     Gets the solution with maximum number of points in front of both cameras.
     :param xwa: 3D points with configuration a
     :param xwb: 3D points with configuration b
     :param xwc: 3D points with configuration c
     :param xwd: 3D points with configuration d
     :return:
         max: Number of points that are in front of both cameras.
         options[idx]: string with the configuration that maximizes max.
     """
    pos_a = 0
    pos_b = 0
    pos_c = 0
    pos_d = 0
    options = ['a: R_plus90 | t ', 'b: R_plus90 | -t', 'c: R_minus90 | t', 'd: R_minus90 | -t']

    for i in range(xwa.shape[1]):
        if xwa[2][i] - T_21a[2][3] >= 0:
            pos_a += 1
        if xwb[2][i] - T_21b[2][3] >= 0:
            pos_b += 1
        if xwc[2][i] - T_21c[2][3] >= 0:
            pos_c += 1
        if xwd[2][i] -T_21d[2][3] >= 0:
            pos_d += 1
    result = [pos_a, pos_b, pos_c, pos_d]
    max = np.amax(result)
    idx = result.index(max)
    return max, options[idx]

def draw3DLine(ax, xIni, xEnd, strStyle, lColor, lWidth):
    """
    Draw a segment in a 3D plot
    Source: Class documentation.
    -input:
        ax: axis handle
        xIni: Initial 3D point.
        xEnd: Final 3D point.
        strStyle: Line style.
        lColor: Line color.
        lWidth: Line width.
    """
    ax.plot([np.squeeze(xIni[0]), np.squeeze(xEnd[0])], [np.squeeze(xIni[1]), np.squeeze(xEnd[1])],
            [np.squeeze(xIni[2]), np.squeeze(xEnd[2])],
            strStyle, color=lColor, linewidth=lWidth)

def drawRefSystem(ax, T_w_c, strStyle, nameStr):
    """
        Draw a reference system in a 3D plot: Red for X axis, Green for Y axis, and Blue for Z axis
        Source: Class documentation.
    -input:
        ax: axis handle
        T_w_c: (4x4 matrix) Reference system C seen from W.
        strStyle: lines style.
        nameStr: Name of the reference system.
    """
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 0:1], strStyle, 'r', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 1:2], strStyle, 'g', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 2:3], strStyle, 'b', 1)
    ax.text(np.squeeze(T_w_c[0, 3] + 0.1), np.squeeze(T_w_c[1, 3] + 0.1), np.squeeze(T_w_c[2, 3] + 0.1), nameStr)

def crossMatrixInv(M):
    x = [M[2, 1], M[0, 2], M[1, 0]]
    return x

def crossMatrix(x):
    M = np.array([[0, -x[2], x[1]],
                  [x[2], 0, -x[0]],
                  [-x[1], x[0], 0]], dtype="object")
    return M

def resBundleProjection(Op,x1Data, x2Data, K_c, nPoints):
    """ - input:
            - Op: Optimization parameters: this must include a
            parametrization for T_21 (reference 1 seen from reference 2)
            in a proper way and for X1 (3D points in ref 1)
            - x1Data: (3xnPoints) 2D points on image 2 (homogeneous coordinates)
            - x2Data: (3xnPoints) 2D points on image 2 (homogeneous coordinates)
            - K_c: (3x3) Intrinsic calibration matrix
            - nPoints: Number of points
        - output:
            - res: residuals from the error between the 2D matched points
            and the projected points from the 3D points (2 equations/residuals per 2D point)

    """
    # Build the projection function
    theta = Op[0:3]
    R = sc.linalg.expm(crossMatrix(list(theta))) # We got the rotation from 1 to 2

    #tita = Op[3]
    #psi = Op[4]
    t =np.zeros((3,1))
    #t[0,0] = math.sin(tita) * math.cos(psi)
    t[0, 0] = Op[3]
    #t[1,0] = math.sin(tita) * math.sin(psi)
    t[1, 0] = Op[4]
    #t[2,0] = math.cos(tita)
    t[2, 0] = Op[5]
    t = t.reshape((3,))
    T_21 = ensamble_T(R,t)
    canonical = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    P_21 = K_c @ canonical @ T_21
    # Now we reconstruct the 3D points
    ones = np.ones((1,nPoints))
    x_3D_1 = np.zeros((4,nPoints))
    x_3D_1[0,:] = Op[6:109]
    x_3D_1[1,:] = Op[109:212]
    x_3D_1[2,:] = Op[212:315]
    x_3D_1[3, :] = ones

    x2projected = np.zeros((x1Data.shape[0], nPoints))
    x1projected = np.zeros((x1Data.shape[0], nPoints))
    #can = np.array(canonical).reshape((3,4))
    for i in range(nPoints):
        x2projected[:,i] = P_21 @ x_3D_1[:,i]
        x1projected[:,i] = K_c @ canonical @ x_3D_1[:,i]

    x2projected = x2projected/x2projected[2,:] # We need to normalize both 2Dpoints in both images
    x1projected = x1projected/x1projected[2,:]
    '''
    plot2dPoints(x1Data, x1projected, x2Data, x2projected)
    '''
    res1 = x1projected[0:2,:] - x1Data[0:2,:]
    res2 = x2projected[0:2,:] - x2Data[0:2,:]
    res1 = res1.flatten()  # x1, y1, x2, y2, x3, y3, x4, y4, ...
    res2 = res2.flatten()  # x1, y1, x2, y2, x3, y3, x4, y4, ...
    res = np.hstack((res1,res2))
    return res
def n_Points_in_Front(x_point, C1_T_C2 ): #C2_T_C1
    votos = 0

    for x in x_point.T:

        x_prj = C1_T_C2 @ x #C2_T_C1
        if x[2] >= 0 and x_prj[2] >= 0:
            votos += 1
    return votos
def resBundleProjection(Op, x1Data, x2Data, Kc, nPoints):
    """
        Residual function for least squares method
        -input:
            Op: Optimization parameters: this must include a
            paramtrization for T_21 (reference 1 seen from reference 2)
            in a proper way and for X1 (3D points in ref 1)
            x1Data: (3xnPoints) 2D points on image 1 (homogeneous
            coordinates)
            x2Data: (3xnPoints) 2D points on image 2 (homogeneous
            coordinates)
            K_c: (3x3) Intrinsic calibration matrix
            nPoints: Number of points
        -output:
            res: residuals from the error between the 2D matched points
            and the projected points from the 3D points
            (2 equations/residuals per 2D point)
    """
    #Construct the project matrix
    canonical = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
    x1proj=np.empty((3,nPoints))
    x2proj=np.empty((3,nPoints))
    R = sc.linalg.expm(crossMatrix([Op[0], Op[1], Op[2]]))
    T=[math.sin(Op[3])*math.cos(Op[4]),math.sin(Op[3])*math.sin(Op[4]),math.cos(Op[3])]
    #print(T)
    T_21=ensamble_T(R,T)
    ones=np.ones((1,x1Data.shape[1]))
    x_3D_1=np.empty((4,x1Data.shape[1]))
    x_3D_1[3]=ones
    j=0

    for i in range(nPoints):
        x_3D_1[0,i] =Op[5+j]
        x_3D_1[1, i] =Op[6+j]
        x_3D_1[2, i] =Op[7+j]
        j+=3
    # print(x1Data)
    # print(x1Data.shape)
    # print(x2Data)
    # print(x2Data.shape)
    x1proj=Kc@canonical@x_3D_1
    x1proj/=x1proj[2,:]
    x2proj=Kc@canonical@T_21@x_3D_1
    x2proj /= x2proj[2, :]
    # print(x1proj)
    # print(x1proj.shape)
    # print(x2proj)
    # print(x2proj.shape)
    #plot2dPoints(x1Data, x1proj, x2Data, x2proj)

    error1=x1proj[0:2,:]-x1Data
    error1=error1.flatten()
    error2=x2proj[0:2,:]-x2Data
    error2 = error2.flatten()

    return (np.concatenate((error1,error2))).flatten()

def TripleMatches(m1,m2,m1_old,m3,p3d):
    '''


    :param m1: Matches imageact ref 1
    :param m2: Matches imageact 2
    :param m1_old: Matches fotact ref with old image
    :param m3: Matches old image
    :param p3d: Points3D
    :return: list of the Matches.
    '''
    matches_1 = []
    matches_2 = []
    matches_3 = []
    matches_3D = []
    j = 0
    for match in m1:
        i = 0
        for cross_match in m1_old:
            if (match[0]==cross_match[0] and match[1]==cross_match[1]):
                matches_1.append(cross_match)
                matches_2.append(m2[j,:])
                matches_3.append(m3[i,:])
                matches_3D.append(p3d[j,:])
            i+=1
        j+=1
    return np.asarray(matches_1), np.asarray(matches_2), np.asarray(matches_3),np.asarray(matches_3D)
def DLT(points2d,points3D):
    '''

    :param points2d: Points 2d of the old image
    :param points3D: points3D
    :return: Projection Matrix P of the old camera.
    '''
    A = []
    for i in range(points2d.shape[1]):

        A.append([
            -points3D[0, i], -points3D[1, i], -points3D[2, i], -points3D[3, i],
            0, 0, 0, 0,
            points2d[0, i] * points3D[0, i], points2d[0, i] * points3D[1, i], points2d[0, i] * points3D[2, i],
            points2d[0, i] * points3D[3, i]
        ])

        A.append([
            0, 0, 0, 0,
            -points3D[0, i], -points3D[1, i], -points3D[2, i], -points3D[3, i],
            points2d[1, i] * points3D[0, i], points2d[1, i] * points3D[1, i], points2d[1, i] * points3D[2, i],
            points2d[1, i] * points3D[3, i]
        ])


    u, s, V = np.linalg.svd(A)

    res = V.T[:, -1]


    P = res.reshape((3, 4))


    return np.asarray(P)
votant=0
votact=0
if __name__ == '__main__':
    if True:
        np.set_printoptions(precision=4,linewidth=1024,suppress=True)

        path_image_vieja = 'fotovieja.jpg'  # Imagen vieja
        path_image_1col='colfotoactual2.jpg'
        path_image_2col='colfotoactual1.jpg'
        #path_image_3col='colfotoactual3.jpg'
        path_image_viejacol='colegiosanagustin1.png'

        canonical = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
        # Read images

        image_pers_1=cv2.imread(path_image_1col)
        #image_pers_1=cv2.imread(path_image_3col)
        image_pers_2=cv2.imread(path_image_2col)
        image_pers_viejacol=cv2.imread(path_image_viejacol)


        img_gris_pers_4 = cv2.cvtColor(image_pers_1, cv2.COLOR_BGR2GRAY)
        img_gris_pers_5 = cv2.cvtColor(image_pers_2, cv2.COLOR_BGR2GRAY)
        img_gris_vieja = cv2.cvtColor(image_pers_viejacol, cv2.COLOR_BGR2GRAY)

        npzpath='C:/Users/jaime/Desktop/MASTER/Computer Vision/TRABAJO/colfotoactual2_colfotoactual1_matches.npz'
        npzpath2 = 'C:/Users/jaime/Desktop/MASTER/Computer Vision/TRABAJO/colegiosanagustin1_colfotoactual1_matches.npz'
        #npzpath3 = 'C:/Users/jaime/Desktop/MASTER/Computer Vision/TRABAJO/colfotoactual3_colfotoactual2_matches.npz'



        matchesListG,keypointsSG1,keypointsSG2=npztox1x2bis(npzpath)

        dMatchesListG = indexMatrixToMatchesList(matchesListG)
        imgMatched = cv2.drawMatches(img_gris_pers_4, cv2.KeyPoint_convert(keypointsSG1), img_gris_pers_5, cv2.KeyPoint_convert(keypointsSG2), dMatchesListG,
                                     None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.title("Matches Superglue")
        plt.imshow(imgMatched, cmap='gray', vmin=0, vmax=255)
        plt.show()
        #plt.waitforbuttonpress()
        srcPts = np.float32([cv2.KeyPoint_convert(keypointsSG1)[m.queryIdx].pt for m in dMatchesListG]).reshape(len(dMatchesListG), 2)
        dstPts = np.float32([cv2.KeyPoint_convert(keypointsSG2)[m.trainIdx].pt for m in dMatchesListG]).reshape(len(dMatchesListG), 2)
        print("Puntos imagen 1:")
        x2 = np.vstack((srcPts.T, np.ones((1, srcPts.shape[0]))))
        print(x2)
        print(x2.shape)
        print("Puntos imagen 2:")
        x1 = np.vstack((dstPts.T, np.ones((1, dstPts.shape[0]))))
        print(x1)
        print(x1.shape)
        '''Puntos en la tercera imagen (foto antigua)'''
        matchesListG2, keypointsSG12, keypointsSG22 = npztox1x2bis(npzpath2)
        dMatchesListG2 = indexMatrixToMatchesList(matchesListG2)
        imgMatched = cv2.drawMatches(img_gris_vieja, cv2.KeyPoint_convert(keypointsSG12), img_gris_pers_5,
                                     cv2.KeyPoint_convert(keypointsSG22), dMatchesListG2,
                                     None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.title("Matches Superglue con foto vieja")
        plt.imshow(imgMatched, cmap='gray', vmin=0, vmax=255)
        plt.show()
        #plt.waitforbuttonpress()
        srcPts2 = np.float32([cv2.KeyPoint_convert(keypointsSG12)[m.queryIdx].pt for m in dMatchesListG2]).reshape(
            len(dMatchesListG2), 2)
        dstPts2 = np.float32([cv2.KeyPoint_convert(keypointsSG22)[m.trainIdx].pt for m in dMatchesListG2]).reshape(
            len(dMatchesListG2), 2)
        print("Puntos en la foto vieja")
        x4 = np.vstack((srcPts2.T, np.ones((1, srcPts2.shape[0]))))
        print(x4)
        print(x4.shape)
        x3 = np.vstack((dstPts2.T, np.ones((1, dstPts2.shape[0]))))
        print("Puntos con la camara actual")
        print(x3)
        print(x3.shape)
    #Calculo inliers fotos actuales
    if True:
        # AQUI VA A TENER QUE HABER OTRO FOR PARA ITERACIONES DE VOTACION
        x1aux = np.empty((3, 4))
        x2aux = np.empty((3, 4))

        # print(x2.shape)
        # print(x1.shape)
        # print(x1)
        indexinlierok = []
        indexBestxok = []
        indexBestx = []
        maskinliers = np.zeros((x1.shape[1],))
        maskmatches = np.zeros((x1.shape[1],))
        maskmatchesx = np.zeros((x1.shape[1],))
        print("Next step")

        # 2) cogere 4 par de puntos (ultimas filis del sift).
        for k in range(10000):
            votact = 0
            selectionant = 0
            indexinlier = []
            indexBestx = []
            A = []
            for i in range(8):

                selection = random.randint(0, x1.shape[1] - 1)
                if (selectionant == selection):
                    selection = selection - 1
                indexBestx = np.append(indexBestx, selection)
                selectionant = selection

                j = 0
                A.append(
                    [x2[0][selection] * x1[0][selection], x2[0][selection] * x1[1][selection], x2[0][selection],
                     x2[1][selection] * x1[0][selection], x2[1][selection] * x1[1][selection], x2[1][selection],
                     x1[0][selection], x1[1][selection], 1])

            u, s, vh = np.linalg.svd(A)
            v = vh.T
            F_21 = v[:, -1]

            # Enforce F to be rank 2 instead of rank 3
            F_21 = np.reshape(F_21, (3, 3))
            u, s, vh = np.linalg.svd(F_21)
            s = np.diag(s)  # Enforce s to be a diagonal matrix
            s[2, 2] = 0  # We enforce rank 2 by changing s3 in S matrix

            # re-estimating the F_12 matrix with rank 2
            F_21 = u @ s @ vh

            for i in range(x2.shape[1]):
                line = F_21 @ x1[:, i]  # l=F@xo linea epipolar

                dist = abs(line[0] * x2[0, i] + line[1] * x2[1, i] + line[2]) / math.sqrt(
                    pow(line[0], 2) + pow(line[1], 2))  # distancia de esa linea al punto x2
                # print(dist)
                # #3
                if (dist < 1):  # 4) COJO DISTANCIAS DE DE LOS NUEVOS Y LOS QUE TENIA  --> MIDO CON DISTANCIA --> VOTACIÓN.
                    votact += 1
                    indexinlier = np.append(indexinlier, i)
                # print("Votacion actual: " + str(votact))

            # time.sleep(5)
            if (votant < votact):
                Fok = F_21
                votant = votact
                indexBestxok = indexBestx
                indice = k
                # x1ok = x1aux
                # x2ok = x2aux
                indexinlierok = indexinlier
            # if(k==10):
            #     maskmatchesx[indexBestx.astype(int)] = 1
            #     maskmatchesx = maskmatchesx.astype(int)
        # print(x2new)
        # print(x1)
        print("The Fundamental matrix selected is: ", '\n', Fok)
        print("Votacion de: " + str(votant))
        print("En la iteracion " + str(indice))
        # print("En la x1ok ", '\n', x1ok)
        # print("En la x2ok ", '\n', x2ok)
        print("En la indexinlier " + str(indexinlierok))

            # print("x2 new:")
            # print(x2new)
            # print("x1:")
            # print(x1)
            # print("The homography matrix selected is: ", '\n', homographyok)
            # print("Votacion de: "+str(votant))
            # print("En la iteracion "+str(indice))
            # print("En la x1ok ", '\n', x1ok)
            # print("En la x2ok ", '\n', x2ok)
            # print("En la indexinlier "+str(indexinlierok))
            #mascara para inliers.
        maskinliers[indexinlierok.astype(int)] = 1
        maskinliers = maskinliers.astype(int)
        # mascara para matches.

        print("Mascara inliers")
        print(maskinliers)
        a=collections.Counter(maskinliers)
        print(a)
        print(a[1])
        plt.figure(2)
        # plt.title("Set of matches")
        # imgMatched2 = cv2.drawMatches(img_gris_pers_4, cv2.KeyPoint_convert(keypointsSG1), img_gris_pers_5, cv2.KeyPoint_convert(keypointsSG2), dMatchesListG,
        #                                  None,None,None,maskmatchesx,
        #                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)  # and cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        # plt.imshow(imgMatched2, cmap='gray', vmin=0, vmax=255)
        # plt.show()

        plt.figure(3)
        plt.title("Inliers")
        imgMatched2 = cv2.drawMatches(img_gris_pers_4, cv2.KeyPoint_convert(keypointsSG1), img_gris_pers_5, cv2.KeyPoint_convert(keypointsSG2), dMatchesListG,
                                         None,None,None,maskinliers,
                                         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)  # and cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        plt.imshow(imgMatched2, cmap='gray', vmin=0, vmax=255)
        plt.show()

        points1=np.empty((2,a[1]))
        points2=np.empty((2,a[1]))

        j=0
        for i in range(a[1]+a[0]):
            if(maskinliers[i]==1):
                points1[:,j]=x1[:2,i]
                points2[:, j] = x2[:2, i]
                j+=1
    # Inliers con foto vieja
    if True:

        print("Calculando inliers con foto vieja")
        x3aux = np.empty((3, 4))
        x4aux = np.empty((3, 4))

        indexinlierok2 = []
        indexBestxok2 = []
        indexBestx2 = []
        maskinliers2 = np.zeros((x3.shape[1],))

        maskmatches2 = np.zeros((x3.shape[1],))
        maskmatchesx2 = np.zeros((x3.shape[1],))
        votact = 0
        votant = 0
        for k in range(10000):
            votact = 0
            selectionant = 0
            indexinlier2 = []
            indexBestx2 = []
            A = []
            for i in range(8):

                selection = random.randint(0, x3.shape[1] - 1)
                if (selectionant == selection):
                    selection = selection - 1
                indexBestx2 = np.append(indexBestx2, selection)
                selectionant = selection

            j = 0
            A.append(
                [x4[0][selection] * x3[0][selection], x4[0][selection] * x3[1][selection], x4[0][selection],
                 x4[1][selection] * x3[0][selection], x4[1][selection] * x3[1][selection], x4[1][selection],
                 x3[0][selection], x3[1][selection], 1])

            u, s, vh = np.linalg.svd(A)
            v = vh.T
            F_21 = v[:, -1]

            # Enforce F to be rank 2 instead of rank 3
            F_21 = np.reshape(F_21, (3, 3))
            u, s, vh = np.linalg.svd(F_21)
            s = np.diag(s)  # Enforce s to be a diagonal matrix
            s[2, 2] = 0  # We enforce rank 2 by changing s3 in S matrix

            # re-estimating the F_12 matrix with rank 2
            F_21 = u @ s @ vh

            # 3) PASAR PUNTOS DE 1 A 2.
            for i in range(x3.shape[1]):
                line = F_21 @ x4[:, i]  # l=F@xo linea epipolar

                dist = abs(line[0] * x3[0, i] + line[1] * x3[1, i] + line[2]) / math.sqrt(
                    pow(line[0], 2) + pow(line[1], 2))  # distancia de esa linea al punto x2
                # print(dist)
                # #3
                if (dist < 500):  # 4) COJO DISTANCIAS DE DE LOS NUEVOS Y LOS QUE TENIA  --> MIDO CON DISTANCIA --> VOTACIÓN.
                    votact += 1
                    indexinlier2 = np.append(indexinlier2, i)
                # print("Votacion actual: " + str(votact))

            # time.sleep(5)
            if (votant < votact):
                Fok2 = F_21
                votant = votact
                indexBestxok2 = indexBestx2
                indice = k
                # x1ok = x3aux
                # x2ok = x4aux
                indexinlierok2 = indexinlier2
            # if(k==10):
            #     maskmatchesx[indexBestx.astype(int)] = 1
            #     maskmatchesx = maskmatchesx.astype(int)
        # print("x2 new:")
        # print(x2new)
        # print("x1:")
        # print(x1)
        print("The Fundamental matrix selected is: ", '\n', Fok2)
        print("Votacion de: " + str(votant))
        print("En la iteracion " + str(indice))
        # print("En la x1ok ", '\n', x1ok)
        # print("En la x2ok ", '\n', x2ok)
        print("En la indexinlier " + str(indexinlierok2))
        # mascara para inliers.
        print(indexinlierok2)
        maskinliers2[indexinlierok2.astype(int)] = 1
        maskinliers2 = maskinliers2.astype(int)
        # mascara para matches.

        print("Mascara inliers foto antigua")
        print(maskinliers2)
        a2 = collections.Counter(maskinliers2)
        print(a2)
        print(a2[1])
        plt.figure(2)

        plt.figure(3)
        plt.title("Inliers foto antigua")
        imgMatched22 = cv2.drawMatches(img_gris_vieja, cv2.KeyPoint_convert(keypointsSG12), img_gris_pers_5,
                                       cv2.KeyPoint_convert(keypointsSG22), dMatchesListG2,
                                       None, None, None, maskinliers2,
                                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)  # and cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        plt.imshow(imgMatched22, cmap='gray', vmin=0, vmax=255)
        plt.show()

        points3 = np.empty((2, a2[1]))
        points4 = np.empty((2, a2[1]))

        j = 0
        for i in range(a2[1] + a2[0]):
            if (maskinliers2[i] == 1):
                points3[:2, j] = x3[:2, i]
                points4[:2, j] = x4[:2, i]
                j += 1
        print("Puntos foto cam ref con foto antigua")
        print(points3)
        points3aux = points3.astype(int)
    # Calculo de E  y representacion 2 camaras actuales sin BA
    if True:

        K_c=calibrationFunction()
        #print(K_c)
        # # 2.4 Pose estimation from two views
        # # We will obtain E from F, and then calculate the resulting four estimations of T.
        #
        #E_21 = (K_c)^T @ Fok @ K_c1
        E_21 = K_c.T @ Fok @ K_c
        #
        # Computing SVD to get R and t
        u, s, vh = np.linalg.svd(E_21)
        t = u[:, 2].reshape(3, 1)  # t = u3
        W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        #
        # R+90 and R-90
        R_plus_90 = u @ W @ vh
        R_minus_90 = u @ W.T @ vh
        # If determinants are -1, we multiply by -1 since they have to be +1.
        if np.linalg.det(R_plus_90) < 0:
            R_plus_90 = -R_plus_90
        if np.linalg.det(R_minus_90) < 0:
            R_minus_90 = -R_minus_90

        # Now we recover the four possible T matrices
        T_21a = ensamble_T(R_plus_90, t.reshape(3, ))
        T_21b = ensamble_T(R_plus_90, -t.reshape(3, ))
        T_21c = ensamble_T(R_minus_90, t.reshape(3, ))
        T_21d = ensamble_T(R_minus_90, -t.reshape(3, ))
        # # print(T_21a)
        #
        # Recover the camera matrices (P1, P2)
        P_c1 = K_c @ canonical
        P_c2_a = K_c @ canonical @ T_21a
        P_c2_b = K_c @ canonical @ T_21b
        P_c2_c = K_c @ canonical @ T_21c
        P_c2_d = K_c @ canonical @ T_21d
        # Now lets calc 3D points with each P matrix to see which one is the correct representation
        x_w23a = np.empty((4, points1.shape[1]))
        x_w23b = np.empty((4, points1.shape[1]))
        x_w23c = np.empty((4, points1.shape[1]))
        x_w23d = np.empty((4, points1.shape[1]))
        for i in range(points1.shape[1]):
            x1_2d = points1[:, i].reshape(2, 1)
            x2_2d = points2[:, i].reshape(2, 1)
            A23a = rightCrosssP(x1_2d, x2_2d, P_c1, P_c2_a)  # We perform the right cross prod to generate A matrix for both images.
            A23b = rightCrosssP(x1_2d, x2_2d, P_c1, P_c2_b)
            A23c = rightCrosssP(x1_2d, x2_2d, P_c1, P_c2_c)
            A23d = rightCrosssP(x1_2d, x2_2d, P_c1, P_c2_d)
            point23a = svd(A23a, True)
            point23b = svd(A23b, True)
            point23c = svd(A23c, True)
            point23d = svd(A23d, True)
            x_w23a[:, i] = point23a  # Add to the vector
            x_w23b[:, i] = point23b
            x_w23c[:, i] = point23c
            x_w23d[:, i] = point23d
        best, idx = getMax(x_w23a, x_w23b, x_w23c, x_w23d)
        print(f'The correct solution is option {idx} with {best} points in front of both cameras.')
        #print(x_w23a)
        n_votos_a = n_Points_in_Front(x_w23a, T_21a)
        n_votos_b = n_Points_in_Front(x_w23b, T_21b)
        n_votos_c = n_Points_in_Front(x_w23c, T_21c)
        n_votos_d = n_Points_in_Front(x_w23d, T_21d)
        print('nº votos A: ', n_votos_a)
        print('nº votos B: ', n_votos_b)
        print('nº votos C: ', n_votos_c)
        print('nº votos D: ', n_votos_d)

        votos=[n_votos_a, n_votos_b, n_votos_c, n_votos_d]
        max_value = max(votos)
        if(votos.index(max_value)==0):
            c2_T_c1=T_21a
            w_T_c1 = np.eye((4))
            c1_T_w=w_T_c1
            c2_T_w=c2_T_c1@c1_T_w
            w_T_c2=np.linalg.inv(c2_T_w)
            pc1=K_c@canonical@c1_T_w
            pc2=K_c@canonical@c2_T_w

        if (votos.index(max_value) == 1):
            c2_T_c1 = T_21b
            w_T_c1 = np.eye((4))
            c1_T_w = w_T_c1
            c2_T_w = c2_T_c1 @ c1_T_w
            w_T_c2 = np.linalg.inv(c2_T_w)
            pc1 = K_c @ canonical @ c1_T_w
            pc2 = K_c @ canonical @ c2_T_w
        if (votos.index(max_value) == 2):
            c2_T_c1 = T_21c
            w_T_c1 = np.eye((4))
            c1_T_w = w_T_c1
            c2_T_w = c2_T_c1 @ c1_T_w
            w_T_c2 = np.linalg.inv(c2_T_w)
            pc1 = K_c @ canonical @ c1_T_w
            pc2 = K_c @ canonical @ c2_T_w

        if (votos.index(max_value) == 3):
            c2_T_c1 = T_21d
            w_T_c1 = np.eye((4))
            c1_T_w = w_T_c1
            c2_T_w = c2_T_c1 @ c1_T_w
            w_T_c2 = np.linalg.inv(c2_T_w)
            pc1 = K_c @ canonical @ c1_T_w
            pc2 = K_c @ canonical @ c2_T_w
        x_w23 = np.empty((4, points1.shape[1]))
        for i in range(points1.shape[1]):
            x1_2d = points1[:, i].reshape(2, 1)
            x2_2d = points2[:, i].reshape(2, 1)
            A23 = rightCrosssP(x1_2d, x2_2d, pc1, pc2)
            point23 = svd(A23, True)
            x_w23[:, i] = point23  # Add the vector
        plt.figure(0)
        #print(x_w23)
        ax = plt.axes(projection='3d', adjustable='box')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        #print(w_T_c2)
        #drawRefSystem(ax, np.eye(4, 4), '-', 'W')
        drawRefSystem(ax, w_T_c1, '-', 'C1')
        drawRefSystem(ax, w_T_c2, '-', 'C2')
        #x_w23=w_T_c2@x_w23
        # ax.set_xlim(-5, 5)
        # ax.set_ylim(-5, 5)
        # ax.set_zlim(-5, 5)
        # # The estimated camera positions (SfM)
        # drawRefSystem(ax, T_w_c1_estimated, '-', 'C1_new')
        # drawRefSystem(ax, T_w_c2_estimated, '-', 'C2_new')
        #print(x_w23.shape)
        #ax.scatter(X_w[0, :], X_w[1, :], X_w[2, :], marker='.')
        ax.scatter(x_w23[0, :], x_w23[1, :], x_w23[2, :], marker='.', c='red')

        # Matplotlib does not correctly manage the axis('equal')
        xFakeBoundingBox = np.linspace(0, 4, 2)
        yFakeBoundingBox = np.linspace(0, 4, 2)
        zFakeBoundingBox = np.linspace(0, 4, 2)
        plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
        print('Close the figure to continue. Left button for orbit, right button for zoom.')
        plt.show()
    #BA cámaras actuales
    if True:
        print("Aplicando BA")
        rotation, traslation = disassemble_T(c2_T_c1)
        theta = crossMatrixInv(sc.linalg.logm(rotation.astype(np.float64)))

        phi = math.atan(traslation[1] / traslation[0])
        th = math.acos(traslation[2])
        # row = np.ones((1,points1.shape[1]))
        # points1 = np.vstack([points1, row])
        # row = np.ones((1, points2.shape[1]))
        # points2 = np.vstack([points2, row])
        points_3d = np.empty((4, points1.shape[1]))
        for i in range(int(points1.shape[1])):
            x1_2d = points1[:, i].reshape(2, 1)
            x2_2d = points2[:, i].reshape(2, 1)
            A = rightCrosssP(x1_2d, x2_2d, pc1,
                             pc2)  # We perform the right cross prod to generate A matrix for both images.
            point3d = svd(A, True)
            points_3d[:, i] = point3d
        #print(points_3d)
        points_3d/=points_3d[2,:]
        Op = []
        # Op = [theta[0], theta[1], theta[2], th, phi, points_3dref1[0, i], points_3dref1[1, i], points_3dref1[2, i], 1]
        Op = [theta[0], theta[1], theta[2], th, phi]
        for i in range(points_3d.shape[1]): #Aqui y abajo va x_w23 si eso
            Op = np.append(Op, points_3d[0, i])
            Op = np.append(Op, points_3d[1, i])
            Op = np.append(Op, points_3d[2, i])
            # Op = np.append(Op, points_3dref1[3, i])
        #print(points1.shape)
        #print(points2.shape)
        res = resBundleProjection(Op, points1, points2, K_c, nPoints=points1.shape[1])
        OpOptim = scOptim.least_squares(resBundleProjection, Op, args=(points1,points2,K_c,points1.shape[1]), method='lm')
        R_opt = sc.linalg.expm(crossMatrix([OpOptim.x[0], OpOptim.x[1], OpOptim.x[2]]))
        T_opt = [math.sin(OpOptim.x[3]) * math.cos(OpOptim.x[4]), math.sin(OpOptim.x[3]) * math.sin(OpOptim.x[4]),
              math.cos(OpOptim.x[3])]
        T_21_opt = ensamble_T(R_opt, T_opt)
        ones = np.ones((1, points1.shape[1]))
        x_3D_1_opt = np.empty((4, points1.shape[1]))
        x_3D_1_opt[3] = ones
        j = 0
        for i in range(points1.shape[1]):
            x_3D_1_opt[0, i] = OpOptim.x[5 + j]
            x_3D_1_opt[1, i] = OpOptim.x[6 + j]
            x_3D_1_opt[2, i] = OpOptim.x[7 + j]
            j += 3
        T_c2_w_opt = T_21_opt @ c1_T_w
        T_w_c2_opt = np.linalg.inv(T_c2_w_opt)
        #T_c1_w_opt = np.linalg.inv(T_21) @c2_T_w
        #T_w_c1_opt = np.linalg.inv(T_c1_w_opt)
        #
        plt.figure(8)

        ax = plt.axes(projection='3d', adjustable='box')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        x_3D_w_opt = w_T_c1 @ x_3D_1_opt
        #print(x_3D_w_opt)
        #print(x_3D_1_opt)
        #print(x_3D_w.shape)
        #drawRefSystem(ax, T_w_c1_opt, '-', 'C1_opt')
        drawRefSystem(ax, T_w_c2_opt, '-', 'C2_opt')
        drawRefSystem(ax, np.eye(4, 4), '-', 'W')
        drawRefSystem(ax, w_T_c1, '-', 'C1')
        drawRefSystem(ax, w_T_c2, '-', 'C2')
        # ax.set_xlim(0, 5)
        # ax.set_ylim(0, 5)
        # ax.set_zlim(0, 5)
        ax.scatter(x_3D_w_opt[0, :], x_3D_w_opt[1, :], x_3D_w_opt[2, :], marker='.', c='red')  # Optimal estimation from solution

        # #ax.scatter(x_3D_w[0, :], x_3D_w[1, :], x_3D_w[2, :], marker='.', c='red')  # Optimal estimation from solution
        xFakeBoundingBox = np.linspace(0, 4, 2)
        yFakeBoundingBox = np.linspace(0, 4, 2)
        zFakeBoundingBox = np.linspace(0, 4, 2)
        plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
        print('Close the figure to continue. Left button for orbit, right button for zoom.')
        plt.show()
        #plt.waitforbuttonpress()


    if True:
        matches_1, matches_2, matches_3, points_3D = TripleMatches(points1.T, points2.T,points3.T, points4.T,
                                                                                         x_3D_1_opt.T)
        matches_1 = matches_1.T
        matches_2 = matches_2.T
        matches_3 = matches_3.T
        points_3D = points_3D.T
        P3_est = DLT(matches_3, points_3D)

        Kcold, c3_R_c1, c3_t_c1,s,s1,s2,s3 = cv2.decomposeProjectionMatrix(P3_est)
        Kcold /= Kcold[2, 2]

        c3_t_c1 /= c3_t_c1[3, 0]

        c3_t_c1 = -c3_R_c1 @ c3_t_c1[:3]
        c3_t_c1=np.resize(c3_t_c1,(3,))

        c3_T_c1=ensamble_T(c3_R_c1,c3_t_c1)

        plt.figure(100)
        ax = plt.axes(projection='3d', adjustable='box')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        drawRefSystem(ax, T_w_c2_opt, '-', 'C2_opt')
        drawRefSystem(ax, np.eye(4, 4), '-', 'W')
        drawRefSystem(ax, w_T_c1, '-', 'C1')

        drawRefSystem(ax, w_T_c2, '-', 'C2')

        drawRefSystem(ax, c3_T_c1@c1_T_w, '-', 'C3')
        final_points_3D = w_T_c1 @ points_3D
        print(final_points_3D)
        ax.scatter(final_points_3D[0, :] , final_points_3D[1, :] ,
                   final_points_3D[2, :] , marker='.')
        xFakeBoundingBox = np.linspace(0, 4, 2)
        yFakeBoundingBox = np.linspace(0, 4, 2)
        zFakeBoundingBox = np.linspace(0, 4, 2)
        plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
        plt.title("3 cams pose")
        print('Close the figure to continue. Left button for orbit, right button for zoom.')
        plt.show()
        plt.waitforbuttonpress()