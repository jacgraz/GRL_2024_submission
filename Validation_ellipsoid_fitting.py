#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Codes to reproduce the evaluation of the estimators of orientation, based on
simulated 3D ellipsoid projected to simulate 2D MASC images.

This code is used to produce the results of the manuscript by Grazioli et al
submitted to GRL in 2024, entitled "Observation of falling snowflakes orientation in
 sheltered and unsheltered sites"


@author: Micky Condolf / Jacopo Grazioli 
"""
import os

# Adapt to local paths 
os.chdir("/home/grazioli/CODES/python/MASC-Anemometer")
out_path  = '/home/grazioli/Documents/Publications/Grazioli_GRL_2024/Raw_images/'


import numpy as np
from numpy.linalg import eig, inv
from numpy.random import rand, normal
import matplotlib.pyplot as plt
import math
import scipy.stats

import pandas as pd

from tqdm import tqdm
from scipy.spatial import ConvexHull
from scipy.linalg import svd

import plotly.io as pio
pio.renderers.default='browser'

from math import atan2


#----------------------------------------------------------------------------
# Utility functions

def __fit_ellipse(x, y):
    x, y = x[:, np.newaxis], y[:, np.newaxis]
    D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
    S, C = np.dot(D.T, D), np.zeros([6, 6])
    C[0, 2], C[2, 0], C[1, 1] = 2, 2, -1
    U, s, V = svd(np.dot(inv(S), C))
    a = U[:, 0]
    return a

def ellipse_center(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    num = b * b - a * c
    x0 = (c * d - b * f) / num
    y0 = (a * f - b * d) / num
    return np.array([x0, y0])

def ellipse_axis_length(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    up = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
    down1 = (b * b - a * c) * (
        (c - a) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a)
    )
    down2 = (b * b - a * c) * (
        (a - c) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a)
    )
    res1 = np.sqrt(up / down1)
    res2 = np.sqrt(up / down2)
    return np.array([res1, res2])

def ellipse_angle_of_rotation(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    return atan2(2 * b, (a - c)) / 2

def fit_ellipse(x, y):
    """@brief fit an ellipse to supplied data points: the 5 params
        returned are:
        M - major axis length
        m - minor axis length
        cx - ellipse centre (x coord.)
        cy - ellipse centre (y coord.)
        phi - rotation angle of ellipse bounding box
    @param x first coordinate of points to fit (array)
    @param y second coord. of points to fit (array)
    """
    a = __fit_ellipse(x, y)
    centre = ellipse_center(a)
    phi = ellipse_angle_of_rotation(a)
    M, m = ellipse_axis_length(a)
    # assert that the major axix M > minor axis m
    if m > M:
        M, m = m, M
    # ensure the angle is betwen 0 and 2*pi
    phi -= 2 * np.pi * int(phi / (2 * np.pi))
    return [M, m, centre[0], centre[1], phi]

def fit_ellipsoid(xx,yy,zz):

   # change xx from vector of length N to Nx1 matrix so we can use hstack
   x = xx[:,np.newaxis]
   y = yy[:,np.newaxis]
   z = zz[:,np.newaxis]

   #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz = 1
   J = np.hstack((x*x,y*y,z*z,x*y,x*z,y*z, x, y, z))
   K = np.ones_like(x) #column of ones

   #Solve the least square problem , i.e.compute (JT*J)^-1 * JT * K  
   polynomial= np.dot(np.linalg.inv(np.dot(J.transpose(),J)), np.dot(J.transpose(),K))
      
   coeff = np.append(polynomial,-1)
   
   #compute mean squared error
   predicted_points = np.dot(J, polynomial)
   distance = predicted_points - K
   squared_error = distance**2
   mean_squared_error = np.mean(squared_error)
   
   return (coeff, mean_squared_error)

def ls_ellipse(xx,yy):

  x = xx[:,np.newaxis]
  y = yy[:,np.newaxis]

  J = np.hstack((x*x, x*y, y*y, x, y))
  K = np.ones_like(x) #column of ones

  JT=J.transpose()
  JTJ = np.dot(JT,J)
  InvJTJ=np.linalg.inv(JTJ);
  ABC= np.dot(InvJTJ, np.dot(JT,K))

  eansa=np.append(ABC,-1)

  return eansa

def polyToParams3D(coeff, printMe):

   # convert the polynomial form of the 3D-ellipsoid to parameters
   # center, axes, and transformation matrix
   # coeff is the vector whose elements are the polynomial coefficients A..J
   # returns (center, axes, rotation matrix)

   if printMe: print('\npolynomial\n',coeff)

   Amat=np.array(
   [
   [ coeff[0],     coeff[3]/2.0, coeff[4]/2.0, coeff[6]/2.0 ],
   [ coeff[3]/2.0, coeff[1],     coeff[5]/2.0, coeff[7]/2.0 ],
   [ coeff[4]/2.0, coeff[5]/2.0, coeff[2],     coeff[8]/2.0 ],
   [ coeff[6]/2.0, coeff[7]/2.0, coeff[8]/2.0, coeff[9]     ]
   ])

   if printMe: print('\nAlgebraic form of polynomial\n',Amat)

   #See B.Bartoni, Preprint SMU-HEP-10-14 Multi-dimensional Ellipsoidal Fitting
   # equation 20 for the following method for finding the center
   A3=Amat[0:3,0:3]
   A3inv=inv(A3)
   ofs=coeff[6:9]/2.0
   center=-np.dot(A3inv,ofs)
   if printMe: print('\nCenter at:',center)

   # Center the ellipsoid at the origin
   Tofs=np.eye(4)
   Tofs[3,0:3]=center
   R = np.dot(Tofs,np.dot(Amat,Tofs.T))
   if printMe: print('\nAlgebraic form translated to center\n',R,'\n')

   R3=R[0:3,0:3]
   R3test=R3/R3[0,0]
   if printMe: print('normed \n',R3test)
   s1=-R[3, 3]
   R3S=R3/s1
   (el,ec)=eig(R3S)

   recip=1.0/np.abs(el)
   axes=np.sqrt(recip)
   if printMe: print('\nAxes are\n',axes  ,'\n')
   axes_ordered = np.flip(np.sort(axes, axis = 0), axis = 0)
   if printMe: print('\nSorted axes are\n',axes_ordered  ,'\n')

   inve=inv(ec) #inverse is actually the transpose here
   if printMe: print('\nRotation matrix\n',inve)
   sorted_rot = sort_matrix(axes, axes_ordered, inve)
   if printMe: print('\nSorted rotation matrix\n',sorted_rot)   
   
   #Calculate euler angles
   alpha = np.degrees(np.arctan2(inve[1,2],inve[0,2]))
   beta = np.degrees(np.arctan2(np.sqrt(1-inve[2,2]**2),inve[2,2]))
   gamma = np.degrees(np.arctan2(-inve[2,1],inve[2,0]))
   
   #Calculate orientation
   position = np.argmax(axes, axis=0)
   major_axis = inve[position,:]
   vertical_axis = [0,0,1]
   unit_vector_1 = major_axis / np.linalg.norm(major_axis)
   unit_vector_2 = vertical_axis / np.linalg.norm(vertical_axis)
   dot_product = np.dot(unit_vector_1, unit_vector_2)
   ####orientation = 90 - round(np.degrees(np.arccos(dot_product)),2)
   orientation = 90-round(np.arccos(dot_product),2)*180/np.pi  
   
   if printMe: print('\nOrientation\n',orientation)
   
   
   return (center,axes_ordered, sorted_rot, orientation, alpha, beta, gamma)

def polyToParams(v,printMe):

   # convert the polynomial form of the ellipse to parameters
   # center, axes, and tilt
   # v is the vector whose elements are the polynomial
   # coefficients A..F
   # returns (center, axes, tilt degrees, rotation matrix)

   #Algebraic form: X.T * Amat * X --> polynomial form

   Amat = np.array(
   [
   [v[0],     v[1]/2.0, v[3]/2.0],
   [v[1]/2.0, v[2],     v[4]/2.0],
   [v[3]/2.0, v[4]/2.0, v[5]    ]
   ])

   if printMe: print('\nAlgebraic form of polynomial\n',Amat)

   A2=Amat[0:2,0:2]
   A2Inv=inv(A2)
   ofs=v[3:5]/2.0
   cc = -np.dot(A2Inv,ofs)
   if printMe: print( '\nCenter at:',cc)

   # Center the ellipse at the origin
   Tofs=np.eye(3)
   Tofs[2,0:2]=cc
   R = np.dot(Tofs,np.dot(Amat,Tofs.T))
   if printMe: print( '\nAlgebraic form translated to center\n',R,'\n')

   R2=R[0:2,0:2]
   s1=-R[2, 2]
   RS=R2/s1
   (el,ec)=eig(RS)

   recip=1.0/np.abs(el)
   axes=np.sqrt(recip)
   #axes = np.flip(np.sort(axes, axis = 0), axis = 0)
   if printMe: print( '\nAxes are\n',axes  ,'\n')

   rads=np.arctan2(ec[1,0],ec[0,0])
   deg=np.degrees(rads) #convert radians to degrees (r2d=180.0/np.pi)
   if printMe: print( 'Rotation is ',deg,'\n')

   inve=inv(ec) #inverse is actually the transpose here
   if printMe: print( '\nRotation matrix\n',inve)
   return (cc[0],cc[1],axes[0],axes[1],deg,inve)

def rotation_matrix(axis, theta):
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def sort_matrix(vector, sorted_vector, matrix):
    sorted_matrix = np.zeros((3,3))
    for i in range(len(vector)):
        if vector[i] == sorted_vector[0]:
            sorted_matrix[:,0] = matrix[:,i] 
        elif vector[i] == sorted_vector[1]:
            sorted_matrix[:,1] = matrix[:,i] 
        elif vector[i] == sorted_vector[2]:
            sorted_matrix[:,2] = matrix[:,i] 
    
    return sorted_matrix


#############################################################################
#############################################################################
# SCRIPT STARTS HERE

#Initialize empty lists
#Initial references of ellipsoid geometry
reference_a = []
reference_b = []
reference_c = []
reference_yaw = []
reference_pitch = []
reference_roll = []
initial_fit_pitch = []

#Results of fitted ellipseoid
ellipsoid_fit_yaw = []
ellipsoid_fit_pitch = []
ellipsoid_fit_roll = []
ellipsoid_fit_a = []
ellipsoid_fit_b = []
ellipsoid_fit_c = []
ellipsoid_fit_mse = []
ellipsoid_fit_mse_norm = []

estimated_orientation = []

#Other methods
ell_fit_ori_cam0 = []
ell_fit_ori_cam1 = []
ell_fit_ori_cam2 = []
orientations3D = []
ell_fit_ori_min = []
ell_fit_ori_mean = [] # absolute value
ell_fit_ori_sign_mean = [] #signed

for i in tqdm(range(0,10000)):
    try:
        #generate random axis ratios (b/a and c/a) and sort them
        a = 1
        mu = 0.63
        sigma = 0.15
        ar_random_ba = scipy.stats.truncnorm.rvs((0-mu)/sigma,(1-mu)/sigma,loc=mu,scale=sigma,size=1)
        
        mu = 0.52
        sigma = 0.15
        ar_random_ca = scipy.stats.truncnorm.rvs((0-mu)/sigma,(1-mu)/sigma,loc=mu,scale=sigma,size=1)
       
        b = ar_random_ba
        c = ar_random_ca
        
        
        ellipsoid_axis = np.flip(np.sort(np.array([a, b, c]), axis = 0), axis = 0)
        reference_a.append(ellipsoid_axis[0])
        reference_b.append(ellipsoid_axis[1])
        reference_c.append(ellipsoid_axis[2])
        
        # Number of points to generate ellipsoid
        point_n = 1000
        
        #Generate random angles
        roll = (rand(1)*np.pi - np.pi/2)[0]
        pitch = normal(loc=0,scale=25.) #25° standard deviation
        pitch = pitch*np.pi/180
        yaw = (rand(1)*2*np.pi -np.pi)[0]
        
        #Store values of pitch angle
        reference_yaw.append(np.degrees(yaw))
        reference_pitch.append(np.degrees(pitch))
        reference_roll.append(np.degrees(roll))
        
        #Generate points of ellipsoid (not rotated)
        theta = rand(point_n)*np.pi
        phi = rand(point_n)*2*np.pi
        xx = ellipsoid_axis[2]*np.sin(theta)*np.cos(phi)
        yy = ellipsoid_axis[1]*np.sin(theta)*np.sin(phi)
        zz = ellipsoid_axis[0]*np.cos(theta)
        
        #Matrix of ellipsoid points
        points = np.transpose(np.vstack([xx, yy, zz]))
        
        #Compute rotation matrix and rotate the ellipse (ZYZ rotation)
        rot_matrix = np.dot(
            np.dot(rotation_matrix([0,0,1], yaw), rotation_matrix([0,1,0], np.pi/2 - pitch)), 
            rotation_matrix([0,0,1], roll))
        rotated_points = []
        for k in range(0,point_n):
            rotated_points.append(np.dot(rot_matrix, points[k,:]))
        rotated_points = np.array(rotated_points)
        
        #Fit the original points and store the orientation value
        ellipsoid, mean_squared_error_true = fit_ellipsoid(rotated_points[:,0],
                                                           rotated_points[:,1],
                                                           rotated_points[:,2])
        data_true = polyToParams3D(ellipsoid, False)
        initial_fit_pitch.append(data_true[3])
    
        #Define normal vectors of the 3 planes
        normal_cam0 = np.dot(rotation_matrix([0,0,1], np.radians(-36)), np.array([1,0,0]))
        normal_cam1 = np.array([1,0,0])
        normal_cam2 = np.dot(rotation_matrix([0,0,1], np.radians(36)), np.array([1,0,0]))
        
        #Project points to each of the 3 planes
        projected_point_cam0 = []
        projected_point_cam1 = []
        projected_point_cam2 = []
        for k in range(0,point_n):
            t0 = (-normal_cam0[0]* rotated_points[k,0]-normal_cam0[1]* rotated_points[k,1]-normal_cam0[2]* rotated_points[k,2])/(normal_cam0[0]**2 + normal_cam0[1]**2 + normal_cam0[2]**2)
            t1 = (-normal_cam1[0]* rotated_points[k,0]-normal_cam1[1]* rotated_points[k,1]-normal_cam1[2]* rotated_points[k,2])/(normal_cam1[0]**2 + normal_cam1[1]**2 + normal_cam1[2]**2)
            t2 = (-normal_cam2[0]* rotated_points[k,0]-normal_cam2[1]* rotated_points[k,1]-normal_cam2[2]* rotated_points[k,2])/(normal_cam2[0]**2 + normal_cam2[1]**2 + normal_cam2[2]**2)
            projected_point_cam0.append([rotated_points[k,0] + t0*normal_cam0[0], rotated_points[k,1] + t0*normal_cam0[1], rotated_points[k,2] + t0*normal_cam0[2]])
            projected_point_cam1.append([rotated_points[k,0] + t1*normal_cam1[0], rotated_points[k,1] + t1*normal_cam1[1], rotated_points[k,2] + t1*normal_cam1[2]])
            projected_point_cam2.append([rotated_points[k,0] + t2*normal_cam2[0], rotated_points[k,1] + t2*normal_cam2[1], rotated_points[k,2] + t2*normal_cam2[2]])    
    
        # Convert to np array
        projected_point_cam0 = np.array(projected_point_cam0)
        projected_point_cam1 = np.array(projected_point_cam1)
        projected_point_cam2 = np.array(projected_point_cam2)
        
        # Rotate the left and right plane to have a front view
        fw_points_cam0 = []
        fw_points_cam1 = projected_point_cam1
        fw_points_cam1[:,0] = 0
        fw_points_cam2 = []
        for k in range(0,len(projected_point_cam0)):
            fw_points_cam0.append(np.dot(rotation_matrix([0,0,1], np.radians(36)), projected_point_cam0[k,:]))
             
        for j in range(0,len(projected_point_cam2)):
            fw_points_cam2.append(np.dot(rotation_matrix([0,0,1], np.radians(-36)), projected_point_cam2[j,:]))
        
        # Set x = 0
        fw_points_cam0 = np.array(fw_points_cam0)
        fw_points_cam0[:,0] = 0
        fw_points_cam2 = np.array(fw_points_cam2)
        fw_points_cam2[:,0] = 0
        
        # Eliminate x-axis and keep only 2D points
        ellipse_points_cam0 = fw_points_cam0[:,1:3]
        ellipse_points_cam1 = fw_points_cam1[:,1:3]
        ellipse_points_cam2 = fw_points_cam2[:,1:3]
        
        # Initialize convex hull for each ellipse
        hull_cam0 = ConvexHull(ellipse_points_cam0)
        hull_cam1 = ConvexHull(ellipse_points_cam1)
        hull_cam2 = ConvexHull(ellipse_points_cam2)
        
        # Obtain contour of projected points cloud
        contour_ellipse_cam0 = ellipse_points_cam0[hull_cam0.vertices]
        contour_ellipse_cam1 = ellipse_points_cam1[hull_cam1.vertices]
        contour_ellipse_cam2 = ellipse_points_cam2[hull_cam2.vertices]
        
        #Fit the 2D points to ellipses and store orientation
        fit_cam0 = ls_ellipse(contour_ellipse_cam0[:,0], contour_ellipse_cam0[:,1])
        fit_cam1 = ls_ellipse(contour_ellipse_cam1[:,0], contour_ellipse_cam1[:,1])
        fit_cam2 = ls_ellipse(contour_ellipse_cam2[:,0], contour_ellipse_cam2[:,1])
        
        data_cam0 = polyToParams(fit_cam0, False)
        data_cam1 = polyToParams(fit_cam1, False)
        data_cam2 = polyToParams(fit_cam2, False)
        
        angle_cam0 = fit_ellipse(contour_ellipse_cam0[:,0], contour_ellipse_cam0[:,1])[4]*180/np.pi
        angle_cam1 = fit_ellipse(contour_ellipse_cam1[:,0], contour_ellipse_cam1[:,1])[4]*180/np.pi
        angle_cam2 = fit_ellipse(contour_ellipse_cam2[:,0], contour_ellipse_cam2[:,1])[4]*180/np.pi

        # Compute fitted orientation for the ellipse of each projection (Cam0, Cam1, Cam2)        
        ell_fit_ori_cam0.append(angle_cam0)
        ell_fit_ori_cam1.append(angle_cam1)
        ell_fit_ori_cam2.append(angle_cam2)
            
        # Compute minimum ell_fit_ori, keep sign ("Min" method)
        vec_ref = [np.abs(angle_cam0),np.abs(angle_cam1),np.abs(angle_cam2)]
        vec_ref_sign = [angle_cam0, angle_cam1, angle_cam2]
        
        ell_fit_ori_min.append(vec_ref_sign[np.argmin(vec_ref)])
        ell_fit_ori_mean.append(np.mean([np.abs(angle_cam0),np.abs(angle_cam1),np.abs(angle_cam2)]))
        ell_fit_ori_sign_mean.append(np.mean([angle_cam0,angle_cam1,angle_cam2]))
        
        # Compute the estimated orientation with orientation3D method (not shown in the paper)
        # Define camera orientations
        cam0_ori = np.radians(-36)*2
        cam1_ori = 0
        cam2_ori = np.radians(36)*2
        
        # Solve system of equations
        a = np.array([[np.sin(cam0_ori), np.cos(cam0_ori), 1], [np.sin(cam1_ori), np.cos(cam1_ori), 1], [np.sin(cam2_ori), np.cos(cam2_ori), 1]])
        b = np.array([np.abs(angle_cam0), np.abs(angle_cam1), np.abs(angle_cam2)])
        A, B, C = np.linalg.solve(a, b)
        A_prime = np.sqrt(A**2 + B**2)
        B_prime = np.arctan2(B, A)
        C_prime = C
        
        # Generate fitted sinus function
        x = np.arange(-np.pi, np.pi , 0.02)
        y   = A_prime*np.sin(2*x + B_prime) + C_prime
        
        # Store the minimum value if this is positive
        if min(y) >= 0:
            orientations3D.append(min(y))
            pass
        else:
            orientations3D.append(np.nan)
            pass               
        
        # Generate points of the 3 fitted ellipses
        phi = np.linspace(0, 2*np.pi, 300)#rand(n_point)*2*np.pi
        xx_fit_cam0 = data_cam0[2]*np.cos(phi)*np.cos(np.radians(data_cam0[4])) - data_cam0[3]*np.sin(phi)*np.sin(np.radians(data_cam0[4]))
        yy_fit_cam0 = data_cam0[2]*np.cos(phi)*np.sin(np.radians(data_cam0[4])) + data_cam0[3]*np.sin(phi)*np.cos(np.radians(data_cam0[4]))
        xx_fit_cam1 = data_cam1[2]*np.cos(phi)*np.cos(np.radians(data_cam1[4])) - data_cam1[3]*np.sin(phi)*np.sin(np.radians(data_cam1[4]))
        yy_fit_cam1 = data_cam1[2]*np.cos(phi)*np.sin(np.radians(data_cam1[4])) + data_cam1[3]*np.sin(phi)*np.cos(np.radians(data_cam1[4]))
        xx_fit_cam2 = data_cam2[2]*np.cos(phi)*np.cos(np.radians(data_cam2[4])) - data_cam2[3]*np.sin(phi)*np.sin(np.radians(data_cam2[4]))
        yy_fit_cam2 = data_cam2[2]*np.cos(phi)*np.sin(np.radians(data_cam2[4])) + data_cam2[3]*np.sin(phi)*np.cos(np.radians(data_cam2[4]))
              
        # Create matrix of points
        fw_fitted_cam0 = np.transpose(np.vstack([np.zeros_like(xx_fit_cam0), xx_fit_cam0, yy_fit_cam0]))
        fw_fitted_cam1 = np.transpose(np.vstack([np.zeros_like(xx_fit_cam1), xx_fit_cam1, yy_fit_cam1]))
        fw_fitted_cam2 = np.transpose(np.vstack([np.zeros_like(xx_fit_cam2), xx_fit_cam2, yy_fit_cam2]))
        
        # Rotete the ellipses on their correct plane
        rotated_points_cam0 = []
        rotated_points_cam1 = fw_fitted_cam1
        rotated_points_cam2 = []
        
        for k in range(0,len(fw_fitted_cam0)):
            rotated_points_cam0.append(np.dot(rotation_matrix([0,0,1], np.radians(-36)), fw_fitted_cam0[k,:]))
             
        for j in range(0,len(fw_fitted_cam2)):
            rotated_points_cam2.append(np.dot(rotation_matrix([0,0,1], np.radians(36)), fw_fitted_cam2[j,:]))
        
        # Convert to np array
        rotated_points_cam0 = np.array(rotated_points_cam0)
        rotated_points_cam2 = np.array(rotated_points_cam2)
        
        # Create array of x, y and z points
        xx= np.concatenate((rotated_points_cam0[:,0], rotated_points_cam1[:,0], rotated_points_cam2[:,0]), axis=0)
        yy= np.concatenate((rotated_points_cam0[:,1], rotated_points_cam1[:,1], rotated_points_cam2[:,1]), axis=0)
        zz= np.concatenate((rotated_points_cam0[:,2], rotated_points_cam1[:,2], rotated_points_cam2[:,2]), axis=0)
        
        # Fit the points to an ellipsoid and store the predicted orientation (ellipsoid_fit method)
        ellipsoid, mean_squared_error = fit_ellipsoid(xx,yy,zz)
        data = polyToParams3D(ellipsoid, False)
            
        ellipsoid_fit_mse.append(mean_squared_error)
        ellipsoid_fit_a.append(data[1][0])
        ellipsoid_fit_b.append(data[1][1])
        ellipsoid_fit_c.append(data[1][2])
        ellipsoid_fit_yaw.append(data[4])
        
        predicted_pitch = 90 - data[5]
        if predicted_pitch > 90 :
            predicted_pitch = 180 - predicted_pitch
        
        ellipsoid_fit_pitch.append(predicted_pitch)
        ellipsoid_fit_roll.append(data[6])
        
        estimated_pitch = data[3] ####
        if estimated_pitch > 90:
            estimated_pitch = 180 - estimated_pitch
        if estimated_pitch < -90:
            estimated_pitch = estimated_pitch + 180
            
        estimated_orientation.append(estimated_pitch)
    except ValueError as err:  #raised if `y` is empty.
        print(err)
        pass
    except RuntimeError as err:
        print(err)
        pass
    except TypeError as err:
        print(err)
        pass

############################################################################
# Store results in a dataframe
results_df = pd.DataFrame({'reference_a': reference_a, 
                           'reference_b': reference_b,
                           'reference_c': reference_c,
                           'reference_yaw': reference_yaw,
                           'reference_pitch': reference_pitch,
                           'reference_roll': reference_roll,
                           'initial_fit_pitch': initial_fit_pitch,
                           'ellipsoid_fit_yaw': ellipsoid_fit_yaw,
                           'ellipsoid_fit_pitch': ellipsoid_fit_pitch, 
                           'ellipsoid_fit_roll': ellipsoid_fit_roll,
                           'ellipsoid_fit_a': ellipsoid_fit_a,
                           'ellipsoid_fit_b': ellipsoid_fit_b,
                           'ellipsoid_fit_c': ellipsoid_fit_c,
                           'ellipsoid_fit_mse': ellipsoid_fit_mse,
                           'ell_fit_ori_cam0': ell_fit_ori_cam0,
                           'ell_fit_ori_cam1': ell_fit_ori_cam1,
                           'ell_fit_ori_cam2': ell_fit_ori_cam2,
                           'orientations3D': orientations3D, 
                           'ell_fit_ori_min':ell_fit_ori_min,
                           'ell_fit_ori_mean':ell_fit_ori_mean,
                           'ell_fit_ori_sign_mean':ell_fit_ori_sign_mean,
                           'ellipsoid_fit_ori': estimated_orientation
                           }).astype(float, errors = 'raise')

dist_df = pd.DataFrame({'Ref': reference_pitch,
                           'Min':  ell_fit_ori_min,
                           'Mean_abs': ell_fit_ori_mean,
                           'Mean': ell_fit_ori_sign_mean,
                           'Cam0': ell_fit_ori_cam0,
                           'Cam1': ell_fit_ori_cam1,
                           'Cam2': ell_fit_ori_cam2,
                           'Ellipsoid_fit':estimated_orientation
                           }).astype(float, errors = 'raise')


reference_p = np.abs(reference_pitch)


# Compute the errors for each method
errors_df = pd.DataFrame({
    'Min':(np.array(np.abs(ell_fit_ori_min))-np.array(reference_p)),
    'Mean_abs':(np.array(ell_fit_ori_mean)-np.array(reference_p)),
    'Mean': (np.array(np.abs(ell_fit_ori_sign_mean))-np.array(reference_p)),
    'Cam0':(np.array(np.abs(ell_fit_ori_cam0))-np.array(reference_p)),
    'Cam1':(np.array(np.abs(ell_fit_ori_cam1))-np.array(reference_p)),
    'Cam2':(np.array(np.abs(ell_fit_ori_cam2))-np.array(reference_p)),
    'Ellipsoid_fit':(np.array(np.abs(estimated_orientation))-np.array(reference_p))
    })

errors_df_sign = pd.DataFrame({
    'Min':(np.array(np.abs(ell_fit_ori_min))-np.array(reference_pitch)),
    'Mean_abs':(np.array(ell_fit_ori_mean)-np.array(reference_pitch)),
    'Mean': (np.array(ell_fit_ori_sign_mean)-np.array(reference_pitch)),
    'Cam0':(np.array(ell_fit_ori_cam0)-np.array(reference_pitch)),
    'Cam1':(np.array(ell_fit_ori_cam1)-np.array(reference_pitch)),
    'Cam2':(np.array(ell_fit_ori_cam2)-np.array(reference_pitch)),
    'Ellipsoid_fit':(np.array(estimated_orientation)-np.array(reference_pitch))
    })

# Save to pickle, such that if needed the results can be loaded directly below
results_df.to_pickle('/home/grazioli/tmp/ellipsoid_pickles/results_df.pkl')
dist_df.to_pickle('/home/grazioli/tmp/ellipsoid_pickles/dist_df.pkl')
errors_df.to_pickle('/home/grazioli/tmp/ellipsoid_pickles/errors_df.pkl')
errors_df_sign.to_pickle('/home/grazioli/tmp/ellipsoid_pickles/errors_df_sign.pkl')

# --------------------------------------------------------------------
# Load pickles
results_df = pd.read_pickle('/home/grazioli/tmp/ellipsoid_pickles/results_df.pkl')
dist_df = pd.read_pickle('/home/grazioli/tmp/ellipsoid_pickles/dist_df.pkl')
errors_df = pd.read_pickle('/home/grazioli/tmp/ellipsoid_pickles/errors_df.pkl')
errors_df_sign = pd.read_pickle('/home/grazioli/tmp/ellipsoid_pickles/errors_df_sign.pkl')
                         

# Plots as in the submitted manuscript
import seaborn as sns
import matplotlib as mpl

# Set font settings
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica']

# Define palette and line styles
palette = {'Min': 'black',
           'Mean_abs': 'red',
           'Mean': 'purple',
           'Cam0': 'grey',
           'Cam1': 'grey',
           'Cam2': 'grey',
           'Ellipsoid_fit': 'green'}

# Increase font size
sns.set(rc={'axes.labelsize': 20, 'axes.titlesize': 22, 'xtick.labelsize': 18, 'ytick.labelsize': 18})
sns.set_style('whitegrid')

# Create a figure and axis
plt.figure(figsize=(10, 6))
ax = sns.boxenplot(data=errors_df, palette=palette)

# Add a title and adjust ylabel
ax.set_title('Distribution of error on absolute orientation', fontsize=22)
ax.set_ylabel('Error [°]', fontsize=20)

# Adjust tick parameters for better visibility
ax.tick_params(axis='both', which='major', labelsize=16, width=2, length=6)

# Add grid lines
plt.grid(True, linestyle='--', linewidth=0.5)

# Show the plot
plt.tight_layout()  # Adjust layout to prevent clipping of labels

plt.savefig(out_path+'00_ellipsoid_error.png',dpi=450)
plt.savefig(out_path+'00_ellipsoid_error.pdf')
plt.show()
#-------------------------------------------------------------------
# Plot distribution (abs)
palette={ 'Ref':'blue',
         'Min':'black',
         'Mean_abs':'red',
         'Mean':'purple',
         'Cam0':'grey',
         'Cam1':'grey',
         'Cam2':'grey',
         'Ellipsoid_fit':'green'}


# Define line styles for each distribution
line_styles = {
    'Ref':'-',
    'Min': '-',
    'Mean_abs': '-',
    'Mean': '-',
    'Cam0': '-',
    'Cam1': '-',
    'Cam2': '-',
    'Ellipsoid_fit': '-'
}

# Create a figure and axis
plt.figure(figsize=(10, 6))
ax = sns.violinplot(data=dist_df, 
                    palette=palette,
                    split=True,
                    inner='quart',
                    inner_kws=dict(color=".8",linewidth=2))

# Add a title and adjust ylabel
ax.set_title('Distribution of orientation values', fontsize=22)
ax.set_ylabel('[°]', fontsize=22)
ax.set_ylim(-91,91)

# Adjust tick parameters for better visibility
ax.tick_params(axis='both', which='major', labelsize=16, width=2, length=6)

# Add grid lines
plt.grid(True, linestyle='--', linewidth=0.5)

# Show the plot
plt.tight_layout()  # Adjust layout to prevent clipping of labels


plt.savefig(out_path+'00_ellipsoid_dist.png',dpi=450)
plt.savefig(out_path+'00_ellipsoid_dist.pdf')
plt.show()
#------------------------------------------------------------------------------

from scipy.stats import shapiro

# Define a Function for the Shapiro test
def shapiro_test(data, alpha = 0.05):
    stat, p = shapiro(data)
    if p > alpha:
        print('Data looks Gaussian')
    else:
        print('Data look does not look Gaussian')
        

