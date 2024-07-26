"""
Codes to reproduce the evaluation of the estimators of orientation, based on
simulated aggregates projected to simulate 2D MASC images.

This code is used to produce the results of the manuscript by Grazioli et al
submitted to GRL in 2024, entitled "Observation of falling snowflakes orientation in
 sheltered and unsheltered sites"
 
 NOTE: it requires the aggregation package, by Jussi Leinonen, available here:
     https://github.com/jleinonen/aggregation

"""


import numpy as np
from scipy.ndimage import convolve, zoom
import math
from numpy.linalg import eig, inv
from aggregation import riming, rotator

# Setting for contour of snowflake
n_point = 100
phi = np.linspace(0, 2*np.pi, n_point)

# ---------------------------------------------------------------------------
# Some utility functions

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

def aggregate(mono_size=650e-6, mono_min_size=100e-6,
    mono_max_size=3000e-6, mono_type="dendrite", grid_res=35e-6,
    num_monos=5, riming_lwp=0.0, riming_mode="subsequent",
    rime_pen_depth=120e-6, compact_dist=0.0):

    mono_generator = riming.gen_monomer(psd="exponential", size=mono_size, 
        min_size=mono_min_size, max_size=mono_max_size,
        mono_type=mono_type, grid_res=grid_res, rimed=True)
        
    agg = riming.generate_rimed_aggregate(mono_generator, N=num_monos,
        align=False, riming_lwp=riming_lwp, riming_mode=riming_mode,
        rime_pen_depth=rime_pen_depth, compact_dist=compact_dist)

   
    agg.align() 
    agg.rotate(rotator.HorizontalRotator())
    agg.rotate(rotator.PartialAligningRotator(exp_sig_deg=25))
    
    
    #-------------
    
    # Get the orientation of principal axis
    PA = agg.principal_axes()
    PA /= np.sqrt((PA**2).sum(0))
    
    vector = PA[:,0]
    angle_z = np.arccos(vector[2] / np.linalg.norm(vector))
    angle_z_degrees = np.degrees(angle_z)
    pitch = 90-angle_z_degrees

    print('Hi')
    return agg, pitch


def projection(**kwargs):
    grid_size_3d = 32
    p_size = 256
    p_downscale = 2
    grid_res = 35e-6
    k = np.array([0.25,0.5,0.25])
    kernel_2d = k[None,:]*k[:,None]
    kernel_3d = k[None,None,:]*k[None,:,None]*k[:,None,None]
    agg, pitch = aggregate(**kwargs)

    cam_angles = np.array([
        -36.0, 0.0, 36.0
    ]) * (np.pi/180)
    
    # compute 2d projections
    def project(angle):
        p = agg.project_on_dim(direction=(angle, 0.0))
        (i,j) = np.mgrid[:p.shape[0],:p.shape[1]]
        i_c = int(round((i*p).sum()/p.sum()))
        j_c = int(round((j*p).sum()/p.sum()))
        i0 = p_size//2 - i_c
        i1 = i0+p.shape[0]
        j0 = p_size//2 - j_c
        j1 = j0+p.shape[1]
        if (i0 < 0) or (i1 >= p_size) or (j0 < 0) or (j1 > p_size):
            raise ValueError("Cannot fit projection in bounds.")
        p_eq = np.zeros((p_size,p_size), dtype=np.uint8)
        p_eq[i0:i1,j0:j1] = p
        return p_eq.astype(np.float32)

    proj = np.stack([project(a) for a in cam_angles], axis=-1)
    proj_any = (proj>0).any(axis=-1)
    (i,j) = np.mgrid[:proj.shape[0],:proj.shape[1]]
    i_act = i[proj_any]
    i_ext = max(proj.shape[0]//2-i_act.min(), i_act.max()-proj.shape[0]//2)
    j_act = j[proj_any]
    j_ext = max(proj.shape[1]//2-j_act.min(), j_act.max()-proj.shape[1]//2)
    proj_ext = max(i_ext,j_ext)+1
    proj_box = proj[
        proj.shape[0]//2-proj_ext:proj.shape[0]//2+proj_ext,
        proj.shape[1]//2-proj_ext:proj.shape[1]//2+proj_ext,
        :
    ]
    proj_size = proj_box.shape[0]*grid_res
    zoom_factor = p_size//p_downscale / proj_box.shape[0] + 1e-8
    proj = zoom(proj_box, (zoom_factor,zoom_factor,1), order=1)
    for k in range(proj.shape[-1]):
        proj[:,:,k] = convolve(proj[:,:,k], kernel_2d, mode='constant')
    proj[proj<0.5] = 0
    proj[proj>=0.5] = 1

    
    ext = (
        min(s0 for (s0,s1) in agg.extent)-grid_res*0.5,
        max(s1 for (s0,s1) in agg.extent)+grid_res*0.5
    )
    d_ext = (ext[1]-ext[0])/grid_size_3d
    ext = (ext[0]-d_ext, ext[1]+d_ext)
    grid_edges = np.linspace(ext[0], ext[1], grid_size_3d+1)
    grid_3d = np.zeros((grid_size_3d,grid_size_3d,grid_size_3d))
    cell_vol = (grid_edges[1]-grid_edges[0])**3

    for (i,gx0) in enumerate(grid_edges[:-1]):
        gx1 = grid_edges[i+1]
        for (j,gy0) in enumerate(grid_edges[:-1]):
            gy1 = grid_edges[j+1]
            for (k,gz0) in enumerate(grid_edges[:-1]):
                gz1 = grid_edges[k+1]
                X = agg.X
                in_cell = \
                    (gx0 <= X[:,0]) & (X[:,0] < gx1) & \
                    (gy0 <= X[:,1]) & (X[:,1] < gy1) & \
                    (gz0 <= X[:,2]) & (X[:,2] < gz1)

                cell_ice_vol = grid_res**3 * \
                    np.count_nonzero(in_cell)
                grid_3d[i,j,k] = cell_ice_vol / cell_vol
    grid_3d = convolve(grid_3d, kernel_3d, mode='constant').astype(np.float32)
    grid_3d_size = grid_edges[-1]-grid_edges[0]

    return (agg, proj, proj_size, grid_3d, grid_3d_size,pitch)


def create_random_snowflake():
    kwargs = {}
    kwargs["riming_lwp"] = 0
    kwargs["num_monos"] = max(10,int(round((7*np.random.rand())**2))) #1
    kwargs["mono_size"] = 500e-6 + 800e-6*np.random.rand()# 0.001
    mono_types = ["dendrite", "dendrite", "dendrite", "needle"]
    if kwargs["num_monos"] < 5:
        mono_types += ["rosette", "plate", "column"]
    
    kwargs["mono_type"] = np.random.choice(mono_types,1)[0] 
    kwargs["compact_dist"] = 0.62 * 35e-6 * np.random.rand()
    kwargs["riming_mode"] = np.random.choice(
        ["simultaneous", "subsequent"], 1)[0]

    print(kwargs)

    return (projection(**kwargs)+(kwargs,))

#----------------------------------------------------------------------------
# End of utility functions

"""
This code below creates random aggregate snowflake or monomer
following some reasonable constraints as given by the function 'create_random_snowflake'

The main limitation in this example is that only aggregates composed of the same monomers 
are generated. However the code 'aggregation' by Jussi Leinonen has also 
the option to generate combination of monomers. 

10'000 snowflakes are generated, rotaded according to a distribution centered at 0° and
with a standard deviation of 25°
"""

import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes
from skimage import measure

def get_orientation(snow_proj):
    oris =[100,100,100]
    cameras = {'cam0':[np.nan,np.nan,np.nan],
               'cam1':[np.nan,np.nan,np.nan],
               'cam2':[np.nan,np.nan,np.nan]}

    cam_vec = ['cam0','cam1','cam2']
    for i in [0,1,2]:
        img=snow_proj[1][:,:,i]
        
        binary_mask = img > 0
        binary_mask = binary_fill_holes(binary_mask)

        # --> Background is label 0 
        labels = measure.label(binary_mask)

        # Remove spurious stuffs. Force connectivity as in this case we know it is one aggregate 
        labels[labels > 1] = 1  

        #plt.imshow(labels)
        #plt.show()
        # compute properties of regions    
        region_props = measure.regionprops(labels)
        oris[i] = region_props[0].orientation*180./np.pi
        cameras[cam_vec[i]]=[region_props[0].major_axis_length,region_props[0].minor_axis_length, oris[i]]

    oris_abs = np.abs(oris)
    argmin   = np.argmin(oris_abs)

    return oris[argmin], cameras

reference_p = []
orientation = []
orientation_ell = []
orientation_cam0 = []
orientation_cam1 = []
orientation_cam2 = []

for i in range(10000):

    try:
        test=create_random_snowflake() 
        reference_pitch = test[5]
        
        ori,cameras = get_orientation(test)

        orientation = np.append(orientation,ori)

         #Left-Ellipse
        a1 = cameras['cam0'][0]
        b1 = cameras['cam0'][1]
        tilt_1 = np.radians(cameras['cam0'][2])
        orientation_cam0 = np.append(orientation_cam0,cameras['cam0'][2])
        
        x1 = np.zeros(n_point)
        y1 = a1*np.cos(phi)*np.cos(tilt_1) - b1*np.sin(phi)*np.sin(tilt_1)
        z1 = a1*np.cos(phi)*np.sin(tilt_1) + b1*np.sin(phi)*np.cos(tilt_1)
        
        ell1= np.transpose(np.vstack([x1, y1, z1]))

        #Middle-Ellipse
        a2 = cameras['cam1'][0]
        b2 = cameras['cam1'][1]
        tilt_2 = np.radians(cameras['cam1'][2])
        orientation_cam1 = np.append(orientation_cam1,cameras['cam1'][2])

        
        x2 = np.zeros(n_point)
        y2 = a2*np.cos(phi)*np.cos(tilt_2) - b2*np.sin(phi)*np.sin(tilt_2)
        z2 = a2*np.cos(phi)*np.sin(tilt_2) + b2*np.sin(phi)*np.cos(tilt_2)
        
        ell2= np.transpose(np.vstack([x2, y2, z2]))

        #Right-Ellipse
        a3 = cameras['cam2'][0]
        b3 = cameras['cam2'][1]
        tilt_3 = np.radians(cameras['cam2'][2])
        orientation_cam2 = np.append(orientation_cam2,cameras['cam2'][2])

        
        x3 = np.zeros(n_point)
        y3 = a3*np.cos(phi)*np.cos(tilt_3) - b3*np.sin(phi)*np.sin(tilt_3)
        z3 = a3*np.cos(phi)*np.sin(tilt_3) + b3*np.sin(phi)*np.cos(tilt_3)
        
        ell3= np.transpose(np.vstack([x3, y3, z3]))

        axis = [0, 0, 1]
        theta = np.radians(36)
        additional_rotation = np.radians(180)

        #Rotate the left and right ellpse
        ell1_rot=[]
        ell2_rot=[]
        ell3_rot=[]
        for k in range(0,n_point):
            ell1_rot.append(np.dot(rotation_matrix(axis, additional_rotation + theta), ell1[k,:]))
            ell2_rot.append(np.dot(rotation_matrix(axis, additional_rotation), ell2[k,:]))
            ell3_rot.append(np.dot(rotation_matrix(axis, additional_rotation - theta), ell3[k,:]))

        #Convert from list to matrix
        ellipse1_rot = np.array(ell1_rot)
        ellipse2_rot = np.array(ell2_rot)
        ellipse3_rot = np.array(ell3_rot)

        #Create 3 vector containing all coordinates in x, y and z
        xx= np.concatenate((ellipse1_rot[:,0], ellipse2_rot[:,0], ellipse3_rot[:,0]), axis=0)
        yy= np.concatenate((ellipse1_rot[:,1], ellipse2_rot[:,1], ellipse3_rot[:,1]), axis=0)
        zz= np.concatenate((ellipse1_rot[:,2], ellipse2_rot[:,2], ellipse3_rot[:,2]), axis=0)    
        
        #Fit the points to an ellipsoid and store the predicted orientation
        ellipsoid, mean_squared_error = fit_ellipsoid(xx,yy,zz)
        data = polyToParams3D(ellipsoid, False)

        estimated_pitch = data[3] ####
        if estimated_pitch > 90:
            estimated_pitch = 180 - estimated_pitch
        if estimated_pitch < -90:
            estimated_pitch = estimated_pitch + 180
        
        orientation_ell=np.append(orientation_ell,estimated_pitch)
        reference_p    =np.append(reference_p,reference_pitch)

        if np.abs(estimated_pitch) == 90:
            print('Test')

        # Print
        print(i,
              np.round(reference_pitch),
              np.round(ori),
              np.round(estimated_pitch),
              np.round(cameras['cam0'][2]),
              np.round(cameras['cam1'][2]),
              np.round(cameras['cam2'][2]))

    except Exception as error:
        print("Something went wrong, skipping", error)


#-----------------------------------------------------------------------------


from scipy.stats import norm
import seaborn as sns
import pandas as pd

#-------------------------------------------------------------------------------
# Plotting distribution of orientation

orientation_mean = (orientation_cam0 + orientation_cam1 + orientation_cam2)/3. 
orientation_median = [np.median([orientation_cam0[i],
                                 orientation_cam1[i],
                                 orientation_cam2[i]]) 
                      for i in range(len(orientation_cam0))] 

orientation_mean_abs = (np.abs(orientation_cam0)+np.abs(orientation_cam1)+np.abs(orientation_cam2))/3.

errors_df = pd.DataFrame({
    'Min':(np.array(np.abs(orientation))-np.array(np.abs(reference_p))),
    'Mean': (np.array(np.abs(orientation_mean))-np.array(np.abs(reference_p))),
    'Mean_abs': (np.array(orientation_mean_abs)-np.array(np.abs(reference_p))),
    'Cam0':(np.array(np.abs(orientation_cam0))-np.array(np.abs(reference_p))),
    'Cam1':(np.array(np.abs(orientation_cam1))-np.array(np.abs(reference_p))),
    'Cam2':(np.array(np.abs(orientation_cam2))-np.array(np.abs(reference_p))),
    'Ellipsoid_fit':(np.array(np.abs(orientation_ell))-np.array(np.abs(reference_p)))
    })


dist_df = pd.DataFrame({   'Min':  orientation,
                           'Mean_abs': orientation_mean_abs,
                           'Ref': reference_p,
                           'Mean': orientation_mean,
                           'Cam0': orientation_cam0,
                           'Cam1': orientation_cam1,
                           'Cam2': orientation_cam2,
                           'Ellipsoid_fit':orientation_ell
                           }).astype(float, errors = 'raise')

# Save to pickle as the process above is very cumbersome on a standard laptop if not 
# parallelized

dist_df.to_pickle('/home/grazioli/tmp/aggregate_pickles/dist_df.pkl')
errors_df.to_pickle('/home/grazioli/tmp/aggregate_pickles/errors_df.pkl')

#----------------------------------------------------
# PLOTS as in the submitted manuscript
out_path  = '/home/grazioli/Documents/Publications/Grazioli_GRL_2024/Raw_images/'

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
    'Ellipsoid_fit': ':'
}

import matplotlib as mpl

# Set font settings
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica']

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

# Customize legend if needed
# plt.legend(title='Legend', title_fontsize='16', fontsize='12', loc='upper right')

# Show the plot
plt.tight_layout()  # Adjust layout to prevent clipping of labels


plt.savefig(out_path+'00_aggregate_error.png',dpi=450)
plt.savefig(out_path+'00_aggregate_error.pdf')
plt.show()

###########################3
# Figure for the full distribution
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

# Customize legend if needed
# plt.legend(title='Legend', title_fontsize='16', fontsize='12', loc='upper right')

# Show the plot
plt.tight_layout()  # Adjust layout to prevent clipping of labels


plt.savefig(out_path+'00_agg_dist.png',dpi=450)
plt.savefig(out_path+'00_agg_dist.pdf')
plt.show()


#---------------------------------------------------------
#----------Normality tests
from scipy.stats import shapiro

# Define a Function for the Shapiro test
def shapiro_test(data, alpha = 0.05):
    stat, p = shapiro(data)
    if p > alpha:
        print('Data looks Gaussian')
    else:
        print('Data look does not look Gaussian')
        





