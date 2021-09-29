#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 13:57:30 2017

@author: jleftley
"""

import numpy as np
import matplotlib.pylab as plt
import emcee
#import scipy.optimize as op
#import astropy.modeling.models as mod
#import oifits3 as oi
#from astropy.io import fits
#import matplotlib.gridspec as grd
#import gc
#from multiprocessing import Pool
# import corner
#import sys
#from copy import deepcopy as dc
import datetime
from UVplotter2py3 import UVpoint
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
import galario
if galario.HAVE_CUDA:
    from galario import double_cuda as double
    print('GPU used')
else:
    from galario import double as double
#from galario import double as double
from astropy.io import fits
from copy import deepcopy as dc
from glob import glob
from tqdm import tqdm
from time import sleep
# double.threads(32)


f=glob('../../NGC1068FinalLM/*.fits') ## Fits files of 1068

paranal=EarthLocation.of_site('paranal')
ESO323=SkyCoord.from_name('NGC1068')


def Gauss(amp,mx,my,stdx,stdy,theta,varx,vary):
    Ga=(np.cos(theta)**2)/(2*stdx**2)+(np.sin(theta)**2)/(2*stdy**2)
    Gb=(-np.sin(2*theta))/(2*stdx**2)+(np.sin(2*theta))/(2*stdy**2)
    Gc=(np.sin(theta)**2)/(2*stdx**2)+(np.cos(theta)**2)/(2*stdy**2)
    gaussian=amp*np.exp(-Ga*(varx-mx)**2-Gb*(varx-mx)*(vary-my)-Gc*(vary-my)**2)
    return gaussian


def IMpoint(pf,x_off,y_off, nxy,dxy,scale,XL,YL):
    ''' Produces an image comprised of point sources for GALARIO to input.
    Variable inputs should be an array with length No. of Point Sources - 1.
    Physical units for variables should be input in mas, angle should be in degrees, x,y (u,v) are in meters '''
    
    pixel_scale_conv=dxy*180*3600*1000/np.pi
    
    # Create a blank image and coords, changing image size will not effect the scaling, changing the step will.
    x = np.arange(-pixel_scale_conv*(nxy/2-1), pixel_scale_conv*(nxy/2+1), pixel_scale_conv)
    y = np.arange(-pixel_scale_conv*(nxy/2-1), pixel_scale_conv*(nxy/2+1), pixel_scale_conv)
    xx, yy = np.meshgrid(x, y)
    comp_IM=np.zeros((len(y),len(x)))
    
    if len(x)!=nxy:
        raise ValueError('Size of image incorrect')
    
    photx=np.average(x_off,weights=pf)
    photy=np.average(y_off,weights=pf)
    
    xcentre=int(nxy/2)
    ycentre=int(nxy/2)
    
    xpix=-(x_off-photx)
    ypix=(y_off-photy)

    for i in range(len(pf)):
        idv_gauss=Gauss(pf[i],xpix[i],ypix[i],pixel_scale_conv*1,pixel_scale_conv*1,0,xx,yy)
        comp_IM += idv_gauss
        
    comp_IM += Gauss(scale,-(XL-photx),YL-photy,50/2.335,70/2.335,0,xx,yy)
    comp_IM = comp_IM/np.sum(comp_IM)
    return comp_IM



def uv_model_point(perc_resf,x_off,y_off,scale,XL,YL,u_co,v_co, nxy, dxy):
    '''Produce a model uv data set. Variable inputs should be an array with length No. of Gaussians. Wavelength is currently a single value.
    Physical units for variables should be input in mas, angle should be in degrees, x,y (u,v) are in meters'''
    
    #Produce Image
    Im = IMpoint(perc_resf,x_off,y_off, nxy,dxy,scale,XL,YL)
    
    #produce uv data -- Pixel scale is set at 10 micro-arcseconds per pixel
#    pixel_scale = 1e-2*np.pi*1e-3/(3600*180)
#    
#    #double pixel scale
#    pixel_scale = pixel_scale*2
    pixel_scale=dxy
    


    model_uv = double.sampleImage(Im,pixel_scale,u_co,v_co,origin='lower')
    
    #covert complex data to Visibility and Phase
    model_vis = (np.absolute(model_uv))**2#-scale
#    model_vis=(np.sum(np.real(model_uv)**2)+np.sum(np.imag(model_uv)**2)-
#               np.sum(np.var(np.real(model_uv)))-np.sum(np.var(np.imag(model_uv))))-scale
#    model_phase = np.angle(model_uv) * 180/np.pi
    
    return model_vis, model_uv
    

def lnlike_point(theta, lnf, scale,XL,YL, u_co, v_co, vis, viserr, u_phase, v_phase, c_phase,c_phaseerr, nxy, dxy):
    ''' Log likelihood function. u_co, v_co, u_phase, v_phase, and wavelen are in meters; phases and angle should be in degrees.
    u_phase and v_phase should be of shape (n,3) for each set of coords with each set being [T3-T2,T2-T1,T3-T1].'''
    
    perc_resf,x_off,y_off = theta
    
    #Create model vis and phase
    #For closure phase handling option 2 (see below) append phase uv's to full list
    
    utot=np.append(u_co, u_phase.flatten())
    vtot=np.append(v_co, v_phase.flatten())
    
    
    model_vis, model_diffphase = uv_model_point(perc_resf,x_off,y_off,scale,XL,YL,utot,vtot, nxy, dxy)
    
    #convert diff phase into closure phase
    
    #split off vis and closure phase comp
    model_vis=model_vis[:len(u_co)]
    model_cdiffphase=model_diffphase[len(u_co):].reshape(u_phase.shape)
    
    #model_cphase=model_cdiffphase[:,0]+model_cdiffphase[:,1]-model_cdiffphase[:,2]
    trip = model_cdiffphase[:,0]*model_cdiffphase[:,1]*np.conj(model_cdiffphase[:,2])
    model_cphase = -np.angle(trip,deg=True) #####For N band remove the negative of image will be rotated by 180 deg
    
    
    if True in np.isnan(model_vis) or True in np.isnan(model_cphase):
        model_vis=np.zeros(model_vis.shape)-99999
        model_cphase=np.zeros(model_cphase.shape)-99999
    
# =============================================================================
#       Consider importing a dictionary due to the large number of inputs
#       Could import the fits files but may want custom vis and phase data
    
#       Also consider making robust to phases with no vis counterparts --- Done
# =============================================================================
    '''
    #option 1 for phase selection
    model_cphase=[]
    for i in range(len(u_phase)):
        u1diff=np.abs(u_co-u_phase[i][0])
        u2diff=np.abs(u_co-u_phase[i][1])
        u3diff=np.abs(u_co-u_phase[i][2])
        pos1_ind=np.argmin(u1diff)
        pos2_ind=np.argmin(u2diff)
        pos3_ind=np.argmin(u3diff)
        
        if u1diff[pos1_ind]>1e-3:
            u1diff=np.abs(-u_co-u_phase[i][0])
            pos1_ind=np.argmin(u1diff)
            if u1diff[pos1_ind]>1e-3:
                raise ValueError('No visibility associated with the '+str(i)+' index closure phase')
    '''

    #option 2 is appending the closure phase uv's to the model --more robust but could be slower ^^^^^^

            
    residuals=np.abs(c_phase-model_cphase)
    residuals[residuals>180]=360-residuals[residuals>180]
    
    # May need a 2nd lnf
    inv_sigma2_vis = 1.0/(viserr**2 + model_vis**2*np.exp(2*lnf))
    inv_sigma2_cphase = 1.0/(c_phaseerr**2 + model_cphase**2*np.exp(2*lnf))
    return -0.5*(np.nansum((vis-model_vis)**2*inv_sigma2_vis - np.log(inv_sigma2_vis))+
                 np.nansum((residuals)**2*inv_sigma2_cphase - np.log(inv_sigma2_cphase)))


def lnprior_point(theta,lnf,scale,dxy):
    ''' log prior '''
    perc_resf,x_off,y_off= theta
    pixel_scale_conv=dxy*180*3600*1000/np.pi
    photx=np.average(x_off,weights=perc_resf)
    photy=np.average(y_off,weights=perc_resf)
    xpix=-(x_off-photx)/pixel_scale_conv
    ypix=(y_off-photy)/pixel_scale_conv
    
    # create flat prior
    ####### Must be a good way to clean this up!
    
#    if not (np.all(perc_resf>-10) and np.all(perc_resf<10)) or not -0.001 < perc_resf.sum() < 0.001 or not (np.all(x_off>-15) and np.all(x_off<15)) or not (np.all(y_off>-15) and np.all(y_off<15)) or not -15.0 < lnf < 15.0 or not 0.00 < scale < 0.3:
#        return -np.inf
    if not np.all(perc_resf>0.0) or not (np.sum(perc_resf)>0.999 and np.sum(perc_resf)<1.0001) or not (np.all((x_off-photx)>-45) and np.all((x_off-photx)<45)) or  not (np.all((y_off-photy)>-45) and np.all((y_off-photy)<45)) or not -15.0 < lnf < 15.0 or not 0.00 < scale < 0.9:
#    if not np.all(perc_resf>0.0) or not (np.sum(perc_resf)>0.999 and np.sum(perc_resf)<1.0001) or not (np.all((x_off-photx)>-10) and np.all((x_off-photx)<10)) or  not (np.all((y_off-photy)>-10) and np.all((y_off-photy)<10)) or not -15.0 < lnf < 15.0 or not 0.00 < scale < 0.3:
        return -np.inf
    return 0.0
    
def lnprior_point_unfixed(theta,lnf,scale,XL,YL,dxy):
    ''' log prior '''
    perc_resf,x_off,y_off= theta
    pixel_scale_conv=dxy*180*3600*1000/np.pi
    photx=np.average(x_off,weights=perc_resf)
    photy=np.average(y_off,weights=perc_resf)
    xpix=-(x_off-photx)/pixel_scale_conv
    ypix=(y_off-photy)/pixel_scale_conv
    # print(np.sum(perc_resf)>999.0)
    # create flat prior
    ####### Must be a good way to clean this up!
    
#    if not (np.all(perc_resf>-10) and np.all(perc_resf<10)) or not -0.001 < perc_resf.sum() < 0.001 or not (np.all(x_off>-15) and np.all(x_off<15)) or not (np.all(y_off>-15) and np.all(y_off<15)) or not -15.0 < lnf < 15.0 or not 0.00 < scale < 0.3:
#        return -np.inf
    if not np.all(perc_resf>0.0) or not (np.sum(perc_resf)>999.0 and np.sum(perc_resf)<1001.0) or not (np.all((x_off-photx)>-45) and np.all((x_off-photx)<45)) or  not (np.all((y_off-photy)>-45) and np.all((y_off-photy)<45)) or not -15.0 < lnf < 15.0 or not 0.00 < scale < 0.5 or not (XL-photx)>-45 or not (XL-photx)<45 or not (YL-photy)>-45 or not (YL-photy)<45:
#    if not np.all(perc_resf>0.0) or not (np.sum(perc_resf)>0.999 and np.sum(perc_resf)<1.0001) or not (np.all((x_off-photx)>-10) and np.all((x_off-photx)<10)) or  not (np.all((y_off-photy)>-10) and np.all((y_off-photy)<10)) or not -15.0 < lnf < 15.0 or not 0.00 < scale < 0.3:
        return -np.inf
    return 0.0


def lnprob_point(theta, u_co, v_co, vis, viserr, u_phase, v_phase, c_phase,c_phaseerr, nxy, dxy):
    ''' Log probability, introduce flattened array of all variables in order '''
    
    #Sort arguments here
    
    lnf=theta[-1]
    scale=theta[-4]
    XL=theta[-3]
    YL=theta[-2]
    
    theta=np.reshape(theta[:-4],(3,int(len(theta[:-4])/3)))
    
    
    lp = lnprior_point_unfixed(theta,lnf,scale,XL,YL,dxy)
    
    
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_point(theta, lnf, scale,XL,YL, u_co, v_co, vis, viserr, u_phase, v_phase, c_phase,c_phaseerr, nxy, dxy)


def RunGravEmcee(FileList,ESO323,paranal,NGauss,wavcent):
#    m=0
    BICc=999999
 #    [array([ 0.14950884,  0.84087721,  0.00961394,  0.        , -0.39345396,
 #         3.7053691 ,  0.        , -0.37194343,  1.88568882,  0.05969262,
 #        -3.27962726]),
 # array([ 0.15010857,  0.48675974,  0.00950677,  0.35362493,  0.        ,
 #        -0.294376  ,  3.62630263, -0.33798657,  0.        ,  0.79550092,
 #         2.2357734 ,  0.58273104,  0.09683767, -2.77878568]),
 # array([ 0.15000389,  0.01203808,  0.36099492,  0.01896371,  0.4579994 ,
 #         0.        , -1.44400499,  0.06258258, -2.05324145,  0.31495417,
 #         0.        ,  4.79092355, -0.80542688,  1.25380632, -0.51828286,
 #         0.01189053, -2.82677547])]
    #resarr=np.array([ 0.15      ,  0.84023372,  0.00976628,  0.        , -0.302614  ,
                     # 3.58471713,  0.        , -0.73768995,  2.09760997,  0.08717338,
                     # -2.15478112])
    resarr=None
    
    #### Starting values from initial run, should make resarr function input
    if NGauss==6:
        resarr=np.array([0.25,0.25,0.125,0.125,0.125,0.125,0,-3,-6,6,3,-1,-3,3,9,-0,5,-17,0.1,-0.6])
    if NGauss==7:
        resarr=np.array([  0.14522231,   0.18315319,   0.1187228 ,   0.14729787,
             0.16475089,   0.07784906,   0.16300388,   0.        ,
             9.30694332,  -4.2688833 ,   1.09569878,   5.56764035,
            -7.43808185,   3.67309052,   0.        ,  -6.85843067,
            -1.4639342 , -11.82446919,   0.8949273 ,  -6.03332575,
            -4.08547887,   0.08377976,  -0.58815455])
    elif NGauss==9:
        if wavcent==3.45:
            resarr=np.array([ 1.15838929e+02,  5.90911311e+01,  7.86437602e+01,  1.43777926e+02,
                                1.50855790e+02,  1.52604470e+02,  1.06144697e+02,  1.00571076e+02,
                                9.24528472e+01,  5.77796185e+00,  1.77355436e+00,  2.27736151e+00,
                                1.99201310e+00, -1.67904493e-01, -3.32517173e+00, -5.99219743e+00,
                                7.31589155e+00,  2.34813700e+00,  4.01225781e+00, -2.10031294e+00,
                                1.06967046e+01,  3.57232126e+00,  2.74989921e+00,  3.50771000e+00,
                                9.63592032e+00, -2.39054642e+00, -9.55805696e+00,  6.69792445e-02,
                               -1.60782242e-01, -1.10575154e+00, -1.20194060e+00])
        elif wavcent==3.55:
            resarr=np.array([ 1.15995190e+02,  5.79910869e+01,  7.91573264e+01,  1.43535379e+02,
              1.50867765e+02,  1.53347343e+02,  1.06245535e+02,  1.00633143e+02,
              9.27875930e+01,  5.82133370e+00,  1.32781794e+00,  2.41493802e+00,
              2.13387872e+00,  2.64395910e-02, -3.27165656e+00, -6.08429345e+00,
              7.47590668e+00,  2.39535786e+00,  4.44448801e+00, -1.14223129e+00,
              1.16285603e+01,  3.39021490e+00,  2.84951381e+00,  3.64457001e+00,
              9.99834834e+00, -2.26026923e+00, -9.74741897e+00, # 3.28613061e-01,
              0.1, 0, 0,
             -1.19264123e+00])
        elif wavcent==3.65:
            resarr=np.array([139.33440321,  78.883856  ,  48.48312139, 121.83477974, 161.56858296,
             133.62859057, 126.45263007,  98.37453337,  90.51515441,  13.75898254,
               6.94117615,  11.01601202,  10.47683198,   8.93997083,   5.61771744,
               2.09954157,  15.40155163,  10.88868058,  11.08160335,   8.73248825,
              18.17578227,   9.09764764,   8.8942426 ,  10.10658747,  16.32335389,
               4.82827657,  -4.10568499,   #0.47720319,
               0.1, 0, 0,  -1.20269184])
        elif wavcent==3.8:
            resarr=np.array([135.38108003,  78.55159185,  64.10357293, 134.39228247, 159.27373853,
             116.87705603, 107.62909564,  97.20412088, 105.71029189,   2.42995998,
              -3.76924134,  -0.80812551,  -0.50669764,  -2.8579857 ,  -6.20866229,
              -9.46062285,   4.57011108,  -0.766354  ,   1.40133794,  -2.57654545,
              10.51733933,  -1.44627108,  -1.07868138,   0.53496205,   6.7434967,
              -5.53193117, -13.81113405,   #0.39018033,
              0.1, 0, 0,  -1.27318916])
        elif wavcent==3.9:
            resarr=np.array([ 1.16434237e+02,  4.95395167e+01,  5.48238570e+01,  1.56692079e+02,
              1.69097874e+02,  1.48806720e+02,  1.14901893e+02,  9.15816489e+01,
              9.78587391e+01,  3.33483728e+00,  2.45714495e-01,  8.79199284e-01,
              4.71451201e-01, -2.83662088e+00, -4.63194014e+00, -8.87361110e+00,
              5.59496681e+00,  1.14962781e-01, -1.74177651e+00, -8.61425538e+00,0.1, 0, 0,  -1.20269184])
        elif wavcent==4.65:
            resarr=np.array([ 1.25348655e+02,  6.53919976e+01,  7.42969814e+01,  1.29123299e+02,
              1.63230319e+02,  1.18205247e+02,  1.19461468e+02,  8.66178755e+01,
              1.17443129e+02,  2.15404969e+00,  1.39598046e+00,  2.32994201e+00,
             -2.42977827e-01,  2.60363949e+00, -3.35569806e+00, -5.25083085e+00,
              8.93722363e+00, -5.87809001e-01,  2.43824394e+00,  1.47736626e+00,
              1.24246181e+01,  2.41828667e+00,  2.82833223e+00,  5.81500270e+00,
              8.88522323e+00, -1.25939691e-01, -1.20595179e+01, # 4.94703753e-01,
              0.1, 0, 0,
             -1.49537677e+00])
        elif wavcent==4.8:
            resarr=np.array([189.83583704,  54.79953441,  88.30224294, 139.8844352 , 184.17200231,
              97.56031341, 146.79962319,  49.36153152,  48.63209826,  -1.97993683,
              -4.63092002,   3.47489848,  -5.53666355,   1.52168003, -10.86516145,
              -4.37871345,   6.62640901,   1.1752101 ,   4.02657572,  -0.47804081,
               8.1226475 ,   5.38007513,   1.07501977,  14.14054798,   4.78653631,
               0.9271004 , -11.0995111 , #  0.49806851,
               0.1, 0, 0,  -1.53312206])
    elif NGauss==27:
        if wavcent==3.25:
            resarr=np.array([48.8976726 , 42.23165736, 46.80231131, 26.91319025, 29.1616891 ,
                           23.88447155, 15.50981437, 17.31955982, 18.39457567, 41.78955899,
                           41.44089551, 40.48958067, 55.28480088, 51.75603571, 54.4174208 ,
                           47.97249025, 45.14545862, 43.7849676 , 38.15442011, 40.65037723,
                           41.86010618, 32.61027605, 33.15375053, 31.19503192, 30.1934474 ,
                           31.57199187, 29.593694  , 12.56183186, 13.92652594, 16.01912576,
                            7.0051236 ,  5.88548895,  9.24336492,  8.92533943, 10.64949905,
                           11.09735723, 10.55455398, 11.26126988, 10.00040923,  6.95542333,
                            8.63897734,  9.90430539,  4.75335974,  5.41349601,  7.1059712 ,
                            1.01063905,  2.05033254,  3.43290868, 14.7357419 , 14.59736602,
                           17.14021607,  9.1982824 ,  5.33846771, 11.50499547, 12.18000105,
                            8.45619991, 10.0826723 ,  8.36398429,  9.64011855,  8.3560227 ,
                           19.16570518, 19.34798016, 17.95541486,  8.32437227,  6.57657405,
                            8.12420349,  9.75157476,  8.52955445, 13.64426383, 10.77550501,
                           13.09417948,  7.89219863, 16.20666332, 17.84198436, 13.89873252,
                            5.07172337,  7.80987557,  4.53608168, -0.25354809, -4.35454067,
                           -4.57810912,  0.12553132, -0.44642949,  4.88306308, -1.12685183])
        elif wavcent==3.5:
            resarr=np.array([50.03811687, 41.76989829, 47.25537241, 27.81995574, 28.97545188,
                    23.32565041, 15.12610242, 16.8996816 , 16.83255573, 41.50177139,
                    41.26010762, 41.31066   , 55.52556346, 51.67600576, 53.15179404,
                    45.91469498, 45.16427646, 44.1322914 , 38.99638857, 41.38901937,
                    42.62254863, 32.48648418, 33.97198901, 30.4158587 , 30.68064842,
                    31.09054647, 30.56344352, 12.86187476, 14.60055771, 16.8022336 ,
                     7.28226133,  5.59490484,  8.73929247,  9.44356425, 11.3902456 ,
                    10.9680435 , 10.39129054, 11.11310161, 10.83293014,  7.00485589,
                     9.26184533, 10.3287377 ,  5.67449481,  5.32268078,  6.86532847,
                     0.8993758 ,  2.74665038,  3.12023234, 14.47692457, 15.21700141,
                    17.999668  ,  9.44266853,  5.51291969, 13.04033284, 12.52672357,
                     8.73283095, 10.35098867,  8.31785528,  8.26596225,  8.28847784,
                    19.50445393, 18.9158704 , 18.42201288,  9.17713255,  6.85061275,
                     8.17613992, 10.39602169,  8.9349952 , 13.96200336,  9.70024987,
                    12.32926708,  8.96112242, 16.65557566, 16.58115341, 14.10757106,
                     4.84574387,  7.56735472,  6.51852081, -0.60608244, -4.97995144,
                    -4.39341702,  0.10795217, -0.87328982,  4.19769128, -0.83765308])
        elif wavcent==3.6 or wavcent==3.7:
             resarr=np.array([49.28195591, 45.63149805, 45.14547739, 26.34382658, 28.72380407,
                    23.83415918, 17.02017213, 15.71524141, 13.95002295, 41.04502273,
                    41.01102348, 41.7445898 , 55.4278084 , 51.70452204, 55.29271455,
                    45.36221974, 45.30366537, 44.72638313, 41.01061228, 41.08446469,
                    42.79961059, 33.2956861 , 32.68389928, 29.88880879, 30.60180639,
                    31.46331455, 30.04793221, 12.6591575 , 14.94577483, 16.06783494,
                     7.90455223,  6.40447453,  7.97124062,  9.53719787, 10.20015726,
                     8.56504102, 10.43243109, 10.58006024, 11.14487991,  6.56317189,
                    10.2730657 ,  8.19432647,  5.93505649,  4.40944991,  7.40638491,
                     0.93013217,  2.63853066,  2.85805658, 13.92780735, 13.14349209,
                    17.71872875,  7.64867865,  6.82532459, 14.37492931, 12.6131575 ,
                     8.22782555, 10.67437171,  8.43278714,  8.68555079,  7.52719295,
                    17.00580764, 19.2341727 , 17.96384322,  8.61795132,  7.32712511,
                     7.80693159,  9.78167092, 10.67140182, 11.60573712,  9.68495328,
                    12.37611158,  8.74560265, 16.32702509, 16.3275652 , 14.66612454,
                     5.31603137,  5.5822494 ,  5.05490798, -0.63480666, -3.14188005,
                    -4.62074117,  0.14997389,  0.7919444 ,  1.86115915, -1.11289144])
        elif wavcent==3.8:
             resarr=np.array([48.27424708, 46.74017358, 45.56925593, 26.14151829, 28.00546574,
                    23.84069559, 15.73627815, 16.3305028 , 15.64946055, 40.65965399,
                    41.3622798 , 41.57687491, 55.02467524, 52.95176655, 54.40865864,
                    44.11750695, 45.05615276, 44.13282307, 43.21478053, 40.37686551,
                    42.93786752, 33.97551354, 32.77148309, 32.06613003, 28.39973689,
                    30.48233094, 29.45433648, 11.88753546, 15.13548895, 15.85556755,
                     7.74059738,  6.11157683,  7.88110625, 10.10792948,  9.97092978,
                     7.44636233, 11.23457972, 11.2098301 , 10.23646502,  6.39827386,
                     9.02168154,  8.27114623,  6.45654369,  4.8927023 ,  6.28823756,
                     0.55865539,  2.32118839,  2.27371286, 13.61441734, 12.3551942 ,
                    16.92476644,  7.24307825,  7.10640675, 14.86548532, 12.73102198,
                     7.82703342, 11.29125543,  8.28887691,  9.08374304,  8.37460769,
                    17.08207497, 19.60351663, 17.59819354,  8.78181319,  7.75708201,
                     8.99087573,  9.96124256,  9.12600404, 10.65764214,  9.87659991,
                    12.58495193,  8.2463037 , 17.65644059, 15.53119695, 14.09135778,
                     4.97456462,  6.89342919,  5.81954148, -1.4789194 , -3.98523376,
                    -5.85054316,  0.1696268 ,  1.14842757,  0.30617765, -1.30541897])
        # elif wavcent==3.9:
        #      resarr=np.array([47.67193423, 46.23341292, 45.44779108, 26.47808218, 28.08194188,
        #             23.37985759, 16.66889289, 15.69058708, 13.95372519, 41.30497295,
        #             41.62037297, 40.55644262, 54.98083007, 51.20898211, 54.20185117,
        #             45.14232338, 44.74898606, 43.74128966, 42.08678117, 40.85611455,
        #             43.24900225, 34.08375622, 33.19388172, 30.41373612, 32.09299048,
        #             31.72574643, 31.13847403, 13.06917383, 13.49345543, 15.58261741,
        #              6.89127359,  6.34075697,  7.77873329, 10.27738501, 10.99035564,
        #              8.42407536, 10.35660859, 11.37918208, 10.87623032,  6.84857588,
        #              9.11204615,  7.79370041,  6.03901347,  5.21138008,  8.36320002,
        #              0.95082791,  3.11913204,  1.29658039, 15.27682044, 13.66107035,
        #             17.94928507,  8.04493241,  6.93309527, 13.43821399, 11.61877406,
        #              8.67842155, 11.84852283,  9.14343115,  8.0599932 ,  6.45863369,
        #             17.7260256 , 19.14920619, 18.15708657,  7.86057348,  7.68210369,
        #              7.84720022, 10.2049998 ,  8.84496611, 11.97519918,  9.05992028,
        #             11.6544648 ,  9.12327091, 18.47890308, 14.30687507, 15.43494347,
        #              4.30835105,  6.11519591,  5.65395798, -1.94722352, -4.34623654,
        #             -4.65180632,  0.15969528,  0.82600505,  1.27624902, -1.43409636])
        elif wavcent==4.62 or wavcent==4.67:
             resarr=np.array([48.98724913, 43.68968906, 46.59288629, 28.11740499, 28.97803448,
                    25.58747299, 15.99254105, 17.42208356, 14.01727114, 41.94356291,
                    43.5725864 , 41.16904248, 54.63255155, 54.14835553, 53.80501166,
                    43.46316326, 45.61184972, 42.92003541, 41.27601246, 41.48968348,
                    44.09046293, 34.12400453, 31.11570925, 28.14993109, 27.6227304 ,
                    29.40270658, 31.3081007 ,  8.55760547, 14.67721075, 17.42082055,
                     6.60181847,  6.86736024, 10.82434962, 10.5491853 , 12.00103959,
                    12.00918028,  8.63598844,  9.87318423, 11.22729794,  7.83751059,
                     9.44118193, 10.10402259,  7.03987166,  4.99266183,  6.58894695,
                     2.97213095,  1.3238294 ,  0.85639067, 14.62550146, 15.15146441,
                    14.20391614,  6.49818757,  8.0524488 , 13.6706872 , 12.07895777,
                     7.22772339, 10.12796154,  9.79314174,  6.02951683,  9.17790733,
                    15.74606949, 18.91726082, 16.71427263, 10.62401194,  7.16886307,
                     7.84764427,  9.97439066,  9.78015394,  9.3905039 , 10.56658804,
                    11.79318616,  9.6924954 , 15.2269294 , 19.03948687, 18.38565637,
                     5.87414921,  7.79057887,  6.71805392, -2.03669527, -6.36086352,
                    -4.05830184,  0.2148845 ,  0.82309966,  0.34153859, -1.56286402])
        elif wavcent==4.77 or wavcent==4.87:
             resarr=np.array([47.95757735, 44.77436271, 45.56310715, 29.18123787, 28.48773296,
                    26.17386571, 15.82182609, 16.63966192, 14.90223869, 42.15671843,
                    41.35085231, 41.6779737 , 55.49689748, 53.33737089, 54.24309923,
                    43.49492585, 45.04065068, 43.48745277, 39.51442465, 40.77286713,
                    45.35505969, 33.47074912, 31.47796296, 28.77443767, 28.42757721,
                    30.57512489, 32.38470314,  9.4103108 , 14.7002162 , 17.79092916,
                     7.94270229,  6.33475068, 10.43918959, 10.41499941, 12.29998116,
                    11.70489874,  9.18007919,  9.04854085, 10.71159828,  7.17407242,
                     9.06083825, 10.20370753,  6.38879265,  4.81651081,  6.55302379,
                     3.11592331,  1.87656445,  0.17880559, 14.87014196, 14.06535883,
                    13.5949504 ,  6.79599618,  7.91414604, 14.17454514, 12.11621353,
                     7.12217182, 10.05749838,  9.77147153,  6.73423903,  7.9180506 ,
                    15.95010188, 20.64296274, 18.00937617, 10.6508032 ,  8.566464  ,
                     7.4228087 , 10.11389548,  9.70990599,  9.62259569, 10.4863635 ,
                    12.76570831,  9.5734124 , 15.0988345 , 17.86251991, 19.7722981 ,
                     6.50440301,  8.25387507,  7.17178823, -2.20616474, -7.01120936,
                    -4.13313673,  0.21737672,  0.83410065,  2.16002771, -1.57964532])
        

        
    elif NGauss==14:

        resarr=np.array([  0071.4    ,   0071.4    ,   0071.4    ,   0071.4    ,
                         0071.4    ,   0071.4    ,   0071.4    ,   0071.4    ,
                         0071.4    ,   0071.4    ,   0071.4    ,   0071.4    ,
                         0071.4    ,   0071.4    ,   0.        ,  -0.        ,
                         -2.65197857,  -2.65197857,  -5.30395714,  -5.30395714,
                         2.65197857,   5.30395714,   5.30395714,   2.65197857,
                         -0.        ,  -0.        ,  -2.65197857,  -2.65197857,
                         0.        ,  -2.65197857,   0.        ,   2.65197857,
                         5.30395714,   7.95593571,  -2.65197857,  -2.65197857,
                         0.        ,   2.65197857, -16.        , -18.65197857,
                         -18.65197857, -16.        ,   0.1       ,  -6.        ])
    for wav_index in range(0,1):
        SciVis=np.array([])
        SciVisErr=np.array([])
        SciCPhase=np.array([])
        SciCPhaseErr=np.array([])
        U=np.array([])
        V=np.array([])
        vflag=np.array([],dtype=bool)
        cflag=np.array([],dtype=bool)
        wavelength=np.array([])
        LL=(wavcent-0.05)*1e-6
        LH=(wavcent+0.05)*1e-6
        

        for num,i in enumerate(FileList):
            if num==4 or num==6:
                print(i)
            for j in range(fits.open(i)['OI_VIS2'].data['vis2data'].shape[0]):

                wav=fits.open(i)['OI_WAVELENGTH'].data['eff_wave']

                maskl=wav>LL
                maskh=wav<LH
                mask=maskl*maskh
                wavelength=np.append(wavelength,wav[mask])
                #mask[:MAH]=False
                vflag=np.append(vflag,fits.open(i)['OI_VIS2'].data['flag'][j][mask])
                SciVis=np.append(SciVis,fits.open(i)['OI_VIS2'].data['vis2data'][j][mask])
                SciVisErr=np.append(SciVisErr,fits.open(i)['OI_VIS2'].data['vis2err'][j][mask])
                U=np.append(U,fits.open(i)['OI_VIS2'].data['ucoord'][j]/wav[mask])
                V=np.append(V,fits.open(i)['OI_VIS2'].data['vcoord'][j]/wav[mask])
            for j in range(fits.open(i)['OI_T3'].data['t3phi'].shape[0]):
                #mask[:MAH]=False
                if num==4 or num==6:
                    cflag=np.append(cflag,fits.open(i)['OI_T3'].data['flag'][j][mask])
                    SciCPhase=np.append(SciCPhase,fits.open(i)['OI_T3'].data['t3phi'][j][mask])
                    SciCPhaseErr=np.append(SciCPhaseErr,fits.open(i)['OI_T3'].data['t3phierr'][j][mask])
                else:
                    cflag=np.append(cflag,fits.open(i)['OI_T3'].data['flag'][j][mask])
                    SciCPhase=np.append(SciCPhase,fits.open(i)['OI_T3'].data['t3phi'][j][mask])
                    SciCPhaseErr=np.append(SciCPhaseErr,fits.open(i)['OI_T3'].data['t3phierr'][j][mask])

    
        # SciVis[SciVis>0.25]=np.nan
        SciVis[SciVis<0.0]=np.nan
        SciVis[vflag]=np.nan
        SciCPhase[cflag]=np.nan

        
        #Recalculate uv positions, unnecessary but safe and pretty quick anyway
        sta_ind=[]
        sta_array=[]
        sta_array_ind=[]
        timeobs_vis_array=[]
        int_time_vis=[]
        phase_sta_ind=[]
        timeobs_phase_array=[]
        int_time_phase=[]
        
        for num,i in enumerate(FileList):
            
            hd=fits.open(i)
            # for h in wav_index:
            for j in dc(hd[4].data['STA_INDEX']):
                sta_ind.append(j)
            for j in dc(hd[5].data['STA_INDEX']):
                phase_sta_ind.append(j)
        
            for j in dc(hd[2].data['STA_NAME']):
                sta_array.append(j)
            for j in dc(hd[2].data['STA_INDEX']):
                sta_array_ind.append(j)
                
            for j in dc(hd[4].data['MJD']):
                timeobs_vis_array.append(Time(j,format='mjd'))
            for j in dc(hd[5].data['MJD']):
                timeobs_phase_array.append(Time(j,format='mjd'))
                
            for j in hd[4].data['INT_TIME']:
                int_time_vis.append(datetime.timedelta(seconds=j/2.0))
            for j in dc(hd[5].data['INT_TIME']):
                int_time_phase.append(datetime.timedelta(seconds=j/2.0))
            hd.close()
     
        sta_ind=np.array(sta_ind)
        phase_sta_ind=np.array(phase_sta_ind)
        sta_array=np.array(sta_array)
        sta_array_ind=np.array(sta_array_ind)
        
        U=np.zeros((len(SciVis)))
        V=np.zeros((len(SciVis)))
        
        
        
        
        
        
        #New uv coords for vis data
        

        for i in range(len(SciVis)//len(wav[mask])):
            # for j in wav_index:
                
            #Find station names
            T2i = np.argwhere(sta_array_ind==sta_ind[i][0])[0][0]
            T1i = np.argwhere(sta_array_ind==sta_ind[i][1])[0][0]
            
            #Generate half way times for each observation
            timeobs=timeobs_vis_array[i]
            time=Time(timeobs.datetime + int_time_vis[i])
            
            # Asjust the observational parameters to the new time
            altnight=AltAz(obstime=time,location=paranal)
            ESO323night=ESO323.transform_to(altnight)
            azr=ESO323night.az.radian
            altr=ESO323night.alt.radian
            
            T2 = sta_array[T2i]
            T1 = sta_array[T1i]
            
            #Fill uv arrays
            U[i*len(wav[mask]):(i+1)*len(wav[mask])] = UVpoint(ESO323, paranal, [T2,T1], azr, altr, time)[1]/wavelength[i*len(wav[mask]):(i+1)*len(wav[mask])]
            V[i*len(wav[mask]):(i+1)*len(wav[mask])] = UVpoint(ESO323, paranal, [T2,T1], azr, altr, time)[2]/wavelength[i*len(wav[mask]):(i+1)*len(wav[mask])]
            
                
        if 0 in U:
            raise ValueError('Mismatched UV and Vis length')
        
        
        #New uv coords for cphase data
        
        Ucphase=np.zeros((len(SciCPhase),3))
        Vcphase=np.zeros((len(SciCPhase),3))
            
        for i in range(len(Ucphase)//len(wav[mask])):
            
            #Find station names
            T3i = np.argwhere(sta_array_ind==phase_sta_ind[i][0])[0][0]
            T2i = np.argwhere(sta_array_ind==phase_sta_ind[i][1])[0][0]
            T1i = np.argwhere(sta_array_ind==phase_sta_ind[i][2])[0][0]
            
            #Generate half way times for each observation
            timeobs=timeobs_phase_array[i]
            time=Time(timeobs.datetime + int_time_phase[i])
            
            # Asjust the observational parameters to the new time
            altnight=AltAz(obstime=time,location=paranal)
            ESO323night=ESO323.transform_to(altnight)
            azr=ESO323night.az.radian
            altr=ESO323night.alt.radian
            
            T3 = sta_array[T3i]
            T2 = sta_array[T2i]
            T1 = sta_array[T1i]
            
            #Fill uv arrays
            Ucphase[i*len(wav[mask]):(i+1)*len(wav[mask]),0] = UVpoint(ESO323, paranal, [T3,T2], azr, altr, time)[1]/wavelength[i*len(wav[mask]):(i+1)*len(wav[mask])]
            Vcphase[i*len(wav[mask]):(i+1)*len(wav[mask]),0] = UVpoint(ESO323, paranal, [T3,T2], azr, altr, time)[2]/wavelength[i*len(wav[mask]):(i+1)*len(wav[mask])]
            Ucphase[i*len(wav[mask]):(i+1)*len(wav[mask]),1] = UVpoint(ESO323, paranal, [T2,T1], azr, altr, time)[1]/wavelength[i*len(wav[mask]):(i+1)*len(wav[mask])]
            Vcphase[i*len(wav[mask]):(i+1)*len(wav[mask]),1] = UVpoint(ESO323, paranal, [T2,T1], azr, altr, time)[2]/wavelength[i*len(wav[mask]):(i+1)*len(wav[mask])]
            Ucphase[i*len(wav[mask]):(i+1)*len(wav[mask]),2] = UVpoint(ESO323, paranal, [T3,T1], azr, altr, time)[1]/wavelength[i*len(wav[mask]):(i+1)*len(wav[mask])]
            Vcphase[i*len(wav[mask]):(i+1)*len(wav[mask]),2] = UVpoint(ESO323, paranal, [T3,T1], azr, altr, time)[2]/wavelength[i*len(wav[mask]):(i+1)*len(wav[mask])]
            
                
    
    # =============================================================================
    #     Make the uv arrays symmetric? Probably just a waste of time
    # =============================================================================
        
 
        nparam=3
        ndim, nwalkers = NGauss*nparam+4, 200
        
        #Declare image parameters
        nxy,dxy = double.get_image_size(np.append(U, Ucphase.flatten()), np.append(V, Vcphase.flatten()),f_max=10)
        print(nxy,dxy)
        # nxy *= 2
        # dxy/=2
#        dxy /= 100
        if type(resarr)==type(None):
            pos=[]
            for position in range(nwalkers):
            #Generate the initial params
                res=[[],[],[]]
                for i in range(NGauss):
                    res[0].append(np.random.uniform(0,1.0))
    #                if i > NGauss/2:
    #                    res[1].append(np.random.uniform(2,10))
    #                else:
    #                    res[1].append(np.random.uniform(-10,-2))
                    res[1].append(np.random.uniform(-40,40))
                    res[2].append(np.random.uniform(-40,40))
    #                res[3].append(0.5)
                    #res[5].append(100)
                # res[0][0]=0.15
                
                res[0]=[1000*num/np.sum(res[0]) for num in res[0]]
#                res[0][:]=[1*num/np.sum(res[0]) for num in res[0]]
#                if NGauss>2:
#                    res[0][2]=1.-res[0][0]-res[0][1]
    #            res[0][1]=-2.0
                res[1][0]=0.0
                res[2][0]=0.0
############## While is not proofed against getting stuck, problematic for large NGauss
                while np.isinf(lnprior_point_unfixed(np.array(res),-2.7,0.25,dxy)):
                    res=[[],[],[]]
                    for i in range(NGauss):
                        res[0].append(np.random.uniform(0,1.0))
                        res[1].append(np.random.uniform(-40,40))
                        res[2].append(np.random.uniform(-40,40))
                   
                    ################Fixed Point###########
                    # res[0][0]=0.15
                    # res[0][1:]=[0.85*num/np.sum(res[0][1:]) for num in res[0][1:]]
                    
                    ################unfixed##############
                    res[0][0:]=[1000.0*num/np.sum(res[0][0:]) for num in res[0][0:]]
                    

                    res[1][0]=0.0
                    res[2][0]=0.0
                 
                
                res=res.flatten()
                res = np.append(res,np.random.uniform(0,0.2))
                res = np.append(res,np.random.uniform(-10,0))
                pos.append(res)
                
        else:
            res=dc(resarr)
            pos = np.array([res + 1e-2*np.random.randn(ndim) for i in range(nwalkers)])
            print(pos[0])
        
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_point, args=(U, V, SciVis, SciVisErr,
                                                                      Ucphase, Vcphase, SciCPhase,
                                                                      SciCPhaseErr, nxy, dxy))
        print('start')
        print(res)
        
        
        nsteps = 1200

        for i, result in enumerate(tqdm(sampler.sample(pos, iterations=nsteps),total=nsteps)):
    #        n = int((width+1) * float(i) / nsteps)
    #        sys.stdout.write("\r[{0}{1}]".format('#' * n, ' ' * (width - n)))
            pass
    #    sys.stdout.write("\n")
        
        ### Burn in plots
        # plt.figure()
        # for j in range(0,ndim):
        #     plt.subplot(ndim,1,j+1)
        #     for i in range(0,nwalkers):plt.plot(sampler.chain[i,:,j])
        
        # samples = sampler.chain[:, 2000:, :].reshape((-1, ndim))
    
        ##### Corner plots
#        fig = corner.corner(samples,quantiles=[0.16, 0.5, 0.84],show_titles=True, title_kwargs={"fontsize": 12})
#        fig.suptitle('ESO -- Test')
        samples = sampler.chain[:, 1000:, :].reshape((-1, nparam*NGauss+4))
        resarr= map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                 zip(*np.percentile(samples, [16, 50, 84],
                                                    axis=0)))
        resarr=np.array(list(resarr))[:,0]
#        resarr2=resarr[:-2,0].reshape(3,int(len(resarr[:-2])/3)).flatten()
#        resarr2=np.append(resarr2,resarr[-2])
#        resarr=np.append(resarr2,resarr[-1])
        # BIC=np.log(len(SciVis)+len(SciCPhase))*nparam*(NGauss)-2*lnlike_point(np.reshape(resarr[:-2],(3,int(len(resarr[:-2])/3))), resarr[-1], resarr[-2], U, V, SciVis, SciVisErr, Ucphase, Vcphase, SciCPhase,SciCPhaseErr, nxy, dxy)
#        print(BIC)
        # if BIC<BICc:
        #     samplerbest=dc(sampler)
        #     BICc=dc(BIC)
    return U,V,Ucphase,Vcphase,SciVis,SciCPhase,SciVisErr,SciCPhaseErr,sampler,nxy,dxy,9999999

BICcurr=99999
BICarray=[]
allminres=[]
NGauss=27 #Number of points
samplerlist=[]
reslist=[]
for wavcent in [3.25,3.50,3.60,3.70,3.80,4.62,4.67,4.77,4.87]:
    
    result=RunGravEmcee(f,ESO323,paranal,NGauss,wavcent)
    U,V,Ucphase,Vcphase,SciVis,SciCPhase,SciVisErr,SciCPhaseErr,sampler,nxy,dxy,BIC=result
    samplerlist.append(sampler)
    samples = sampler.chain[:, 1000:, :].reshape((-1, 3*NGauss+4))
    #Enable BICs and argmin allminres if resarr is unset or median allminres if set. Min BIC can be compared for diff NGauss
    #BICs=[np.log(len(SciVis)+len(SciCPhase))*3*(NGauss)-2*lnlike_point(np.reshape(i[:-4],(3,int(len(i[:-4])/3))), i[-1], i[-4],i[-3],i[-2], U, V, SciVis, SciVisErr, Ucphase, Vcphase, SciCPhase,SciCPhaseErr, nxy, dxy) for i in tqdm(samples)]    # BICarray.append(BICs)
    #allminres.append(samples[np.argmin(BICs)])
    # allminres.append(np.median(samples,axis=0))
    # if np.min(BICs)<BICcurr:
    reslist.append(result)
    resultfinal=dc(result)
        # array=np.array(samples[np.argmin(BICs)])
        # BICcurr=dc(np.min(BICs))
    # print(NGauss,': ',np.min(BICs))
U,V,Ucphase,Vcphase,SciVis,SciCPhase,SciVisErr,SciCPhaseErr,sampler,nxy,dxy,BIC=resultfinal #set to whichever wav required
