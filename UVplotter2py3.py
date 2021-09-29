import numpy as np
import matplotlib.pylab as plt
#import oifits3 as oi
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
import datetime
import astropy.modeling.models as mod
import scipy.optimize as op
from mpl_toolkits.mplot3d import Axes3D
from astropy.io import fits
#import utm


def linefit(x,std):
    ym=mod.Gaussian1D(0.16,0,std)(x)+0.23
#    ym=mod.Gaussian1D(0.38,0,std)(x)
    return ym

#model call and maximum likelyhood function
def lnlike(theta, x, y, yerr):
    std, lnf = theta
    model = anylfftgauss(x,std)
    inv_sigma2 = 1.0/(yerr**2 + model**2*np.exp(2*lnf))
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

#def Gauss(amp,mx,my,stdx,stdy,theta,varx,vary):
#    Ga=(np.cos(theta)**2)/(2*stdx**2)+(np.sin(theta)**2)/(2*stdy**2)
#    Gb=(np.sin(2*theta))/(2*stdx**2)-(np.sin(2*theta))/(2*stdy**2)
#    Gc=(np.sin(theta)**2)/(2*stdx**2)+(np.cos(theta)**2)/(2*stdy**2)
#    gaussian=amp*np.exp(-Ga*(varx-mx)**2-Gb*(varx-mx)*(vary-my)-Gc*(vary-my)**2)
#    return gaussian

def anylfftgauss(x,ys):
#    gauss=
#    Gmodel=mod.Gaussian2D(1.0-a,0.0,0.0,lam*6.48e8/((ys/r)*np.pi*np.pi*2.0),lam*6.48e8/(ys*np.pi*np.pi*2.0),-t)+mod.Gaussian2D(a,0.0,0.0,lam*6.48e8/(0.2*np.pi*np.pi*2.0),lam*6.48e8/(0.2*np.pi*np.pi*2.0),0.0)
     anygauss=mod.Gaussian1D(0.16,0.0,11.8e-6*6.48e8/((ys)*np.pi*np.pi*2.0))(x)+0.23
#     anygauss=Gauss(0.16,0.0,0.0,11.8*6.48e8/((ys/r)*np.pi**2*2.0),11.8*6.48e8/(ys*np.pi**2*2.0),-t,x,amp)+Gauss(0.23,0.0,0.0,11.8*6.48e8/(0.2*np.pi**2*2.0),11.8*6.48e8/(0.2*np.pi**2*2.0),0.0,x,amp)
#    anygauss=Gmodel(x,amp)
#    del(Gmodel)
     return anygauss
def UVpoint(ESO323,paranal,Stat,azr,altr,time):
    #station coords relative to VLTI (E,N,A)
    stations={'U1':np.array([-9.925,-20.335,0.0]),'U2':np.array([14.887, 30.502,0.0]),
              'U3':np.array([44.915,66.183,0.0]),'U4':np.array([103.306,43.999,0.0]),
              'A0':np.array([-14.642,-55.812,0.0]),'A1':np.array([-9.434,-70.949,0.0]),
              'B0':np.array([-7.065,-53.212,0.0]),'B1':np.array([-1.863,-68.334,0.0]),
              'B2':np.array([0.739,-75.899,0.0]),'B3':np.array([3.348,-83.481,0.0]),
              'B4':np.array([5.945,-91.030,0.0]),'B5':np.array([8.547,-98.594,0.0]),
              'C0':np.array([0.487,-50.607,0.0]),'C1':np.array([5.691,-65.735,0.0]),
              'C2':np.array([8.296,-73.307,0.0]),'C3':np.array([10.896,-80.864,0.0]),
              'D0':np.array([15.628,-45.397,0.0]),'D1':np.array([26.039,-75.660,0.0]),
              'D2':np.array([31.243,-90.787,0.0]),'E0':np.array([30.760,-40.196,0.0]),
              'G0':np.array([45.896,-34.990,0.0]),'G1':np.array([66.716,-95.501,0.0]),
              'G2':np.array([38.063,-12.289,0.0]),'H0':np.array([76.150,-24.572,0.0]),
              'I1':np.array([96.711,-59.789,0.0]),'J1':np.array([106.648,-39.444,0.0]),
              'J2':np.array([114.460,-62.151,0.0]),'J3':np.array([80.628,36.193,0.0]),
              'J4':np.array([75.424,51.320,0.0]),'J5':np.array([67.618,74.009,0.0]),
              'J6':np.array([59.810,96.706,0.0]),'K0':np.array([106.397,-14.165,0.0]),
              'L0':np.array([113.977,-11.549,0.0]),'M0':np.array([121.535,-8.951,0.0])}
    #OLD CODE
    Stanames=[str(Stat[0])[:2],str(Stat[1])[:2]]
#    Stacoord=[stations[i] for i in Stanames]
#    star=[np.sin(azr)*np.cos(altr),np.cos(azr)*np.cos(altr),np.sin(altr)]
#    baseline=[Stacoord[1][0]-Stacoord[0][0],Stacoord[1][1]-Stacoord[0][1],Stacoord[1][2]-Stacoord[0][2]]
#    sidOPD=star[0]*baseline[0]+star[1]*baseline[1]+star[2]*baseline[2]
#    PBase=[baseline[i]-sidOPD*star[i] for i in range(0,3)]
    #Ucoord=-PBase[1]*np.sin(-0.7069)+PBase[0]*np.cos(-0.7069)
    #Vcoord=-PBase[1]*np.sin(altr)*np.cos(-0.7069)-PBase[0]*np.sin(altr)*np.sin(-0.7069)-PBase[2]*np.cos(altr)
#    theta=np.arctan2(PBase[1],PBase[0])/np.pi*180
    #NEW CODE
#    Stanames=['U1','U3']

    Stacoord=[stations[i] for i in Stanames]
    baseline=[Stacoord[1][0]-Stacoord[0][0],Stacoord[1][1]-Stacoord[0][1],Stacoord[1][2]-Stacoord[0][2]]
    A_b=np.arctan2(baseline[0],baseline[1])
    A=azr
    sinpsi=np.sin(A_b-A)
    cospsi=np.cos(A_b-A)*np.sin(altr)
    psi=np.arctan2(sinpsi,cospsi)
    #HA=np.arccos((np.sin(altr)-np.sin(ESO323.dec.radian)*np.sin(paranal.latitude.radian))/(np.cos(ESO323.dec.radian)*np.cos(paranal.latitude.radian)))
    lst=time.sidereal_time('apparent',paranal.lon)
    HA=(lst.deg-ESO323.ra.deg)*np.pi/180.0
    y=np.sin(HA)
    x=np.tan(paranal.lat.radian)*np.cos(ESO323.dec.radian)-np.sin(ESO323.dec.radian)*np.cos(HA)
    p=np.arctan2(y,x)
    pang=p+psi+np.pi
    colati=(np.pi/2.0+paranal.lat.radian)
    pen=baseline[1]*np.cos(colati)+baseline[2]*np.sin(colati)
    pew=-baseline[0]
    pea=-baseline[1]*np.sin(colati)+baseline[2]*np.cos(colati)
    ucoord=pen*np.sin(HA)-pew*np.cos(HA)
    #vcoord=ucoord*np.tan(pang)
    vcoord=-pen*np.sin(ESO323.dec.radian)*np.cos(HA)-pew*np.sin(ESO323.dec.radian)*np.sin(HA)-pea*np.cos(ESO323.dec.radian)
    pang=(pang)%(2*np.pi)
    
#    print Stanames, pang, ucoord, vcoord
    return (pang, ucoord, vcoord)
