#ipython3 TO RUN IN HELADA
#os.chdir('/Users/M51/Downloads/SEDfittingplotspickles7Feb')
#run sed.py

from scipy.optimize import curve_fit
from scipy.constants import h,k,c
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from mcdb import wutil as wu
from numpy import asarray
from numpy import save

wlLMN=[3.25,3.50,3.60,3.70,3.80,4.62,4.67,4.77,4.87,8.5,9.0,9.5,10.0,10.5,11.0,11.5,12.0,12.5]
wln=[8.5,9.0,9.5,10.0,10.5,11.0,11.5,12.0,12.5]
wllm=[3.20,3.50,3.60,3.70,3.80,4.62,4.67,4.77,4.87]
wlmn=[4.62,4.67,4.77,4.87,8.5,9.0,9.5,10.0,10.5,11.0,11.5,12.0,12.5]

nus=np.logspace(13,14.5,5000) 
ls=(c/nus)*1e6 #in um  descending   wl
revls = ls[::-1]


mas=[10,8,9,10,9,68,42,63,28,30,34,9]#
mis=[7,8,7,8,7,37,20,30,7,8,3,7]
print('12 ellipses')#flux ratios to compare are DE1-DE5 and DE2-DE4
ens=['E1','E2','E3','E4','E5','DE1','DE2','DE3','DE4','DE5','t','void','DE1-DE4','DE2-DE5']
#ens=['e1','e2','e3','e4','e5','de1','de2','de3','nc','sc','t','void']
colors=['magenta','lime', 'yellow', 'orange', 'lightblue','red','green','brown','purple','cyan','pink','white']
colors2=['green','royalblue','darkorange','sienna','brown','green','royalblue','darkorange','sienna','brown','gold','salmon']
i=9  # CHANGE ELLIPSE HERE i=0 to 4        #   <-----------------------------CHANGE--
Cper=20   #CHANGE %Carbon 
p1=1.0; p2=1.0#for b1LMN2 CONCLUSION:USE 1.0 for both 
ma1=mas[i]
mi1=mis[i]
en=ens[i]
col=colors[i]
run='5' #refining grid                     # run nr.  <-----------------------------CHANGE--
indexmedei=i#np.int(en[1])-1
indexstdei=i#np.int(en[1])-1
col2=colors2[i]
ymaxs=[2.0,0.9,0.5,1.0,0.4,7,2,3,1.4, 1 ,0.9,0.6]#for 1bb fits
ym=ymaxs[i]
print('ellipse: '+en+'run:'+run)

if(i==9):
   limchisq=30#r1  
   #temperatures1=np.linspace(180,600,20)#R1   
   #ffs1=np.linspace(0.001,0.1,20)
   #logNexts1=np.linspace(-6,-3,20)#

   #temperatures2=np.linspace(600,1200,20)#15->chi2:20-24
   #ffs2=np.linspace(0.0001,0.1,20)
   #fine grid final
   temperatures1=np.linspace(130,400,30)#20 
   ffs1=np.linspace(0.01,1,30)
   logNexts1=np.linspace(-5,-3,30)#30

   temperatures2=np.linspace(950,1500,30)#
   ffs2=np.linspace(0.0001,0.001,35)#.
def singlebb1AmMgFeolrvp1C(nu,T2,ff,N):
    h = 6.63*10**(-34) #J*s=W*s=1*10^7erg=kg*m^2*s^-2
    c = 2.998*10**8 #m/s  to cm
    k = 1.38*10**(-23)  #J/K = kg*m^2*s / K
    a=(2*h*np.power(nu,3)) / (np.power(c,2))
    enu2=np.exp((h*nu)/(k*T2))
    I2=a*(1/(enu2-1))#W m^-2 Hz^-1 sr^-1
    marad = (ma1*2*np.pi)/(1000*360*3600)
    mirad = (mi1*2*np.pi)/(1000*360*3600)
    Arad2=marad * mirad * 2*np.pi
    #print('area rad',Arad2)
    Iint2=I2*Arad2/1e-26
    #extdummy,wlext=loadextAmMgFeolrvp1()#to use wl
    ext=mix(Cper)#CHANGE % OF CARBON
    revwlext=wlext[::-1]
    revext=ext[::-1]
    nuext=np.divide(c,revwlext)*1e6
    intext2=np.interp( nu, nuext, revext)
    #print('intext2*N',intext2*N)
    #print((1.0-np.exp(-1.0*np.power(c/nu,-1.0*alemiss))))
    return( ff*Iint2 * np.exp(-1.0*intext2*N) )#Jy/sr  #returns with wliii


def singlebb1AmMgFeolrv1p5(nu,T2,ff,N):
    h = 6.63*10**(-34) #J*s=W*s=1*10^7erg=kg*m^2*s^-2
    c = 2.998*10**8 #m/s  to cm
    k = 1.38*10**(-23)  #J/K = kg*m^2*s / K
    a=(2*h*np.power(nu,3)) / (np.power(c,2))
    enu2=np.exp((h*nu)/(k*T2))
    I2=a*(1/(enu2-1))#W m^-2 Hz^-1 sr^-1
    marad = (ma1*2*np.pi)/(1000*360*3600)
    mirad = (mi1*2*np.pi)/(1000*360*3600)
    Arad2=marad * mirad * 2*np.pi
    #print('area rad',Arad2)
    Iint2=I2*Arad2/1e-26
    ext,wlext=loadextAmMgFeolrv1p5()
    revwlext=wlext[::-1]
    revext=ext[::-1]
    nuext=np.divide(c,revwlext)*1e6
    intext2=np.interp( nu, nuext, revext)
    #print('intext2*N',intext2*N)
    #print((1.0-np.exp(-1.0*np.power(c/nu,-1.0*alemiss))))
    return( ff*Iint2 * np.exp(-1.0*intext2*N) )#Jy/sr  #returns with wliii


def singlebb1AmMgFeolrvp1(nu,T2,ff,N):
    h = 6.63*10**(-34) #J*s=W*s=1*10^7erg=kg*m^2*s^-2
    c = 2.998*10**8 #m/s  to cm
    k = 1.38*10**(-23)  #J/K = kg*m^2*s / K
    a=(2*h*np.power(nu,3)) / (np.power(c,2))
    enu2=np.exp((h*nu)/(k*T2))
    I2=a*(1/(enu2-1))#W m^-2 Hz^-1 sr^-1
    marad = (ma1*2*np.pi)/(1000*360*3600)
    mirad = (mi1*2*np.pi)/(1000*360*3600)
    Arad2=marad * mirad * 2*np.pi
    #print('area rad',Arad2)
    Iint2=I2*Arad2/1e-26
    ext,wlext=loadextAmMgFeolrvp1()
    revwlext=wlext[::-1]
    revext=ext[::-1]
    nuext=np.divide(c,revwlext)*1e6
    intext2=np.interp( nu, nuext, revext)
    #print('intext2*N',intext2*N)
    #print((1.0-np.exp(-1.0*np.power(c/nu,-1.0*alemiss))))
    return( ff*Iint2 * np.exp(-1.0*intext2*N) )#Jy/sr  #returns with wliii

def singlebb1AmMgolrvp1(nu,T2,ff,N):
    h = 6.63*10**(-34) #J*s=W*s=1*10^7erg=kg*m^2*s^-2
    c = 2.998*10**8 #m/s  to cm
    k = 1.38*10**(-23)  #J/K = kg*m^2*s / K
    a=(2*h*np.power(nu,3)) / (np.power(c,2))
    enu2=np.exp((h*nu)/(k*T2))
    I2=a*(1/(enu2-1))#W m^-2 Hz^-1 sr^-1
    marad = (ma1*2*np.pi)/(1000*360*3600)
    mirad = (mi1*2*np.pi)/(1000*360*3600)
    Arad2=marad * mirad * 2*np.pi
    #print('area rad',Arad2)
    Iint2=I2*Arad2/1e-26
    ext,wlext=loadextAmMgolrvp1()
    revwlext=wlext[::-1]
    revext=ext[::-1]
    nuext=np.divide(c,revwlext)*1e6
    intext2=np.interp( nu, nuext, revext)
    #print('intext2*N',intext2*N)
    #print((1.0-np.exp(-1.0*np.power(c/nu,-1.0*alemiss))))
    return( ff*Iint2 * np.exp(-1.0*intext2*N) )#Jy/sr  #returns with wliii

def singlebb1newol(nu,T2,ff,N):
    h = 6.63*10**(-34) #J*s=W*s=1*10^7erg=kg*m^2*s^-2
    c = 2.998*10**8 #m/s  to cm
    k = 1.38*10**(-23)  #J/K = kg*m^2*s / K
    a=(2*h*np.power(nu,3)) / (np.power(c,2))
    enu2=np.exp((h*nu)/(k*T2))
    I2=a*(1/(enu2-1))#W m^-2 Hz^-1 sr^-1
    marad = (ma1*2*np.pi)/(1000*360*3600)
    mirad = (mi1*2*np.pi)/(1000*360*3600)
    Arad2=marad * mirad * 2*np.pi
    #print('area rad',Arad2)
    Iint2=I2*Arad2/1e-26
    ext,wlext=loadextnewolivine()
    revwlext=wlext[::-1]
    revext=ext[::-1]
    nuext=np.divide(c,revwlext)*1e6
    intext2=np.interp( nu, nuext, revext)
    #print('intext2*N',intext2*N)
    #print((1.0-np.exp(-1.0*np.power(c/nu,-1.0*alemiss))))
    return( ff*Iint2 * np.exp(-1.0*intext2*N) )#Jy/sr  #returns with wliii

def singlebb1ol(nu,T2,ff,N):
    h = 6.63*10**(-34) #J*s=W*s=1*10^7erg=kg*m^2*s^-2
    c = 2.998*10**8 #m/s  to cm
    k = 1.38*10**(-23)  #J/K = kg*m^2*s / K
    a=(2*h*np.power(nu,3)) / (np.power(c,2))
    enu2=np.exp((h*nu)/(k*T2))
    I2=a*(1/(enu2-1))#W m^-2 Hz^-1 sr^-1
    marad = (ma1*2*np.pi)/(1000*360*3600)
    mirad = (mi1*2*np.pi)/(1000*360*3600)
    Arad2=marad * mirad * 2*np.pi
    #print('area rad',Arad2)
    Iint2=I2*Arad2/1e-26
    ext,wlext=loadextolivine()
    revwlext=wlext[::-1]
    revext=ext[::-1]
    nuext=np.divide(c,revwlext)*1e6
    intext2=np.interp( nu, nuext, revext)
    #print('intext2*N',intext2*N)
    #print((1.0-np.exp(-1.0*np.power(c/nu,-1.0*alemiss))))
    return( ff*Iint2 * np.exp(-1.0*intext2*N) )#Jy/sr  #returns with wlii

def singlebb1(nu,T2,ff,N):
    h = 6.63*10**(-34) #J*s=W*s=1*10^7erg=kg*m^2*s^-2
    c = 2.998*10**8 #m/s  to cm
    k = 1.38*10**(-23)  #J/K = kg*m^2*s / K
    a=(2*h*np.power(nu,3)) / (np.power(c,2))
    enu2=np.exp((h*nu)/(k*T2))
    I2=a*(1/(enu2-1))#W m^-2 Hz^-1 sr^-1
    marad = (ma1*2*np.pi)/(1000*360*3600) 
    mirad = (mi1*2*np.pi)/(1000*360*3600) 
    Arad2=marad * mirad * 2*np.pi
    #print('area rad',Arad2)
    Iint2=I2*Arad2/1e-26
    ext,wlext=loadext()
    revwlext=wlext[::-1]
    revext=ext[::-1]
    nuext=np.divide(c,revwlext)*1e6
    intext2=np.interp( nu, nuext, revext)
    #print('intext2*N',intext2*N)
    #print((1.0-np.exp(-1.0*np.power(c/nu,-1.0*alemiss))))
    return( ff*Iint2 * np.exp(-1.0*intext2*N) )#Jy/sr  #returns with wli

#fsbb=singlebb(nus,200,0.1,0.0,ma1,mi1)
#plt.plot(ls,fsbb)
#plt.plot(ls,singlebb(nus,700,0.01,0.0,ma1,mi1))

def plotsed(selwl,b1):
    plt.plot(selwl,b1,'o',color='red',label='ce')
    plt.xlabel('Wavelength [$\\mu m$] ')
    plt.title('Flux fraction of 3 areas with wl')
    plt.xlabel('Wavelength [$\\mu m$] ')
    plt.ylabel('Flux [Jy]')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()

def getspeclm():
    df=pd.read_pickle('ngc1068septphotdata.pk')
    #DRS old data we trusted:3.16 -4.13 and 4.44 -4.88
    ydatal=[]
    ydatam=[]
    ydatalm=[]
    wldatal=[] #decreasing
    wldatam=[] #decreasing
    wldatalm=[] #decreasing
    yerrorlm=[]
    for i in range(df['wave'].shape[0]):
        if((i>1)and(i<34)):  #skipping band in the middle
            ydatalm.append(df['meanjy'][i])#flux   meanjy
            ydatam.append(df['meanjy'][i])#flux
            wldatalm.append(df['wave'][i]) #wave
            wldatam.append(df['wave'][i])
            yerrorlm.append(df['stddevjy'][i])#sigma flux
        if((i>49)and(i<102)):
            ydatalm.append(df['meanjy'][i])
            ydatal.append(df['meanjy'][i])
            wldatalm.append(df['wave'][i])
            wldatal.append(df['wave'][i])
            yerrorlm.append(df['stddevjy'][i])#sigma flux

    return(wldatalm,ydatalm,yerrorlm)


def getspecN():
    df=pd.read_pickle('n1068-5.pk')#NovPhotFlux.pkoldspectrum
    ydata=[]
    wldata=[] #increasing!!! 
    yerror=[]
    for i in range(df['wave'].shape[0]):
        if((i>0)and(i<156)):  #to use if we want to cut it
            ydata.append(df['flux'][i])#flux   meanjy
            wldata.append(df['wave'][i]) #wave
            yerror.append(df['flux'][i])#sigma flux  #CHANGE!! for new spectrum 
    
    wldatarev=wldata[::-1]
    ydatarev=ydata[::-1]
    yerrorrev=yerror[::-1]
    return(wldatarev,ydatarev,yerrorrev)


def plotspeclmn(wldatalm,ydatalm,yerrorlm,wldatan,ydatan,yerrorn):
    plt.plot(wldatalm,ydatalm)
    plt.errorbar(wldatalm,ydatalm,yerr=yerrorlm)
    plt.plot(wldatan,ydatan)
    plt.errorbar(wldatan,ydatan,yerr=yerrorn)
    return


def getrelfn(wls,wldata,ydata):
    rf=np.load('stdrelflxN12e.npy',allow_pickle=True)#
    fractsurfb1=[]
    for i in range(0,9):#(8,17):#for N band had 7 to 16, maybe this is the problem with 8.5umfeat
       fractsurfb1.append(np.float(rf[i][1][indexmedei]))#(rf[i][indexei])

    print(fractsurfb1)
    selwl=[]
    b1=[]
    ind=[]
    for i in range(0,np.size(wls)):
        absolute_val_array = np.abs(np.array(wldata) - wls[i])
        #print(wls[i],wldata) 
        smallest_difference_index = absolute_val_array.argmin()
        selwl.append(wldata[smallest_difference_index])
        b1.append(float(fractsurfb1[i])*ydata[smallest_difference_index])
        ind.append(smallest_difference_index)

    nus=c/np.array(selwl)*1e6
    return(nus,selwl,b1,ind)


def getrelflm(wls,wldata,ydata):
    rf=np.load('stdrelflxLM12e.npy',allow_pickle=True)#new M models
    fractsurfb1=[]
    for i in range(0,9):#for LM band, 9 including 3.2um
       fractsurfb1.append(np.float(rf[i][1][indexmedei]))#(rf[i][indexei])

    print(fractsurfb1)
    selwl=[]
    b1=[]
    ind=[]
    for i in range(0,np.size(wls)):
        absolute_val_array = np.abs(np.array(wldata) - wls[i])
        #print(wls[i],wldata) 
        smallest_difference_index = absolute_val_array.argmin()
        selwl.append(wldata[smallest_difference_index])
        b1.append(float(fractsurfb1[i])*ydata[smallest_difference_index])
        ind.append(smallest_difference_index)

    nus=c/np.array(selwl)*1e6
    return(nus,selwl,b1,ind)

def getstdrelflm(wls,wldata,ydata):   
    stdrf=np.load('stdrelflxLM12e.npy',allow_pickle=True)#for 12e new M models
    errrelfluxlm=[]
    for i in range(9):#9 including 3.2um
       errrelfluxlm.append(stdrf[i][2][indexstdei])
    
    print('errrelfluxlm',errrelfluxlm)  
    selwl=[]
    abserrfluxlm=[]
    ind=[]
    print(np.size(wls))
    for i in range(0,np.size(wls)):
        absolute_val_array = np.abs(np.array(wldata) - wls[i])
        #print(wls[i],wldata) 
        smallest_difference_index = absolute_val_array.argmin()
        selwl.append(wldata[smallest_difference_index])
        abserrfluxlm.append(float(errrelfluxlm[i])*ydata[smallest_difference_index])
        ind.append(smallest_difference_index)

    return(abserrfluxlm)


def getstdrelfn(wls,wldata,ydata):
    stdrf=np.load('stdrelflxN12e.npy',allow_pickle=True)#for 12 ellipses new N
    errrelflux=[]
    for i in range(9):
       errrelflux.append(stdrf[i][2][indexstdei])
     
    print('errrelflux',errrelflux)  
    selwl=[]
    abserrflux=[]
    ind=[]
    for i in range(0,np.size(wls)):
        absolute_val_array = np.abs(np.array(wldata) - wls[i])
        #print(wls[i],wldata) 
        smallest_difference_index = absolute_val_array.argmin()
        selwl.append(wldata[smallest_difference_index])
        abserrflux.append(float(errrelflux[i])*ydata[smallest_difference_index])
        ind.append(smallest_difference_index)
    
    return(abserrflux)

def getrfcsvJam(fname):
    ind=(np.int(en[1]))
    print(ind)
    ta=pd.read_csv(fname)#'SmallGaussPhot.csv')
    wl=ta[ta.keys()[0]] 
    fractsurfb1=[]
    n=np.shape(wl)[0]
    if(en=='e1'):lab='ce'
    elif(en=='e2'):lab='ne'
    else: lab='se'
    #print(ta[lab])
    for i in range(n):
       fractsurfb1.append(ta[lab][i])
    
    plt.plot(wl,fractsurfb1,'x',color=col2,label=en+'_point_sources')
    return(wl,fractsurfb1)

def getrfcsvJac(fnamelm,fnamen):
    ind=((np.int(en[1]) -1)*3)+3
    print(ind)
    ta=pd.read_csv(fnamelm)#'SmallGaussPhot.csv')
    wl=ta[ta.keys()[0]]
    fractsurfb1lm=[]
    eb1lm=[]
    n=np.shape(wl)[0]
    if(en=='e1'):
       lab=' centerflux [relative]'
       labe=' centererr [relative]' 
    elif(en=='e2'):
       lab=' neflux [relative]'
       labe=' neerr [relative]' 
    else: 
       lab=' swflux [relative]'
       labe=' swerr [relative]  '
    #print(ta[lab])
    for i in range(n):
       fractsurfb1lm.append(ta[lab][i])
       eb1lm.append(ta[labe][i])

    plt.errorbar(wl,fractsurfb1lm,yerr=eb1lm,fmt='*',color='darkmagenta',label=en+'_img_rec')
    #plt.plot(wl,fractsurfb1lm,'*',color='darkmagenta')#,label=en+'_img_rec')
    ta=pd.read_csv(fnamen)#'SmallGaussPhot.csv')
    wl=ta[ta.keys()[0]]
    fractsurfb1=[]
    eb1n=[]
    n=np.shape(wl)[0]
    #print(ta[lab])                                  
    for i in range(n):                                     
       fractsurfb1.append(ta[lab][i])
       eb1n.append(ta[labe][i])

    plt.errorbar(wl,fractsurfb1,yerr=eb1n,fmt='*',color='darkmagenta')
    #plt.plot(wl,fractsurfb1,'*',color='darkmagenta')#,label=en+'_img_rec')
    return(wl,fractsurfb1)



def getrf(fname,fnameelm,fnameen):
    rf=np.load(fname)
    fractsurfb1=[]
    for i in range(18):#for LM&N band#should be 18!! 
       fractsurfb1.append(np.float(rf[i][indexei]))

    stdrf=np.load(fnameelm,allow_pickle=True)
    errrelfluxlm=[]
    for i in range(9):
       errrelfluxlm.append(np.float(stdrf[i][2][indexstdei]))
    
    stdrf=np.load(fnameen,allow_pickle=True)
    errrelfluxn=[]
    for i in range(9):
       errrelfluxn.append(np.float(stdrf[i][2][indexstdei]))

    print(errrelfluxlm,fractsurfb1[0:8])
    plt.plot(wlLMN,fractsurfb1,color=col)#,label=en+'_small_Gaussians')
    plt.errorbar(wllm,fractsurfb1,yerr=errrelfluxlm,fmt='o',markersize=2,mfc=col,mec=col,ecolor='black',capsize=3,label=en+'_small_Gaussians')
    plt.errorbar(wln,fractsurfb1,yerr=errrelfluxn,fmt='o',markersize=2,mfc=col,mec=col,ecolor='black',capsize=3)
    return(fractsurfb1,errrelfluxlm,errrelfluxn)# in fractions to compare between methods

'''
getrf('relflxLMN.npy','stdrelflxLM.npy','stdrelflxN.npy')
getrfcsvJam('BigGaussPhot.csv')
getrfcsvJac('lm_image_florentin.csv','n_image_florentin.csv')

plt.legend(fontsize=7)
plt.tight_layout()
plt.xlim(xmin=3.2)
plt.xlim(xmax=13.5)
plt.ylim(ymin=-0.05)
#plt.ylim(ymax=0.5)
plt.grid(True)
plt.savefig('CompRelFlxs3ellipses'+en+'.png')
plt.xlabel('Wavelength [$\\mu m$] ')
plt.title('Relative flux inside ellipses with wl')
plt.xlabel('Wavelength [$\\mu m$] ')
plt.ylabel('Relative flux [counts]')
'''
def loadextC():
    table=pd.read_csv('carbon1to30um.csv')
    wlext=table[table.keys()[0]]
    ext=table[table.keys()[1]]
    return(ext,wlext)

def loadext():
    ext=[
     39793.801,
     67494.584,
     78152.768,
     116309.28,
     154490.88,
     210697.36,
     200780.75,
     267667.88,
     323695.77,
     356932.44,
     404597.06,
     448591.37,
     507437.96,
     565666.00,
     637002.05,
     931517.27,
     1137652.6,
     919333.49,
     631419.27,
     567468.25,
     503253.84,
     368146.13,
     278863.55,
     199173.93,
     135122.01,
     85026.363,
     66102.880,
     50531.502,
     38448.414,
     28846.463,
     21354.264,
     15798.436,
     11838.844,
     9010.1833,
     6986.4643,
     5493.4506,
     4383.0545,
     3583.7993,
     3206.5404,
     2924.1727,
     2806.9763,
     2639.7689,
     3053.1844,
     4800.5646,
     7260.0788,
     9651.0855,
     12116.474,
     15534.693,
     17369.322,
     16458.639,
     15175.293,
     13640.735,
     11497.354,
     9885.0501,
     8538.0231,
     7301.4810,
     5923.5105,
     4831.8437,
     4063.4057,
     3988.1476,
     4400.6627,
     4952.5537,
     5557.2525,
     6054.0399,
     6276.3328,
     6124.0335,
     5641.0919,
     5136.5000,
     4713.1708,
     4310.8821,
     3989.8932,
     3696.9629,
     3180.8610,
     2760.5042,
     2395.2222,
     2081.1324,
     1789.5720,
     1516.7320,
     1273.3178,
     1061.1560,
     879.22762,
     725.49803,
     491.86208,
     333.69313,
     228.05272,
     189.83898,
     99.169315,
     55.711456,
     31.821471,
     18.266502,
     10.507832,
     6.0476670,
     3.4824221,
     1.9172626,
     1.2454002,
     0.65250673]
    #Need to be aligned in the left side
    wlext=[
       0.0010000000,
       0.0013180000,
       0.0017380000,
       0.0022910000,
       0.0030199999,
       0.0039809998,
       0.0052479999,
       0.0069180001,
       0.0091199996,
       0.012020000,
       0.015850000,
       0.020889999,
       0.027540000,
       0.036309998,
       0.047860000,
       0.063100003,
       0.083180003,
       0.10960000,
       0.14450000,
       0.19050001,
       0.25119999,
       0.33109999,
       0.43650001,
       0.57539999,
       0.75860000,
       1.0000000,
       1.1480000,
       1.3180000,
       1.5140001,
       1.7380000,
       1.9950000,
       2.2909999,
       2.6300001,
       3.0200000,
       3.4670000,
       3.9809999,
       4.5710001,
       5.2480001,
       5.7540002,
       6.3099999,
       6.6069999,
       7.2440000,
       7.5860000,
       7.9429998,
       8.2220001,
       8.5109997,
       8.8100004,
       9.1199999,
       9.4410000,
       9.7720003,
       10.120000,
       10.470000,
       10.840000,
       11.220000,
       11.610000,
       12.020000,
       12.590000,
       13.180000,
       13.800000,
       14.450000,
       15.140000,
       15.850000,
       16.600000,
       17.379999,
       18.200001,
       19.049999,
       19.950001,
       20.889999,
       21.879999,
       22.910000,
       23.990000,
       25.120001,
       27.540001,
       30.200001,
       33.110001,
       36.310001,
       39.810001,
       43.650002,
       47.860001,
       52.480000,
       57.540001,
       63.099998,
       75.860001,
       91.199997,
       109.60000,
       120.20000,
       166.00000,
       218.80000,
       288.39999,
       380.20001,
       501.20001,
       660.70001,
       871.00000,
       1171.0000,
       1451.0000,
       2000.0000]
    return(ext,wlext)

def loadextolivine():  #olivine in 0.1 μm GRF approximation.Mg-rich member (Mg2SiO4) Gielen 2008 cm^2/g
    ext=[
    29.4   ,
    29.4   ,
    44.8   ,
    201.8  ,
    429.6  ,
    756.8  ,
    1240.4 ,
    1738.1 ,
    2107.9 ,
    2335.6 ,
    2236.3 ,
    2023.3 ,
    1810.3 ,
    1455.1 ,
    1199.6 ,
    1043.5 ,
    802.0  ,
    503.6  ,
    304.9  ,
    163.0  ,
    163.3  ,
    291.8  ,
    434.6  ,
    634.7]
    wlext=[
    3.0     ,
    6.1722      ,
    7.6439      ,
    8.366       ,
    8.7383      ,
    8.935       ,
    9.1803      ,
    9.3007      ,
    9.522       ,
    9.7695      ,
    10.0697     ,
    10.321      ,
    10.6222     ,
    11.0245     ,
    11.4508     ,
    11.8263     ,
    12.1028     ,
    12.4048     ,
    12.7807     ,
    13.1561     ,
    13.5552     ,
    14.1279     ,
    14.9        ,
    16.2204]
    return(ext,wlext)


#The conversion for absorption efficiency (Q) to mass absorption coefficient (f) is:
#f = (3/4) Q/(rho*a)
#Where a is the particle radius (0.1 microns in our case) and rho is the material
#density of the grains.  Wikipedia lists the density of olivine to be 3.2-4.5 g/cm^3
#(probably depends on Fe/Mg ratio).  For our purposes I would use rho=4 g/cm^3.
def loadextAmMgFeolrv1p5():
    fn='/allegro6/matisse/jaffe/minerals/Q_Am_MgFeolivine_Dor_DHS_f1.0_rv1.5.dat'
    it=readit(fn,20,600)    
    wlext, Q = zip(*it)
    a=0.1*1e-4#um
    rho=4. #g/cm^3
    ext= (3/4)* np.array(Q)/(rho*a)
    return(ext,wlext)

def loadextAmMgFeolrvp1():
    fn='/allegro6/matisse/jaffe/minerals/Q_Am_MgFeolivine_Dor_DHS_f1.0_rv0.1.dat'
    it=readit(fn,20,600)    
    wlext, Q = zip(*it)
    a=0.1*1e-4#um
    rho=4. #g/cm^3
    ext= (3/4)* np.array(Q)/(rho*a)
    return(ext,wlext)

def loadextAmMgolrvp1():
    fn='/allegro6/matisse/jaffe/minerals/Q_Am_Mgolivine_Jae_DHS_f1.0_rv0.1.dat'
    it=readit(fn,20,600)    
    wlext, Q = zip(*it)
    a=0.1*1e-4#um
    rho=4. #g/cm^3
    ext= (3/4)* np.array(Q)/(rho*a)
    return(ext,wlext)

def loadextnewolivine():
    fn='/allegro6/matisse/jaffe/minerals/Q_olivine_new_rv0.1.dat'
    it=readit(fn,20,600)
    wlext, Q = zip(*it)
    a=0.1*1e-4#um
    rho=4. #g/cm^3
    ext= (3/4)* np.array(Q)/(rho*a)
    return(ext,wlext)

def readit(file_name,start_line = 2,end_line=500): # start_line - where your data starts (2 line mean 3rd line, because we start from 0th line) 
    with open(file_name,'r') as f:
        data = f.read().split('\n')
    data = [i.split(' ') for i in data[start_line:end_line]]
    for i in range(len(data)):
       row = [(sub) for sub in data[i] if len(sub)!=0]
       yield float(row[0]),float(row[1])

def plotb1(selwl,b1,stdb1):
    #plt.plot(selwl,b1,'o',color='black',label=en)
    plt.errorbar(selwl,b1,yerr=stdb1,fmt='o',markersize=2,mfc=col,mec=col,ecolor='black',capsize=3,label=en)
    plt.legend(fontsize=7.2)
    plt.xlim(xmin=1.8)
    plt.xlim(xmax=13.5)
    plt.ylim(ymin=0.0)
    plt.ylim(ymax=ymaxs[i])
    plt.grid(True)
    plt.xlabel('Wavelength [$\\mu m$] ')
    plt.title('Flux inside ellipses with wl')
    plt.xlabel('Wavelength [$\\mu m$] ')
    plt.ylabel('Flux [Jy]')
    plt.tight_layout()
    #plt.savefig('sed'+en+'newspec3ellipses.png')


def mix(p):#returns in wlextsm
    pd=p/100
    intextC=np.interp( wlext, wlextC, extC)#wlext defined outside
    return(pd*intextC+(1-pd)*extsm)


'''
ext,wlext=loadext()
#m2/kg= 10000cm2/1000g=10cm2/g
#plt.plot(np.log10(wlext),np.log10(np.array(ext)/10))#as Fig.3 Schartmann 2005 
ext2,wlext2=loadextolivine()
ext3,wlext3=loadextnewolivine()
extsm,wlext=loadextAmMgFeolrvp1()
extC,wlextC=loadextC()

plt.plot(wlext,extsm,label='OlivineSmallGrains')
plt.plot(wlextC,extC,label='Carbon')
extmix10=mix(10)
print(extmix10)
plt.plot(wlext,extmix10,label='mix with 10%C')
extmix25=mix(25)
plt.plot(wlext,extmix25,label='mix with 25%C')
extmix50=mix(50)
plt.plot(wlext,extmix50,label='mix with 50%C')
extmix75=mix(75)
plt.plot(wlext,extmix75,label='mix with 75%C')
extmix90=mix(90)
plt.plot(wlext,extmix90,label='mix with 90%C')


#plt.plot(wlext,ext)
#plt.plot(wlext2,ext2,label='OlivineNoLMdata')
#plt.plot(wlext3,ext3*1.23,label='NewOlivine')
#plt.plot(wlext2,ext2,label='OlivineNoLMdata')
#plt.plot(wlext2,ext2,label='OlivineNoLMdata')

#plt.xlim(xmin=1.0,xmax=20)
#plt.ylim(ymin=0,ymax=40000)
#plt.ylim(ymin=0,ymax=2500)
#plt.legend(fontsize=7)
#plt.xlabel('Wavelength [$\\mu m$] ')
#plt.ylabel('Mass absorption coefficient [cm^2/g]')
#plt.tight_layout()
#plt.savefig('ComparissonSmallOlivinesCarbon.png')

plt.plot(np.log10(wlext),np.log10(np.array(ext)/10))#as Fig.3 Schartmann 2005 

#ext2,wlext2=loadextolivine()#loadextironpoorinol()
plt.plot(wlext,ext)
#plt.plot(wlext2,ext2)
#plt.plot(wlext2,np.array(ext2)*292.5)#@8.5  9651.0855 and ~300
plt.xlim(xmin=1.0,xmax=20)
plt.ylim(ymin=0,ymax=20000)


plt.plot(np.log10(wlext),np.log10(ext))
plt.plot(np.log10(wlext2),np.log10(np.array(ext2)*10000))
pilt.plot(np.log10(wlext2),np.log10(np.array(ext2)*25000))
plt.plot(np.log10(wlext2),np.log10(np.array(ext2)*50000))
plt.plot(np.log10(wlext2),np.log10(np.array(ext2)*100000))
plt.xlim(xmin=np.log(1.0),xmax=np.log(30))


plt.plot(ls,singlebb1ol(nus,300,1.0,1e-3))
plt.plot(ls,singlebb1(nus,300,1.0,1e-4))

'''


#basic set up
wldatalm,ydatalm,yerrorlm=getspeclm()
wldatan,ydatan,yerrorn=getspecN()
#plotspeclmn(wldatalm,ydatalm,yerrorlm,wldatan,ydatan,yerrorn)

nusn,selwln,b1n,indn=getrelfn(wln,wldatan,ydatan)
#plot3seds(selwln,b1n,b2n,b3n)
nuslm,selwllm,b1lm,indlm=getrelflm(wllm,wldatalm,ydatalm)
#plot3seds(selwllm,b1lm,b2lm,b3lm)
#ext,wlext=loadext()loaded inside singlebb1
nuslmn=np.concatenate((nusn,nuslm),axis=0)
#nusmn=np.concatenate((nusn,nuslm[0:4?]),axis=0)
selwl=np.concatenate((selwllm,selwln),axis=0)
b1=np.concatenate((b1lm,b1n),axis=0)
stdb1lm=getstdrelflm(wllm,wldatalm,ydatalm)
stdb1n=getstdrelfn(wln,wldatan,ydatan)
stdb1=np.concatenate((stdb1lm,stdb1n),axis=0)
print('got error bars',stdb1)
#plot3seds(selwl,b1,b2,b3)
#plt.close()
plotb1(selwl,b1,stdb1)
gravfl=np.array([0.05333333, 0.0665 , 0.11, 0.23, 0.36])
wlsk=np.array([2.0,2.1,2.2,2.3,2.4])
nusk=c/np.array([2.0,2.1,2.2,2.3,2.4])
#plotting Gravity fluxes
#plt.plot(wlsk,gravfl,'o',color='black',markersize='1',label='Gravity Collaboration 2020')#change error bar!!




print('b1=',b1)
b1LMN2=[p1*b1[0],p1*b1[1],p1*b1[2],p1*b1[3],p1*b1[4],p2*b1[5],p2*b1[6],p2*b1[7],p2*b1[8],b1[9],b1[10],b1[11],b1[12],b1[13],b1[14],b1[15],b1[16],b1[17]]
nl=4; nm=4; nn=9
print('b1LMN2',b1LMN2)
#plt.plot(selwl,b1LMN2,color='black')

#CHANGES to MAKE: avoid the for with the plt.plot inside /do not save fluxes. 
goodparams1=[]
revwlLMN=wlLMN[::-1]
selwls1=revwlLMN
selnus1=c/np.array(selwls1)*1e6
revb1LMN2=b1LMN2[::-1]#10% flux at LM


print('Using singlebb1AmMgFeolrvp1')
extsm,wlext=loadextAmMgFeolrvp1()
extC,wlextC=loadextC()#extmix=mix(Cper)
for i in range(len(temperatures1)):
    for j in range(len(ffs1)):
        for k in range(len(logNexts1)):
            fmod=singlebb1AmMgFeolrvp1C(selnus1, temperatures1[i], ffs1[j], np.float_power(10,logNexts1[k]) )
            residlmn=np.subtract(fmod,revb1LMN2) #including 8um
            residlm=np.subtract(fmod[9:],revb1LMN2[9:]) #in LM only
            #residlmnno8=np.concatenate((np.subtract(fmod[0:2],revb1LMN2[0:2]),residlm),axis=0) #no 8um
            #chisq1=np.sum(np.square(residlmnn/revabserrfluxNB3))#[3:] #not using it
            if(np.all(np.array(residlm)<0.0)):  #always less than 10%/20%fluxes data
                goodparams1.append([i,j,k])

print(np.shape(goodparams1),',goodparams1') #6471 short   / (3666, 3) finer/9175lastlong
print('Finished Goodparams1')


goodparams2=[] 
b1LMN=b1
revb1LMN=b1LMN[::-1]
count=0
revabserrfluxNB3=stdb1[::-1]
lgood1 = len(goodparams1)
print('good2 ',lgood1,len(temperatures2),len(ffs2))#,len(logNexts2))

class good2():
#  def __init__(self, goodparams1, temperatures2, ffs2)#, logNexts2):
   def __init__(self, inputs):
      self.temperatures1 = inputs['t1']
      self.ffs1 = inputs['f1']
      self.logNexts1 = inputs['l1']
      self.temperatures2 = inputs['t2']
      self.ffs2 = inputs['f2']
      self.selnus1 = inputs['selnus1']
      self.goodparams1 = inputs['g1']

   def loop1(self, m):
      count = 0
      for i in range(len(self.temperatures2)):
         for j in range(len(self.ffs2)):
            #for k in range(len(self.logNexts2)):
            if(self.temperatures1[m[0]]<self.temperatures2[i]):#print('for >Hotter in N')
               fmodcold=singlebb1AmMgFeolrvp1C(self.selnus1, self.temperatures1[m[0]],self.ffs1[m[1]], np.float_power(10,self.logNexts1[m[2]]) )
               fmodhot=singlebb1AmMgFeolrvp1C(self.selnus1, self.temperatures2[i],ffs2[j], np.float_power(10,self.logNexts1[m[2]]) )#USING NCold too
               totalf=fmodcold+fmodhot
               residlmn=np.subtract(totalf,revb1LMN)
               #chisq=np.sum(np.square(residlmn/revabserrfluxNB3))
               chisq=np.sum(np.square(residlmn/revabserrfluxNB3))#revb1LMN)
               #print(chisq)
               if(chisq<limchisq):#first selection USING 100
                  count+=1
                  goodparams2.append([temperatures1[m[0]], ffs1[m[1]], np.power(10,logNexts1[m[2]]), temperatures2[i], ffs2[j], np.power(10,logNexts1[m[2]]), chisq, totalf])
                  if(count%50==0): print(count)

      dfn=pd.DataFrame(goodparams2,columns=['T1','ff1','N1', 'T2', 'ff2', 'N1','chi2','f'])#SAVING NCold on both, Nhot and NCold
      #pd.to_pickle(dfn,en+run+'goodparams2.pk')
      #print('Finished Goodparams2')
      return dfn


   def runloop(self):
      with Pool(10) as pool:
         output = pool.map(self.loop1, self.goodparams1)
      return output

# try running it
inputs = {}
inputs['t1'] = temperatures1
inputs['f1'] = ffs1
inputs['l1'] = logNexts1
inputs['t2'] = temperatures2
inputs['f2'] = ffs2
#inputs['l2'] = logNexts2
inputs['g1'] = goodparams1
inputs['selnus1'] = selnus1
testme = good2(inputs)


#output = testme.loopy()
output = pd.concat(testme.runloop())
print('created output')
minchi2s=np.min(output['chi2'] )  #6.359697581813473  lower than minc

dfsel1sn=output[(output['chi2']-minchi2s < np.sqrt(2*6))]   #6 free params fitted  3.46410
print('selecting values inside 1 sigma',np.shape(dfsel1sn) )# 4237
pd.to_pickle(dfsel1sn, 'finaldf'+en+run+'12enewMnewNolivinesmallC'+np.str(Cper)+'.pk') 
#pd.to_pickle(dfsel1sn, 'df'+en+run+'old5newMolivinesmallC'+np.str(Cper)+'NCold.pk')
#print('ran testme, saved df'+en+run+'old5newMolivinesmallC'+np.str(Cper)+'NCold.pk - ', len(output))
minc='%.0f' %(np.min(dfsel1sn['chi2'])) #2.9
maxc='%.0f' %(np.max(dfsel1sn['chi2'])) #6.4
minT1='%.0f' %(np.min(dfsel1sn['T1']))
maxT1='%.0f' %(np.max(dfsel1sn['T1']))
minff1='%.4f' %(np.min(dfsel1sn['ff1']))
maxff1='%.4f' %(np.max(dfsel1sn['ff1']))
minT2='%.0f' %(np.min(dfsel1sn['T2']))
maxT2='%.0f' %(np.max(dfsel1sn['T2']))
minff2='%.4f' %(np.min(dfsel1sn['ff2']))
maxff2='%.4f' %(np.max(dfsel1sn['ff2']))
minN1='%.6f' %(np.min(dfsel1sn['N1'])[0])
maxN1='%.6f' %(np.max(dfsel1sn['N1'])[0])
print('min max',minT1,maxT1,minff1,maxff1,minN1,maxN1,minT2,maxT2,minff2,maxff2,minc,maxc)
print('FINISHED FIT ellipse: '+en+'run:'+run)
amm=[Cper,minT1,maxT1,minff1,maxff1,minT2,maxT2,minff2,maxff2,minN1,maxN1,minc,maxc]
save('final'+en+run+'E12olivinesmallC'+np.str(Cper)+'.npy',amm)


dfn=dfsel1sn


#namepk='df'+en+run+'old5newMolivinesmallC25.pk' # <---------------------CHANGE!!--
#extsm,wlext=loadextAmMgFeolrvp1()
#extC,wlextC=loadextC()
#dfn=pd.read_pickle(namepk)#+'erasethis')
#revwlLMN=wlLMN[::-1]; selwls1=revwlLMN;

coln1=[]
for i in range(np.size(selwls1)):
    coln1.append('%.2f' %selwls1[i])

print(coln1)
cmap = plt.get_cmap('gist_rainbow')#('nipy_spectral')#('gist_ncar')# #

nusshort=nus[1100:3900]
lsshort=ls[1100:3900]


def fp(dfn,label,grav):   #leg=fp(dfn,'no')
    #selecting values inside 1 sigma
    minchi2s=np.min(dfn['chi2'] )  #6.359697581813473  lower than minc
    dfsel1sn=dfn[(dfn['chi2']-minchi2s < np.sqrt(2*6))]   #6 free params fitted  3.46410
    print('selecting values inside 1 sigma',np.shape(dfsel1sn) )# 4237
    minc=np.min(dfsel1sn['chi2']) #2.95
    maxc=np.max(dfsel1sn['chi2']) #6.4
    dfnbord1 = dfsel1sn.sort_values('chi2',ascending=False)
    dfnbord1[coln1]=pd.DataFrame(dfnbord1.f.tolist(), index= dfnbord1.index)#dfnbord1.columns
    minfwl=np.ones((lsshort.size))*20.
    maxfwl=np.zeros((lsshort.size))
    minT1=np.min(dfsel1sn['T1'])
    maxT1=np.max(dfsel1sn['T1'])
    minT2=np.min(dfsel1sn['T2'])
    maxT2=np.max(dfsel1sn['T2'])
    minff1=np.min(dfsel1sn['ff1'])
    maxff1=np.max(dfsel1sn['ff1'])
    minff2=np.min(dfsel1sn['ff2'])
    maxff2=np.max(dfsel1sn['ff2'])
    minN1=np.min(dfsel1sn['N1'])
    maxN1=np.max(dfsel1sn['N1'])
    #print('min max',minT1,maxT1,minff1,maxff1,minN1,maxN1,minT2,maxT2,minff2,maxff2,'chi2',minc,maxc)
    for index,row in dfnbord1.iterrows():#print(index,row)
        fmodcold=singlebb1AmMgFeolrvp1C(nusshort,row[0],row[1],row[2])
        fmodhot=singlebb1AmMgFeolrvp1C(nusshort,row[3],row[4],row[5])
        totalf=fmodcold+fmodhot
        for i in range(totalf.size):
           if(totalf[i]<minfwl[i]):minfwl[i]=totalf[i]
           if(totalf[i]>maxfwl[i]):maxfwl[i]=totalf[i]

    print(np.shape(nusshort))
    #   AVOID THIS!!!plt.plot(ls,totalf,color='grey')#,label=str(row[0])+'K &
    #plt.plot(lsshort,minfwl)#label='models inside 1 sigma')
    #plt.plot(lsshort,maxfwl)
    plt.fill_between(lsshort,minfwl,maxfwl,color='grey')
    print('grey')
     
    #plt.plot(selwl,b1,'o',color='black',label=en)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.xlim(xmin=1.8)
    plt.xlim(xmax=13.5)
    plt.ylim(ymin=0.0)
    plt.ylim(ymax=3.0)
    plt.grid(True)
    #plt.grid(which='minor',color='#CCCCCC', linestyle=':')
    if(grav==True):
       plt.xlim(xmin=1.8)
       gravfl=np.array([0.05333333, 0.0665 , 0.11, 0.23, 0.36])
       wlsk=np.array([2.0,2.1,2.2,2.3,2.4])
       nusk=c/np.array([2.0,2.1,2.2,2.3,2.4])
       #plotting Gravity fluxes
       plt.plot(wlsk,gravfl,'o',color='black',markersize='1',label='Gravity Collaboration 2020')#change error bar!!
    return#(legend,indsel)



#print(i)16
#plotb1(selwl,b1,stdb1)
#leg,ind=
fp(dfn,'no',True)
del(dfn)
plt.title('sed fit '+en+run)
print('[',Cper,',',minT1,',',maxT1,',',minff1,',',maxff1,',',minT2,',',maxT2,',',minff2,',',maxff2,',',minN1,',',maxN1,',',minc,',',maxc,']')

plt.ylim(ymax=ym)
plt.savefig('final'+en+run+'2bbnewMnewNolivinesmallp'+np.str(Cper)+'C-2.png')
plt.pause(5)
plt.xlim(xmin=2.5)
plt.savefig('final'+en+run+'2bbnewMnewNolivinesmallp'+np.str(Cper)+'C-1.png')
plt.pause(5)
#plt.xlim(xmin=1.9,xmax=2.5)
#plt.ylim(ymax=0.5)
#plt.savefig('final'+en+run+'1bbnewMnewNolivinesmallp'+np.str(Cper)+'C-3.png')
#plt.pause(5)

#leg,ind=fp(dfn,'yes')
#plt.savefig('celeg.png')
#plt.clf()

#pd.to_pickle(leg,'sedfitleg'+en+run+'newphot5ellipses.pk')

print('Finished plot'+en+run+'old5newMolivinesmallC'+np.str(Cper)+'NCold.png')
print('saved:   '+ en+run+'E12olivinesmallC'+np.str(Cper)+'.npy')
#print(leg)
#plt.clf()

#PLOT TO HERE
#next: basic_chi2grid for triangle plots
