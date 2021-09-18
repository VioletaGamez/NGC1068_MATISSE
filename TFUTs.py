from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
hdu=fits.open('/Users/M51/Downloads/NbandNGC1068/newN7CPN21VIS/2018-09-24T045210_NGC1068_U1U2U3U4_IR-N_LOW_IN_IN_noChop_cal_oifits_0.fits')
hdu=fits.open('/Users/M51/Downloads/NbandNGC1068/newN7CPN21VIS/2018-09-24T045210_NGC1068_U1U2U3U4_IR-N_LOW_OUT_OUT_noChop_cal_oifits_0.fits')
hdu=fits.open('/Users/M51/Downloads/NbandNGC1068/newN7CPN21VIS/2018-09-25T052136_NGC1068_U1U2U3U4_IR-N_LOW_IN_IN_noChop_cal_oifits_0.fits')
hdu=fits.open('/Users/M51/Downloads/NbandNGC1068/newN7CPN21VIS/2018-09-25T052136_NGC1068_U1U2U3U4_IR-N_LOW_OUT_OUT_noChop_cal_oifits_0.fits')
hdu=fits.open('/Users/M51/Downloads/NbandNGC1068/newN7CPN21VIS/2019-11-07T013000_M77_U1U2U3U4_IR-N_LOW_IN_IN_noChop_cal_oifits_0.fits')#A
hdu=fits.open('/Users/M51/Downloads/NbandNGC1068/newN7CPN21VIS/2019-11-07T013000_M77_U1U2U3U4_IR-N_LOW_IN_OUT_noChop_cal_oifits_0.fits')
hdu=fits.open('/Users/M51/Downloads/NbandNGC1068/newN7CPN21VIS/2019-11-07T013000_M77_U1U2U3U4_IR-N_LOW_OUT_IN_noChop_cal_oifits_0.fits')
hdu=fits.open('/Users/M51/Downloads/NbandNGC1068/newN7CPN21VIS/2019-11-07T013000_M77_U1U2U3U4_IR-N_LOW_OUT_OUT_noChop_cal_oifits_0.fits')
hdu=fits.open('/Users/M51/Downloads/NbandNGC1068/newN7CPN21VIS/2019-11-07T051740_M77_U1U2U3U4_IR-N_LOW_IN_IN_noChop_cal_oifits_0.fits')#B
hdu=fits.open('/Users/M51/Downloads/NbandNGC1068/newN7CPN21VIS/2019-11-07T051740_M77_U1U2U3U4_IR-N_LOW_OUT_IN_noChop_cal_oifits_0.fits')
hdu=fits.open('/Users/M51/Downloads/NbandNGC1068/newN7CPN21VIS/2019-11-07T051740_M77_U1U2U3U4_IR-N_LOW_OUT_OUT_noChop_cal_oifits_0.fits')
hdu=fits.open('/Users/M51/Downloads/NbandNGC1068/newN7CPN21VIS/2019-11-07T065701_M77_U1U2U3U4_IR-N_LOW_IN_IN_noChop_cal_oifits_0.fits')#C
hdu=fits.open('/Users/M51/Downloads/NbandNGC1068/newN7CPN21VIS/2019-11-07T065701_M77_U1U2U3U4_IR-N_LOW_IN_OUT_noChop_cal_oifits_0.fits')
hdu=fits.open('/Users/M51/Downloads/NbandNGC1068/newN7CPN21VIS/2019-11-07T065701_M77_U1U2U3U4_IR-N_LOW_OUT_IN_noChop_cal_oifits_0.fits')
hdu=fits.open('/Users/M51/Downloads/NbandNGC1068/newN7CPN21VIS/2019-11-07T065701_M77_U1U2U3U4_IR-N_LOW_OUT_OUT_noChop_cal_oifits_0.fits')

hduii0=fits.open('/Users/M51/Downloads/NbandNGC1068/newN7CPN21VIS/2018-09-24T045210_NGC1068_U1U2U3U4_IR-N_LOW_IN_IN_noChop_cal_oifits_0.fits')
hduoo0=fits.open('/Users/M51/Downloads/NbandNGC1068/newN7CPN21VIS/2018-09-24T045210_NGC1068_U1U2U3U4_IR-N_LOW_OUT_OUT_noChop_cal_oifits_0.fits')
hduii1=fits.open('/Users/M51/Downloads/NbandNGC1068/newN7CPN21VIS/2018-09-25T052136_NGC1068_U1U2U3U4_IR-N_LOW_IN_IN_noChop_cal_oifits_0.fits')
hduoo1=fits.open('/Users/M51/Downloads/NbandNGC1068/newN7CPN21VIS/2018-09-25T052136_NGC1068_U1U2U3U4_IR-N_LOW_OUT_OUT_noChop_cal_oifits_0.fits')
hduii2=fits.open('/Users/M51/Downloads/NbandNGC1068/newN7CPN21VIS/2019-11-07T013000_M77_U1U2U3U4_IR-N_LOW_IN_IN_noChop_cal_oifits_0.fits')#A
hduio2=fits.open('/Users/M51/Downloads/NbandNGC1068/newN7CPN21VIS/2019-11-07T013000_M77_U1U2U3U4_IR-N_LOW_IN_OUT_noChop_cal_oifits_0.fits')
hduoi2=fits.open('/Users/M51/Downloads/NbandNGC1068/newN7CPN21VIS/2019-11-07T013000_M77_U1U2U3U4_IR-N_LOW_OUT_IN_noChop_cal_oifits_0.fits')
hduoo2=fits.open('/Users/M51/Downloads/NbandNGC1068/newN7CPN21VIS/2019-11-07T013000_M77_U1U2U3U4_IR-N_LOW_OUT_OUT_noChop_cal_oifits_0.fits')
hduii3=fits.open('/Users/M51/Downloads/NbandNGC1068/newN7CPN21VIS/2019-11-07T051740_M77_U1U2U3U4_IR-N_LOW_IN_IN_noChop_cal_oifits_0.fits')#B
hduoi3=fits.open('/Users/M51/Downloads/NbandNGC1068/newN7CPN21VIS/2019-11-07T051740_M77_U1U2U3U4_IR-N_LOW_OUT_IN_noChop_cal_oifits_0.fits')
hduoo3=fits.open('/Users/M51/Downloads/NbandNGC1068/newN7CPN21VIS/2019-11-07T051740_M77_U1U2U3U4_IR-N_LOW_OUT_OUT_noChop_cal_oifits_0.fits')
hduii4=fits.open('/Users/M51/Downloads/NbandNGC1068/newN7CPN21VIS/2019-11-07T065701_M77_U1U2U3U4_IR-N_LOW_IN_IN_noChop_cal_oifits_0.fits')#C
hduio4=fits.open('/Users/M51/Downloads/NbandNGC1068/newN7CPN21VIS/2019-11-07T065701_M77_U1U2U3U4_IR-N_LOW_IN_OUT_noChop_cal_oifits_0.fits')
hduoi4=fits.open('/Users/M51/Downloads/NbandNGC1068/newN7CPN21VIS/2019-11-07T065701_M77_U1U2U3U4_IR-N_LOW_OUT_IN_noChop_cal_oifits_0.fits')
hduoo4=fits.open('/Users/M51/Downloads/NbandNGC1068/newN7CPN21VIS/2019-11-07T065701_M77_U1U2U3U4_IR-N_LOW_OUT_OUT_noChop_cal_oifits_0.fits')

#FILES FOR 3um  wrong, non-chopped
hdu=fits.open('/Users/M51/Downloads/binning_experiment-2/merged_mjd_split/m77_2018-09-22T04_11_40_in-in.fits')
hdu=fits.open('/Users/M51/Downloads/binning_experiment-2/merged_mjd_split/m77_2018-09-22T04_11_40_out-out.fits')
hdu=fits.open('/Users/M51/Downloads/binning_experiment-2/merged_mjd_split/m77_2018-09-25T05_21_36_in-in.fits')
hdu=fits.open('/Users/M51/Downloads/binning_experiment-2/merged_mjd_split/m77_2018-09-25T05_21_36_out-out.fits')
hdu=fits.open('/Users/M51/Downloads/binning_experiment-2/merged_mjd_split/m77_2019-11-07T01_30_00_in-in.fits')
hdu=fits.open('/Users/M51/Downloads/binning_experiment-2/merged_mjd_split/m77_2019-11-07T01_30_00_out-out.fits')
hdu=fits.open('/Users/M51/Downloads/binning_experiment-2/merged_mjd_split/m77_2019-11-07T05_17_40_in-in.fits')
hdu=fits.open('/Users/M51/Downloads/binning_experiment-2/merged_mjd_split/m77_2019-11-07T05_17_40_out-out.fits')
hdu=fits.open('/Users/M51/Downloads/binning_experiment-2/merged_mjd_split/m77_2019-11-07T06_57_01_in-in.fits')
hdu=fits.open('/Users/M51/Downloads/binning_experiment-2/merged_mjd_split/m77_2019-11-07T06_57_01_out-out.fits')



ti='Sep22II'#'Nov7COO'
tiv2=ti+'vis2'
#ti='25SeptOOvis2'  #25Sept baselines with UT2 look great!! OO bl6 weird
for i in range(0,6):  #by BL
   print(np.shape(hdu[4].data['VIS2DATA'])[0])
   n=np.shape(hdu[4].data['VIS2DATA'])[0]/6
   for j in range(0,int(n)):
      plt.plot(hdu[3].data['EFF_WAVE']*1e6, hdu[4].data['VIS2DATA'][i+6*j],label='mjd'+str(j)+':'+str( hdu[4].data['MJD'][i+6*j]))
   plt.title(tiv2+', baseline'+str(i+1)+':'+str(hdu[4].data['STA_INDEX'][i]))
   plt.xlabel('wl [um]')
   plt.ylabel('vis2')
   plt.legend()
   plt.grid(linestyle="--",linewidth=0.1,color='.25')
   #plt.savefig(ti+str(i)+'.png')#has to go before show
   plt.show()

ti='Sep25OO'#'Nov7COO'
ticp=ti+'t3phi'
for i in range(0,4):
   print(np.shape(hdu[5].data['T3PHI'])[0])
   n=np.shape(hdu[5].data['T3PHI'])[0]/4
   for j in range(0,int(n)):
      plt.plot(hdu[3].data['EFF_WAVE']*1e6, hdu[5].data['T3PHI'][i+4*j],label='mjd'+str(j)+':'+str( hdu[5].data['MJD'][i+4*j]))
   plt.title(ticp+', triplet'+str(i+1)+':'+str(hdu[5].data['STA_INDEX'][i]))
   plt.xlabel('wl [um]')
   plt.ylabel('t3phi [deg]')
   plt.legend()
   plt.grid(linestyle="--",linewidth=0.1,color='.25')
   #plt.savefig(ti+str(i)+'.png')#has to go before show
   plt.show()
#chopped but with averages
hdus=['/Users/M51/Downloads/chopped_and_binned/m77_LM_2018-09-22T04_11_40_in-in.fits','/Users/M51/Downloads/chopped_and_binned/m77_LM_2018-09-22T04_11_40_out-out.fits','/Users/M51/Downloads/chopped_and_binned/m77_LM_2018-09-25T05_21_36_in-in.fits','/Users/M51/Downloads/chopped_and_binned/m77_LM_2018-09-25T05_21_36_out-out.fits','/Users/M51/Downloads/chopped_and_binned/m77_LM_2019-11-07T01_30_00_in-in.fits','/Users/M51/Downloads/chopped_and_binned/m77_LM_2019-11-07T01_30_00_out-out.fits','/Users/M51/Downloads/chopped_and_binned/m77_LM_2019-11-07T05_17_40_in-in.fits','/Users/M51/Downloads/chopped_and_binned/m77_LM_2019-11-07T05_17_40_out-out.fits','/Users/M51/Downloads/chopped_and_binned/m77_LM_2019-11-07T06_57_01_in-in.fits','/Users/M51/Downloads/chopped_and_binned/m77_LM_2019-11-07T06_57_01_out-out.fits']
#good ones
hdus=['/Users/M51/Downloads/chopped_and_binned-2/m77_LM_2018-09-22T04_11_40_in-in.fits','/Users/M51/Downloads/chopped_and_binned-2/m77_LM_2018-09-22T04_11_40_out-out.fits','/Users/M51/Downloads/chopped_and_binned-2/m77_LM_2018-09-25T05_21_36_in-in.fits','/Users/M51/Downloads/chopped_and_binned-2/m77_LM_2018-09-25T05_21_36_out-out.fits','/Users/M51/Downloads/chopped_and_binned-2/m77_LM_2019-11-07T01_30_00_in-in.fits','/Users/M51/Downloads/chopped_and_binned-2/m77_LM_2019-11-07T01_30_00_out-out.fits','/Users/M51/Downloads/chopped_and_binned-2/m77_LM_2019-11-07T05_17_40_in-in.fits','/Users/M51/Downloads/chopped_and_binned-2/m77_LM_2019-11-07T05_17_40_out-out.fits','/Users/M51/Downloads/chopped_and_binned-2/m77_LM_2019-11-07T06_57_01_in-in.fits','/Users/M51/Downloads/chopped_and_binned-2/m77_LM_2019-11-07T06_57_01_out-out.fits']
#The tuple function casts an iterable (in this case a list) as a tuple object, and np.vstack takes a tuple of numpy arrays as argument. test
a = []
b= np.array([1,2,3])
for i in range(10):
  a.append(b)
a = np.vstack( tuple(a) )
np.mean(a[:,[1]])
np.std(a[:,[1]])

for k in range(10):
   hdu=fits.open(hdus[k])
   ti=hdus[k][39:]  #Only 1 data, when there were 2 sub-observations, is it an avg?
   tiv2=ti+'vis2'
   for i in range(0,6):  #by BL
      print(np.shape(hdu[4].data['VIS2DATA'])[0])
      n=np.shape(hdu[4].data['VIS2DATA'])[0]/6
      #vis2s=np.zeros(26)#from 2.84 to 3.53. needed?
      table=[]
      for j in range(0,int(n)):
         plt.plot(hdu[3].data['EFF_WAVE']*1e6, hdu[4].data['VIS2DATA'][i+6*j],label='mjd'+str(j)+':'+str( hdu[4].data['MJD'][i+6*j]))
         vis2s=np.array(hdu[4].data['VIS2DATA'][i+6*j][80:])
         #print('vis2s',vis2s)
         table.append(vis2s)
         
      table=np.vstack(tuple(table))
      #print('table',table)
      means=[]; stds=[]
      for l in range(26):
         means.append(np.mean(table[:,[l]]))
         stds.append(np.std(table[:,[l]]))
         if(np.std(table[:,[l]])>0.25*np.mean(table[:,[l]])):print('BL'+str(i+1), 'mean:', "%.4f" % np.mean(table[:,[l]]), 'std:', "%.4f" % np.std(table[:,[l]]), 'ratio:',  "%.4f" % (np.std(table[:,[l]])/np.mean(table[:,[l]])), 'wl:',hdu[3].data['EFF_WAVE'][80+l]*1e6)
      plt.title(tiv2+', baseline'+str(i+1)+':'+str(hdu[4].data['STA_INDEX'][i]))
      plt.xlabel('wl [um]')
      plt.ylabel('vis2')
      plt.legend()
      plt.xlim(xmin=2.84,xmax=3.53)
      plt.ylim(ymin=-0.1,ymax=0.5)
      plt.grid(linestyle="--",linewidth=0.1,color='.25')
      #plt.savefig(ti+str(i)+'.png')#has to go before show
      plt.show()
      input()


for k in range(10):
   hdu=fits.open(hdus[k])
   ti=hdus[k][39:]  #Only 1 data, when there were 2 sub-observations, is it an avg?
   ticp=ti+'t3phi'
   for i in range(0,4):
      print(np.shape(hdu[5].data['T3PHI'])[0])
      n=np.shape(hdu[5].data['T3PHI'])[0]/4
      print(int(n))
      table=[]
      for j in range(0,int(n)):
         print(j)
         plt.plot(hdu[3].data['EFF_WAVE']*1e6, hdu[5].data['T3PHI'][i+4*j],label='mjd'+str(j)+':'+str( hdu[5].data['MJD'][i+4*j]))
         cps=np.array(hdu[5].data['T3PHI'][i+4*j][80:])
         #print('cps',cps)
         table.append(cps)
                
      table=np.vstack(tuple(table))
      #print('table',table)
      means=[]; stds=[]
      for l in range(26):
         means.append(np.mean(table[:,[l]]))
         stds.append(np.std(table[:,[l]]))
         if(np.std(table[:,[l]])>30):print('TR:'+str(i+1), 'mean:', "%.4f" % np.mean(table[:,[l]]), 'std:', "%.4f" % np.std(table[:,[l]]), 'ratio:',  "%.4f" % (np.std(table[:,[l]])/np.mean(table[:,[l]])), 'wl:',hdu[3].data['EFF_WAVE'][80+l]*1e6)
    
      plt.title(ticp+', triplet'+str(i+1)+':'+str(hdu[5].data['STA_INDEX'][i]))
      plt.xlabel('wl [um]')
      plt.ylabel('t3phi [deg]')
      plt.legend()
      plt.xlim(xmin=2.84,xmax=3.53)
      plt.ylim(ymin=-190,ymax=190)
      plt.grid(linestyle="--",linewidth=0.1,color='.25')
      #plt.savefig(ti+str(i)+'.png')#has to go before show
      plt.show()
      input()



for i in range(10):
   hdu=fits.open(hdus[i])
   ti=hdus[i][39:]  #Only 1 data, when there were 2 sub-observations, is it an avg?
   n=hdu[7].data['FLUXDATA'].shape[0]
   for i in range(0,n):
      plt.plot(hdu[3].data['EFF_WAVE']*1e6, hdu[7].data['FLUXDATA'][i],linestyle='-',label='order:'+str(i+1)+',T:'+str(hdu[7].data['STA_INDEX'][i])+'mjd:'+str( hdu[5].data['MJD'][i]))
  
   plt.title(ti)
   plt.xlabel('wl [um]')
   plt.ylabel('flux [instrumental units]')
   plt.legend()
   #plt.savefig(ti+'.png')#has to go before show
   plt.show()
   plt.pause(20)
  
   n=hdu[7].data['FLUXDATA'].shape[0]
   for i in range(0,n):
      plt.plot(hdu[3].data['EFF_WAVE']*1e6, hdu[7].data['FLUXDATA'][i],linestyle='--',label='order:'+str(i+1)+',T:'+str(hdu[7].data['STA_INDEX'][i])+'mjd:'+str( hdu[5].data['MJD'][i]))


   plt.title(ti)
   plt.xlabel('wl [um]')
   plt.ylabel('flux [instrumental units]')
   plt.legend()
   #plt.savefig(ti+'.png')#has to go before show
   plt.show()
   plt.pause(20)



----------
ti='25SeptAllBCDscalibratedphotometry'  #Only 1 data, when there were 2 sub-observations, is it an avg?
hdu=fits.open('/Users/M51/Downloads/NbandNGC1068/newN7CPN21VIS/2018-09-25T052136_NGC1068_U1U2U3U4_IR-N_LOW_IN_IN_noChop_cal_oifits_0.fits')
bcd='II'
n=hdu[7].data['FLUXDATA'].shape[0]
for i in range(0,n):
   plt.plot(hdu[3].data['EFF_WAVE']*1e6, hdu[7].data['FLUXDATA'][i],linestyle='-',label=str(bcd)+',order:'+str(i+1)+',T:'+str(hdu[7].data['STA_INDEX'][i])+'mjd:'+str( hdu[5].data['MJD'][i]))
  
hdu=fits.open('/Users/M51/Downloads/NbandNGC1068/newN7CPN21VIS/2018-09-25T052136_NGC1068_U1U2U3U4_IR-N_LOW_OUT_OUT_noChop_cal_oifits_0.fits')
bcd='OO'
n=hdu[7].data['FLUXDATA'].shape[0]
for i in range(0,n):
   plt.plot(hdu[3].data['EFF_WAVE']*1e6, hdu[7].data['FLUXDATA'][i],linestyle='--',label=str(bcd)+',order:'+str(i+1)+',T:'+str(hdu[7].data['STA_INDEX'][i])+'mjd:'+str( hdu[5].data['MJD'][i]))


plt.title(ti)
plt.xlabel('wl [um]')
plt.ylabel('flux [instrumental units]')
plt.legend()
#plt.savefig(ti+'.png')#has to go before show
plt.show()



ti='Nov7-A AllBCDscalibratedphotometry'
hdu=fits.open('/Users/M51/Downloads/NbandNGC1068/newN7CPN21VIS/2019-11-07T013000_M77_U1U2U3U4_IR-N_LOW_IN_IN_noChop_cal_oifits_0.fits')#A
bcd='II'
n=hdu[7].data['FLUXDATA'].shape[0]
for i in range(0,n):
   plt.plot(hdu[3].data['EFF_WAVE']*1e6, hdu[7].data['FLUXDATA'][i],linestyle='-',label=str(bcd)+',order:'+str(i+1)+',T:'+str(hdu[7].data['STA_INDEX'][i])+'mjd:'+str( hdu[5].data['MJD'][i]))
  

bcd='IO'
hdu=fits.open('/Users/M51/Downloads/NbandNGC1068/newN7CPN21VIS/2019-11-07T013000_M77_U1U2U3U4_IR-N_LOW_IN_OUT_noChop_cal_oifits_0.fits')
n=hdu[7].data['FLUXDATA'].shape[0]
for i in range(0,n):
   plt.plot(hdu[3].data['EFF_WAVE']*1e6, hdu[7].data['FLUXDATA'][i],linestyle=':',label=str(bcd)+',order:'+str(i+1)+',T:'+str(hdu[7].data['STA_INDEX'][i])+'mjd:'+str( hdu[5].data['MJD'][i]))

bcd='OI'
hdu=fits.open('/Users/M51/Downloads/NbandNGC1068/newN7CPN21VIS/2019-11-07T013000_M77_U1U2U3U4_IR-N_LOW_OUT_IN_noChop_cal_oifits_0.fits')
n=hdu[7].data['FLUXDATA'].shape[0]
for i in range(0,n):
   plt.plot(hdu[3].data['EFF_WAVE']*1e6, hdu[7].data['FLUXDATA'][i],linestyle='-.',label=str(bcd)+',order:'+str(i+1)+',T:'+str(hdu[7].data['STA_INDEX'][i])+'mjd:'+str( hdu[5].data['MJD'][i]))

hdu=fits.open('/Users/M51/Downloads/NbandNGC1068/newN7CPN21VIS/2019-11-07T013000_M77_U1U2U3U4_IR-N_LOW_OUT_OUT_noChop_cal_oifits_0.fits')
bcd='OO'
n=hdu[7].data['FLUXDATA'].shape[0]
for i in range(0,n):
   plt.plot(hdu[3].data['EFF_WAVE']*1e6, hdu[7].data['FLUXDATA'][i],linestyle='--',label=str(bcd)+',order:'+str(i+1)+',T:'+str(hdu[7].data['STA_INDEX'][i])+'mjd:'+str( hdu[5].data['MJD'][i]))


plt.title(ti)
plt.xlabel('wl [um]')
plt.ylabel('flux [instrumental units]')
plt.legend()
#plt.savefig(ti+'.png')#has to go before show
plt.ylim(ymin=0.0)
plt.show()


ti='Nov7-B AllBCDscalibratedphotometry'
bcd='II'
hdu=fits.open('/Users/M51/Downloads/NbandNGC1068/newN7CPN21VIS/2019-11-07T051740_M77_U1U2U3U4_IR-N_LOW_IN_IN_noChop_cal_oifits_0.fits')#B
n=hdu[7].data['FLUXDATA'].shape[0]
for i in range(0,n):
   plt.plot(hdu[3].data['EFF_WAVE']*1e6, hdu[7].data['FLUXDATA'][i],linestyle='-',label=str(bcd)+',order:'+str(i+1)+',T:'+str(hdu[7].data['STA_INDEX'][i])+'mjd:'+str( hdu[5].data['MJD'][i]))

bcd='OI'
hdu=fits.open('/Users/M51/Downloads/NbandNGC1068/newN7CPN21VIS/2019-11-07T051740_M77_U1U2U3U4_IR-N_LOW_OUT_IN_noChop_cal_oifits_0.fits')
n=hdu[7].data['FLUXDATA'].shape[0]
for i in range(0,n):
   plt.plot(hdu[3].data['EFF_WAVE']*1e6, hdu[7].data['FLUXDATA'][i],linestyle='-.',label=str(bcd)+',order:'+str(i+1)+',T:'+str(hdu[7].data['STA_INDEX'][i])+'mjd:'+str( hdu[5].data['MJD'][i]))

bcd='OO'
hdu=fits.open('/Users/M51/Downloads/NbandNGC1068/newN7CPN21VIS/2019-11-07T051740_M77_U1U2U3U4_IR-N_LOW_OUT_OUT_noChop_cal_oifits_0.fits')
n=hdu[7].data['FLUXDATA'].shape[0]
for i in range(0,n):
   plt.plot(hdu[3].data['EFF_WAVE']*1e6, hdu[7].data['FLUXDATA'][i],linestyle='--',label=str(bcd)+',order:'+str(i+1)+',T:'+str(hdu[7].data['STA_INDEX'][i])+'mjd:'+str( hdu[5].data['MJD'][i]))


plt.title(ti)
plt.xlabel('wl [um]')
plt.ylabel('flux [instrumental units]')
plt.legend()
#plt.savefig(ti+'.png')#has to go before show
plt.ylim(ymin=0.0)
plt.show()


ti='Nov7-C AllBCDscalibratedphotometry'
bcd='II'
hdu=fits.open('/Users/M51/Downloads/NbandNGC1068/newN7CPN21VIS/2019-11-07T065701_M77_U1U2U3U4_IR-N_LOW_IN_IN_noChop_cal_oifits_0.fits')#C
n=hdu[7].data['FLUXDATA'].shape[0]
for i in range(0,n):
   plt.plot(hdu[3].data['EFF_WAVE']*1e6, hdu[7].data['FLUXDATA'][i],linestyle='-',label=str(bcd)+',order:'+str(i+1)+',T:'+str(hdu[7].data['STA_INDEX'][i])+'mjd:'+str( hdu[5].data['MJD'][i]))
  

bcd='IO'
hdu=fits.open('/Users/M51/Downloads/NbandNGC1068/newN7CPN21VIS/2019-11-07T065701_M77_U1U2U3U4_IR-N_LOW_IN_OUT_noChop_cal_oifits_0.fits')
n=hdu[7].data['FLUXDATA'].shape[0]
for i in range(0,n):
   plt.plot(hdu[3].data['EFF_WAVE']*1e6, hdu[7].data['FLUXDATA'][i],linestyle=':',label=str(bcd)+',order:'+str(i+1)+',T:'+str(hdu[7].data['STA_INDEX'][i])+'mjd:'+str( hdu[5].data['MJD'][i]))

bcd='OI'
hdu=fits.open('/Users/M51/Downloads/NbandNGC1068/newN7CPN21VIS/2019-11-07T065701_M77_U1U2U3U4_IR-N_LOW_OUT_IN_noChop_cal_oifits_0.fits')
n=hdu[7].data['FLUXDATA'].shape[0]
for i in range(0,n):
   plt.plot(hdu[3].data['EFF_WAVE']*1e6, hdu[7].data['FLUXDATA'][i],linestyle='-.',label=str(bcd)+',order:'+str(i+1)+',T:'+str(hdu[7].data['STA_INDEX'][i])+'mjd:'+str( hdu[5].data['MJD'][i]))

bcd='OO'
hdu=fits.open('/Users/M51/Downloads/NbandNGC1068/newN7CPN21VIS/2019-11-07T065701_M77_U1U2U3U4_IR-N_LOW_OUT_OUT_noChop_cal_oifits_0.fits')
n=hdu[7].data['FLUXDATA'].shape[0]
for i in range(0,n):
   plt.plot(hdu[3].data['EFF_WAVE']*1e6, hdu[7].data['FLUXDATA'][i],linestyle='--',label=str(bcd)+',order:'+str(i+1)+',T:'+str(hdu[7].data['STA_INDEX'][i])+'mjd:'+str( hdu[5].data['MJD'][i]))


plt.title(ti)
plt.xlabel('wl [um]')
plt.ylabel('flux [instrumental units]')
plt.legend()
#plt.savefig(ti+'.png')#has to go before show
plt.ylim(ymin=0.0)
plt.show()

CONCLUSION:
on Sept spread is much larger.
ON NovB one T looks worst.
In general it looks good.

28March2020 looking at corr flxs
from mcdb import wutilmod as wu
#reduced in corr flux   NON CALIBR
fitsii='/Volumes/MD/nov2019data/065701NGC1068afterctSel/ReducedCorrF/Iter1/mat_raw_estimates.2019-11-07T06_57_01.HAWAII-2RG.rb/TARGET_RAW_INT_0001.fits'
fitsoo='/Volumes/MD/nov2019data/065701NGC1068afterctSel/ReducedCorrF/Iter1/mat_raw_estimates.2019-11-07T06_57_01.HAWAII-2RG.rb/TARGET_RAW_INT_0002.fits'
mavgvis2,mstdvis2,mcalcp,mcalcpcomplex,mstdcp1,mstdcp2,avgbl,avgpa=wu.getavgstddrs(fitsii,fitsoo)  #order as in OO?

#reduced in corr flux  CALIBRATED  USING SPECTRA: ---

hdu=fits.open('/Volumes/MD/nov2019data/065701NGC1068afterctSel/ReducedCorrF/Iter1/mat_raw_estimates.2019-11-07T06_57_01.HAWAII-2RG.rb/TARGET_RAW_INT_0001.fits')#CHANGE
hdu=fits.open('/Volumes/MD/nov2019data/065701NGC1068afterctSel/ReducedCorrF/Iter1/mat_raw_estimates.2019-11-07T06_57_01.HAWAII-2RG.rb/TARGET_RAW_INT_0002.fits')#CHANGE  01 or 02 is the same
hdu[4].data['STA_INDEX']
from matplotlib import cm
#plasma hdu[4].data['STA_INDEX']
baseline=['UT3-UT4, '+str(round(avgbl[0],0))+'m, '+str(round(avgpa[0],0))+'deg','UT1-UT2, '+str(round(avgbl[1],0))+'m, '+str(round(avgpa[1],0))+'deg','UT2-UT3, '+str(round(avgbl[2],0))+'m, '+str(round(avgpa[2],0))+'deg','UT2-UT4, '+str(round(avgbl[3],0))+'m, '+str(round(avgpa[3],0))+'deg','UT1-UT3, '+str(round(avgbl[4],0))+'m, '+str(round(avgpa[4],0))+'deg','UT1-UT4, '+str(round(avgbl[5],0))+'m, '+str(round(avgpa[5],0))+'deg' ]#based on OO
plt.clf()
cmap = plt.get_cmap('nipy_spectral')
os.chdir('/Users/M51/Downloads/NovemberDataDRS')for bl in range(6):  #plots last defined magvis2
    c = cmap(float(bl)/6)
    plt.plot(hdu[3].data['EFF_WAVE']*1e6,mavgvis2[bl],color=c,label=baseline[bl])
    y=np.array(mavgvis2[bl],dtype=float)
    sigma=np.array(mstdvis2[bl],dtype=float)
    plt.fill_between(hdu[3].data['EFF_WAVE']*1e6,y+sigma,y-sigma,color=c,alpha=0.5)

#plt.ylim(0,0.15)
plt.legend()
plt.xlabel('wl [um]')
plt.ylabel('Correlated Flux? (label vis2 in fits file)')
plt.title(str(hdu[0].header['OBJECT'])+' '+str(hdu[0].header['DATE-OBS'])+'non-cal '+str(hdu[0].header['HIERARCH ESO INS BCD1 ID'])+str(hdu[0].header['HIERARCH ESO INS BCD2 ID']))


#comparing to reduced for VIS2

Dec212019
#Plotting vis2 and clph , calibrated by BCD!!!
from mcdb import wutilmod as wu
fitsii='/Users/M51/Dropbox/MATISSEfiles/dataNGC1068paper/NGC1068CalibratedWithSelectedFiles/S22HD16658selected/TARGET_CAL_INT_0002.fits'#3
fitsoo='/Users/M51/Dropbox/MATISSEfiles/dataNGC1068paper/NGC1068CalibratedWithSelectedFiles/S22HD16658selected/TARGET_CAL_INT_0001.fits'#4
mavgvis2,mstdvis2,mcalcp,mcalcpcomplex,mstdcp1,mstdcp2,avgbl,avgpa=wu.getavgstddrs(fitsii,fitsoo)
fitsii='/Users/M51/Dropbox/MATISSEfiles/dataNGC1068paper/NGC1068CalibratedWithSelectedFiles/S25hd209240allgood/TARGET_CAL_INT_0002.fits'
fitsoo='/Users/M51/Dropbox/MATISSEfiles/dataNGC1068paper/NGC1068CalibratedWithSelectedFiles/S25hd209240allgood/TARGET_CAL_INT_0001.fits'
mavgvis2,mstdvis2,mcalcp,mcalcpcomplex,mstdcp1,mstdcp2,avgbl,avgpa=wu.getavgstddrs(fitsii,fitsoo)
#Novdata
fitsf='/Users/M51/Downloads/NovemberDataDRS/1TARGET_CAL_INT_0001.fits'
#wu.plotuvpoints(fitsf,'blue') # 3Ts
fitsf='/Users/M51/Downloads/NovemberDataDRS/1TARGET_CAL_INT_0002.fits'
#plotuvpoints(fitsf,colr)
fitsoo='/Users/M51/Downloads/NovemberDataDRS/2TARGET_CAL_INT_0001.fits'
fitsii='/Users/M51/Downloads/NovemberDataDRS/2TARGET_CAL_INT_0002.fits'
mavgvis2,mstdvis2,mcalcp,mcalcpcomplex,mstdcp1,mstdcp2,avgbl,avgpa=wu.getavgstddrs(fitsii,fitsoo)
fitsoo='/Users/M51/Downloads/NovemberDataDRS/3TARGET_CAL_INT_0001.fits'
fitsii='/Users/M51/Downloads/NovemberDataDRS/3TARGET_CAL_INT_0002.fits'
mavgvis2,mstdvis2,mcalcp,mcalcpcomplex,mstdcp1,mstdcp2,avgbl,avgpa=wu.getavgstddrs(fitsii,fitsoo)

from matplotlib import cm
#plasma hdu[4].data['STA_INDEX']
baseline=['UT3-UT4, '+str(round(avgbl[0],0))+'m, '+str(round(avgpa[0],0))+'deg','UT1-UT2, '+str(round(avgbl[1],0))+'m, '+str(round(avgpa[1],0))+'deg','UT2-UT3, '+str(round(avgbl[2],0))+'m, '+str(round(avgpa[2],0))+'deg','UT2-UT4, '+str(round(avgbl[3],0))+'m, '+str(round(avgpa[3],0))+'deg','UT1-UT3, '+str(round(avgbl[4],0))+'m, '+str(round(avgpa[4],0))+'deg','UT1-UT4, '+str(round(avgbl[5],0))+'m, '+str(round(avgpa[5],0))+'deg' ]#based on OO
plt.clf()
cmap = plt.get_cmap('nipy_spectral')
from astropy.io import fits
hdu=fits.open('/Users/M51/Dropbox/MATISSEfiles/dataNGC1068paper/NGC1068CalibratedWithSelectedFiles/S22HD16658selected/TARGET_CAL_INT_0002.fits') #only to get array wl
for bl in range(6):  #plots last defined magvis2
    c = cmap(float(bl)/6)
    plt.plot(hdu[3].data['EFF_WAVE']*1e6,mavgvis2[bl],color=c,label=baseline[bl])
    y=np.array(mavgvis2[bl],dtype=float)
    sigma=np.array(mstdvis2[bl],dtype=float)
    plt.fill_between(hdu[3].data['EFF_WAVE']*1e6,y+sigma,y-sigma,color=c,alpha=0.5)

plt.ylim(0,0.15)
plt.legend()
plt.xlabel('wl [um]')
plt.ylabel('vis2')
#PLOTTING  CLOSURE PHASES  w/error<----PENDING!
#viridis, rainbow, jet
triplet=["UT2-UT3-UT4","UT1-UT2-UT3","UT1-UT2-UT4","UT1-UT3-UT4",] #based on OO
plt.clf()
cmap = plt.get_cmap('viridis')
for tr in range(4):
    c = cmap(float(tr)/4)
    plt.plot(hdu[3].data['EFF_WAVE']*1e6, mcalcp[tr][:],color=c,label=triplet[tr])
    y=np.array(mcalcp[tr][:],dtype=float)
    sigma=np.array(mstdcp1[tr],dtype=float)
    plt.fill_between(hdu[3].data['EFF_WAVE']*1e6,y+sigma,y-sigma,color=c,alpha=0.5)

plt.title('Closure Phases Calibrated by BCD NGC 1068 Nov 6 b) and c), Sept 22 and 25 ')     #CHANGE
plt.ylim(-182,182)
plt.legend()
plt.xlabel('wavelength [um]')
plt.ylabel('closure phase [deg]')
#final plot uv plane
#mcdb not working Oct292020
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle
import glob
import numpy as np
#from mcdb import fitsutil as fu
#from mcdb import matutil  as mu
#from mcdb import Mcdb as mc
#from mcdb import wutil as wu
import fnmatch
import pandas as pd
from astropy.io import fits
from numpy import pi, cos, sin
import scipy
from scipy.special import jv
from scipy.optimize import curve_fit
from astropy import units as u
from astropy.coordinates import SkyCoord
pd.set_option("display.precision", 8)
import math
from matplotlib import cm
cmap = plt.get_cmap('plasma')  #('viridis')   #('jet_r')
import matplotlib.colors as colors
import matplotlib.cm as cmx
from statistics import stdev
#DRS calibrated data
#from mcdb import wutil
from importlib import reload as re
#hdu[3].data['EFF_WAVE']*1e6
import os
os.chdir('/Users/M51/matissepy/cdb/mcdb')
import pandas as pd
import wutilmod as wu  #calling upoints,vpoints,vis2=uvplane('Llong')  to make final uv plane plot

cmap = plt.get_cmap('viridis')

#uv plane with ... wl vis2
df=pd.read_pickle('/Users/M51/TablesPicklesScripts/Aug29NGC1068Sept22DRSv1.4.1') #2 : one II and one OO  HD 16658
#wu.visdrsagn(df, 3.4,3.6,0,1, avg=True)   #test
jet = cm = plt.get_cmap('plasma')      #('RdBu') 0x11b622278
j=1
c = cmap(float(j)/4)
print(c)
wu.visdrsagnmultexp(df, 'HD 16658', c, 'Lshort', 'lime', avg=True)
j=2
c = cmap(float(j)/4)
print(c)
wu.visdrsagnmultexp(df, 'HD 16658', c, 'Lband','lime', avg=True)
j=3
c = cmap(float(j)/4)
print(c)
wu.visdrsagnmultexp(df, 'HD 16658', c, 'Llong','lime',  avg=True)
j=4
c = cmap(float(j)/4)
print(c)
wu.visdrsagnmultexp(df, 'HD 16658', c, 'Mband', 'lime', avg=True)

jet = cm = plt.get_cmap('RdBu')
cm = plt.get_cmap('Greens')   #('Reds')      #('RdBu')
df=pd.read_pickle('/Users/M51/TablesPicklesScripts/Aug29NGC1068Sept25DRSv1.4.1')
#wu.visdrsagnmultexp(df, 'HD 29', 'red', 'Lshort', avg=True)  #test
j=1
c = cmap(float(j)/4)
wu.visdrsagnmultexp(df, 'HD 29920', c, 'Lshort', 'blue' ,avg=True)
j=2
c = cmap(float(j)/4)
wu.visdrsagnmultexp(df, 'HD ', c, 'Lband', 'blue', avg=True)
j=3
c = cmap(float(j)/4)
wu.visdrsagnmultexp(df, 'HD ', c, 'Llong', 'blue', avg=True)
j=4
c = cmap(float(j)/4)
wu.visdrsagnmultexp(df, 'HD ', c, 'Mband', 'blue', avg=True)
plt.legend()

go to #To plot all dates

sjump=0.15/450       #np.max(zavg)/600
gll = plt.scatter([],[], s=0.01/sjump, alpha=0.9, facecolors='none', edgecolors='black')
gl = plt.scatter([],[], s=0.05/sjump, alpha=0.9, facecolors='none', edgecolors='black')
ga = plt.scatter([],[], s=0.10/sjump, alpha=0.9, facecolors='none', edgecolors='black')
legend1=plt.legend((gll,gl,ga),
                   ('0.01', '0.05', '0.10'),
                   scatterpoints=1,
                   loc='lower left',
                   ncol=1,
                   fontsize=11, title='VIS2')
plt.gca().add_artist(legend1)


gll1 = plt.scatter([],[], s=10, marker='o', color='lime')
#gl1 = plt.scatter([],[], s=10, marker='o', color='blue')
ga1 = plt.scatter([],[], s=10, marker='o', color='blue')
plt.legend((gll1,ga1),
           ('22Sept', '25Sept'),
           scatterpoints=1,
           loc='upper right',
           ncol=1,
           fontsize=8, title='Dates')

plt.title('VIS2 on uv plane NGC 1068')


Dec52019

plotting spectra of ngc1068 cal by drs
from astropy.io import fits
hdu=fits.open('/Users/M51/Dropbox/MATISSEfiles/dataNGC1068paper/NGC1068CalibratedWithSelectedFiles/S22HD16658selected/TARGET_CAL_INT_0001.fits')
hdu.info()
hdu[7].columns
plt.plot(hdu[3].data['EFF_WAVE']*1e6,hdu[7].data['FLUXDATA'][0],label='Sept22,cal:HD 16658')
plt.legend()
#hdu=fits.open('/Users/M51/Dropbox/MATISSEfiles/dataNGC1068paper/NGC1068CalibratedWithSelectedFiles/S22delPhe/TARGET_CAL_INT_0001.fits')
#plt.plot(hdu[3].data['EFF_WAVE']*1e6,hdu[7].data['FLUXDATA'][0],label='Sept22,cal:delPhe')
#plt.legend()  #HAS SAME AS 16658!!! :(
hdu=fits.open('/Users/M51/Dropbox/MATISSEfiles/dataNGC1068paper/NGC1068CalibratedWithSelectedFiles/S24HD20356selected/TARGET_CAL_INT_0001.fits')
plt.plot(hdu[3].data['EFF_WAVE']*1e6,hdu[7].data['FLUXDATA'][0],label='Sept24,cal:HD 20356')
plt.legend()
hdu=fits.open('/Users/M51/Dropbox/MATISSEfiles/dataNGC1068paper/NGC1068CalibratedWithSelectedFiles/S25hd209240allgood/TARGET_CAL_INT_0001.fits')
plt.plot(hdu[3].data['EFF_WAVE']*1e6,hdu[7].data['FLUXDATA'][0],label='Sept25,cal:HD 209240')
plt.legend()
plt.title('NGC1068 calibrated OI_FLUX (after chopping test, selected data)')
plt.xlabel('wl [um]')
plt.ylabel('OI_FLUX [?]')








agn=['NGC 424',
'NGC 1068',
'NGC 3783',
'Circinus',
'NGC 5506',
'MCG-5-23-16',
'NGC 4507']

HXR=[21.51,
37.90,
173.84,
273.17,
239.40,
209.62,
184.54]

fr=[

IZwicky1 NGC 1365 (u) IRAS 05189-2524 H0557-385
IRAS 09149-6206 Mrk 1239 (u) NGC 3281 NGC 4151
3C273 (u) NGC 4593 ESO 323-77 (u) NGC 5128
IRAS 13349+2438 (u) IC4329A NGC 5995 NGC 7469


Sep 25
from astropy.io import fits
hdu=fits.open('/Volumes/MD/Reduced/Sept24NGC1068Nband/Iter1/mat_raw_estimates.2018-09-22T04_11_40.AQUARIUS.rb/TARGET_RAW_INT_0001.fits'#din LOW
              hdu['OI_VIS'].data['VISAMP'].shape #(24, 121
for i in range(24):
    plt.plot(hdu['OI_WAVELENGTH'].data['EFF_WAVE']*1e6,hdu['OI_VIS'].data['VISAMP'][i])

plt.clf()
              
              
              ???
              
April7 2021
1. How exactly are the calibrators selected. In the current text it
just says "nearby", "bright enough", and "small", but what were the
exact cuts you used in the selection? Here we guessed nearby was <10
deg and small is unresolved, but we need some help with the flux cuts.
In my notes of when we searched calibrators I wrote:
For the catalogue catalog2=fits.open("/Users/M51/msdfcc-v9.fits")(L and M bands):
#angdist max 10deg
#diammax 1 mas
#K star or G5 max.
#magnitude min16 max -4  should not matter because we are using a very small diameter
#Sort by ang dist, display 50, HA -5 to +5
#Lmag ~8 !!! or larger
#BD-02 473  Back up only! Prefer K stars, this is G5

for Cohen calibrators (Nband):
'HD10380' Nmag=1.2=13Jy 6to8times bright in Lband  vizier diam=2.77mas  <-------  1st
'HD10550' Nmag=1.98 =6.43J  UDDL=diam=2mas <------   2nd
'HD18322', 1.39Nmag     <------   3rd

For the catalogue of the Hybrid calibrators df=pd.read_pickle('/Users/M51/TablesPicklesScripts/PureHybridCalsNGC106815deg.pk') (L, M and N bands):
#Hybrid  we want: non-resolved in L/1068 is 1Jy at 8um  in N at 12 um it can reach 5 Jy
#Pick the closest
Thinking about it a bit more, maybe we can rather explain why we choose the calibrators in that way.
"Nearby" is to have an airmass comparable to the one of the target at the moment of the observation.
"small" is to not to depend on a measurement of the diameter of the star, which varies with wavelength and many times is not reliable when the star is very extended.
"Bright enough" means, if possible, with the same flux of the target, so any flux-dependant instrumental errors get "divided-out" when we do the calibration.

Sept11  Calibrators
#ESO wants guide * 20 deg from the moon
Best scenario: 424 / HybCalfor424 /THEN Pick up NGC1068 at 20deg / lowHYB / 1068 /
If not :
#-----------NGC 1068--------------
#FAINT  https://www.eso.org/observing/etc/bin/simu/calvin
ngc1068 02:45:03.962 00:25:33.61
#angdist max 10deg
#diammax 1 mas
#K star or G5 max.
#magnitude min16 max -4  should not matter because we are using a very small diameter
#Sort by ang dist, display 50, HA -5 to +5
#Lmag ~8 !!! or larger

#poscat2=pd.read_pickle('posmsdfcc.pk') # These pickles contain ra and decs for each catalog in degrees
#poscat1=pd.read_pickle('posjsdc_2017.pk')
#cat are poscat1 or poscat2, it only needs the columns with ra and dec
def search_cat(cat,rad,decd):  #send cat as data frame, ra,dec have to be in degrees
    catras = cat['ra']*15. #it comes in hours with decimal, we need them in degrees to compare directly
    catdecs = cat['de']
    eps = 5./3600 #5 arcseconds
    s = np.where((np.absolute(catras-rad)<eps) & (np.absolute(catdecs-decd)<eps))[0]
    if len(s) == 0:    #if it fails, try again
        eps = 10./3600  #more tolerance
        s = np.where( np.logical_and(np.absolute(catras-rad)<eps, np.absolute(catdecs-decd)<eps) )[0]  #(array([5]),)
    if len(s) == 0:    #if it fails, try again
        eps = 13./3600  #more tolerance
        s = np.where( np.logical_and(np.absolute(catras-rad)<eps, np.absolute(catdecs-decd)<eps) )[0]  #(array([5]),)
        if len(s) == 0:
            return(np.float('nan'))

    return(s[0]) #index

def search_cat2(cat,rad,decd,eps):  #send cat as data frame, ra and dec in degrees
    catras = cat['ra']*15. #it comes in hours with decimal, we need them in degrees to compare directly
    print(catras[1000])
    catdecs = cat['de']
    print(catdecs[1000])
    print(rad,decd,eps)
    s = np.where((np.absolute(catras-rad)<eps) & (np.absolute(catdecs-decd)<eps))[0]
    if len(s) == 0:    #if it fails
            print('No match around ',rad,' ',decd,' in a radius of ',eps*3600,' arcseconds')
            return(np.float('nan'))

    print(s)
    return(s) #array of indexes
    
    
#BD-00 423    <-------------------1st
from astropy.coordinates import SkyCoord
from astropy import units as u
import pandas as pd
from astropy.io import fits
import numpy as np
#from mcdb import wutilmod as wu
#poscat2=pd.read_pickle('/Users/M51/TablesPicklesScripts/posmsdfcc.pk') #in hexagesimal units!!!
poscat2=pd.read_pickle('/Users/M51/Correctionspaper2021/1068PaperOther/TablesPicklesScripts/posmsdfcc.pk') #in hexagesimal units!!!

catalog2=fits.open("/Users/M51/Correctionspaper2021/1068PaperOther/2018-2019/msdfcc-v9.fits")
catalog2[1].data.columns
cat=pd.DataFrame(catalog2[1].data)
ra='02:45:03.962' #NGC 1068 coordinates
de='00:25:33.610'
ct = SkyCoord(ra[0:12], de[0:12],unit=(u.hourangle, u.deg))
eps = 10./3600  #arcsec in deg
rat=ct.ra.degree
dect=ct.dec.degree
aindex = search_cat2(poscat2,rat,dect,eps)  #[374093]  #wu. if mcdb is working and uncomment
cat['Name'][374093]
Out[228]: 'BD-00   423                      '
cat['med_Lflux'][374093]
Out[232]: 0.09948508
cat['disp_Lflux'][374093]
Out[233]: 0.027268965
#BD-01 373   2nd  <-----------------2nd
ra='02:38:43.629'
de='-00:32:28.59'
ct = SkyCoord(ra[0:12], de[0:12],unit=(u.hourangle, u.deg))
eps = 15./3600  #arcsec in deg
rat=ct.ra.degree
dect=ct.dec.degree
aindex = wu.search_cat2(poscat2,rat,dect,eps)#343914
cat['Name'][343914]  #BD-01   373
cat['med_Lflux'][343914]  #0.15167382
cat['disp_Lflux'][343914]  #0.00057767244


#BD-02 473  Back up only! Prefer K stars, this is G5
ra='02:41:23.00'#'02:38:44.000'
de='-01:44:13.000'#'00:32:29.000'
ct = SkyCoord(ra[0:12], de[0:12],unit=(u.hourangle, u.deg))
eps = 10./3600  #arcsec in deg
rat=ct.ra.degree
dect=ct.dec.degree
aindex = wu.search_cat2(poscat2,rat,dect,eps)  #[256206
cat['Name'][256206]
Out[238]: 'BD-02   473                      '
cat['med_Lflux'][256206]
Out[239]: 0.16512977
cat['disp_Lflux'][256206]
Out[240]: 0.007279915
#Cohen  for N
df2=pd.read_pickle("/Users/M51/TablesPicklesScripts/cohencalsNGC1068.pk")
ngc1068 02 45 03.962 00 25 33.61
HD10380 01 41 25.89 +05 29 15.40
HD10824 01 45 59.260 -05 43 59.87
HD12076 01 58 27.67 -07 04 42.12
HD18322 02 56 25.64 -08 53 53.32
HD18700 03 00 44.14 +10 52 13.41
HD18884 03 02 16.77 +04 05 23.05
HD21120 03 24 48.79 +09 01 43.99
chararray(['HD10380' Nmag=1.2=13Jy 6to8times bright in Lband  vizier diam=2.77mas  <-------  1st
           , 'HD10550' Nmag=1.98 =6.43J  UDDL=diam=2mas <------   2nd
           'HD10824', 2Nmag
   NO    'HD11353', 1.2Nmag   Spectroscopic binary
           'HD12076',  2.21Nmag  UDDL=2.87mas
           'HD18322', 1.39Nmag     <------   3rd
           'HD18700',1.9mag
           'HD18884',
           'HD21120',1.6Nmag
   NO    'HD21754'],  1.7mag  Spectroscopic binary
ra='02:56:25.640'
de='-08:53:53.32'
ct = SkyCoord(ra[0:12], de[0:12],unit=(u.hourangle, u.deg))
eps = 10./3600  #arcsec in deg
rat=ct.ra.degree
dect=ct.dec.degree
aindex = wu.search_cat2(poscat2,rat,dect,eps)  #[403]
cat['Name'][403]  #same* eta Eri
cat['med_Lflux'][403]  86
cat['med_Nflux'][403]  13
#Hybrid  we want: non-resolved in L/1068 is 1Jy at 8um  in N at 12 um it can reach 5 Jy (for LMN bands)
#Pick the closest
df=pd.read_pickle('/Users/M51/TablesPicklesScripts/PureHybridCalsNGC106815deg.pk')
df['Name']
0    * nu. Cet    Lflux=22Jy dispL=0.2 Nflux=3.38J dispN=.96 UDDL=1.19mas
1    * rho02 Eri   NO  Double or multiple star
2    HD  19270
3    *  81 Cet   <----- same :)  1st!!  HD16400
4    HD  16247  Lflux=14.  Nflux=2.03Jy  <--------2nd
5    HD  20356
6    HD  13763  <--- good choice to be first than anything else Lflux=16 Nf=2.7
 7    HD  19723
#WE had chosen ESO calvis webpage
#HD19082
ra='03:04:06.188'
de='-05:14:35.95'
ct = SkyCoord(ra[0:12], de[0:12],unit=(u.hourangle, u.deg))
eps = 10./3600  #arcsec in deg
rat=ct.ra.degree
dect=ct.dec.degree
aindex = wu.search_cat2(poscat2,rat,dect,eps)  #16579
cat['Name'][16579]  #'HD  19082
cat['IRflag'][16579]   # 1  IR excess   NOT GOOD
cat['UDDL_est'][16579]  #1.138
cat['UDDN_est'][16579]   #1.148
#HD16400   = 81 cet
ra='02:37:41.801'
de='-03:23:46.226'
ct = SkyCoord(ra[0:12], de[0:12],unit=(u.hourangle, u.deg))
eps = 10./3600  #arcsec in deg
rat=ct.ra.degree
dect=ct.dec.degree
aindex = wu.search_cat2(poscat2,rat,dect,eps)        #   [2899]
cat['Name'][2899]  #81 Cet
cat['IRflag'][2899]   # 0  IR excess            <---------------- 1st
cat['UDDL_est'][2899]  #1.069
cat['UDDN_est'][2899]  #1.077
cat['med_Lflux'][2899] # 14 disp .25
cat['med_Nflux'][2899]    #2.14   disp  .62
#---------NGC 0424------------
#Cohen
chararray(['HD1632', 'HD3627', 'HD4502', 'HD11961'], dtype='<U16')

#Hybrid   PureHybridCalsNGC042415deg.pk
df['Name']    Lflux=  Nflux=
0    * chi Eri  71     9.6     <---    30minlate  ~same airmass
1    HD   6245  14     2.1   faintest in L
2    HD   1089  20     3.9
3    HD   3605  20     2.9   <----30 min earlier same airmass
4    HD   1923  39     5.6
ngc424  01:11:27.600   -38:05:00.00
chiEri  01:55:57.4721  -51:36:32.032
HD6245  01:02:49.1901  -46:23:50.318
HD1089  00:14:58.2620  -34:54:15.201
HD3605  00:38:48.7832  -25:06:28.144
HD1923  00:23:22.1450  -29:50:50.109
df=pd.read_pickle("/Users/M51/TablesPicklesScripts/PureHybridCalsNGC042415deg.pk")


Sept10
Re-calculated avg and std for clph. Now as complex nrs.avg
fitsii='/Users/M51/S22HD16658sel/TARGET_CAL_INT_0001.fits'
fitsoo='/Users/M51/S22HD16658sel/TARGET_CAL_INT_0002.fits'
fitsii='/Users/M51/S25hd20356/TARGET_CAL_INT_0001.fits'
fitsoo='/Users/M51/S25hd20356/TARGET_CAL_INT_0002.fits'

#modified to complex
mavgvis2,mstdvis2,mcalcp,mcalcpcomplex,mstdcp1,mstdcp2,avgbl,avgpa=wu.getavgstddrs(fitsii,fitsoo)

np.save('S22mavgvis2',mavgvis2)
np.save('S22mstdvis2',mstdvis2)
np.save('S22mcalcp',mcalcp)
np.save('S22mcalcpcomplex',mcalcpcomplex)
np.save('S22mstdcp1',mstdcp1)
np.save('S22avgbl',avgbl)
np.save('S22avgpa',avgpa)


np.save('S25mavgvis2',mavgvis2)
np.save('S25mstdvis2',mstdvis2)
np.save('S25mcalcp',mcalcp)
np.save('S25mcalcpcomplex',mcalcpcomplex)
np.save('S25mstdcp1',mstdcp1)
np.save('S25avgbl',avgbl)
np.save('S25avgpa',avgpa)

restore()


Sept 6
FINAL PLOTS for paper:  bcd calibration is performed here
1.-VIS2 S22
hdu=fits.open('/Users/M51/TARGET_CAL_INT_0001.fits') #S25
fitsii='/Users/M51/TARGET_CAL_INT_0002.fits'
fitsoo='/Users/M51/TARGET_CAL_INT_0001.fits'
mavgvis2ii,mstdvis2ii,mavgcpii,mstdcpii,avgblii,avgpaii=wu.getavgstddrs(fitsii)
[35 34]
[33 32]
[32 35]
[32 34]
[33 35]
[33 34]
[32 35 34]
[33 32 35]
[33 32 34]
[33 35 34]
mavgvis2oo,mstdvis2oo,mavgcpoo,mstdcpoo,avgbloo,avgpaoo=wu.getavgstddrs(fitsoo)
[34 35]
[32 33]
[33 34]
[33 35]
[32 34]
[32 35]
[33 34 35]
[32 33 34]
[32 33 35]
[32 34 35]
#Plotting vis2 and clph , calibrated by BCD
fitsii='/Volumes/MD/Reduced/Aug5DRSNGC1068SeptAfterChoppingTestv1.4.1/Iter1/CalibratedFiles/S25hd209240/TARGET_CAL_INT_0002.fits'
fitsoo='/Volumes/MD/Reduced/Aug5DRSNGC1068SeptAfterChoppingTestv1.4.1/Iter1/CalibratedFiles/S25hd209240/TARGET_CAL_INT_0001.fits'
#MODIFIED:
mstdvis2,mcalcp,mstdcp,avgbl,avgpa=wu.getavgstddrs(fitsii,fitsoo)

mavgvis2,mstdvis2,mavgcp,mstdcp=wu.getBCDavgstddrs(mavgvis2ii,mstdvis2ii,mavgcpii,mstdcpii, mavgvis2oo,mstdvis2oo,mavgcpoo,mstdcpoo)#NOT IN USE any more



#PLOTTING VIS2  w/error
from matplotlib import cm
#plasma
baseline=['UT3-UT4, '+str(round(avgbloo[0],0))+'m, '+str(round(avgpa[0],0))+'deg','UT1-UT2, '+str(round(avgbloo[1],0))+'m, '+str(round(avgpa[1],0))+'deg','UT2-UT3, '+str(round(avgbloo[2],0))+'m, '+str(round(avgpa[2],0))+'deg','UT2-UT4, '+str(round(avgbloo[3],0))+'m, '+str(round(avgpa[3],0))+'deg','UT1-UT3, '+str(round(avgbloo[4],0))+'m, '+str(round(avgpa[4],0))+'deg','UT1-UT4, '+str(round(avgbloo[5],0))+'m, '+str(round(avgpa[5],0))+'deg' ]#based on OO
plt.clf()
cmap = plt.get_cmap('nipy_spectral')
for bl in range(6):
    c = cmap(float(bl)/6)
    plt.plot(hdu[3].data['EFF_WAVE']*1e6,mavgvis2[bl],color=c,label=baseline[bl])
    y=np.array(mavgvis2[bl],dtype=float)
    sigma=np.array(mstdvis2[bl],dtype=float)
    plt.fill_between(hdu[3].data['EFF_WAVE']*1e6,y+sigma,y-sigma,color=c,alpha=0.5)

plt.ylim(0,0.15)
plt.legend()
plt.xlabel('wl [um]')
plt.ylabel('vis2')
#PLOTTING  CLOSURE PHASES  w/error
#viridis, rainbow, jet
triplet=["UT2-UT3-UT4","UT1-UT2-UT3","UT1-UT2-UT4","UT1-UT3-UT4",] #based on OO
plt.clf()
cmap = plt.get_cmap('viridis')
for tr in range(4):
    c = cmap(float(tr)/4)
    plt.plot(hdu[3].data['EFF_WAVE']*1e6, mcalcp[tr][:],color=c,label=triplet[tr])
    y=np.array(mcalcp[tr][:],dtype=float)
    sigma=np.array(mstdcp1[tr],dtype=float)
    plt.fill_between(hdu[3].data['EFF_WAVE']*1e6,y+sigma,y-sigma,color=c,alpha=0.5)

plt.title('Calibrated NGC 1068 Sept. 25')     #CHANGE
plt.ylim(-182,182)
plt.legend()
plt.xlabel('wavelength [um]')
plt.ylabel('closure phase [deg]')







df=pd.read_pickle('Aug29NGC1068Sept25DRSv1.4.1') #2  HD 209240
#taking avgs and stds.
band='Lshort' #3.25 to 3.45
band='Lband'
band='Llong'
band='Mband'
plt.clf()
df=pd.read_pickle('/Users/M51/TablesPicklesScripts/Aug29NGC1068Sept22DRSv1.4.1') #2 : one II and one OO  HD 16658
wu.visdrsagnmultexp(df, 'HD 16658', 'lightgreen', 'Lshort', avg=True)

1.- Table of selected observations of the calibrators







Aug 29  add april cals  PENDING!
Erase badmjds rows from 3 cals didnt work
re-reduce cals using only selected mjds and re-calibrate 1068  DONE
*Ask Jacob to calibrate BCDs!!!!   DONE
new tables for NGC1068:
dir='/Volumes/MD/Reduced/Aug5DRSNGC1068SeptAfterChoppingTestv1.4.1/Iter1/CalibratedFiles/S22HD16658sel'
df=wu.CalDatatableDRS('/Volumes/MD/Reduced/Aug5DRSNGC1068SeptAfterChoppingTestv1.4.1/Iter1/CalibratedFiles/S22HD16658sel')
df.to_pickle('Aug29NGC1068Sept22DRSv1.4.1')
dir='/Volumes/MD/Reduced/Aug5DRSNGC1068SeptAfterChoppingTestv1.4.1/Iter1/CalibratedFiles/S24HD20356sel'
df=wu.CalDatatableDRS(dir)
df.to_pickle('Aug29NGC1068Sept24DRSv1.4.1')
df=wu.CalDatatableDRS('/Volumes/MD/Reduced/Aug5DRSNGC1068SeptAfterChoppingTestv1.4.1/Iter1/CalibratedFiles/s25hd20356')
df.to_pickle('Aug29NGC1068Sept25DRSv1.4.1')
#plotting uv plane for each date  DONE
FALTA CALIBRAR BCDs
#needs to be fixed
df=pd.read_pickle('Aug29NGC1068Sept22DRSv1.4.1') #2 : one II and one OO  HD 16658
df=pd.read_pickle('Aug29NGC1068Sept24DRSv1.4.1') #2  HD 20356
df=pd.read_pickle('Aug29NGC1068Sept25DRSv1.4.1') #2  HD 209240
wu.visdrsagnmultexp(df,3.4,3.5,0,1,'HD 16658')#CHANGEcal!!!  checck wl
wu.visdrsagnmultexp(df,3.4,3.5,0,1,'HD 20356')#CHANGEcal!!!  checck wl
wu.visdrsagnmultexp(df,3.4,3.5,0,1,'HD 209240')#CHANGEcal!!!  checck wl
df=pd.read_pickle('ngc1068DRSv1.4.1Sel.pk') #18,75  DOESNT WORK
wu.visdrsagnmultexp(df,3.4,3.5,0,2,'HD 16658, HD 20356 and HD 209240','red')

#To plot all dates ********************************************
df=pd.read_pickle('/Users/M51/TablesPicklesScripts/Aug29NGC1068Sept25DRSv1.4.1')
              #wu.visdrsagnmultexp(df, 'HD 29', 'red', 'Lshort', avg=True)  #test
j=1
c = cmap(float(j)/4)
wu.visdrsagnmultexp(df, 'HD 29920', c, 'Lshort', 'blue' ,avg=True)
band='Lshort'
band='Lband'
band='Llong'
band='Mband'
plt.clf()
df=pd.read_pickle('/Users/M51/TablesPicklesScripts/Aug29NGC1068Sept22DRSv1.4.1') #2 : one II and one OO  HD 16658
wu.visdrsagnmultexp(df,3.4,3.5,0,2, 'HD 16658, HD 20356 and HD 209240','lightgreen', band, avg=True)
              #last vs fn
wu.visdrsagnmultexp(df, 'HD 16658, HD 20356 and HD 209240','lightgreen', band, 'black', avg=True)
df=pd.read_pickle('/Users/M51/TablesPicklesScripts/Aug29NGC1068Sept24DRSv1.4.1') #2  HD 20356
wu.visdrsagnmultexp(df,3.4,3.5,0,2, 'HD 16658, HD 20356 and HD 209240','blue', band, avg=True)
wu.visdrsagnmultexp(df, 'HD 16658, HD 20356 and HD 209240','blue', band, 'blue', avg=True)
df=pd.read_pickle('/Users/M51/TablesPicklesScripts/Aug29NGC1068Sept25DRSv1.4.1') #2  HD 209240
wu.visdrsagnmultexp(df,3.4,3.5,0,2, 'HD 16658, HD 20356 and HD 209240','red', band,  avg=True)

sjump=0.3/600       #np.max(zavg)/600
gll = plt.scatter([],[], s=0.01/sjump, alpha=0.9, facecolors='none', edgecolors='black')
gl = plt.scatter([],[], s=0.05/sjump, alpha=0.9, facecolors='none', edgecolors='black')
ga = plt.scatter([],[], s=0.15/sjump, alpha=0.9, facecolors='none', edgecolors='black')
legend1=plt.legend((gll,gl,ga),
           ('0.01', '0.05', '0.15'),
           scatterpoints=1,
           loc='lower left',
           ncol=1,
           fontsize=8, title='VIS2')
plt.gca().add_artist(legend1)


gll1 = plt.scatter([],[], s=10, marker='o', color='lightgreen')
gl1 = plt.scatter([],[], s=10, marker='o', color='blue')
ga1 = plt.scatter([],[], s=10, marker='o', color='red')
plt.legend((gll1,gl1,ga1),
           ('22Sept', '24Sept', '25Sept'),
           scatterpoints=1,
           loc='upper right',
           ncol=1,
           fontsize=8, title='Dates')

df=pd.read_pickle('Aug29NGC1068Sept22DRSv1.4.1') #2 : one II and one OO  HD 16658
outch=wu.plotvis2(df,'table','T','F',calibrator='HD 16658')      #.17
df=pd.read_pickle('Aug29NGC1068Sept24DRSv1.4.1') #2  HD 20356
outch=wu.plotvis2(df,'table','T','F',calibrator='HD 20356')      #.32
df=pd.read_pickle('Aug29NGC1068Sept25DRSv1.4.1') #2  HD 209240
outch=wu.plotvis2(df,'table','T','F',calibrator='HD 209240')     #.11
outch=wu.plotvis2(df,'table','T','F',calibrator='HD 20356 and HD 209240')      #.32
wu.plotclph(df,'table','T','F','jet',calibrator='HD 16658')  #FIX!!!

plt.clf()
file='/Volumes/MD/Reduced/Aug5DRSNGC1068SeptAfterChoppingTestv1.4.1/Iter1/CalibratedFiles/S22HD16658sel/TARGET_CAL_INT_0001.fits'
wu.plotclph(file,'fits','T','F','jet',calibrator='HD 16658')
plt.clf()
file='/Volumes/MD/Reduced/Aug5DRSNGC1068SeptAfterChoppingTestv1.4.1/Iter1/CalibratedFiles/S22HD16658sel/TARGET_CAL_INT_0002.fits'
wu.plotclph(file,'fits','T','F','jet',calibrator='HD 16658')
plt.clf()
file='/Volumes/MD/Reduced/Aug5DRSNGC1068SeptAfterChoppingTestv1.4.1/Iter1/CalibratedFiles/S24HD20356sel/TARGET_CAL_INT_0001.fits'
wu.plotclph(file,'fits','T','F','jet',calibrator='HD 20356')#OO PROBLEM!!!
plt.clf()
file='/Volumes/MD/Reduced/Aug5DRSNGC1068SeptAfterChoppingTestv1.4.1/Iter1/CalibratedFiles/S24HD20356sel/TARGET_CAL_INT_0002.fits'
wu.plotclph(file,'fits','T','F','jet',calibrator='HD 20356')
plt.clf()
file='/Volumes/MD/Reduced/Aug5DRSNGC1068SeptAfterChoppingTestv1.4.1/Iter1/CalibratedFiles/S25hd209240/TARGET_CAL_INT_0001.fits'
wu.plotclph(file,'fits','T','F','jet',calibrator='HD 209240')
plt.clf()
file='/Volumes/MD/Reduced/Aug5DRSNGC1068SeptAfterChoppingTestv1.4.1/Iter1/CalibratedFiles/S25hd209240/TARGET_CAL_INT_0002.fits'
wu.plotclph(file,'fits','T','F','jet',calibrator='HD 209240')







wu.visewsagn(DF1068,67,91,viidp22,voodp22,0,9) #To plot visibilities on the uv plane, color code is the visibility,  10 is chopped!!
#CHECK right wls to average in,
#3.21 to 3.81:
wu.visewsagn(DF1068,67,91,viidp22,voodp22,1,9,avg='True')  #To plot visibilities on the uv plane, color code is the visibility,  USING AVGS  0 is in red in table
for i in range(df.shape[0]):
    for j in range(df.shape[1]):
        plt.plot(df['WL'][i],df['VIS2'][i][j])
plt.ylim(0,0.5)

#trying to change fits file to delete bad mjds.   DIDNT WORK
amjds50=wu.getbadmjds('/Users/M51/picklesews',0.5)#april may and sept
len(amjds50) #221
amjds50=np.array(amjds50T)
amjds50=np.sort(amjds50)
hdu=fits.open('/Users/M51/modifiedfitsforcal/HD16658/CALIB_RAW_INT_0001.fits') #OO
In [38]: hdu[4].data['mjd']
58383.20997588  in list of mjds
58383.21096678
58383.21195812
58383.21295051
array([58383.21039851, 58383.21039851, 58383.21039851, 58383.21039851,
       58383.21039851, 58383.21039851,
       
       58383.21139703, 58383.21139703,
       58383.21139703, 58383.21139703, 58383.21139703, 58383.21139703,
       
       58383.21239233, 58383.21239233, 58383.21239233, 58383.21239233,
       58383.21239233, 58383.21239233,
       
       58383.21337817, 58383.21337817,
       58383.21337817, 58383.21337817, 58383.21337817, 58383.21337817])

CALIB_CAL  1-3 In In  / 4-7Out Out
58383.20997672444
rows to delete: 0,3
hdu=fits.open('/Users/M51/modifiedfitsforcal/HD16658/CALIB_RAW_INT_0001.fits') #OO
type(hdu)  #astropy.io.fits.hdu.hdulist.HDUList
aind=[0,0,0,0,0,0]

#OI_VIS2  4
hdu[4].data.shape
for i in aind:
    hdu[4].data=np.delete(hdu[4].data,i)  #23
    hdu.writeto('/Users/M51/modifiedfitsforcal/HD16658/CALIB_RAW_INT_0001.fits',overwrite=True)

hdu[4].data.shape

newdata=np.delete(newdata,2)  #22 :) row3 moved to row2 after del row0
len(newdata)  #22
len(out4data)  #24
out4data=newdata  #24
#out=fits.HDUList([hdu.copy()])  PENDING
hdu.writeto('/Users/M51/modifiedfitsforcal/HD16658/modCALIB_RAW_INT_0001.fits')
hdutest=fits.open('/Users/M51/modifiedfitsforcal/HD16658/modCALIB_RAW_INT_0001.fits')

ValueError: could not broadcast input array from shape (18,110) into shape (12,110)
#OI_T3  5
hdu[5].data=np.delete(hdu[5].data,0)

out5data=out[5].data  #24
newdata=np.delete(out5data,0)  #23
newdata=np.delete(newdata,2)  #22 :) row3 moved to row2 after del row0
len(newdata)  #22
len(out5data)
#OI_VIS 6
#TF2  7

with fits.open('/Users/M51/modifiedfitsforcal/HD16658/CALIB_RAW_INT_0001.fits', mode='update') as hdul:
    out4data=hdul[4].data  #24
    newdata=np.delete(out4data,0)  #23
    newdata=np.delete(newdata,2)  #22 :) row3 moved to row2 after del row0
    len(newdata)  #22
    len(out4data)  #24
    out4data=newdata
    # Change something in hdul.
    hdul.flush()  # changes are written back to original.fits   PENDING

# closing the file will also flush any changes and prevent further writing
new=wu.removemjdinfits('/Users/M51/modifiedfitsforcal/HD16658/CALIB_RAW_INT_0001.fits',rownr)

Aug 28
amjds50T=wum.getbadmjds('/Users/M51/picklesews/NGC1068',0.5)#28->18
amjds25T=wum.getbadmjds('/Users/M51/picklesews/NGC1068',0.25)#28->23
adir=np.array(["/Volumes/MD/Reduced/Aug5DRSNGC1068SeptAfterChoppingTestv1.4.1/Iter1"])
df1=wum.tableDRSCalRIVIS2(adir)
#df1.to_pickle('ngc1068DRSv1.4.1.pk')#ch
for i in range(df1.shape[0]):
    temp=df1['mjdo'][i] in amjds25T        #CHANGE!!!
    if(temp==True): #not in list of bad mjds
        df1=df1.drop(i)
        print(df1.shape[0])  #28->18

#df1.to_pickle('ngc1068DRSv1.4.1Sel.pk')#ch and nonch

dfc=pd.read_pickle('/Users/M51/TablesPicklesScripts/ngc1068DRSv1.4.1.pk')#ch/ not using selected
dfnc=pd.read_pickle('/Users/M51/TablesPicklesScripts/DRSNGC1068v1.4.1noCT.pk') PENDING

arrayname2, arraymedL2, arraymedLNC2 ,arraymedM2,  arraymedMNC2, arraystdL2,arraystdLNC2,arraystdM2,arraystdMNC2 =wu.plotrawvis2sel(amjds50T,dfc,dfnc)  #doesnt work for a target since there is no diameter

visdrsagn(df,3.4,3.6,i1,i2,avg=None):  PENDING
wu.visdrsagn(df,3.4,3.5,0,1)



from astropy.io import fits
hdu=fits.open('/Users/M51/CALIB_RAW_INT_0001.fits')
hdu['OI_VIS2'].data.columns
len(hdu['OI_VIS2'].data)
 hdu['OI_VIS2'].data['MJD']
new=np.delete(hdu['OI_VIS2'].data,0,axis=0)
len(new)  #29
new['OI_VIS2'].data
new['OI_VIS2'].data['MJD']
#double checked with 16658 Good :)
t=wu.mrestore('2018-09-22T04:44:55.tpl.pk')
for tt in t: print(tt['chopped'],np.around(tt['fflag'],3),tt['mjd-obs'])
False [0.568 0.45  0.635 0.62  0.432 0.45 ] 58383.20465691
False [0.635 0.38  0.594 0.535 0.325 0.365] 58383.2037238
False [0.432 0.391 0.31  0.314 0.351 0.288] 58383.19973116
False [0.535 0.679 0.491 0.52  0.616 0.627] 58383.20280817
False [0.59  0.435 0.642 0.421 0.421 0.424] 58383.20092859
False [0.705 0.542 0.561 0.583 0.579 0.576] 58383.20187058
True [0.625 0.735 0.676 0.647 0.625 0.581] 58383.21195812  <--
True [0.741 0.807 0.667 0.719 0.793 0.696] 58383.20886846  <--
True [0.733 0.8   0.8   0.704 0.704 0.593] 58383.21096678  <--
True [0.533 0.644 0.607 0.615 0.519 0.481] 58383.20787761
True [0.422 0.563 0.504 0.496 0.407 0.585] 58383.20997588
True [0.588 0.743 0.522 0.485 0.485 0.574] 58383.21295051
True [0.397 0.324 0.316 0.265 0.228 0.287] 58383.20688422
True [0.412 0.691 0.463 0.478 0.382 0.39 ] 58383.20571736
PENDING:
wu.TFvis2tauDRS(df,'LFlux','T0E','VIS2',0,135,1,2,0,correction,0,'scatter')
#2D W
wu.TFvis2tauDRS(df,'LFlux','T0E','VIS2',0,135,1,2,0,correction,0,None)
#color,x,y  L
wu.TFvis2tauDRS(df,'LFlux','T0E','VIS2',0,135,1,2,0,correction,0,'scatter2')


Aug 27
used tol=50% to make df
dfnc=  #first may
wum.getbadmjds()
#may?
dfnc1=pd.read_pickle('/Users/M51/TablesPicklesScripts/mayCalsDRSv1.4.1.pk')
#Sept
dfnc2=pd.read_pickle('/Users/M51/TablesPicklesScripts/septCalsDRSv1.4.1.pk')
arrayname1, arraymedL1, arraymedLNC1, arraymedM1, arraymedMNC1,arraystdL1,arraystdLNC1,arraystdM1,arraystdMNC1 =wum.plotrawvis2sel(amjds50,df,dfnc1)
dfnc3=pd.read_pickle('/Users/M51/TablesPicklesScripts/aprCalsDRSv1.4.1.pk')  PENDING
dfc1=pd.read_pickle('/Users/M51/TablesPicklesScripts/ngc1068DRSv1.4.1.pk')

arrayname2, arraymedL2, arraymedLNC2 ,arraymedM2,  arraymedMNC2, arraystdL2,arraystdLNC2,arraystdM2,arraystdMNC2 =wum.plotrawvis2sel(amjds50,df,dfnc2)
x = np.linspace(0,0.6)
plt.plot(x,x,color='black')
plt.errorbar(arraymedL1,arraymedLNC1,xerr=arraystdL1,yerr=arraystdLNC1,fmt='o',ecolor='orange',capthick=2)
plt.errorbar(arraymedL2,arraymedLNC2,xerr=arraystdL2,yerr=arraystdLNC2,fmt='o',ecolor='orange',capthick=2)
for i in range(len(arrayname1)):
    plt.text(arraymedL1[i],arraymedLNC1[i],arrayname1[i],size=7)
for i in range(len(arrayname2)):
    plt.text(arraymedL2[i],arraymedLNC2[i],arrayname2[i],size=7)
for i in range(len(arrayname1)):
    if(arraymedMNC1[i]<=1 and arraymedM1[i]<=1):
        print(arrayname1[i])
        plt.errorbar(arraymedM1[i], arraymedMNC1[i] ,xerr=arraystdM1[i] ,yerr=arraystdMNC1[i] ,fmt='o',ecolor='g',capthick=2)
        plt.text(arraymedM1[i],arraymedMNC1[i],arrayname1[i],size=7)
        print(arraystdM1[i]-arraystdMNC1[i])
for i in range(len(arrayname2)):
    if(arraymedMNC2[i]<=1 and arraymedM2[i]<=1):
        print(arrayname2[i])
        plt.errorbar(arraymedM2[i], arraymedMNC2[i] ,xerr=arraystdM2[i] ,yerr=arraystdMNC2[i] ,fmt='o',ecolor='g',capthick=2)
        plt.text(arraymedM2[i],arraymedMNC2[i],arrayname2[i],size=7)
        print(arraystdM2[i]-arraystdMNC2[i])

plt.xlim(0,0.6)
plt.ylim(0,0.6)
plt.grid(linestyle="--",linewidth=0.1,color='.25')
plt.title('raw VIS2 L(3.5+-0.1um,orange) & M(4.75+-0.1um,green) bands chopped vs non-chopped')
plt.xlabel('Raw VIS2 chopped')
plt.ylabel('Raw VIS2 non-chopped')

#for sel mjds
for i in range(18): print(arrayname1[i],np.around(arraymedL1[i],2),np.around(arraystdL1[i],2))
HD138492 0.26 0.08
50_Vir 0.34 0.08
37_Lib 0.24 0.08
HR_6778 0.18 0.04
HD 89998 0.28 0.07
UCAC2 3446325 0.27 0.06
HD 101666 0.25 0.08
UCAC2 5719521 0.28 0.06
HD184996 0.3 0.09
UCAC2  29197485 0.22 0.1
UCAC2 5719521 0.29 0.08
HD 101666 0.25 0.08
UCAC2 14406654 0.32 0.07
UCAC2 14406654 0.32 0.07
UCAC2  29197485 0.28 0.07
UCAC2 5719521 0.33 0.08
HD184996 0.29 0.08
for i in range(len(arrayname2)): print(arrayname2[i],np.around(arraymedL2[i],2),np.around(arraystdL2[i],2))
HD 183925 0.33 0.08
HD 6290 0.32 0.09
84 Aqr 0.36 0.1
del Phe 0.33 0.15
HD 16658 0.36 0.08
HD 195677 0.23 0.11
sig Cap 0.23 0.08
sig Cap 0.28 0.08
HD 20356 0.25 0.11
HD 172453 0.32 0.07
V4026 Sgr 0.24 0.09
HD209240 0.36 0.08
47 Cap 0.2 0.1
HD 20356 0.27 0.09
HD33162 0.18 0.08

for i in range(len(arrayname1)): print(arraystdL1[i]-arraystdLNC1[i])
differences in L band stds of vis2
HD138492 0.006136533945775263
50_Vir -0.002169175273309204
37_Lib 0.016304530019998695
HR_6778 3.552371124385112e-06
HD 89998 -0.0053125938035329945
UCAC2 3446325 -0.012532718899401558
HD 101666 0.002207373085957212
UCAC2 5719521 -0.011455632200830151
HD184996 0.0018043011368679834
UCAC2  29197485 -0.001146715981906063
UCAC2 5719521 0.010101348411930972
HD 101666 -0.0032675516954199513
UCAC2 14406654 -0.00707816056046956
UCAC2 14406654 -0.00638430764105645
UCAC2  29197485 -0.06158863926322093
UCAC2 5719521 -0.002173495741025641
HD184996 -0.0053682455424109154
UCAC2  29197485 -0.046430184088342444
differences in M band stds of vis2
HD138492 nan
50_Vir nan
37_Lib nan
HR_6778 0.006418261226917403
HD 89998 0.007201516194717719
UCAC2 3446325 0.528760152243173
HD 101666 0.011723469212927953
UCAC2 5719521 0.060704005432940744
HD184996 0.0736110648711026
UCAC2  29197485 -53.88639889131986
UCAC2 5719521 -0.2522987619581627
HD 101666 0.0032041042517196167
UCAC2 14406654 -0.26864907318231757
UCAC2 14406654 -0.12584614660874527
UCAC2  29197485 4.619247174980353
UCAC2 5719521 -0.9653769134351691
HD184996 0.008120582344812652
UCAC2  29197485 -0.5044096706104938

differences in L band stds of vis2
HD 183925 -0.0005759401122673036
HD 6290 0.01573793911861969
84 Aqr 0.007395818298759599
del Phe 0.052018442526997555  <---
HD 16658 0.003190446574381417
HD 195677 -0.007637830520312319
sig Cap 0.006088263833129368
sig Cap -0.324526467238094   <---
HD 20356 0.0017966939325204656
HD 172453 0.0031748470607071005
V4026 Sgr -0.007752683877010988
HD209240 0.00029924181169935227
47 Cap 0.017389649853523703
HD 20356 -0.0007956232379002365
HD33162 -0.0009770311253133879
differences in M band stds of vis2
HD 183925 0.0036072139794901675
HD 6290 0.011003969580048123
84 Aqr -0.011847815230070446
del Phe 0.1238089361410699 <---  2206
HD 16658 -2.716720775014652
HD 195677 0.029625094402569455
sig Cap 0.01155011247836435
sig Cap -0.35603242105302074  <---
HD 20356 0.00889184906633221
HD 172453 -0.017094541841275096
V4026 Sgr 0.011685655540495596
HD209240 0.004323213833001341
47 Cap 0.027890083329781448
HD 20356 -0.00035129243844202185
HD33162 0.012447184469600026

HIERARCH ESO ISS IAS IRIS DIT = 0.010 / IRIS Detector Integration Time [s].
HIERARCH ESO ISS IAS IRIS_ERRSTDFLUX1 = 594.724 / Std deviation Flux (ADU).
HIERARCH ESO ISS IAS IRIS_ERRSTDFLUX2 = 491.187 / Std deviation Flux (ADU).
HIERARCH ESO ISS IAS IRIS_ERRSTDFLUX3 = 325.709 / Std deviation Flux (ADU).
HIERARCH ESO ISS IAS IRIS_ERRSTDFLUX4 = 449.355 / Std deviation Flux (ADU).

#
HIERARCH ESO ISS IAS IRIS_ERRSTDX1 = 0.568 / Std deviation X.  error detector in pixels, above 1 not ok


HIERARCH ESO ISS IAS IRIS_ERRSTDX2 = 0.718 / Std deviation X.
HIERARCH ESO ISS IAS IRIS_ERRSTDX3 = 0.676 / Std deviation X.
HIERARCH ESO ISS IAS IRIS_ERRSTDX4 = 0.454 / Std deviation X.
HIERARCH ESO ISS IAS IRIS_ERRSTDY1 = 0.671 / Std deviation Y.
HIERARCH ESO ISS IAS IRIS_ERRSTDY2 = 0.723 / Std deviation Y.
HIERARCH ESO ISS IAS IRIS_ERRSTDY3 = 0.846 / Std deviation Y.
HIERARCH ESO ISS IAS IRIS_ERRSTDY4 = 0.538 / Std deviation Y.
HIERARCH ESO ISS IAS IRIS_GUID = 'ON      ' / IRIS labguiding state (ON, OFF).


#Plots TF cals
Aug 26 units of clph?
              df=pd.read_pickle('marchCalsDRSv1.4.1.pk')#no UTs
              df=pd.read_pickle('aprilCalsDRSv1.4.1.pk')#no UTs   <---faint ATs
              df=pd.read_pickle('mayCalsDRSv1.4.1.pk')  #lots
              df=pd.read_pickle('juneCalsDRSv1.4.1.pk') #few UTs
              df=pd.read_pickle('SeptCalsDRSv1.4.1.pk') #few UTs
#PLOT FROM HERE Plots cals  tables made in: Aug15 df=wum.tableDRSCalRIVIS2
from mcdb import wutilmod as wu
#amjds70=wum.getMjds('/Users/M51/picklesews',0.70) #len(amjds70) NOT FOR PAPER!
#Out[71]: 382  #more bad ones with a more strict criteria
amjds50=wu.getbadmjds('/Users/M51/picklesews',0.50) #contains may, april and sept data #FINAL FOR PAPER!! :) AS WE DID SAME WITH NGC 1068
len(amjds50) #221
df1=pd.read_pickle('/Users/M51/TablesPicklesScripts/amjCalsCTDRSv1.4.1.pk')#178 created Aug 17
df2=pd.read_pickle('/Users/M51/TablesPicklesScripts/septCalsCTDRSv1.4.1.pk')#104
np.sum(df2['name']=='del Phe') #3  #Why we could not filter out these 2?
np.sum(df1['name']=='UCAC2  29197485')#24
for i in range(df1.shape[0]):
    temp=df1['mjdo'][i] in amjds50
    temp4=df1['name'][i] == 'UCAC2  29197485'  #there are 24
    if(temp==True): #not in list of bad mjds
        df1=df1.drop(i)  #108
        print(df1.shape[0])
    elif(temp4==True): #not in list of bad mjds
        df1=df1.drop(i)  #100
        print(df1.shape[0])

for i in range(df2.shape[0]):
    temp2=df2['mjdo'][i] in amjds50
    temp3=df2['name'][i] == 'del Phe'  #there are 3
    #print(temp2)
    if(temp2==True):
        df2=df2.drop(i)#78 ams    104
        print(df2.shape[0])
    elif(temp3==True): #not in list of bad mjds
        df2=df2.drop(i)  #75
        print(df2.shape[0])

df=pd.concat([df1,df2], ignore_index=True)#183 no delPhe/  186 ams     212  re-enumerates
np.shape(df)#175
#just checking, running for df
for index, row in df2.iterrows():
    temp=df2['mjdo'][index] in amjds50
    if(temp==True): #not in list of bad mjds
        print('bad mjd still in df')
for i in range(df.shape[0]):
    temp=df['mjdo'][i] in amjds50
    #if(df['mjdo'][i]=='58383.15932782'):#testing a good one
    if(df['mjdo'][i]=='58383.16131374'):#testing a bad one
    #if(temp==True): #not in list of bad mjds
        print('bad mjd still in df')

np.sum(df['LFlux']==0.0)#43  <0.1)
anames=[]
for i in range(np.shape(df)[0]):
    if(df['LFlux'][i]==0.0):
        anames.append(df['name'][i])
print(np.unique(anames))#['UCAC2  29197485 Gone' 'UCAC2 14406654' 'UCAC2 3446325' 'UCAC2 5719521']

for i in range(np.shape(df)[0]):
    for j in range(6):
        if(df['VIS2'][i][j][35]<0.15):#for 3.5um
            print(df['name'][i])
#*re-did plots of Aug23? using last df
#LEAVE BLANK WINDOW OPEN TO GET LARGE SIZE FIGURE
from importlib import reload as re
list(df.columns)
#OO ch0 3.35um UCAC2  29197485 vis2<0.2

#df=pd.concat([df1,df2], ignore_index=True)#
#df.to_pickle('MaySeptCalsCTselmjds.pk')  #amj + sept
#df.shape#(282, 75)
#Q: loss of vis2 w/flux in L??  Make TF plots
statvis2Ls=wu.errorBarsByTPL2(df,data='VIS2',X='LFlux',BCD1='OUT',BCD2='OUT', Title='No', band='Lshort') #For paper
statvis2Ls=wu.errorBarsByTPL2(df,data='T3PHI',X='LFlux',BCD1='OUT',BCD2='OUT', Title='No', band='Lshort') #For paper

statvis2Ls=wu.errorBarsByTPL2(df,data='VIS2',X='LFlux',BCD1='OUT',BCD2='OUT', Title='No', band='Lshort')
statvis2Ls=wu.errorBarsByTPL2(df,data='VIS2',X='LFlux',BCD1='OUT',BCD2='OUT', Title='Yes', band='Lshort')
statvis2Lm=wu.errorBarsByTPL2(df,data='VIS2',X='LFlux',BCD1='OUT',BCD2='OUT', Title='Yes', band='Lband')
statvis2Ll=wu.errorBarsByTPL2(df,data='VIS2',X='LFlux',BCD1='OUT',BCD2='OUT', Title='Yes', band='Llong')
statvis2Mm=wu.errorBarsByTPL2(df,data='VIS2',X='LFlux',BCD1='OUT',BCD2='OUT', Title='Yes', band='Mband')

statvis2Ls=wu.errorBarsByTPL2(df,data='VIS2',X='tau0' ,BCD1='OUT',BCD2='OUT', Title='Yes', band='Lshort')
statvis2Lm=wu.errorBarsByTPL2(df,data='VIS2',X='tau0' ,BCD1='OUT',BCD2='OUT', Title='Yes', band='Lband')
statvis2Ll=wu.errorBarsByTPL2(df,data='VIS2',X='tau0' ,BCD1='OUT',BCD2='OUT', Title='Yes', band='Llong')
statvis2Mm=wu.errorBarsByTPL2(df,data='VIS2',X='tau0' ,BCD1='OUT',BCD2='OUT', Title='Yes', band='Mband')

statvis2Ls=wu.errorBarsByTPL2(df,data='VIS2',X='mjd'  ,BCD1='OUT',BCD2='OUT', Title='Yes', band='Lshort')
statvis2Lm=wu.errorBarsByTPL2(df,data='VIS2',X='mjd'  ,BCD1='OUT',BCD2='OUT', Title='Yes', band='Lband')
statvis2Ll=wu.errorBarsByTPL2(df,data='VIS2',X='mjd'  ,BCD1='OUT',BCD2='OUT', Title='Yes', band='Llong')
statvis2Mm=wu.errorBarsByTPL2(df,data='VIS2',X='mjd'  ,BCD1='OUT',BCD2='OUT', Title='Yes', band='Mband')

statvis2Ls=wu.errorBarsByTPL2(df,data='VIS2',X='LFlux',BCD1='IN',BCD2='IN'  , Title='Yes', band='Lshort')
statvis2Lm=wu.errorBarsByTPL2(df,data='VIS2',X='LFlux',BCD1='IN',BCD2='IN'  , Title='Yes', band='Lband')
statvis2Ll=wu.errorBarsByTPL2(df,data='VIS2',X='LFlux',BCD1='IN',BCD2='IN'  , Title='Yes', band='Llong')
statvis2Mm=wu.errorBarsByTPL2(df,data='VIS2',X='LFlux',BCD1='IN',BCD2='IN'  , Title='Yes', band='Mband')

statvis2Ls=wu.errorBarsByTPL2(df,data='VIS2',X='tau0' ,BCD1='IN',BCD2='IN'  , Title='Yes', band='Lshort')
statvis2Lm=wu.errorBarsByTPL2(df,data='VIS2',X='tau0' ,BCD1='IN',BCD2='IN'  , Title='Yes', band='Lband')
statvis2Ll=wu.errorBarsByTPL2(df,data='VIS2',X='tau0' ,BCD1='IN',BCD2='IN'  , Title='Yes', band='Llong')
statvis2Mm=wu.errorBarsByTPL2(df,data='VIS2',X='tau0' ,BCD1='IN',BCD2='IN'  , Title='Yes', band='Mband')

statvis2Ls=wu.errorBarsByTPL2(df,data='VIS2',X='mjd'  ,BCD1='IN',BCD2='IN'  , Title='Yes', band='Lshort')
statvis2Lm=wu.errorBarsByTPL2(df,data='VIS2',X='mjd'  ,BCD1='IN',BCD2='IN'  , Title='Yes', band='Lband')
statvis2Ll=wu.errorBarsByTPL2(df,data='VIS2',X='mjd'  ,BCD1='IN',BCD2='IN'  , Title='Yes', band='Llong')
statvis2Mm=wu.errorBarsByTPL2(df,data='VIS2',X='mjd'  ,BCD1='IN',BCD2='IN'  , Title='Yes', band='Mband')


statvis2Ls=wu.errorBarsByTPL2(df,data='T3PHI',X='LFlux',BCD1='OUT',BCD2='OUT', Title='Yes', band='Lshort')
statvis2Lm=wu.errorBarsByTPL2(df,data='T3PHI',X='LFlux',BCD1='OUT',BCD2='OUT', Title='Yes', band='Lband')
statvis2Ll=wu.errorBarsByTPL2(df,data='T3PHI',X='LFlux',BCD1='OUT',BCD2='OUT', Title='Yes', band='Llong')
statvis2Mm=wu.errorBarsByTPL2(df,data='T3PHI',X='LFlux',BCD1='OUT',BCD2='OUT', Title='Yes', band='Mband')

statvis2Ls=wu.errorBarsByTPL2(df,data='T3PHI',X='tau0' ,BCD1='OUT',BCD2='OUT', Title='Yes', band='Lshort')
statvis2Lm=wu.errorBarsByTPL2(df,data='T3PHI',X='tau0' ,BCD1='OUT',BCD2='OUT', Title='Yes', band='Lband')
statvis2Ll=wu.errorBarsByTPL2(df,data='T3PHI',X='tau0' ,BCD1='OUT',BCD2='OUT', Title='Yes', band='Llong')
statvis2Mm=wu.errorBarsByTPL2(df,data='T3PHI',X='tau0' ,BCD1='OUT',BCD2='OUT', Title='Yes', band='Mband')

statvis2Ls=wu.errorBarsByTPL2(df,data='T3PHI',X='mjd'  ,BCD1='OUT',BCD2='OUT', Title='Yes', band='Lshort')
statvis2Lm=wu.errorBarsByTPL2(df,data='T3PHI',X='mjd'  ,BCD1='OUT',BCD2='OUT', Title='Yes', band='Lband')
statvis2Ll=wu.errorBarsByTPL2(df,data='T3PHI',X='mjd'  ,BCD1='OUT',BCD2='OUT', Title='Yes', band='Llong')
statvis2Mm=wu.errorBarsByTPL2(df,data='T3PHI',X='mjd'  ,BCD1='OUT',BCD2='OUT', Title='Yes', band='Mband')

statvis2Ls=wu.errorBarsByTPL2(df,data='T3PHI',X='LFlux',BCD1='IN',BCD2='IN'  , Title='Yes', band='Lshort')
statvis2Lm=wu.errorBarsByTPL2(df,data='T3PHI',X='LFlux',BCD1='IN',BCD2='IN'  , Title='Yes', band='Lband')
statvis2Ll=wu.errorBarsByTPL2(df,data='T3PHI',X='LFlux',BCD1='IN',BCD2='IN'  , Title='Yes', band='Llong')
statvis2Mm=wu.errorBarsByTPL2(df,data='T3PHI',X='LFlux',BCD1='IN',BCD2='IN'  , Title='Yes', band='Mband')

statvis2Ls=wu.errorBarsByTPL2(df,data='T3PHI',X='tau0' ,BCD1='IN',BCD2='IN'  , Title='Yes', band='Lshort')
statvis2Lm=wu.errorBarsByTPL2(df,data='T3PHI',X='tau0' ,BCD1='IN',BCD2='IN'  , Title='Yes', band='Lband')
statvis2Ll=wu.errorBarsByTPL2(df,data='T3PHI',X='tau0' ,BCD1='IN',BCD2='IN'  , Title='Yes', band='Llong')
statvis2Mm=wu.errorBarsByTPL2(df,data='T3PHI',X='tau0' ,BCD1='IN',BCD2='IN'  , Title='Yes', band='Mband')

statvis2Ls=wu.errorBarsByTPL2(df,data='T3PHI',X='mjd'  ,BCD1='IN',BCD2='IN'  , Title='Yes', band='Lshort')
statvis2Lm=wu.errorBarsByTPL2(df,data='T3PHI',X='mjd'  ,BCD1='IN',BCD2='IN'  , Title='Yes', band='Lband')
statvis2Ll=wu.errorBarsByTPL2(df,data='T3PHI',X='mjd'  ,BCD1='IN',BCD2='IN'  , Title='Yes', band='Llong')
statvis2Mm=wu.errorBarsByTPL2(df,data='T3PHI',X='mjd'  ,BCD1='IN',BCD2='IN'  , Title='Yes', band='Mband')
#PLOT TO HERE

#to do all channels and bcds oo ii at once saving
#no mjd selection
#3.3 to 3.5um
wu.errorBarsByTPL2(df,data='VIS2',X='LFlux',BCD1='OUT',BCD2='OUT')  #old version
wu.errorBarsByTPL2(df,data='VIS2',X='LFlux',BCD1='IN',BCD2='IN')
wu.errorBarsByTPL2(df,data='VIS2',X='tau0', BCD1='OUT',BCD2='OUT')
wu.errorBarsByTPL2(df,data='VIS2',X='tau0', BCD1='IN',BCD2='IN')
#Zoom  ?? pending
wu.errorBarsByTPL2(df,data='VIS2',X='mjd', BCD1='OUT',BCD2='OUT')
wu.errorBarsByTPL2(df,data='VIS2',X='mjd', BCD1='IN',BCD2='IN')

for i in range(110): print(i,df['WL'][0][i])
0 4.9053774
1 4.892955
2 4.8804116
3 4.8677473
4 4.8549633
5 4.842058
6 4.8290324
7 4.815886
8 4.802619  <---
9 4.7892313
10 4.775723
11 4.762094
12 4.7483444
13 4.734474
14 4.7204833
15 4.706372  incl
16 4.6921396
17 4.677787
18 4.6633134
19 4.6487193
20 4.6340046
21 4.6191688
22 4.604213  <---
23 4.5891366
24 4.5739393
25 4.5586214
26 4.5431824
27 4.527623
28 4.511944
29 4.496143
30 4.4802217
31 4.46418
32 4.448018
33 4.431735
34 4.4153314
35 4.3988066
36 4.3821616
37 4.365396
38 4.34851
39 4.3315024
40 4.3143754
41 4.297127
42 4.2797585
43 4.2622685
44 4.2446585
45 4.226928
46 4.2090764
47 4.1911044
48 4.173012
49 4.154798
50 4.136464
51 4.1180096
52 4.0994344
53 4.0807385
54 4.061922
55 4.042985
56 4.023927
57 4.0047483
58 3.9854496
59 3.96603
60 3.9464893
61 3.9268284
62 3.9070468
63 3.8871443
64 3.8671215
65 3.846978  <---
66 3.8267136
67 3.8063288
68 3.785823
69 3.765197
70 3.74445  incl
71 3.7235827
72 3.7025948
73 3.681486
74 3.6602566
75 3.6389065   <---
76 3.617436
77 3.5958447
78 3.5741327
79 3.5523  incl
80 3.5303469
81 3.5082731
82 3.4860787
83 3.4637635
84 3.4413276  <---
85 3.4187713
86 3.396094
87 3.3732965
88 3.3503783
89 3.3273392  incl
90 3.3041794
91 3.280899
92 3.2574983
93 3.2339766   <---
94 3.2103345
95 3.1865716
96 3.1626883
97 3.1386843
98 3.1145594
99 3.090314
100 3.0659478
101 3.0414612
102 3.016854
103 2.992126
104 2.967277
105 2.942308
106 2.9172182
107 2.8920076
108 2.8666763
109 2.8412247
In [168]: for i in range(64): print(i,df['WL'][40][i])
0 4.2090764
1 4.1911044
2 4.173012
3 4.154798
4 4.136464
5 4.1180096
6 4.0994344
7 4.0807385
8 4.061922
9 4.042985
10 4.023927
11 4.0047483
12 3.9854496
13 3.96603
14 3.9464893
15 3.9268284
16 3.9070468
17 3.8871443
18 3.8671215
19 3.846978  <----
20 3.8267136
21 3.8063288
22 3.785823
23 3.765197
24 3.74445    incl
25 3.7235827
26 3.7025948
27 3.681486
28 3.6602566
29 3.6389065  <----
30 3.617436
31 3.5958447
32 3.5741327
33 3.5523
34 3.5303469  incl
35 3.5082731
36 3.4860787
37 3.4637635
38 3.4413276  <----
39 3.4187713
40 3.396094
41 3.3732965
42 3.3503783
43 3.3273392 incl
44 3.3041794
45 3.280899
46 3.2574983
47 3.2339766  <----
48 3.2103345
49 3.1865716
50 3.1626883
51 3.1386843
52 3.1145594
53 3.090314
54 3.0659478
55 3.0414612
56 3.016854
57 2.992126
58 2.967277
59 2.942308
60 2.9172182
61 2.8920076
62 2.8666763
63 2.8412247
#Added option for Lshort,Lband, Llarge and Mband
#Added returns array with VIS2 medias and std
#Corrected size of plots
#FALTA: Add  quality test to get rid of mjd's, also passing a list of mjds to avoid  AND RE-DO all Plots!!!!
statvis2Ls=wu.errorBarsByTPL2(df,data='VIS2',X='LFlux',BCD1='OUT',BCD2='OUT',band='Lshort')
statvis2L=wu.errorBarsByTPL2(df,data='VIS2',X='LFlux',BCD1='OUT',BCD2='OUT',band='Lband')
statvis2Ll=wu.errorBarsByTPL2(df,data='VIS2',X='LFlux',BCD1='OUT',BCD2='OUT',band='Llarge')
statvis2M=wu.errorBarsByTPL2(df,data='VIS2',X='LFlux',BCD1='OUT',BCD2='OUT',band='Mband')

wu.errorBarsByTPL2(df,data='VIS2',X='LFlux',BCD1='IN',BCD2='IN')
wu.errorBarsByTPL2(df,data='VIS2',X='tau0', BCD1='OUT',BCD2='OUT')
wu.errorBarsByTPL2(df,data='VIS2',X='tau0', BCD1='IN',BCD2='IN')
#Zoom  ?? pending
wu.errorBarsByTPL2(df,data='VIS2',X='mjd', BCD1='OUT',BCD2='OUT')
wu.errorBarsByTPL2(df,data='VIS2',X='mjd', BCD1='IN',BCD2='IN')





PLOT: Noise uncertainty and TF uncertainty as function of flux for diff. wls.

Aug16 plotting chopped and non-chopped  SeptCals
W: mjand j (430 files LOW/LM/+UTs?) there were no files completely misslabeled (hdr copping but it wasnt or the other way around)
os.chdir('/Users/M51/TablesPicklesScripts')
df=pd.read_pickle('SeptCalsDRSv1.4.1.pk')
selnonch=[]
for i in range(df.shape[0]):
    if((df['resolution'][i]=='LOW')&(df['chopping'][i]=='F')&((df['STATIONV2'][i][0][0]==34) or (df['STATIONV2'][i][0][0]==35))):  #for V2
        selnonch.append(i)
    print(df['resolution'][i],df['chopping'][i],df['STATIONV2'][i][0][0])

len(selnonch) #189

df=pd.read_pickle('SeptCalsCTDRSv1.4.1.pk')
selchCT=[]
for i in range(df.shape[0]):
    print(df['resolution'][i],df['chopping'][i],df['STATIONV2'][i][0][0])
    if((df['resolution'][i]=='LOW')&(df['chopping'][i]=='T')&((df['STATIONV2'][i][0][0]==34) or (df['STATIONV2'][i][0][0]==35))):  #UT1 or UT2
        selchCT.append(i)

len(selchCT)  #104

df=pd.read_pickle('amjCalsCTDRSv1.4.1.pk')  #178,75

selchCT=[]
for i in range(df.shape[0]):
    print(df['resolution'][i],df['chopping'][i],df['STATIONV2'][i][0][0])
    if((df['resolution'][i]=='LOW')&(df['chopping'][i]=='T')&((df['STATIONV2'][i][0][0]==34) or (df['STATIONV2'][i][0][0]==35))):  #UT1 or UT2
        selchCT.append(i)

len(selchCT) #178


#RAW VIS2 vs wl
Plots: select LOW , CHOPPED, UTs
#DRS calibrated data  or raw data (no corrections)
#UTS ONLY!!!!
#df=pd.read_pickle('/Users/M51/TablesPicklesScripts/SeptCalsDRSv1.4.1.pk')

#Aug23?
Made it to a function in wutilmod
plotrawvis2sel(amjds50,dfc,dfnc)
#may?
dfc=pd.read_pickle('/Users/M51/TablesPicklesScripts/amjCalsCTDRSv1.4.1.pk')
dfnc=pd.read_pickle('/Users/M51/TablesPicklesScripts/mayCalsDRSv1.4.1.pk')
#Sept
dfc=pd.read_pickle('/Users/M51/TablesPicklesScripts/septCalsCTDRSv1.4.1.pk')
dfnc=pd.read_pickle('/Users/M51/TablesPicklesScripts/septCalsDRSv1.4.1.pk')

anamesc=np.unique(df['name']) #21
atplsc=np.unique(dfc['tplstart']) #26may   102sept

#To do Quality Control 1: vis by eye
using dfc=df  from above Aug26
#for i in range(np.size(anames)):
arraymedLNC=[]
arraystdLNC=[]
arraymedL=[]
arraystdL=[]
arraymedMNC=[]
arraystdMNC=[]
arraymedM=[]
arraystdM=[]
arrayname=[]
count2=0
count22=0
for i in range(np.size(atplsc)):
    #print(anames[i])
    #print(atpls[i])
    #df2=df[df['name']==anames[i]]
    df2c=dfc[dfc['tplstart']==atplsc[i]] #BASED ON CHOPPED ONES
    df2c=df2c.reset_index()
    df2nc=dfnc[dfnc['tplstart']==atplsc[i]]
    df2nc=df2nc.reset_index()
    selnonch=[]  #or chopped   23 tpls
    selch=[]
    print(df2.shape[0])
    print('here',df2c.keys())
    for j in range(df2c.shape[0]):
        if((df2c['resolution'][j]=='LOW')&(df2c['chopping'][j]=='T')&((df2c['STATIONV2'][j][0][0]==34) or (df2c['STATIONV2'][j][0][0]==35))):  #for V2
            print(df2c['mjdo'][j],j)
            temp=df2c['mjdo'][j] in amjds50
            if(temp==True):
                print('HERE')
                selch.append(j)
            #print(df2c['resolution'][j],df2c['chopping'][j],df2c['STATIONV2'][j][0][0])
            #print(df2c['name'][j],df2c['tplstart'][j],np.str("%.2f" % df2c['T0S'][j]),df2c['mjdo'][j],np.str("%.2f" % df2c['LFlux'][j]),np.str("%.2f" % df2c['UDDL'][j]))
    for j in range(df2nc.shape[0]):
        if((df2nc['resolution'][j]=='LOW')&(df2nc['chopping'][j]=='F')&((df2nc['STATIONV2'][j][0][0]==34) or (df2nc['STATIONV2'][j][0][0]==35))):  #for V2
            print(df2c['mjdo'][j],j)
            temp=df2c['mjdo'][j] in amjds50
            if(temp==True):
                print('THERE')
                selnonch.append(j)

    print(selch,len(selch))
    print(selnonch,len(selnonch))
    if(len(selch)!=0):# to grab the associated ones to the chopped files
        #print(count2)
        outch=wu.plotvis2(df2c,'table','T','F',selch)  #3.5 and 4.75+-0.1
        tit= str(outch[0])+'chopped(CT)-'+str(i)
        if(len(selnonch)!=0):
            arraymedL. append(outch[1])  #for plotvis2 on selnonch
            arraystdL.append(outch[2])
            arraymedM.append(outch[3])  #for plotvis2 on selnonch
            arraystdM.append(outch[4])
            arrayname.append(outch[0])
            plt.savefig(tit)
        count2+=1
        plt.clf()
        if(len(selnonch)!=0):#Include to plot nonchopped, change title in plotvis2
            outnonch=wu.plotvis2(df2nc,'table','T','F',selnonch) #3.5 and 4.75+-0.1

            arraymedLNC.append(outnonch[1])  #for plotvis2 on selnonch
            arraystdLNC.append(outnonch[2])
            arraymedMNC.append(outnonch[3])  #for plotvis2 on selnonch
            arraystdMNC.append(outnonch[4])
            tit= str(outch[0])+'non-chopped-'+str(i)
            plt.savefig(tit)
            count22+=1
            plt.clf()


#CONCLUSIONS  @3.5um and @3.8um
arraymedL
arraystdL    #Look very similar  to non ch
arraymedLNC  #is larger most of the times
arraystdLNC  #Look very similar  to ch
arraymedM    #is larger in the 6 stars left, contrary to L band
arraystdM    #Look very similar  to non ch
arraymedMNC
arraystdMNC  #Look very similar  to ch  , largest diff=.06 only in 1 case
The non-chopped ones in L tend to a larger dispession from 3.5 to larger wls for the faintest *s, while the chopped ones keep a straight vis2, in these cases the M band is gone in both ch and nonch.
The resolution effects are clearly defined in the chopped, in gral there seems more "order" in bls with chopped.
When M band is good, in the non-chopped it decays with larger wl, or remains flat while chopped ones continue with the same slope as in L band, growing signal with larger wl.
HD16658 M band has neg values.

#3.5 and 4.75+-0.1
len(arrayname)  #21  15sept
len(arraymedL)  #21
len(arraymedLNC)
len(arraymedM)   #21
len(arraymedMNC)

x = np.linspace(0,0.5)
plt.plot(x,x,color='black')
plt.errorbar(arraymedL,arraymedLNC,xerr=arraystdL,yerr=arraystdLNC,fmt='o',ecolor='orange',capthick=2)
for i in range(len(arrayname)):
    plt.text(arraymedL[i],arraymedLNC[i],arrayname[i],size=7)
plt.xlim(0,0.5)
plt.ylim(0,0.5)
plt.title('raw VIS2 L band chopped vs non-chopped')
plt.xlabel('Raw VIS2 L band chopped')
plt.ylabel('Raw VIS2 L band non-chopped')
plt.grid(linestyle="--",linewidth=0.1,color='.25')


for i in range(len(arrayname)): print(arraystdL[i]-arraystdLNC[i])
0.009748840021906194
-0.0075277257835586475
-0.0074463545266025485
0.00976708061861574
-0.00888636220150088
-0.00837677981133516
-0.0034364075335908406
-0.00823843556738052
0.0009631576716501583
-0.031019801528358676
-0.1338298605011813
-0.0011214482764730915
-0.007438740800890728
0.0050678481758877625
-0.005103569687501003
0.02216158040192573
-0.004457252821981755
-0.046693777158324345
-0.001991811759799067
-0.005865931971516997
-0.00885532973908787

CT
0.00986233765249385
-0.006242209576586832
-0.008048435230958002
0.01076096233240989
-0.006881296980571008
-0.003389319888414788
-0.0053125938035329945
-0.012532718899401558
0.002207373085957212
-0.017927381311689507
-0.13430860057748098
-0.0005318748974860593
-0.006831615119271425
0.001252917545187096
-0.004382640508401994
0.024704987365439246
0.0009513102516756355
-0.047697602131514596
-0.002173495741025641
-0.0053682455424109154
-0.0036789265358529027

x = np.linspace(0,1)
plt.plot(x,x,color='black')
for i in range(len(arrayname)):
    if(arraymedMNC[i]<=1 and arraymedM[i]<=1):
        print(arrayname[i])
        plt.errorbar(arraymedM[i], arraymedMNC[i] ,xerr=arraystdM[i] ,yerr=arraystdMNC[i] ,fmt='o',ecolor='g',capthick=2)
        plt.text(arraymedM[i],arraymedMNC[i],arrayname[i],size=7)
        print(arraystdM[i]-arraystdMNC[i])

plt.title('raw VIS2 M band chopped vs non-chopped')
plt.xlabel('Raw VIS2 M band chopped')
plt.ylabel('Raw VIS2 M band non-chopped')
plt.grid(linestyle="--",linewidth=0.1,color='.25')
plt.xlim(0,0.6)
plt.ylim(0,0.6)

plt.title('raw VIS2 L(3.5+-0.1um,orange) & M(4.75+-0.1um,green) bands chopped vs non-chopped')
plt.xlabel('Raw VIS2 chopped')
plt.ylabel('Raw VIS2 non-chopped')

HR_6778
HD 89998
HD 101666
HD184996
HD 101666
HD184996

CT
HR_6778
0.005401387494516678
HD 89998
0.007201516194717719
HD 101666
0.011723469212927953
HD184996
0.07096400852977619
HD 101666
0.004652955177482834
HD184996
0.008120582344812652
    ...:
HR_6778
0.0001456941129679909
HD 89998
0.008212622122359459
HD 101666
0.00818720561617807
HD184996
0.06991588816570554
HD 101666
0.0037683386733484286
HD184996
0.009151894891550591



#list vis2<1 usnig or , M
HR_6778
HD 89998
UCAC2 3446325
HD 101666
UCAC2 5719521
HD184996
UCAC2  29197485
UCAC2 5719521
HD 101666
UCAC2 14406654
UCAC2 14406654
UCAC2  29197485
UCAC2 5719521
HD184996
UCAC2  29197485
#List vis2>1  M, using or
for i in range(len(arrayname)):
    if(arraymedMNC[i]>1 or arraymedM[i]>1):
        print(arrayname[i])
UCAC2 3446325
UCAC2 10873525
UCAC2 5719521
UCAC2  29197485
UCAC2 5719521
UCAC2 14406654
UCAC2 14406654
UCAC2  29197485
UCAC2 5719521
UCAC2  29197485



Aug15
modif.  df=wum.tableDRSCalRIVIS2  to search in various directories, it now TAKES AN ARRAY
adir=np.array(["/Volumes/MD/Reduced/Aug14MarAprCalsDRSv1.4.1/Iter1","/Volumes/MD/Reduced/Aug8DRSMayJunCals/Iter1"])
df=wu.tableDRSCalRIVIS2(adir)
1790
/Volumes/MD/Reduced/Aug14MarAprCalsDRSv1.4.1/Iter1/mat_raw_estimates.2019-03-22T23_47_42.HAWAII-2RG.rb/RAW_VIS2_0001.fits
1810  #not good!!!  should be exactly the same
/Volumes/MD/Reduced/Aug14MarAprCalsDRSv1.4.1/Iter1/mat_raw_estimates.2019-03-22T23_47_42.HAWAII-2RG.rb/RAW_DPHASE_0001.fits
1790

/Volumes/MD/Reduced/Aug8DRSMayJunCals/Iter1/mat_raw_estimates.2019-04-30T03_24_33.HAWAII-2RG.rb
15 of all except cphase (12) raw_spec raw_tf2 and raw_vis2
#modified it!!
#crash at /Volumes/MD/Reduced/Aug8DRSMayJunCals/Iter1/mat_raw_estimates.2019-04-02T08_14_15.HAWAII-2RG.rb/RAW_VIS2_0001.fits
#/Volumes/MD/Reduced/Aug8DRSMayJunCals/Iter1/mat_raw_estimates.2019-04-25T01_48_42.HAWAII-2RG.rb/RAW_VIS2_0001.fits
#Moved the files away
#crashed at /Volumes/MD/Reduced/Aug8DRSMayJunCals/Iter1/mat_raw_estimates.2019-04-26T04_33_49.HAWAII-2RG.rb/RAW_VIS2_0001.fits
#264 tet_Sco 264.329688     -42.99782
#37  changed the check for index2 is nan to include index1
'''crashes!!ß
    /Volumes/MD/Reduced/Aug8DRSMayJunCals/Iter1/mat_raw_estimates.2019-06-01T05_04_13.HAWAII-2RG.rb/RAW_VIS2_0007.fits
1615 HD_151773 252.986201     -50.02949
297445 198174
(1616, 72)
/Volumes/MD/Reduced/Aug8DRSMayJunCals/Iter1/mat_raw_estimates.2019-06-01T05_04_13.HAWAII-2RG.rb/RAW_VIS2_0008.fits
ERROR:root:Internal Python error in the inspect module.
Below is the traceback from this internal error.

Traceback (most recent call last):
  ...
OSError: [Errno 24] Too many open files

During handling of the above exception, another exception occurred:
...
OSError: [Errno 24] Too many open files'''
#Different chop, chopf, res and dit!!
#data with NO CHOPPING TEST!!!
dirmarch=['/Volumes/MD/Reduced/Aug14MarAprCalsDRSv1.4.1/Iter1'] #200
dirapril=['/Volumes/MD/Reduced/Aug8DRSMayJunCals/Iter1/april']#263
dirmay=['/Volumes/MD/Reduced/Aug8DRSMayJunCals/Iter1/may']  #800
dirjune=['/Volumes/MD/Reduced/Aug8DRSMayJunCals/Iter1/june'] #512
dirsept=['/Volumes/MD/COPYLaCieSeptemberData/SeptemberData/May25CalsSeptDRSv.1.4'] #485
dirseptCT=['/Volumes/MD/Reduced/Aug4DRSSeptCalsAfterChoppingTestv1.4.1/Iter1'] #104  #has chopped data only
diramjCT=['/Volumes/MD/Reduced/Aug16DRSamjCalsCTv1.4.1/Iter1']  #has chopped data only
df=wu.tableDRSCalRIVIS2(dirmarch)#includes now bl and angles
df.to_pickle('marchCalsDRSv1.4.1.pk')
df=wu.tableDRSCalRIVIS2(dirapril)
df.to_pickle('aprilCalsDRSv1.4.1.pk')
df=wu.tableDRSCalRIVIS2(dirmay)
df.to_pickle('mayCalsDRSv1.4.1.pk')
df=wu.tableDRSCalRIVIS2(dirjune)
df.to_pickle('juneCalsDRSv1.4.1.pk')
df=wu.tableDRSCalRIVIS2(dirsept)
df.to_pickle('septCalsDRSv1.4.1.pk')
df=wu.tableDRSCalRIVIS2(dirseptCT) #has chopped data only
df.to_pickle('septCalsCTDRSv1.4.1.pk')
df=wu.tableDRSCalRIVIS2(diramjCT) #has chopped data only
df.to_pickle('amjCalsCTDRSv1.4.1.pk')

df=pd.read_pickle('marchCalsDRSv1.4.1.pk')#no UTs
df=pd.read_pickle('aprilCalsDRSv1.4.1.pk')#no UTs   <---faint ATs
df=pd.read_pickle('mayCalsDRSv1.4.1.pk')  #lots
df=pd.read_pickle('juneCalsDRSv1.4.1.pk') #few UTs
df=pd.read_pickle('SeptCalsDRSv1.4.1.pk') #few UTs

         
notfound=[]
for i in range(df.shape[0]):
    if(df['UDDK'][i]==0):
        notfound.append(df['name'][i])



print(np.unique(notfound))  #march
['HD66424' 'epsSco']
print(np.unique(df['name']))  #march
['HD 101666' 'HD66424' 'N_Vel' 'V532Car' 'VV1068Sco' 'delSgr' 'epsSco'
 'gamAql' 'lamSgr' 'lam_Vel' 'upsLib' 'zet_Ara']

print(np.unique(notfound))  #april
['tet Cen' 'tet_Cen' 'tet_Sco']
print(np.unique(df['name'])) #april
['HD115418' 'HD142425' 'HD74600' 'HD_128154' 'IRAS10153-5540'
 'IRAS12563-6100' 'VV396Cen' 'bet_TrA' 'eps Crt' 'gam cru' 'iot_Ant'
 'tet Cen' 'tet_Cen' 'tet_Sco']

print(np.unique(notfound))  #may
['HD123139' 'IRAS08534-2405' 'UCAC2  29197485' 'UCAC2 10873525'
 'UCAC2 14406654' 'UCAC2 3446325' 'UCAC2 5719521' 'tet_Cen']
print(np.unique(df['name']))  #may
['106Vir' '35Vir' '37Lib' '37_Lib' '50_Vir' '56_Hya' '75Vir' 'HD  90798'
 'HD 101666' 'HD 89998' 'HD115418' 'HD123139' 'HD135367' 'HD136422'
 'HD138492' 'HD138938' 'HD142425' 'HD149447' 'HD151680' 'HD152786'
 'HD168454' 'HD181109' 'HD183925' 'HD184996' 'HD186251' 'HD189695'
 'HD74600' 'HD85139' 'HD86355' 'HD_114889' 'HD_135367' 'HD_138742'
 'HD_148103' 'HD_156637' 'HD_80479' 'HIP55282' 'HR_6778' 'IRAS08534-2405'
 'IRAS10153-5540' 'IRAS12563-6100' 'N_Vel' 'UCAC2  29197485'
 'UCAC2 10873525' 'UCAC2 14406654' 'UCAC2 3446325' 'UCAC2 5719521'
 'V849Ara' 'VV396Cen' 'V_V602_Car' 'alf Hya' 'alfCirc' 'del_Sgr' 'e01Sgr'
 'i Aql' 'nuLib' 'tet_Cen' 'tet_Lib']

print(np.unique(notfound))  #June W:bad weather , also July
['HD107446' 'HD122451' 'HD125687' 'HD129078' 'HD151680' 'HD169420'
 'HD97048']
print(np.unique(df['name']))   #june
['HD102839' 'HD102964' 'HD107446' 'HD122451' 'HD125687' 'HD129078'
 'HD133550' 'HD136422' 'HD138492' 'HD139997' 'HD147084' 'HD149447'
 'HD151680' 'HD151773' 'HD152161' 'HD158106' 'HD160668' 'HD162744'
 'HD165135' 'HD168454' 'HD169420' 'HD171443' 'HD174796' 'HD177716'
 'HD181109' 'HD183799' 'HD183925' 'HD6805' 'HD88624' 'HD97048' 'HD_148952'
 'HD_151773' 'HD_321151' 'c02_Aqr' 'del_Sgr' 'e01_Sgr' 'zet_Ara'
 'zeta_Ara']

print(np.unique(notfound))
['CD-55 855']
print(np.unique(df['name']))
['47 Cap' '84 Aqr' 'CD-55 855' 'HD 16658' 'HD 172453' 'HD 183925'
 'HD 195677' 'HD 20356' 'HD 212849' 'HD 29246' 'HD 6290' 'HD14823'
 'HD175786' 'HD186791' 'HD209240' 'HD25286' 'HD33162' 'V4026 Sgr' 'b Sgr'
 'del Phe' 'sig Cap']


print(np.unique(notfound))  #septCT  has chopped data only, chopping test applied
[]
print(np.unique(df['name'])) #septCT
['47 Cap' '84 Aqr' 'HD 16658' 'HD 172453' 'HD 183925' 'HD 195677'
 'HD 20356' 'HD 6290' 'HD209240' 'HD33162' 'V4026 Sgr' 'del Phe' 'sig Cap']

selection=[]
NC=[]
for i in range(df.shape[0]):
    if((df['resolution'][i]=='LOW')&(df['chopping'][i]=='F')&((df['STATION'][i][0][0]==32) or (df['STATION'][i][0][0]==33))):  #UT1
        NC.append(i)
    #print(df['resolution'][i],df['chopping'][i],df['STATION'][i][0][0])
    if((df['resolution'][i]=='LOW')&(df['chopping'][i]=='T')&((df['STATION'][i][0][0]==32) or (df['STATION'][i][0][0]==33))):  #UT1
         selection.append(i)


len(selection)
may:  88
jun: [242, 243, 244, 245, 254, 255, 256, 257]
sept: 111

Aug 7 copying data from Merkske:
scp -r gamez@merkske.strw.leidenuniv.nl:/data2/0501to0701 .
Dec182019
copying data from Westerschelde:
scp -r gamez@westerschelde.strw.leidenuniv.nl:/data2/drs/nov2019data/MAT*.fits .
scp -r gamez@westerschelde.strw.leidenuniv.nl:/data2/drs/nov2019data/MAT*.fits 

Aug 5
Plotting reduced chopped data! :)
wu.plotvis2("/Volumes/MD/Reduced/Aug5DRSNGC1068Sept/Iter1/mat_raw_estimates.2018-09-25T05_21_36.HAWAII-2RG.rb/RAW_VIS2_0001.fits",'F')


df=wu.tableDRSCalRIVIS2("/Volumes/MD/Reduced/Aug5DRSNGC1068Sept/Iter1")


Aug 1st  WORKING WITH CHOPPED DATA LM BANDS
0.- DO CHOPPING TEST!!
1.- For mjj data make table with diameters for the calibrators:
#From the two catalogs, it saves two picklƒhdue files
# wu.tableDiam('/Volumes/LaCie/SeptemberData/All_AllBCDs')
#Dir where the EWS or DRS reduced files are. It could return a sub-table of the diameter -catalogs. Set up which one to return.
#CHANGE!! Fin1 and Fin2.to_pickle("TableDiamsTemp2msdfcc19.pk")
def tableDiam(Dir):
    wum.tableDiam("/Volumes/MD/Reduced/Jun4CalsAprilUTs")
 ['HR 6778','HD 80479','HD 89998','HD 101666','HD 114889','HD 138492','HD 148103','HD 156637','HD 184996','HD 321151','tet_Lib','UCAC2 3446325','UCAC2 5719521','UCAC2 10873525','UCAC2 14406654','UCAC2 29197485','50_Vir','37_Lib','alfCirc']

2.- make table with reduced data:
There are 4 OPTIONS:  Use 2, looks like the best and it will have the complete data
#DF.to_pickle('posmsdfcc.pk') # (465857, 2)  Done
#DF.to_pickle('posjsdc_2017.pk')  #(465877, 2) Done :)
#To make a table with all the reduced data
#Also for calibrated data!! :)
def tableDRSCalRI(inDIRvis2,inDIRcf,icf):  #icf to get INCCORRFLUX
    #df=wu.tableDRS('/Volumes/LaCie/DecemberData/All_AllBCDs/Iter1vis2','/Volumes/LaCie/DecemberData/All_AllBCDs/Iter1cf',True)
    patternData='*CALIB_RAW_INT_*' #gets them stacked up  #'RAW_VIS2*'

#To make a table with all the DRS incoherentlyreduced data with only visibiities (no corr-flxs, nor coherent integration)
#Also for calibrated data :)
#CAN NOT pass the sub-table
#wu.tableDRSCalRIVIS2('/Volumes/LaCie/Reduced/May25CalsSeptDRSv.1.4')
def tableDRSCalRIVIS2(inDIRvis2):
    #df=wu.tableDRS('/Volumes/LaCie/DecemberData/All_AllBCDs/Iter1vis2','/Volumes/LaCie/DecemberData/All_AllBCDs/Iter1cf',True)
    #patternData='2018*'   for FS CMa
    #fileList1 = []
    #for root, dirnames, filenames in os.walk(inDIRvis2):
    #    for filename in fnmatch.filter(filenames, patternData):
    #        fileList1.append(os.path.join(root, filename))
    patternData='RAW_VIS2*' #recovers all the mjds#/'*CALIB_RAW_INT_*'gets them stacked up
    patternData2='RAW_DPHASE*'
    patternData3='RAW_CPHASE*'


#makes a table from the CALIBRATED DATA
#wu.CalDatatableDRS('/Users/M51/NGC106822SeptDRScal')
def CalDatatableDRS(inDIRvis2):
    #df=wu.tableDRS('/Volumes/LaCie/DecemberData/All_AllBCDs/Iter1vis2','/Volumes/LaCie/DecemberData/All_AllBCDs/Iter1cf',True)
    patternData='TARGET*'#September22.fits'#'2018*'  #20for FS CMa

#df1=wu.tableDRSRawV2('/Volumes/LaCie/Reduced/SeptCals','/Volumes/LaCie/Reduced/SeptCals',False)
#USING ONLY RAW_VIS2 FILES
def tableDRSRawV2(inDIRvis2,inDIRcf,icf):  #icf to get INCCORRFLUX
    #df=wu.tableDRSRawV2('/Volumes/LaCie/DecemberData/All_AllBCDs/Iter1vis2','/Volumes/LaCie/DecemberData/All_AllBCDs/Iter1cf',True)
    patternData='RAW_VIS2*'

df=wum.tableDRSCalRIVIS2("/Volumes/MD/Reduced/Jun4CalsAprilUTs")
df.shape # (138, 72)
df.to_pickle('')
df.to_pickle('DRSAprilcals.pk')

#Copied data
df=wum.tableDRSCalRIVIS2("/Volumes/MD/AugustDRSreduced/vtest2") #253,72
df.to_pickle('vtest2.pk')



3.- Make plots of VIS2,VIS2PHI and T3PHI and check by eye
Quick check:

wu.plotvis2('RAW_VIS2_0001.fits','F')


cd /data2/drs/septdata/vtest2/Iter1
ll *.rb | grep -c rb
find . -name "*VIS2_0001*" | grep -c fits    #24
hdu=fits.open('RAW_VIS2_0001.fits')

from astropy.io import fits
import matplotlib.pylab as plt

for i in range(6):
    #hdu=fits.open(f)
    plt.plot(hdu[3].data['EFF_WAVE']*1e6,hdu[4].data['VIS2DATA'][0])
    plt.ylim(-1,1)
    object=hdu[0].header["HIERARCH ESO OBS TARG NAME"]
    plt.title(object)
    plt.show()
    plt.pause(25)
    plt.clf()


f=['mat_raw_estimates.2018-09-21T04_40_25.HAWAII-2RG.rb/RAW_VIS2_0001.fits','mat_raw_estimates.2018-09-24T06_48_55.HAWAII-2RG.rb/RAW_VIS2_0001.fits','mat_raw_estimates.2018-09-23T08_08_57.HAWAII-2RG.rb/RAW_VIS2_0001.fits','mat_raw_estimates.2018-09-21T09_33_42.HAWAII-2RG.rb/RAW_VIS2_0001.fits','mat_raw_estimates.2018-09-21T02_59_52.HAWAII-2RG.rb/RAW_VIS2_0001.fits','mat_raw_estimates.2018-09-22T04_44_55.HAWAII-2RG.rb/RAW_VIS2_0001.fits','mat_raw_estimates.2018-09-23T00_58_27.HAWAII-2RG.rb/RAW_VIS2_0001.fits','mat_raw_estimates.2018-09-23T09_13_51.HAWAII-2RG.rb/RAW_VIS2_0001.fits','mat_raw_estimates.2018-09-21T05_16_29.HAWAII-2RG.rb/RAW_VIS2_0001.fits','mat_raw_estimates.2018-09-21T09_01_35.HAWAII-2RG.rb/RAW_VIS2_0001.fits','mat_raw_estimates.2018-09-22T06_48_41.HAWAII-2RG.rb/RAW_VIS2_0001.fits','mat_raw_estimates.2018-09-21T03_54_26.HAWAII-2RG.rb/RAW_VIS2_0001.fits','mat_raw_estimates.2018-09-21T03_18_05.HAWAII-2RG.rb/RAW_VIS2_0001.fits','mat_raw_estimates.2018-09-21T05_56_34.HAWAII-2RG.rb/RAW_VIS2_0001.fits','mat_raw_estimates.2018-09-23T01_50_29.HAWAII-2RG.rb/RAW_VIS2_0001.fits','mat_raw_estimates.2018-09-22T05_18_44.HAWAII-2RG.rb/RAW_VIS2_0001.fits','mat_raw_estimates.2018-09-23T08_45_55.HAWAII-2RG.rb/RAW_VIS2_0001.fits','mat_raw_estimates.2018-09-21T02_44_43.HAWAII-2RG.rb/RAW_VIS2_0001.fits','mat_raw_estimates.2018-09-23T04_05_22.HAWAII-2RG.rb/RAW_VIS2_0001.fits','mat_raw_estimates.2018-09-23T23_25_55.HAWAII-2RG.rb/RAW_VIS2_0001.fits','mat_raw_estimates.2018-09-23T09_19_42.HAWAII-2RG.rb/RAW_VIS2_0001.fits','mat_raw_estimates.2018-09-24T08_29_25.HAWAII-2RG.rb/RAW_VIS2_0001.fits','mat_raw_estimates.2018-09-21T02_33_16.HAWAII-2RG.rb/RAW_VIS2_0001.fits','mat_raw_estimates.2018-09-23T07_32_18.HAWAII-2RG.rb/RAW_VIS2_0001.fits']




Pulling reduced data from Westerschelde: one file
scp gamez@westerschelde.strw.leidenuniv.nl:/data2/drs/septdata/vtest2/Iter1/mat_raw_estimates.2018-09-24T06_48_55.HAWAII-2RG.rb/RAW_VIS2_0001.fits ./
multiple files inside the folder that I give
scp -r gamez@westerschelde.strw.leidenuniv.nl:/data2/drs/septdata/vtest2/Iter1/mat_raw_estimates.2018-09-24T06_48_55.HAWAII-2RG.rb ./
#Only this kind
scp -r gamez@westerschelde.strw.leidenuniv.nl:/data2/drs/septdata/vtest2/Iter1/mat_raw_estimates.2018-09-24T06_48_55.HAWAII-2RG.rb/TARGET_RAW_INT* ./
Used shell scripts /Volumes/MD/AugustDRSreduced/vtest2/scriptpullDRSreddata.sh to 4
#westerschelde.strw.leidenuniv.nl  connection closed

#Selecting good mjds by eye
df=pd.read_pickle('/Users/M51/TablesPicklesScripts/.pk') #484, includes IO and OI
anames=np.unique(df['name']) #10
#To do Quality Control 1: vis by eye
for i in range(np.size(anames)):
    df2=df[df['name']==anames[i]]
    df2=df2.reset_index()
    plt.clf()
    wum.VIS2wldrscal(df2,'WL',50,100,0,130)

4.- Make TF plots
correction=[1,1,1,1,1,1]
#2d mine
wu.TFvis2tauDRS(df,'LFlux','T0E','VIS2',0,135,1,2,0,correction,0,'scatter')
#2D W
wu.TFvis2tauDRS(df,'LFlux','T0E','VIS2',0,135,1,2,0,correction,0,None)
#color,x,y  L
wu.TFvis2tauDRS(df,'LFlux','T0E','VIS2',0,135,1,2,0,correction,0,'scatter2')
python3 $autopath/mat_autoPipeline.py --nbCore=4 --dirCalib=$calmap --dirResult=$outdir --skipN --paramL=/cumulBlock=FALSE/useOpdMod=FALSE/compensate="pb,rb,nl,if,bp,od" $rawdata &
esorex --output-dir=/Volumes/LaCie/DecemberData/All_AllBCDs/Iter1/mat_raw_estimates.2018-12-03T03_31_23.HAWAII-2RG.rb mat_raw_estimates  --cumulBlock=FALSE --corrFlux=TRUE --coherentAlgo=2 --useOpdMod=TRUE /Volumes/LaCie/DecemberData/All_AllBCDs/Iter1/mat_raw_estimates.2018-12-03T03:31:23.HAWAII-2RG.sof
/data1/jvarga/matisse/drs/bin/esorex --output-dir=/data2/archive/data/MATISSE/matisse_red2//coherent/2019-03-22/2019-03-23T08_41_19/Iter1/mat_raw_estimates.2019-03-23T08_41_19.HAWAII-2RG.rb mat_raw_estimates --cumulBlock=FALSE --compensate="pb,rb,nl,if,bp,od" --hampelFilterKernel=10 --corrFlux=TRUE --useOpdMod=FALSE --coherentAlgo=2 -- /data2/archive/data/MATISSE/matisse_red2//coherent/2019-03-22/2019-03-23T08_41_19/Iter1/mat_raw_estimates.2019-03-23T08:41:19.HAWAII-2RG.sof

 mc.hdrsFromDisk(fileName='septheaders.pk')
In [14]: mc.select('tplstart','2018-09-22T04:44:55')
Out[14]: 28

In [15]: mc.values('detchopfreq',True)
Out[15]: array([0.])

In [16]: mc.values('detchopst',True)
Out[16]: array([False])

In [17]: mc.values('chopfreq',True)
Out[17]: array([0.      , 0.457875])

In [18]: mc.values('chopst',True)
Out[18]: array(['F       ', 'T       '], dtype=object)

In [19]: mc.values('chopthrow',True)
Out[19]: array([0., 4.])

In [20]: mc.select('chopst','T')
Out[20]: 14
In [22]: mc.selectedTplDirectories(mother='/Volumes/LaCie/SeptemberData/HD16658CHOPPED',minfiles=2)processed  1  templates
python3 /Users/M51/TestGit/tools/mat_tools/mat_autoPipeline.py --nbCore=8 --dirCalib=/Users/M51/MATISSEDATA/CalibMap --dirResult=/Volumes/LaCie/Reduced --skipN --paramL=/cumulBlock=FALSE/useOpdMod=FALSE/compensate="pb,rb,nl,if,bp,od" /Volumes/LaCie/SeptemberData/HD16658CHOPPED/2018-09-22T04:44:55.tpl DINDT WORK
    File "/Users/M51/TestGit/tools/mat_tools/mat_autoPipeline.py", line 114
    tplid    = hdr['HIERARCH ESO TPL ID']
    ^
IndentationError: expected an indented block
Using Ws Version:
python3 /Users/M51/TestGit/tools/mat_tools/mat_autoPipeline.py --nbCore=8 --dirCalib=/Users/M51/MATISSEDATA/CalibMap --dirResult=/Volumes/LaCie/Reduced --skipN --paramL=/cumulBlock=FALSE/useOpdMod=FALSE/compensate="pb,rb,nl,if,bp,od" /Volumes/LaCie/SeptemberData/HD16658CHOPPED/2018-09-22T04:44:55.tpl &
Recovered:
/Users/M51/TestGit/tools/mat_tools/automaticPipelineAug1st.py

--dirRaw=/Users/M51/MATISSEDATA/2018-07-13 --dirCalib=/Users/M51/MATISSEDATA/CalibMap --dirResult=/Users/M51/MATISSEDATA/Reduced —tplSTART=2018-07-13T10:08:15

python3 /Users/M51/TestGit/tools/mat_tools/automaticPipelineAug1st.py --dirRaw=/Volumes/LaCie/SeptemberData/HD16658CHOPPED/2018-09-22T04:44:55.tpl --dirCalib=/Users/M51/MATISSEDATA/CalibMap --dirResult=/Volumes/LaCie/Reduced --nbCore=1  --skipN --paramL=/cumulBlock=FALSE/useOpdMod=FALSE/compensate="pb,rb,nl,if,bp,od"  NOT WORKING
needs tidyup oifits??

gave:
esorex --output-dir=/Volumes/LaCie/Reduced/Iter1/mat_raw_estimates.2018-09-22T04_44_55.HAWAII-2RG.rb mat_raw_estimates  --cumulBlock=FALSE --useOpdMod=FALSE --compensate=pb,rb,nl,if,bp,od /Volumes/LaCie/Reduced/Iter1/mat_raw_estimates.2018-09-22T04:44:55.HAWAII-2RG.sof

python3 /Users/M51/TestGit/tools/mat_tools/automaticPipelineAug1st.py --dirRaw=/Volumes/LaCie/SeptemberData/HD16658CHOPPED/2018-09-22T04:44:55.tpl --dirCalib=/Users/M51/MATISSEDATA/CalibMap --dirResult=/Volumes/LaCie/Reduced --nbCore=1  --skipN --paramL=/cumulBlock=FALSE/useOpdMod=FALSE NOT WORKING
gave:
esorex --output-dir=/Volumes/LaCie/Reduced/Iter1/mat_raw_estimates.2018-09-22T04_44_55.HAWAII-2RG.rb mat_raw_estimates  --cumulBlock=FALSE --useOpdMod=FALSE /Volumes/LaCie/Reduced/Iter1/mat_raw_estimates.2018-09-22T04:44:55.HAWAII-2RG.sof
LOW

17:07:28 [WARNING] : lt_dlopen (/opt/local/lib/esopipes-plugins/iiinstrument-0.1.8/matisse-1.2.7/mat_ext_beams.so) returned NULL,
17:14:36 [ INFO  ] mat_raw_estimates: Starting mat_ext_beams
17:14:36 [ ERROR ] mat_ext_beams_lib: NO TARGET_CAL or CALIB_CAL frames found
17:14:36 [WARNING] mat_raw_estimates: Error in mat_ext_beams_lib

python3 /Users/M51/TestGit/tools/mat_tools/automaticPipelineAug1st.py --dirRaw=/Volumes/LaCie/SeptemberData/HD16658CHOPPED/2018-09-22T04:44:55.tpl --dirCalib=/Users/M51/MATISSEDATA/CalibMap --dirResult=/Volumes/LaCie/Reduced --nbCore=1  --skipN --paramL=/cumulBlock=FALSE/useOpdMod=FALSE > mylog.log -DCPL_ADD_FLOPS & DIDNT wORK

starlight:mat_tools starlight$ which esorex
/opt/local/bin/esorex
starlight:mat_tools starlight$ esorex
    
    ***** ESO Recipe Execution Tool, version 3.13.2  *****


Libraries used: CPL = 7.1.1, CFITSIO = 3.45, WCSLIB = 6.2, FFTW (normal precision) = 3.3.8-sse2, FFTW (single precision) = 3.3.8-sse2, CPL FLOP counting is unavailable, enable with -DCPL_ADD_FLOPS  DIDNT WORK

[WARNING] esorex: lt_dlopen (/opt/local/lib/esopipes-plugins/iiinstrument-0.1.8/matisse-1.2.7/mat_merge_results.so) returned NULL,

/opt/local/etc/macports/sources.conf
sudo port -f clean --all all
sudo port -f uninstall inactive
REINSTALLED MATISSE PIP 1.4.4
python3 /Users/M51/TestGit/tools/mat_tools/automaticPipelineAug1st.py --dirRaw=/Volumes/LaCie/SeptemberData/HD16658CHOPPED/2018-09-22T04:44:55.tpl --dirCalib=/Users/M51/MATISSEDATA/CalibMap --dirResult=/Volumes/LaCie/Reduced --nbCore=1  --skipN --paramL=/cumulBlock=FALSE/useOpdMod=FALSE > mylog.log -DCPL_ADD_FLOPS &
in esorex.log 21:51:28 [ INFO  ] cpl_recipedefine_init: Run-time version 7.1.1 of CPL is higher than the version (70100) used to compile visir_util_spc_txt2fits

python3 /Users/M51/TestGit/tools/mat_tools/automaticPipelineAug1st.py --dirRaw=/Volumes/LaCie/SeptemberData/HD16658CHOPPED/2018-09-22T04:44:55.tpl --dirCalib=/Users/M51/MATISSEDATA/CalibMap --dirResult=/Volumes/LaCie/Reduced --skipN  DIDNT WORK
did
esorex --output-dir=/Volumes/LaCie/Reduced/Iter1/mat_raw_estimates.2018-09-22T04_44_55.HAWAII-2RG.rb mat_raw_estimates --useOpdMod=FALSE --compensate=[pb,nl,if,rb,bp,od] /Volumes/LaCie/Reduced/Iter1/mat_raw_estimates.2018-09-22T04:44:55.HAWAII-2RG.sof


python3 automaticPipelineAug1st.py --dirRaw=/Volumes/LaCie/SeptemberData --dirCalib=/Users/M51/MATISSEDATA/CalibMap --dirResult=/Volumes/LaCie/Reduced --tplSTART=2018-09-22T04:44:55 --skipN  DIDNT WORK

python3 automaticPipelineAug1st.py --dirRaw=/Volumes/LaCie/SeptemberData --dirCalib=/Users/M51/MATISSEDATA/CalibMap --dirResult=/Users/M51 --tplSTART=2018-09-22T04:44:55 --maxIter=1 --skipN  WORKS!!!!! :)

python3 automaticPipelineAug1st.py --dirRaw=/Volumes/LaCie/SeptemberData --dirCalib=/Users/M51/MATISSEDATA/CalibMap --dirResult=/Users/M51 --maxIter=1 --nbCore=8 --skipN

En : /opt/local/var/macports/registry/portfiles/esopipe-matisse-recipes-1.4.1_6/95944563d55b7e84076588caf3513ac891c22df0724ff9920b238527e831dc56-1650/Portfile
configure.args  --with-cpl=${prefix} --with-erfa=${prefix} --with-gsl=${prefix}



post-destroot {
    system "cd ${worksrcpath}/reflex && make DESTDIR=${destroot} uninstall"
}

static int mat_raw_estimates_destroy(cpl_plugin *);

/Users/M51/matisse/install.sh/
UNINSTALL FIRST Users/M51/SOFT/INTROOT/matis/install.sh



#....+......+.......+.......+.......+.......+.........+......+

Jul19

 from mcdb import maxlike
tg=maxlike.twoGaussOneCentered
 tg.makeimage()
params=[17,10,]
10.36949135,  17.04338876,  44.02156558,   0.23058534,
    26.59059548,  86.14140077,   2.57778532,  29.00377378,
        133.86839425,  -1.0216261
''' params= majoraxis1(mas), minoraxis1(mas), pa(deg E of North), separation (mas), pasep(deg), amp, majoraxis2, minoraxis2, pa2, photfact
     lam = average wavelength (micron)'''
Jul17 plotting vis2drs on uvplane
Make  data frame with all
os.chdir('/Users/M51/TablesPicklesScripts')
try first: change file name pattern data
df=wu.tableDRSCalRIVIS2("/Users/M51/Downloads/HD45677/LMOUTOUT/preSelection") #didnt work
df=wu.CalDatatableDRS("/Users/M51/Downloads/HD45677/LMOUTOUT/preSelection/Selection") #made a few changes and WORKED
df.to_pickle('hd45677.pk')
In [94]: df.keys()
Out[94]:
Index(['i', 'n', 'name', 'pBLe', 'pBLAe', 'airmass', 'mjdo', 'mjd', 'dateobs', 'mode', 'object', 'categ', 'type', 'tplstart', 'T0E', 'T0S','windspeed', 'seeinge', 'seeings', 'chopping', 'chopfreq', 'BCD1','BCD2', 'resolution', 'wlrange', 'wl0', 'DIT', 'fringetracker','readoutMd', 'readout', 'filter', 'WL', 'lenWL', 'VIS2', 'VIS2ERR','LENVIS2', 'RA', 'DEC', 'UCOORD', 'VCOORD', 'BL', 'AO', 'CNAME', 'dist','ST', 'Kmag', 'Lmag', 'Mmag', 'Nmag', 'LFlux', 'MFlux', 'NFlux', 'UDDK','UDDL', 'UDDM', 'UDDN', 'interpflux', 'T3AMP', 'T3AMPERR', 'T3PHI','T3PHIERR', 'U1COORD', 'V1COORD', 'U2COORD', 'V2COORD', 'STATION', 'i1','i2'],dtype='object')
def tableDRSCalRI(inDIRvis2,inDIRcf,icf):  #didnt try
wu.visdrsagn(df,3.4,3.5,0,1)  #
In [228]: df['tplstart'][[1,2,4,5]]
Out[228]:
1    2018-12-10T04:48:14
2    2018-12-10T06:02:43
4    2018-12-10T08:40:41
5    2018-12-10T07:50:30

new funt in wutilmod
f="/Users/M51/NGC1068Sept22DRScaldelPhe/TARGET_CAL_INT_0001September22.fits"
f="/Users/M51/NGC1068Sept25DRScalHD20356/TARGET_CAL_INT_0001September25.fits"
f="/Users/M51/NGC1068Sept24DRScalHD25286/TARGET_CAL_INT_0001September24.fits"
f="/Users/M51/NGC1068Sept22HD16658/TARGET_CAL_INT_0001.fits"
f="/Users/M51/NGC1068Sept25HD209240/TARGET_CAL_INT_0001.fits"
f="/Users/M51/modelfittingresults/delSco/delsco_19may.fits"

wu.plotvis2(f,'F')
wu.plotvis2(f,'T')  #with error bars

wu.plotclph(f,'F')
wu.plotclph(f,'T')
df=wu.CalDatatableDRS('/Users/M51/NGC1068Sept24DRScalHD25286')#changed TARGET
df['VIS2'][0].shape
Out[121]: (12, 110)
df['mjd'][0][[0,6]]
Out[138]: array([58385.22799755, 58385.23226796])
df.to_pickle('S241068.pk')
df=wu.CalDatatableDRS('/Users/M51/NGC1068Sept22DRScaldelPhe')#changed TARGET
In [152]: df['mjd'][0].shape
Out[152]: (48,)
df.to_pickle('S221068.pk')
df=wu.CalDatatableDRS('/Users/M51/NGC1068Sept25DRScalHD20356')
In [158]: df['mjd'][0].shape
Out[158]: (42,)
In [159]: df.to_pickle('S251068.pk')
df=wu.CalDatatableDRS('/Users/M51/NGC1068Sept22HD16658')
df['mjd'][0].shape  48
df.to_pickle('S221068-2.pk')
df=wu.CalDatatableDRS('/Users/M51/NGC1068Sept25HD209240')
df['mjd'][0].shape  42
df.to_pickle('S251068-2.pk')

#plotting uv plane
df['mjd'][0].shape   #12
df=pd.read_pickle('S241068.pk')
wu.visdrsagnmultexp(df,3.4,3.5,0,1,'HD25286')#check wl
wu.visdrsagnmultexp(df,3.4,3.5,0,2,'HD25286')#check wl
df=pd.read_pickle('S221068.pk') #48
#Choose good mjds
wu.visdrsagnmultexp(df,3.4,3.5,0,8,'delPhe')#check wl
df=pd.read_pickle('S251068.pk') #42
wu.visdrsagnmultexp(df,3.4,3.5,0,6,'delPhe')#check wl




July 11
plotting vis2 April cals with neg values
Jul4

In order to add the ATs you can do the following steps:
1. Open the calibrated data in ipython and look at hdu[4].data['sta_index']
2. Copy the 6 baseline combinations
3. Add this to img2vis in ascending order(!!) in the same format as the others in match_bls
Had [1,28],[18,23],[23,28],[1,23],[18,28],[1,18]




Epsilon in model_fitter: 0.1um


15-20 for Sept 25th ews pickle

#H=?? −α
#NGC 1068: RA=40.670992deg=02:42:41 RA (J2000)  DEC=-00:00:47.8
#Sept24 DTATE-OBS=2018-09-24T05:30:54 , UTC=05:30:52
H = 01:01:15 - 02:42:41=1h-2.7h=-1.7h #End: 1h-3.7h=-2.7
#-2.7 to -1.7
#Conclusion: aspro has a different (in crescent order) definition of the bls and clphs, so it is not comparable tothe data of DRS!!!
#-------------------flipping fits file
from astropy.io import fits
hdu=fits.open('/Users/M51/modelfittingresults/v0.16/cluster/UsingWs/S24HD25286/1-mostprob_model.fits')
data2=hdu[0].data.copy()
import numpy as np
data2=np.flip(data2,axis=1)
hdu[0].data=data2.copy()
hdu.writeto('/Users/M51/modelfittingresults/v0.16/cluster/UsingWs/S24HD25286/flip.fits')

#-------------------------Chain plots
samples = np.load('samples.npy')
labels = ['fwhmx_1', 'fwhmy_1', 'pa_1', 'amp'] + ['sep','sep_angle', 'fwhmx_2', 'fwhmy_2', 'pa_2'] + ['logf']
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator
fig1 = pl.figure()
fig2 = pl.figure()
for i in range(samples.shape[-1]): #10 num of params
    ax = fig1.add_subplot(5, 2, i + 1)  #to have 5*2plots
    ax.plot(samples[:, :, i].T, "k", alpha=0.3)
    ax.set_xlim(0, samples.shape[1])
if i < 2:
    ax.set_xticklabels([])
else:
    ax.set_xlabel("Step Number")
    ax.set_ylabel(labels[i])
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax = fig2.add_subplot(5, 2, i + 1)
    ax.plot(samples[:, :, i].T, "k", alpha=0.6)
    #ax.set_xlim(0, 200)
if i < 2:
    ax.set_xticklabels([])
else:
    ax.set_xlabel("Step Number")
    ax.set_ylabel(labels[i])
    ax.yaxis.set_major_locator(MaxNLocator(4))

fig1.savefig("time.png", transparent=True, dpi=1000)
fig2.savefig("burnin.png", transparent=True, dpi=1000)
#plt.savefig('destination_path.eps', format='eps', dpi=1000)

x2_best = [3.282257, 6.859977, 45, 0.74424, 30.74578, np.degrees(-1.47862), 20.2627556, 12.57016, 40, -0.413009]


The simplest solution to run two Python processes concurrently is to run them from a bash file, and tell each process to go into the background with the & shell operator.

python script1.py &
python script2.py &



MIERC email alumno ese , Integrar la intensidad en los pixeles , HACER PRESENTACION!! Metropolis Hastings sampling
como se pasa el error en los datos al modelo?
PENDING DRS: amplitude cphase is =0 always
calibrar NGC1068 con los otros calibradores opcionales, mandar a Jacob DONE
1 Diapositiva del fit del modelo: MonteCarloMarkocChain, Th.Bayes, priors and maximum likelihoods, fitting parameters  LEFT FOR LATER

TARDE 1 Diapositiva datos y comportqmiento DRS
1 diapos modelo solo vis, y modelo J.

Ver si hay mas uv points nuevos, SI HAY, al menos 6 mas.
fit the model usando estos nuevos uv
hacer el modelo cromatico, incluir bandas L y M.  DONE but needs to be repeated with a model that we are completely sure that is right. Sampling the whole parameter space.

*go to HERE:)



PENDING but not so urgent: April cals problems for DRS
Reducir los dos April nuevos


massBokGob=2*10^(4)*0.835*10^(-27)*(4/3)*pi*(6.172*10^(18))^3  Kg
=16.447*10^(33)/(2*10^30) Mo
distance Roche=r*((1.7*10^7)/(8223.5)*2)^(1/3)=r*16.04996439937953
=.32pc*16.04996439937953unitless
=5.1pc
counting pixels projected distance is like 2.8pc!!

9.8752*10^(17)*(2*1.7*10^(7)*2*10^(30)987520000000000000^3*pi*(4/3)*(1.67*10^(-27)/2)*2*10^(4))^(1/3)


MT=1.7*1e7*2*1e30 #in Kg
G=6.674*1e-11 #N*m2/kg2  N=kgm/s2
v=60000 #m/s
r=(MT*G/np.square(v))/(3.086*1e16) #in m-> pc   20.42pc
P=(2*np.pi*(MT*G/np.square(v))/v )/( 3.154*1e7) #in s->yrs


GitHub DONE y hacer un par de modelos,
checar option modelvis in errorBarsByTPL2 DONE
volver a hacer errorBarsByTPL2 for LFlux corregir size *10 or 2** y cambiar color!!  DONE

#Jun12erased everything, pulled everything again
cd projects
git clone https://github.com/jwisbell/matisse_agn_scripts.git
cd matisse_agn_scripts/
git checkout modelfitting
git pull origin modelfitting

First, you need to switch to command mode. hit Esc key. Next, you can type the following commands:
:q to quit.
:q! to quit without saving data/file.
:x save and quit.
:qa to quit all open files.

jacob [1:55 PM]
git fetch origin
git reset --hard origin/modelfitting

`git fetch --all && git reset --hard origin/modelfitting`
#downloading all the rest of the  UT Apr data, i have:
MATIS.2019-04-16T04/49/18.051.fits
MATIS.2019-04-18T05/33/27.692.fits
from mcdb import img2vis #old way
from mcdb import model_fitter
#workd with from mcdb import img2vis from mcdb.img2vis import modelvis,modelcphase, modelbispec
#new way from directory projects
cd /Users/M51/projects/matisse_agn_scripts/modelfitting
add azimuthalAverage
ipython3
import model_fitter
s = 256 # num pixels to a side of square model image
bounds = [[0.9,0.9,-np.pi/2,0.05,0,0,.9,.9,-np.pi]\
          ,[s/2,s/2,np.pi/2,1.0,s,s,s/2,s/2,np.pi/2]]
pxscale = 0.5

#To change the Gaussian
#1st model
x0_rest = [15/pxscale , 12/pxscale, np.radians(30.0), 0.15, \
           110, 130, 4/pxscale, 4/pxscale, np.radians(125.0-180),\
           -1.0]
x0 = x0_rest
labels_rest = ['fwhmx', 'fwhmy', 'pa', 'amp'] + ['x','y', 'fwhmx', 'fwhmy', 'pa'] + ['logf']
labels=labels_rest
model_fitter.do_mcmc("/Users/M51/NGC1068Sept24DRScal/TARGET_CAL_INT_0001.fits",0.5,x0,modelfunc = two_gauss_one_centered,labels,n_components=2,method='vis')

#2nd model
x0_rest = [15/pxscale , 6.5/pxscale, np.radians(42.0), 0.15, \
           110, 130, 4/pxscale, 4/pxscale, np.radians(125.0-180),\
           -1.0]
x0 = x0_rest
labels_rest = ['fwhmx', 'fwhmy', 'pa', 'amp'] + ['x','y', 'fwhmx', 'fwhmy', 'pa'] + ['logf']
labels=labels_rest
model_fitter.do_mcmc("/Users/M51/NGC1068Sept24DRScal/TARGET_CAL_INT_0001.fits",0.5,x0,labels,n_components=2,method='vis')
#3rd model
x0_rest = [15/pxscale , 6.5/pxscale, np.radians(42.0), 0.15, \
           110, 110, 4/pxscale, 4/pxscale, np.radians(125.0-180),\
           15/pxscale, 6.5/pxscale, np.radians(135),\
           -1.0]
x0 = x0_rest
labels_rest = ['fwhmx', 'fwhmy', 'pa', 'amp'] + ['x','y', 'fwhmx', 'fwhmy', 'pa'] + ['fwhmx', 'fwhmy', 'pa']+['logf']
labels=labels_rest
model_fitter.do_mcmc("/Users/M51/NGC1068Sept24DRScal/TARGET_CAL_INT_0001.fits",0.5,x0,labels,n_components=2,method='vis')






#labels = ['x','y', 'fwhmx', 'fwhmy', 'pa', 'amp']*n_components + ['logf']
#amp=1.


#HERE not working
labels_rest = ['x','y','fwhmx', 'fwhmy', 'pa','amp']+ ['logf']
labels=labels_rest
x0_rest = [0,0,15/pxscale , 6.52/pxscale, np.radians(42.0), 1.0, -1.0]
#   4/pxscale, 4/pxscale, np.radians(125.0-180),
x0 = x0_rest
model_fitter.do_mcmc("/Users/M51/NGC1068Sept24DRScal/TARGET_CAL_INT_0001.fits",0.5,x0,labels,n_components=2,method='vis')




model_fitter.do_mcmc("/Users/M51/NGC1068Sept24DRScal/TARGET_CAL_INT_0001.fits",0.5,x0,labels,n_components=1,method='vis')


fname="/Users/M51/NGC1068Sept24DRScal/TARGET_CAL_INT_0001.fits"
model_fitter.do_mcmc(fname,pxscale,x0, labels, s=s, n_components=2, method='vis')
 [32, 34] is not in list

/usr/local/lib/python3.7/site-packages/emcee/ensemble.py:335: RuntimeWarning: invalid value encountered in subtract
    lnpdiff = (self.dim - 1.) * np.log(zz) + newlnprob - lnprob0
/usr/local/lib/python3.7/site-packages/emcee/ensemble.py:336: RuntimeWarning: invalid value encountered in greater
[ 2.99337488e+01  2.44128117e+01  3.51259817e-01 -1.18587067e-01
 1.09999831e+02  1.29870674e+02  8.42843790e+00  7.96925911e+00
 -1.10762129e+00 -1.08981418e+00] results
WARNING:root:Too few points to create valid contours
model_fitter.do_mcmc("/Users/M51/NGC1068Sept24DRScal/TARGET_CAL_INT_0001.fits",0.5,x0,labels,method='cphase')
WARNING:root:Too few points to create valid contours
[ 30.09899959  23.80142044   0.38922582  -0.18498362 109.90076606
 130.02770817   8.21273952   8.01820396  -0.4963971   -1.06362293] results
model_fitter.do_mcmc("/Users/M51/NGC1068Sept24DRScal/TARGET_CAL_INT_0001.fits",0.5,x0,labels,method='bispec')
WARNING:root:Too few points to create valid contours
[ 30.17089876  23.87567934   0.62344187   0.1450903  110.30277198
 129.9452873    7.83647837   8.05210751  -1.14342536  -1.26368359] results
model_fitter.check_fit("/Users/M51/NGC1068Sept24DRScal/TARGET_CAL_INT_0001.fits", 0.5, save=True)
[32, 34] is not in list

#LAST VERSION++++++++++++++++++++++Jun14th
import model_fitter
oifitsngc1068Sept24HD25286 = "/Users/M51/NGC1068Sept24DRScalHD25286/TARGET_CAL_INT_0001.fits"
oifitsngc1068Sept22delPhe = "/Users/M51/NGC1068Sept22DRScaldelPhe/TARGET_CAL_INT_0001.fits"
oifitsngc1068Sept22HD16658 = "/Users/M51/NGC1068Sept22HD16658/TARGET_CAL_INT_0001.fits"
oifitsngc1068Sept25HD20356 = "/Users/M51/NGC1068Sept25DRScalHD20356/TARGET_CAL_INT_0001.fits"
oifitsngc1068Sept25HD209240 = "/Users/M51/NGC1068Sept25HD209240/TARGET_CAL_INT_0001.fits"
delSco19may = "/Users/M51/modelfittingresults/delSco/delsco_19may.fits"

testclph="/Users/M51/modelfittingresults/ASPROtests/testsimple/GaussianPunct.fits"
alldates="/Users/M51/Downloads/TARGET_CAL_INT_0001.fits"

# pixel scale of image (mas / pixel)
pxscale = 1
# num pixels to a side of square model image,s:    the length in px of the model image sides (default=64px)
s = 256
fname = alldates#pk1068  HERE
#initial fitting guess
x0_rest = [25 , 18, -.772, 0.8, \
           s/2+10, s/2-10, 10, 8, np.radians(125.0-180),\
           -1.0]
x0_best =  [  7.60351634,   7.76163087,   0.,0.51198132,\
                240.14389472, 224.37750026,  16.72100893,   9.96399313,  -0.92684539,\
                -0.75273277]
#x0_rest = [4, 4, 0, 0.2,     30, np.radians(-60), 20, 15, np.radians(60),         -.75]
x0_one = [20, 20, 0.0, 0.5, -1.0]

#bounds: boundaries of fitted parameters for fitting
bounds_rest = [[0.9,0.9,-np.pi/2,0.05,     0,-np.pi,.9,.9,-np.pi] + [-10.0]\
    ,[s/2,s/2,np.pi/2,1.0,              s/2,np.pi,s/2,s/2,np.pi/2] + [1.0]]
#OUR   Have to be in degrees!! logf can be in [-10,1)
#now bounds2 =  [[0.9,0.9,-179,0.2,  0,-179,.9,.9,-179] + [-10.0],[s/2,s/2,179,1.0,     s/2,179,s/2,s/2,179] + [1.0]]
ourbounds_rest = [[0.9,0.9,-90,0.55,     3,-180,.9,.9,-180] + [-10.0]\
               ,[20,20,90,0.9,         s/4,180,20,20,90] + [1.0]]
labels2 = ['fwhmx_1', 'fwhmy_1', 'pa_1', 'amp'] + ['sep','sep_angle', 'fwhmx_2', 'fwhmy_2', 'pa_2'] + ['logf']
weights2 = np.array([12,12,11,11,    10,11,12,12,11,    11]) #sigmas of gaussian ball around initial guess
x0 = [3.282257, 6.859977, 1.2724, 0.74424, 30.74578, -1.47862, 20.2627556, 12.57016, 0.50497, -0.413009] #best fit
#with x0 = [2,7,40, 0.6,       30, -170, 15, 10, 30,  0]
bounds = ourbounds_rest
labels=labels2
weights=weights2
modelfunc = model_fitter.two_gauss_one_centered
model_fitter.do_mcmc(fname,pxscale,x0,bounds,modelfunc,labels, weights, s=s, method='mix',niter=1000,nwalkers=500,uniform_guess=True) #Fitting took 23.98 minutes Best Fit Values:
x0=[2.8605143279063743, 2.487028726947051, 0.3083350064172956, 0.734736669716154, 35.05892702506377, -2.5108591547552925, 19.68328395577154, 3.6509369037518637, 0.4872262041796586, -0.5052863025206997]
Js:x0 = np.load('most_probable_vals.npy')  in rad
(0.41559119*180)/np.pi
x0=[ 2.82919722,  7.58331964,  58 ,  0.75828249, 30.93793678,
       -85, 20.09215737, 13.6608034 ,  24, -0.76569389]
model_fitter.check_fit(fname, 0.5, labels2, save=True)
#... plot chains
model_fitter.check_fit(fname, 1.0, labels2, save=True)
os.chdir('..')
fname = "/Users/M51/NGC1068Sept24DRScalHD25286/TARGET_CAL_INT_0001.fits"
import model_fitter
model_fitter.do_mcmc(fname,pxscale,x0,bounds,modelfunc,labels, weights, s=s, method='mix',niter=1000,nwalkers=500,uniform_guess=True) #116min Fitting took 23.98 minutes Best Fit Values:
model_fitter.check_fit(fname, 1.0, labels2, save=True)


np.save('samples.npy', samples)
for m, b, d2 in samples[np.random.randint(len(samples), size=24)]:
    ys = m * xs + b
    pl.fill_between(xs, ys + np.sqrt(d2), ys - np.sqrt(d2), color="k",
                    alpha=0.1)
pl.savefig("results.png", transparent=True, dpi=300)

#J: In model_fitter the center wavelength is set in the function lnlike, and then the wavelength range is set in img2vis in __init__. I think it is dlam=0.2e-6 right now.

#J: Here is the final array:
x0 = [3.282257, 6.859977, 1.2724, 0.74424, 30.74578, -1.47862, 20.2627556, 12.57016, 0.50497, -0.413009]
#EXPERIMENTING:
x0 = [7, 7, 0, 1,     10,     0, 7, 7, 0, -0.413009]
x0 = [7, 7, 0, 1,     10,    45, 7, 7, 0, -0.3]

logf, scaling factor for the errors for the inputs recommended by emcee.10 A good fit should give 0, it shouldn't be above 1.

#J: bounds! Here they are:
bounds_rest = [[0.9,0.9,-np.pi/2,0.05, 0,-np.pi,.9,.9,-np.pi] ,[s/2,s/2,np.pi/2,1.0, s/2,np.pi,s/2,s/2,np.pi/2] + [-10.0]]
labels2 = ['fwhmx_1', 'fwhmy_1', 'pa_1', 'amp'] + ['sep','sep_angle', 'fwhmx_2', 'fwhmy_2', 'pa_2'] + ['logf']  #only for the plots!!
bounds2 =  [[0.9,0.9,-179,0.05,     0,-179,.9,.9,-179] + [-10.0]\
                ,[s/2,s/2,179,1.0,     s/2,179,s/2,s/2,179] + [1.0]]
weights2 = np.array([2,2,1,1,    10,1,2,2,1,    1]) #sigmas of gaussian ball around initial guess



x0_testsimple = [6,6,0,0.5,       10,30,6,6,0,   10]
labels2 = ['fwhmx_1', 'fwhmy_1', 'pa_1', 'amp'] + ['sep','sep_angle', 'fwhmx_2', 'fwhmy_2', 'pa_2'] + ['logf']
bounds2 =  [[0.9,0.9,-179,0.05,     0,-179,.9,.9,-179] + [-10.0]\
            ,[s/2,s/2,179,1.0,     s/2,179,s/2,s/2,179] + [1.0]] #define the 0 probability
#The weights ...   THERMAL ANNEALING
#short run with many walkers to get the priors
weights2 = np.array([2,2,1,1,    10,1,2,2,1,    1]) #sigmas of gaussian ball around
#Uniform_guess=True  in model_fitter.py  then weights not matter any more
x0=x0_testsimple
bounds = ourbounds_rest
labels=labels2
weights=weights2
modelfunc = model_fitter.two_gauss_one_centered
#-....-....-....
model_fitter.do_mcmc(fname,pxscale,x0,bounds,modelfunc,labels, weights, s=s, method='mix',niter=1000,nwalkers=400) #Took 1hr. with 200 walkers
RE-DO Sept24 con 1000inter,400 walkers
model_fitter.check_fit(fname, 0.5, labels2, save=True)
#ran for 3.6, 4.7, 3.8, 3.4   checkfit changing wl.
fname='/Users/M51/modelfitting/ASPROtests/test1/test1.fits'
x0 = [3, 3, 0, 1,     10, 0, 12, 12, 0, 5]
bounds_rest = [[0.9,0.9,-np.pi/2,0.05, 0,-np.pi,.9,.9,-np.pi] + [-10.0] ,[s/2,s/2,np.pi/2,1.0, s/2,np.pi,s/2,s/2,np.pi/2] + [1.0]]
labels_rest = ['fwhmx', 'fwhmy', 'pa', 'amp']  + ['logf'] + ['sep','sep_angle', 'fwhmx', 'fwhmy', 'pa'] + ['logf']



#BINARIES: deltaSco(both?) y gamPhe UTs



#line 244: ndim, nwalkers = len(x0), 30
#line 247  sampler.run_mcmc(pos, 100)  production run of 1000 steps
#Pg 13
#def do_mcmc(fname, pxscale,x0,labels, s=64, n_components=1, method='bispec'):
modelfunc=model_fitter.gauss2d#(x,y,x0,y0,fwhmx,fwhmy,pa, amp=1.)
bounds_rest = [[0,0,0.9,0.9,-np.pi/2,0.05] + [-1.0]]
x0 = [0,0,20, 20, 0.0, 0.5, -1.0]
labels_rest = ['x0','yo','fwhmx', 'fwhmy', 'pa', 'amp']
x0=x0_1
model_fitter.do_mcmc(fname,pxscale,x0,bounds,modelfunc,labels, s=s, method='mix',niter=1000,nwalkers=50)
# n_components: number of gaussian components to fit (default=1)
#    method: options are 'vis' to fit visibilities alone
#       'cphase' to fit closure phase alone (not recommended)
#       'bispec' to fit the bispectrum (default, best option)
#HERE:)

def check_fit(fname, pixelscale, save=True):
    ''' Make plots to check the quality of the fit.
        args are fname (filename), pixelscale (pixelscale in mas/px) and save.
        save defaults to True, but setting to false calls plt.show() instead of
        saving as a png
        '''
#To get the latest version, please do `git checkout modelfitting` then `git pull origin modelfitting`

#Jun 11 TFvis2tauDRS()
#joining Sept and Apr cals
Fin3=pd.read_pickle('TableDiamsAprilUTsjsdc17.pk')
Fin2=pd.read_pickle('TableDiamsSeptmsdfcc19.pk')
#Avoiding the bad mjds
df1=pd.read_pickle('Jun4superpAprCals.pk')#16,90
df2=pd.read_pickle('May31superpSeptCals.pk')#156,90
df=pd.concat([df1, df2], ignore_index=True)#172,90
mjds=[58590.05766807, 58590.05246034, 58381.04560605, 58381.05829687, 58381.11948348, 58381.15815322, 58382.12718112, 58382.13937325, 58382.14056971, 58382.14151070, 58382.14244882, 58382.14338623, 58382.14647846, 58383.13341208, 58383.13461323, 58383.13554598, 58383.13647738, 58383.13741494, 58383.13836445, 58383.13942234, 58383.14308607, 58383.14435676, 58383.14797656, 58383.15507120, 58383.20571736, 58384.07957074, 58384.17850135, 58384.98769468, 58385.00456004, 58385.38429503, 58385.38560493, 58386.03318726, 58386.07974087, 58386.08855096, 58386.10701578, 58386.11361284, 58386.11911919, 58386.18925231, 58386.19161547, 58386.19635368, 58386.20523466, 58386.40622892, 58386.4108629,  58385.35656910, 58385.36001311, 58385.36108241, 58385.36422807, 58385.36542380, 58589.97297293, 58591.16714752, 58591.17005860, 58590.24013409, 58590.24305453, 58383.04362422, 58383.04483624, 58383.04578132, 58383.04673491, 58383.04775873, 58383.07146526, 58383.07601124, 58383.07693048, 58383.08061932]  #Apr+Sept  last 9 bSgr, first2#3C 273
#for index, row in df2.iterrows():
for i in range(np.size(mjds)):
    df=df[df['index']!=mjds[i]]
    print(mjds[i])
    print(df.shape)  #120, 90

df=df.reset_index() #index was mjds  !!#for index, row in df2.iterrows():  for j in range(6):
df.to_pickle('superpSeptApr.pk') #index was mjds  !!
df=pd.read_pickle('/Users/M51/TablesPicklesScripts/superpSeptApr.pk')
correction=[1,1,1,1,1,1]
#2d mine
wu.TFvis2tauDRS(df,'LFlux','T0E','VIS2',0,135,1,2,0,correction,0,'scatter')
#2D W
wu.TFvis2tauDRS(df,'LFlux','T0E','VIS2',0,135,1,2,0,correction,0,None)
#color,x,y  L
wu.TFvis2tauDRS(df,'LFlux','T0E','VIS2',0,135,1,2,0,correction,0,'scatter2')

#change OO  and channel by j and title
wu.errorBarsByTPL(df,data='modvis',X='LFlux')
wu.errorBarsByTPL(df,data='modvis',X='tau0')
wu.errorBarsByTPL(df,data='VIS2',X='tau0')
wu.errorBarsByTPL(df,data='VIS2',X='LFlux')

#WORKING WELL, correcting by diam done well
wu.errorBarsByTPL(df,data='rawvis2',X='tau0')#prints raw and TFvis2


wu.errorBarsByTPL(df,data='VIS2',X='LFlux') #READY
#Change title, bcd channel and for in j for channel

#to do all channels and bcds oo ii at once saving
wu.errorBarsByTPL2(df,data='VIS2',X='LFlux',BCD1='OUT',BCD2='OUT')
wu.errorBarsByTPL2(df,data='VIS2',X='LFlux',BCD1='IN',BCD2='IN')
wu.errorBarsByTPL2(df,data='VIS2',X='tau0', BCD1='OUT',BCD2='OUT')
wu.errorBarsByTPL2(df,data='VIS2',X='tau0', BCD1='IN',BCD2='IN')
wu.errorBarsByTPL2(df,data='VIS2',X='mjd', BCD1='OUT',BCD2='OUT') #REDO!!
np.min(df['index'])
Out[66]: 58381.05417115
np.max(df['index'])
Out[65]: 58591.21621246
210 days
for i in range(df['targ'].shape[0]):
    print(df['UDDL'][i])
    for j in range(6):
        if(np.mean(df['VIS2'][i][j][51:103])<0):# [51:103]  110
            print(df['index'][i])
            print(df['tplstart'][i])  #2019-04-17T02:32:41
            print(df['targ'][i])  #[5:57] HD 97765 64 58590.11240974, 58590.10950099
            print(np.size(df['WL'][i]))

for i in range(df['targ'].shape[0]):print(df['targ'][i],df['name'][i])
for i in range(df['targ'].shape[0]):print(df['targ'][i],df['name'][i])


fits just vis or the bisoectra and phases, based in m2vis. Its modified, to get aspro, ews or drs. Either using curvefit or mcmc.
Run the model a number of times amd you look at the probabilities.
What to fit?? When you fit the comb of the 3 bls, by the complex 3-spectrum (amplitude is sensitive by the atmosp affects) whereas the cl ph are not.
Leo: VIS2 (6)and cl phases(4, as phases, not as complex nrs)  from DRS BOTH as a function of wl. divide the L band in 3 and maybe include M
So far it fits the tri-spectra (W) and the
3.6-3.8 FIRTST,
/Users/M51/PLOTS/EWSvsDRSnochannelcorr/AprilCals/vis2wl2.png

self-calibrar delPhe Sept23 and try models 2018-09-22T03:30:30 using cal_to_targ_converter.py
0.Measure errors substracting the model for each bl and wl!! :)
0. calibrar April cals and Sept cals in DRS and plot
1. model   TODAY@!!              Prezi presentation
2. hacer tabla DRS? agns
1.Selection mjds by quality tests for all the AGNs!!!!!
2.lista badmjds y hacer plots comparing both pplns
2.self-calibrate sept con drs

mat_est_aphase pg.63
OPCIONES: mismas plots para Abril DONE!! looks sad ...
Walter: ask bcd code, phase referencing technique

diff phase def in Millour 2016 Pg 12, how to use it?? compare to midi, is it the same as in Leos code?
Download our manual and check deffinitions!! MATISSE_pipeline_user_manual compare to book
self-cal DRS and plot
*Matter2016 talks about errors
*Lopez_9146-22 errors in cl ph
Look at QC DET
SEE VISs2 DATA AGNS  !!!! DONE  :(
PENDING using DRS using OPD=TRUE
#9 Jun
esorex mat_merge_results /Volumes/LaCie/Reduced/May25CalsSeptDRSv.1.4/mat_raw_estimates.2018-09-22T03_30_30.HAWAII-2RG.rb/test/cal3.sof
#[ INFO  ] esorex: Created product CALIB_RAW_INT_0001.fits (in place)
from mcdb import cal_to_targ_converter as ct
test=ct.Head_Info('CALIB_RAW_INT_0001.fits')
test.write_sim('testTARGET_RAW_INT_0001.fits')  # :D
esorex mat_cal_vis /Users/M51/cross-cal/cal1.sof
esorex mat_cal_cphase /Users/M51/cross-cal/cal1.sof
esorex mat_cal_dphase /Users/M51/cross-cal/cal1.sof
esorex mat_merge_results /Users/M51/cross-cal/cal2.sof
os.chdir('img2vis-master')
import img2vis as i2
bl,pa,u,v,vis,vis_noise, u1,u2,v1,v2,cphase,cphase_noise, bispec, bispec_noise, sta_indices=i2.read_matisse_oifits('/Users/M51/cross-cal/TARGET_CAL_INT_0001.fits',3,8,phot=False)

                                  
                                
                                  
i2.modelvis(u,v,vis,1,0)  NO
i2.modelcphase(u1,v1,u2,v2,phase,1,0, sta_index) NO
#Example: in test.py
## compute px scale
L_scale = np.float(m.split(".fits")[0].split("L")[1])
pxscale_pc = 0.04 * np.sqrt(L_scale)
mas_per_pc=1000/20
pxscale_mas = pxscale_pc * mas_per_pc
lam=1e-5
                                  #oifits="../MIDI_data/NGC1068_lopez-gonzaga2014.oifits"
                                  #phot=10.0
                                  #i=img2vis(m, pxscale_mas, lam, oifits=oifits, phot=phot)
                                  
                                  oifits = "../MIDI_data/Circinus_clean.oifits"
                                  i=img2vis(m, pxscale_mas, lam, oifits=oifits)
                                  i.optimize_pa(fixed_pa=44)
                                  i.make_plot()
#Example 2
 from img2vis import img2vis
pxscale_circ = 0.01333 * 1000/20 ### 0.08 pc per px, 1000 mas = 20 pc in Circinus
oifitscirc="../MIDI_data/Circinus_clean.oifits"
i=img2vis("../models/Bernd_2016-05-13/circinus_bild_10_25L0.111111.fits", pxscale_circ, 1e-5, oifits=oifitscirc)
i.optimize_pa()
i.make_plot() #These steps are covered by ASPRO
Data1UntitledFolder/PythonCode/CodeandDatafiles
FTinpython
#GOOD: ModelxExperiments.py
'''Simply start ipython and then import model_fitter from here the file path to the data, the pixelscale, the initial model guess, etc. can be specified. Then run do_mcmc(...) to do the fit. More info is found by calling help(do_mcmc).'''
oifits='/Users/M51/modelfitting/Aspro2_NGC_1068_MATISSE_LM_2_83876-4_13905-13ch_UT1-UT2-UT3-UT4_2019-09-18.fits'
import model_fitter
mf.do_mcmc()
                                  
 
                                  
                                  
#For DRS
#Selecting good mjds by eye
df=pd.read_pickle('/Users/M51/TablesPicklesScripts/May31stDRSSeptCals.pk') #484, includes IO and OI
anames=np.unique(df['name']) #21
#To do Quality Control 1: vis by eye
df2=df[df['name']=='HD 16658']
wu.VIS2wldrscal(df2,'WL',50,100,0,130)
for i in range(np.size(anames)):
    df2=df[df['name']==anames[i]]
    df2=df2.reset_index()
    plt.clf()
    wu.VIS2wldrscal(df2,'WL',50,100,0,130)  #HERE
#HD 97765 has only L band(64 wls), code takes care of that

#May24ewsAprCalsUTs.pk
os.chdir('TablesPicklesScripts')
df=pd.read_pickle('Jun4DRSAprCals.pk')
df2=pd.read_pickle('May24ewsAprCalsUTs.pk')
df['mjdo']=pd.to_numeric(df['mjdo'])
type(df['mjdo'][0])
#make sure that both mjds have the same number of decimal digits
DF=pd.concat([df2.set_index('mjd-obs'),df.set_index('mjdo')], axis=1, join='inner').reset_index()  #16,90  16 of 18 :)
DF.shape #( keeps only ii and oo, cause only those match with EWS
DF.to_pickle('Jun4superpAprCals.pk') #in M51!!
#Doing the comparisson of pplns
#Look at calibrated vis with wl   for all AGNs     :(
from astropy.io import fits  #PENDING: plot errors, plot only selected mjds!!!
hdu=fits.open('/Users/M51/NGC1068Sept22DRScaldelPhe/TARGET_CAL_INT_0001.fits')
hdu=fits.open('/Users/M51/NGC1068Sept24DRScal/TARGET_CAL_INT_0001.fits')
hdu=fits.open('/Users/M51/NGC1068Sept25DRScal/TARGET_CAL_INT_0001.fits')
hdu=fits.open('/Users/M51/NGC22SeptDRScal/TARGET_CAL_INT_0001.fits')
hdu=fits.open('/Users/M51/NGC1365Sept24DRScal/TARGET_CAL_INT_0001.fits')
hdu=fits.open('/Users/M51/NGC1566Sept23DRScal/TARGET_CAL_INT_0001.fits')
hdu=fits.open('/Users/M51/NGC7469Sept25DRScal/TARGET_CAL_INT_0001.fits')
hdu=fits.open('/Users/M51/modelfittingresults/delSco/delsco_19may.fits')
hdu=fits.open('/Users/M51/modelfittingresults/delSco/delsco_20may.fits')
                                  
hdu=fits.open('/Users/M51/NGC1068Sept24DRScalHD25286/TARGET_CAL_INT_000
hdu=fits.open("/Users/M51/NGC1068Sept22DRScaldelPhe/TARGET_CAL_INT_0001September22.fits")
hdu=fits.open("/Users/M51/NGC1068Sept25DRScalHD20356/TARGET_CAL_INT_0001September25.fits")
hdu=fits.open("/Users/M51/NGC1068Sept24DRScalHD25286/TARGET_CAL_INT_0001September24.fits")
    
hdu=fits.open('/Users/M51/NGC1068Sept22HD16658/TARGET_CAL_INT_0001.fits')
hdu=fits.open('/Users/M51/NGC1068Sept25HD209240/TARGET_CAL_INT_0001.fits')

n=hdu[5].data.shape[0]  #for cl ph
cmap = plt.get_cmap('brg')#('plasma')
cmap = plt.get_cmap('plasma')
#Novdata
hdu=fits.open('/Users/M51/Dropbox/MATISSEfiles/dataNGC1068paper/NovemberDataDRS/1TARGET_CAL_INT_0001.fits')
hdu=fits.open('/Users/M51/Downloads/NovemberDataDRS/1TARGET_CAL_INT_0002.fits')
hdu=fits.open('/Users/M51/Dropbox/MATISSEfiles/dataNGC1068paper/NovemberDataDRS/2TARGET_CAL_INT_0001.fits')
hdu=fits.open('/Users/M51/Downloads/NovemberDataDRS/2TARGET_CAL_INT_0002.fits')
hdu=fits.open('/Users/M51/Dropbox/MATISSEfiles/dataNGC1068paper/NovemberDataDRS/3TARGET_CAL_INT_0001.fits')#oo
hdu=fits.open('/Users/M51/Downloads/NovemberDataDRS/3TARGET_CAL_INT_0002.fits')#ii
#sept fpr paper
hdu=fits.open('/Users/M51/Dropbox/MATISSEfiles/dataNGC1068paper/NGC1068CalibratedWithSelectedFiles/S22HD16658selected/TARGET_CAL_INT_0002.fits')#3ii
hdu=fits.open('/Users/M51/Dropbox/MATISSEfiles/dataNGC1068paper/NGC1068CalibratedWithSelectedFiles/S22HD16658selected/TARGET_CAL_INT_0001.fits')#4oo
hdu=fits.open('/Users/M51/Dropbox/MATISSEfiles/dataNGC1068paper/NGC1068CalibratedWithSelectedFiles/S25hd209240allgood/TARGET_CAL_INT_0002.fits')#ii
hdu=fits.open('/Users/M51/Dropbox/MATISSEfiles/dataNGC1068paper/NGC1068CalibratedWithSelectedFiles/S25hd209240allgood/TARGET_CAL_INT_0001.fits')#oo
n=hdu[5].data.shape[0]  #for cl ph
n
cmap = plt.get_cmap('brg')#('plasma') 1
cmap = plt.get_cmap('plasma')  #2
#plotclph

for j in range(n):
    if(j%4==3):
        t=j%4  #j%4 for clph, 6 for vis2
        c = cmap(float(t)/4)  #/4 for clph, /6 for vis2
        plt.plot(hdu[3].data['EFF_WAVE']*1e6, hdu[5].data['T3PHI'][j],color=c, label=hdu[5].data['STA_INDEX'][j])#DEG  no error
        #plt.errorbar(hdu[3].data['EFF_WAVE']*1e6, hdu[5].data['T3PHI'][j], yerr=hdu[5].data['T3PHIERR'][j],color=c, label=hdu[5].data['STA_INDEX'][j])  #deg with errors
        #plt.plot(hdu[3].data['EFF_WAVE']*1e6,hdu[4].data['VIS2DATA'][j],color=c) #VIS2  No err
        #plt.errorbar(hdu[3].data['EFF_WAVE']*1e6,hdu[4].data['VIS2DATA'][j],yerr=hdu[4].data['VIS2ERR'][j],color=c, label=hdu[4].data['STA_INDEX'][j]) #VIS2 with err

        if(j<4):plt.legend() #j<4 for clph, 6 for vis2


plt.grid(linestyle="--",linewidth=0.1,color='.25')
#plt.minorticks_on()
plt.grid(b=True, which='minor', color='0.25', linestyle='--', alpha=0.2)
#plt.title('NGC 1068 closure phases 25 Sept using HD 20356')
plt.ylabel('T3PHI [deg]')
#plt.title('NGC 1068 VIS2 24 Sept using HD 25286')
#plt.title('September 22th OO')
#plt.title('September 25th OO')
#plt.title('September 25th OO/II')
plt.title('November 6th a) OO')
#plt.ylabel('VIS2')
plt.xlabel('wl [um]')
plt.ylim(-180,180)

plt.ylim(-230,230)
#plt.ylim(0,0.4)  #for vis2
plt.savefig("25Septclph-2.png", dpi=1000)
              
#plotvis2    APRIL Cals faint UTs
hdu=fits.open('/Volumes/MD/Reduced/Jun4CalsAprilUTs/mat_raw_estimates.2019-04-17T00_12_18.HAWAII-2RG.rb/RAW_VIS2_0003.fits')
hdu=fits.open('/Volumes/MD/Reduced/Jun4CalsAprilUTs/mat_raw_estimates.2019-04-17T00_12_18.HAWAII-2RG.rb/CALIB_RAW_INT_0004.fits') #HD 62753 ++ DIL NAME HIGH
hdu=fits.open('/Volumes/MD/Reduced/Jun4CalsAprilUTs/mat_raw_estimates.2019-04-17T00_49_14.HAWAII-2RG.rb/CALIB_RAW_INT_0002.fits') #HD112091  ++ DIL NAME HIGH
 hdu=fits.open('/Volumes/MD/Reduced/Jun4CalsAprilUTs/mat_raw_estimates.2019-04-16T23_14_47.HAWAII-2RG.rb/CALIB_RAW_INT_0001.fits') LamPyx GOOD
hdu=fits.open('/Volumes/MD/Reduced/Jun4CalsAprilUTs/mat_raw_estimates.2019-04-17T01_11_57.HAWAII-2RG.rb/CALIB_RAW_INT_0002.fits') #3C273 neg Mband
hdu=fits.open('/Volumes/MD/Reduced/Jun4CalsAprilUTs/mat_raw_estimates.2019-04-17T01_36_02.HAWAII-2RG.rb/CALIB_RAW_INT_0003.fits')#3C273noiseL
hdu=fits.open('/Volumes/MD/Reduced/Jun4CalsAprilUTs/mat_raw_estimates.2019-04-17T02_57_00.HAWAII-2RG.rb/CALIB_RAW_INT_0002.fits')#HD 97765 L band only noise?
hdu=fits.open('/Volumes/MD/Reduced/Jun4CalsAprilUTs/mat_raw_estimates.2019-04-17T03_42_28.HAWAII-2RG.rb/CALIB_RAW_INT_0004.fits')#61 Virg LHIGH good
hdu=fits.open('/Volumes/MD/Reduced/Jun4CalsAprilUTs/mat_raw_estimates.2019-04-17T04_11_56.HAWAII-2RG.rb/CALIB_RAW_INT_0001.fits')#HD98591 LhighGood
hdu=fits.open('/Volumes/MD/Reduced/Jun4CalsAprilUTs/mat_raw_estimates.2019-04-17T04_45_35.HAWAII-2RG.rb/CALIB_RAW_INT_0003.fits')#HD101517LHIGH ++
hdu=fits.open('/Volumes/MD/Reduced/Jun4CalsAprilUTs/mat_raw_estimates.2019-04-17T05_31_00.HAWAII-2RG.rb/CALIB_RAW_INT_0004.fits')#HD150071negM
hdu=fits.open('/Volumes/MD/Reduced/Jun4CalsAprilUTs/mat_raw_estimates.2019-04-17T05_40_53.HAWAII-2RG.rb/CALIB_RAW_INT_0001.fits')#HD150071negM
hdu=fits.open('/Volumes/MD/Reduced/Jun4CalsAprilUTs/mat_raw_estimates.2019-04-18T01_22_04.HAWAII-2RG.rb/CALIB_RAW_INT_0001.fits')#LamPyxGood
hdu=fits.open('/Volumes/MD/Reduced/Jun4CalsAprilUTs/mat_raw_estimates.2019-04-18T03_54_24.HAWAII-2RG.rb/CALIB_RAW_INT_0001.fits')#HD97705negjump
hdu=fits.open('/Volumes/MD/Reduced/Jun4CalsAprilUTs/mat_raw_estimates.2019-04-18T05_02_02.HAWAII-2RG.rb/CALIB_RAW_INT_0001.fits')#HD150071negM
hdu=fits.open('/Volumes/MD/Reduced/Jun4CalsAprilUTs/mat_raw_estimates.2019-04-18T05_12_38.HAWAII-2RG.rb/CALIB_RAW_INT_0001.fits')#HD 101517++HIGHL

cmap = plt.get_cmap('plasma')
n=hdu[4].data.shape[0]  #for vis2
for j in range(n):
    t=j%6  #j%4 for clph, 6 for vis2
    c = cmap(float(t)/6)  #/4 for clph, /6 for vis2
    #plt.plot(hdu[3].data['EFF_WAVE']*1e6,hdu[4].data['VIS2DATA'][j],color=c) #VIS2  No err
    plt.errorbar(hdu[3].data['EFF_WAVE']*1e6,hdu[4].data['VIS2DATA'][j],yerr=hdu[4].data['VIS2ERR'][j],color=c, label=hdu[4].data['STA_INDEX'][j]) #VIS2 with err
    if(j<6):plt.legend() #j<4 for clph, 6 for vis2

plt.grid(linestyle="--",linewidth=0.1,color='.25')
#plt.minorticks_on()
plt.grid(b=True, which='minor', color='0.25', linestyle='--', alpha=0.2)
#plt.title('NGC 1068 VIS2 24 Sept using HD 25286')
targ=hdu[0].header['HIERARCH ESO OBS TARG NAME']
plt.title(targ)
plt.ylabel('VIS2')
plt.xlabel('wl [um]')
plt.ylim(-1,1.2)  #for vis2
#plt.savefig("25Septclph-2.png", dpi=1000)

              
              plt.show()

#New function to make table of calibrated data
wu.CalDatatableDRS('/Users/M51/NGC106822SeptDRScal')
df.to_pickle('NGC1068DRSSept22')



wu.corrvisY(DF,0,135,'VIS2','BL') #
wu.corrVIS2Y(DF,0,135,'VIS2','WL')  #only DRS
wu.corrvisY(DF,0,135,'VIS2','tau')
wu.corrvisY(DF,0,135,'VIS2','diam') #mjd
wu.errorBarsByTarget(DF)#To include errorbars
Haver rutina para seleccion de mjds, rehacer para TODO, en especial para las AGNS, para despues calibrarlas
error tiene que tomar en cuenta la resolucion esperada!
next Email:  include calvis and cavisavg with error bars  !!!
Comparacion pipelines  Calibradores, AGNs tambien :(
ChecK!! There is some symmetry in 84Aqr calib/
Leave out resolved sources or quantify error taking this into account
wu.table2D(df,50,100,Fin2)  change size for diameter,
make plots like /Users/M51/Desktop/modelvis2vsBLHD209042.png
#plot phot
#Look at chrom phases Sept cals
#June 4
#Routine to make the arrays to calibrate, self-cal the Cals, and calibrate AGNs

#Diccionario de visibilidades, el error tiene que tomar en cuenta la resolucion esperada!!   DONE
wu.modvisdict(df['wave'][0],35,135,0.25,4.0,plotbl=None)
wu.modvisdict(df['wave'][0],35,135,0.25,4.0,plotbl=True)


#Jun3
df=pd.read_pickle('scSeptCals.pk') #only selected mjds and self-calibrated
Fin2=pd.read_pickle('/Users/M51/TableDiamsSeptmsdfcc19.pk')
wu.table2D(df,50,100,Fin2)  #18,49
#1.changed std to divide by to n-1 and re-do errvis.png. Used stdev from ststistics  in table2D and TFvisEWSflux  DONE
corrvis=[1,1,1,1,1,1]  #No channel correction needed since it is for calibrated data
df1=pd.read_pickle('HD33162sc.pk')
wu.modvisews(df1,Fin2)  #CORRECT!!!!  spatfreq[j] was taking only 1 val
#2.Check for symmetries and  why there is no error bar at 1?? DONE
wu.TFvisEWSflux(df,0,130,'diam',corrvis, tableDiams=Fin2,yaxis='calvis')  #DONE




June 4th
#section #FOR AGNS Quality control
1.CHECK!!!!#Symmetries in 84Aqr, HD 195677, HD6290, HD 183925, 47 Cap,  and V4026Sgr
#Make table with list of tables!!
#REFILL all calibrated tables and TFs with new code, compare to old plots!!!
Re-do eye qual tests


#2.Download STARTED!! :)  DONE
#and Reduce April DRS// STARTED-DONE
#2 Install and re-reduce EWS after DRS STARTING
#3.Re- do plots using wu.TFvisEWSflux(df,0,130,'diam',corrvis, tableDiams=Fin2,yaxis='calvisavg',yerr=True)   #HD 172453 and HD 16658
wu.TFvisEWSflux(df,0,130,'diam',corrvis,tableDiams=Fin2,yaxis='calvisavg',yerr=True)   #HD 172453 and HD 16658
#6.CALIBRATE AGNs WITH CHOSEN CALIBRATORS in both, DRS and EWS
#7.CROSSCALIBRATING APRIL DATA
#8.Models!!
#4.Making tables for AGNs/
#5.include mjd in  wu.corrvisY(DF,0,135,'VIS2','diam') mjd








wu.errorBarsByTarget(DF)#To include errorbars
df=wu.tableDRSCalRIVIS2('/Volumes/LaCie/Reduced/May27AGNSeptDRSv1.4/NGC1068')  #47,72
df.to_pickle('DRSNGC1068Sept.pk')
DF=pd.read_pickle('DRSNGC1068Sept.pk')

abadmjd=[
for i in range(np.size(abadmjd)):  #155??
    DF=DF[DF['index']!=abadmjd[i]]
    print(abadmjd[i])
    print(DF.shape)  #97,90
         
DF=DF.reset_index()
         
df=wu.tableDRSCalRIVIS2('/Volumes/LaCie/Reduced/May27AGNSeptDRSv1.4/NGC1365')  #14,72
df.to_pickle('DRSNGC1365Sept.pk')
df=wu.tableDRSCalRIVIS2('/Volumes/LaCie/Reduced/May27AGNSeptDRSv1.4/NGC1566')  #6,72
df.to_pickle('DRSNGC1566Sept.pk')
df=wu.tableDRSCalRIVIS2('/Volumes/LaCie/Reduced/May27AGNSeptDRSv1.4/NGC7469')
df.to_pickle('DRSNGC7469Sept.pk')  #14,72

#May 31st  COMPARING BOTH PIPELINES  redoing Jun4
#Check std /n-1
#not able to add April data.  Are there more UTs data July?? More April??
#mjd=format(hducal[0].header["MJD-OBS"],'.8f')
df=wu.tableDRSCalRIVIS2('/Volumes/LaCie/Reduced/May25CalsSeptDRSv.1.4')
df.shape  #(484, 72)
df.to_pickle('May31stDRSSeptCals.pk')
DF1=pd.read_pickle('/Users/M51/May31stDRSSeptCals.pk')  #484
DF2=pd.read_pickle('/Users/M51/May14SeptCalsEWS.pk')  #166
df['mjdo']=pd.to_numeric(df['mjdo'])
type(df['mjdo'][0])
#make sure that both mjds have the same number of decimal digits
DF=pd.concat([DF2.set_index('mjd-obs'),DF1.set_index('mjdo')], axis=1, join='inner').reset_index()  #156,90
DF.shape #( keeps only ii and oo, cause only those match with EWS
DF.to_pickle('May31superpSeptCals.pk') #in M51!!
#PLOTS DRS VS EWS--**--**--**--**--**
os.chdir('TablesPicklesScripts')
DF=pd.read_pickle('May31superpSeptCals.pk')  #156,90 ->LOST 10 ONLY :D
#Only good mjds:  #bSgr 50 has 118 wls!!! completely left out
df=DF[DF['targ']=='HD 33162']  #mjd 7 digits 58386.4108629
#SeptCals
abadmjd=[58382.12718112, 58382.13937325, 58382.14056971, 58382.14151070, 58382.14244882, 58382.14338623, 58382.14647846, 58384.07957074, 58381.04560605, 58381.05829687, 58381.11948348, 58381.15815322, 58383.13341208, 58383.13461323, 58383.13554598, 58383.13647738, 58383.13741494, 58383.13836445, 58383.13942234, 58383.14308607, 58383.14435676, 58383.14797656, 58383.15507120, 58384.31596017, 58384.31937302, 58384.32040372, 58384.32355232, 58383.13341208, 58383.13461323, 58383.13554598, 58383.13647738, 58383.13741494, 58383.13836445, 58383.13942234, 58383.14308607, 58383.14435676, 58383.14797656, 58383.14981545, 58383.15075302, 58383.15213772, 58383.15307554, 58383.15401894, 58383.15507120, 58383.20571736, 58384.17850135, 58384.98769468, 58385.00456004, 58385.00903409, 58385.38429503, 58385.38560493, 58386.03318726, 58386.07974087, 58386.08855096, 58386.10701578, 58386.11361284, 58386.11911919, 58386.18925231, 58386.19161547, 58386.19635368, 58386.20523466, 58385.35656910, 58385.36001311, 58385.36108241, 58385.36422807, 58385.36542380, 58386.25508756, 58386.27472541, 58386.40622892, 58386.4108629, 58383.04362422, 58383.04483624, 58383.04578132, 58383.04673491, 58383.04775873, 58383.07146526, 58383.07601124, 58383.07693048, 58383.08061932, 58386.02775214, 58386.03201258]
#April Cals
abadmjd=[58589.97297293, 58591.16714752, 58591.17005860, 58590.24013409, 58590.24305453, 58590.05766807, 58590.05246034]
#only faint
abadmjd=[58589.97297293, 58591.16714752, 58591.17005860, 58590.24013409, 58590.24305453, 58590.05766807, 58590.05246034, 58589.97585821, 58591.06268634, 58591.05981072]
         #last2 3c273
for i in range(np.size(abadmjd)):  #155??
    DF=DF[DF['index']!=abadmjd[i]]
    print(abadmjd[i])
    print(DF.shape)  #97,90   9,90

DF=DF.reset_index()
DF['UDDL'][48]=0.5   #CD-55 855: 48-54   does not work!!!! Not included
wu.corrvis11(DF,0,135,'VISAMP','diam') #function CORRECTED in Jun4. No channel correction
wu.corrvis11(DF,0,135,'VISAMP','tau')
wu.corrvis11(DF,0,135,'VIS2','diam')
wu.corrvis11(DF,0,135,'VIS2','tau')
wu.visvis2(DF,'DRS')
wu.visvis2(DF,'EWS')


#Look at vis with wl
#por que vis2neg in bl a no se ven en vis2wl ,
#targ es HD 97 765 tenia t~8.5
#
wu.corrvisY(DF,0,135,'VISAMP','BL')
wu.corrvisY(DF,0,135,'VISAMP','WL')
DF['VISAMP'][75].shape  #HD 172453 (6, 327)  !!!!  and 77
wu.corrvisY(DF,0,135,'VISAMP','tau')
wu.corrvisY(DF,0,135,'VISAMP','diam')
wu.corrvisY(DF,0,135,'VIS2','BL')
wu.corrvisY(DF,0,135,'VIS2','WL')
wu.corrvisY(DF,0,135,'sqrt','WL')  #<-----NEW
wu.corrvisY(DF,0,135,'VIS2','tau')
wu.corrvisY(DF,0,135,'sqrt','tau')  #<-----NEW
wu.corrvisY(DF,0,135,'VIS2','diam')
wu.corrvisY(DF,0,135,'sqrt','diam') #<-----NEW
wu.corrvisY(DF,0,135,'VIS2','mjd') #<-----NEW
wu.corrvisY(DF,0,135,'sqrt','mjd') #<-----NEW
         #using .1mas diam for CD-55 855
#wu.corrvisY(DF,0,135,'VIS2','diam')
wu.errorBarsByTarget(DF,'VIS2')#To include errorbars
#wu.corrvisY(DF,0,135,'sqrt','diam') #<-----NEW
wu.errorBarsByTarget(DF,'sqrt')#To include errorbars PENDING!!!


anames=['47 Cap', '84 Aqr', 'CD-55 855', 'HD 16658', 'HD 172453', 'HD 183925', 'HD 195677', 'HD 20356', 'HD 29246', 'HD 6290', 'HD14823', 'HD209240', 'HD25286', 'HD33162', 'V4026 Sgr', 'sig Cap']
DRSstd=[0.08006266619843484, 0.07729911689057035, 1.091696555448394e+26, 0.080855017423361, 0.06318619105293355, 0.08705272564402355, 0.11165605310429533, 0.0901650932129324, 0.0832597694312303, 0.06390830585067846, 0.08653509391263062, 0.07970972908831706, 0.09354342278478908, 0.10259142301147851, 0.08641359938752133, 0.07536004104564437]
EWSstd=[0.0572499987607066, 0.10009899991212921, 7.210408480928431e+25, 0.1062913044510037, 0.0998955221769378, 0.09807117937428167, 0.11973682679484561, 0.072324449064323, 0.10050803952843075, 0.08882150165120081, 0.1005334635602377, 0.08818833319894627, 0.07778110984289215, 0.07792956517747306, 0.08479017740554753, 0.09317733538456191]
#WITH CORRECTION IN visamps  and including CD-55 855
DRSstd=[0.08921329218877523, 0.06659371964319867, 0.11514642656639083, 0.07457313177853417, 0.05605246852409594, 0.07945097764645209, 0.1180862342280175, 0.08088492348727246, 0.06984958901464429, 0.06256736456831437, 0.07175344136506442, 0.06987357741533935, 0.07472454431699077, 0.08400241854241802, 0.09429518384630875, 0.07197869752872049]
EWSstd=[0.07374722360107376, 0.08559697355650546, 0.16105335679015534, 0.1020309331914704, 0.09133619317680712, 0.09228669196428241, 0.13930985551344413, 0.07407750016256011, 0.08572657673821328, 0.08519115690800343, 0.08063401014926273, 0.08257781676410121, 0.0672012960707044, 0.0814197667696257, 0.08674421051092887, 0.09473233166836094]
for i in range(np.size(DRSstd)):
    print(EWSstd[i]-DRSstd[i]) #before: +-.025 Now: .046  but it is because CD-55 855 is bad.
         -0.01546606858770147
         0.01900325391330679
         0.045906930223764505
         0.02745780141293623
         0.03528372465271118
         0.012835714317830316
         0.021223621285426625
         -0.006807423324712347
         0.01587698772356899
         0.02262379233968906
         0.008880568784198309
         0.012704239348761859
         -0.0075232482462863765
         -0.0025826517727923215
         -0.007550973335379879
         0.022753634139640447
DRSmd=[0.2603373516465553, 0.3677837802119522, 2.8119061519137877e+25, 0.3184723424311813, 0.31978432593573874, 0.33650025009970724, 0.22994485246723403, 0.30712710132268406, 0.3835322679056103, 0.30622598932442124, 0.39607144768588665, 0.35239135203596433, 0.4082527537342794, 0.31872938209708246, 0.32180365930380295, 0.28755030683849936]
EWSmd=[0.17423471698167653, 0.3121449487859636, 2.2382442263875227e+25, 0.2661740542310481, 0.27495809388511355, 0.2686988960325718, 0.19672695516054026, 0.2572018968685976, 0.31778269061290165, 0.24948720244477904, 0.36917223156496726, 0.3058656512863521, 0.3609378476561465, 0.2501795727850632, 0.22698666175075333, 0.2420820707100451]
         #WITH CORRECTION IN visamps  and including CD-55 855
DRSmd=[0.449499786533174, 0.5984755024656346, 0.48733982122614333, 0.5540473784363316, 0.5544150568108515, 0.5632779666054749, 0.455214961243255, 0.5315412894835132, 0.6100770554638564, 0.5332289385075301, 0.6202963201065245, 0.5854842973143322, 0.6309412709987899, 0.51226327776584, 0.5154498119548465, 0.5113620055120796]
EWSmd=[0.4037175456925735, 0.5453280511913862, 0.47056784432211124, 0.49178226358390886, 0.5048738756156647, 0.5024525750354968, 0.4101312254980165, 0.493827788668813, 0.5495236769121982, 0.4820059615374206, 0.5953915460934489, 0.5407956375599766, 0.5895775914051495, 0.48702411086899616, 0.46119149003712423, 0.47529610314291243]
for i in range(np.size(DRSmd)):
    print(EWSmd[i]-DRSmd[i])  #at most -.095, at least .026
         Now: All negative from .016 to .062
         
         -0.04578224084060051
         -0.05314745127424847
         -0.016771976904032093
         -0.0622651148524227
         -0.0495411811951868
         -0.06082539156997813
         -0.045083735745238496
         -0.03771350081470021
         -0.06055337855165821
         -0.05122297697010947
         -0.02490477401307556
         -0.04468865975435554
         -0.0413636795936404
         -0.025239166896843857
         -0.054258321917722285
         -0.036065902369167124
wu.corrvisY(DF,0,135,'VISPHI','WL')
#Their deff of differential phase s different!!!  Pg 18 MAN
#COMPARING CL>PHS    PENDING
#APPLYING CHANNEL CORRECTION with Philippe's code PENDING
         #END COMPARING BOTH PIPELINES ***************************
#INDEX OF FUNCTIONS see file indexfunctions
dir(wu)    #line 908
f(x, A, B) #this is your 'straight line' y=f(x)
tableDiam
checkFiles
#poscat2=pd.read_pickle('posmsdfcc.pk') # These pickles contain ra and decs for each catalog in degrees
#poscat1=pd.read_pickle('posjsdc_2017.pk')
#cat are poscat1 or poscat2, it only needs the columns with ra and dec
search_cat(cat,ra,dec):  #send cat as data frame
positionscat(cat):  #To run once for each catalog, send it as pandas data frame , saved to pickles

#DF has to be a superp, VISAMP or VIS2, diam or tau.
corrvis11(DF,bl1,bl2,DRSvis,color)
#to VISAMP AND VIS2 seem to not to be so related
visvis2(DF,'DRS')
'TFcf11',
corrvisY(DF,0,135,'VISAMP','WL') #VIS2, BL,tau,diam, WL
#To include error after wu.corrvisY(DF,0,135,'VIS2','diam')
wu.errorBarsByTarget(DF)
viswl
'TFcfbl',
'TFcfflux',
'TFcfmjd',
'TFcftau',

'TFvis2tauDRS',
'TFcorflux2tauDRS',
#To make a table with all the reduced data from CAL_RAW_INT, icf to get INCCORRFLUX
tableDRSCalRI(inDIRvis2,inDIRcf,icf)
#To make a table with all the reduced data from RAW_VIS2,DPHASE and CPHASE
tableDRSCalRIVIS2(inDIRvis2)
#Maybe it was initially meant to be the above function
tableDRSRawV2
arraytocal
#To make a table from reduced data
tableEWS(inDIR)
#To take out a bad template from the superp.pk
omitTemplate(DF,DRS,EWS)
#to plot corrected visibility with wl, BOTH
corrviswl(DF,bl1,bl2)
#to plot corrected visibility with bl, BOTH
corrvisbl(DF,bl1,bl2)
#to model the visibilities of the 6bls for the observations in DF
modvisews(DF,tableDiams)
modvis(DF)  #to plot the ideal visibility with wl  BOTH
modvisbl(DF)  #to plot the ideal visibility with bl  BOTH
cfluxbl(DF,bl1,bl2,a)  # BOTH
viswlewsagn
viswlewsagn2
viswlewsagnold
viswlewscal
#To plot calibrated or non-cal vis,chromph or closure phases.
#To plt them also averaged  and with errors for the diff bls
#to get pandas data frame with the calvis,calchrph and calclph
visewsagn

'TFvisEWSbl',
'TFvisEWSflux',
'TFvisEWSmjd',
'TFvisEWStau',
'TFcftauEWS'
TFcftauEWS2nd
'TFvistauEWS'
'TFphotEWSflux',
'TFcfEWSbl',
'TFcfEWSflux',
'TFcfEWSfluxlim',
'TFcfEWSmjd',
'TFcfEWStau',
#May27-30
#I have:
/Users/M51/TableDiamsAprilUTsjsdc17.pk
#Dir where the EWS reduced files are. Returns a sub-table of the diameter -catalogs. Set up which one to return.
df=wu.tableDiam('/Volumes/LaCie/SeptemberData')

CD-55 855 63.856969 -55.63668
CD-55 855 63.856969 -55.63668
954
954
18
18
{'HR  8394                         ', 'HIP97351                         ', 'HD193150                         '}
{'HR  8394                         ', 'HIP97351                         ', 'HD193150


#PROCESS:
df.to_pickle('TDSept.pk')
#BEFORE ALL THIS MJDs are selected by eye in SELECTION CALIBRATORS
#self calibration CALIBRATING CALIBRATORS
Fin2=pd.read_pickle('/Users/M51/TableDiamsSeptmsdfcc19.pk')
vii1=[1,1,1,1,1,1]
voo1=[1,1,1,1,1,1]
DF2=pd.read_pickle('/Users/M51/May14SeptCalsEWS.pk')
#HD 183925'---------------------
df=DF2[DF2['targ']=='HD 183925']   #6,19
df=df.reset_index()
badmjd=[58381.04560605, 58381.05829687]
for i in range(np.size(badmjd)):
    df=df[df['mjd-obs']!=badmjd[i]]
    print(badmjd[i])
    print(df.shape)  #4,20

df=df.reset_index()
vii,voo=wu.arraytocalews(df,Fin2)
#wl1,wl2,bl1,bl2,ang1[deg],ang2[deg]  plots only between i1 and i2 #9ischopped
#np.mean(dat),np.var(dat),np.max(dat)-np.mean(dat)
wu.viswlewsagn(df,50,100,0,130,0,135,vii1,voo1,0,4,toplot='vis') #5
wu.viswlewsagn(df,50,100,0,130,0,135,vii,voo,0,4,toplot='vis')#5
#returns dataframe.          #DFtemp=DFtemp.reset_index()  #comment if you sent the good mjds only
HD183925sc =  wu.viswlewsagn(df,50,100,0,130,0,135,vii,voo,0,4)
HD183925sc.to_pickle('HD183925sc.pk')
#HD 6290--------------------
df=DF2[DF2['targ']=='HD 6290']  #5
df=df.reset_index()
badmjd=[58381.11948348]
for i in range(np.size(badmjd)):
    df=df[df['mjd-obs']!=badmjd[i]]
    print(badmjd[i])
    print(df.shape)

df=df.reset_index() #4
vii,voo=wu.arraytocalews(df,Fin2)
wu.viswlewsagn(df,50,100,0,130,0,135,vii1,voo1,0,4,toplot='vis') #5
wu.viswlewsagn(df,50,100,0,130,0,135,vii,voo,0,4,toplot='vis') #5
HD6290sc=wu.viswlewsagn(df,50,100,0,130,0,135,vii,voo,0,4)
HD6290sc.to_pickle('HD6290sc.pk')
#84Aqr----------------------
df=DF2[DF2['targ']=='84 Aqr']  #5
df=df.reset_index()
badmjd=[58381.15815322]

for i in range(np.size(badmjd)):
    df=df[df['mjd-obs']!=badmjd[i]]
    print(badmjd[i])
    print(df.shape)

df=df.reset_index() #4
vii,voo=wu.arraytocalews(df,Fin2)
wu.viswlewsagn(df,50,100,0,130,0,135,vii1,voo1,0,4,toplot='vis') #5
wu.viswlewsagn(df,50,100,0,130,0,135,vii,voo,0,4,toplot='vis') #5
df84Aqrsc=wu.viswlewsagn(df,50,100,0,130,0,135,vii,voo,0,4)
df84Aqrsc.to_pickle('df84Aqrsc.pk')
#HD14823----------------------
df=DF2[DF2['targ']=='HD14823']  #20
df=df.reset_index()  #NO bad mjds
vii,voo=wu.arraytocalews(df,Fin2)
wu.viswlewsagn(df,50,100,0,140,0,180,vii1,voo1,0,20,toplot='vis') #5
wu.viswlewsagn(df,50,100,0,140,0,180,vii,voo,0,20,toplot='vis') #5
HD14823sc=wu.viswlewsagn(df,50,100,0,140,0,180,vii,voo,0,20)
HD14823sc.to_pickle('HD14823sc.pk')
#del Phe-Sept 22--------------------
df=DF2[DF2['targ']=='del Phe']  #20
df=df.reset_index()
badmjd=[58383.13341208, 58383.13461323, 58383.13554598, 58383.13647738, 58383.13741494, 58383.13836445, 58383.13942234, 58383.14308607, 58383.14435676, 58383.14797656, 58383.15507120, 58384.31596017, 58384.31937302, 58384.32040372, 58384.32355232]

for i in range(np.size(badmjd)):
    df=df[df['mjd-obs']!=badmjd[i]]
    print(badmjd[i])
    print(df.shape)

df=df.reset_index() #5  Sept 22
vii,voo=wu.arraytocalews(df,Fin2)
wu.viswlewsagn(df,50,100,0,130,0,135,vii1,voo1,0,5,toplot='vis') #5
wu.viswlewsagn(df,50,100,0,130,0,135,vii,voo,0,5,toplot='vis') #5
delPheSept22sc=wu.viswlewsagn(df,50,100,0,130,0,135,vii,voo,0,5)
delPheSept22sc.to_pickle('delPheSept22sc.pk')
#delPhe --4Sept 23--------------------
df=DF2[DF2['targ']=='del Phe']  #20
df=df.reset_index()
badmjd=[58383.13341208, 58383.13461323, 58383.13554598, 58383.13647738, 58383.13741494, 58383.13836445, 58383.13942234, 58383.14308607, 58383.14435676, 58383.14797656, 58383.14981545, 58383.15075302, 58383.15213772, 58383.15307554, 58383.15401894, 58383.15507120]
for i in range(np.size(badmjd)):
    df=df[df['mjd-obs']!=badmjd[i]]
    print(badmjd[i])
    print(df.shape)

df=df.reset_index() #4  Sept 23
vii,voo=wu.arraytocalews(df,Fin2)
wu.viswlewsagn(df,50,100,0,130,0,135,vii1,voo1,0,4,toplot='vis') #5
wu.viswlewsagn(df,50,100,0,130,0,135,vii,voo,0,4,toplot='vis') #5
delPheSept23sc=wu.viswlewsagn(df,50,100,0,130,0,135,vii,voo,0,4)
delPheSept23sc.to_pickle('delPheSept23sc.pk')
#HD 16658 ---------------------
df=DF2[DF2['targ']=='HD 16658']
df=df.reset_index()
badmjd=[58383.20571736]
for i in range(np.size(badmjd)):
    df=df[df['mjd-obs']!=badmjd[i]]
    print(badmjd[i])
    print(df.shape)

df=df.reset_index() #6 Sept 23
vii,voo=wu.arraytocalews(df,Fin2)
wu.viswlewsagn(df,50,100,0,130,0,135,vii1,voo1,0,6,toplot='vis') #5
wu.viswlewsagn(df,50,100,0,130,0,135,vii,voo,0,6,toplot='vis') #5
HD16658sc=wu.viswlewsagn(df,50,100,0,130,0,135,vii,voo,0,6)
HD16658sc.to_pickle('HD16658sc.pk')
#HD 195677-------------------
df=DF2[DF2['targ']=='HD 195677']
df=df.reset_index()  #5
badmjd=[58384.17850135]
for i in range(np.size(badmjd)):
    df=df[df['mjd-obs']!=badmjd[i]]
    print(badmjd[i])
    print(df.shape)

df=df.reset_index() #4 Sept 23
vii,voo=wu.arraytocalews(df,Fin2)
wu.viswlewsagn(df,50,100,0,130,0,135,vii1,voo1,0,4,toplot='vis') #5
wu.viswlewsagn(df,50,100,0,130,0,135,vii,voo,0,4,toplot='vis') #5
HD195677sc=wu.viswlewsagn(df,50,100,0,130,0,135,vii,voo,0,4)
HD195677sc.to_pickle('HD195677sc.pk')
#sigCap --4Sept 23--------------------
df=DF2[DF2['targ']=='sig Cap']  #10
df=df.reset_index()
badmjd=[58384.98769468, 58385.00456004, 58385.00903409]
for i in range(np.size(badmjd)):
    df=df[df['mjd-obs']!=badmjd[i]]
    print(badmjd[i])
    print(df.shape)

df=df.reset_index() #7
vii,voo=wu.arraytocalews(df,Fin2)
wu.viswlewsagn(df,50,100,0,130,0,135,vii1,voo1,0,7,toplot='vis') #5
wu.viswlewsagn(df,50,100,0,130,0,135,vii,voo,0,7,toplot='vis') #5
sigCapsc=wu.viswlewsagn(df,50,100,0,130,0,135,vii,voo,0,7)
sigCapsc.to_pickle('sigCapsc.pk')
#HD 25286-----------------------------
df=DF2[DF2['targ']=='HD25286']  #14
df=df.reset_index()
badmjd=[58385.38429503, 58385.38560493]
for i in range(np.size(badmjd)):
    df=df[df['mjd-obs']!=badmjd[i]]
    print(badmjd[i])
    print(df.shape)

df=df.reset_index() #12
vii,voo=wu.arraytocalews(df,Fin2)
wu.viswlewsagn(df,50,100,0,130,0,135,vii1,voo1,0,12,toplot='vis') #5
wu.viswlewsagn(df,50,100,0,130,0,135,vii,voo,0,12,toplot='vis') #5
HD25286sc=wu.viswlewsagn(df,50,100,0,130,0,135,vii,voo,0,12)
HD25286sc.to_pickle('HD25286sc.pk')
#HD 172453-----------------------------
df=DF2[DF2['targ']=='HD 172453']  #6
df=df.reset_index()
badmjd=[58386.03318726]
for i in range(np.size(badmjd)):
    df=df[df['mjd-obs']!=badmjd[i]]
    print(badmjd[i])
    print(df.shape)

df=df.reset_index() #5
vii,voo=wu.arraytocalews(df,Fin2)
wu.viswlewsagn(df,50,100,0,130,0,135,vii1,voo1,0,5,toplot='vis') #5
wu.viswlewsagn(df,50,100,0,130,0,135,vii,voo,0,5,toplot='vis') #5
HD172453sc=wu.viswlewsagn(df,50,100,0,130,0,135,vii,voo,0,5)
HD172453sc.to_pickle('HD172453sc.pk')
#V026 Sgr-----------------------------
df=DF2[DF2['targ']=='V4026 Sgr']  #5
df=df.reset_index()
badmjd=[58386.07974087, 58386.08855096]
for i in range(np.size(badmjd)):
    df=df[df['mjd-obs']!=badmjd[i]]
    print(badmjd[i])
    print(df.shape)

df=df.reset_index() #3
vii,voo=wu.arraytocalews(df,Fin2)
wu.viswlewsagn(df,50,100,0,130,0,135,vii1,voo1,0,3,toplot='vis') #5
wu.viswlewsagn(df,50,100,0,130,0,135,vii,voo,0,3,toplot='vis') #5  SOMETHING IS WRONG!!!! EVEN CAL DOESNT GET TO 1
V026Sgrsc=wu.viswlewsagn(df,50,100,0,130,0,135,vii,voo,0,3)
V026Sgrsc.to_pickle('V026Sgrsc.pk')
#HD 209240----------------------------
df=DF2[DF2['targ']=='HD209240']  #9
df=df.reset_index()
badmjd=[58386.10701578, 58386.11361284, 58386.11911919]
for i in range(np.size(badmjd)):
    df=df[df['mjd-obs']!=badmjd[i]]
    print(badmjd[i])
    print(df.shape)

df=df.reset_index() #6
vii,voo=wu.arraytocalews(df,Fin2)
wu.viswlewsagn(df,50,100,0,140,0,180,vii1,voo1,0,6,toplot='vis') #5
wu.viswlewsagn(df,50,100,0,140,0,180,vii,voo,0,6,toplot='vis') #5
HD209240sc=wu.viswlewsagn(df,50,100,0,140,0,180,vii,voo,0,60)
HD209240sc.to_pickle('HD209240sc.pk')
#47 Cap ---------------------------
df=DF2[DF2['targ']=='47 Cap']  #7
df=df.reset_index()
badmjd=[58386.18925231, 58386.19161547, 58386.19635368, 58386.20523466]
for i in range(np.size(badmjd)):
    df=df[df['mjd-obs']!=badmjd[i]]
    print(badmjd[i])
    print(df.shape)

df=df.reset_index() #3
vii,voo=wu.arraytocalews(df,Fin2)
wu.viswlewsagn(df,50,100,0,140,0,180,vii1,voo1,0,6,toplot='vis') #5  STRUCTURE???
wu.viswlewsagn(df,50,100,0,140,0,180,vii,voo,0,6,toplot='vis') #5 BELOW 1 after cal
df47Capsc=wu.viswlewsagn(df,50,100,0,140,0,180,vii,voo,0,6)
df47Capsc.to_pickle('df47Capsc.pk')
#HD 20356 ---------------------------
df=DF2[DF2['targ']=='HD 20356']  #13
df=df.reset_index()
badmjd=[58385.35656910, 58385.36001311, 58385.36108241, 58385.36422807, 58385.36542380, 58386.25508756, 58386.27472541]
for i in range(np.size(badmjd)):
    df=df[df['mjd-obs']!=badmjd[i]]
    print(badmjd[i])
    print(df.shape)

df=df.reset_index() #6
vii,voo=wu.arraytocalews(df,Fin2)
wu.viswlewsagn(df,50,100,0,140,0,180,vii1,voo1,0,6,toplot='vis')
wu.viswlewsagn(df,50,100,0,140,0,180,vii,voo,0,6,toplot='vis')
HD20356sc=wu.viswlewsagn(df,50,100,0,140,0,180,vii,voo,0,6)
HD20356sc.to_pickle('HD20356sc.pk')
#HD33162--------------------------------
df=DF2[DF2['targ']=='HD33162']  #5
df=df.reset_index()
badmjd=[58386.40622892, 58386.4108629]
for i in range(np.size(badmjd)):
    df=df[df['mjd-obs']!=badmjd[i]]
    print(badmjd[i])
    print(df.shape)

df=df.reset_index() #3
vii,voo=wu.arraytocalews(df,Fin2)
wu.viswlewsagn(df,50,100,0,140,0,180,vii1,voo1,0,3,toplot='vis')
wu.viswlewsagn(df,50,100,0,140,0,180,vii,voo,0,3,toplot='vis')
HD33162sc=wu.viswlewsagn(df,50,100,0,140,0,180,vii,voo,0,3)
HD33162sc.to_pickle('HD33162sc.pk')  #includes a col with calvis
#print(HD33162sc['UDDL'][0])
#*to test: HD33162sc['vis'][0][0]-HD33162sc['calvis'][0][0]
#PLOTS
df=pd.concat([HD183925sc,HD6290sc,df84Aqrsc,HD14823sc,delPheSept22sc,delPheSept23sc,HD16658sc,HD195677sc,sigCapsc,HD25286sc,V026Sgrsc,HD209240sc,df47Capsc,HD20356sc, HD33162sc], ignore_index=True)#left HD172453sc out   91,32
df.to_pickle('scSeptCals.pk') #selected
'''/usr/local/bin/ipython3:1: FutureWarning: Sorting because non-concatenation axis is not aligned. A future version
of pandas will change to not sort by default.

To accept the future behavior, pass 'sort=False'.

To retain the current behavior and silence the warning, pass 'sort=True'.'''

corrvis=[1,1,1,1,1,1]  #No channel correction needed since
corrcf=[1,1,1,1,1,1]  #it is for calibrated data
#VIS

wu.TFvisEWSflux(df,0,130,'diam',corrvis,tableDiams=Fin2,yaxis=None)   #1  CHECK that it is not dividing by model again!!
wu.TFvisEWSflux(df,0,130,'diam',corrvis,tableDiams=Fin2,yaxis='calvis')   #1  CHECK is not dividing by model again!!
wu.TFvisEWSflux(df,0,130,'tau',corrvis,tableDiams=Fin2,yaxis=None)   #1  CHECK that it is not dividing by model again!!
wu.TFvisEWSflux(df,0,130,'tau',corrvis,tableDiams=Fin2,yaxis='calvis')    #2
wu.TFvisEWStau(df,0,130,'diam',corrvis,tableDiams=Fin2,yaxis=None)    #3-
wu.TFvisEWStau(df,0,130,'diam',corrvis,tableDiams=Fin2,yaxis='calvis')    #3-
wu.TFvisEWStau(df,0,130,'tau',corrvis,tableDiams=Fin2,yaxis=None)     #4

wu.TFvisEWStau(df,0,130,'tau',corrvis,tableDiams=Fin2,yaxis='calvis')     #4
wu.TFvisEWSmjd(df,0,130,'diam',corrvis,tableDiams=Fin2,yaxis=None)    #5-
wu.TFvisEWSmjd(df,0,130,'diam',corrvis,tableDiams=Fin2,yaxis='calvis')    #5-
wu.TFvisEWSmjd(df,0,130,'tau',corrvis,tableDiams=Fin2,yaxis=None)     #6

wu.TFvisEWSmjd(df,0,130,'tau',corrvis,tableDiams=Fin2,yaxis='calvis')     #6
wu.TFvisEWSbl(df,0,130,'diam',corrvis,tableDiams=Fin2,yaxis=None)     #7-

wu.TFvisEWSbl(df,0,130,'diam',corrvis,tableDiams=Fin2,yaxis='calvis')     #7-
wu.TFvisEWSbl(df,0,130,'tau',corrvis,tableDiams=Fin2,yaxis=None)      #8

wu.TFvisEWSbl(df,0,130,'tau',corrvis,tableDiams=Fin2,yaxis='calvis')      #8
# Using averages and error bars
#-df must not include bad mjds
         #uses stdev from statistics , divides by n-1
wu.TFvisEWSflux(df,0,130,'diam',corrvis,tableDiams=Fin2,yaxis='calvisavg')   #1
wu.TFvisEWSflux(df,0,130,'diam',corrvis, tableDiams=Fin2,yaxis='calvisavg',yerr=True)   #HD 172453 and HD 16658
wu.TFvisEWSflux(df,0,130,'tau',corrvis,tableDiams=Fin2,yaxis='calvisavg')    #2
wu.TFvisEWSflux(df,0,130,'tau',corrvis,  tableDiams=Fin2,yaxis='calvisavg',yerr=True)
wu.TFvisEWStau(df,0,140,'diam',corrvis,tableDiams=Fin2,yaxis='calvisavg')    #3-
wu.TFvisEWStau(df,0,130,'diam',corrvis, tableDiams=Fin2,yaxis='calvisavg',yerr=True)
wu.TFvisEWStau(df,0,130,'tau',corrvis,tableDiams=Fin2,yaxis='calvisavg')    #4
wu.TFvisEWStau(df,0,130,'tau',corrvis, tableDiams=Fin2,yaxis='calvisavg',yerr=True)
wu.TFvisEWSmjd(df,0,130,'diam',corrvis, tableDiams=Fin2,yaxis='calvisavg')   #5
wu.TFvisEWSmjd(df,0,130,'diam',corrvis, tableDiams=Fin2,yaxis='calvisavg',yerr=True)  #WHY SYMMETRY??  CHECK!!!
wu.TFvisEWSmjd(df,0,130,'tau',corrvis, tableDiams=Fin2,yaxis='calvisavg')   #6
wu.TFvisEWSmjd(df,0,130,'tau',corrvis, tableDiams=Fin2,yaxis='calvisavg',yerr=True)
wu.TFvisEWSbl(df,0,130,'diam',corrvis, tableDiams=Fin2,yaxis='calvisavg')   #7
wu.TFvisEWSbl(df,0,130,'diam', corrvis, tableDiams=Fin2,yaxis='calvisavg',yerr=True)
wu.TFvisEWSbl(df,0,130,'tau',corrvis, tableDiams=Fin2,yaxis='calvisavg')   #8
wu.TFvisEWSbl(df,0,130,'tau',corrvis, tableDiams=Fin2,yaxis='calvisavg',yerr=True)
wu.table2D(df,50,100,Fin2)
#2D TABLE  avgviserror vs tau0 and flux
# now in function wu.table2D
ayavg=np.zeros(df['targ'].shape[0])
ayerr=np.zeros(df['targ'].shape[0])
afluxes=np.zeros(df['targ'].shape[0])
dfpos=wu.positionscat(Fin2)
for i in range(df['targ'].shape[0]):
    ayavg[i]=np.mean([np.mean(np.abs(DF['calvis'][i][0][wl1:wl2])),np.mean(np.abs(DF['calvis'][i][1][wl1:wl2])),np.mean(np.abs(DF['calvis'][i][2][wl1:wl2])),np.mean(np.abs(DF['calvis'][i][3][wl1:wl2])),np.mean(np.abs(DF['calvis'][i][4][wl1:wl2])),np.mean(np.abs(DF['calvis'][i][5][wl1:wl2]))])
    ayerr[i]=np.std([np.mean(np.abs(DF['calvis'][i][0][wl1:wl2])),np.mean(np.abs(DF['calvis'][i][1][wl1:wl2])),np.mean(np.abs(DF['calvis'][i][2][wl1:wl2])),np.mean(np.abs(DF['calvis'][i][3][wl1:wl2])),np.mean(np.abs(DF['calvis'][i][4][wl1:wl2])),np.mean(np.abs(DF['calvis'][i][5][wl1:wl2]))])
    k=wu.search_cat(dfpos,DF['ra'][i],DF['dec'][i])
    afluxes[i]=Fin2['med_Lflux'][k]
    if (ayerr[i]>0.05):
        plt.text(DF['tau'][i],afluxes[i],'x')

plt.rcParams['axes.facecolor'] = 'white'
plt.scatter(DF['tau'],afluxes,s=ayerr*7000, c=ayerr,cmap='cool' , alpha=0.5) #='solar')
plt.yscale('log')
title= 'viserror by mjd, with LFlux and tau'  #Lflux=cat2['med_Lflux'][index2]
plt.title(title)
plt.ylabel('Lflux [Jy]')
plt.xlabel('tau0 [ms]')
plt.grid(linestyle="--",linewidth=0.1,color='.25')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='0.25', linestyle='--', alpha=0.2)
cbar = plt.colorbar()
cbar.set_label('viserr',rotation=270)
for i in range(df['targ'].shape[0]):
    if (ayerr[i]>0.05):
        plt.text(DF['tau'][i],afluxes[i],'x')
         
         
#CONCLUSION !!!!
Eleccion Cals:  Mandar updated table
         delPhe 78Jy Cal for NGC 1068 Sept22
         delPhe 78Jy Cal for NGC 1566 Sept23
         HD 25286 1.9Jy Cal for NGC 1566  Sept24
         HD 25286 1.9Jy Cal for NGC 1068 Sept24
         HD 20356 28Jy Cal for NGC 1068 Sept25
         HD 20356 28Jy Cal for NGC 7469 Sept25
         
#NEXT STEP: CALIBRATE AGNs with selected calibrators: go to CALIBRATING AGNS
         
#May 24/Volumes/LaCie/Reduced/May25CalsSeptDRSv.1.4
#Sept vis to vis

#2Q. DRS telescope number convention for J
#Calibrate calibrators April, then Sept to see what Leo suggests #  also to see if bcds are right for evry source  DONE not for April

AN ARRAY GEOMETRY:
UT1=U1=32
UT2=U2-33
UT3=U3=34
UT4=U4=35
http://www.eso.org/observing/etc/doc//viscalc/vltistations.html
The physical location of the array centre is defined as follows:
Longitude: -70.40479659 degrees
Latitude: -24.62794830 degrees
Station Position Coordinates

Following is a list of the station platform coordinates. The platform coordinates (p,q) are rotated from the geographic grid coordinates (e,n). This platform rotation, nu, defined as a clockwise rotation from geometric north (i.e. east = +90 degrees), is:
    nu  =  -18.984 degrees
The geographic coordinates may be derived by rotating the (p,q) coordinates:
    e  =  p * cos(nu)  +  q * sin(nu)
    n  =  q * cos(nu)  -  p * sin(nu)
where positive e is defined as the distance to the geometric east of the array centre and positive n is defined as the distance to the geometric north of the array centre.
Each station is allocated an letter-digit identifier for easier specification, e.g. "A1".

The following data is the station identifier, plus the coordinates of that station in the reference frame of the rotated platform, followed by the coordinates of the station as per an East-North grid, all specified in metres.
_ID_______P__________Q__________E__________N____
    U1    -16.000    -16.000     -9.925    -20.335
        U2     24.000     24.000     14.887     30.502
            U3     64.0013    47.9725    44.915     66.183
                U4    112.000      8.000    103.306     43.999
Baseline data, sorted by position angle.
    U1-U2     56.569     26.016  32-33
    U1-U3    102.434     32.369  32-34
    U1-U4    130.231     60.396  32-35
    U2-U3     46.635     40.082  33-34
    U2-U4     89.443     81.321  33-35
    U3-U4     62.463    110.803  34-35

SUMMARY AND PLAN
CALIBRATE A CALIBRATOR Jacob: HD 29246 (*target*, TPL_START: 2018-09-21T09:33:42) and HD212849 (*calib*, TPL_START: 2019-09-21T05:56:34) which are both marked "very good". Even though they are "faint" (~5 Jy) the results are exactly what we should expect for a point source.

#... go to #MAY 24

#May 23

#make sure aprcorrvis is made with selected mjds   PENDING

hdu=fits.open('/Users/M51/redDataDRSSept/2018-09-22T04_44_55_vis01.fit

v2=hdu['OI_VIS2'].data
v2['MJD']
v2['UCOORD'][0:6]
v2['VCOORD'][0:6]
DF2=pd.read_pickle('/Users/M51/May14SeptCalsEWS.pk') # (166, 19)
HD16658=DF2[DF2['targ']=='HD 16658']   #Sept 22 NGC1068 2018-09-22T04:44:55
HD16658=HD16658.reset_index()
vii1=[1,1,1,1,1,1]
voo1=[1,1,1,1,1,1]
tbl=wu.viswlewsagn(HD16658,50,100,30,110,0,135,vii1,voo1,2,3) #9ischopped
u=np.zeros(6)
v=np.zeros(6)
for j in range(6):
    u[j]=tbl['pbl'][0][j]*np.sin(tbl['pbla'][0][j])
    v[j]=tbl['pbl'][0][j]*np.cos(tbl['pbla'][0][j])
In [168]: u
Out[168]:
array([52.14869873,  6.60037731, 73.80307745, 21.33323522, 67.19114895,
       14.69451099])

In [169]: v
Out[169]:
array([-20.23238286,  46.37666822,  59.36040876,  79.06448469,
       12.59268931,  32.6967698 ])

In [171]:  v2['UCOORD'][0:6]
Out[171]:
array([52.0918798 ,  6.89014193, 14.8590231 , 66.95090291, 21.74916504,
                               73.84104484])
                        
In [172]:  v2['VCOORD'][0:6]
Out[172]:
array([-20.0466714 ,  46.29903655,  32.55176286,  12.50509146,
                               78.85079941,  58.80412802])

In [173]: tbl['bcd']
                        Out[173]:
                        0    0
                        Name: bcd, dtype: int64
                        
In [174]: tbl['mjd-obs']
                        Out[174]:
                        0    58383.20187058
                        
                        
                        t3['STA_INDEX']
                        Out[194]:
                        array([[33, 34, 35],
                               [32, 33, 34],
                               [32, 33, 35],
                               [32, 34, 35],
                               [33, 34, 35],
                               [32, 33, 34],
                               [32, 33, 35],
                               [32, 34, 35],
                               [33, 34, 35],
                               [32, 33, 34],
                               [32, 33, 35],
                               [32, 34, 35],
                               [33, 34, 35],
                               [32, 33, 34],
                               [32, 33, 35],
                               [32, 34, 35],
                               [33, 34, 35],
                               [32, 33, 34],
                               [32, 33, 35],
                               [32, 34, 35],
                               [33, 34, 35],
                               [32, 33, 34],
                               [32, 33, 35],
                               [32, 34, 35],
                               [33, 34, 35],
                               [32, 33, 34],
                               [32, 33, 35],
                               [32, 34, 35]], dtype=int16)
In [195]: t3['V1COORD'][0:4]
Out[195]: array([32.55176286, 46.29903655, 46.29903655, 78.85079941])
                        
In [196]: t3['V2COORD'][0:4]
Out[196]: array([-20.0466714 ,  32.55176286,  12.50509146, -20.0466714 ])
                        
In [197]: v2['VCOORD'][0:6]
Out[197]:
array([-20.0466714 ,  46.29903655,  32.55176286,  12.50509146,
                               78.85079941,  58.80412802])
In [199]: v2['sta_index'][0:6]
array([[34, 35],
[32, 33],
[33, 34],
[33, 35],
[32, 34],
[32, 35]], dtype=int16)
#Avoiding the bad mjds
df=pd.read_pickle('AprilCalsUTs.pk')  #Has NO 3C 273
amjds=[58589.97297293, 58591.16714752, 58591.17005860, 58590.24013409, 58590.24305453]
for i in range(np.size(amjds)):
    df=df[df['mjd-obs']!=amjds[i]]
    print(amjds[i])
    print(df.shape)  #11, 20

dfA=df.reset_index() #11,21
Fin3=pd.read_pickle('TableDiamsAprilUTsjsdc17.pk')
Fin2=pd.read_pickle('/Users/M51/TableDiamsSeptmsdfcc19.pk')
#SELECTION CALIBRATORS
DF2=pd.read_pickle('/Users/M51/May14SeptCalsEWS.pk') # (166, 19)
df=DF2[DF2['targ']!='CD-55 855']  #159,18
df=df[df['targ']!='b Sgr']  #150
amjd=[58381.04560605,58381.05829687,58381.11948348,58381.15815322, 58382.12718112,58382.13937325,58382.14056971,58382.14151070, 58382.14244882,58382.14338623,58382.14647846, 58383.13341208,58383.13461323,58383.13554598,58383.13647738, 58383.13741494,58383.13836445,58383.13942234,58383.14308607, 58383.14435676,58383.14797656, 58383.15507120, 58383.20571736, 58384.07957074, 58384.17850135, 58384.98769468, 58385.00456004, 58385.38429503, 58385.38560493, 58386.03318726, 58386.07974087, 58386.08855096, 58386.10701578, 58386.11361284, 58386.11911919, 58386.18925231,58386.19161547, 58386.19635368, 58386.20523466, 58386.40622892, 58386.4108629,  58385.35656910, 58385.36001311, 58385.36108241, 58385.36422807, 58385.36542380]   #41 Lst 6 HD20356
for i in range(np.size(amjd)):
    df=df[df['mjd-obs']!=amjd[i]]
    print(amjd[i])
    print(df.shape)  #96,18   #104,19

dfS=df.reset_index()   #104,20
aprcorrvis=[1.0158460507124838, 1.1496059666487555, 1.1256564665409685, 0.8531561895438786, 1.0178854238281492, 0.8378499027257653]

septcorrvis=[1.0171356326838124, 1.126052655208009, 1.1070565294110437, 0.8673201939895662, 1.0308546161084602, 0.8515803725991087]
#Plotting April and Sept UT calibrators together
wu.TFvisEWSflux(dfA,0,130,'diam',aprcorrvis,tableDiams=Fin3)
wu.TFvisEWSflux(dfS,0,130,'diam',septcorrvis,tableDiams=Fin2)   #1SAvis

wu.TFvisEWSflux(dfA,0,130,'tau',aprcorrvis,tableDiams=Fin3)    #2 <----
wu.TFvisEWSflux(dfS,0,130,'tau',septcorrvis,tableDiams=Fin2)  #

wu.TFvisEWStau(dfA,0,130,'diam',aprcorrvis,tableDiams=Fin3)    #3-
wu.TFvisEWStau(dfS,0,130,'diam',septcorrvis,tableDiams=Fin2)

wu.TFvisEWStau(dfA,0,130,'tau',aprcorrvis,tableDiams=Fin3)     #4
wu.TFvisEWStau(dfS,0,130,'tau',septcorrvis,tableDiams=Fin2)     #4

wu.TFvisEWSmjd(dfA,0,130,'diam',aprcorrvis,tableDiams=Fin3)
wu.TFvisEWSmjd(dfS,0,130,'diam',septcorrvis,tableDiams=Fin2)

wu.TFvisEWSmjd(dfA,0,130,'tau',aprcorrvis,tableDiams=Fin3)     #6
wu.TFvisEWSmjd(dfS,0,130,'tau',septcorrvis,tableDiams=Fin2)     #6

   #5-#missing 2 bls  CHECK
wu.TFvisEWSbl(dfA,0,130,'diam',aprcorrvis,tableDiams=Fin3)     #7-
wu.TFvisEWSbl(dfS,0,130,'tau',septcorrvis,tableDiams=Fin2)      #8

septcorrcf=[1.0461615749200677,1.1432393108263472,1.06764454906736,0.9009873729422595,1.0163537336808484,0.8256134585631165]


#May 22 downloading files
ssh ssh.strw.leidenuniv.nl -p 80 -l gamez
scp gamez@ssh.strw.leidenuniv.nl:/disks/cool1/jaffe/drs/aprildata/apriltpl/2019-04-16T23:14:47.tpl.pk ./   #Yessss!!!! :)
scp gamez@ssh.strw.leidenuniv.nl:/disks/cool1/jaffe/drs/aprildata/apriltpl/2019-04-17T01:11:57.tpl.pk ./
scp gamez@ssh.strw.leidenuniv.nl:/disks/cool1/jaffe/drs/aprildata/apriltpl/2019-04-17T02:32:41.tpl.pk ./
scp gamez@ssh.strw.leidenuniv.nl:/disks/cool1/jaffe/drs/aprildata/apriltpl/2019-04-17T05:31:00.tpl.pk ./
scp gamez@ssh.strw.leidenuniv.nl:/disks/cool1/jaffe/drs/aprildata/apriltpl/2019-04-17T05:40:53.tpl.pk ./
scp gamez@ssh.strw.leidenuniv.nl:/disks/cool1/jaffe/drs/aprildata/apriltpl/2019-04-18T01:22:04.tpl.pk ./
scp gamez@ssh.strw.leidenuniv.nl:/disks/cool1/jaffe/drs/aprildata/apriltpl/2019-04-18T03:54:24.tpl.pk ./
scp gamez@ssh.strw.leidenuniv.nl:/disks/cool1/jaffe/drs/aprildata/apriltpl/2019-04-18T04:25:34.tpl.pk ./
scp gamez@ssh.strw.leidenuniv.nl:/disks/cool1/jaffe/drs/aprildata/apriltpl/2019-04-18T05:02:02.tpl.pk ./

scp -r gamez@ssh.strw.leidenuniv.nl:/disks/cool1/jaffe/drs/gamez/redDataSept/2018*.fits ./

#table observations AprilCals
df=df[df['targ']!='3C273']   #16,19
df=df.reset_index()
df.to_pickle('AprilCalsUTs.pk')
df[['targ','file','tau']]
df=pd.read_pickle('AprilCalsUTs.pk')  #16,20
for j in range(len(df['header'])):
    print(df['header'][j]['HIERARCH ESO TPL START'])
for j in range(len(df['header'])):
    print(df['header'][j]['HIERARCH ESO INS DIL ID']) #All LOW
#HD 97765 has 64 wls instead of 110. From 2.73815408808 to 4.206

#Fin3
        Name    med_Lflux
0  * lam Pyx                          28.10558701
1  HD 150071                           0.31112576
2  HD 134102                           4.66320658
3  HD  97765                           0.12568563
4  HD  97705                           0.04386224
#Selecting good mjds by eye
anames=np.unique(df['targ'])
#For every source look at the photometry
#To do Quality Control 1: vis by eye
for i in range(np.size(anames)):
    df2=df[df['targ']==anames[i]]
    df2=df2.reset_index()
    plt.clf()
    wu.viswlewscal(df2,'wave',50,100,0,130,tableDiams=Fin3)
    #HD 97765 has only L band(64 wls), code takes care of that
#To do Quality Control 2: opd tracking by eye
for i in range(df.shape[0]):
    print(df[['targ','file','tau','mjd-obs']][i:i+1])
    cf=df['cflux'][i]
    plt.imshow(cf['opdimage'])  #cf['opdimage'].shape (271, 768)
    plt.pause(15)
#Quality Control 3: Photometry
wu.photews(df)  #check S/N
#Quality Control 4: Walter's OPD tracking opdflag<0.3
for i in range(df.shape[0]):
    print(df['opdflag'][i])

#Avoiding the bad mjds
df=pd.read_pickle('AprilCalsUTs.pk')  #Has NO 3C 273
amjds=[58589.97297293, 58591.16714752, 58591.17005860, 58590.24013409, 58590.24305453]
for i in range(np.size(amjds)):
    df=df[df['mjd-obs']!=amjds[i]]
    print(amjds[i])
    print(df.shape)  #11, 20

df=df.reset_index() #11,21

Fin3=pd.read_pickle('TableDiamsAprilUTsjsdc17.pk')
#Plots April cals  WITH CHANNEL CORRECTION
correction=(1,1,1,1,1,1)
wu.TFvistauEWS(dfA,'tau','vis',0,150,1,1,1,correction,Fin3) #channelcorrection vis ews
mean= 0.565849143209848 std= 0.04416939767784047
popt a=0.0023681406610297434 b=0.5524125283846789
mean= 0.6403564307809081 std= 0.0834994840282275
popt a=0.02548741033987503 b=0.4957431817072323
mean= 0.6270160194983219 std= 0.08838776462896195
popt a=0.019096920170085494 b=0.5186618305369087
mean= 0.47522722418322216 std= 0.058993570336804746
popt a=0.013969638708806756 b=0.3959647641166078
mean= 0.5669851199942979 std= 0.09649229866672546
popt a=0.024064577358320224 b=0.43044489566017763
mean= 0.4667012774852171 std= 0.06116952763021346
popt a=0.013126111189655579 b=0.3922249158779468
correction2=[1.0158460507124838, 1.1496059666487555, 1.1256564665409685, 0.8531561895438786, 1.0178854238281492, 0.8378499027257653]
wu.TFvistauEWS(dfA,'tau','vis',0,130,1,1,1,correction2,Fin3)
mean= 0.5570225358586357 std= 0.04348040497559782
popt a=0.0023312024420329607 b=0.543795505130033
mean= 0.5570225358586358 std= 0.07263313383074975
popt a=0.022170561833463012 b=0.43122878346107796
mean= 0.5638145393490317 std= 0.08267394317243959
popt a=0.02039056076609558 b=0.4557241767279592
mean= 0.5570225358586357 std= 0.06914744458262019
popt a=0.01637406911194563 b=0.4641175562691935
mean= 0.5570225358586357 std= 0.09479681740979166
popt a=0.023641735481904246 b=0.4228814779829914
mean= 0.5572709129346552 std= 0.06333928842363752
popt a=0.013688971271771043 b=0.4849094897952657
Out[399]:
[0.9978978765474654,
 0.9978978765474655,
 1.010065653296608,
 0.9978978765474654,
 0.9978978765474654,
 0.9983428405135298]
wu.TFcftauEWS(dfA,'tau','cflux',0,130,1,1,1,correction,Fin3)
mean= 1537.2823102353002 std= 266.37770070944026
popt a=12.793629168089538 b=1464.692421326178
mean= 1782.2399955649373 std= 454.8530738898536
popt a=87.31138420532841 b=1286.84314416389
mean= 1758.720079933615 std= 444.9160592876114
popt a=98.76535256003531 b=1235.1649434451288
mean= 1324.4761537751647 std= 289.2675597412579
popt a=45.44865884588292 b=1066.6045957334713
mean= 1525.0610666974928 std= 388.75048249591896
popt a=64.52631232283889 b=1158.944636606098
mean= 1326.6873709406277 std= 236.5937334381691
popt a=51.139314452258 b=1056.3592728763504
correction2=[0.9966747824794958, 1.1554895597764698, 1.1402407621810584, 0.8587049845523091, 0.9887513157463045, 0.8601385952643621]
wu.TFcftauEWS(dfA,'tau','cflux',0,130,1,1,1,correction2,Fin3)
mean= 1542.4111628578562 std= 267.26641969083875
popt a=12.836312858942897 b=1469.5790903731584
mean= 1542.4111628578562 std= 393.64533417147027
popt a=75.5622449254665 b=1113.6778549600056
mean= 1542.4111628578564 std= 390.19483783106796
popt a=86.61798090478253 b=1083.2492469342956
mean= 1542.4111628578562 std= 336.86488950808786
popt a=52.92697314704856 b=1242.1083279063387
mean= 1542.4111628578562 std= 393.1731632665308
popt a=65.26040722696354 b=1172.129544860644
mean= 1542.4111628578567 std= 275.0646636957994
popt a=59.45473603569039 b=1228.126822091415
Out[82]: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0000000000000002]


aprcorrvis=[1.0158460507124838, 1.1496059666487555, 1.1256564665409685, 0.8531561895438786, 1.0178854238281492, 0.8378499027257653]
aprcorrcf=[0.9966747824794958, 1.1554895597764698, 1.1402407621810584, 0.8587049845523091, 0.9887513157463045, 0.8601385952643621]

wu.TFvisEWSflux(dfA,0,130,'diam',aprcorrvis,tableDiams=Fin3)   #1Apr
wu.TFvisEWSflux(dfA,0,130,'tau',aprcorrvis,tableDiams=Fin3)    #2
wu.TFvisEWStau(dfA,0,130,'diam',aprcorrvis,tableDiams=Fin3)    #3-
wu.TFvisEWStau(dfA,0,130,'tau',aprcorrvis,tableDiams=Fin3)     #4
wu.TFvisEWSmjd(dfA,0,130,'diam',aprcorrvis,tableDiams=Fin3)    #5-#missing 2 bls  CHECK
wu.TFvisEWSmjd(dfA,0,130,'tau',aprcorrvis,tableDiams=Fin3)     #6
wu.TFvisEWSbl(dfA,0,130,'diam',aprcorrvis,tableDiams=Fin3)     #7-
wu.TFvisEWSbl(dfA,0,130,'tau',aprcorrvis,tableDiams=Fin3)      #8
#NOW FOR CORR FLXS
wu.TFcfEWSflux(dfA,0,130,'diam',aprcorrcf,tableDiams=Fin3)   #9-
wu.TFcfEWSflux(dfA,0,130,'tau',aprcorrcf,tableDiams=Fin3)    #10
wu.TFcfEWStau(dfA,0,130,'diam',aprcorrcf,tableDiams=Fin3)    #11-
wu.TFcfEWStau(dfA,0,130,'tau',aprcorrcf,tableDiams=Fin3)     #12
wu.TFcfEWSmjd(dfA,0,130,'diam',aprcorrcf,tableDiams=Fin3)    #13 #missing 2 bls  CHECK
wu.TFcfEWSmjd(dfA,0,130,'tau',aprcorrcf,tableDiams=Fin3)     #14
wu.TFcfEWSbl(dfA,0,130,'diam',aprcorrcf,tableDiams=Fin3)     #15
wu.TFcfEWSbl(dfA,0,130,'tau',aprcorrcf,tableDiams=Fin3)      #16
#PHOTOMETRY   PENDING
wu.TFphotEWSflux(dfA,0,130,'tau',correction,tableDiams=Fin3)   #9-

#CONCLUSION:


#*******END SELECTION CALIBRATORS
#*******CALIBRATING AGNS
##May 17,18,20,21
# Around 3.2,3.4,3.6,3.8  CHECK vis looks good with wl edges!!!!!
#3.1-4.1
# get pickles from table indexpickles, os.chdir('TablesPicklesScripts')

SELECTION Cals:  Mandar updated table
              delPhe 78Jy Cal for NGC 1068 Sept22 delPheSept22sc.pk 2018-09-22T03:30:30
              delPhe 78Jy Cal for NGC 1566 Sept23 delPheSept23sc.pk 2018-09-23T07:32:18
              HD 25286 1.9Jy Cal for NGC 1365  Sept24 HD25286sc.pk 2018-09-24T09:08:58
              HD 25286 1.9Jy Cal for NGC 1068 Sept24 HD25286sc.pk 2018-09-24T09:08:58
              HD 20356 28Jy Cal for NGC 1068 Sept25 HD20356sc.pk 2018-09-25T06:20:02
              HD 20356 28Jy Cal for NGC 7469 Sept25 HD20356sc.pk 2018-09-25T06:20:02
              

delPheSept22sc=pd.read_pickle('delPheSept22sc.pk')
viidp22, voodp22 =wu.arraytocalews(delPheSept22sc,Fin2)
DF=pd.read_pickle('May14NGC1068ews.pk')
Fin2=pd.read_pickle('/Users/M51/TableDiamsSeptmsdfcc19.pk')
 # 0 is ININ, 3 is OUTOUT
#RETURNS pandas data frame with calibrated vis, chrom ph and cl ph, also u1,v2,u2,v2, over all wls. Includes ALL mjds  :D #3.21 to 3.81:
DF1068=wu.viswlewsagn(DF,50,100,30,130,0,135,viidp22,voodp22,0,9) #9ischopped
DF1068.to_pickle('Jun4NGC1068Sept22Cal.pk')
DF1068.shape  #9,30  includes all
#NGC 1566     PENDING to do selection mjds for AGNs... not finished
delPheSept23sc=pd.read_pickle('delPheSept23sc.pk')
viidp23, voodp23 =wu.arraytocalews(delPheSept23sc,Fin2)
DF=pd.read_pickle('May14NGC1566ews.pk')
# 0 is ININ, 3 is OUTOUT
#RETURNS pandas data frame with calibrated vis, chrom ph and cl ph, also u1,v2,u2,v2, over all wls. Includes ALL mjds  :D #3.21 to 3.81:
DF1566=wu.viswlewsagn(DF,50,100,30,130,0,135,viidp23,voodp23,0,11)
DF1566.to_pickle('Jun4NGC1566Sept23Cal.pk')
DF1566.shape  #9,27  includes all
#NGC 156
              delPheSept23sc=pd.read_pickle('delPheSept23sc.pk')
              viidp23, voodp23 =wu.arraytocalews(delPheSept23sc,Fin2)
              DF=pd.read_pickle('May14NGC1566ews.pk')
              # 0 is ININ, 3 is OUTOUT
              #RETURNS pandas data frame with calibrated vis, chrom ph and cl ph, also u1,v2,u2,v2, over all wls. Includes ALL mjds  :D #3.21 to 3.81:
              DF1566=wu.viswlewsagn(DF,50,100,30,130,0,135,viidp23,voodp23,0,11)
              DF1566.to_pickle('Jun4NGC1566Sept23Cal.pk')
              DF1566.shape  #9,27  includes all
              
              
#AND PLOTS: vis,chromphase,closphase  #3.21 to 3.81:
#it will get rid of bad mjds but not chopped ones
#around 3.2   (3.1 to 3.3 , = 95 to 87)
#around 3.4  (3.3 to 3.5 ,=  87 to 79)
#around 3.6  (3.5 to 3.7, = 79 to 70)
#around 3.8  (3.7 to 3.9, = 70 to 61)
#BY BINS for each range of wl: [30,45,60,75,90,105,120,135] 7 bins bl lengths /  for angles in rad
#wl1,wl2,bl1,bl2,ang1[deg],ang2[deg]  plots only between i1 and i2 #9ischopped
#np.mean(dat),np.var(dat),np.max(dat)-np.mean(dat)
dummy1=wu.viswlewsagn(DF,50,100,25,45,0,45,vii,voo,0,8,toplot='vis') #5 h and noisier, spread@3.5um: 0.18424051386236368 0.0011167667171687538 0.06755478797907322    7
dummy1=wu.viswlewsagn(DF,50,100,25,45,0,45, viidp22,voodp22,0,8,toplot='vis') #5 h and noisier, spread@3.5um: 0.18424051386236368 0.0011167667171687538 0.06755478797907322    7

dummy1=wu.viswlewsagn(DF,50,100,25,45,0,45,vii,voo,0,8,toplot='chromphase')
dummy1=wu.viswlewsagn(DF,50,100,25,45,0,45, viidp22,voodp22,0,8,toplot='chromphase')
#5.080674958462422 2.1626048140704013 1.604905641709041
dummy2=wu.viswlewsagn(DF,50,100,45,65,0,45,vii,voo,0,8,toplot='vis')
#0.16685019594840786 0.000543484420758606 0.03324338459339518   7
dummy2=wu.viswlewsagn(DF,50,100,45,65,0,45, viidp22,voodp22,0,8,toplot='vis')
#0.16685019594840786 0.000543484420758606 0.03324338459339518   7
dummy2=wu.viswlewsagn(DF,50,100,45,65,0,45, viidp22,voodp22,0,8,toplot='chromphase')
#13.255041069231792 30.441451515611938 6.239882644941554  #??
dummy3=wu.viswlewsagn(DF,50,100,65,85,0,45,vii,voo,0,8,toplot='vis')
#0.17510775467501932 0.0004242177209588848 0.03436640842171851  8
dummy3=wu.viswlewsagn(DF,50,100,65,85,0,45, viidp22,voodp22,0,8,toplot='vis')
#0.17510775467501932 0.0004242177209588848 0.03436640842171851  8
dummy3=wu.viswlewsagn(DF,50,100,65,85,0,45,viidp22,voodp22,0,8,toplot='chromphase')
#14.447198617171157 5.622590435795349 5.575520276694462
dummy4=wu.viswlewsagn(DF,50,100,45,65,45,90,vii,voo,0,8,toplot='vis')
#0.02515563090119851 1.2333542879697795e-05 0.0068429364506186405  7
dummy4=wu.viswlewsagn(DF,50,100,45,65,45,90, viidp22,voodp22,0,8,toplot='vis')
#0.02515563090119851 1.2333542879697795e-05 0.0068429364506186405  7
dummy4=wu.viswlewsagn(DF,50,100,45,65,45,90, viidp22,voodp22,0,8,toplot='chromphase')                             #WOOOWWWW!!!
#24.840291319101834 13.954334090198845 4.48986760686568
dummy5=wu.viswlewsagn(DF,50,100,65,85,45,90,vii,voo,0,8,toplot='vis')
#0.2375313109361151 0.0015123456777886076 0.051623841645696694    6
dummy5=wu.viswlewsagn(DF,50,100,65,85,45,90, viidp22,voodp22,0,8,toplot='vis')
#0.2375313109361151 0.0015123456777886076 0.051623841645696694    6
dummy5=wu.viswlewsagn(DF,50,100,65,85,45,90, viidp22,voodp22,0,8,toplot='chromphase')
#11.532553756829323 6.646390624143969 4.550218670440994
dummy6=wu.viswlewsagn(DF,50,100,25,65,90,135,vii,voo,0,8,toplot='vis')
#0.09970536439028203 0.0006044534828848272 0.04215099749274  7
dummy6=wu.viswlewsagn(DF,50,100,25,65,90,135, viidp22,voodp22,0,8,toplot='vis')
#0.09970536439028203 0.0006044534828848272 0.04215099749274  7
#Whole range: 3.2 to 3.8
dummy6=wu.viswlewsagn(DF,50,100,25,65,90,135, viidp22,voodp22,0,8,toplot='chromphase')  #Different

#13.75777726122998 10.62224338748525 4.748694264445836
#For closure phases all the BLS and ANGLES are used (ignores bli and angi)
dummy1=wu.viswlewsagn(DF,50,100,25,45,0,45, viidp22,voodp22,1,2,toplot='closphase')
dummy1=wu.viswlewsagn(DF,50,100,25,45,0,45, viidp22,voodp22,1,8,toplot='closphase')
              
              
#PLOT UVPLANE
#ALLbls:30-130m Allangles:0-115(deg)
#wave has 110 wls, from  2.73815408808 to 4.9670940904575 or 64
              100->2.98
              91->3.21
              67->3.81
              50->4.12
#3.21 to 3.81:  using DF1068 because it has been calibrated
wu.visewsagn(DF1068,67,91,viidp22,voodp22,0,9) #To plot visibilities on the uv plane, color code is the visibility,  10 is chopped!!
#CHECK right wls to average in,
#3.21 to 3.81:
wu.visewsagn(DF1068,67,91,viidp22,voodp22,1,9,avg='True')  #To plot visibilities on the uv plane, color code is the visibility,  USING AVGS  0 is in red in table
wu.visewsagn(DF1068,67,91, viidp22,voodp22,0,9,avg='True')
# Around 3.2,3.4,3.6,3.8  CHECK vis looks good with wl edges!!!!!
              
#PENDING TO RE_DO...
S22x1,S22y1,S22z1=wu.visewsagn(DF,87,95,vii,voo,0,8,avg='True') #last 2 entries are to have the same color range
S22x2,S22y2,S22z2=wu.visewsagn(DF,79,87,vii,voo,0,8,avg='True')
S22x3,S22y3,S22z3=wu.visewsagn(DF,70,79,vii,voo,0,8,avg='True')
S22x4,S22y4,S22z4=wu.visewsagn(DF,61,70,vii,voo,0,8,avg='True')


#3.59213308377 to 3.77215650585  returns vector with averages and uv points
S22x2,S22y2,S22z2=wu.visewsagn(DF,67,75,vii,voo,0,8,avg='True')  #To plot visibilities on the uv plane, color code is the visibility,  USING AVGS
[10.336821607907144, 2.1773475112471252, 12.527426484431045, 57.74103008781093, 47.40973758339904, 59.9176074742527, 105, 105]
[32.4132246074734, 46.167057029755036, 78.58006238705032, 12.182430416137493, -20.23716927469514, 58.35644904739005, 105, 105]
[0.19017443878595158, 0.1899452541563472, 0.19699528102536012, 0.0359967230792409, 0.11579285701308398, 0.2559505741749046, 0.28, 0.015]#<----  pending

avgallwlsII=np.empty([4,9],dtype=object)
avgallwlsOO=np.empty([4,9],dtype=object)
for i in range(4):  #II   4 closure phases
    k=0
    for n in range(67,76):
        avgallwlsII[i][k]=np.mean([DF1068['calclph'][1][i][n],DF1068['calclph'][3][i][n],DF1068['calclph'][5][i][n],DF1068['calclph'][7][i][n]])
        avgallwlsOO[i][k]=np.mean([DF1068['calclph'][2][i][n],DF1068['calclph'][4][i][n],DF1068['calclph'][6][i][n],DF1068['calclph'][8][i][n]])
        k+=1

aclphii=np.zeros([4])  #array([0.01286546, 0.17555723, 0.09889884, 0.26159062])    #[-0.10952522,  0.07373101,  0.01207774,  0.19533397])
aclphoo=np.zeros([4])
for i in range(4):  #4   closure phases
    aclphii[i]=np.mean(avgallwlsII[i])
    aclphoo[i]=np.mean(avgallwlsOO[i])

for i in range(4):
    #print(aclphii[i]*(180/np.pi))  #around 7 deg different only :)
    #print(aclphoo[i]*(180/np.pi))
    #print(np.mean([aclphii[i]*(180/np.pi),aclphoo[i]*(180/np.pi)]))
    #[-2.7690980317958234, 7.141582111275697, 3.179245030919417, 13.089925173990933].  #in DEG
    print(np.mean([aclphii[i],aclphoo[i]]))
    #clph=[ -0.04832987796533174, 0.12464412164328897, 0.05548829351721275, 0.22846229312583338]  #IN RAD


0.7371366759931437
-6.27533273958479
10.058688625494455
4.224475597056939
5.66648631520991
0.6920037466289245
14.988038264711216
11.19181208327065
#FOR CLOSURE PHASES USE ALL RANGE OF BLS AND ANGLES!!
#range of wls: 3.21 to 3.81
closph=wu.viswlewsagn(DF,67,91,30,110,0,115,vii,voo,1,9)  #9,4
clphii=[np.mean([closph[1][0],closph[3][0],closph[5][0],closph[7][0]]),np.mean([closph[1][1],closph[3][1],closph[5][1],closph[7][1]]),np.mean([closph[1][2],closph[3][2],closph[5][2],closph[7][2]]),np.mean([closph[1][3],closph[3][3],closph[5][3],closph[7][3]])]
clphoo=[np.mean([closph[2][0],closph[4][0],closph[6][0],closph[8][0]]),np.mean([closph[2][1],closph[4][1],closph[6][1],closph[8][1]]),np.mean([closph[2][2],closph[4][2],closph[6][2],closph[8][2]]),np.mean([closph[2][3],closph[4][3],closph[6][3],closph[8][3]])]
[0.2258162091007355,
               -0.030000360143742216,
               0.2685352841595194,
               0.012718714915041654]
#END
              
              
              
              #TESTS CALIBRATING A CALIBRATOR   #MAY 24
vii1=[1,1,1,1,1,1]
voo1=[1,1,1,1,1,1]
dummy1=wu.viswlewsagn(HD16658,50,100,0,130,0,135,vii1,voo1,0,6,toplot='vis')
              
              
#del Phe
dummy1=wu.viswlewsagn(delPhe, 50,100,0,130,0,135, vii1,voo1,0,6,toplot='vis')
dummy1=wu.viswlewsagn(delPhe,  50,100,0,130,0,135, viidp,voodp,0,6,toplot='vis')
dummy1=wu.viswlewsagn(delPhe,  50,100,0,130,0,135, vii1,voo1,0,6,toplot='chromphase')
dummy1=wu.viswlewsagn(delPhe,  50,100,0,130,0,135, viidp,voodp,0,6,toplot='chromphase')

              
              
dummy1=wu.viswlewsagn(HD16658,50,100,0,130,0,135,vii,voo,0,6,toplot='vis')
dummy1=wu.viswlewsagn(HD16658,50,100,0,130,0,135,viidp,voodp,4,5,toplot='vis')
   #CHANGE ylim

dummy1=wu.viswlewsagn(delPhe, 50,100,0,130,0,135,vii,voo,0,9,toplot='vis')


dummy1=wu.viswlewsagn(HD16658,50,100,0,130,0,135,vii,voo,1,2,toplot='chromphase')
dummy1=wu.viswlewsagn(HD16658,50,100,0,130,0,135,viidp,voodp,1,2,toplot='chromphase')


HD25286=df[df['targ']=='HD25286']   #Sept 24 NGC1068 2018-09-24T09:08:58
#WRONG: 209240 intended to be for 1068 Sept23-24  @03:16
# HD 20356 was the cal for NGC 1068, but it did not pass Quality Tests
#tau0 between 4 and 8.5
HD25286=HD25286.reset_index()  #12,20  1.9Jy
vii2,voo2=wu.arraytocal(HD25286,Fin2)
dummy1=wu.viswlewsagn(HD25286, 50,100,0,130,0,135,vii1,voo1,0,12,toplot='vis')
dummy1=wu.viswlewsagn(HD25286, 50,100,0,130,0,135,vii2,voo2,0,12,toplot='vis')
# SEPTEMBER CALIBRATORS
delPhe=df[df['targ']=='del Phe']
amjd=[58384.31596017, 58384.31937302, 58384.32040372, 58384.32355232]
for i in range(np.size(amjd)):
    delPhe=delPhe[delPhe['mjd-obs']!=amjd[i]]
delPhe=delPhe.reset_index()
viidp,voodp=wu.arraytocal(delPhe,Fin2)
dummy1=wu.viswlewsagn(delPhe, 50,100,0,130,0,135,vii1,voo1,0,12,toplot='vis')
dummy1=wu.viswlewsagn(delPhe, 50,100,0,130,0,135,viidp,voodp,0,12,toplot='vis')
dummy1=wu.viswlewsagn(HD16658,50,100,0,130,0,135,viidp,voodp,0,6,toplot='vis')
dummy1=wu.viswlewsagn(delPhe, 50,100,0,130,0,135,vii,voo,0,12,toplot='vis')




dummy1=wu.viswlewsagn(HD16658,50,100,25,45,0,45,vii1,voo1,0,6,toplot='closphase')
dummy1=wu.viswlewsagn(HD16658,50,100,25,45,0,45,viidp,voodp,0,6,toplot='closphase')





bSgr=df[df['targ']=='H']





#12, 14, 15, 16  May
#PREP
#PLOTTING CHROMATIC PHASES
DF=pd.read_pickle('May14NGC1068ews.pk')
Fin2=pd.read_pickle('/Users/M51/TableDiamsSeptmsdfcc19.pk')
#SELECTION CALIBRATORS
DF2=pd.read_pickle('/Users/M51/May14SeptCalsEWS.pk') # (166, 19)
df=DF2[DF2['targ']!='CD-55 855']  #159,18
df=df[df['targ']!='b Sgr']  #150
amjd=[58381.04560605,58381.05829687,58381.11948348,58381.15815322, 58382.12718112,58382.13937325,58382.14056971,58382.14151070, 58382.14244882,58382.14338623,58382.14647846, 58383.13341208,58383.13461323,58383.13554598,58383.13647738, 58383.13741494,58383.13836445,58383.13942234,58383.14308607, 58383.14435676,58383.14797656, 58383.15507120, 58383.20571736, 58384.07957074, 58384.17850135, 58384.98769468, 58385.00456004, 58385.38429503, 58385.38560493, 58386.03318726, 58386.07974087, 58386.08855096, 58386.10701578, 58386.11361284, 58386.11911919, 58386.18925231,58386.19161547, 58386.19635368, 58386.20523466, 58386.40622892, 58386.4108629,  58385.35656910, 58385.36001311, 58385.36108241, 58385.36422807, 58385.36542380]   #41 Lst 6 HD20356
for i in range(np.size(amjd)):
    df=df[df['mjd-obs']!=amjd[i]]
    print(amjd[i])
    print(df.shape)  #96,18   #104,19

df=df.reset_index()   #104,20
#NGC 1068 22 Sept
HD16658=df[df['targ']=='HD 16658']   #Sept 22 NGC1068 2018-09-22T04:44:55
HD16658=HD16658.reset_index()  #6,20   1Jy
vii,voo=wu.arraytocal(HD16658,Fin2) #6,110Sept 22  averaged over frames but not over wl
#USE FOR ALL BLS: [30,45,60,75,90,105,120,135]    7 bins
#ALLbls:30-130m Allangles:0-115(deg)
#wave has 110 wls, from  2.73815408808 to 4.9670940904575
100->2.98
91->3.21
67->3.81
50->4.12
#3.21 to 3.81:
wu.visewsagn(DF,67,91,vii,voo,0,9) #To plot visibilities on the uv plane, color code is the visibility,  10 is chopped!!
#CHECK right wls to average in,
#3.21 to 3.81:
wu.visewsagn(DF,67,91,vii,voo,0,9,avg='True')  #To plot visibilities on the uv plane, color code is the visibility,  USING AVGS
Didnt use 0
1 IN IN
2 OUT OUT
3 IN IN
4 OUT OUT
5 IN IN
6 OUT OUT
7 IN IN
8 OUT OUT
[10.872072617172773, 2.7716509845342374, 13.613178401342546, 58.72723376656111, 47.85927714589538, 61.485781941725264]
[32.28718071232128, 45.98620103068335, 78.27855991536596, 12.135881127866504, -20.15723917353567, 58.128909571543204]
[0.15660571518833083, 0.15111637778400155, 0.20628468625769258, 0.03240795727006722, 0.10560381519085443, 0.23115704912868718]
wu.viswlewsagn(DF,50,100,30,100,0,115,vii,voo,0,2)  #calibrated chromatic phases vs wl, color coded by mjd, BETWEEN BLS & ANGLES.
wu.viswlewsagn2(DF,50,100,0,25,25,50,vii,voo,0,9) #calibrated chromatic phases vs wl, color coded by mjd, BETWEEN CARTESIAN COORDS.


#FOR CLOSURE PHASES USE ALL RANGE OF BLS AND ANGLES!!
#range of wls: 3.21 to 3.81
closph=wu.viswlewsagn(DF,67,91,30,110,0,115,vii,voo,1,9)  #9,4
clphii=[np.mean([closph[1][0],closph[3][0],closph[5][0],closph[7][0]]),np.mean([closph[1][1],closph[3][1],closph[5][1],closph[7][1]]),np.mean([closph[1][2],closph[3][2],closph[5][2],closph[7][2]]),np.mean([closph[1][3],closph[3][3],closph[5][3],closph[7][3]])]
clphoo=[np.mean([closph[2][0],closph[4][0],closph[6][0],closph[8][0]]),np.mean([closph[2][1],closph[4][1],closph[6][1],closph[8][1]]),np.mean([closph[2][2],closph[4][2],closph[6][2],closph[8][2]]),np.mean([closph[2][3],closph[4][3],closph[6][3],closph[8][3]])]
[0.2258162091007355,
 -0.030000360143742216,
 0.2685352841595194,
 0.012718714915041654]

df=wu.viswlewsagn(DF,67,91,30,110,0,115,vii,voo,1,9)

#USING WHOLE RANGE 50,100
[0.1540355462430816, 0.15601892609516294, 0.21204332913545332, 0.03841630918817976, 0.11130872192090943, 0.23399241884468824]
#around 3.2   (3.1 to 3.3 , = 95 to 87)
wu.visewsagn(DF,87,95,vii,voo,0,9,avg='True')  #USING AVGS
[0.1369007327146784, 0.11179575691227503, 0.14911826651570192, 0.01976539832674824, 0.07527830749419188, 0.19443387893324068]
#around 3.4  (3.3 to 3.5 ,=  87 to 79)
wu.visewsagn(DF,79,87,vii,voo,0,9,avg='True')  #USING AVGS
[0.15460378689948556, 0.1413813961002711, 0.20135294175674315, 0.02974531203403489, 0.09547894966887338, 0.22440164063083143]
#around 3.6  (3.5 to 3.7, = 79 to )
wu.visewsagn(DF,,79,vii,voo,0,9,avg='True')  #USING AVGS
#around3.8

wu.viswlewsagn(DF,50,100,30,100,0,115,vii,voo,0,9)  #calibrated chromatic phases vs wl, color coded by mjd, BETWEEN BLS & ANGLES
wu.viswlewsagn(DF,50,100,30,100,0,115,vii,voo,0,9)
wu.viswlewsagn(DF,50,100,30,100,0,115,vii,voo,1,2)


#BINS for Sept 22 NGC 1068: see visewsagn
wu.viswlewsagn2(DF,50,100,0,25,25,50,vii,voo,0,9) #Calibrated visibilities or calibrated chromatic phases vs wl, color coded by mjd between two ranges of CARTESIAN COORDS.

#fsel in ve CHOPPING test   in ve, it uses
#def chopchop(file,*args,**kwargs):
def chopchop(file, choptest=False):
    #   sky=kwargs.get('sky',False)
    if (not choptest):
        if (fu.getkeyword('chopfreq',file,fill=0)>0): return 1
        else:return 0
    dprtype = fu.getkeyword('dprtype',file)
    sky     = 'SKY' == dprtype
    data    = fu.getData(file,"tartyp")
    if not sky :
        count=np.sum(data[:100]=='S')
    else:
        count=np.sum(data[:100]=='T')
    if (count>5):
        return 1  #It was chopping
    else:
        return 0 #Not chopping
#PENDING: MARK DUBIOUS MJDS/
df2=wu.tableEWS('/Volumes/LaCie/SeptAGN/NGC1068') #(22, 18)
df2.to_pickle('May13NGC1068ews.pk')
for i in range(df2.shape[0]):
    print(df2[['targ','file','tau','mjd-obs']][i:i+1])
    cf=df2['cflux'][i]
    plt.imshow(cf['opdimage'],vmin=0.,vmax=1500.)  #cf['opdimage'].shape (271, 768)
    plt.pause(2)

#THE plot
#Dropbox is fine or the Sterrewacht-version of it, https://owncloud.strw.leidenuniv.nl/index.php/login -- login with your normal login)?
#Plot uv plane:
'''img2vis
- point source
- 2D Gauss distribution (elongated)
- 2-fold, 3-fold combinations of these

My tool to produce visibilities from arbitrary images:
https://github.com/astroleo/img2vis

Modeling with Astropy:
http://docs.astropy.org/en/stable/modeling/index.html#id1

wu.modelvis(u,v,vis,fftscale,roll)'''
#NGC 1068
DF=pd.read_pickle('May14NGC1068ews.pk') #22,18
#DF=DF[DF['mjd-obs']>58383.16988163]  # (21, 18)
df1=df2[df2['mjd-obs']==58383.18053153]
amjds=[58386.15734864, 58383.16988163, 58383.18053153, 58383.18334492, 58383.18439130, 58385.23524700, 58386.232164]
plt.imshow(cf['opdimage'],vmin=0,vmax=2000)
#DF=DF.reset_index()
#To avoid 58383.16988163 and 58383.18439130 use index 1 to 8
#2018-09-22T04:01:55  DF 0-9
#2018-09-24T05:21:50  DF 10-14
#2018-09-25T05:21:36  DF 15-21
for i in range(DF.shape[0]):
    print(DF[['targ','file','tau','mjd-obs']][i:i+1])
    cf=DF['cflux'][i]
    plt.imshow(cf['opdimage'],vmin=0.,vmax=1500.)  #cf['opdimage'].shape (271, 768)
    plt.pause(2)

#PASS DF calibrated!!!!
#Use SELECTION CALIBRATORS df=...    # (104, 18)
HD16658=df[df['targ']=='HD 16658']   #Sept 22 NGC1068 2018-09-22T04:44:55
HD16658=HD16658.reset_index()  #6,20   1Jy
HD25286=df[df['targ']=='HD25286']   #Sept 24 NGC1068 2018-09-24T09:08:58
#WRONG: 209240 intended to be for 1068 Sept23-24  @03:16
# HD 20356 was the cal for NGC 1068, but it did not pass Quality Tests
#tau0 between 4 and 8.5
HD25286=HD25286.reset_index()  #12,20  1.9Jy
HD20356=df[df['targ']=='HD 20356']   #Sept 25  2018-09-25T06:20:02 Closer in time but it has 28Jy
HD20356=HD20356.reset_index()
#ALSO TRY:  It is 8.4 Jy
HD209240=df[df['targ']=='HD209240'] #Sept 25  2018-09-25T02:37:24 Has 28Jy
HD209240=HD209240.reset_index()

Fin2=pd.read_pickle('/Users/M51/TableDiamsSeptmsdfcc19.pk')
wu.viswlewscal(HD16658,'wave',50,100,0,130,tableDiams=Fin2)
#Get different BCDs matched

vii,voo=wu.arraycalagn(HD16658,Fin2) #6,110Sept 22  averaged over frames but not over wl
wu.visewsagn(DF,50,100,vii,voo,0,9)        #Range for TPL-START
wu.visewsagn(DF,50,100,vii,voo,1,8)
#CHECK ANGLE GOES TO RIGHT DIRECTION! (left, right)
vii2,voo2=wu.arraycalagn(HD25286,Fin2)   #Sept 24
wu.visewsagn(DF,50,100,vii2,voo2,10,14)   #Range for TPL-START
#CHECK ANGLE GOES TO RIGHT DIRECTION! (left, right)
vii3,voo3=wu.arraycalagn(HD20356,Fin2)   #Sept 25
wu.visewsagn(DF,50,100,vii3,voo3,15,21)   #Range for TPL-START
#CHECK ANGLE GOES TO RIGHT DIRECTION! (left, right)
vii4,voo4=wu.arraycalagn(HD209240,Fin2)  #Sept 25
wu.visewsagn(DF,50,100,vii4,voo4,15,21)   #Range for TPL-START
#CHECK ANGLE GOES TO RIGHT DIRECTION! (left, right)


#NGC 1365
DF=pd.read_pickle('25Aprews1365.pk')  #14,18 AllSept24
vii2,voo2=wu.arraycalagn(HD25286,Fin2)
wu.visewsagn(DF,50,100,vii2,voo2,0,14)

#NGC 1566
DF=pd.read_pickle('25Aprews1566.pk')  #(11, 18) #Sept 23#2018-09-23T08:08:57
HD195677=df[df['targ']=='HD 195677']
HD195677=HD195677.reset_index()
vii5,voo5=wu.arraycalagn(HD195677,Fin2) #6,110Sept
wu.visewsagn(DF,50,100,vii5,voo5,0,11)   #vis too high!!

#NGC 7469
DF=pd.read_pickle('25Aprews7469.pk')  #(7, 18) Sept 25 MATIS.2018-09-25T03:40:20.197.fits
vii4,voo4=wu.arraycalagn(HD209240,Fin2)  #Sept 25
wu.visewsagn(DF,50,100,vii4,voo4,0,7)

HD 172453??  Looks dubious, but try it...


#---------------------------------------------
t1=np.abs(HD16658['vis'][0])
t2=np.abs(HD16658['vis'][2])
t3=np.abs(HD16658['vis'][4])
cii=[0,2,4]
cavgvii=np.mean(np.abs(HD16658['vis'][cii]))  #6,110
t=[]
for j in range(110):
    t.append(np.mean([t1[0][j],t2[0][j],t3[0][j]]))

cavgvii[0]-t =   0...  :)
cii[0]-t = 0... :)


l = .0000038
print (l,"lambda in microns")
c = 1.0/l
cmap = plt.get_cmap('jet_r')
#
delta0 =  math.radians(DF["dec"][i]) #dec of the source in deg
print (DF["dec"][i], delta0, "dec of the source in deg and rad")
a=(D)*cos(deltaB)  #major axis: almost 0
print (a, "major axis")
b=(D)*cos(deltaB)*sin(delta0) #minor axis: almost 0
print (b, "minor axis")
v0=(D)*sin(deltaB)*cos(delta0) # center over v axis (u center=0)
print (v0, "center on v axis")
deltaB = math.rad ians(-90) #dec of the baseline E-W=-90, E-W baselines have no v
theta=l/D #rad
u=1/thetax
v=1/thetay

t = np.linspace(-0.9*pi, .5*pi, 400)
plt.plot(c*cos(deltaB)*sin(t),c*(sin(deltaB)*cos(delta0)-cos(deltaB)*sin(delta0)*cos(t)),color='red')
plt.show()


for i in range(7):
    for j in range (6):
        c = cmap(float(j)/6)
        plt.plot(df['UCOORD'][i][j],df['UCOORD'][i][j],'bo', color=c,markersize=3)
        plt.plot(-1.0*df['UCOORD'][i][j],-1.0*df['UCOORD'][i][j],'bo',color=c,markersize=3)  # u,v for which lambda???

#installing DRS
if you would like to install the latest "tagged" release (actually we
                                                          tagged version 1.1.5 today) you can already do it by following the
instruction written in the email (see below)

(Please note that we have renamed matis into matisSE recently so you
 have to install the packages with the full name - there is also no
 version 1.1.5 with the short name)

You could also use the install_esoreflex script
ftp://ftp.eso.org/pub/dfs/reflex/install_esoreflex [1] and install it
with

./install_esoreflex -r testing (will install the latest tagged version
                                1.1.5)

./install_esoreflex -r devel (will install the latest version from svn
                              where I created a kit  )

In both cases you have to follow the instruction and install the MATISSE
pipeline (and not the MATIS).

Follow the installation instruction on

http://www.eso.org/sci/software/pipelines/installation/rpm.html [2]

or

http://www.eso.org/sci/software/pipelines/installation/macports.html [3]


to install matisse pipeline version 1.0.2 on fedora or on a mac,
respectively

The only thing you have to do is replace in all the commands

STABLE with TESTING e.g.

curl ftp://ftp.eso.org/pub/dfs/pipelines/repositories/
[4]STABLE/macports/setup/Portfile -o Portfile

becomes
curl ftp://ftp.eso.org/pub/dfs/pipelines/repositories/
[4]TESTING/macports/setup/Portfile -o Portfile

or

sudo dnf config-manager
--add-repo=ftp://ftp.eso.org/pub/dfs/pipelines/repositories/
[4]STABLE/fedora/esorepo.repo

becomes

sudo dnf config-manager
--add-repo=ftp://ftp.eso.org/pub/dfs/pipelines/repositories/
[4]TESTING/fedora/esorepo.repo

#10th May
scp 2018*tpl.pk gamez@ssh.strw.leidenuniv.nl:/disks/cool1/jaffe/drs/gamez/RedDataDec/v.Apr15
scp -r /Volumes/LaCie/SeptemberData/RedLast gamez@ssh.strw.leidenuniv.nl:/disks/cool1/jaffe/drs/gamez/RedDataDec/v.Apr15    #Yessss!!!! :)

ssh ssh.strw.leidenuniv.nl -p 80 -l gamez  #Yessss!!!! :)
chmod 777 -R *.tpl*


#FOR AGNS Quality control
DF=pd.read_pickle('25Aprews1068')
#vis
wu.viswlewsagn(DF,50,100,0,130)
#vis by bl length and angle
OUT-OUT: 34,12,23,24,13,14
IN-IN: 3<->4, 1<->2
#To save the angles of the bls in the same order as pbls
aangles=np.zeros([DF.shape[0], 6])
angles1=[]
pbls1=[]
for i in range(DF.shape[0]):
    for i in range(DF.shape[0]):
        pbls=[DF['header'][i]["HIERARCH ESO ISS PBL12 START"], DF['header'][i]["HIERARCH ESO ISS PBL13 START"], DF['header'][i]["HIERARCH ESO ISS PBL14 START"], DF['header'][i]["HIERARCH ESO ISS PBL23 START"], DF['header'][i]["HIERARCH ESO ISS PBL24 START"], DF['header'][i]["HIERARCH ESO ISS PBL34 START"]]
        pblas=[DF['header'][i]["HIERARCH ESO ISS PBLA12 START"], DF['header'][i]["HIERARCH ESO ISS PBLA13 START"], DF['header'][i]["HIERARCH ESO ISS PBLA14 START"], DF['header'][i]["HIERARCH ESO ISS PBLA23 START"], DF['header'][i]["HIERARCH ESO ISS PBLA24 START"], DF['header'][i]["HIERARCH ESO ISS PBLA34 START"]]
        for j in range(6):
            for k in range(6):
                if (DF['pbl'][i][j]==pbls[k]):
                    aangles[i][j]=pblas[k]
                    angles1.append(pblas[k])
                    pbls1.append(pbls[k])

np.size(angles1)  # 2904

for i in range(22):
    for j in range(6):
        plt.plot(aangles[i][j],'o')


for i in range(22):
    plt.plot(aangles[i],'o')

plt.hist(angles1,bins=20)
plt.hist(angles1,bins=6)
maxa=np.max(angles1)  #114.133  u=bl*sinposang,v=bl*cosposang
mina=np.min(angles1)  #0.2
#0-120
#mybins = np.linspace(0, 120, 6)   NO!
plt.hist(angles,bins=[0,20,40,60,80,100,120])

maxbl=np.max(pbls1)  #109.815
minbl=np.min(pbls1)  #33.479
jump=(maxbl-minbl)/6  #76.336   12.722666666666667
binsbls=[minbl+0*jump,minbl+1*jump,minbl+2*jump,minbl+3*jump,minbl+4*jump,minbl+5*jump,minbl+6*jump]
#[33.479,46.20166666666667,58.92433333333334,71.64699999999999,84.36966666666666,97.09233333333333,109.815]
plt.hist(pbls1,bins=6)
plt.hist(pbls1,bins=binsbls)
#SEPARATE BY TPL START
DF['tps-']
for i in range(DF.shpe([0]):
    df1=
#Scale to 0.6  wl/bl=resolution  3um/100m=3mas
#size src*wl=diff uv coord w/diff vis. ->normalize by the resolution.
#/  diff in bl 100/5=20m
#10-20% lower with 4-6 tau  ??
#check airmass!!  NGC 1068
#not-calibrated
#USE: [33.4,46.20,58.92,71.65,84.34,97.09,109.82]
wu.viswlewsagn(DF,50,100,33.4,46.20,0,20,vii,voo)     #9-h,   0-l
20,40,60)    #nothing
wu.viswlewsagn(DF,50,100,33.4,46.20,60,80)    #nothing
wu.viswlewsagn(DF,50,100,33.4,46.20,80,100)   #nothing
wu.viswlewsagn(DF,50,100,33.4,46.20,100,120)  #nothing
#---
wu.viswlewsagn(DF,50,100,46.20,58.92,0,20)    #9,14 higher/2 main groups!!
wu.viswlewsagn(DF,50,100,46.20,58.92,20,40)   #nothing
wu.viswlewsagn(DF,50,100,46.20,58.92,40,60)   #nothing
wu.viswlewsagn(DF,50,100,46.20,58.92,60,80)   #all below 0.25
wu.viswlewsagn(DF,50,100,46.20,58.92,80,100)  #nothing
wu.viswlewsagn(DF,50,100,46.20,58.92,100,120) #0,9 a little higher
#---
wu.viswlewsagn(DF,50,100,58.92,71.65,0,20)    #nothing
wu.viswlewsagn(DF,50,100,58.92,71.65,20,40)   #nothing
wu.viswlewsagn(DF,50,100,58.92,71.65,40,60)   #nothing
wu.viswlewsagn(DF,50,100,58.92,71.65,60,80)   #9-hnoiser, all below ~0.5
wu.viswlewsagn(DF,50,100,58.92,71.65,80,100)  #nothing
wu.viswlewsagn(DF,50,100,58.92,71.65,100,120) #14,21hnoisier/all<0.25
#---
wu.viswlewsagn(DF,50,100,71.65,84.34,0,20)    #9-h,0-l
wu.viswlewsagn(DF,50,100,71.65,84.34,20,40)   #nothing
wu.viswlewsagn(DF,50,100,71.65,84.34,40,60)   #0-l
wu.viswlewsagn(DF,50,100,71.65,84.34,60,80)   #nothing
wu.viswlewsagn(DF,50,100,71.65,84.34,80,100)  #14 little higher/all<~0.5
wu.viswlewsagn(DF,50,100,71.65,84.34,100,120) #nothing
#---
wu.viswlewsagn(DF,50,100,84.34,97.09,0,20)    #nothing
wu.viswlewsagn(DF,50,100,84.34,97.09,20,40)   #14,21-h/all below 0.2
wu.viswlewsagn(DF,50,100,84.34,97.09,40,60)   #9-h/all below 0.2
wu.viswlewsagn(DF,50,100,84.34,97.09,60,80)   #nothing
wu.viswlewsagn(DF,50,100,84.34,97.09,80,100)  #nothing
wu.viswlewsagn(DF,50,100,84.34,97.09,100,120) #nothing
#---
wu.viswlewsagn(DF,50,100,97.09,109.82,0,20)    #nothing
wu.viswlewsagn(DF,50,100,97.09,109.82,20,40)   #nothing
wu.viswlewsagn(DF,50,100,97.09,109.82,40,60)   #14,21-hnoisier/all<0.15
wu.viswlewsagn(DF,50,100,97.09,109.82,60,80)   #nothing
wu.viswlewsagn(DF,50,100,97.09,109.82,80,100)  #nothing
wu.viswlewsagn(DF,50,100,97.09,109.82,100,120) #nothing
#CONCLUSION (which was obvious) : treat different dates appart. Mke chopping test to really discard the chopped ones.
#NGC 7469  [42.65, 56.75333333333333, 70.85666666666667, 84.96000000000001, 99.06333333333333, 113.16666666666666, 127.27000000000001]
#not-calibrated
wu.viswlewsagn(DF,50,100,42.65,56.75,0,20)     #nothing
wu.viswlewsagn(DF,50,100,42.65,56.75,20,40)    ##very noisy, 6 a lot higher/ super noisy >1
wu.viswlewsagn(DF,50,100,42.65,56.75,40,60)    #6, a lot higher/super noisy >1
wu.viswlewsagn(DF,50,100,42.65,56.75,60,80)    #nothing
wu.viswlewsagn(DF,50,100,42.65,56.75,80,100)   #nothing
wu.viswlewsagn(DF,50,100,42.65,56.75,100,120)  #nothing
               #---
wu.viswlewsagn(DF,50,100,56.75,70.85,0,20)    #nothing
wu.viswlewsagn(DF,50,100,56.75,70.85,20,40)   #nothing
wu.viswlewsagn(DF,50,100,56.75,70.85,40,60)   #nothing
wu.viswlewsagn(DF,50,100,56.75,70.85,60,80)   #nothing
wu.viswlewsagn(DF,50,100,56.75,70.85,80,100)  #nothing
wu.viswlewsagn(DF,50,100,56.75,70.85,100,120) #2-l,4-h,6 a lot higher
               #---
wu.viswlewsagn(DF,50,100,58.92,71.65,0,20)    #nothing
wu.viswlewsagn(DF,50,100,70.85,84.96,20,40)   #nothing
wu.viswlewsagn(DF,50,100,70.85,84.96,40,60)   #nothing
wu.viswlewsagn(DF,50,100,70.85,84.96,60,80)   #nothing
wu.viswlewsagn(DF,50,100,70.85,84.96,80,100)  #nothing
wu.viswlewsagn(DF,50,100,70.85,84.96,100,120) #nothing
               #---
wu.viswlewsagn(DF,50,100,84.96,99.1,0,20)    #nothing
wu.viswlewsagn(DF,50,100,84.96,99.1,20,40)   #6 a lot higher
wu.viswlewsagn(DF,50,100,84.96,99.1,40,60)   #nothing
wu.viswlewsagn(DF,50,100,84.96,99.1,60,80)   #nothing
wu.viswlewsagn(DF,50,100,84.96,99.1,80,100)  #6 a lot higher
wu.viswlewsagn(DF,50,100,84.96,99.1,100,120) #nothing
               #---
wu.viswlewsagn(DF,50,100,99.1,113.17,0,20)    #nothing
wu.viswlewsagn(DF,50,100,99.1,113.17,20,40)   #nothing
wu.viswlewsagn(DF,50,100,99.1,113.17,40,60)   #nothing
wu.viswlewsagn(DF,50,100,99.1,113.17,60,80)   #nothing
wu.viswlewsagn(DF,50,100,99.1,113.17,80,100)  #nothing
wu.viswlewsagn(DF,50,100,99.1,113.17,100,120) #nothing
               #---
wu.viswlewsagn(DF,50,100,113.17,127.28,0,20)    #nothing
wu.viswlewsagn(DF,50,100,113.17,127.28,20,40)   #nothing
wu.viswlewsagn(DF,50,100,113.17,127.28,40,60)   #nothing
wu.viswlewsagn(DF,50,100,113.17,127.28,60,80)   #6 a lot higher
wu.viswlewsagn(DF,50,100,113.17,127.28,80,100)  #nothing
wu.viswlewsagn(DF,50,100,113.17,127.28,100,120) #nothing
#NGC 1365  DF=pd.read_pickle('25Aprews1365.pk')  #(14, 18)
#[46.567, 60.4945, 74.422, 88.3495, 102.277, 116.2045, 130.132]
#not-calibrated
wu.viswlewsagn(DF,50,100,46.5,60.49,0,20)     #Ok
wu.viswlewsagn(DF,50,100,46.5,60.49,20,40)    #Two groups!! 2 bls!!
wu.viswlewsagn(DF,50,100,46.5,60.49,40,60)    #nothing
wu.viswlewsagn(DF,50,100,46.5,60.49,60,80)    #nothing
wu.viswlewsagn(DF,50,100,46.5,60.49,80,100)   #nothing
wu.viswlewsagn(DF,50,100,46.5,60.49,100,120)  #Ok
               #---
wu.viswlewsagn(DF,50,100,56.75,70.85,0,20)    #nothing
wu.viswlewsagn(DF,50,100,56.75,70.85,20,40)   #nothing
wu.viswlewsagn(DF,50,100,56.75,70.85,40,60)   #nothing
wu.viswlewsagn(DF,50,100,56.75,70.85,60,80)   #nothing
wu.viswlewsagn(DF,50,100,56.75,70.85,80,100)  #nothing
wu.viswlewsagn(DF,50,100,56.75,70.85,100,120) #Ok lot higher
               #---
wu.viswlewsagn(DF,50,100,58.92,71.65,0,20)    #nothing
wu.viswlewsagn(DF,50,100,70.85,84.96,20,40)   #nothing
wu.viswlewsagn(DF,50,100,70.85,84.96,40,60)   #nothing
wu.viswlewsagn(DF,50,100,70.85,84.96,60,80)   #nothing
wu.viswlewsagn(DF,50,100,70.85,84.96,80,100)  #nothing
wu.viswlewsagn(DF,50,100,70.85,84.96,100,120) #nothing
               #---
wu.viswlewsagn(DF,50,100,84.96,99.1,0,20)    #nothing
wu.viswlewsagn(DF,50,100,84.96,99.1,20,40)   #nothing
wu.viswlewsagn(DF,50,100,84.96,99.1,40,60)   #nothing
wu.viswlewsagn(DF,50,100,84.96,99.1,60,80)   #2 groups
wu.viswlewsagn(DF,50,100,84.96,99.1,80,100)  #nothing
wu.viswlewsagn(DF,50,100,84.96,99.1,100,120) #nothing
               #---
wu.viswlewsagn(DF,50,100,99.1,113.17,0,20)    #nothing
wu.viswlewsagn(DF,50,100,99.1,113.17,20,40)   #Ok
wu.viswlewsagn(DF,50,100,99.1,113.17,40,60)   #nothing
wu.viswlewsagn(DF,50,100,99.1,113.17,60,80)   #nothing
wu.viswlewsagn(DF,50,100,99.1,113.17,80,100)  #nothing
wu.viswlewsagn(DF,50,100,99.1,113.17,100,120) #nothing
               #---
wu.viswlewsagn(DF,50,100,113.17,127.28,0,20)    #nothing
wu.viswlewsagn(DF,50,100,113.17,127.28,20,40)   #nothing
wu.viswlewsagn(DF,50,100,113.17,127.28,40,60)   #nothing
wu.viswlewsagn(DF,50,100,113.17,127.28,60,80)   #nothing
wu.viswlewsagn(DF,50,100,113.17,127.28,80,100)  #nothing
wu.viswlewsagn(DF,50,100,113.17,127.28,100,120) #nothing

#NGC 1566  DF=pd.read_pickle('25Aprews1566.pk')
#not-calibrated
#[42.93, 57.19583333333333, 71.46166666666667, 85.72749999999999, 99.99333333333334,  114.25916666666666, 128.525]
wu.viswlewsagn(DF,50,100,42.9,57.2,0,20)     #Nothing
wu.viswlewsagn(DF,50,100,42.9,57.2,20,40)    #Super noisy
wu.viswlewsagn(DF,50,100,42.9,57.2,40,60)    #3 super noisy
wu.viswlewsagn(DF,50,100,42.9,57.2,60,80)    #nothing
wu.viswlewsagn(DF,50,100,42.9,57.2,80,100)   #nothing
wu.viswlewsagn(DF,50,100,42.9,57.2,100,120)  #nothing
               #---
wu.viswlewsagn(DF,50,100,57.2,71.5,0,20)    #nothing
wu.viswlewsagn(DF,50,100,57.2,71.5,20,40)   #nothing
wu.viswlewsagn(DF,50,100,57.2,71.5,40,60)   #nothing
wu.viswlewsagn(DF,50,100,57.2,71.5,60,80)   #nothing
wu.viswlewsagn(DF,50,100,57.2,71.5,80,100)  #nothing
wu.viswlewsagn(DF,50,100,57.2,71.5,100,120) #Some noisier, 10 lower
               #---
wu.viswlewsagn(DF,50,100,71.5,85.7,0,20)    #nothing
wu.viswlewsagn(DF,50,100,71.5,85.7,20,40)   #nothing
wu.viswlewsagn(DF,50,100,71.5,85.7,40,60)   #nothing
wu.viswlewsagn(DF,50,100,71.5,85.7,60,80)   #nothing
wu.viswlewsagn(DF,50,100,71.5,85.7,80,100)  #nothing
wu.viswlewsagn(DF,50,100,71.5,85.7,100,120) #nothing
               #---
wu.viswlewsagn(DF,50,100,85.7,100,0,20)    #nothing
wu.viswlewsagn(DF,50,100,85.7,100,20,40)   #some noisier, some higher, 10 -l
wu.viswlewsagn(DF,50,100,85.7,100,40,60)   #nothing
wu.viswlewsagn(DF,50,100,85.7,100,60,80)   #Ok
wu.viswlewsagn(DF,50,100,85.7,100,80,100)  #very noisy
wu.viswlewsagn(DF,50,100,85.7,100,100,120) #nothing
               #---
wu.viswlewsagn(DF,50,100,100,114.3,0,20)    #nothing
wu.viswlewsagn(DF,50,100,100,114.3,20,40)   #nothing
wu.viswlewsagn(DF,50,100,100,114.3,40,60)   #nothing
wu.viswlewsagn(DF,50,100,100,114.3,60,80)   #nothing
wu.viswlewsagn(DF,50,100,100,114.3,80,100)  #nothing
wu.viswlewsagn(DF,50,100,100,114.3,100,120) #nothing
               #---
wu.viswlewsagn(DF,50,100,114.3,128.6,0,20)    #nothing
wu.viswlewsagn(DF,50,100,114.3,128.6,20,40)   #nothing
wu.viswlewsagn(DF,50,100,114.3,128.6,40,60)   #Ok, noisy
wu.viswlewsagn(DF,50,100,114.3,128.6,60,80)   #very noisy
wu.viswlewsagn(DF,50,100,114.3,128.6,80,100)  #nothing
wu.viswlewsagn(DF,50,100,114.3,128.6,100,120) #nothing
               
#[33.4,46.20,58.92,71.65,84.34,97.09,109.82]  1068
#[42.93, 57.19583, 71.46166, 85.7274999, 99.99,  114.25916, 128.525]  1566
#[46.567, 60.4945, 74.422, 88.3495, 102.277, 116.2045, 130.132]  1365
#[42.65, 56.753, 70.856, 84.960, 99.0633, 113.16, 127.2700]  7469

#USE FOR ALL: [30,45,60,75,90,105,120,135]    7 bins bl lengths

#Phot
wu.photews(DF)  #check S/N - Done.

#CONCLUSION: check BCDS maybe certain combination of telescopes over/under-perform
    
#NEXT:
#RE_DO with calibrated data
#MAKE the chopping test  and discard those mjds, because EWS current version can not handle them
#any measurement of the S/N of the phot? Measure is it looks tilted or displaced from 60
#10-20% lower with 4-6 tau  ??
#check airmass!!


#FOR cals looking at the photom
DF=pd.read_pickle('/Users/M51/allseptCalsEWS.pk') # (166, 18)
Fin2=pd.read_pickle('/Users/M51/TableDiamsSeptmsdfcc19.pk')
septcorrvis=[1.0171356326838124, 1.126052655208009, 1.1070565294110437, 0.8673201939895662, 1.0308546161084602, 0.8515803725991087]
septcorrcf=[1.0461615749200677,1.1432393108263472,1.06764454906736,0.9009873729422595,1.0163537336808484,0.8256134585631165]
DF=DF[DF['targ']!='CD-55 855'] # (159, 18) no diam.
DF=DF.reset_index()
anames=np.unique(DF['targ'])
#For every source look at the photometry
for i in range(np.size(anames)):
    df=DF[DF['targ']==anames[i]]
    df=df.reset_index()
    plt.clf()
    for j in range(df.shape[0]):
        print(df[['targ','file','tau','mjd-obs']][j:j+1])
        for k in range(0,1):  #4 fot all Ts
            #print(k)
            #plt.imshow(df['photim'][j][k])  #photobeamsDRS
            #plt.plot(df['phot'][j][k])  #last 4 TbyT, change 10 for k, as a function of wl
            plt.plot(df['photim'][j][k][60]) #k 0 to 4



#By channel look at one slice of the photometry, all cals at once
for k in range(4):
    for j in range(df.shape[0]):
        print(df[['targ','file','tau','mjd-obs']][j:j+1])
        #print(k)
        #plt.imshow(df['photim'][j][k])  #photobeamsDRS
        #plt.plot(df['phot'][j][k])  #last 4 TbyT, change 10 for k, as a function of wl
        c=df['photim'][j][k][60]  #horizontal cut
        plt.plot(c/np.max(c))

    plt.pause(20)
    plt.clf()


df1=DF[DF['targ']=='HD 20356']
df1=df1.reset_index()
temp=df1['header'][0]
temp['HIERARCH ISS AIRM END']  1.080   high would be >1.5
#'HD 183925'  wrong wl??   plt.imshow(DF['photim'][0][0])

#For all AGNs looking at the visibilities
DF=pd.read_pickle('25Aprews1068.pk')

#9th May
#Qs Js plots,  300 Jys cals?? Only O-O?? Y

DF=pd.read_pickle('/Users/M51/allseptCalsEWS.pk') # (166, 18)
... only good mjds in df
Fin2=pd.read_pickle('/Users/M51/TableDiamsSeptmsdfcc19.pk')
#Making channel correction for both vis and cf
correction=(1,1,1,1,1,1)
#Includes one weird sig Cap subobservation that I could not explain...
#For visibilities
#Done taking averages over a range pf wls first for the vis and idealvis, and then dividing these.
wu.TFvistauEWS(df,'tau','vis',0,130,1,1,1,correction,Fin2) #channelcorrection vis ews
mean= 0.5421263779191837 std= 0.08704649512297952
popt a=0.0067362711231937 b=0.5037027576019775
mean= 0.6002572996164631 std= 0.12315190965051542
popt a=0.02752134114234006 b=0.4432758564211924
mean= 0.5902349979920465 std= 0.12912442411564598
popt a=0.01957143495280236 b=0.4798675444370998
mean= 0.462563058500089 std= 0.08471382076836416
popt a=0.01587446736092044 b=0.37201526191938844
mean= 0.5498330527855652 std= 0.0981869046950826
popt a=0.014848564004880273 b=0.4651369983755729
mean= 0.45408716425548457 std= 0.0864955831409827
popt a=0.01847087451417514 b=0.3483416199681105
correction2=[1.0167723058742606,
 1.1257983811661547,
 1.107001290399351,
 0.867549203948711,
 1.031226377643633,
 0.851652440967889]
wu.TFvistauEWS(df,'tau','vis',0,130,1,1,1,correction2,Fin2)
mean= 0.5331836585114721 std= 0.0856106078224992
popt a=0.006625151700994736 b=0.4953938622608827
mean= 0.533183658511472 std= 0.10939073257766539
popt a=0.024446065182732367 b=0.39374355734902805
mean= 0.5331836585114722 std= 0.11664342691873857
popt a=0.017679684795586965 b=0.4334841835424508
mean= 0.5331836585114721 std= 0.09764728084906685
popt a=0.018298059605269586 b=0.428811717127802
mean= 0.5331836585114721 std= 0.09521372496254515
popt a=0.01439893767931872 b=0.4510522680480972
mean= 0.5331836585114721 std= 0.1015620680223518
popt a=0.02168827726182335 b=0.4090185204029058
Out[42]:
[0.9999999999999999,
 0.9999999999999997,
 1.0000000000000002,
 0.9999999999999999,
 0.9999999999999999,
 0.9999999999999999]
#Done taking average over a range of wls,  but first dividing  vis and idealvis over a range of wl one to one. The difference is very small.
#This is the last vertion of the function
wu.TFvistauEWS(df,'tau','vis',0,130,1,1,1,correction,Fin2)
mean= 0.5419676278291257 std= 0.08698837456454697
popt a=0.006764925784318717 b=0.5033805616233488
mean= 0.6000026611431147 std= 0.12326019338513174
popt a=0.02760317044923392 b=0.44255446443371027
mean= 0.5898808200579098 std= 0.12869113804809557
popt a=0.01965392428686277 b=0.47904819169962065
mean= 0.46214039996271117 std= 0.08461383414377202
popt a=0.01598229963130603 b=0.370977529457111
mean= 0.5492776115362787 std= 0.09822779397369716
popt a=0.015007281879028065 b=0.463676232024155
mean= 0.4537536387606364 std= 0.08631254517040339
popt a=0.018542746339447632 b=0.34759662910231404
correction2=[1.0171356326838124,
             1.126052655208009,
             1.1070565294110437,
             0.8673201939895662,
             1.0308546161084602,
             0.8515803725991087]
wu.TFvistauEWS(df,'tau','vis',0,130,1,1,1,correction2,Fin2)
mean= 0.5328371265482961 std= 0.08552288580729356
popt a=0.0066509593063034675 b=0.4949001238547381
mean= 0.5328371265482961 std= 0.10946219327759821
popt a=0.02451321375966549 b=0.39301401056606616
mean= 0.5328371265482958 std= 0.11624622106385074
popt a=0.017753316322957995 b=0.432722428070849
mean= 0.5328371265482961 std= 0.09755778169370044
popt a=0.01842721888942652 b=0.42772846195328984
mean= 0.532837126548296 std= 0.09528772771519727
popt a=0.01455809749319408 b=0.449797890085872
mean= 0.5328371265482961 std= 0.10135572395470888
popt a=0.021774510995235384 b=0.408178301373864
Out[74]: [1.0, 1.0, 0.9999999999999996, 1.0, 0.9999999999999997, 1.0]


#Fis correlated fluxes
#Done taking averages over a range of wls first for the cf and idealvisand flux, and then dividing these.
wu.TFcftauEWS(df,'tau','cflux',0,130,1,1,1,correction,Fin2)
mean= 1982.697644408625 std= 553.424401116737
popt a=116.16111971382313 b=1320.1158278188875
mean= 2166.9077736651893 std= 649.4083792701747
popt a=196.2421844870234 b=1047.5443984430638
mean= 2023.835377023198 std= 604.5668945007903
popt a=161.05262677228495 b=1115.6256128724112
mean= 1708.6190702675879 std= 485.3485113702807
popt a=126.5816830707474 b=986.5984668374817
mean= 1927.747793580071 std= 551.936863663088
popt a=136.87567521688732 b=1147.0103671135362
mean= 1565.4891804499377 std= 466.1277284634386
popt a=123.32898639222114 b=859.4321534443538
correction2=[1.0457912469811963,
 1.1429544938963607,
 1.0674897045399157,
 0.901226101291887,
 1.016807466634514,
 0.8257309866561264]
wu.TFcftauEWS(df,'tau','vis',0,130,1,1,1,correction2,Fin2)
mean= 1895.8828065657683 std= 529.19203781277
popt a=111.07486356932046 b=1262.3129409506164
mean= 1895.882806565768 std= 568.1839327271248
popt a=171.69728766907053 b=916.523268459428
mean= 1895.8828065657685 std= 566.3444733280651
popt a=150.87042562771347 b=1045.0926296301598
mean= 1895.8828065657683 std= 538.5424486425156
popt a=140.45496886763658 b=1094.7291260793024
mean= 1895.8828065657688 std= 542.813543147868
popt a=134.61316791327553 b=1128.0506975861094
mean= 1895.8828065657679 std= 564.5031323713135
popt a=149.35734225617531 b=1040.8137414043576
Out[67]:
[1.0000000000000002,
 1.0,
 1.0000000000000002,
 1.0000000000000002,
 1.0000000000000002,
 0.9999999999999999]
#Done taking average over a range of wls,  but first dividing  vis and idealvis over a range of wl one to one. The difference is very small.
#This is the last vertion of the function
wu.TFcftauEWS(df,'tau','cflux',0,130,1,1,1,correction,Fin2)
mean= 1982.1557953676063 std= 553.3739085666473
popt a=116.28966374288193 b=1318.8407649252422
mean= 2166.088374656327 std= 649.3796763667491
popt a=196.50633549669138 b=1045.2182886071005
mean= 2022.8594521722844 std= 604.7661430481334
popt a=161.30694416667995 b=1113.2155371551435
mean= 1707.0951425137046 std= 485.30036445873606
popt a=126.98910367592939 b=982.7506167136562
mean= 1925.6790649311722 std= 551.6379152706448
popt a=137.49098059153175 b=1141.4319438347895
mean= 1564.2846581795086 std= 466.19772510398343
popt a=123.65828251034185 b=856.3424126268349
correction2=[1.0461615749200677,
 1.1432393108263472,
 1.06764454906736,
 0.9009873729422595,
 1.0163537336808484,
 0.8256134585631165]
wu.TFcftauEWS(df,'tau','cflux',0,130,1,1,1,correction2,Fin2)
mean= 1894.6937479701003 std= 528.956445957144
popt a=111.15841802081609 b=1260.6472865332846
mean= 1894.6937479701007 std= 568.0172735639833
popt a=171.88556343767914 b=914.2602853543538
mean= 1894.6937479701007 std= 566.4489586692746
popt a=151.0867492399235 b=1042.683673361562
mean= 1894.6937479701003 std= 538.6317045420313
popt a=140.9443796208312 b=1090.7484715630833
mean= 1894.693747970101 std= 542.7617344138847
popt a=135.2786678948703 b=1123.065636195459
mean= 1894.693747970101 std= 564.6682721419609
popt a=149.7774525426126 b=1037.2195539570207
Out[78]:
[0.9999999999999999,
 1.0,
 1.0,
 0.9999999999999999,
 1.0000000000000002,
 1.0000000000000002]
#CONCLUSION
septcorrvis=[1.0171356326838124,
             1.126052655208009,
             1.1070565294110437,
             0.8673201939895662,
             1.0308546161084602,
             0.8515803725991087]

septcorrcf=[1.0461615749200677,
            1.1432393108263472,
            1.06764454906736,
            0.9009873729422595,
            1.0163537336808484,
            0.8256134585631165]

#To upload files
#scp tetGru.png gamez@oosterschelde:/disks/cool1/jaffe/drs/gamez
#scp /Volumes/LaCie/DecemberData/alpEri/2018-12-09T01\:17\:59.tpl.pk gamez@oosterschelde:/disks/cool1/jaffe/drs/gamez/alfEri  has weird permissions
scp -r *tpl* gamez@oosterschelde:/disks/cool1/jaffe/drs/decdata/alferi/dilution_test/
ssh gamez@oosterschelde
mkdir RedDataDec
cd ../..
cd /disks/cool1/jaffe/drs/decdata/alferi/dilution_test/
chmod 777 -R *.tpl*

scp 2018*tpl.pk gamez@oosterschelde:/disks/cool1/jaffe/drs/gamez/RedDataDec/
scp *vis2.fits gamez@oosterschelde:/disks/cool1/jaffe/drs/gamez/RedDataDec/
#To download : scp gamez@oosterschelde:/disks/cool1/jaffe/drs/gamez/mscd-v1.pk ./



#7,8 My local hostname: starlight.local, network address:smb://192.168.0.102
ssh ssh.strw.leidenuniv.nl -p 80 -l gamez
'''scp starlight.local:/Volumes/LaCie/SeptemberData/RedLast .
scp starlight.local@smb://192.168.0.102:/Volumes/LaCie/SeptemberData/RedLast .
scp starlight.local@192.168.0.102:/Volumes/LaCie/SeptemberData/RedLast .
scp 2018-09-20T00/:57/:18.tpl.pk gamez@oosterschelde:/disks/cool1/jaffe/drs/gamez/RedDataSept/
scp 2018-09-20T00/:57/:18.tpl.pk gamez@ssh.strw.leidenuniv.nl:/disks/cool1/jaffe/drs/gamez/RedDataSept/
 scp /Volumes/LaCie/SeptemberData/RedLast/2018-09-20T00/:57/:18.tpl.pk gamez@ssh.strw.leidenuniv.nl:/disks/cool1/jaffe/drs/gamez/RedDataSept/   ALL THIS DIDNT WORK  '''
scp -r /Users/M51/SeptAGNs.pdf gamez@ssh.strw.leidenuniv.nl:/disks/cool1/jaffe/drs/gamez/redDataSept
scp -r /Volumes/LaCie/SeptemberData/RedLast gamez@ssh.strw.leidenuniv.nl:/disks/cool1/jaffe/drs/gamez/redDataSept    #Yessss!!!! :)

ssh ssh.strw.leidenuniv.nl -p 80 -l gamez
chmod 777 -R *.tpl*
#Plots for ews: last version has correction for vis, and corrected bls, version after April 15th.
DF=pd.read_pickle('/Users/M51/allseptCalsEWS.pk') # (166, 18)
df=DF[DF['targ']!='CD-55 855'] # (159, 18) no diam.
df=df.reset_index()
#(PENDING  function working fine
 wu.TFvistauEWS(DF,'tau','vis',0,130,1,1,1,correction,Fin2) ##FIRST-2 vis EWS   to correct channels  with limiting tau0
def TFcorflux2tauDRS(DF,x,y,bl1,bl2,a,p,fit,correction) not ready
 #)


df=df[df['targ']!='HD 212849']   #really high cfs when using wrong wl: 20:46, right:50:100
df=df.reset_index()  #(152, 20)
Fin2=pd.read_pickle('/Users/M51/TableDiamsSeptmsdfcc19.pk')
#NOT CORRECTED BY CHANNEL YET
#VIS      FIRST to SIXTH -Cals
#There are a few above 2.0, program uses xlim
#SELECTION CALIBRATORS
DF=pd.read_pickle('/Users/M51/allseptCalsEWS.pk') # (166, 18)
df=DF[DF['targ']!='CD-55 855']  #159,18
df=df[df['targ']!='b Sgr']  #150
#df=df[df['targ']!='HD 20356']  #137 Needed for 1068 Sept 25
amjd=[58381.04560605,58381.05829687,58381.11948348,58381.15815322, 58382.12718112,58382.13937325,58382.14056971,58382.14151070, 58382.14244882,58382.14338623,58382.14647846, 58383.13341208,58383.13461323,58383.13554598,58383.13647738, 58383.13741494,58383.13836445,58383.13942234,58383.14308607, 58383.14435676,58383.14797656, 58383.15507120, 58383.20571736, 58384.07957074, 58384.17850135, 58384.98769468, 58385.00456004, 58385.38429503, 58385.38560493, 58386.03318726, 58386.07974087, 58386.08855096, 58386.10701578, 58386.11361284, 58386.11911919, 58386.18925231,58386.19161547, 58386.19635368, 58386.20523466, 58386.40622892, 58386.4108629,  58385.35656910, 58385.36001311, 58385.36108241, 58385.36422807, 58385.36542380]   #41 Lst 6 HD20356
for i in range(np.size(amjd)):
    df=df[df['mjd-obs']!=amjd[i]]
    print(amjd[i])
    print(df.shape)  #96,18   #104,18

df=df.reset_index()
for i in range(df.shape[0]): print(df['wave'][i][50])  #4.128
for i in range(df.shape[0]): print(df['wave'][i][100]) #2.978
wu.TFvisEWSflux(df,0,130,'diam',tableDiams=Fin2)   #1-Cals
wu.TFvisEWSflux(df,0,130,'tau',tableDiams=Fin2)    #2
wu.TFvisEWStau(df,0,130,'diam',tableDiams=Fin2)    #3-Cals  small tend
wu.TFvisEWStau(df,0,130,'tau',tableDiams=Fin2)     #4
wu.TFvisEWSmjd(df,0,130,'diam',tableDiams=Fin2)    #5-Cals
wu.TFvisEWSmjd(df,0,130,'tau',tableDiams=Fin2)     #6
wu.TFvisEWSbl(df,0,130,'diam',tableDiams=Fin2)     #7-Cals
wu.TFvisEWSbl(df,0,130,'tau',tableDiams=Fin2)      #8
#CF     SEVENTH to 14TH - Cals
wu.TFcfEWSflux(df,0,130,'diam',tableDiams=Fin2)   #9-Cals
wu.TFcfEWSflux(df,0,130,'tau',tableDiams=Fin2)    #10
wu.TFcfEWStau(df,0,130,'diam',tableDiams=Fin2)    #11-Cals <- dep with t0
wu.TFcfEWStau(df,0,130,'tau',tableDiams=Fin2)     #12
wu.TFcfEWSmjd(df,0,130,'diam',tableDiams=Fin2)    #13
wu.TFcfEWSmjd(df,0,130,'tau',tableDiams=Fin2)     #14
wu.TFcfEWSbl(df,0,130,'diam',tableDiams=Fin2)     #15
wu.TFcfEWSbl(df,0,130,'tau',tableDiams=Fin2)      #16

cat[['Name','SpType','diam_midi','diam_cohen','diam_gaia','UDDL_est','Kmag','CalFlag','IRflag','med_Lflux']]
df1=df[df['targ']=='HD 195677']       #'HD209240']
df1=df1.reset_index()
wu.TFcfEWSflux(df1,0,130,'diam',tableDiams=Fin2)

wu.TFcfEWSfluxlim(df,0,130,'diam',7.,tableDiams=Fin2)
df1=df[df['targ']=='HD 212849']
df1=df1.reset_index()
wu.TFcfEWSflux(df1,0,130,'diam',tableDiams=Fin2)

#TO SAVE ALL PLOTS MANUALLY
DF=pd.read_pickle('/Users/M51/allseptCalsEWS.pk') # (166, 18)
DF=DF[DF['targ']!='CD-55 855'] # (159, 18) no diam.
DF=DF.reset_index()
anames=np.unique(DF['targ'])
for i in range(np.size(anames)):
    df=DF[DF['targ']==anames[i]]
    df=df.reset_index()
    #L band ONLY
    #3 to 4.1um  (92.9782114669574997 to 4.1282540599575)=50:100 SeptCals
    wu.viswlews(df,'wave',50,100,0,130,tableDiams=Fin2) #ylim=2.
    #test: plt.plot(DF[keyx][i],np.abs(DF['vis'][i][j]))

DF=pd.read_pickle('/Users/M51/allseptCalsEWS.pk') # (166, 18)
DF=DF[DF['targ']=='CD-55 855']
DF=DF.reset_index()
import matplotlib.colors as colors
import matplotlib.cm as cmx
wl1=50
wl2=100
jet = cm = plt.get_cmap('prism')      #('RdBu')
cNorm  = colors.Normalize(vmin=0, vmax=np.max(DF['tau']))
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
cNorm  = colors.Normalize(vmin=1.671, vmax=11.086)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
for i in range(DF.shape[0]):
    for j in range(6):
        plt.text(DF[keyx][i][wl1+(i*2)], np.abs(DF['vis'][i][j][wl1+(i*2)]),'.'+str(i),fontsize=8 )
        colorVal = scalarMap.to_rgba(DF['tau'][i])
        if (j==5):
            plt.plot(DF[keyx][i][wl1:wl2],np.abs(DF['vis'][i][j][wl1:wl2]), linestyle='-', color=colorVal, label=str(i)+':mjd='+str(DF['mjd-obs'][i])+',t0='+"{0:.1f}".format(DF['tau'][i]))
            print(i,'    ', j)
        else:
            plt.plot(DF[keyx][i][wl1:wl2],np.abs(DF['vis'][i][j][wl1:wl2]), linestyle='-', color=colorVal)

plt.ylim(0.,2.)
plt.legend(prop={'size': 7})
title= DF['targ'][0]+' EWS TF vis vs wl'  #Lflux=cat2['med_Lflux'][index2]
plt.title(title)
plt.xlabel('wl [um]')
plt.ylabel('Corrected vis []')
plt.grid(linestyle="--",linewidth=0.5,color='.25')

DF=pd.read_pickle('/Users/M51/allseptCalsEWS.pk') # (166, 18)
DF=DF[DF['targ']=='del Phe'] #20.18
DF=DF.reset_index()
wu.viswlews(DF,'wave',50,100,0,130,tableDiams=Fin2)


#CONCLUSIONS:  nicer in email
#For vis:
#Leave out chopped ones, most of the times they have larger spread and noise
#Leave delPhe out or choose carefully mjds to include DONE 2nd, looks good
#vrange3.2 and 3.8 (no chann corr) for accepted cals can give us another parameter for unresolved cals to be acceptable in vis
#I think that we can use the original calibrators avoding red mjds except for cd-55 855
#We need a EWS that uses the chopped frames right, to get to a conclusion like: The chopping seems to ruin/improve(HD 6290 shows smaller spread but larger noise) the visibilities, because even when the obs looks good, the values are altered.
#Plot1: check sigCap has high values, while 47 Cap has lower values but it has diam ~3.65mas
#Plot2: seems to be stable with tau0
#Plot3&4: There is a slight tendency of the vis to increment with tau even a little below 3
#5, the 2 largest stars, with 3~ and ~3.5 seem to have lower values
#6 maybe there is a smaller spread with larger tau
#7  Spread seems to decrease with larger bls
#9 cf not so stable
#11&12 seems to be a dep with tau
#16 THere is a clear dependance with tau0

#NEXT: RE-Check cfs making the channel correction ...



#baselines are now correct in pbl
#first quality check, in the OPD tracking images
df=pd.read_pickle('allseptCalsEWS.pk')
for i in range(df.shape[0]):
    print(dfews[['targ','file','tau','mjd-obs']][i:i+1])
    cf=dfews['cflux'][i]
    plt.imshow(cf['opdimage'])  #cf['opdimage'].shape (271, 768)
    plt.pause(15)
#Use ctrl c to stop
from astropy.io import fits
import os
os.chdir('/Volumes/LaCie/SeptemberData')
os.chdir('/Volumes/LaCie//SeptAGN/NGC1365')
for i in range(dfews.shape[0]):
    #print(df[['targ','file','tau','mjd-obs']][i:i+1])
    fl=dfews['file'][i]
    hducal=fits.open(fl)
    tpls=hducal[0].header['HIERARCH ESO TPL START']
    print(tpls)
    hducal.close()

df=pd.read_pickle('/Users/M51/25Aprews1068.pk')
df=pd.read_pickle('/Users/M51/25Aprews1365.pk')
for i in range(0,22):#(df.shape[0]):
    print(df[['targ','file','tau','mjd-obs']][i:i+1])
    cf=df['cflux'][i]
    plt.imshow(cf['opdimage'])  #cf['opdimage'].shape (271, 768)
    plt.pause(5)
#5 May
df1=DF.loc[DF["targ"]=='CD-55 855']
df1=df1.reset_index()
df1.shape  #(7, 19)
cf=df1['cflux'][0]
cf.keys()   #dict_keys(['localopd', 'mjd', 'wave', 'fdata', 'opdf', 'delay', 'opdfm', 'delaym', 'opd0', 'opds'(6, 271, 128), 'opd'(6, 271), 'opdp'(6, 271), 'opdm'(6, 271), 'opdpm'(6, 271), 'cdata'(6, 271, 110), 'phase'(6, 271), 'mphase'(6, 271), 'pdata'(6, 271, 110), 'flux'(6, 110), 'opdimage'])

plt.imshow(cf['opdimage'])  # plt.imshow(cf['opdf'][0])    plt.imshow(cf['opdfm'][0])  plt.imshow(cf['opd0'][0])
#To recover the tracking:
plt.plot(cf['opdp'][0])  # for each bl #strength of the tracking

plt.imshow(cf['fdata'][0].real)

for j in range(cf['flux'][0].shape[0]):  #??????
    plt.plot(,np.abs(cf['flux'][0][j]),'o')   #110  x=????
#28 April
#Plot uv plane:
l = .0000038
print (l,"lambda in microns")
c = 1.0/l
cmap = plt.get_cmap('jet_r')
#
delta0 =  math.radians(hdr["DEC"]) #dec of the source in deg
print (hdr["DEC"], delta0, "dec of the source in deg and rad")
a=(D)*cos(deltaB)  #major axis: almost 0
print (a, "major axis")
b=(D)*cos(deltaB)*sin(delta0) #minor axis: almost 0
print (b, "minor axis")
v0=(D)*sin(deltaB)*cos(delta0) # center over v axis (u center=0)
print (v0, "center on v axis")
deltaB = math.rad ians(-90) #dec of the baseline E-W=-90, E-W baselines have no v
theta=l/D #rad
u=1/thetax
v=1/thetay

t = np.linspace(-0.9*pi, .5*pi, 400)
plt.plot(c*cos(deltaB)*sin(t),c*(sin(deltaB)*cos(delta0)-cos(deltaB)*sin(delta0)*cos(t)),color='red')
plt.show()


for i in range(7):
    for j in range (6):
        c = cmap(float(j)/6)
        plt.plot(df['UCOORD'][i][j],df['UCOORD'][i][j],'bo', color=c,markersize=3)
        plt.plot(-1.0*df['UCOORD'][i][j],-1.0*df['UCOORD'][i][j],'bo',color=c,markersize=3)  # u,v for which lambda???



#  25 April

#Rehaciendp tabla hdrs  mc.hdrsFromFitsFiles("MA*.fits",fileName="headers1566.pk"), seleccion y los folders con templates y re-reduciendo
#Re haciendo tabla:
os.getcwd()
df2=wu.tableEWS('/Volumes/LaCie/SeptAGN/NGC1566')
os.chdir('/Users/M51')
df2.to_pickle('25Aprews1566.pk')
#19Abril  df1365=wu.tableEWS(inDIR)  #ews1365.pk
print(df[['targ','file','tau','mjd-obs']][0:30])
cat.loc[cat['name']=='HD 183925']
print(df[['file']][0:30])
    cat[['Name','UDDL_est']]
    Name    UDDL_est
0   * del Phe                          2.27699995
1   * b Sgr                            2.92400002
2   * sig Cap                          2.02800012
3   *  47 Cap                          3.83599997
4   HD 209240                          0.70899999
5   HD  20356                          1.88399994
6   HD  33162                          2.98399997
7   HD 183925                          1.45000005
8   V* V4026 Sgr                       3.23699999
9   HD   6290                          1.89800000
10  *  84 Aqr                          0.68400002
11  HD  29246                          0.75000000
12  HD 172453                          0.67299998
13  HD 195677                          1.01900005
14  HD  16658                          0.28999999
15  HD  25286                          0.40099999
16  HD  14823                          0.77600002
17  HD 212849                          0.92900002
df=pd.read_pickle('allseptCalsEWS.pk')  #Only 6
Fin2=pd.read_pickle('/Users/M51/TableDiamsSeptmsdfcc19.pk')
df1=df.loc[df['targ']=='CD-55 855']
df1=df1.reset_index()
wu.viswl(df1)  #raw vis
wu.TFvisEWSflux(df1,0,130,tableDiams=Fin2)    #corrected vis  has no diameter!!

df1=df.loc[df['targ']=='HD 20356']
df1=df1.reset_index()
df1.shape
cf=df1['cflux']
cf.keys()   #dict_keys(['localopd', 'mjd', 'wave', 'fdata', 'opdf', 'delay', 'opdfm', 'delaym', 'opd0', 'opds'(6, 271, 128), 'opd'(6, 271), 'opdp'(6, 271), 'opdm'(6, 271), 'opdpm'(6, 271), 'cdata'(6, 271, 110), 'phase'(6, 271), 'mphase'(6, 271), 'pdata'(6, 271, 110), 'flux'(6, 110), 'opdimage'])

plt.imshow(cf['opdimage'])  # plt.imshow(cf['opdf'][0])    plt.imshow(cf['opdfm'][0])  plt.imshow(cf['opd0'][0])
#To recover the tracking:
plt.plot(cf['opd'][0])  #for each bl

plt.plot(cf['opdp'][0])  # for each bl #strength of the tracking

plt.imshow(cf['fdata'][0].real)

for j in range(cf['flux'][0].shape[0]):
    plt.plot(np.abs(cf['flux'][0][j]))   #110

#18Abril
#FIRST-Cals  plotting all EWS reduced Sept DATA
DF=pd.read_pickle('/Users/M51/allseptCalsEWS.pk') # (166, 18)
df=DF[DF['targ']!='CD-55 855'] # (159, 18)
df=df.reset_index()
Fin2=pd.read_pickle('/Users/M51/TableDiamsSeptmsdfcc19.pk')
2 is wth tau   8,9,10,11
Plotted 1-7 for diam same as below
wu.TFvisEWSmjd(df,0,130,'diam',tableDiams=Fin2)  or 'tau'

#17 Abril
Not corrected by channel
#In all functions I had to use 'pbl', since I don't have 'BL'. The stars are small so it should not make a big difference.
df= septCalsEWS.pk   Fin2=  TableDiamsSeptmsdfcc19.pk'
wu.TFvisEWSflux(df,0,130,tableDiams=Fin2) septCalsvisfluxdiam.png
wu.TFvisEWStau(df,0,130,tableDiams=Fin2) SeptCalsvistaudiam
wu.TFvisEWSmjd(df,0,130,tableDiams=Fin2)   SeptCalsvismjdtau / diam
wu.TFvisEWSbl(df,0,130,tableDiams=Fin2)  SeptCalsvisbltau / diam  #bls not in the right order, but since stars are small it looks fine
wu.TFcfEWSflux(df,0,130,tableDiams=Fin2)  SeptCalscfluxLfluxdiam /tau
wu.TFcfEWSmjd(df,0,130,tableDiams=Fin2)  SeptCalscfluxmjdtau  / diam
wu.TFcfEWSbl(df,0,130,tableDiams=Fin2)   SeptCalscfluxbl /diam GOOD



In [303]: Fin2['UDDL_est']
Out[303]:
0    1.45000005
1    1.89800000
2    0.68400002
3    0.75000000
4    0.77600002
5    0.92900002
Name: UDDL_est, dtype: float64

In [304]: Fin2['med_Lflux']
Out[304]:
0    30.36823273
1    16.33547211
2     5.75891304
3     5.87307978
4     4.53761578
5     6.05695581
In [306]: Fin2['Name']
Out[306]:
0    HD 183925
1    HD   6290
2    *  84 Aqr
3    HD  29246
4    HD  14823
5    HD 212849
#92-95  'V482 Car' #Misslabeled in raw data!!  V* AF Col
#April 12th
df=pd.read_pickle('GoodDecCalsAll.pk')  #
df.shape  #193,81     96 II + 96 OO
wu.TFvisEWSflux(df,0,130)
wu.TFvisEWStau(df,0,130)
wu.TFvisEWSmjd(df,0,130) #tau or UDDL
wu.TFvisEWSbl(df,0,130)  #tau or UDDL
#tau or UDDL
wu.TFcfEWSmjd(df,0,130)  #tau or UDDL
wu.TFcfEWSbl(df,0,130)  #tau or UDDL
#++++++SeptAGN+++++++++
# April 11th
from mcdb import wutil as wu
t=wu.mrestore('/Volumes/LaCie/SeptAGN/NGC1365/2018-09-24T06:48:55.tpl.pk')
len(t) # 14
t0=t[0]
t0.keys()  #Out[194]: dict_keys(['phot', 'photim', 'wave'110, 'cflux', 'flux', 'vis'(6, 110), 'mjd', 'opd', 'pbl', 'header', 'bcd', 'tau', 'ra', 'dec', 'mjd-obs', 'targ', 'file', 'sky'])
#**   REDUCE DATA for calibrator
#**   Make visibility plot for each baseline with grid
#**   Also for calibrator
for n in range(len(t)):
    for i in range(1):
        plt.plot(t[n]['wave'],t[n]['vis'][i].real)  #NO, take np.abs

plt.ylim(-0.1,1.0)
cf=t0['cflux']
cf.keys()   #dict_keys(['localopd', 'mjd', 'wave', 'fdata', 'opdf', 'delay', 'opdfm', 'delaym', 'opd0', 'opds'(6, 271, 128), 'opd'(6, 271), 'opdp'(6, 271), 'opdm'(6, 271), 'opdpm'(6, 271), 'cdata'(6, 271, 110), 'phase'(6, 271), 'mphase'(6, 271), 'pdata'(6, 271, 110), 'flux'(6, 110), 'opdimage'])

plt.imshow(cf['opdimage'])  # plt.imshow(cf['opdf'][0])    plt.imshow(cf['opdfm'][0])  plt.imshow(cf['opd0'][0])
#To recover the tracking:
plt.plot(cf['opd'][0])  #for each bl
plt.xlim(10,300)

plt.plot(cf['opdp'][0])  # 271 #strength of the tracking

plt.imshow(cf['fdata'][0].real)

#++++++++++++++++++++++++++++^^^^^^^^^^^^^++++++++++++++++++++


2.- correction for the channels: avg,all visibilities in same  run, by channel, v6=[a,b,c,d,e,f] , vc6=v6*6/(a+b+c+d+e+f).
For every observation,divide raw visibilies
for the six channels by these six vc6 numbers.
3.-

#-----Concatenating both data frames-----
DF1=pd.read_pickle('decCalsDRSvis.pk')  #382
DF2=pd.read_pickle('decCalsEWS.pk')  #225
DF=pd.concat([DF1.set_index('mjdo'),DF2.set_index('mjd-obs')], axis=1, join='inner').reset_index()
DF.shape #(225, 80)  keeps only ii and oo, cause only those match with EWS
DF.to_pickle('superp.pk') #in M51!!
#Leave out tpls
df=wu.omitTemplate(DF,True,False)  #(193, 81)
dfoo=df.loc[df['BCD1']=='OUT']
dfoo=dfoo.loc[dfoo['BCD2']=='OUT']
dfoo=dfoo.reset_index(drop=True)
dfoo.shape #(97, 81)
dfoo.to_pickle('superoo.pk')
dfii=df.loc[df['BCD1']=='IN']
dfii=dfii.loc[dfii['BCD2']=='IN']
dfii=dfii.reset_index(drop=True)
dfii.shape #(96, 81)
dfii.to_pickle('superii.pk')
#FIRST
#correction=[1,1,1,1,1,1]
wu.TFvis2tauDRS(dfoo,'T0E','VIS2',0,130,1,2,1,correction)  #Channelsooselected.png
For to>3  plt.plot(X,a*X+b)
mean= 0.7195404749345954 std= 0.10591060541932423
popt [0.01711671 0.61982412]
mean= 0.6654039944251154 std= 0.07631253507036513
popt [0.0116379  0.59760541]
mean= 0.5203049525757195 std= 0.06990079851456633
popt [0.01045762 0.45938231]
mean= 0.49840430646463607 std= 0.09105426439460582
popt [0.00359148 0.47676621]
mean= 0.4735591991994335 std= 0.07647238320006569
popt [0.01283483 0.39878772]
mean= 0.4149350521369565 std= 0.05224796064886438
popt [0.00571524 0.38168876]
for t0>4
mean= 0.7286863873530545 std= 0.09873548879211222
popt a=0.017375475101478424 b=0.6183100536398057
mean= 0.6722323236985942 std= 0.07777851472296506
popt a=0.011083691509760829 b=0.6018240484111158
mean= 0.529440305075255 std= 0.06639692294116267
popt a=0.006486714108853205 b=0.48823396575003475
mean= 0.4972270796242682 std= 0.09643210720327344
popt a=0.006941245004421347 b=0.45142874508509623
mean= 0.4817884953043245 std= 0.07180258699405413
popt a=0.01144668208076766 b=0.4090743510432683
mean= 0.41936291939374715 std= 0.04747100066418501
popt a=0.004656228198292114 b=0.38980178016269007
channelsooselectedcorrected.png
correction=[1.313446407352279,
 1.21168879478491,
 0.9543082987107666,
 0.8962444375324078,
 0.8684166182378906,
 0.7558954433817469]

#correction=[1,1,1,1,1,1]
wu.TFvis2tauDRS(dfii,'T0E','VIS2',0,130,1,2,1,correction)
for t0>3
mean= 0.734972203347128 std= 0.10100591578762337
popt [0.02436001 0.593419  ]
mean= 0.6838129949208789 std= 0.07239763851354739
popt [0.02014729 0.56673945]
mean= 0.553557472178488 std= 0.06934589518547495
popt [0.01278854 0.47925852]
mean= 0.5327953726414557 std= 0.09243359941500187
popt [0.02264037 0.4012348 ]
mean= 0.4630731912627401 std= 0.06985977293503247
popt [0.00710488 0.42041968]
mean= 0.40036661095146786 std= 0.05032829830576718
popt [0.01223118 0.32929264]

for t0>4
mean= 0.7501918984436711 std= 0.08260946055672692
popt a=0.021548537462845862 b=0.6151654189629071
mean= 0.6977408277574401 std= 0.05781532787048559
popt a=0.01649937027087155 b=0.594353213394124
mean= 0.5623807631810621 std= 0.05822639294317941
popt a=0.011547477950978538 b=0.4893160211572334
mean= 0.5467111576353789 std= 0.0811649200916686
popt a=0.020013530479252847 b=0.4213032700689154
mean= 0.4696041347092097 std= 0.06881322522293014
popt a=0.005287109370842267 b=0.4350157450583895
mean= 0.4107866768463049 std= 0.04017543501955046
popt a=0.00807511496088609 b=0.36018675346186113
correction=[1.3094580637426168,
 1.21790485235163,
 0.9816342015541816,
 0.954282944655742,
 0.819692830911078,
 0.7170271067847526]
mean= 0.5729025764288443 std= 0.06308675538689447
popt a=0.016456074023020614 b=0.4697862631370562
mean= 0.5729025764288445 std= 0.04747113681241275
popt a=0.013547339408224094 b=0.4880128455235557
mean= 0.5729025764288442 std= 0.05931577450234713
popt a=0.011763524522867625 b=0.49847083559781974
mean= 0.5729025764288441 std= 0.08505330682709507
popt a=0.020972323754224285 b=0.4414867414232203
mean= 0.5729025764288443 std= 0.08395001472252127
popt a=0.006450107958226775 b=0.5307058235727619
mean= 0.5729025764288443 std= 0.0560305665425992
popt a=0.011261938366376229 b=0.5023335230829703

#for correlated fluxes    superoo.pk/ii has vis in icflux
#HACER TABLE con nueva pipeline con ICFLUX
correction=[1,1,1,1,1,1]
wu.TFcorflux2tauDRS(df,'T0E','ICFLUX2',0,130,1,2,1,correction)


#CONCLUSION: There is a clear difference in the vis2 for the different channels (Considering good weather observations i.e. t0>4), which for direct comparisson can be omited by multiplying by a correction factor. The medians and spreads of the different channels for the different BCDs II and OO are very close so there is no need to treat them separately.

#FIRST  for cflux EWS
#correction=[1,1,1,1,1,1]

Check range of wls: [20:45]
y understanding of what walter said yesterday is that he finds the mean of the vis2 for each channel for each good night, and then divides the channels by these values. Then to rescale away from 1 to ~0.7 he multiplies by the mean of the ratios between each channel with every other channel.

dfoo=pd.read_pickle('superoo.pk')
correction=[1,1,1,1,1,1]
wu.TFcftauEWS(dfoo,'T0E','cflux',0,130,1,1,1,correction)
mean= 106.90222923698711 std= 28.103051475301992
popt a=-0.09329555675514922 b=107.49807225186589
mean= 119.17230324947262 std= 29.358350961557303
popt a=-0.16502982200141192 b=120.22628563205708
mean= 109.61841988192144 std= 27.860381460547845
popt a=-0.44719240747088085 b=112.47446636986187
mean= 102.50531538614784 std= 21.390898289511732
popt a=0.8260812401824487 b=97.06255981357506
mean= 101.55485414147397 std= 27.66336236697876
popt a=-0.2405087552882017 b=103.09089141036605
mean= 87.12373551898933 std= 22.914745732000295
popt a=-0.569734861987317 b=90.76219705922016
Out[100]:
[1.0231887935165984,
 1.14062883489649,
 1.049186154364803,
 0.9811047976041906,
 0.9720076880194479,
 0.8338837315984703]
corrcfoo=[1.0231887935165984,
      1.14062883489649,
      1.049186154364803,
      0.9811047976041906,
      0.9720076880194479,
      0.8338837315984703]

wu.TFcftauEWS(df,'T0E','cflux',0,130,1,1,1,corr)
mean= 104.47947623583204 std= 27.46614471676785
popt a=-0.09118113543941453 b=105.06181525109594
mean= 104.47947623583205 std= 25.73874170401936
popt a=-0.14468393907194965 b=105.40351711369767
mean= 104.47947623583204 std= 26.554278613612706
popt a=-0.42622726088797025 b=107.20162641444239
mean= 104.47947623583207 std= 21.802867891123604
popt a=0.8419905919889819 b=98.93189959742789
mean= 104.47947623583204 std= 28.4600242446079
popt a=-0.24743485759288686 b=106.05974786538857
mean= 104.47947623583207 std= 27.479545245564463
popt a=-0.6832304290286424 b=108.84274729683553
Out[102]:
[0.9999999999999998,
 1.0,
 0.9999999999999998,
 1.0000000000000002,
 0.9999999999999998,
 1.0000000000000002]





dfii=pd.read_pickle('superii.pk')
wu.TFcftauEWS(dfii,'T0E','cflux',0,130,1,1,1,correction)
mean= 110.25487102311843 std= 28.35209772477549
popt a=-2.516570937264457 b=126.31603929735282
mean= 124.97041435734795 std= 30.205141708721612
popt a=-2.7005100973195995 b=142.20551250062726
mean= 108.83463642925662 std= 29.09683660147649
popt a=-2.9585010348548817 b=127.93170582218866
mean= 95.63613347528066 std= 25.28344638830414
popt a=-2.0188579402285716 b=108.52081581013746
mean= 117.61094659043087 std= 23.730672354894093
popt a=-0.3750923534811801 b=120.07296116668108
mean= 93.68996671163839 std= 23.178168226272433
popt a=-2.4052795586898337 b=109.04085536517567
Out[104]:
corrcfii=[1.0161786583653332,
 1.15180641742696,
 1.0030888776530422,
 0.8814431226878043,
 1.0839769055671113,
 0.8635060182997496]

mean= 108.49949476451214 std= 27.900701802165226
popt a=-2.4765047339435875 b=124.30495408549989
mean= 108.49949476451216 std= 26.224147783615756
popt a=-2.344586935808719 b=123.46303294735748
mean= 108.49949476451212 std= 29.007236795960946
popt a=-2.9493902616327623 b=127.5377542850134
mean= 108.49949476451214 std= 28.684149592326225
popt a=-2.290400505831054 b=123.11720626821865
mean= 108.49949476451216 std= 21.892230575224996
popt a=-0.3460324580310379 b=110.77076731785935
mean= 108.49949476451214 std= 26.841930148802476
popt a=-2.7854810135155574 b=126.27689122900016
Out[106]: [1.0, 1.0000000000000002, 0.9999999999999998, 1.0, 1.0000000000000002, 1.0]

dfoo=pd.read_pickle('superoo.pk')
wu.TFvistauEWS(dfoo,'T0E','vis',0,130,1,1,1,correction)
mean= 0.9298814983943139 std= 0.21958093217228375
popt a=0.0037954545239172314 b=0.9056413852742881
mean= 1.135456330653814 std= 0.2526000806581711
popt a=0.00534795578887326 b=1.101300996401917
mean= 1.0477017359244185 std= 0.24312023326110938
popt a=0.006191365644122704 b=1.0081598642515255
mean= 0.9197198414401095 std= 0.15210580166662238
popt a=0.018220306792040182 b=0.7996727183855504
mean= 0.9244635059276091 std= 0.21658730053067357
popt a=0.007318628704872331 b=0.8777222298580384
mean= 0.8155467658222287 std= 0.20047198916027137
popt a=0.005128185650478512 b=0.7827969601416646
Out[158]:corrvisoo=[0.9664839065849935,
 1.1801506666192538,
 1.0889418365895118,
 0.9559222619803481,
 0.960852648694487,
 0.8476486795314051]
mean= 0.9621282796937491 std= 0.227195642551523
popt a=0.003927069987353784 b=0.9370475890569024
mean= 0.9621282796937494 std= 0.214040535503647
popt a=0.004531581554061484 b=0.9331868054142209
mean= 0.9621282796937489 std= 0.22326282735407119
popt a=0.005685669762979105 b=0.9258160864243442
mean= 0.9621282796937491 std= 0.15911942604151777
popt a=0.019060448723023506 b=0.8365457601169324
mean= 0.962128279693749 std= 0.22541156630514714
popt a=0.007616808755615768 b=0.9134826419635896
mean= 0.962128279693749 std= 0.2365036293940737
popt a=0.006049890725577842 b=0.9234922475727373
Out[160]:
[1.0000000000000002,
 1.0000000000000002,
 0.9999999999999999,
 1.0000000000000002,
 1.0,
 1.0]
dfii=pd.read_pickle('superii.pk')
wu.TFvistauEWS(dfii,'T0E','vis',0,130,1,1,1,correction)
mean= 0.9209975101859754 std= 0.2160498168262353
popt a=-0.015487535570089011 b=1.0198415011345412
mean= 1.1002446812126565 std= 0.24508067196566935
popt a=-0.011137264287966858 b=1.1713245276794917
mean= 0.9632804106882383 std= 0.23760161446518302
popt a=-0.019652618882697738 b=1.0901377016386657
mean= 0.8748295753507856 std= 0.20352386675190398
popt a=-0.010009768855664307 b=0.9387135610278153
mean= 1.0665602147692537 std= 0.17919688474224368
popt a=0.00830919771054495 b=1.012020672167075
mean= 0.8516596012359324 std= 0.19628263406275487
popt a=-0.013607018945485862 b=0.9385018268048667
Out[138]: corrvisii=[0.9564545569293599,
 1.142602479859734,
 1.0003652867829225,
 0.908509224646954,
 1.1076212111036208,
 0.8844472406774082]
mean= 0.9629286655738072 std= 0.2258861283696011
popt a=-0.016192653205490813 b=1.0662728328645623
mean= 0.9629286655738073 std= 0.21449338355694403
popt a=-0.009747276912768459 b=1.0251373847453435
mean= 0.9629286655738073 std= 0.2375148534284778
popt a=-0.019645440758061877 b=1.0897396218626383
mean= 0.9629286655738073 std= 0.22401959301072938
popt a=-0.011017796713333671 b=1.0332460503944851
mean= 0.9629286655738069 std= 0.16178534949118029
popt a=0.007501852618816326 b=0.913688338770092
mean= 0.9629286655738067 std= 0.22192689968982182
popt a=-0.015384770894823463 b=1.0611167955181595
Out[145]:
[1.0,
 1.0000000000000002,
 1.0000000000000002,
 1.0000000000000002,
 0.9999999999999997,
 0.9999999999999997]




index=[]
for i in range(len(tpls)):
    for j in range(len(df1['header'])):
        if (df1['header'][j]['HIERARCH ESO TPL START'] == tpls[i]):
            index.append(j)
            break

Index=[185,
       187,
       189,
       191,
       193,
       195,
       197,
       199,
       201,
       203,
       205,
       207,
       209,
       211,
       213,
       215,
       217,
       219,
       221,
       223]

dDt=Ddf.loc[Ddf['header']['HIERARCH ESO TPL START'] == '2018-11-28T01:28:25']

wu.plotDRS(DF1,'T0E','VIS2',0,130,0,0)

wu.TFcfbl(df,0,130,30,1)  #(1 - corrected)
wu.TFcfflux(df,0,130,30,1)
wu.TFcfmjd(df,0,130,30,1)
wu.TFcftau(df,0,130,30,1)
Interm prod vs final prod

incoh cflux drs (which is sq)vs cohn cflux ews **2  GOOD
photom same also GOOD
vis same and corr. by diam : BAD!!!

NEXT TF as fn of weather, bls or vis.also
Leo: is it stable with time???
Leo: COH flx vs time
ews makes a complex corr flux If u take sqr rt u loose the phase inform
and for the diff phases we need them
NEED test closure phases, diff phases,
253Jy zero point for l band flux but use Kmags.

cat='/Volumes/LaCie/CalibMap/jsdc_2017_03_03.fits'
wu.search_cat(cat,24.42692,-57,23663)

#13 Marzo
#dDF1=wu.tableDRS('/Volumes/LaCie/DecemberData/All_AllBCDs/Iter1vis2','/Volumes/LaCie/DecemberData/All_AllBCDs/Iter1cf2',True)
#DF1.to_pickle('decCalsDRS.pk')
DF1.shape #(292, 62)
DF1.keys()
Index(['i', 'n', 'name', 'pBLe', 'pBLAe', 'BL', 'airmass', 'mjdo', 'dateobs',
       'mode', 'object', 'categ', 'type', 'tplstart', 'T0E', 'T0S',
       'windspeed', 'seeinge', 'seeings', 'chopping', 'chopfreq', 'BCD1',
       'BCD2', 'resolution', 'wlrange', 'wl0', 'DIT', 'fringetracker',
       'readoutMd', 'readout', 'filter', 'WL', 'lenWL', 'VIS2', 'lenVIS2',
       'RA', 'DEC', 'UCOORD', 'VCOORD', 'AO', 'CNAME', 'diam', 'Kmag', 'Lmag',
       'Mmag', 'Nmag', 'LFlux', 'MFlux', 'NFlux', 'UUDK', 'UDDL', 'UDDM',
       'UDDN', 'interpflux', 't3amp', 'T3PHI', 'VISAMP', 'VISPHI', 'CCFXAMP',
       'TF2', 'FLUX', 'ICFLUX2'],
      dtype='object')
#DF2=wu.tableEWS(inDIR='/Volumes/LaCie/DecemberData/All_AllBCDs')
#DF2.to_pickle('decCalsEWS.pk')
#DF2.shape = (199, 16)  dec cals
DF2.keys()
Out[30]:
Index(['phot', 'wave', 'cflux', 'flux', 'vis', 'mjd', 'opd', 'pbl', 'header',
       'tau', 'ra', 'dec', 'mjd-obs', 'targ', 'file', 'sky'],
      dtype='object')
all=wu.tpldata(diam=0.1)  #For the pk files at the directory
all.keys()
type(all)   #dict
all.keys()
Out[93]: dict_keys(['mjd', 'chan', 'flux', 'vis', 'phot', 'base', 'tau', 'airy', 'tf', 'tv'])
#-----Concatenating both data frames-----
#DF1=pd.read_pickle('decCalsDRS.pk') DF2=pd.read_pickle('decCalsEWS.pk') in M51!!
pd.set_option("display.precision", 8)
DF1['mjdo'] = pd.to_numeric(DF1['mjdo'], errors='coerce')
#DF2 has less decimals
for i in range(len(DF2['targ'])):
    format(DF2['mjd-obs'][i], '.8f')
#DF=pd.concat([DF1.set_index('mjdo'),DF2.set_index('mjd-obs')], axis=1, join='inner').reset_index()
DF.shape #(148, 77)
#DF.to_pickle('superp.pk') in M51!!
pd.options.display.float_format = '{:,.8f}'.format
pd.set_option('precision', 8)
DF1[['BCD1','BCD2']][0:20]
DF2["mjd-obs"] = pd.to_numeric(DF2["mjd-obs"])
pd.options.display.float_format = '${:,.8f}'.format
count=0
listNotFound=[]
for i in range(len(DF1['name'])):
    for j in range(len(DF2['targ'])):
        #print(DF1['mjdo'][i],DF2['mjd-obs'][j])  Some have 7 and some 8 digits
        if (DF1['mjdo'][i]==DF2['mjd-obs'][j]):
            count=count+1
            print(count)
            break
        if(j==len(DF2['targ'])-1): listNotFound.append(DF1['mjdo'][i])
            
#Didn't find:  from DF1
pdlist=pd.DataFrame(listNotFound,columns=['mjdo'])
pdlist.to_pickle('listNotFoundDF1.pk')
#Didn't find:  from DF2
count=0
listNotFound=[]
for i in range(len(DF2['targ'])):
    for j in range(len(DF1['name'])):
        #print(DF1['mjdo'][i],DF2['mjd-obs'][j])  Some have 7 and some 8 digits
        if (DF1['mjdo'][j]==DF2['mjd-obs'][i]):
            count=count+1
            print(count)
            break
        if(j==len(DF1['name'])-1): listNotFound.append(DF2['mjd-obs'][i])

pdlist=pd.DataFrame(listNotFound,columns=['mjd-obs'])
pdlist.to_pickle('listNotFoundDF2.pk')
#To Make table with diameters: function in wu tableDiam(Dir)

#rename column
DF2.rename(columns={'mjd-obs': 'mjdo'}, inplace=True)
DF.shape (148, 77)    DF1(292, 62) DF2(199, 16)
np.sum(DF['DEC']==DF['dec'])  # 148
np.sum(DF['RA']==DF['ra'])  #148 :) :) :)

#Pending
tpl=[]
for i in range(DF2.shape[0]):
    tpl.append(DF2['header'][i]["HIERARCH ESO TPL START"])

tpl=np.array(tpl)

#Both pipelines have 64 data points for the wavelength (for LOW resolution), but there is a difference between them. The difference in the large wavelengths is very small (0.00226) but for the short wavelengths it is a large difference (0.10307). So we should plot each set of data with its own wavelength array.
for i in range(df.shape[0]):
    plt.plot(df['WL'][0],'o',color='blue',label='DRS')
    plt.plot(df['wave'][0],'o',color='black',label='EWS')
    plt.axhline(y=min(df['wave'][0]), linewidth=2, color = 'black')
    plt.axhline(y=min(df['WL'][0]), linewidth=2, color = 'blue')
    plt.axhline(y=max(df['wave'][0]), linewidth=2, color = 'black')
    plt.axhline(y=max(df['WL'][0]), linewidth=2, color = 'blue')
    if (i==0): plt.legend()
#The same for all observations
minsWL=[]
for i in range(df.shape[0]):
    minsWL.append(np.min(df['WL'][i]))
maxsWL=[]
for i in range(df.shape[0]):
    maxsWL.append(np.max(df['WL'][i]))
#The same for all observations
minswave=[]
for i in range(df.shape[0]):
    minswave.append(np.min(df['wave'][i]))
maxswave=[]
for i in range(df.shape[0]):
    maxswave.append(np.max(df['wave'][i]))
minswave[0]-minsWL[0]  # -0.10307058233015631
maxswave[0]-maxsWL[0]   # -0.0022628878740338365

#----------------------PLOTTING-------------------------
df=pd.read_pickle('superp.pk')
DFmjd=df.sort_values(by=["index"])  #mjd-obs is the index in superp
#np.sum(df['BCD1']==df['BCD2']) #148 They are all either II or OO!! ':|
DFO=DFmjd.loc[DFmjd['BCD1'] == 'OUT']  #74
DFOO=DFO.loc[DFO['BCD2'] == 'OUT']  #  74
DF=DFOO.reset_index() #Keeps n and mjd-obs
DF.shape  #(74, 78)
DF.to_pickle('decOO.pk')
#functions in wutil

#Quick test
for i in range(6): plt.plot(df['WL'][0],df['VIS2'][0][i],color='red')
for i in range(6): plt.plot(df['WL'][0],df['vis'][0][i],color='blue')
df['BCD2'][0]
df['BCD1'][0]
df['tplstart'][0]
t=wu.mrestore('2018-11-12T04:58:24.tpl.pk')
t=t[0]
for i in range(6): plt.plot(df['WL'][0],np.abs(t['vis'][i]),color='black')
for i in range(6): plt.plot(df['WL'][0],np.abs(df['vis'][0][i]*np.exp((0+1j)*df['visphi'][0][i])),color='green')







#--------------Search files inside a directory using a pattern in their names--------------
#20 Feb to plot VIS2 from one calibrator dec7CetOO, having different tplstarts.
inDIR ='/Volumes/LaCie/DecemberData/All_AllBCDs'
patternVIS2='*RAW_VIS2_*'    #'*RAW_TF2_*'        #'*RAW_VIS2_*'
patternData=patternVIS2
patternData2='*.tpl.pk'
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.time import Time
import os, fnmatch
import pandas as pd
import statistics as stat
#For DRS reduced files
fileList = []
for root, dirnames, filenames in os.walk(inDIR):
    for filename in fnmatch.filter(filenames, patternData):   # Match string patternData=patternVIS2= '*RAW_VIS2_0001.fits'
        fileList.append(os.path.join(root, filename))

len(fileList)  # 95
#For EWS reduced files
fileList2 = []
for root, dirnames, filenames in os.walk(inDIR):
    for filename in fnmatch.filter(filenames, patternData2):   # Match string patternData2 ='*tpl.pk'
        fileList2.append(filename)

len(fileList2)  # 22 each one for each tpstart, contains multiple observations
#--------------------------Make Data Frame with the  reduced data----------------------
#For DRS reduced data Generate pandas data tables with code below
#To plot EWS using wutils

all=wu.tpldata(diam=0.1)  #For the pk files at the directory
all.keys()
type(all)   #dict
wu.zcplot(vi['mjd'],vi['vis'],vi['tau'],cb=True)
plt.ylim((0,2.0))
scl=(vi['base']<80)&(vi['base']>30)
wu.zcplot(vi['mjd'],vi['vis'],vi['tau'],cb=True,select=scl)

scl=scl&(vi['vis']<2.0)
wu.zcplot(vi['mjd'],vi['vis'],vi['tau'],cb=True,select=scl)

wu.zcplot(vi['mjd'],vi['vis'],vi['tau'],cb=True,select=scl)
#To plot EWS
from mcdb import wutil as wu
DFEWS=pd.DataFrame(columns=(['phot', 'wave', 'cflux', 'flux', 'vis', 'mjd', 'opd', 'pbl', 'header', 'tau', 'ra', 'dec', 'mjd-obs', 'targ', 'file', 'sky']))
count=0
for i in range(len(fileList2)):  #235
    red=wu.mrestore(fileList2[i])
    print('i=',i)
    for j in range(len(red)):  #Number of observation under the same tplstart
        dcr=red[j]
        dfdcrT=pd.DataFrame.from_dict(dcr, orient='index')
        dfdcr=dfdcrT.transpose()
        DFEWS=pd.concat([DFEWS, dfdcr], ignore_index=True)
        #DFEWS.append(dcr,ignore_index=True)
        count=count+1
        print(count)
        print('j=',j)

wav=DF2['wave'].convert_objects(convert_numeric=True)
/usr/local/bin/ipython3:1: FutureWarning: convert_objects is deprecated.  To re-infer data dtypes for object columns, use Series.infer_objects()
For all other conversions use the data-type specific converters pd.to_datetime, pd.to_timedelta and pd.to_numeric.
#!/usr/local/opt/python/bin/python3.7
DFEWS.to_pickle('novdecCals.pk')
DFEWS.shape  # (223, 16)


def tableEWS(inDIR):   #DOESNT WORK, converting to string is a mess :( not converting it looses decimals
    #inDIR='/Volumes/LaCie/DecemberData/All_AllBCDs'
    DFEWS=pd.DataFrame(columns=(['phot', 'wave', 'cflux', 'flux', 'vis', 'mjd', 'opd', 'pbl', 'header', 'tau', 'ra', 'dec', 'mjd-obs', 'targ', 'file', 'sky']))
    patternData='*tpl.pk'
    fileList = []
    for root, dirnames, filenames in os.walk(inDIR):
        for filename in fnmatch.filter(filenames, patternData):
            fileList.append(os.path.join(root, filename))

print(len(fileList))
count=0
    for i in range(5):#len(fileList)):  #235
        red=mrestore(fileList[i])
        print('template i=',i)
        for j in range(len(red)):  #Number of observation under the same tplstart
            dcr=red[j]
            print(dcr['mjd-obs'])
            dfdcrT=pd.DataFrame.from_dict(dcr, orient='index') #.keys()Out[47]: RangeIndex(start=0, stop=1, step=1)
            dfdcrs=dfdcrT.astype(str)
            dfdcrs=dfdcrs.transpose()  #recovers keys
            #print(dfdcr['mjd-obs'])
            DFEWS=pd.concat([DFEWS, dfdcrs], ignore_index=True)
            #print(DFEWS['mjd-obs'][count])
            #DFEWS.append(dcr,ignore_index=True)
            count=count+1
            print(count)
            print('j=',j)

print(DFEWS.shape)
return(DFEWS)

#-----------------------READING DATA FRAMES----------------------
import pandas as pd
#os.chdir('/Volumes/LaCie/DecemberData/dec7CetOO')
DF=pd.read_pickle("dec7CetOOVIS2")
DFmjd=DF.sort_values(by=["mjd"])
#DF=DFmjd.reset_index()
#Select Non-chopped:
DFNC=DFmjd.loc[DFmjd['chopping'] == 'F']
DF=DFNC.reset_index()
DF.shape  #(95, 39) NC:(23, 39)

DFEWS=pd.read_pickle("dec7CetOOred")
DF2mjd=DFEWS.sort_values(by=["mjd-obs"])  #mjd is an array of mjds
DF2=DF2mjd.reset_index()
DF2.shape   # (23, 23)  233,17
#----Making an array with ALL baselines values-----
#RedData1
arrayBLs=np.empty((len(DFmjd['name']),6))
for i in range(len(DFmjd['name'])):
    for j in range(6): arrayBLs[i][j]=np.sqrt(DFmjd["UCOORD"][i][j]**2+DFmjd["VCOORD"][i][j]**2)
np.min(arrayBLs)  # 10.7613089818378
np.max(arrayBLs) # 137.2266647606971
#RedData2
arrayBLs2=np.empty((len(DF2['targ']),6))
for i in range(len(DF2['targ'])):
    for j in range(6): arrayBLs2[i][j]=DF2["pbl"][i][j]
np.min(arrayBLs2)  # 10.851     7.915
np.max(arrayBLs2) # 136.494     137.296
#--------------Plot--------------------
#To plot ALL vis2  EWS
DF2['header'][0]['HIERARCH ESO INS BCD1 NAME']
from matplotlib import cm
cmap = plt.get_cmap('jet_r')
for i in range(DF2.shape[0]):
    hdr=DF2['header'][i]
    if((hdr['HIERARCH ESO INS BCD1 NAME']=='OUT')&(hdr['HIERARCH ESO INS BCD2 NAME']=='OUT')):   #Checking for BCDs Out-Out
        print(i)
        for b in range(1):     #Modify
            print('BL=',b)
            print('BL=',DF2['pbl'][i])
            c = cmap(float(b)/6)
            vis2=np.square(np.abs(DF2['vis'][i][b]))
            #plt.plot(DF2['wave'][i],vis2,linestyle='--',color=c,label='BL:')#+str(b)+'='+DF2['pbl'][i])
            corrflux=DF2['flux'][i][b].real
#           plt.plot(DF2['wave'][i],corrflux,linestyle='--',color=c,label='BL:')

#plt.ylim((-0.1,1.2))
plt.grid(linestyle="--",linewidth=0.5,color='.25')
#plt.legend(prop={'size': 7})


#---------------Plot only Bls with projected length between ...
#---------------less than 35m--------------------------------------------------
from matplotlib import cm
cmap = plt.get_cmap('jet_r')
keyy="VIS2"
keyx="WL"
count=0
for i in range(len(DF["targ"])):
    for j in range(0,6):
        c = cmap(float(j)/6)  #red to blue
        DF['UCOORD'][j]
        BLval=np.sqrt(DF["UCOORD"][i][j]**2+DF["VCOORD"][i][j]**2)
        if (BLval<35.):
            plt.plot(DF[keyx][i],DF[keyy][i][j], color = c, label='BL:'+str(j))
            count=count+1
            plt.text(DF[keyx][i][i+2+j*5],DF[keyy][i][j][i+2+j*5],str(j)+'  '+str(int(BLval)),fontsize=8)
            print(count)


title= str(keyy) + " vs " + str(keyx)
plt.title(title)
plt.xlabel(keyx)
plt.ylabel(keyy)
plt.ylim((-0.1,1.7))
plt.grid(linestyle="--",linewidth=0.5,color='.25')
#plt.legend(prop={'size': 7})
#plt.savefig(title)
#---------------... ADD  EWS ...-------------------
keyx='wave'
keyy='vis'
count2=0
for i in range(DF2.shape[0]):
    print(i)
    for b in range(6):
        if (DF2['pbl'][i][b]<35):
            print('BL=',b)
            count2=count2+1
            c = cmap(float(b)/6)
            vis2=np.square(np.abs(DF2[keyy][i][b]))
            plt.plot(DF2[keyx][i],vis2,linestyle='--',color=c,label='BL:'+str(b))
            plt.text(DF2[keyx][i][i+2+j*5],vis2[i+2+j*5],str(j)+'  '+str(int(DF2['pbl'][i][b])),fontsize=8)
            print(count2)

#--------------------------------End plot bl<35--------------------------------------------

#-----------------------------BINS: 35 to 60, 60 to 140------------
from matplotlib import cm
cmap = plt.get_cmap('jet_r')
minbl=60
maxbl=140
keyy="VIS2"
keyx="WL"
count=0
for i in range(len(DF["name"])):
    for j in range(0,6):
        c = cmap(float(j)/6)  #red to blue
        DF['UCOORD'][j]
        BLval=np.sqrt(DF["UCOORD"][i][j]**2+DF["VCOORD"][i][j]**2)
        if ((BLval>minbl)&(BLval<maxbl)):
            plt.plot(DF[keyx][i],DF[keyy][i][j], color = c, label='BL:'+str(j))
            count=count+1
            plt.text(DF[keyx][i][i+2+j*5],DF[keyy][i][j][i+2+j*5],str(j)+'  '+str(int(BLval)),fontsize=8)
            print(count)


title= str(keyy) + " vs " + str(keyx)
plt.title(title)
plt.xlabel(keyx)
plt.ylabel(keyy)
plt.ylim((-0.1,1.7))
plt.grid(linestyle="--",linewidth=0.5,color='.25')
#plt.legend(prop={'size': 7})
#plt.savefig(title)
#---------------... ADD  EWS ...-------------------
keyx='wave'
keyy='vis'
count2=0
for i in range(DF2.shape[0]):
    print(i)
    for b in range(6):
        if ((DF2['pbl'][i][b]>minbl)&(DF2['pbl'][i][b]<maxbl)):
            print('BL=',b)
            count2=count2+1
            c = cmap(float(b)/6)
            vis2=np.square(np.abs(DF2[keyy][i][b]))
            plt.plot(DF2[keyx][i],vis2,linestyle='--',color=c,label='BL:'+str(b))
            plt.text(DF2[keyx][i][i+2+j*5],vis2[i+2+j*5],str(j)+'  '+str(int(DF2['pbl'][i][b])),fontsize=8)
            print(count2)

#-----------------------------  END plot bins  ----------------------

#-------------------and taking average between 2 wls.----------------
from matplotlib import cm
cmap = plt.get_cmap('jet_r')
keyy="VIS2"
keyx="mjd"  #mjd       for BL comment plt.plot
minbl=10  #MIN: 10m
maxbl=140 #MAX: 140m
minwl=20   # 3.8267136
maxwl=48   # 3.2103345
count=0
for i in range(len(DF["name"])):
    for j in range(6):
        c = cmap(float(j)/6)  #red to blue
        BLval=np.sqrt(DF["UCOORD"][i][j]**2+DF["VCOORD"][i][j]**2)
        if ((BLval>minbl) & (BLval<maxbl)):
            plt.plot(BLval,np.mean(DF[keyy][i][j][minwl:maxwl]), 'o',color = c, label='BL:'+str(j))
            #plt.plot(DF[keyx][i],np.mean(DF[keyy][i][j][minwl:maxwl]), 'o',color = c, label='BL:'+str(j))
            count=count+1
                #plt.text(DF[keyx][i],DF[keyy][i][j][i+1+j],int(BLval),fontsize=8)
            print(count)

title= str(keyy) + " vs " + str(keyx)
plt.title(title)
plt.xlabel('BL(m)')
#plt.xlabel(keyx)
plt.ylabel(keyy)
plt.ylim((-0.1,0.8))
plt.grid(linestyle="--",linewidth=0.5,color='.25')
plt.legend(prop={'size': 7})
#---------------... ADD  EWS ...-------------------
#cmap=plt.get_cmap('gist_heat')
keyy='vis'
keyx='pbl'     #'wave'
count2=0
for i in range(DF2.shape[0]):
    print(i)
    for b in range(6):
        if ((DF2['pbl'][i][b]>minbl)&(DF2['pbl'][i][b]<maxbl)):
            count2=count2+1
            c = cmap(float(b)/6)
            meanvis2=np.mean(np.square(np.abs(DF2[keyy][i][b][minwl:maxwl])))
            plt.plot(DF2[keyx][i][b],meanvis2/2.0,'*',color=c,label='BL:'+str(b))
            print(count2)


#-------------------COMPARISON between 2 reductions (average between 2 wls.)----------------
#concatenating both dataframes
#Check right matching / order with  DFNC['dateobs'] and DF2['file']
DFi=DF.set_index('mjd') #data frames ordered by mjd
DF2i=DF2.set_index('mjd-obs')
DF=pd.concat([DFi,DF2i],axis=1,join_axes=[DFi.index])
#DF=pd.concat([DF,DF2],axis=1,join_axes=[DF.index])
#DF=DF.reset_index()

from matplotlib import cm
cmap = plt.get_cmap('jet_r')
keyy="VIS2"
minbl=10  #MIN: 10m
maxbl=140 #MAX: 140m
minwl=20   # 3.8267136
maxwl=48   # 3.2103345
count=0
for i, row in DF.iterrows():
    for j in range(6):
        c = cmap(float(j)/6)  #red to blue
        BLval=np.sqrt(DF["UCOORD"][i][j]**2+DF["VCOORD"][i][j]**2)
        meanvis2=np.mean(np.square(np.abs(DF['vis'][i][j][minwl:maxwl])))
        if ((BLval>minbl) & (BLval<maxbl)):
            #print(DF['T0S'][i][j])
            plt.plot(np.mean(DF[keyy][i][j][minwl:maxwl]),meanvis2, 'o',color = c, label='BL:'+str(j))
            #plt.plot(DF[keyx][i],np.mean(DF[keyy][i][j][minwl:maxwl]), 'o',color = c, label='BL:'+str(j))
            count=count+1
            #plt.text(DF[keyx][i],DF[keyy][i][j][i+1+j],int(BLval),fontsize=8)
            print(count)

plt.plot([-0.1, 1.5], [-0.1, 1.5], color = 'black', linewidth = 0.5)
title= ('VIS2 DRS vs VIS2 EWS')
plt.title(title)
plt.xlabel('VIS2 DRS')
plt.ylabel('VIS2 EWS')
plt.ylim((-0.1,1.5))
plt.grid(linestyle="--",linewidth=0.5,color='.25')
plt.legend(prop={'size': 7})
markersize=int(DF['T0S'][i][j]),

#Using pipeline version 1.1.3:  ***********************************************************
#/Users/M51/SOFT/INTROOT/matis/matisp/matis-1.1.3/matisse/mat_est_corr_lib.c
# 24 Oct  Working with L band  - 31 Oct
#First Version
#Change the name of the file according to the one generated by the search for parameters
FileTplStart = ("/Users/M51/Downloads/calsSept21-22.txt")        #Change  calib.txt
# ("/Users/M51/Downloads/ngc1068Sept.txt")
/Users/M51/Downloads/star.txt

inDIR ='/Volumes/LaCie/Reduced/Iter1Nov22sept21-22CalsLband'
/Volumes/LaCie/Reduced/Iter120NovNGC1068SeptLband
/Volumes/LaCie/Reduced/Iter1Nov22sept21-22CalsLband
/Volumes/LaCie/Reduced/Iter1Nov21sept22-23CalsLband
/Volumes/LaCie/Reduced/Iter1Nov22Sept23-24CalsLband
/Volumes/LaCie/Reduced/Iter1Nov21Sept24-25CalsLband

patternVIS2='*RAW_VIS2_*'    #'*RAW_TF2_*'        #'*RAW_VIS2_*'
patternData=patternVIS2


import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
import os, fnmatch
import pandas as pd
import statistics as stat

#The .txt needs to have no '', no commas, _ instead of :, line by line
with open(FileTplStart) as f:
    firstlistTplStart=f.read().splitlines()

firstlistTplStart[0]
len(firstlistTplStart)  #
listTplStart=pd.unique(firstlistTplStart).tolist() #To eliminate duplicates
len(listTplStart)   #
firstlistTplStart[15]   #

#To find the directory where the reduced files for each calibrators are using tplstarts
directoryList = []
fileList=[]
for i in range(len(listTplStart)):
    patternTplStart = '*'+str(listTplStart[i])+'*'
    for fList in os.walk(inDIR):
        for DirName in fList[1]:
            if fnmatch.fnmatch(DirName, patternTplStart): # Match search string
                directoryList.append(os.path.join(fList[0],DirName))

len(directoryList)    #
#To compare all arrangements of the BCDs 01-15
for i in range(len(directoryList)):
    for fList in os.walk(directoryList[i]):
        for FileName in fList[2]:
            if fnmatch.fnmatch(FileName, patternData): # Match string patternData=patternVIS2= '*RAW_VIS2_0001.fits'
                fileList.append(os.path.join(fList[0],FileName))

len(fileList)  #
#**********************************3rd version*********************************************
#Make a data frames DF from the file with the TPL_STARTs
#For VIS2 data,  n gives the number of file 0001-00n
[4].data["Sta INDEX"]  UCOORD, VCOORD
hdu[2].data
Use sqroot u^2 + v^2

#did u check residual delays?  take second diff in the tracking delays correlates well with visibilities, but for hd 16062 ... Maybe the AO is doing a bad job, but with a small spatial filter ... these were taken with pinhole...
#Diam vs flux line : constant surface brightness, in N band you can find dusty envelopes.
#CorrFlux=True doesnt work (still) By now  You can use a template spectrum and the correlate for the target
USE only the ones with NO CHOPPING
from importlib import reload as re

#pBLe and pBLAe are not in the right order.
DFtable=pd.DataFrame(columns=["i","n","name", "pBLe", "pBLAe", "airmass", "mjd", "dateobs", "mode", "object", "categ", "type", "tplstart", "T0E","T0S", "windspeed", "seeinge", "seeings" , "chopping", "chopfreq", "BCD1", "BCD2", "resolution", "wlrange"," wl0", "DIT", "fringetracker", "readoutMd", "readout", "filter", "ra", "dec"])
DF=pd.DataFrame(columns=["i", "n", "name", "pBLe", "pBLAe", "airmass", "mjd", "dateobs", "mode", "object", "categ", "type", "tplstart", "T0E","T0S", "windspeed", "seeinge", "seeings", "chopping", "chopfreq", "BCD1", "BCD2", "resolution", "wlrange", "wl0", "DIT", "fringetracker", "readoutMd", "readout", "filter", "WL","lenWL", "VIS2", "lenVIS2", "RA","DEC","UCOORD","VCOORD"])
for i in range(len(fileList)):
    n=fileList[i][-7:-5]
    hducal=fits.open(str(fileList[i]))
    object=hducal[0].header["HIERARCH ESO OBS TARG NAME"]
    pble=[hducal[0].header["HIERARCH ESO ISS PBL12 END"], hducal[0].header["HIERARCH ESO ISS PBL13 END"], hducal[0].header["HIERARCH ESO ISS PBL14 END"], hducal[0].header["HIERARCH ESO ISS PBL23 END"], hducal[0].header["HIERARCH ESO ISS PBL24 END"], hducal[0].header["HIERARCH ESO ISS PBL34 END"]]
    pblae=[hducal[0].header["HIERARCH ESO ISS PBLA12 END"], hducal[0].header["HIERARCH ESO ISS PBLA13 END"], hducal[0].header["HIERARCH ESO ISS PBLA14 END"], hducal[0].header["HIERARCH ESO ISS PBLA23 END"], hducal[0].header["HIERARCH ESO ISS PBLA24 END"], hducal[0].header["HIERARCH ESO ISS PBLA34 END"]]
    airmass=np.mean([hducal[0].header["HIERARCH ESO ISS AIRM END"],hducal[0].header["HIERARCH ESO ISS AIRM START"]])
    mjd=hducal[0].header["MJD-OBS"]
    datobs=hducal[0].header["DATE-OBS"]
    mode=hducal[0].header["HIERARCH ESO INS MODE"]
    temp=hducal[2].data["STA_INDEX"]   #temporal var    #EXTNAME = 'OI_ARRAY'
    staind=(temp == [1,5,13,10])      #indexes of the stations      #Check   ??
    si1=staind[0]                    #returns T or F
    si2=staind[1]
    si3=staind[2]
    si4=staind[3]
    objectSTD=hducal[0].header["OBJECT"]
    catg=hducal[1].data["CATEGORY"] #'CAL'#hducal[0].header["HIERARCH ESO DPR CATG"] only in original files
    typ="NA" #hducal[0].header["HIERARCH ESO DPR TYPE"]
    tpls=hducal[0].header["HIERARCH ESO TPL START"]
    tau0end=hducal[0].header["HIERARCH ESO ISS AMBI TAU0 END"]*1000# Coherence time [seconds]. in ms
    tau0start=hducal[0].header["HIERARCH ESO ISS AMBI TAU0 START"]*1000# Coherence time[seconds]  in ms
    wind=hducal[0].header["HIERARCH ESO ISS AMBI WINDSP"] # Observatory wind speed [m/s].
    seee=hducal[0].header["HIERARCH ESO ISS AMBI FWHM END"]   #Observatory seeing [arcsec].
    sees=hducal[0].header["HIERARCH ESO ISS AMBI FWHM START"] #Observatory seeing [arcsec].
    chop=hducal[0].header["HIERARCH ESO ISS CHOP ST"] #ISSChopping status=DET Chopping enabled?Sometimes it is not right!  See table Imaging data
    chopf=hducal[0].header["HIERARCH ESO ISS CHOP FREQ"]  #DET ??
    bcd1=hducal[0].header["HIERARCH ESO INS BCD1 ID"]   #'OUT' changes with 0001-0012
    bcd2=hducal[0].header["HIERARCH ESO INS BCD2 ID"]   #'OUT'
    res=hducal[0].header["HIERARCH ESO INS DIL NAME"]   #dispersorLband wewantLOW
    walrange=len(hducal[3].data["EFF_WAVE"])
    wal0=hducal[3].data["EFF_WAVE"][0]*1e6  #EXTNAME = 'OI_WAVELENGTH'
    dit=hducal[0].header["EXPTIME"]  #we want 50-120 ms
    frtr=hducal[0].header["HIERARCH ESO DEL FT STATUS"]  # 'OFF     '/Fringe Tracker Status
    ro=hducal[0].header["HIERARCH ESO DET READ CURID"]# 1 /Used readout mode id
    romd=hducal[0].header["HIERARCH ESO DET READ CURNAME"]#'SCI-SLOW-SPEED'Usedreadoutmodename
    filter=hducal[0].header["HIERARCH ESO INS SFL NAME"]#1.50/L&MSpatialFilterdevicename.
    r=hducal[0].header["RA"]
    d=hducal[0].header["DEC"]
    wl=hducal[3].data["EFF_WAVE"]*1e6
    lenwl=len(hducal[3].data["EFF_WAVE"])
    print(lenwl)
    vis2=hducal[4].data["VIS2DATA"]    # EXTNAME = 'OI_VIS2   #Saves all BL
    lenvis2=[len(hducal[4].data["VIS2DATA"][0]),len(hducal[4].data["VIS2DATA"][1]),len(hducal[4].data["VIS2DATA"][2]),len(hducal[4].data["VIS2DATA"][3]),len(hducal[4].data["VIS2DATA"][4]),len(hducal[4].data["VIS2DATA"][5])]
    print(len(vis2[0]))
    ucoord=hducal[4].data["UCOORD"]                          #Saves for all BL
    vcoord=hducal[4].data["VCOORD"]                         #Saves for all BL
    DFtable.loc[i] = [i,n,object,pble,pblae,airmass,mjd,datobs,mode,objectSTD,catg,typ,tpls,tau0end,tau0start,wind,seee,sees,chop,chopf,bcd1,bcd2,res,walrange,wal0,dit,frtr,ro,romd,filter,r,d]
    DF.loc[i] = [i,n,object,pble,pblae,airmass,mjd,datobs,mode,objectSTD,catg,typ,tpls,tau0end,tau0start,wind,seee,sees,chop,chopf,bcd1,bcd2,res,walrange,wal0,dit,frtr,ro,romd,filter,wl,lenwl,vis2,lenvis2,r,d,ucoord,vcoord]
    hducal.close()

DFtable.shape  # 42,32
DF.shape   #42, 36
#DFtable.to_csv('dec7CetOOVIS2.csv')#('ngc1068Sept-VIS2.csv')     #no vis2 data
#DF.to_pickle('dec7CetOOVIS2')#('ngc1068Sept-VIS2')         #To save in M51   to plot                    #Change
#To make subplots
vis2_01=DF.loc[DF['n'] == '01']
vis2_SourceName=DF.loc[DF['name'] == 'SourceName']

#--------------------------------------PLOTS!!-------------------
#FOR SCIENCE OBJECT
import pandas as pd
DF=pd.read_pickle("ngc1068Sept-VIS2")
#To make subtables in .csv files
#DFx=pd.read_pickle("calJuly11X-01")
#DF = DF.join(DFx)   #name=name1
import matplotlib.pyplot as plt
#colormap = plt.cm.gist_heat  #https://matplotlib.org/users/colormaps.html
#********** VIS2 vs WL - For Science Sources **********************
#PANDAS DATA FRAME
tpl=DF.loc[DF['tplstart'] == '2018-09-22T04:11:40']
#NGC 1068
#'2018-09-25T05:21:36']      #'2018-09-24T05:21:50'] 2018-09-22T04:01:55-Bad  #'2018-09-22T04:11:40']
tplsort=tpl.sort_values(by=["mjd"])
tplnew=tplsort.reset_index()
tplnew2=tplnew.loc[tplnew["chopping"]=='F']  #We want the ones where there was no chopping
DF=tplnew2
DF['n']  #7,1,8,2,9,3,10,4,11

vis2=DF.loc[DF['n'] == '01']    #Check!  vis2["n"]
vis2new= vis2.reset_index()
#TO PRINT TABLE
DFtable=pd.read_csv('/Users/M51/ngc1068Sept-VIS2.csv')
tpltable=DFtable.loc[DFtable['tplstart'] ==  '2018-09-25T05:21:36']
tpltablesort=tpltable.sort_values(by=["mjd"])
tpltablereset=tpltablesort.reset_index()
tpltablereset.to_csv('NGC10682018-09-25T052136.csv')

vis2table=DFtable.loc[DFtable['n'] == 7]  #Contains data that does not depend on BL
vis2table = vis2table.reset_index()
vis2table.to_csv('NGC1068-Sept07.csv')


DF=vis2new  #tplnew
DF.sort_values(by=["mjd"])  #order according to desired color code

DataType='RAW_VIS2'
from matplotlib import cm
cmap = plt.get_cmap('jet_r')   #viridis ('Blues')
object=". NGC 1068"
date=" Sept "                                                 #Change


keyy="VIS2"
keyx="WL"
for i in range(len(DF["name"])):
    for j in range(6):
        c = cmap(float(i)/len(DF["name"]))  #red to blue
        plt.plot(DF[keyx][i],DF[keyy][i][j], color = c, label=DF["n"][i])
        plt.text(DF[keyx][i][i*10],DF[keyy][i][j][i*10],'.'+DF["BCD1"][i]+DF["BCD2"][i],fontsize=6)

title= str(keyy) + " vs " + str(keyx) + str(object) + str(date) + " Files:" + str(DataType) + ".pdf"
plt.title(title)
plt.xlabel(keyx)
plt.ylabel(keyy)
#plt.xlim((3.0,4.25))
plt.ylim((-.05,0.6))
plt.grid(linestyle="--",linewidth=0.5,color='.25')
plt.legend(prop={'size': 7})
#plt.savefig(title)
#fig.savefig(title + '.svg', format='svg', dpi=1200)  #Has low res.
plt.show()
#Plot average visibilities between two wavelengths vs  something
import numpy as np
#Check!! It depends on the DF["lenWL"][0]  (252,110,
a=67   #67 to 3.8063288
b=86   # 86 from  3.396094
jump=math.floor((b-a)/len(DF["name"]))
keyy="VIS2"  #calculates de average
keyx='BL'     #T0S, airmass, seeinge, windspeed, diamL, fluxK/fluxL, diamK/DIT
for i in range(len(DF["name"])):
    for BL in range(6):
        #c = [float(i)/float(len(DF["name"])), 0.0, float(len(DF["name"])-i)/float(len(DF["name"]))] #R,G,B  red to blue
        print(BL)
        c = cmap(float(i)/len(DF["name"]))  #red to blue
        BLval=np.sqrt(DF["UCOORD"][i][BL]**2+DF["VCOORD"][i][BL]**2)
        plt.plot(BLval,np.mean(DF[keyy][i][BL][a:b]),'o',color=c, label=DF["n"][i])
        plt.text(BLval,np.mean(DF[keyy][i][BL][a:b]),'.'+DF["i"][i]+'_'+DF["BCD1"][i]+DF["BCD2"][i],fontsize=6)

title= str(keyy) + " vs " + str(keyx) + str(object) + str(date) + " Files:" + str(DataType) + ".pdf"
plt.title(title)
plt.xlabel(keyx)
plt.ylabel(keyy)
#plt.ylim((-0.01,.06))
plt.grid(linestyle="--",linewidth=0.5,color='.25')
#plt.legend(prop={'size': 7})
#plt.savefig(title)
plt.show()



BLm=[]
for i in range(len(DF["name"])):
    BLm.append([np.sqrt(DF["UCOORD"][i][0]**2+DF["VCOORD"][i][0]**2),np.sqrt(DF["UCOORD"][i][1]**2+DF["VCOORD"][i][1]**2),np.sqrt(DF["UCOORD"][i][2]**2+DF["VCOORD"][i][2]**2),np.sqrt(DF["UCOORD"][i][3]**2+DF["VCOORD"][i][3]**2),np.sqrt(DF["UCOORD"][i][4]**2+DF["VCOORD"][i][4]**2),np.sqrt(DF["UCOORD"][i][5]**2+DF["VCOORD"][i][5]**2)])

BLm[16:29]




#Identificar los telescopios con esos BCDs ay ver si son los mismos los que causan una mayor vis2
#Ver si se ha tomado en cuenta el extra OPD por el bcd
#Always the two shortest and the longest baselines



#********************* VIS2 vs WL - For Calibrators **********************
import matplotlib.pyplot as plt
#colormap = plt.cm.gist_heat  #https://matplotlib.org/users/colormaps.html
#PANDAS DATA FRAME
import pandas as pd
DF=pd.read_pickle("Sept21-22-VIS2")

tpl=DF.loc[DF["tplstart"]=='2018-09-22T01:40:12']
sub=DF.loc[DF["resolution"]=='LOW']  #
tpl['name']
#'2018-09-22T04:11:40'(['CAL-gam-Phe-LR-LM', 'Cal-NGC1068-LR-LM', 'Cal1-c-Sgr-(b-Sgr)','hd-183925'], dtype=object)
HD183925 '2018-09-21T23:36:26', '2018-09-21T23:49:50' HD183925-2
     bSgr-1   '2018-09-22T01:00:14', '2018-09-22T01:40:12'  bSgr 1.292Kmag
   nt so good   delPhe '2018-09-22T03:09:37', '2018-09-22T03:30:30' delPhe
HD 16658  -> '2018-09-22T04:44:55'] Kmag K 6.114 2.39Jy
#['HD 20356', 'HD175786', 'HD186791', 'HD209240'Cal-N-NGC1068, 'HD25286 ','sig Cap ']
([sigCap'2018-09-23T23:25:55', '2018-09-23T23:56:04', sigCap
HD175786  '2018-09-24T01:26:20', '2018-09-24T01:54:17',  HD175786-2
  HD175786-3   '2018-09-24T02:11:00', '2018-09-24T02:51:55',  HD186791
HD209240  '2018-09-24T03:24:24', '2018-09-24T03:52:16', HD209240-2
 HD20356 '2018-09-24T08:03:17', '2018-09-24T08:29:25', HD20356-2
HD25286  '2018-09-24T09:08:58', '2018-09-24T09:38:05'] HD25286-2
#Calsngc1068:2018-09-25T05:     21:36: 2018-09-25T06:02:54' 47Cap-3 2018-09-25T04:39:44 47 Cap-2
array(['2018-09-25T00:32:12'  HD 172453, '2018-09-25T01:52:02'  -1,
     HD209240-1  '2018-09-25T02:29:14', '2018-09-25T02:37:24',   HD209240-3
    47 Cap   '2018-09-25T04:28:31', '2018-09-25T04:39:44',  47 Cap-2
   47Cap-3    '2018-09-25T06:02:54', '2018-09-25T06:20:02'HD 20356
       '2018-09-25T09:41:58']  HD33162
#PENDING
#array(['CAL-gam-Phe-LR-LM', 'Cal-MWC-300-HR-LR', 'Cal-NGC1566','HD-195677-AllR-N'], dtype=object)
      '2018-09-23T01:50:29', '2018-09-23T03:43:28',
      '2018-09-23T04:05:22', '2018-09-23T04:30:46',
      '2018-09-23T04:49:00', '2018-09-23T05:02:13',
      '2018-09-23T07:32:18', '2018-09-23T09:13:51',
      '2018-09-23T09:19:42', '2018-09-23T09:32:24']
666.7*10**(-0.4*KMAG)
      
tplsort=tpl.sort_values(by=["mjd"])
tplnew=tplsort.reset_index()
DF=tplnew
      
tplnew2=tplnew.loc[tplnew["chopping"]=='F']  #We want the ones where there was no chopping?
tplsort2=tplnew2.reset_index()
DF=tplsort2
DF['n']  #7,1,8,2,9,3,10,4,11

vis2=DF.loc[DF['n'] == '12']    #Check!  vis2["n"]
vis2new= vis2.reset_index()
DF=vis2new
#TO PRINT TABLE
DFtable=pd.read_csv('/Users/M51/-VIS2.csv')
tpltable=DFtable.loc[DFtable['tplstart'] ==  '2018-09-22T01:00:14']
tpltablesort=tpltable.sort_values(by=["mjd"])
tpltablereset=tpltablesort.reset_index()
tpltablereset.to_csv('36.csv')

vis2table=DFtable.loc[DFtable['n'] == 7]  #Contains data that does not depend on BL
vis2table = vis2table.reset_index()
vis2table.to_csv('NGC1068-Sept07.csv')


DF=vis2new  #tplnew
DF.sort_values(by=["mjd"])  #order according to desired color code

DataType='RAW_VIS2'
from matplotlib import cm
cmap = plt.get_cmap('jet_r')   #viridis ('Blues')
object=". Cals"
date=" Sept "                                                 #Change

# Plot 1
keyy="VIS2"
keyx="WL"
for i in range(len(DF["name"])):
    for j in range(0,6):
        c = cmap(float(j)/6)  #red to blue
        plt.plot(DF[keyx][i],DF[keyy][i][j], color = c, label=DF["n"][i])
        plt.text(DF[keyx][i][70+j*3],DF[keyy][i][j][70+j*3],'.'+str(j) + str(DF["BCD1"][i])+str(DF["BCD2"][i]),fontsize=8)
    title= str(i)+'.' + str(DF["name"][0]) + str(DF["diammas"][0]) + 'mas' + str(DF["tplstart"][i]) +'.pdf'
    plt.title(title)
    plt.xlabel(keyx)
    plt.ylabel(keyy)
    #plt.xlim((3.0,4.25))
    plt.ylim(-0.05,1.0)
    plt.grid(linestyle="--",linewidth=0.5,color='.25')
    plt.legend(prop={'size': 6})
    plt.savefig(title)
    plt.show()

title= str(i)+'.' + str(DF["name"][0]) + str(DF["diammas"][0]) + 'mas' + str(DF["tplstart"][i]) +'.pdf'
plt.title(title)
plt.xlabel(keyx)
plt.ylabel(keyy)
#plt.xlim((3.0,4.25))
plt.ylim(-0.05,1.0)
plt.grid(linestyle="--",linewidth=0.5,color='.25')
plt.legend(prop={'size': 6})
plt.savefig(title)
plt.show()

#For only one source
#CD-55 855: 2018-09-23T09:13:51 and 2018-09-23T09:19:42
t=DF.loc[DF["name"]=='CD-55 855'] # HD 16658 2018-09-22T04:44:55
m=31
n=38
for i in range(m,n):
    for j in range(0,6):
      c = cmap(float(j)/6)
      plt.plot(t[keyx][i],t[keyy][i][j],color=c,label=int(t["BLm"][i][j]))
      plt.text(t[keyx][i][70+j*6],t[keyy][i][j][70+j*6],'.'+str(j) + str(t["BCD1"][i])+str(t["BCD2"][i]),fontsize=6)


plt.text(4.2, .30, str(t["name"][m+1]) , {'color': 'r', 'fontsize': 20})
plt.text(4.2, .27, 'T0=' + str(t["T0S"][m+1]) , {'color': 'b', 'fontsize': 20})
plt.text(4.2, .24, 'seeing:' + str(t["seeinge"][m+1]) , {'color': 'b', 'fontsize': 20})
plt.text(4.2, .21, 'airmass:' + str(t["airmass"][m+1]) , {'color': 'b', 'fontsize': 20})
plt.text(4.2, .18, 'windspeed:' + str(t["windspeed"][m+1]), {'color': 'b', 'fontsize': 20})
title = str(t["tplstart"][m+1]) +'.pdf'
plt.title(title)
plt.xlabel(keyx)
plt.ylabel(keyy)
plt.ylim(-0.05,.3)
plt.grid(linestyle="--",linewidth=0.5,color='.25')
plt.legend(prop={'size': 6})
plt.show()
# Plot 2
#Plot average visibilities between two wavelengths vs  something
import numpy as np
a=75   # 3.8063288
b=95   # 3.3732965
jump=math.floor((b-a)/len(DF["name"]))
keyy="VIS2"  #calculates de average
keyx="BLm"     #BLm   T0S, airmass, seeinge, windspeed, diamL, fluxK/fluxL, diamK/DIT
for i in range(len(DF["name"])):
    for j in range(0,6):
        #c = [float(i)/float(len(DF["name"])), 0.0, float(len(DF["name"])-i)/float(len(DF["name"]))] #R,G,B redtoblue
        c = cmap(float(i)/len(DF["name"]))  #red to blue
        plt.plot(DF[keyx][i][j],np.mean(DF[keyy][i][j][a:b]),'o',color=c, label=DF["n"][i])
        plt.text(DF[keyx][i][j],np.mean(DF[keyy][i][j][a:b]),'.'+DF["n"][i]+'_'+DF["BCD1"][i]+DF["BCD2"][i],fontsize=6)
        #BLval=np.sqrt(DF["UCOORD"][i][j]**2+DF["VCOORD"][i][j]**2)
        #plt.plot(BLval,np.mean(DF[keyy][i][j][a:b]),'o',color=c, label=DF["n"][i])
        #plt.text(BLval,np.mean(DF[keyy][i][j][a:b]),'.'+DF["n"][i]+'_'+DF["BCD1"][i]+DF["BCD2"][i]+DF["chopping"][i],fontsize=6)
      
for bl in range(0,125):
    plt.plot(bl,np.square(2*jv(1,pi*DF["diamrad"][0]*bl*1e6/3.7)/(pi*DF["diamrad"][0]*bl*1e6/3.7)),'o', color='black')

title= str(keyy) + " vs " + str(keyx) + ".pdf"
plt.title(title)
plt.xlabel(keyx)
plt.ylabel(keyy)
plt.ylim((-0.01,.8))
plt.grid(linestyle="--",linewidth=0.5,color='.25')
#plt.legend(prop={'size': 7})
#plt.savefig(title)
plt.show()
      
#Plot 3   T0S, airmass, seeinge, windspeed, only 1st BL
import numpy as np

a=75   # 3.8063288
b=95   # 3.3732965
jump=math.floor((b-a)/len(DF["name"]))
keyy="VIS2"  #calculates de average
keyx="diammas"     #T0S, airmass, seeinge, windspeed, diamL, fluxK/fluxL, diamK/DIT
#["BLm"][0] is always around 60m. ["BLm"][1] 55m
for i in range(len(DF["name"])):
   for j in range(0,1):  #For the first base line only
      #c = [float(i)/float(len(DF["name"])), 0.0, float(len(DF["name"])-i)/float(len(DF["name"]))] #R,G,B redtoblue
      c = cmap(float(i)/len(DF["name"]))  #red to blue
      plt.plot(DF[keyx][i],np.mean(DF[keyy][i][j][a:b]),'o',color=c, label=DF["n"][i])
      plt.text(DF[keyx][i],np.mean(DF[keyy][i][j][a:b]),'.'+DF["name"][i]+DF["n"][i],fontsize=6)
      #+'_'+DF["BCD1"][i]+DF["BCD2"][i]
      #BLval=np.sqrt(DF["UCOORD"][i][j]**2+DF["VCOORD"][i][j]**2)
      #plt.plot(BLval,np.mean(DF[keyy][i][j][a:b]),'o',color=c, label=DF["n"][i])
      #plt.text(BLval,np.mean(DF[keyy][i][j][a:b]),'.'+DF["n"][i]+'_'+DF["BCD1"][i]+DF["BCD2"][i]+DF["chopping"][i],fontsize=6)
      
      
title= str(keyy) + " vs " + str(keyx) + ".pdf"
plt.title(title)
plt.xlabel(keyx)
plt.ylabel(keyy)
plt.ylim((-0.01,.8))
plt.grid(linestyle="--",linewidth=0.5,color='.25')
#plt.legend(prop={'size': 7})
#plt.savefig(title)
plt.show()
      
#Plot 4  T0S, airmass, seeinge, windspeed, only 1st BL
a=75   # 3.8063288
b=95   # 3.3732965
keyy="VIS2"  #calculates de average
# fluxK/fluxL
#["BLm"][0] is always around 60m. ["BLm"][1] 55m
DF=DF.loc[DF["tplstart"]!='2018-19-22T03:09:37'] delPhe
DF=DF.loc[DF["tplstart"]!='2018-19-22T01:40:12'] BSgr
      
for i in range(len(DF["name"])):
   temp=cat.loc[cat["name1"] == DF["name"][i]]
   temp=temp.reset_index()
   flux=temp["FluxK"][0]
   for j in range(1,2):  #For the first base line only
      #c = [float(i)/float(len(DF["name"])), 0.0, float(len(DF["name"])-i)/float(len(DF["name"]))] #R,G,B redtoblue
      c = cmap(float(i)/len(DF["name"]))  #red to blue
      plt.plot(flux,np.mean(DF[keyy][i][j][a:b]),'o',color=c, label=DF["name"][i]+DF["tplstart"][i])
      plt.text(flux,np.mean(DF[keyy][i][j][a:b]),'.'+DF["name"][i]+DF["tplstart"][i]+'_'+DF["n"][i],fontsize=6)
      #+'_'+DF["BCD1"][i]+DF["BCD2"][i]
      #BLval=np.sqrt(DF["UCOORD"][i][j]**2+DF["VCOORD"][i][j]**2)
      #plt.plot(BLval,np.mean(DF[keyy][i][j][a:b]),'o',color=c, label=DF["n"][i])
      #plt.text(BLval,np.mean(DF[keyy][i][j][a:b]),'.'+DF["n"][i]+'_'+DF["BCD1"][i]+DF["BCD2"][i]+DF["chopping"][i],fontsize=6)
      
      
title= str(keyy) + " vs " + "FluxK[Jy]" + ".pdf"
plt.title(title)
plt.xlabel("FluxK[Jy]")
plt.ylabel(keyy)
plt.ylim((-0.01,.5))
plt.grid(linestyle="--",linewidth=0.5,color='.25')
#plt.legend(prop={'size': 4})
#plt.savefig(title)
plt.show()
      
#PLot 5
keyy="VIS2"
keyx="WL"
DF=DF.loc[DF["tplstart"]=='2018-09-22T03:09:37'] #  delPhe
DF=DF.loc[DF["tplstart"]=='2018-09-22T01:00:14']  #bSgr  'T01:40:12'
DF=DF.reset_index()
for i in range(7,9):   #len(DF["name"])):
   for j in range(0,6):
      c = cmap(float(i)/(len(DF["name"])))  #red to blue
      plt.plot(DF[keyx][i],DF[keyy][i][j], color = c, label=DF["n"][i])
      plt.text(DF[keyx][i][70+j*3],DF[keyy][i][j][70+j*3],'.'+str(j) + str(DF["BCD1"][i])+str(DF["BCD2"][i]),fontsize=8)

title= str(i)+'.' + str(DF["name"][0]) + str(DF["diammas"][0]) + 'mas' + str(DF["tplstart"][i]) +'.pdf'
plt.title(title)
plt.xlabel(keyx)
plt.ylabel(keyy)
#plt.xlim((3.0,4.25))
plt.ylim(-0.05,1.0)
plt.grid(linestyle="--",linewidth=0.5,color='.25')
plt.legend(prop={'size': 6})
plt.savefig(title)
plt.show()
#****************  Scatter with color coding ********
#Plot averaged visibilities and TF between two wavelengths vs VARIOUS
a=75   #index 30 to 3.617436,    20 to 3.8267136
b=95   #From  3.396094
keyy="VIS2"      #calculates the average   TF, modelVIS2, VIS2
keyx="FluxK"  #  fluxK      diamL, seeing/fluxL diamK/
keycolor="T0S"    #seeinge, T0S, mjd, airmass, windspeed
x=np.zeros(8)   #-6 for Jul11th
y=np.zeros(8)
z=np.zeros(8)
l=[]
s=np.zeros(8)
j=0
for i in range(len(DF["name"])):
    x[j]=DF[keyx][i] #+6 for July 11th
    y[j]=np.mean(DF[keyy][i][BL][a:b])
    z[j]=DF[keycolor][i]
    l.append(DF["name"][i])
    s=DF["diamL"][i]
    j=j+1
      
      #fig, ax = plt.subplots()
      #ax.scatter(x,y,c=z)
      #cbar = plt.colorbar()  #doesn't work
      #cbar.set_label(keycolor,labelpad=-1)
      #for i, txt in enumerate(l):
      #    ax.annotate(txt, (x[i], y[i]))
      
plt.scatter(x,y,c=z,s=(s*10)**3, label=l)  #between.1 and 10:(s*10)**3
cbar = plt.colorbar()
cbar.set_label(keycolor,labelpad=-1)
title= "SP " + str(keyy) + " vs " + str(keyx) + str(object) + str(date) + " BL:" + str(BL) + " Files:" + str(DataType) + ".pdf"
plt.title(title)
plt.xlabel(keyx)
plt.ylabel(keyy)
plt.xlim(0,30)  #To lim flux
#plt.ylim((-0.1,0.6))
plt.grid(linestyle="--",linewidth=0.5,color='.25')
#plt.legend(prop={'size': 7})
for i, txt in enumerate(l):
    plt.annotate(txt, (x[i], y[i]), size=9)
      
#plt.savefig(title)
plt.show()

      
      
# ** Algo que prenda y apague , calentador, enfriador
# ** Fringe tracker se queda fijo en el cambio de BCDs la senal va a bajar

#TO DO
#Now using the raw files to check for chopping
#1.- meter la estrella 7 9 mas, hacer modelo, comparar las BLs coomo cambian

      
666.7*10**(-0.4*KMAG)
cat=pd.DataFrame(columns=["name1","name2","SpType","Kmag","FluxK","Lmag","Mmag","Nmag","UDDK","UDDL","UDDM","UDDN"])
cat.loc[0] = ['HD 172453', '','K1III',4.418   , 11.40   , 3.865   , 3.494   , 4.342000  ,   0.666512  ,  0.672591   , 0.675094 ,   0.678026]
cat.loc[1] = ['CD-55 855', '' , '' , 6.779, 1.30, 0, 0, 0, 0, 0, 0, 0 ]
cat.loc[2] = ['HD 195677', '',   'K5/M0III',    3.809  ,  19.97 ,   3.828 ,   3.466  ,  3.738000   , 1.012132,    1.018803  ,  1.021276  ,  1.024689]
cat.loc[3] = ['del Phe',  '',  'G9III' ,   1.629 ,   148.71 ,  0  ,   0  ,  1.766000 ,   2.257172  ,  2.276834 ,   2.283964  ,  2.293497]
cat.loc[4] = [ 'HD 20356',  '',  'K4III',    2.310  ,  79.42 , 0   ,  0  ,   2.319000  ,  1.866445  ,  1.884123,    1.891639 ,   1.899920]
cat.loc[5] = [ 'HD175786'  ,'',  'M1/2III',    2.899  ,  46.17  ,  2.979  ,  2.826 ,   2.928000 ,   1.636825,    1.663902,    1.664566 ,   1.669321]
cat.loc[6] = ['HD186791','* gam Aql' ,'K3II',  -0.720,  1293 ,  0   ,    0  ,-0.452000,    7.195430  ,  7.262040,    7.289916  ,  7.321740]
cat.loc[7] = ['HD209240' ,'HR 8394' ,   'K0III' ,   4.091  ,  15.40  ,  3.887  ,  3.742  ,  3.932000  ,  0.702797 ,   0.708984  ,  0.711296 ,   0.714291]
cat.loc[8] = [ 'HD25286', '',    'K2III' ,    5.538  ,  4.06  ,  5.462 ,   5.436 ,   5.487000  ,  0.397596 ,   0.401215 ,   0.402692  ,  0.404437]
cat.loc[9] = ['sig Cap',  'HD193150',  'K2III',    2.048  ,  101.10  ,  1.948 , 1.985  ,  2.099000  ,  2.009662  ,  2.027952 ,   2.035420  ,  2.044238]
cat.loc[10] = [ '47 Cap' , '',    'M2III' ,    1.097  ,  242.73 ,  0   ,   0  , 1.095000  ,  3.801393  ,  3.836039,    3.835713   , 3.846619]
cat.loc[11] = ['HD 172453','',    'K1III' ,   4.418 ,   11.40 ,   3.865 ,   3.494  ,  4.342000 ,   0.666512,    0.672591 ,   0.675094  ,  0.678026]
cat.loc[12] = ['HD 20356', '',    'K4III',    2.310 ,   79.42 ,  0  ,   0  ,  2.319000  ,  1.866445  ,  1.884123,    1.891639  ,  1.899920]
cat.loc[13] = ['HD209240','HR 8394',    'K0III',    4.091 ,   15.40  ,  3.887 ,   3.742  ,  3.932000  ,  0.702797  ,  0.708984  ,  0.711296 ,   0.714291]
cat.loc[14] = ['HD33162' , '',    'M1III' ,   1.602  ,  152.45 , 0, 0,  1.760000   , 2.961983  ,  2.983991  ,  2.986506  ,  2.995028]
cat.loc[15] = ['V4026 Sgr' , 'HIP97351',    'M2III'   , 1.536  ,  162.01   ,0 , 0,   1.669000   , 3.207913,    3.237149  ,  3.236874  ,  3.246078]
cat.loc[16] = ['HD 183925', '',  'K5III',2.915, 45.49, 0, 2.673, 2.88, 1.435707  ,  1.449880  ,  1.455678  ,  1.461931]
cat.loc[17] = ['b Sgr', '* b Sgr', 'K3III', 1.292, 202.83, 0, 0, 1.362000, 2.897324  ,  2.924145  ,  2.935370 ,   2.948184]
cat.loc[18] = ['HD 16658', '', 'K0III', 6.114, 2.3896, 6.104  ,  6.090 ,   6.093000, 0.287772   , 0.290305  ,  0.291252  ,  0.292478 ]
      
cat.to_pickle('catSept')
      
#Now using the files RAW_TF2_00-n
DFx=pd.DataFrame(columns=['tpls','n2','TF2','ModVIS2'])
for i in range(len(fileList)):
    n=fileList[i][-7:-5]
    hducal=fits.open(str(fileList[i]))
    tf2=hducal[3].data['TF2']  #Saves all BL
    tp=hducal[0].header["HIERARCH ESO TPL START"]
    hducal.close()
    mv2=0  #PEND
    DFx.loc[i]=[tp,n,tf2,mv2]

DFx.to_pickle("Sept21-22-TF2")
      
      
#Plotting for Calibrators the TF
import pandas as pd
import matplotlib.pyplot as plt
#Assuming that for all VIS2 we could get a TF2

DF=pd.read_pickle("Sept21-22-VIS2")
DFx=pd.read_pickle("Sept21-22-TF2")
DF = DF.join(DFx)
      tpl=DF.loc[DF['tplstart'] ==  '2018-09-24T02:51:55'] #'2018-09-22T04:44:55']
tpl['tpls'] #Check that is the same
tpl['name']
tplsort=tpl.sort_values(by=["mjd"])
tplnew=tplsort.reset_index()
DF=tplnew

From the raw data
hdu[2].data["TARTYP"]
chararray(['T', 'T', 'T', 'T'], dtype='<U1')
MATIS.2018-09-25T02:37:24.
#'2018-09-22T04:11:40'(['CAL-gam-Phe-LR-LM', 'Cal-NGC1068-LR-LM', 'Cal1-c-Sgr-(b-Sgr)','hd-183925'], dtype=object)
HD183925 '2018-09-21T23:36:26', '2018-09-21T23:49:50' HD183925-2
bSgr-1   '2018-09-22T01:00:14', '2018-09-22T01:40:12'  bSgr 1.292Kmag
nt so good   delPhe '2018-09-22T03:09:37', '2018-09-22T03:30:30' delPhe
HD 16658  -> '2018-09-22T04:44:55'] Kmag K 6.114 2.39Jy
#['HD 20356', 'HD175786', 'HD186791', 'HD209240'Cal-N-NGC1068, 'HD25286 ','sig Cap ']
([sigCap'2018-09-23T23:25:55', '2018-09-23T23:56:04', sigCap
HD175786  '2018-09-24T01:26:20', '2018-09-24T01:54:17',  HD175786-2
HD175786-3   '2018-09-24T02:11:00', '2018-09-24T02:51:55',  HD186791
HD209240  '2018-09-24T03:24:24', '2018-09-24T03:52:16', HD209240-2
HD20356 '2018-09-24T08:03:17', '2018-09-24T08:29:25', HD20356-2
HD25286  '2018-09-24T09:08:58', '2018-09-24T09:38:05'] HD25286-2
#Calsngc1068:2018-09-25T05:     21:36: 2018-09-25T06:02:54' 47Cap-3 2018-09-25T04:39:44 47 Cap-2
array(['2018-09-25T00:32:12'  HD 172453, '2018-09-25T01:52:02'  -1,
HD209240-1  '2018-09-25T02:29:14', '2018-09-25T02:37:24',   HD209240-3
47 Cap   '2018-09-25T04:28:31', '2018-09-25T04:39:44',  47 Cap-2
47Cap-3    '2018-09-25T06:02:54', '2018-09-25T06:20:02'HD 20356
'2018-09-25T09:41:58']  HD33162
#PENDING
#array(['CAL-gam-Phe-LR-LM', 'Cal-MWC-300-HR-LR', 'Cal-NGC1566','HD-195677-AllR-N'], dtype=object)
2018-09-23T01:50:29', '2018-09-23T03:43:28',
'2018-09-23T04:05:22', '2018-09-23T04:30:46',
'2018-09-23T04:49:00', '2018-09-23T05:02:13',
'2018-09-23T07:32:18', '2018-09-23T09:13:51',
'2018-09-23T09:19:42', '2018-09-23T09:32:24']
666.7*10**(-0.4*KMAG)

             
DataType='RAW_TF2'
from matplotlib import cm
cmap = plt.get_cmap('jet_r')   #viridis ('Blues')
object=". Cals"
date=" Sept "                                                 #Change
             
             
keyy="TF2"
keyx="WL"
for i in range(len(DF["name"])):
    for j in range(0,6):
      c = cmap(float(i)/len(DF["name"]))  #red to blue
      plt.plot(DF[keyx][i],DF[keyy][i][j], color = c, label=DF["n"][i])
      plt.text(DF[keyx][i][i*3],DF[keyy][i][j][i*3],'.'+DF["BCD1"][i]+DF["BCD2"][i]+DF["chopping"][i],fontsize=6)
#ModelVIS
keyy="TF2/VIS2"
keyx="WL"
for i in range(len(DF["name"])):
    for j in range(0,6):
      c = cmap(float(j)/6)  #red to blue
      plt.plot(DF[keyx][i],DF["TF2"][i][j]/DF["VIS2"][i][j], color = c, label=DF["n"][i])
      #plt.text(DF[keyx][i][i*3],DF["TF2"][i][j]/DF["VIS2"][i][j],'.'+DF["BCD1"][i]+DF["BCD2"][i]+DF["chopping"][i],fontsize=6)

title= str(keyy) + " vs " + str(keyx) + str(object) + str(date) + " Files:" + str(DataType) + ".pdf"
plt.title(title)
plt.xlabel(keyx)
plt.ylabel(keyy)
#plt.xlim((3.0,4.25))
#plt.ylim(-0.05,1.0)
plt.grid(linestyle="--",linewidth=0.5,color='.25')
plt.legend(prop={'size': 7})
#plt.savefig(title)
plt.show()


import math
from scipy.special import jv
import numpy as np
from numpy import pi, cos, sin
DF=pd.read_pickle("Sept24-25-VIS2")
DFxx=pd.DataFrame(columns=["BLm","BLum","diammas","diamrad","modelvis2","tf"])  #missing FluxK
for i in range(len(DF["name"])):
    #nam=DF["name"][i]
    temp=cat.loc[cat["name1"] == DF["name"][i]]
    temp=temp.reset_index()
    diammas=temp["UDDL"][0] #in case it found multiple coincidences
    diamrad=math.radians(diammas)/3.6e6 #in mas -> rad
     #error index out of bounds meanas it did not find a match in cat
    BLm=[np.sqrt(DF["UCOORD"][i][0]**2+DF["VCOORD"][i][0]**2),np.sqrt(DF["UCOORD"][i][1]**2+DF["VCOORD"][i][1]**2),np.sqrt(DF["UCOORD"][i][2]**2+DF["VCOORD"][i][2]**2),np.sqrt(DF["UCOORD"][i][3]**2+DF["VCOORD"][i][3]**2),np.sqrt(DF["UCOORD"][i][4]**2+DF["VCOORD"][i][4]**2),np.sqrt(DF["UCOORD"][i][5]**2+DF["VCOORD"][i][5]**2)]
    BLum=[BLm[0]*1e6, BLm[1]*1e6, BLm[2]*1e6, BLm[3]*1e6, BLm[4]*1e6, BLm[5]*1e6]  # m -> um, index for the cal, Carries info for 6  diff BLs
    SpatFreq=[BLum[0]/DF["WL"][i], BLum[1]/DF["WL"][i], BLum[2]/DF["WL"][i], BLum[3]/DF["WL"][i], BLum[4]/DF["WL"][i], BLum[5]/DF["WL"][i]]   #For 6 diff BLs
    modelvis2 = [np.square(2*jv(1,pi*diamrad*BLum[0]/DF["WL"][i])/(pi*diamrad*BLum[0]/DF["WL"][i])), np.square(2*jv(1,pi*diamrad*BLum[1]/DF["WL"][i])/(pi*diamrad*BLum[1]/DF["WL"][i])), np.square(2*jv(1,pi*diamrad*BLum[2]/DF["WL"][i])/(pi*diamrad*BLum[2]/DF["WL"][i])), np.square(2*jv(1,pi*diamrad*BLum[3]/DF["WL"][i])/(pi*diamrad*BLum[3]/DF["WL"][i])), np.square(2*jv(1,pi*diamrad*BLum[4]/DF["WL"][i])/(pi*diamrad*BLum[4]/DF["WL"][i])), np.square(2*jv(1,pi*diamrad*BLum[5]/DF["WL"][i])/(pi*diamrad*BLum[5]/DF["WL"][i]))] #For 6 diff BLs
    print(i)
    print(len(modelvis2[0]))
    print(len(DF["VIS2"][i][0]))
    print(len(DF["WL"][i]))   #  95-3.396 to 75-3.806
    #AvgVIS2=[np.mean(DF["VIS2"][i][0][75:95]),np.mean(DF["VIS2"][i][1][75:95]),np.mean(DF["VIS2"][i][2][75:95]),np.mean(DF["VIS2"][i][3][75:95]),np.mean(DF["VIS2"][i][4][75:95]),np.mean(DF["VIS2"][i][5][75:95])]
    #AvgModelVIS2=[np.mean(modelvis2[0][75:95]), np.mean(modelvis2[1][75:95]), np.mean(modelvis2[2][75:95]), np.mean(modelvis2[3][75:95]), np.mean(modelvis2[4][75:95]), np.mean(modelvis2[5][75:95])]   #For 6 diff BLs
    #AvgTF=[AvgVIS2[0]/AvgModelVIS2[0], AvgVIS2[1]/AvgModelVIS2[1], AvgVIS2[2]/AvgModelVIS2[2], AvgVIS2[3]/AvgModelVIS2[3], AvgVIS2[4]/AvgModelVIS2[4], AvgVIS2[5]/AvgModelVIS2[5]]
    TF=[np.divide(DF["VIS2"][i][0],modelvis2[0]), np.divide(DF["VIS2"][i][1],modelvis2[1]), np.divide(DF["VIS2"][i][2],modelvis2[2]), np.divide(DF["VIS2"][i][3],modelvis2[3]), np.divide(DF["VIS2"][i][4],modelvis2[4]), np.divide(DF["VIS2"][i][5],modelvis2[5]) ]#Dividestermbyterm For 6 diff BLs
    DFxx.loc[i]=[BLm,BLum,diammas,diamrad,modelvis2,TF]
     
      
DFxx.to_pickle('DFxxSept24-25')
      
      
>>> len(np.unique(DF1["name"])) low res.
      3
>>> len(np.unique(DF2["name"]))
      4
>>> len(np.unique(DF3["name"]))
      3
>>> len(np.unique(DF4["name"]))
      6
DF=pd.concat([DF1, DF2, DF3, DF4], ignore_index=True)
#pickles containing data frames Nov26
#cat.to_pickle('catSept')
#df  =pd.concat([OO,II], ignore_index=True) (low res only II and OO, same columns)
#SeptLOWNCOO
#SeptLOWNCII
#SeptLOWNC (non-chopped)
#SeptLOW  (join for the 4 days and only with low res)
#Model21-22 (join DF and DFxx, diff columns)
#DFxxSept21-22
#Sept21-22-VIS2 (has table .csv)
#---------------------------------MORE-----PLOTS!!-------------------
import pandas as pd
DF=pd.read_pickle("Sept21-22-VIS2")
DFxx=pd.read_pickle("DFxxSept21-22")
DF = DF.join(DFxx)
DF.to_pickle("Model21-22")
#DF=pd.read_pickle("ngc1068Sept-VIS2")
#To make subtables in .csv files
#DFx=pd.read_pickle("calJuly11X-01")

import matplotlib.pyplot as plt
#colormap = plt.cm.gist_heat  #https://matplotlib.org/users/colormaps.html
      
DataType='modelvis2'
from matplotlib import cm
cmap = plt.get_cmap('jet_r')   #viridis ('Blues')
object=". Cals"
date=" Sept "                                                 #Change
      
tpl=DF.loc[DF['tplstart'] == '2018-09-22T04:44:55'] #HD16658 #'2018-09-24T02:51:55' ]  # '2018-09-24T03:24:24']
#HD209240  '2018-09-24T03:24:24', '2018-09-24T03:52:16', HD209240-2
#'2018-09-24T02:51:55']    HD186791 diam7.26204
tpl['tplstart'] #Check that is the same
tpl['name']
tplsort=tpl.sort_values(by=["mjd"])
tplnew=tplsort.reset_index()
DF=tplnew
      
keyy="modelvis2"
keyx="WL"
for i in range(len(DF["name"])):
    for j in range(0,6):
      c = cmap(float(i)/len(DF["name"]))  #red to blue
      plt.plot(DF[keyx][i],DF[keyy][i][j],label=DF["n"][i],color=c) #exchange labe and color order.label:j, n,
      #plt.text(DF[keyx][i][30+i*12],DF[keyy][i][j][30+i*12],'.'+DF["BCD1"][i]+DF["BCD2"][i]+DF["chopping"][i]+'____'+str(int(DF["BLm"][i][j])),fontsize=6)  #400+i*120 HD186791

#title= str(keyy) + " vs " + str(keyx) + str(object) + str(date) + " Files:" + DataType + ".pdf"
#plt.title(title)
plt.xlabel(keyx)
plt.ylabel(keyy)
#plt.xlim((3.0,4.25))
#plt.ylim(-0.05,1.0)
plt.grid(linestyle="--",linewidth=0.5,color='.25')
plt.legend(prop={'size': 4})
#plt.savefig(title)
plt.show()
      
>>> DF["BLm"][0]    IN  IN
[40.507278215229675, 55.85751575859407, 123.7508747854928, 102.17465910813404, 74.60910314784135, 46.60265430063254]
>>> DF["BLm"][1]    OUT OUT
[40.29800395224937, 55.8809802883101, 46.59685472089122, 74.3922411062533, 102.19499067550476, 123.59417278554515]
>>> DF["BLm"][2]    IN IN
[39.90221060374678, 55.92402444844926, 123.29480310240986, 102.23084237272532, 73.98060124278263, 46.58466461893916]
>>> DF["BLm"][3]    OUT OUT
[39.59800652881609, 55.95594193769539, 46.5742302882055, 73.66291451197786, 102.25612690158567, 123.06209771380156]
      #See mat_det_opdmod
#Plot average modeled visibilities and TF between two wavelengths vs VARIOUS
  #  95-3.396 to 75-3.806
a=75   #index 30 to 3.617436, 20 to 3.8267136
b=95   #From  3.396094
jump=math.floor((b-a)/len(DF["name"]))
keyy="tf"    #"tf" #"modelvis2"  #calculates de average
keyx="BLm"  # diamL, fluxK/fluxL, diamK/
for i in range(len(DF["name"])):
      for j in range(0,6):
        c = cmap(float(i)/len(DF["name"]))  #red to blue
        plt.plot(DF[keyx][i][j],np.mean(DF[keyy][i][j][a:b]),'o', label=DF["n"][i],color = c) #exchange labe and color order j, n,
        plt.text(DF[keyx][i][j],np.mean(DF[keyy][i][j][a:b]),'.'+DF["BCD1"][i]+DF["BCD2"][i]+DF["chopping"][i]+'____'+str(int(DF["BLm"][i][j])),fontsize=6)

#[np.square(2*jv(1,pi*diamrad*BLum[0]/DF["WL"][i])/(pi*diamrad*BLum[0]/DF["WL"][i]))
      for bl in range(0,125):  #for 3.7um
      plt.plot(bl,np.square(2*jv(1,pi*DF["diamrad"][0]*bl*1e6/3.6)/(pi*DF["diamrad"][0]*bl*1e6/3.6)),'o', color='black')
      
title= str(keyy) + " vs " + str(keyx) + str(object) + str(date) + " Files:" + str(DataType) + ".pdf"
plt.title(title)
plt.xlabel(keyx)
plt.ylabel(keyy)
#plt.ylim((0.0,0.3)) #0.3
plt.grid(linestyle="--",linewidth=0.5,color='.25')
plt.legend(prop={'size': 4})
plt.savefig(title)
plt.show()
      
      
      
      
      
      
      
      
      
      
      
      
      
      

#******************   TF    ***********************
#Plot average modeled visibilities and TF between two wavelengths vs VARIOUS
a=20   #index 30 to 3.617436, 20 to 3.8267136
b=41   #From  3.396094
jump=math.floor((b-a)/len(DF["name"]))
keyy="tf"      #calculates the average
keyx="FluxK"  #  FluxK  diamL, seeinge, T0S, windspeed, airmass/fluxL diamK/
for i in range(len(DF["name"])):
    #c = [float(i)/float(len(DF["name"])), 0.0, float(len(DF["name"])-i)/float(len(DF["name"]))] #R,G,B  red to blue
    c = cmap(float(i)/len(DF["name"]))  #red to blue
    plt.plot(DF[keyx][i],np.mean(DF[keyy][i][BL][a:b]),'o',color=c, label=DF["name"][i])
    plt.text(DF[keyx][i],np.mean(DF[keyy][i][BL][a:b]),'.'+DF["name"][i],fontsize=6)

title= str(keyy) + " vs " + str(keyx) + str(object) + str(date) + " BL:" + str(BL) + " Files:" + str(DataType) + "2.pdf"
plt.title(title)
plt.xlabel(keyx)
plt.ylabel(keyy)
plt.ylim((-0.1,1))
plt.grid(linestyle="--",linewidth=0.5,color='.25')
plt.legend(prop={'size': 7})
plt.savefig(title)
plt.show()
#****************  Scatter with color coding ********
#Plot averaged visibilities and TF between two wavelengths vs VARIOUS
a=20   #index 30 to 3.617436,    20 to 3.8267136
b=41   #From  3.396094
keyy="VIS2"      #calculates the average   TF, modelVIS2, VIS2
keyx="FluxK"  #  fluxK      diamL, seeing/fluxL diamK/
keycolor="seeinge"    #seeinge, T0S, mjd, airmass, windspeed
ind=[14,15,18,19,20,21,22,23]
x=np.zeros(8)   #-6 for Jul11th
y=np.zeros(8)
z=np.zeros(8)
l=[]
s=np.zeros(8)
j=0
for i in ind: #range(len(DF["name"])-6): #len(DF["name"])):  #-6 for July 11th
    x[j]=DF[keyx][i] #+6 for July 11th
    y[j]=np.mean(DF[keyy][i][BL][a:b])
    z[j]=DF[keycolor][i]
    l.append(DF["name"][i])
    s=DF["diamL"][i]
    j=j+1

#fig, ax = plt.subplots()
#ax.scatter(x,y,c=z)
#cbar = plt.colorbar()  #doesn't work
#cbar.set_label(keycolor,labelpad=-1)
#for i, txt in enumerate(l):
#    ax.annotate(txt, (x[i], y[i]))

plt.scatter(x,y,c=z,s=(s*10)**3, label=l)  #between.1 and 10:(s*10)**3
cbar = plt.colorbar()
cbar.set_label(keycolor,labelpad=-1)
title= "SP " + str(keyy) + " vs " + str(keyx) + str(object) + str(date) + " BL:" + str(BL) + " Files:" + str(DataType) + ".pdf"
plt.title(title)
plt.xlabel(keyx)
plt.ylabel(keyy)
plt.xlim(0,30)  #To lim flux
#plt.ylim((-0.1,0.6))
plt.grid(linestyle="--",linewidth=0.5,color='.25')
#plt.legend(prop={'size': 7})
for i, txt in enumerate(l):
    plt.annotate(txt, (x[i], y[i]), size=9)

#plt.savefig(title)
plt.show()

#Plot conditions   Other options: seeinge vs windspeed
jump=math.floor((b-a)/len(DF["name"]))
keyy="windspeed"  # T0S, airmass, seeinge, windspeed,
keyx="mjd"  # mjd,
for i in range(len(DF["name"])):
    #c = [float(i)/float(len(DF["name"])), 0.0, float(len(DF["name"])-i)/float(len(DF["name"]))] #R,G,B  red to blue
    c = cmap(float(i)/len(DF["name"]))  #red to blue
    plt.plot(DF[keyx][i],DF[keyy][i],'o',color=c,label=DF["name"][i])
    plt.text(DF[keyx][i],DF[keyy][i],'.'+DF["name"][i],fontsize=6)

title= str(keyy) + " vs " + str(keyx) + str(object) + str(date) + " BL:" + str(BL) + " Files:" + str(DataType) + "2.pdf"
plt.title(title)
plt.xlabel(keyx)
plt.ylabel(keyy)
plt.grid(linestyle="--",linewidth=0.5,color='.25')
plt.legend(prop={'size': 7})
plt.savefig(title)
plt.show()



#FluxK lowest: 0,1,7,9
DF["name"][1]

from astropy.io import fits
hdu=fits.open("/Volumes/LaCie/Reduced/Iter117NovNGC1068Lband/mat_raw_estimates.2018-07-14T10_23_24.HAWAII-2RG.rb/RAW_VIS2_0001.fits")
#EXACTLY the same as TARGET_RAW_INT_0001!!
hdu=fits.open("/Volumes/LaCie/Reduced/Iter126SeptNGC1068Only/mat_raw_estimates.2018-07-13T09_45_36.HAWAII-2RG.rb/TARGET_CAL_0001.fits") #???
hdu=fits.open("/Volumes/LaCie/Reduced/Iter126SeptNGC1068Only/mat_raw_estimates.2018-07-13T09_45_36.HAWAII-2RG.rb/TARGET_RAW_INT_0001.fits")
hduF=fits.open("/Volumes/LaCie/Reduced/NGC1068oifits/TARGET_CAL_INT_0001.fits")
hdu.info()
BL=4   #3,5doubles
a=20   #index 30 to 3.617436,    20 to 3.8267136
b=41 #len(hdu[3].data["EFF_WAVE"]) #41   #From  3.396094
plt.plot((hdu[3].data["EFF_WAVE"][a:b])*1e6,hdu[4].data["VIS2DATA"][BL][a:b],label="NonCal")
plt.ylim((-0.1,.2))
plt.show()
hdu[0].header["HIERARCH ESO ISS AMBI FWHM END"]  0.56
hdu[0].header["HIERARCH ESO ISS AMBI TAU0 END"]  0.005739
hdu[0].header["HIERARCH ESO ISS AMBI WINDSP"]

plt.plot((hduF[3].data["EFF_WAVE"][a:b])*1e6,hduF[4].data["VIS2DATA"][BL][a:b],label="Calibrated")
plt.legend()
plt.show()

plt.plot((hduF[3].data["EFF_WAVE"][a:b])*1e6,hduF[7].data["FLUXDATA"][BL][a:b],label="Calibrated")
plt.legend()
plt.show()
hdu.close()























from numpy import pi, cos, sin
import math
from scipy.special import jv
#Make table with Diameters  Uses data frame DF (no need of external disk)
#TABLE with Measured vis averaged over 3.4 to 3.6 and the Bessel function and diameters
#Some data from the fits file of the catalogue
from astropy.io import fits
hdudi=fits.open("/opt/local/share/esopipes/datastatic/matisse-1.1.5/jsdc_2017_03_03.fits")
hdudi[1].data.shape   #(465877,)  Good :)
#Working with a pandas dataframe
caldf = pd.DataFrame(hdudi[1].data)
diamrad=[]
modelvis2=[]
SpatFreq=[]
FluxL=[0,0,0,0,0,0,0,0,0,0]
FluxK=[0,0,0,0,0,0,0,0,0,0]
DFx=pd.DataFrame(columns=["i", "name1", "name2", "diam", "AvgVIS2", "ModelVIS2", "pBLe", "AvgWL", "Kmag", "W1mag", "W2mag", "W3mag", "FluxL", "FluxK", "SpecType"])
import pandas as pd
import numpy as np
DF = pd.read_pickle('calJuly13')
a=30   #30-41
b=41   #array([3.617436 , 3.5958447, 3.5741327, 3.5523   , 3.5303469, 3.5082731,3.4860787, 3.4637635, 3.4413276, 3.4187713, 3.396094 ],dtype=float32)
for i in range(len(df['i'])):
    diamrad.append(math.radians(caldf["UDDL"][index[i]])/3.6e6) #in mas -> rad
    #avgwl=np.mean(DF["WL"][i])  #First index for a specific calibrator, second for wl
    #avgvis2=np.mean(DF["VIS2"][i][0])  #First index for a specific calibrator, second for BL, third for wl
    BLum=df["pBLe"][i]*1e6  # m -> um, index for the cal
    SpatFreq.append(BLum/DF["WL"][i])
    modelvis2.append(np.mean(np.square(2*jv(1,pi*diamrad[i]*BLum/DF["WL"][i])/(pi*diamrad[i]*BLum/DF["WL"][i])))) #??had square square
    print(caldf["NAME"][index[i]],caldf["KMAG"][index[i]]) #ALl with NMAG from????
    tablex.loc[i]=[i,DF["name"][i],caldf["NAME"][index[i]],caldf["UDDL"][index[i]],avgvis2,modelvis2[i],DF["pBLe"][i],avgwl,caldf["KMAG"][index[i]],    caldf["LMAG"][index[i]], caldf["MMAG"][index[i]], caldf["NMAG"][index[i]], FluxL[i], FluxK[i], caldf["SPTYPE"][index[i]]]
    DFx.loc[i]=[i,DF["name"][i],caldf["NAME"][index[i]],caldf["UDDL"][index[i]],avgvis2,modelvis2[i],DF["pBLe"][i],avgwl,caldf["KMAG"][index[i]],    caldf["LMAG"][index[i]], caldf["MMAG"][index[i]], caldf["NMAG"][index[i]], FluxL[i], FluxK[i], caldf["SPTYPE"][index[i]]]

DFx.to_csv('calJuly13X.csv')     #no vis2 data
DFx.to_pickle('calJuly13X')
#5Nov
import pandas as pd
dfx = pd.read_pickle('calJuly13X')  #To read back
#https://www.gemini.edu/cgi-bin/sciops/instruments/michelle/magnitudes.pl?magnitude=1.3&wavelength=2.2&filter=2MASS+Ks&option=magnitude
dfx["Kmag"]   #FkuxK
#um   https://old.ipac.caltech.edu/2mass/releases/allsky/faq.html#jansky
0    1.292 AB:  Fnu=202.829Jy,z=666.7Jy For the (effective) wl 2.159 microns:
1    1.446      Fnu=176Jy, z="
2   -0.835      1438
3   -0.399      962
4   -0.257      844
5    0.037      644
6   -0.274      858
7    0.801      318
8   -0.344      915
9    1.577      156
#http://wise2.ipac.caltech.edu/docs/release/prelim/expsup/sec4_3g.html#PhotometricZP
#Monochromatic AB magnitudes are defined by -2.5log[Fnu(Jy)]+8.926 (see e.g., Tokunaga & Vaca 2005). Hence, conversion from the WISE Vega-system magnitudes to flux density using Fnu0  (column 2 of Table 1) and to AB magnitudes follows: mAB=mVega+Deltam   BUT...
#Deltam = W1    2.699,W2    3.339,W3    5.174,W4    6.620
#http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4h.html#example
#Vega Magnitudes to Flux Density  The source flux density, in Jansky [Jy] units, is computed from the calibrated WISE magnitudes, mvega using: Fnu[Jy]=Fnu0x10^(-mVega/2.5), where Fnu0 is the zero magnitude flux density corresponding to the constant that gives the same response as that of Alpha Lyrae (Vega). For sources with steeply rising MIR spectra or with spectra that deviate significantly from Fν=constant (Fnu prop nu^-alpha), including cool asteroids and dusty star-forming galaxies, a color correction is required, especially for W3 due to its wide bandpass. With a given flux correction, fc, the flux density conversion is given by: Fnu[Jy]=(Fnu0/Fc)x10^(-mVega/2.5) where F*nu0 is the zero magnitude flux density derived for sources with power-law spectra: Fnu prop nu^-2
#http://wise2.ipac.caltech.edu/docs/release/allwise/faq.html AllWISE Catalog and Multiepoch Photometry Database magnitudes are given in the Vega system and represent total in-band brightness measurements. Conversion of AllWISE in-band magnitudes to monochromatic values such as flux density or AB magnitude, along with some helpful examples, are described in section IV.4.h.i of the WISE All-Sky Data Release Explanatory Supplement. The broad WISE bandpasses may require applying significant color corrections when deriving monochromatic brightnesses depending on the spectral energy distribution of the particular object. The nominal system flux zero points are defined for a fν∝ν-2 spectrum through the WISE bandpasses. Color corrections for other spectral slopes are given in section IV.4.h.vi of the All-Sky Release Supplement.  (Flux Corrections and Colors for Power Laws and Blackbodies)
dfx["W1mag"]  #W1!  NaN
AB=dfx["Kmag"]+2.699  #ABmag
fcW1nu1=0.9961
fcW1nu0=0.9907
fcW1nu2=1.0084
fcW1nu3=1.0283
cats=pd.read_csv("JSDCAllWISE.csv")  #crossmatch to 5 arcsec
cats.columns #JSDC Johnson mags, AllWISE Vega mags
#Revisar: cats[""]
cats["angDist"][0] #less than 2 arcsec
cats["qph"][0]     #AA
for i in range(11):
    #ind=cats[cats["Name"].str.contains(NamesJSDC[i])]
    ind=cats[cats["Name"].str.contains(dfx["name1"][i])]
    print(ind["angDist"])
    #print(ind["Name"],ind["W1mag"]-ind["W2mag"],ind["W2mag"]-ind["W3mag"])
    MonochrFluxL=(306.682/fcW1nu3)*10**(-ind["W1mag"]/2.5)
    dfx["W1mag"][i]=ind["W1mag"]
    dfx["W2mag"][i]=ind["W2mag"]
    dfx["W3mag"][i]=ind["W3mag"]
    #Filling the values in DFx
    dfx["FluxL"][i]=MonochrFluxL  #Fill by hand to change the fcW1nu3
    FluxKJyAB = (10**23.0)*10**(-(dfx["Kmag"][i] + 48.6)/ 2.5)   #Flux AB
    FluxVega = 666.7*10**(-0.4*dfx["Kmag"][i])                  #Flux Vega
    #dfx["FluxK"][i]=FluxVega


FluxVega = 666.7*10**(-0.4*3.04)
#Fill by hand i=6,8,9
FluxAB = (10**23.0)*10**(-(AB[9] + 48.6)/ 2.5)  #NO
#Equiv to 3631*10**(-0.4*AB[9])
      
FluxVega = 666.7*10**(-0.4*dfx["Kmag"][7])  #YES
dfx["name1"][6]   #'Tet Cen' =NamesJSDC[6]
dfx["name1"][8]   #'nu Tuc'
dfx["name1"][9]   #'HD189831'
#Record used fcnu?: 2, 3, 2, 3, 3, 3, 3, NaN,  3, NaN, NaN,
print(dfx["W1mag"],dfx["Kmag"],dfx["FluxL"],dfx["FluxK"])
#Saving the new added data to the files
dfx.to_csv('calJuly13X.csv')
dfx.to_pickle('calJuly13X')
dfx["Flux"][0]
#In the ABν system, the reference spectrum is simply a constant line in fν
#in AB magnitudes, mag 0 has a flux of 3720 Jy , 1 Jansky = 10^-26 W Hz^-1 m^-2
#To convert AB magnitude to monochromatic flux in Jansky:F/Jy=10^(23-(AB+48.6)/2.5)
#Convert to flux https://www.ukirt.hawaii.edu/astronomy/calib/phot_cal/conver.html
F[Jy]=10^(23-(ABmag+48.6)/2.5)
m=-2.5log(F*/FVega)


#array with indexes! from the JSDC catalogue where the 10 calibrators from July 13th are
ind=[3.82843e+05, 4.48554e+05, 3.86682e+05, 4.41000e+02, 4.09768e+05, 5.88500e+03, 2.52490e+05, 3.91902e+05, 4.44972e+05, 3.87264e+05]

#http://vizier.u-strasbg.fr/viz-bin/VizieR  search:jsdc with 5 arcsec.Use UDDL [mas]
#Kmag=1.292 Nmag=1.362  #* b Sgr

DFx.to_csv('calJuly13X.csv')
DFx.to_pickle('calJuly13X')     #To save changes in M51   to plot           #Change
import pandas as pd
dfx = pd.read_pickle('calJuly13X')  #To read back

#Plot from the DF created
"""DF=pd.DataFrame(columns=["i","name","pBLe", "airmass", "mjd", "dateobs",mode", "SI1","SI2","SI3","SI4","object","categ","type","tplstart","T0E","T0S","windspeed","chopping","BCD1","BCD2","resolution","wlrange","wl0","DIT","fringetracker","readoutMd","readout","filter","WL","VIS2"])"""
#Arrange by MJD
DFmjd=DF.sort_values(by=['mjd'])
#Plot      ****
fig = plt.figure()
ax  = fig.add_subplot(2,1,1)
t = DFmjd['WL'][0] #  array of x-values  [i]
s = DFmjd['VIS2'][0][0] # array of y-values  [i][BL]
line, =ax.plot(t,s) # add plot to figure (not yet visible)
fig.show() # plot is now visible but python is still open for new commands
s1 = DFmjd['VIS2'][1][0] # new y-array
line, =ax.plot(t,s1) # add new line to existing plot; change is not yet visible
fig.show() # plot including both lines is now visible

#Plot
num_plots = len(fileList)
#from cycler import cycler
#colors = plt.get_cmap('Spectral')(np.linspace(0, 1, num_plots))
#ax.set_prop_cycle(cycler('color', colors))
wl = DFmjd['WL'][0]
for i in range(10):                                         #Change
    if len(wl)==len(DFmjd['VIS2'][i][0]):    #64
        plt.plot(wl,DFmjd['VIS2'][i][0],label=DFmjd['name'][i]) #BL changes the BL (same order for all files??)
        plt.text(wl[i*5],DFmjd['VIS2'][i][0][i*5],DFmjd['name'][i])

plt.grid(linestyle="--",linewidth=0.5,color='.25')
plt.legend(prop={'size': 7})
#Add:   title=   BL
plt.title('VIS2 calibrators July 13th')
plt.xlabel('wavelength [um]')
plt.ylabel('VIS2')
plt.ylim(-0.1,1.1)
#plt.savefig('.png')
plt.show()

#TF  using the saved data frame
import pandas as pd
df = pd.read_pickle('calJuly13')
df.shape  #(10, 35)
df['name']
df.query('name.str.contains("Sgr")')  #B Sgr, c SGr
df.query('chopping.str.contains("F")')
df['RA']
"""0       B Sgr
1       g Aqr
2       c Sgr
3      30 Psc
4       k Aqr
5     eta Scl
6     Tet Cen
7    V584 Aql
8      nu Tuc
9    HD189831"""
"""
RA      =           299.235587 / [deg] 19:56:56.5 RA (J2000) pointing
DEC     =            -27.16995 / [deg] -27:10:11.8 DEC (J2000) pointing
"""
# NOOO! hdudi=fits.open("/Users/M51/SOFT/INTROOT/matis/matisp/matis-1.1.3/data/jsdc_2017_03_03.fits")
from astropy.io import fits
hdudi=fits.open("/opt/local/share/esopipes/datastatic/matisse-1.1.5/jsdc_2017_03_03.fits")
hdudi[1].data.shape   #(465877,)  Good :)
hdudi[0].header
"""HIERARCH ESO PRO CATG = 'JSDC_CAT'
HIERARCH ESO PRO TECH = 'CATALOG '
HIERARCH ESO PRO TYPE = 'JMMC    '
DATE    = '2017-05-31T14:39:30' / file creation date (YYYY-MM-DDThh:mm:ss UT) """
hdudi[1].data.columns
hdudi[1].data["Name"].size  # 109 / 465877
ra=hdudi[1].data["RAJ2000"]
de=hdudi[1].data["DEJ2000"]
dL=hdudi[1].data["UDDL"]   #Uniform Diameter
mL=hdudi[1].data["LMAG"]
#Working with a pandas dataframe
caldf = pd.DataFrame(hdudi[1].data)
caldf.columns
#(['NAME', 'RAJ2000', 'DEJ2000', 'SPTYPE', 'BMAG', 'E_BMAG', 'VMAG','E_VMAG', 'RMAG', 'IMAG', 'JMAG', 'E_JMAG', 'HMAG', 'E_HMAG', 'KMAG','E_KMAG', 'LMAG', 'E_LMAG', 'MMAG', 'E_MMAG', 'NMAG', 'E_NMAG', 'LDD','E_LDD', 'LDD_CHI2', 'CALFLAG', 'UDDB', 'UDDV', 'UDDR', 'UDDI', 'UDDJ','UDDH', 'UDDK', 'UDDL', 'UDDM', 'UDDN'],dtype='object')
from astropy import units as u
from astropy.coordinates import SkyCoord
c = SkyCoord(ra=299.235587*u.degree, dec=-27.16995*u.degree, frame='icrs')
c.ra.hms  #hms_tuple(h=19.0, m=56.0, s=56.540880000010816)
c.dec.dms #dms_tuple(d=-27.0, m=-10.0, s=-11.820000000000164)
c.ra.degree
c.dec.degree
dummy=caldf['RAJ2000'][0] + caldf['DEJ2000'][0] #'00 00 00.095  -05 29 39.66  '
test = SkyCoord(dummy, unit=(u.hourangle, u.deg))
test #<SkyCoord (ICRS): (ra, dec) in deg   (0.00039583, -5.49435)>
test.ra.degree
test.dec.degree
c.separation(test)   #<Angle 62.49220327 deg>
from astropy import units as u
from astropy.coordinates import SkyCoord
import time
import numpy as np
start = time.clock()
index=np.zeros(10)
(len(df['name']))  #10
for i in range(4,10):       # 1 to 9
    temp=1000.00
    temp1 = df['RA'][i]
    temp2 = df['DEC'][i]
    C = SkyCoord(ra=temp1*u.degree, dec=temp2*u.degree, frame='icrs')
    for j in range(len(caldf['NAME'])):   #465877
        temp3 = caldf['RAJ2000'][j] + caldf['DEJ2000'][j]
        S = SkyCoord(temp3, unit=(u.hourangle, u.deg))
        sep = C.separation(S)
        sepd = sep.degree
        #print(sepd)
        #print(temp)
        if sepd<temp:
            print(j)
            #print(temp)
            index[i]=j
            temp = sepd


print(time.clock() - start) #255 secs for 100,000       1219 for i=0
#i=382843     sep=0.4090961146179403
index
index[0].astype(int)
#Subtable with crossmatched data
caldf.loc[caldf[382843]]
caldf['NAME'][382843]    #* b Sgr
caldf['LDD'][382843]     #2.9674914
caldf['E_LDD'][382843]   #0.2995214
caldf['LDD_CHI2']        #0.00082179246  Limb Darkened diameter Not to use
caldf['CALFLAG'][382843] #0
caldf['UDDL'][382843]    #2.9241455  Uniform Disk Diameter Use this!!! 0.205674
caldf['LMAG'][382843]    #nan

import pandas as pd
df = pd.read_pickle('calJuly13')
from numpy import pi, cos, sin
import math
from scipy.special import jv  #Bessel function of the first kind of real order and complex argument.
#http://vizier.u-strasbg.fr/viz-bin/VizieR  search:jsdc with 5 arcsec.Use UDDL [mas]
#Kmag=1.292 Nmag=1.362  #* b Sgr
BL=df['pBLe'][0]*1e6  # m -> um
DiamCalStar=math.radians(2.9242455 )/3.6e6 #1.4386804236327141e-08 diam of calstar mas -> rad
wl=df["WL"][0]  #in um
SpatFreq = BL / wl
#Vis(B) = Besinc(πα0 B/λ)
pi*DiamCalStar*SpatFreq
test=np.square(2*jv(1,pi*DiamCalStar*SpatFreq)/(pi*DiamCalStar*SpatFreq)) #3: 0.97 uniformdisk
VisAmp = np.abs(test)
VisSq = np.square(VisAmp)
plt.plot(wl,VisSq)
plt.plot(df["WL"][0],df["VIS2"][0][0])
plt.show()

max(df["VIS2"][0][0])   #0.9308883486738551
max(VisSq)    #0.9647966

#modear vis con diametro dado para cada BL??   could be, but better change the value of the diameter of the calibrator, seeing and noise??
#https://www.eso.org/sci/facilities/paranal/telescopes/vlti/tuto/tutorial_interferometry.html
x = df['WL'][0][1:62] #  array of x-values  [i]  larger index to the left of the plot
y = df['VIS2'][0][0][1:62]
plt.plot(x,y,label=df['name'][0])
DiamCalStar=math.radians(caldf['LDD'][382843])/3.6e6  #diam of the calstar in mas -> rad
y2=np.square(np.square(2*jv(1,pi*DiamCalStar*SpatFreq)/(pi*DiamCalStar*SpatFreq)))
plt.plot(x,y2[1:62],label=DiamCalStar)
DiamCalStar=1.4e-2
y3=np.square(np.square(2*jv(1,pi*DiamCalStar*SpatFreq)/(pi*DiamCalStar*SpatFreq)))
plt.plot(x,y3[1:62],label=DiamCalStar)
DiamCalStar=0.3e-1
y4=np.square(np.square(2*jv(1,pi*DiamCalStar*SpatFreq)/(pi*DiamCalStar*SpatFreq)))
plt.plot(x,y4[1:62],label=DiamCalStar)
DiamCalStar=0.4e-1
y5=np.square(np.square(2*jv(1,pi*DiamCalStar*SpatFreq)/(pi*DiamCalStar*SpatFreq)))
plt.plot(x,y5[1:62],label=DiamCalStar)
DiamCalStar=0.2e-1
y6=np.square(np.square(2*jv(1,pi*DiamCalStar*SpatFreq)/(pi*DiamCalStar*SpatFreq)))
plt.plot(x,y6[1:62],label=DiamCalStar)
#DiamCalStar=0.8e-3
#y7=np.square(np.square(2*jv(1,pi*DiamCalStar*SpatFreq)/(pi*DiamCalStar*SpatFreq)))
#plt.plot(x,y7[1:62],label=DiamCalStar)
plt.legend()
plt.show()

#Plot      ****
fig = plt.figure()
ax  = fig.add_subplot(2,1,1)
x = df['WL'][0][1:62] #  array of x-values  [i]  larger index to the left of the plot
y = df['VIS2'][0][0][1:62] # array of y-values  [i][BL]
line, =ax.plot(x,y) # add plot to figure (not yet visible)
fig.show() # plot is now visible but python is still open for new commands
y2 = VisSq[1:62] # new y-array
line, =ax.plot(x,y2) # add new line to existing plot; change is not yet visible
fig.show() # plot including both lines is now visible
DiamCalStar=1.1e-2
y3=np.square(np.square(2*jv(1,pi*DiamCalStar*SpatFreq)/(pi*DiamCalStar*SpatFreq)))#newy-array
line, =ax.plot(x,y3[1:62])
fig.show()
DiamCalStar=1.1e-1
y3=np.square(np.square(2*jv(1,pi*DiamCalStar*SpatFreq)/(pi*DiamCalStar*SpatFreq)))#newy-array
line, =ax.plot(x,y3[1:62])
fig.show()

#To handle time (MJD) in seconds:
from astropy.time import Time
times = mjd[0:228]  #array([58312.38049544, 58312.38049815])
t = Time(times, format='mjd')  #scale is utc??
t  #<Time object: scale='utc' format='mjd' value=[58312.38049544 ...]>
t.isot  #2018-07-13T09:07:54.806 ... '2018-07-13T09:08:53.800'
t.unix   # seconds since 1970.0 (UTC) array([1.53147287e+09 ...])
d0h = hdu[0].header
d0h['MJD-OBS']     #58312.38023811
#This represents the number of days since midnight on November 17, 1858. For example, 51544.0 in MJD is midnight on January 1, 2000.
d0h['DATE-OBS'] #2018-07-13T09:07:32.5723
dateobs=Time(d0h['DATE-OBS'],format='isot', scale='utc')
starttime = dateobs.mjd  #58312.380238105325
days=(t-starttime)  #<Time object: scale='utc' format='mjd' value=[0.00068558 0.00068828]>
#WARNING: ErfaWarning: ERFA function "taiutc" yielded 228 of "dubious year (Note 4)" [astropy._erfa.core]
seconds=(days.value)*86400
ms=seconds*1000


#To plot wind speed with time
Nmint=Time(Nmjd[0],format='mjd', scale='utc')#<Time object: scale='utc' format='mjd' value=58312.38049543986>
Nmint.isot#'2018-07-13T09:07:54.806'
print(Ndateobs,Nmint.isot, Nd0h['HIERARCH ESO ISS AMBI WINDSP']) #2.98, 2.83, 3.03, 2.8
#1 - 2018-07-13T09:07:32.572 2018-07-13T09:07:54.806 2.98
#2 - 2018-07-13T09:09:14.339 2018-07-13T09:06:57.270 2.83
#3 - 2018-07-13T09:10:34.281 2018-07-13T09:05:36.774 3.03
#4 - 2018-07-13T09:11:54.406 2018-07-13T09:06:57.166 2.8
In order:
#Plot wind vel vs time during the night
NDatesObs=Time(['2018-07-13T09:05:36.774','2018-07-13T09:06:57.270','2018-07-13T09:06:57.166','2018-07-13T09:07:54.806'],format='isot', scale='utc')
NWindSpeed=[3.03,2.83,2.8,2.98]
#FALTAOrdenar automaticamente en t!!!
plt.plot(NDatesObs.value,NWindSpeed,label="Wind speed (m/s)")
plt.grid(linestyle="--",linewidth=0.5,color='.25')
plt.legend()
plt.xticks(rotation=45)
plt.title('Wind Speed against Time N band')
plt.show()
#*******************************************************************
#8th October   To plot the transfer function of one calibrator in the L band vs wl and 3umTF vs 6BLs
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.time import Time
idxFile = 1  #file 1-4 only give BCD IN/OUT choices. Enough to look at nr.1
FileName='/Volumes/LaCie/Reduced/Iter125SeptAllJuly13thLMbands/mat_raw_estimates.2018-07-13T08_34_14.HAWAII-2RG.rb/RAW_TF2_000'+str(idxFile)+'.fits'
LhduRTF = fits.open(FileName)
h=LhduRTF[0].header #h.values

h["UTC"]       #             31188.25 / [s] 08:39:48.250 UTC
calibrator = h["HIERARCH ESO OBS TARG NAME"] #'30 Psc'
airmass = np.mean([h["HIERARCH ESO ISS AIRM END"],h["HIERARCH ESO ISS AIRM START"]]) #
#HIERARCH ESO ISS AIRM END = 1.07 / Airmass at end.
#HIERARCH ESO ISS AIRM START = 1.071 / Airmass at start.
diameter = h["HIERARCH ESO PRO JSDC DIAMETER"]   #7.85043239593506
"""
HIERARCH ESO PRO JSDC DIAMETER = 7.85043239593506
HIERARCH ESO PRO JSDC DIAMETER ERROR = 0.587996244430542
HIERARCH ESO PRO JSDC LMAG = 1000000.
HIERARCH ESO PRO JSDC MMAG = 1000000.
HIERARCH ESO PRO JSDC NMAG = -0.40200001001358
HIERARCH ESO PRO JSDC NAME = '*  30 Psc '
...
"""
d1=LhduRTF[1].data
d1["TEL_NAME"]   #chararray(['AT1', 'AT2', 'AT3', 'AT4'], dtype='<U3')
d1["STA_NAME"]   #chararray(['A0', 'B2', 'D0', 'C1'], dtype='<U2')
d1["STA_INDEX"]   #array([ 1,  5, 13, 10], dtype=int16)
d2=LhduRTF[2].data
LwlRTF=d2["EFF_WAVE"]  #d2.shape   (64,)
val=3e-06
idx = (np.abs(LwlRTF-val)).argmin()   #57   LwlRTF[57]   =2.992126e-06
d3=LhduRTF[3].data
TF2=d3["TF2"]  #d3["TF2"].shape   (6, 64)
d3["STA_INDEX"]     #d3["STA_INDEX"].shape  (6, 2)
"""array([[13, 10],
       [ 1,  5],
       [ 5, 13],
       [ 5, 10],
       [ 1, 13],
       [ 1, 10]], dtype=int16)"""
BL = [h["HIERARCH ESO ISS PBL34 START"],h["HIERARCH ESO ISS PBL12 START"],h["HIERARCH ESO ISS PBL23 START"],h["HIERARCH ESO ISS PBL24 START"],h["HIERARCH ESO ISS PBL13 START"],h["HIERARCH ESO ISS PBL14 START"]]
d3["MJD"]  #array([58312.36134778, 58312.36134778, 58312.36134778, 58312.36134778, 58312.36134778, 58312.36134778])
time=Time(d3["MJD"][0], format='mjd',scale='utc')#<Time object: scale='utc' format='mjd' value=58312.38049543986>
str(time.isot)
d3["INT_TIME"]   #array([0.075, 0.075, 0.075, 0.075, 0.075, 0.075])
#The L band is specified from 3.2 to 3.9 μm and the N band from 8.0 to 13.0 μm. MATISSE will operate also in M band, from 4.5 to 5.0 μm.
LhduRTF.close()
for i in range(6):
    plt.plot(LwlRTF*1e+6,TF2[i],label="TF")
    title=calibrator+' BL:'+str(BL[i])+'m. Diam:'+str(diameter)+' Airm:'+str(airmass)
    plt.title(title)
    plt.grid(linestyle="--",linewidth=0.5,color='.2')
    plt.plot(LwlRTF[idx]*1e+6,TF2[i][idx],'ro', label="TF at 3um")
    plt.xlabel('wavelength')
    plt.ylabel('TF')
    plt.figtext(0.2, 0.2, "T:"+str(time.isot))
    plt.show()

TF3um=[TF2[0][idx],TF2[1][idx],TF2[2][idx],TF2[3][idx],TF2[4][idx],TF2[5][idx]]
plt.plot(BL,TF3um,'bo', label="TF at different BLs")
title=calibrator+' Diam:'+str(diameter)+' Airm:'+str(airmass)+' at 3um.'
plt.title(title)
plt.grid(linestyle="--",linewidth=0.5,color='.2')
plt.xlabel('BL in m')
plt.ylabel('TF')
plt.figtext(0.2, 0.2, "T:"+str(time.isot))
plt.show()

**********************************************************************
# 10 Oct  Working with L band
#To open files containing list of TPLs with find...   DONE
#To plot for all calibrators in DPHASE the VIS   DONE
#To calculate fluxes of the calibrators
#To play with the diameters of the calibrators

#First replace the : in the time for _/Users/M51/Downloads/July13calibrators-2.txt
#Change the name of the file according to the one generated by the search for parameters
FileTplStart = ("/Users/M51/Downloads/July13calibrators-2.txt")  #Change
inDIR = '/Volumes/LaCie/Reduced/Iter125SeptAllJuly13thLMbands/'  #Change
patternData='*RAW_DPHASE_0001.fits'                              #Change
BL = 0               #1 to 6                                     #Change
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.time import Time
import os, fnmatch
import pandas as pd
import statistics as stat

with open(FileTplStart) as f:
    firstlistTplStart=f.read().splitlines()

firstlistTplStart[0]      # '2018-07-13T05_41_36'
len(firstlistTplStart)  #273
listTplStart=pd.unique(firstlistTplStart).tolist()
len(listTplStart)   #15
directoryList = []
fileList=[]
#To find all calibrators
for i in range(len(listTplStart)):
    patternTplStart = '*'+str(listTplStart[i])+'*'
    for fList in os.walk(inDIR):
        for DirName in fList[1]:
            if fnmatch.fnmatch(DirName, patternTplStart): # Match search string
                directoryList.append(os.path.join(fList[0],DirName))

len(directoryList)    #15
#To compare only with one set up of the BCDs (is the same??)
for i in range(len(directoryList)):
    for fList in os.walk(directoryList[i]):
        for FileName in fList[2]:
            if fnmatch.fnmatch(FileName, patternData): # Match search string
                fileList.append(os.path.join(fList[0],FileName))

len(fileList)  #15
#To plot all the calibrators
#To get the same x axis for all the plots
hducal=fits.open(str(fileList[0]))
d3cal=hducal[3].data
wl = d3cal["EFF_WAVE"]*1e6
len(wl)    #64
hducal.close()
Yes=[]
No=[]
Noisy=[]
DFall=pd.DataFrame(columns=["i","name","pble", "airmass", "mjd", "mode", "SI1","SI2","SI3","SI4"])
for i in range(len(fileList)):
    hducal=fits.open(str(fileList[i]))
    airmass=np.mean([hducal[0].header["HIERARCH ESO ISS AIRM END"],hducal[0].header["HIERARCH ESO ISS AIRM START"]])
    object=hducal[0].header["HIERARCH ESO OBS TARG NAME"]
    mode=hducal[0].header["HIERARCH ESO INS MODE"]
    mjd=hducal[0].header["MJD-OBS"]
    temp=hducal[2].data["STA_INDEX"]
    staind=(temp == [1,5,13,10])
    si1=staind[0]
    si2=staind[1]
    si3=staind[2]
    si4=staind[3]
    if BL==0:pble=hducal[0].header["HIERARCH ESO ISS PBL12 END"]
    #elif BL==1:bl=hducal[0].header["HIERARCH ESO ISS PBL13 END"]
    #elif BL==2:bl=hducal[0].header["HIERARCH ESO ISS PBL14 END"]
    #elif BL==3:bl=hducal[0].header["HIERARCH ESO ISS PBL23 END"]
    #elif BL==4:bl=hducal[0].header["HIERARCH ESO ISS PBL24 END"]
    #elif BL==5:bl=hducal[0].header["HIERARCH ESO ISS PBL34 END"]
    d4cal=hducal[4].data
    vacal=d4cal["VISAMP"]    #below: to append the ith row
    #DFall.loc[i] = [i,object,pble,airmass,mjd,mode,si1,si2,si3,si4]  #Comment to plot without Noisy data
    if len(wl)==len(vacal[BL]):    #64
        if i not in Noisy:
            plt.plot(wl,vacal[BL], label=i) #BL changes the BL (same order for all files??)
            #Yes.append(i)                                          #Comment to plot without Noisy data
            hducal.close()
            min = vacal[BL].min()
            #if min<(-1.5):                                   #Comment to plot without Noisy data   #Change
                #Noisy.append(i)                                     #Comment to plot without Noisy data
        else:
            print('Noisy data')
    else:
        #print(len(vacal[BL]))   #118
        No.append(i)      #Comment to plot without Noisy data
        hducal.close()

len(Yes)  #0
len(No)   #4
len(Noisy) #4   before:[21, 28, 124, 131, 135, 199, 201, 218]
#DFall
plt.grid(linestyle="--",linewidth=0.5,color='.25')
plt.legend(prop={'size': 5})
plt.title('VISAMP calibrators July 13th')
plt.show()

#Making subtables by specific value under a specific column
DFall.loc[DFall['mode'] != 'HYBRID'] #Empty DataFrame
DFall.loc[DFall['name'] == 'HD16062']
DFall.loc[DFall['SI4'] != True] #Empty DataFrame 1-4
np.min(DFall['airmass'])   #1.0034999999999998
np.max(DFall['airmass'])   #1.257
hyb=(DFall["mode"].all=='HYBRID')  #False
#DFall.to_pickle('DFallOct10')  #To save in M51                     Change
#import pandas as pd
#DFall = pd.read_pickle('DFallOct10')  To read back
array=[23,40,51,80,81,245]
DFall.ix[:,array]
DFall.loc[DFall['i']==23]
DFall.loc[DFall['i']==40]
DFall.loc[DFall['i']==51]
DFall.loc[DFall['i']==80]
DFall.loc[DFall['i']==81]
DFall.loc[DFall['i']==245]


#Localizing the noisy ones, to plot
#Do they belong to the same cal? around same time?
import statistics
for i in range(len(Noisy)):
    hducal=fits.open(str(fileList[Noisy[i]]))
    d4cal=hducal[4].data
    vacal=d4cal["VISAMP"]
    sd=statistics.stdev(vacal[BL])  #<0.16 is good
    min = vacal[BL].min()
    print(i)
    print(min)
    print(sd)
    plt.plot(wl,vacal[BL], label=Noisy[i])
    hducal.close()

plt.grid(linestyle="--",linewidth=0.5,color='.25')
plt.title('VISAMP Noisy calibrators July 13th')
plt.legend()
plt.show()
corrdf=pd.DataFrame(columns=Yes)  #[0 rows x 248 columns] with names the elements in Yes
corrdf['index']=Yes
corrdf.set_index('index')
#corrdf.loc[0,272] = 'DUMMY'  if the name of the col doesn existi it adds a new one
#Making a correlation table for the array plotted (Yes)
for i in range(len(Yes)):
    hducali=fits.open(str(fileList[Yes[i]]))
    d4cali=hducali[4].data
    vacali=d4cali["VISAMP"]
    for j in range(len(Yes)):
        hducalj=fits.open(str(fileList[Yes[j]]))
        d4calj=hducalj[4].data
        vacalj=d4calj["VISAMP"]
        corrij=np.corrcoef(vacali,vacalj)[0][1]
        corrdf.loc[Yes[i],Yes[j]] = corrij

corrdf.to_pickle('corrdfYesOct10')  #To save in M51                     Change
#import pandas as pd
#corrdf = pd.read_pickle('corrdfYesOct10')  To read back
corrdf[2]  #columns under name: 2
corrdf.iloc[0,2]
corrdf.values[:,>0.99]
corrdf.ix[:,[1,2]]  #columns with names of the array
corrdf.ix[:,Yes] #All
corrdf.ix[:,No]  #empty
DFall.ix[:,Yes]  #Subset

#  Working with files with different number of entries in wl
corrdf[23]
corrdf.ix[:,[23,29]]
corrdf.ix[:,No]
corrdf.ix[No,No]   #NaN are the noisy ones
No[0] #index of the first element in No: 23
hducal=fits.open(str(fileList[No[0]]))
d3cal=hducal[3].data
wlNo = d3cal["EFF_WAVE"]*1e6
d4cal=hducal[4].data
vacal=d4cal["VISAMP"]
hducal.close()
count=0
Noisy2=[]
for i in range(len(No)):
    hducal=fits.open(str(fileList[No[i]]))
    d4cal=hducal[4].data
    vacal=d4cal["VISAMP"]
    if len(wlNo)==len(vacal[BL]):
        min2=vacal[BL].min()
        if min2<(-4):           #Change #Comment to plot without Noisy data
            Noisy2.append(No[i])
        plt.plot(wlNo,vacal[BL], label=No[i]) #BL changes the BL (same order for all files??)
        hducal.close()
        count=count+1
    else:
        print(len(vacal[BL]))   #
        hducal.close()

len(Noisy2)  #10
print(count)  #25
plt.grid(linestyle="--",linewidth=0.5,color='.25')
plt.legend()
plt.title('VISAMP calibrators July 13th')
plt.show()
Noisy2   #[32, 65, 92, 94, 116, 125, 127, 141, 167, 198]

#By eye: there are 2 diff sets: like fileList[No[0]] and like fileList[No[1]]
hducal=fits.open(str(fileList[No[0]]))
d4cal=hducal[4].data
vacal0=d4cal["VISAMP"]
hducal.close()
hducal=fits.open(str(fileList[No[1]]))
d4cal=hducal[4].data
vacal1=d4cal["VISAMP"]
hducal.close()
df = pd.DataFrame(columns=['index','corr0','corr1'])
#Plot w/o Noisy2 data
j=0
for i in range(len(No)):
    hducal=fits.open(str(fileList[No[i]]))
    d4cal=hducal[4].data
    vacal=d4cal["VISAMP"]
    if len(wlNo)==len(vacal[BL]):    #118
        if No[i] not in Noisy2:  #to not to plot the very noisy ones
            corr0=(np.corrcoef(vacal0[BL],vacal[BL])[0][1])
            corr1=(np.corrcoef(vacal1[BL],vacal[BL])[0][1])
            df.loc[j] = [No[i],corr0,corr1]  #To append as row jth
            #plt.figtext(.8, .8, corr0)
            plt.grid(linestyle="--",linewidth=0.5,color='.25')
            plt.legend()
            plt.xlabel('EFF_WAVE in um')
            plt.ylabel('VISAMP')
            plt.title('VISAMP calibrators July 13th')
            if corr0 > 0.99:  #to divide by correlation
                plt.subplot(121)
                plt.plot(wlNo,vacal[BL],label=No[i]) #BL changes the BL (same order for all files??)
                hducal.close()
                j=j+1
            else:
                plt.subplot(122)
                plt.plot(wlNo,vacal[BL],label=No[i]) #BL changes the BL (same order for all files??)
                hducal.close()
                j=j+1
        else:
            print('Noisy data')
            hducal.close()
    else:
        print(len(vacal[BL]))   #
        hducal.close()

df #pandas data frame with correlations
plt.show() #here, to include both subplots

#Simple code to find files in steps and plot ...----------------------------->
patternTplStart = '*'+str(listTplStart[0])+'*'
directoryList = []
#fList saves 3 elements: directory given, subdirectories and files.
for fList in os.walk(inDIR):
    for DirName in fList[1]:
        if fnmatch.fnmatch(DirName, patternTplStart): # Match search string
            directoryList.append(os.path.join(fList[0],DirName))
directoryList  #it shoud find only one
#['/Volumes/LaCie/Reduced/Iter125SeptAllJuly13thLMbands/mat_raw_estimates.2018-07-13T05_41_36.HAWAII-2RG.rb']
#Now we search for a specific type of files inside the directory found
inDIR=directoryList[0]  #it shoud find only one
#'/Volumes/LaCie/Reduced/Iter125SeptAllJuly13thLMbands/mat_raw_estimates.2018-07-13T05_41_36.HAWAII-2RG.rb'
fileList=[]
#To compare only with one set up of the BCDs (is the same??)
patternData='*RAW_DPHASE_0001.fits'
for fList in os.walk(inDIR):
    for FileName in fList[2]:
        if fnmatch.fnmatch(FileName, patternData): # Match search string
            fileList.append(os.path.join(fList[0],FileName))

print(fileList)
#['/Volumes/LaCie/Reduced/Iter125SeptAllJuly13thLMbands/mat_raw_estimates.2018-07-13T05_41_36.HAWAII-2RG.rb/RAW_DPHASE_0001.fits']
#To compare only with all set up of the BCDs
#it can be only for the same source
fileList=[]
#To compare only with one set up of the BCDs
patternData='*RAW_DPHASE*'
for fList in os.walk(inDIR):
    for FileName in fList[2]:
        if fnmatch.fnmatch(FileName, patternData): # Match search string
            fileList.append(os.path.join(fList[0],FileName))

print(fileList)
#Found 0001 - 0012  :)

#To plot the vis
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.time import Time
#plotting Differential Phase--
#etaScl L
hducal=fits.open('/Volumes/LaCie/Reduced/Iter125SeptAllJuly13thLMbands/mat_raw_estimates.2018-07-13T10_08_15.HAWAII-2RG.rb/RAW_DPHASE_0001.fits')
viscal=hducal[4].data
vacal=viscal["VISAMP"]
d3cal=hducal[3].data
#calibrator HD16062 used for NGC1068 L
hdu = fits.open('/Volumes/LaCie/Reduced/Iter1test26SeptNGC1068LMbands/mat_raw_estimates.2018-07-13T09_03_40.HAWAII-2RG.rb/RAW_DPHASE_0001.fits')
hdu.info()
vis=hdu[4].data
vis.columns
va=vis["VISAMP"]
d3=hdu[3].data
wl=d3["EFF_WAVE"]
plt.plot(wl,vacal[0])
plt.plot(wl,va[0])
plt.show()

#FALTA Guardar automaticamente las grficas en archivos
#Plot OPDs and visibilities RAW_VIS2_0001.fits to 12   OI_OPDWVPO_0001.fits to 12
dirL = ['/Volumes/LaCie/Reduced/Iter125SeptAllJuly13thLMbands/']
dirN=['/Volumes/LaCie/Reduced/Iter16OctAllJuly13thNband/']
#Function to pass the file names reading them from a file .txt
files = []
hdu = fits.open("/Volumes/LaCie/Reduced/Iter16OctAllJuly13thNband/mat_raw_estimates.2018-07-13T03_26_00.AQUARIUS.rb/OI_OPDWVPO_0001.fits")/Volumes/LaCie/Reduced/Iter16OctAllJuly13thNband/mat_raw_estimates.2018-07-13T05_41_36.AQUARIUS.rb/OI_OPDWVPO_0001.fits
hdu = fits.open("/Volumes/LaCie/Reduced/Iter16OctAllJuly13thNband/mat_raw_estimates.2018-07-13T03_26_00.AQUARIUS.rb/RAW_VIS2_0001.fits")/Volumes/LaCie/Reduced/Iter16OctAllJuly13thNband/mat_raw_estimates.2018-07-13T05_41_36.AQUARIUS.rb/RAW_VIS2_0001.fits
/Volumes/LaCie/Reduced/Iter125SeptAllJuly13thLMbands/mat_raw_estimates.2018-07-13T08_34_14.HAWAII-2RG.rb/RAW_SPECTRUM_0001.fits
#hdu.info()
d1d = hdu[1].data    #d1d.columns
d1d["TARGET"]     #chararray(['V1068 Sco'], dtype='<U9')
d1d["CATEGORY"]   #chararray(['CAL'], dtype='<U3')
d3d = hdu[3].data
d3d["EFF_WAVE"].shape   #(121,)
wl=d3d["EFF_WAVE"]
d4d = hdu[4].data     #d4d.columns
d4d["VIS2DATA"].shape   #(6, 121)
vis2=d4d["VIS2DATA"]
vis2.shape  #(6, 121)
vis2[5].shape  #(121,)
d4d["STA_INDEX"]
""">>> d4d["STA_INDEX"]
    array([[13, 10],
    [ 1,  5],
    [ 5, 13],
    [ 5, 10],
    [ 1, 13],
    [ 1, 10]], dtype=int16)"""  d4d["UCOORD"] #array([-12.34808013,  11.31281987,  18.5063535 ,   6.15827337, 29.81917338,  17.47109325])
d4d["VCOORD"]   #array([-16.44526312, -22.45603692,  24.66759629,   8.22233317,  2.21155937, -14.23370375])
plt.plot(wl,vis2[0],label="")
#plt.grid(linestyle="--",linewidth=0.5,color='.25')
plt.legend()
#plt.xticks(rotation=45)
plt.title('')
plt.show()
#<--------------------------------D

# Oct 30th


                """
                    hdu[0].header["ORIGFILE"]  #'MATISSE_OBS_SIPHOT_LM_SKY_193_0001.fits'
                    HIERARCH ESO DEL FT STATUS = 'OFF     ' / Fringe Tracker Status
                    HIERARCH ESO DET TPLID VAL   = 'MATISSE_FRINGES' / Current template identificati
                    HIERARCH ESO INS DIL ID = 'LOW     ' / L&M DIL  unique id.
                    HIERARCH ESO INS DIL NAME = 'LOW     ' / L&M DIL name.
                    HIERARCH ESO INS DIL NO =    6 / L&M DIL wheel position index.
                    HIERARCH ESO INS DIN ID = 'LOW     ' / N DIL unique id.
                    HIERARCH ESO INS DIN NAME = 'LOW     ' / N DIL name.
                    HIERARCH ESO INS DIN NO =    6 / N DIL wheel position index.
                    HIERARCH ESO INS FFD ID = 'PUP20   ' / FFD unique ID.
                    HIERARCH ESO INS FFD NAME = 'PUP20   ' / FFD name.
                    HIERARCH ESO INS MODE = 'HYBRID  ' / Instrument mode used.
                    HIERARCH ESO ISS AMBI FWHM END = 1.43 / Observatory seeing [arcsec].
                    HIERARCH ESO ISS AMBI FWHM START = 1.43 / Observatory seeing [arcsec].
                    HIERARCH ESO ISS AMBI IRSKY TEMP = -78.30 / Temperature of the IR sky,from radi
                    HIERARCH ESO ISS AMBI IWV END = 1.81 / Integrated Water Vapor [mm]
                    HIERARCH ESO ISS AMBI IWV START = 1.84 / Integrated Water Vapor [mm].
                    HIERARCH ESO ISS AMBI IWV30D END = 1.91 / IWV at 30deg elev [mm].
                    HIERARCH ESO ISS AMBI IWV30D START = 1.91 / IWV at 30deg elev [mm].
                    HIERARCH ESO ISS AMBI IWV30DSTD END = 0.02 / IWV at 30deg elev. [mm].
                    HIERARCH ESO ISS AMBI IWV30DSTD START = 0.02 / IWV at 30deg elev. [mm].
                    HIERARCH ESO ISS AMBI IWVSTD END = 0.04 / Integrated Water Vapor [mm].
                    HIERARCH ESO ISS AMBI LRATE = -0.0050 / Observatory ambient lapse rate [K/m].
                    HIERARCH ESO ISS AMBI PRES = 745.48 / Observatory ambient air pressure [hPa].
                    HIERARCH ESO ISS AMBI RHUM = 4.50 / Relative humidity [percentage].
                    HIERARCH ESO ISS AMBI TEMP = 13.44 / Observatory ambient temperature [C].
                    HIERARCH ESO ISS AMBI TEMPDEW = -26.74 / Observatory ambient dew temperature
                    HIERARCH ESO ISS AMBI WINDDIR = 7.00 / Wind direction [deg].
                    HIERARCH ESO ISS TRAK STATUS = 'NORMAL  ' / Tracking status.
                    HIERARCH ESO OBS AIRM =    5.0 / Req. max. airmass
                    HIERARCH ESO OBS START = '2018-07-12T01:01:04' / OB start time
                    HIERARCH ESO OBS STREHLRATIO = 0.0 / Req. strehl ratio
                    HIERARCH ESO OBS TARG NAME = 'Tet Cen ' / OB target name
                    HIERARCH ESO OBS TPLNO =     2 / Template number within OB
                    HIERARCH ESO TPL EXPNO =     5 / Exposure number within template
                    HIERARCH ESO TPL ID = 'MATISSE_hyb_obs' / Template signature ID
                    HIERARCH ESO TPL NAME = 'Celestial target observation' / Template name
                    HIERARCH ESO TPL NEXP =     28 / Number of exposures within template
                    HIERARCH ESO TPL PRESEQ = 'MATISSE_obs.seq' / Sequencer script
                    HIERARCH ESO TPL START = '2018-07-12T01:06:44' / TPL start time
                    HIERARCH ESO TPL VERSION = '$Revision: 310644 $' / Version of the template
                    ORIGFILE= 'MATISSE_OBS_SIPHOT_LM_STD_193_0001.fits' / Original File Name
                    ARCFILE = 'MATIS.2018-07-12T01:10:13.746.fits' / Archive File Name
                    d2d["EXPTIME"]   #0.075
"""
#test
hdu=fits.open("/Volumes/LaCie/Reduced/Iter129OctTetCen/mat_raw_estimates.2018-07-12T01_06_44.HAWAII-2RG.rb/RAW_VIS2_0001.fits")
                
d3d = hdu[3].data
d3d["EFF_WAVE"].shape   #(121,)
wl=d3d["EFF_WAVE"]
d4d = hdu[4].data     #d4d.columns
d4d["VIS2DATA"].shape   #(6, 121)
vis2=d4d["VIS2DATA"]
plt.plot(wl,vis2[0])
plt.savefig('TEST.png')
#Plot and keep table for tetCen one BL at a time, 12 obs. for one TPLSTART
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.time import Time
import os, fnmatch
import pandas as pd
inDIR = '/Volumes/LaCie/Reduced/Iter129OctTetCen/mat_raw_estimates.2018-07-12T01_06_44.HAWAII-2RG.rb/'  #Change
pattern= 'RAW_VIS2_00*'                       #   Change
hducal=fits.open('/Volumes/LaCie/Reduced/Iter129OctTetCen/mat_raw_estimates.2018-07-12T01_06_44.HAWAII-2RG.rb/RAW_VIS2_0001.fits' )
d3cal=hducal[3].data
wl = d3cal["EFF_WAVE"]*1e6
len(wl)    #64
hducal.close()
DF=pd.DataFrame(columns=["i","name","pBLe", "airmass", "mjd", "mode", "SI1","SI2","SI3","SI4","object","categ","type","tplstart","T0E","T0S","windspeed","chopping","BCD1","BCD2","resolution","wlrange","wl0","DIT","fringetracker","readoutMd","readout","filter"])
Yes=[]
No=[]
DFvis2=pd.DataFrame(columns=wl)
i=1
for root, dirs, files in os.walk(inDIR):
    for filename in fnmatch.filter(files, pattern):
        #i=1   #uncomment TO PLOT 6 BLs at once from files 0001 to 0012
        for BL in range(5,6): #range(6) does 0 to 5,TO PLOT 6 BLs at once,forfile0001 then 0012
            #print(filename)
            hducal = fits.open(inDIR+filename)
            object=hducal[0].header["HIERARCH ESO OBS TARG NAME"]
            if BL==0:pble=hducal[0].header["HIERARCH ESO ISS PBL12 END"]
            elif BL==1:pble=hducal[0].header["HIERARCH ESO ISS PBL13 END"]
            elif BL==2:pble=hducal[0].header["HIERARCH ESO ISS PBL14 END"]
            elif BL==3:pble=hducal[0].header["HIERARCH ESO ISS PBL23 END"]
            elif BL==4:pble=hducal[0].header["HIERARCH ESO ISS PBL24 END"]
            elif BL==5:pble=hducal[0].header["HIERARCH ESO ISS PBL34 END"]
            airmass=np.mean([hducal[0].header["HIERARCH ESO ISS AIRM END"],hducal[0].header["HIERARCH ESO ISS AIRM START"]])  #hdu[0].header["HIERARCH ESO OBS AIRM"]=5.0/Req.max.airmass
            mjd=hducal[0].header["MJD-OBS"]
            mode=hducal[0].header["HIERARCH ESO INS MODE"]
            temp=hducal[2].data["STA_INDEX"]   #temporal var
            staind=(temp == [1,5,13,10])      #indexes of the stations
            si1=staind[0]                    #returns T or F
            si2=staind[1]
            si3=staind[2]
            si4=staind[3]
            objectSTD=hducal[0].header["OBJECT"]
            catg="NA" #hducal[0].header["HIERARCH ESO DPR CATG"]
            typ="NA" #hducal[0].header["HIERARCH ESO DPR TYPE"]
            tpls=hducal[0].header["HIERARCH ESO TPL START"]
            tau0end=hducal[0].header["HIERARCH ESO ISS AMBI TAU0 END"]# Coherence time [seconds].
            tau0start=hducal[0].header["HIERARCH ESO ISS AMBI TAU0 START"]# Coherence time[seconds]
            wind=hducal[0].header["HIERARCH ESO ISS AMBI WINDSP"] # Observatory wind speed [m/s].
            chop=hducal[0].header["HIERARCH ESO ISS CHOP ST"] # Chopping status.
            bcd1=hducal[0].header["HIERARCH ESO INS BCD1 ID"]   #'OUT'
            bcd2=hducal[0].header["HIERARCH ESO INS BCD2 ID"]
            res=hducal[0].header["HIERARCH ESO INS DIL NAME"]   #dispersor L band we want LOW
            walrange=len(hducal[3].data["EFF_WAVE"])
            wal0=hducal[3].data["EFF_WAVE"][0]*1e6
            dit=hducal[0].header["EXPTIME"]  #we want 50-120 ms
            frtr=hducal[0].header["HIERARCH ESO DEL FT STATUS"]  # 'OFF     '/Fringe Tracker Status
            ro=hducal[0].header["HIERARCH ESO DET READ CURID"]# 1 /Used readout mode id
            romd=hducal[0].header["HIERARCH ESO DET READ CURNAME"]#'SCI-SLOW-SPEED'Usedreadoutmodename
            filter=hducal[0].header["HIERARCH ESO INS SFL NAME"]#1.50 /L&MSpatialFilterdevicename.
            DF.loc[i] = [i,object,pble,airmass,mjd,mode,si1,si2,si3,si4,objectSTD,catg,typ,tpls,tau0end,tau0start,wind,chop,bcd1,bcd2,res,walrange,wal0,dit,frtr,ro,romd,filter]  #Comment to plot without Noisy data
            datacal=hducal[4].data["VIS2DATA"]
            if len(wl)==len(datacal[BL]):    #64
                plt.plot(wl,datacal[BL],label=i) #BL changes the BL (same order for all files??)
                plt.text(wl[i*5],datacal[BL][i*5],i)
                Yes.append(i)
                DFvis2.loc[i]=datacal[BL]
                i=i+1
                hducal.close()
            else:
                print(len(vacal[BL]))   #118
                No.append(i)      #Comment to plot without Noisy data
                hducal.close()
        #TO PLOT 6 BLs at once from files 0001 to 0012
        #plt.grid(linestyle="--",linewidth=0.5,color='.25')
        #plt.legend(prop={'size': 7})
        #title='VIS2 tetCen 6 BLs' + filename
        #plt.title(title)
        #plt.xlabel('wavelength [um]')
        #plt.ylabel('VIS2')
        #plt.show()
    #TO PLOT one BL for the 12 files
    plt.grid(linestyle="--",linewidth=0.5,color='.25')
    plt.legend(prop={'size': 7})
    title='VIS2 tetCen BL' + str(BL+1)
    plt.title(title)
    plt.xlabel('wavelength [um]')
    plt.ylabel('VIS2')
    #plt.savefig('vis2tetCen.png')
    plt.show()
        
len(Yes)  #10
len(No)   #0
                
DF.to_csv('tetCenBL6.csv')
DF.to_pickle('tetCenBL6')         #To save in M51              Change
DFvis2.to_csv('vis2tetCenBL6.csv')
DFvis2.to_pickle('vis2tetCenBL6')
#import pandas as pd
#df = pd.read_pickle('tetCen')  #To read back

hducal=fits.open("/Volumes/LaCie/Reduced/Iter129OctTetCen/mat_raw_estimates.2018-07-12T01_06_44.HAWAII-2RG.rb/RAW_VIS2_0009.fits")
#TPLS:/Volumes/LaCie/Reduced/Iter129OctTetCen/mat_raw_estimates.2018-07-12T02_31_33.HAWAII.rb
hducal[0].header["DATE-OBS"]   #'2018-07-12T01:26:27.0271'
#Imaging data has the index 2  "TARTYP"  U unknown, S sky, T target
#For the July 13th cal
hducal=fits.open("/Volumes/LaCie/Reduced/Iter125SeptAllJuly13thLMbands/mat_raw_estimates.2018-07-13T23_07_24.HAWAII-2RG.rb/RAW_VIS2_0001.fits")
hducal[0].header["DATE-OBS"]   #'2018-07-13T23:13:33.5024'
#GET FOR THE 12 files!!!  Done  :)
#TABLE with Measured vis averaged over 3.4 to 3.6 and the Bessel function and diameters Done :)
#MAIL DATE_OBS OF THE CALS. Done :)
#Cambiar unidades bl m a um y volver a graficar




MJD-OBS =       58312.14971413 / 2018-07-13T03:35:35.3008
DATE-OBS= '2018-07-13T03:35:35.3008' / Observing date
HIERARCH ESO TPL START = '2018-07-13T03:26:00' / TPL start time
HIERARCH ESO TPL EXPNO =     8 / Exposure number within template
HIERARCH ESO TPL ID = 'MATISSE_hyb_obs' / Template signature ID
HIERARCH ESO TPL NAME = 'Celestial target observation' / Template name
HIERARCH ESO TPL NEXP =     12 / Number of exposures within template


#  2 Nov
import pandas as pd
cats=pd.read_csv("JSDCAllWISE.csv")  #crossmatch to 5 arcsec
cats.columns #JSDC Johnson mags, AllWISE Vega mags
"""Index(['angDist', '_RAJ2000', '_DEJ2000', 'Dis', 'Name', 'SpType', 'Bmag',
       'Vmag', 'Rmag', 'Imag', 'Jmag', 'Hmag', 'Kmag', 'Lmag', 'Mmag', 'Nmag',
       'LDD', 'e_LDD', 'LDDCHI', 'CalFlag', 'UDDB', 'UDDV', 'UDDR', 'UDDI',
       'UDDJ', 'UDDH', 'UDDK', 'UDDL', 'UDDM', 'UDDN', 'Simbad', 'RAJ2000',
       'DEJ2000', 'AllWISE', 'RAJ2000.1', 'DEJ2000.1', 'eeMaj', 'eeMin',
       'eePA', 'W1mag', 'W2mag', 'W3mag', 'W4mag', 'Jmag.1', 'Hmag.1',
       'Kmag.1', 'e_W1mag', 'e_W2mag', 'e_W3mag', 'e_W4mag', 'e_Jmag',
       'e_Hmag', 'e_Kmag', 'ID', 'ccf', 'ex', 'var', 'qph', 'pmRA', 'e_pmRA',
       'pmDE', 'e_pmDE', 'd2M'],
      dtype='object')"""
#Revisar:
cats["angDist"][0] #less than 2 arcsec
cats["qph"][0]     #AA
cats["Kmag.1"][0]     #J, H and K are from 2mass
cats["Lmag"][0]  #6.653   L, M and N are = W1-W3 from WISE!!
cats["W1mag"][0]  #6.653
d=cats["W1mag"]-cats["Lmag"]
temp=d==0   sum(temp)455020
import math
totalnan=[]
for i in range(len(cats["W1mag"])):
    totalnan.append(math.isnan(cats["W1mag"][i]))

sum(totalnan)  #6721   360
for i in range(len(cats["W1mag"])):
    totalnan.append(math.isnan(cats["W1mag"][i]))
#For JSDC:
#Column   Lmag,M,N   from table: II/346/jsdc_v2  L magnitude (Cat. II/311/wise W1mag,W2,W3),
#Column  J,H,Kmag from 2MASS
#PENDING
x.append(cats["Kmag"][i],cats["Lmag"],cats["N"]
for i in range(30):
    plt.plot(x, np.poly1d(np.polyfit(x, y, 1))x)
        
#Conv https://www.gemini.edu/sciops/instruments/midir-resources/imaging-calibrations/fluxmagnitude-conversion
cats["Kmag.1"][4]   #10.8  #John0.0307Jy 2MASSKs0.0319Jy mono:0.0018 Jy
cats.loc[cats['Name'] == 'HD 224419']
angDist    _RAJ2000  _DEJ2000  Dis  ...   e_pmRA   pmDE  e_pmDE    d2M
462000  4.490261  359.441372 -7.375179    0  ...     25.0  847.0    24.0    NaN
463488  0.513703  359.441372 -7.375179    0  ...     56.0 -223.0    57.0  0.545

cats.loc[cats["Name"] == NamesJSDC[0]] #NO
cats[cats["Name"].str.contains('B Sgr')]  #322573
cats["Name"][322573]    #'V* BB Sgr'
cats[cats["Name"].str.contains(NamesJSDC[0])] #returns the same
dataWISE=pd.DataFrame(columns="0","1")
for i in range(10):
    ind=cats[cats["Name"].str.contains(NamesJSDC[i])]
    print(ind["Name"],ind["angDist"],ind["W1mag"]-ind["W2mag"],ind["W2mag"]-ind["W3mag"])
    
#print(ind["angDist"],ind["W1mag"],ind["W2mag"],ind["W3mag"])
#dataWISE.loc[i]=[ind]     #Doesnt work



dfx["name1"]
#------------------------------------------------------
# 11 Nov
#EM* MWC 297   catg: science, type: object July 13th
FluxVega = 666.7*10**(-0.4*3.04)  # 40.5443605352521
#Kmag: 3.04 Spec Type:B1.5Ve C diameter as HD 16062 or slightly larger
#Observed with larger integration times under MED and HIGH res.
#We can not use it as a calibrator
         """
         IDL> flux
         10         962         644        1357           7           0
         21         857         857          75          70           0
         87         101       29072         956         692         216
         189           1           0         193         203         109
         1437         156         318         844         914         176
         IDL> help,flux
         FLUX            LONG      = Array[30]
         IDL> weak=where(flux lt 22)
         IDL> print,z.name[weak]
         HD 224726 HD  16062 BD-01   381 HD19994 HD 143192 HD 170058 HD 170759
hhh
"""

         
import numpy as np
import matplotlib.pyplot as plt
my_array = np.arange(20)
my_array2 = my_array
z = -1*np.arange(20)
         
graph = plt.scatter(my_array, my_array2, c=z, cmap=plt.cm.coolwarm)
cb = plt.colorbar(graph)
cb.set_label('mean value')
plt.show()
#Raw data from July
hdu=fits.open('/Users/M51/MATISSEDATA/2018-07-13/MATISSE_OBS_SIPHOT_LM_OBJECT_194_0001.fits' )
hdu[4].data['STA_NAME']#         chararray(['A0', 'B2', 'D0', 'C1'],
hdu[4].data['STA_INDEX']        # array([ 1,  5, 13, 10], dtype=int16)
hdu[4].data['STAXYZ'] array([[ 14.63825159,  55.79954648,   4.53235481],
                [ -0.7373607 ,  75.88981035,   4.53803567],
                [-15.6279567 ,  45.39734809,   4.5397    ],
                [ -5.6900299 ,  65.72884937,   4.5389253 ]])

         

# 29 Nov
#in order of baselines
#Taking the average from each date where there were non-chopped and using both II and OO
#NGC 1068 =[np.average([]),np.average([]),np.average([]),np.average([]),np.average([]),np.average([])]
NGC1068OOvisSept22=[np.average([0.07,0.09,0.09]),np.average([0.16,0.16,0.15]),np.average([0.15,0.18,0.16]),np.average([0.03,0.03,0.02]),np.average([0.12,0.12,0.13]),np.average([0.18,0.15,0.16])]
NGC1068OOvisSept24=[np.average([0.05,0.05]),np.average([0.37,0.33]),np.average([0.28,0.27]),np.average([0.04,0.04]),np.average([0.17,0.16]),np.average([0.13,0.13])]
NGC1068OOvisSept25=[np.average([0.05,0.05,0.05]),np.average([0.15,0.16,0.15]),np.average([0.12,0.15,0.13]),np.average([0.02,0.02,0.02]),np.average([0.16,0.15,0.15]),np.average([0.12,0.11,0.11])]
NGC1068IIvisSept22=[np.average([0.07,0.09,0.09]),np.average([0.16,0.16,0.15]),np.average([0.15,0.18,0.16]),np.average([0.03,0.03,0.02]),np.average([0.12,0.12,0.13]),np.average([0.18,0.15,0.16])]
NGC1068IIvisSept24=[np.average([0.05,0.05]),np.average([0.37,0.33]),np.average([0.28,0.27]),np.average([0.04,0.04]),np.average([0.17,0.16]),np.average([0.13,0.13])]
NGC1068IIvisSept25=[np.average([0.05,0.05,0.05]),np.average([0.15,0.16,0.15]),np.average([0.12,0.15,0.13]),np.average([0.02,0.02,0.02]),np.average([0.16,0.15,0.15]),np.average([0.12,0.11,0.11])]
         
NGC1068OOvisSept22CAL=np.divide(NGC1068OOvisSept22,HD16658OOvisSept22)
NGC1068IIvisSept22CAL=np.divide(NGC1068IIvisSept22,HD16658IIvisSept22)
#Make an average of the II and OO above?

NGC1068CFluxSept22=[np.average([0.27,0.33,0.32]),np.average([0.57,0.56,0.57]),np.average([0.49,0.60,0.57]),np.average([0.09,0.09,0.09]),np.average([0.38,0.44,0.44]),np.average([0.53,0.52,0.55])]
NGC1068CFluxSept24=[np.average([0.30,0.36]),np.average([2.11,2.28]),np.average([1.63,1.72]),np.average([0.20,0.24]),np.average([0.84,0.90]),np.average([0.63,0.76])]
NGC1068CFluxSept25=[np.average([0.27,0.31,0.32]),np.average([0.86,0.93,0.92]),np.average([0.73,0.86,0.83]),np.average([0.09,0.11,0.11]),np.average([0.80,0.80,0.83]),np.average([0.58,0.61,0.64])]
NGC1068CFluxSept22CAL=np.divide(NGC1068CFluxSept22,HD16658CFluxSept22)

#Make an average of the II and OO above?
#Remember that for II the order changes
#Choose by matching numbers
p=pd.read_pickle('ngc1068Sept-VIS2')
p22=p.loc[p["tplstart"]=='2018-09-22T04:11:40']
p22=p22.loc[p["chopping"]=='F'] #On this date there were only OO and II
p22=p22.reset_index()
sum22=np.zeros(6)
sumPA22=np.zeros(6)
for j in range(0,6):
    for i in range(len(p22["name"])):
        sum22[j]=sum22[j]+p22['pBLe'][i][j]
        sumPA22[j]=sumPA22[j]+p22['pBLAe'][i][j]
    
NGC1068Sept22BL=sum22/len(p22["name"])
NGC1068Sept22PA=sumPA22/len(p22["name"])
p24=p.loc[p["tplstart"]=='2018-09-24T05:21:50']
p24=p24.loc[p24["chopping"]=='F'] #On this date there were OI and IO
p24OO=p24.loc[(p24['BCD1']=='OUT')&(p24['BCD2']=='OUT')]
p24=p24.loc[p24["chopping"]=='F']
p24II=p24.loc[(p24['BCD1']=='IN')&(p24['BCD2']=='IN')]
p24=pd.concat([p24OO,p24II], ignore_index=True)
p24=p24.reset_index()
sum24=np.zeros(6)
sumPA24=np.zeros(6)
for j in range(0,6):
    for i in range(len(p24["name"])):
        sum24[j]=sum24[j]+p24['pBLe'][i][j]
        sumPA24[j]=sumPA24[j]+p24['pBLAe'][i][j]
         
NGC1068Sept24BL=sum24/len(p24["name"])
NGC1068Sept24PA=sumPA24/len(p24["name"])
p25=p.loc[p["tplstart"]=='2018-09-25T05:21:36']
p25=p25.loc[p["chopping"]=='F'] #On this date there were only OO and II
p25=p25.reset_index()
sum25=np.zeros(6)
sumPA25=np.zeros(6)
for j in range(0,6):
    for i in range(len(p25["name"])):
        sum25[j]=sum25[j]+p25['pBLe'][i][j]
        sumPA25[j]=sumPA25[j]+p25['pBLAe'][i][j]
         
NGC1068Sept25BL=sum25/len(p25["name"])
NGC1068Sept25PA=sumPA25/len(p25["name"])

#Using all because they had 20% or less good frames on all the baselines
HD16658OOvisSept22=[np.average([1.07,0.72,0.99]),np.average([0.95,1.16,0.99]),np.average([0.98,0.90,1.08]),np.average([0.76,0.81,0.87]),np.average([0.70,0.82,0.56]),np.average([0.60,0.83,0.66])]
HD16658IIvisSept22=[np.average([1.07,0.72,0.99]),np.average([0.95,1.16,0.99]),np.average([0.98,0.90,1.08]),np.average([0.76,0.81,0.87]),np.average([0.70,0.82,0.56]),np.average([0.60,0.83,0.66])]
HD16658CFluxSept22=[np.average([1.01,0.90,1.03]),np.average([0.90,1.24,0.98]),np.average([0.87,1.01,1.02]),np.average([0.79,0.82,0.99]),np.average([0.68,0.88,0.62]),np.average([0.60,0.71,0.71])]
#Array with averaged baselines Remember that for II the order changes
#Choose by matching numbers
p=pd.read_pickle('SeptLOWNCOO')
p22=p.loc[p["name"]=='HD 16658'] #2018-09-22T04:44:55
p22=p22.reset_index()
p=pd.read_pickle('SeptLOWNCII')
p=p.loc[p["name"]=='HD 16658']  #2018-09-22T04:44:55
p22=pd.concat([p22,p], ignore_index=True)
sum22=np.zeros(6) #for each baseline
sumPA22=np.zeros(6)
for j in range(0,6): #baselines
    for i in range(len(p22["name"])):
         sum22[j]=sum22[j]+p22['pBLe'][i][j]
         sumPA22[j]=sumPA22[j]+p22['pBLAe'][i][j]
        
HD16658Sept22BL=sum22/len(p22["name"])
HD16658Sept22PA=sumPA22/len(p22["name"])


import matplotlib.pyplot as plt
#with BL
BLOO=[NGC1068Sept22BL[5],NGC1068Sept22BL[0],NGC1068Sept22BL[3],NGC1068Sept22BL[4],NGC1068Sept22BL[1],NGC1068Sept22BL[2]]
plt.plot(BLOO,NGC1068OOvisSept22CAL,'o',label='visCAL')
plt.plot(BLOO,NGC1068CFluxSept22CAL,'o',label='CFluxCAL')
title= "NGC1068"
plt.title(title)
plt.xlabel('BL[m]')
plt.ylabel('CFluxCAL')
plt.xlim((0.0,100))
plt.grid(linestyle="--",linewidth=0.5,color='.25')
#plt.legend(prop={'size': 7})
#plt.savefig(title)
plt.show()
         
#with PA
PAOO=[NGC1068Sept22PA[5],NGC1068Sept22PA[0],NGC1068Sept22PA[3],NGC1068Sept22PA[4],NGC1068Sept22PA[1],NGC1068Sept22PA[2]]
plt.plot(PAOO,NGC1068OOvisSept22CAL,'o',label='visCAL')
plt.plot(PAOO,NGC1068CFluxSept22CAL,'o',label='CFluxCAL')
title= "NGC1068"
plt.title(title)
plt.xlabel('PA')
plt.ylabel('CFluxCAL')  #Change
#plt.xlim((0.0,120))
plt.grid(linestyle="--",linewidth=0.5,color='.25')
#plt.legend(prop={'size': 7})
#plt.savefig(title)
plt.show()
         
         
NGC 1566 0.3Jy
NGC 1365 0.4Jy <- unresolved
NGC 7469 0.15Jy
BGC 1068 0.5Jy
cf=Jy*counts/1000
plot correlatedfluxnormalized vs wl
compare with BL
         
#18Feb        
#PLOTS TO COMPARE CORRFLUXES, VIS2
test=mu.mrestore("2018-12-02T02:54:59.tpl.pk")
len(test)  #2
test[0].keys()
test[0]['mjd'][0]  #Has 385 entries:58454.12495949056 to 58454.125652823896
dcr=test[0]
dcr.keys() #dict_keys(['phot', 'wave', 'cflux', 'flux', 'vis', 'mjd', 'opd', 'tau', 'ra', 'dec', 'targ', 'file', 'sky'])
dcr['wave'].shape
#CORR FLUX
dcr['flux'].shape     #    (6, 64)
import matplotlib.pyplot as plt
plt.plot(dcr['wave'],dcr['flux'][0].real,linestyle='--')  #Has 6, one for each BL  - Real part
from astropy.io import fits
dci=fits.open("/Volumes/LaCie/DecemberData/7CetOODRS/Iter1/mat_raw_estimates.2018-12-02T02_54_59.HAWAII-2RG.rb/OBJ_CORR_FLUX_0001.fits")
dci[1].data['TIME'].shape #385
dci[1].data['TIME'][0]    #58454.12495949056 to 58454.125652823896
re=dci[1].data['TIME']-test[0]['mjd']  # 0
dci[1].data['CORRFLUXREAL1'].shape  #385 mjd , 64 wl, 625 x detector
         #For one wl
re=dci[1].data['CORRFLUXREAL1'][0,0,:]     #  - Real part
im=dci[1].data['CORRFLUXIMAG1'][0,0,:]     #  - imag part
com=np.zeros(625,dtype=complex)
for i in range(625): com[i]=complex(re[i],im[i])
cf=np.fft.fft(com)
         
for l in range(64):
plt.plot(dcr['wave'],cf1)   #No Wavelength!!
#SQUARED VIS
dci=fits.open("/Volumes/LaCie/DecemberData/7CetOODRS/Iter1/mat_raw_estimates.2018-12-02T02_54_59.HAWAII-2RG.rb/RAW_VIS2_0001.fits")
wav=dci[3].data['EFF_WAVE']*1e6
v2=dci[4].data #6
        
vis2=np.abs(dcr['vis'])
#for b in range(6): plt.plot(dcr['wave'],dcr['vis'][b].real)
for b in range(6): plt.plot(dcr['wave'],vis2[b],linestyle='--',label=str(b)+'co')
for b in range(6): plt.plot(wav,v2['VIS2DATA'][b],label=str(b)+'in')
#FIX, not to use by now
dci=fits.open("/Volumes/LaCie/DecemberData/7CetOODRS/Iter1/mat_raw_estimates.2018-12-02T02_54_59.HAWAII-2RG.rb/RAW_SPECTRUM_0001.fits")
flux0=dci['OI_FLUX'].data['FLUXDATA'][0,:]  #4,64
plt.plot(dci['OI_WAVELENGTH'].data,flux0)  #only 4!!  DOESNT WORK

         
         
         
         
#For DRS
#ARCFILE = 'MATIS.2018-12-02T02:59:56.421.fits' / Archive File Name
dci[4].data['MJD']  58454.12530616
#For EWS
dcr['mjd'].shape    #(385,)
dcr['mjd'][0] #58462.12740143498
dcr['mjd'][384]  #58462.12809476831


         
         
         
         
         
#For 2 very faint targets, BUT weather was bad.
#Reduced with EWS
from mcdb import wutil as wu
t1=wu.mrestore('/Users/M51/2018-12-04T06:31:23.tpl.pk')
t1=t1[0]
for i in range(6):plt.plot(t1['wave'],t1['flux'].real[i])
t2=wu.mrestore('/Users/M51/2018-12-04T06:13:19.tpl.pk')
t2=t2[0]
         
for i in range(6):plt.plot(t1['wave'],t1['flux'].real[i])
for i in range(6):plt.plot(t2['wave'],np.abs(t2['vis'][i]))
wu.tv(t1['cflux']['opdimage'])
wu.tv(t2['cflux']['opdimage'])
print(t2['tau'])
plt.plot(t2['cflux']['opdp'][0])
plt.plot(t1['cflux']['opdp'][0],color='red')
#EWS reduced files
#WHy are the tails for some vis falling at long wl and for some they just remain stable?
#1. the airmass causes a refraction index with wl
#2.the photometry doesnt overlap in pixels
#Both seem to be ok, there must be other reason.
         
t1=wu.mrestore('/Volumes/LaCie/DecemberData/NotV482Car/2018-12-04T05:48:29.tpl.pk')
for i in range(6):plt.plot(t1['wave'],np.abs(t1['vis'][i]))
t1['pbl']
t2=wu.mrestore('/Volumes/LaCie/DecemberData/V482Car/2018-12-04T06\:51\:07.tpl.pk')
for i in range(6):plt.plot(t2['wave'],np.abs(t2['vis'][i]))
t2['pbl']
help(mu.airyBase)
a=mu.airyBase(4.01,t2['wave'],108.8)
plt.plot(t2['wave'],0.5*a)
ax=mu.airyBase(4.6,t1['wave'],108.8)
plt.plot(t2['wave'],0.5*ax)
plt.clf()
wa=t2['wave']
for i in range(6):plt.plot(wa,np.abs(t2['vis'][i]/mu.airyBase(4.01,wa,t2['pbl'][i])))
for i in range(6):plt.plot(wa,np.abs(t1['vis'][i]/mu.airyBase(4.6,wa,t1['pbl'][i])))
for i in range(6):plt.plot(wa,np.abs(t2['vis'][i]/mu.airyBase(4.01,wa,t2['pbl'][i])),color='red')
for i in range(6):plt.plot(wa,np.abs(t1['vis'][i]/mu.airyBase(4.6,wa,t1['pbl'][i])),color='green')
print(t1['tau']) #1.449   less than 4 is bad
print(t2['tau'])   #1.9880000000000002
talpgood=wu.mrestore('/Volumes/LaCie/DecemberData/alpEri/2018-12-09T01:17:59.tpl.pk')
for i in range(6):plt.plot(wa,np.abs(talpgood[0]['vis'][i]),color='blue')
talpbad=wu.mrestore('/Volumes/LaCie/DecemberData/alpEri/2018-12-04T00:24:17.tpl.pk')
for i in range(6):plt.plot(wa,np.abs(talpbad[0]['vis'][i]),color='black')
print(talpbad[0]['tau'])  #1.788
print(talpgood[0]['tau'])  #4.153
h=(talpgood[0]['header'])
print(h['hierarch eso iss airm start'])  #1.189
h=(talpbad[0]['header'])
print(h['hierarch eso iss airm start'])   #1.207 about the same
h=(t1['header'])
print(h['hierarch eso iss airm start'])  #1.039
h=(t2['header'])
print(h['hierarch eso iss airm start'])    #1.344
cflux=t1['cflux']
cflux.keys()
Out[56]: dict_keys(['localopd', 'mjd', 'wave', 'fdata', 'opdf', 'delay', 'opd', 'opdp', 'cdata', 'phase', 'pdata', 'flux', 'opdimage'])
t1.keys()
Out[57]: dict_keys(['phot', 'wave', 'cflux', 'flux', 'vis', 'mjd', 'opd', 'pbl', 'header', 'tau', 'ra', 'dec', 'mjd-obs', 'targ', 'file', 'sky'])
os.chdir('/Volumes/LaCie/DecemberData')
pd=mu.getphotdata(t1['file'],t1['sky'])
pd.keys()
Out[62]: dict_keys(['data', 'sky', 'meansky', 'bad'])
d=pd['data']
dm=np.mean(d,1)
dm.shape#         Out[65]: (4, 64, 150)
for i in range(4): dm[i]-=pd['meansky'][i]
#To plot
wu.tv(dm[0])
pdgood=mu.getphotdata(talpgood[0]['file'],talpgood[0]['sky'])
dgood=pdgood['data']
dmgood=np.mean(dgood,1)
for i in range(4): dmgood[i]-=pdgood['meansky'][i]
plt.clf()
plt.plot(np.mean(dmgood[0],1))
plt.plot(np.mean(dm[0],1))
plt.clf()
plt.plot(wa,np.mean(dm[0],1)/np.mean(dmgood[0],1))
pdbad=mu.getphotdata(talpbad[0]['file'],talpbad[0]['sky'])
dbad=pdbad['data']
dmbad=np.mean(dbad,1)
for i in range(4): dmbad[i]-=pdbad['meansky'][i]
plt.plot(wa,np.mean(dm[0],1)/np.mean(dmbad[0],1))
         
         
         
         
#22 Marzo
         s1=np.where(cat['UDDL']<3)  139873
         s2=np.where(cat['CalFlag']==0)
         
from astropy.io import fits
catalog=fits.open('/Users/M51/msdfcc-v9.fits')
catalog[1].data.shape  # (465857,)
catalog[1].data.columns
cat=pd.DataFrame(catalog[1].data)
cat.keys()
         : (465857, 49)
         (['Name', 'SpType', 'RAJ2000', 'DEJ2000', 'distance', 'teff_midi', 'teff_gaia', 'Comp', 'mean_sep', 'mag1', 'mag2', 'diam_midi', 'e_diam_midi', 'diam_cohen', 'e_diam_cohen', 'diam_gaia', 'LDD_meas', 'e_diam_meas', 'UDD_meas', 'band_meas', 'LDD_est', 'e_diam_est', 'UDDL_est', 'UDDM_est', 'UDDN_est', 'Jmag', 'Hmag', 'Kmag', 'W4mag', 'CalFlag', 'IRflag', 'nb_Lflux', 'med_Lflux', 'disp_Lflux', 'nb_Mflux', 'med_Mflux', 'disp_Mflux', 'nb_Nflux', 'med_Nflux', 'disp_Nflux', 'Lcorflux_30', 'Lcorflux_100', 'Lcorflux_130', 'Mcorflux_30', 'Mcorflux_100', 'Mcorflux_130', 'Ncorflux_30', 'Ncorflux_100', 'Ncorflux_130'], dtype='object')
cat['DEJ2000'][0]   #'-16:42:58.017'
mincf=np.min(cat['Lcorflux_30'],cat['Lcorflux_100'],cat['Lcorflux_130']) #No
#USING THIS FOR PROPOSAL
from astropy import units as u  # Needs 2 steps :(
ra = coord.Angle(first_row["ra"], unit=u.hour)
ra.degree
s1=np.where((cat['UDDL_est']<3) & (cat['CalFlag']==0) & (cat['Lcorflux_130']>0.2)  & (cat['Ncorflux_130']>4.) & ((cat['IRflag']==0) | (cat['IRflag']==2) | (cat['IRflag']==4) | (cat['IRflag']==6)) & (cat['disp_Lflux']/cat['med_Lflux']<0.15) & (cat['disp_Nflux']/cat['med_Nflux']<0.30)  ) # & (np.int(cat['DEJ200']<25) )
#For CALIBRATORS
s3=np.where((cat['UDDL_est']<3) & (cat['CalFlag']==0) & (cat['Lcorflux_130']>0.2)  & (cat['Ncorflux_130']>4.) & ((cat['IRflag']==0) | (cat['IRflag']==2) | (cat['IRflag']==4) | (cat['IRflag']==6)) & (cat['disp_Lflux']/cat['med_Lflux']<0.05) & (cat['disp_Nflux']/cat['med_Nflux']<0.10)  & (cat['med_Lflux']>7) & (cat['med_Lflux']<35) ) # 2961
s3=np.where((cat['UDDL_est']<3) & (cat['CalFlag']==0) & ((cat['IRflag']==0) | (cat['IRflag']==2) | (cat['IRflag']==4) | (cat['IRflag']==6)) & (cat['disp_Lflux']/cat['med_Lflux']<0.05) & (cat['disp_Nflux']/cat['med_Nflux']<0.10)  & (cat['med_Lflux']>4.2) & (cat['med_Lflux']<5.8) ) # 2648
s3=np.where((cat['UDDL_est']<3) & (cat['CalFlag']==0) & ((cat['IRflag']==0) | (cat['IRflag']==2) | (cat['IRflag']==4) | (cat['IRflag']==6)) & (cat['disp_Lflux']/cat['med_Lflux']<0.05) & (cat['disp_Nflux']/cat['med_Nflux']<0.10)  & (cat['med_Lflux']>0.99) & (cat['med_Lflux']<1.01) ) # 44564
len(cat['Name'][s1[0]])   498
Names=cat['Name'][s1[0]]
DECs=cat['DEJ2000'][s1[0]]
DECs=DECs.reset_index()
selecteddec=[]
for i in range(len(DECs)):
    print(cat['DEJ2000'][i][:3])
    if(np.int(cat['DEJ2000'][i][:3])<27):
         selecteddec.append(DECs['DEJ2000'][i])

len(selecteddec)  #372
FDF=pd.DataFrame(columns=['Name', 'SpType', 'RAJ2000', 'DEJ2000', 'distance', 'teff_midi', 'teff_gaia', 'Comp', 'mean_sep', 'mag1', 'mag2', 'diam_midi', 'e_diam_midi', 'diam_cohen', 'e_diam_cohen', 'diam_gaia', 'LDD_meas', 'e_diam_meas', 'UDD_meas', 'band_meas', 'LDD_est', 'e_diam_est', 'UDDL_est', 'UDDM_est', 'UDDN_est', 'Jmag', 'Hmag', 'Kmag', 'W4mag', 'CalFlag', 'IRflag', 'nb_Lflux', 'med_Lflux', 'disp_Lflux', 'nb_Mflux', 'med_Mflux', 'disp_Mflux', 'nb_Nflux', 'med_Nflux', 'disp_Nflux', 'Lcorflux_30', 'Lcorflux_100', 'Lcorflux_130', 'Mcorflux_30', 'Mcorflux_100', 'Mcorflux_130', 'Ncorflux_30', 'Ncorflux_100', 'Ncorflux_130'])


for i in range (len(selecteddec)):
    for j in range (len(s1[0])):
        if(cat['DEJ2000'][s1[0][j]]==selecteddec[i]):
         FDF.loc[i]=cat.iloc[s1[0][j]]  #Add row i

FDF.shape  #(372, 49)  Wiiiiii!!!
FDF.to_pickle('ObsPropCals.pk')
np.max(FDF['med_Lflux'])  # 133.89999389648438
np.min(FDF['med_Lflux'])# 14.889442443847656
np.min(FDF['UDDL_est'])  #0.7689999938011169
np.max(FDF['UDDL_est']) #2.984999895095825
np.max(FDF['med_Nflux'])   #19.399999618530273
np.min(FDF['med_Nflux'])  #4.070000171661377
s2=np.where((FDF['med_Lflux']>19)&(FDF['med_Lflux']<21) ) #No choices with RA between 0 and 12 with LFlux <19
FDF['med_Lflux'][s2[0]].item    341    HD    853
s5=np.where((FDF['med_Lflux']>30)&(FDF['med_Lflux']<50))
FDF['RAJ2000'][s5[0]].item   308, HD 1923,
FDF['Name'][s5[0]].item
         
         
         
#Fainter than 20:
FDF2=pd.DataFrame(columns=['Name', 'SpType', 'RAJ2000', 'DEJ2000', 'distance', 'teff_midi', 'teff_gaia', 'Comp', 'mean_sep', 'mag1', 'mag2', 'diam_midi', 'e_diam_midi', 'diam_cohen', 'e_diam_cohen', 'diam_gaia', 'LDD_meas', 'e_diam_meas', 'UDD_meas', 'band_meas', 'LDD_est', 'e_diam_est', 'UDDL_est', 'UDDM_est', 'UDDN_est', 'Jmag', 'Hmag', 'Kmag', 'W4mag', 'CalFlag', 'IRflag', 'nb_Lflux', 'med_Lflux', 'disp_Lflux', 'nb_Mflux', 'med_Mflux', 'disp_Mflux', 'nb_Nflux', 'med_Nflux', 'disp_Nflux', 'Lcorflux_30', 'Lcorflux_100', 'Lcorflux_130', 'Mcorflux_30', 'Mcorflux_100', 'Mcorflux_130', 'Ncorflux_30', 'Ncorflux_100', 'Ncorflux_130'])
s3=np.where((cat['UDDL_est']<3) & (cat['CalFlag']==0) & (cat['med_Lflux']<20) & ((cat['IRflag']==0) | (cat['IRflag']==2) | (cat['IRflag']==4) | (cat['IRflag']==6)) & (cat['disp_Lflux']/cat['med_Lflux']<0.15) & (cat['disp_Nflux']/cat['med_Nflux']<0.30))
s3=np.where((cat['UDDL_est']<3) & (cat['CalFlag']==0) & (cat['med_Lflux']<20) & ((cat['IRflag']==0) | (cat['IRflag']==2) | (cat['IRflag']==4) | (cat['IRflag']==6)) & (cat['disp_Lflux']/cat['med_Lflux']<0.15) & (cat['disp_Nflux']/cat['med_Nflux']<0.30) & (cat['med_Lflux']>9.9) & (cat['med_Lflux']<10.1) )  #Faster
s3=np.where((cat['UDDL_est']<3) & (cat['CalFlag']==0) & (cat['med_Lflux']<20) & ((cat['IRflag']==0) | (cat['IRflag']==2) | (cat['IRflag']==4) | (cat['IRflag']==6)) & (cat['disp_Lflux']/cat['med_Lflux']<0.15) & (cat['disp_Nflux']/cat['med_Nflux']<0.30) & (cat['med_Lflux']>4.98) & (cat['med_Lflux']<5.01) )  #Faster
s3=np.where((cat['UDDL_est']<3) & (cat['CalFlag']==0) & (cat['med_Lflux']<20) & ((cat['IRflag']==0) | (cat['IRflag']==2) | (cat['IRflag']==4) | (cat['IRflag']==6)) & (cat['disp_Lflux']/cat['med_Lflux']<0.15) & (cat['disp_Nflux']/cat['med_Nflux']<0.30) & (cat['med_Lflux']>2.49) & (cat['med_Lflux']<2.51) )
s3=np.where((cat['UDDL_est']<3) & (cat['CalFlag']==0) & (cat['med_Lflux']<20) & ((cat['IRflag']==0) | (cat['IRflag']==2) | (cat['IRflag']==4) | (cat['IRflag']==6)) & (cat['disp_Lflux']/cat['med_Lflux']<0.15) & (cat['disp_Nflux']/cat['med_Nflux']<0.30) & (cat['med_Lflux']>0.695) & (cat['med_Lflux']<0.705) )
len(s3[0])  #40
DECs=cat['DEJ2000'][s3[0]]
RAs=cat['RAJ2000'][s3[0]]
DECs=DECs.reset_index()
RAs=RAs.reset_index()
selecteddec=[]
selectedra=[]
count=0
#Selecting here RA for October and DE for Paranal
for i in range(len(DECs)):
    #print(cat['DEJ2000'][s3[0][i]][:3])
    if((np.int(cat['DEJ2000'][s3[0][i]][:3])<27)&(np.int(cat['RAJ2000'][s3[0][i]][:2])<6)): #To observe in Oct
        FDF2.loc[count]=cat.iloc[s3[0][i]]  #Add row  s1[0][j]
        count=count+1
        print(count)
        #selectedra.append(cat['RAJ2000'][i])
        #selecteddec.append(DECs['DEJ2000'][i])

         
FDF2.to_pickle('LFLux10.pk')
FDF2.to_pickle('LFLux5.pk')
FDF2.to_pickle('LFLux2.5.pk')
FDF2.to_pickle('LFLux.7.pk')
      
FDF2.shape  #(372, 49)  Wiiiiii!!!
FDF2.to_pickle('FaintObsPropCals.pk')
np.max(FDF2['med_Lflux'])  # 133.89999389648438
         np.min(FDF2['med_Lflux'])# 14.889442443847656
         np.min(FDF2['UDDL_est'])  #0.7689999938011169
         np.max(FDF2['UDDL_est']) #2.984999895095825
         np.max(FDF2['med_Nflux'])   #19.399999618530273
         np.min(FDF2['med_Nflux'])  #4.070000171661377
         
http://morpheus.phys.lsu.edu/~gclayton/magnitude.html
         distance, teff_midi and gaia, SpType,
         RAJ2000, DEJ2000

s=df['Name'].str.contains('b Sgr')  #465851#W1Lmag=-1.4 F_nu = 1123.872 Jy / FluxK=202  DOESNT WORK
df['Name'][s]  #* b Sgr  93.80000305
df['nb_Lflux'][s] #4
         
mask = df.apply(lambda df: df.Name.str.contains('Sgr', na=False))
         print (mask)  #TEST
cat0=pd.read_pickle("/Users/M51/mscd-v1.pk")
s0=np.where(cat0['Name'].str.contains('b Sgr')
index 809 or 829 or some permutation


#NovemberData
df=wu.tableDRS('/Volumes/LaCie/NovemberData/All_AllBCDs/Iter1vis2','/Volumes/LaCie/NovemberData/All_AllBCDs/Iter1cf2',True)
df.shape  #(12, 62)
df.to_pickle('novCalsDRS.pk')
dDF1=wu.tableDRS('/Volumes/LaCie/DecemberData/All_AllBCDs/Iter1vis2','/Volumes/LaCie/DecemberData/All_AllBCDs/Iter1cf2',True)
            DF1=pd.read_pickle('decCalsDRS.pk')
            
wu.plotcf2DRS(DF1,'T0E','ICFLUX2',0,130,2) # p=2 cause cf is squared and we need the corr sq

#---------------------------------------------------------------------
from astropy.io import fits
catalog=fits.open('/Users/M51/msdfcc-v9.fits')
catalog[1].data.shape  # (465857,)
catalog[1].data.columns
cat=pd.DataFrame(catalog[1].data)
cat.keys()
            
#To observe in April in ATs
#1Jy
s1=np.where((cat['UDDL_est']<1) & (cat['CalFlag']==0) & (cat['IRflag']==0) & (cat['disp_Lflux']/cat['med_Lflux']<0.05) &(cat['med_Lflux']>0.95) &(cat['med_Lflux']<1.05)  )
len(s1[0])  #959
#2 Jy
s1=np.where((cat['UDDL_est']<1) & (cat['CalFlag']==0) & (cat['IRflag']==0) & (cat['disp_Lflux']/cat['med_Lflux']<0.05) &(cat['med_Lflux']>1.9) &(cat['med_Lflux']<2.1) )
len(s1[0])  #454
#5 Jy
s1=np.where((cat['UDDL_est']<1) & (cat['CalFlag']==0) & (cat['IRflag']==0) & (cat['disp_Lflux']/cat['med_Lflux']<0.05) &(cat['med_Lflux']>4.5) &(cat['med_Lflux']<5.5) )
s1=np.where((cat['UDDL_est']<1) & (cat['CalFlag']==0) & (cat['IRflag']==0) & (cat['disp_Lflux']/cat['med_Lflux']<0.05) &(cat['med_Lflux']>4.2) &(cat['med_Lflux']<5.8) ) #552 for North 2ndHalfNight
len(s1[0])  #336
len(cat['Name'][s1[0]])
Names=cat['Name'][s1[0]]
DECs=cat['DEJ2000'][s1[0]]
RAs=cat['RAJ2000'][s1[0]]
DECs=DECs.reset_index()
RAs=RAs.reset_index()
# ---- ----- ----
count=0
FDF=pd.DataFrame(columns=['Name', 'SpType', 'RAJ2000', 'DEJ2000', 'distance', 'teff_midi', 'teff_gaia', 'Comp', 'mean_sep', 'mag1', 'mag2', 'diam_midi', 'e_diam_midi', 'diam_cohen', 'e_diam_cohen', 'diam_gaia', 'LDD_meas', 'e_diam_meas', 'UDD_meas', 'band_meas', 'LDD_est', 'e_diam_est', 'UDDL_est', 'UDDM_est', 'UDDN_est', 'Jmag', 'Hmag', 'Kmag', 'W4mag', 'CalFlag', 'IRflag', 'nb_Lflux', 'med_Lflux', 'disp_Lflux', 'nb_Mflux', 'med_Mflux', 'disp_Mflux', 'nb_Nflux', 'med_Nflux', 'disp_Nflux', 'Lcorflux_30', 'Lcorflux_100', 'Lcorflux_130', 'Mcorflux_30', 'Mcorflux_100', 'Mcorflux_130', 'Ncorflux_30', 'Ncorflux_100', 'Ncorflux_130'])
#Selecting here RA for April 6 to 18 and DE for Paranal <27
#Using RA=10+-1 1st half / RA=17+-1  Second half of the night
#DEC in (-60,-40) South / DEC in (-10,10) North
#Lfluxes: 1, 2 and 5.
for i in range(len(DECs)):
    #print(cat['DEJ2000'][s1[0][i]][:3])
    #if((np.int(cat['DEJ2000'][s1[0][i]][:3])>-60)& (np.int(cat['DEJ2000'][s1[0][i]][:3])<-40)& (np.int(cat['RAJ2000'][s1[0][i]][:2])>10)&(np.int(cat['RAJ2000'][s1[0][i]][:2])<12)): #South 1stHalfNight  6,4*s
    #if((np.int(cat['DEJ2000'][s1[0][i]][:3])>-60)& (np.int(cat['DEJ2000'][s1[0][i]][:3])<-40)& (np.int(cat['RAJ2000'][s1[0][i]][:2])>16)&(np.int(cat['RAJ2000'][s1[0][i]][:2])<18)): #South 2ndtHalfNight 2*s
    #if((np.int(cat['DEJ2000'][s1[0][i]][:3])>-10)& (np.int(cat['DEJ2000'][s1[0][i]][:3])<10)& (np.int(cat['RAJ2000'][s1[0][i]][:2])>10)&(np.int(cat['RAJ2000'][s1[0][i]][:2])<12)): #North 1stHalfNight
    if((np.int(cat['DEJ2000'][s1[0][i]][:3])>-10)& (np.int(cat['DEJ2000'][s1[0][i]][:3])<10)& (np.int(cat['RAJ2000'][s1[0][i]][:2])>16)&(np.int(cat['RAJ2000'][s1[0][i]][:2])<18)): #North 2ndHalfNight
            print('Name',cat['Name'][s1[0][i]],'DE',np.int(cat['DEJ2000'][s1[0][i]][:3]),'RA',np.int(cat['RAJ2000'][s1[0][i]][:2]))
            FDF.loc[count]=cat.iloc[s1[0][i]]  #Add row  s1[0][j]
            count=count+1

FDF=FDF.reset_index()
# Selecting the 1st star from each set
idxa=[0,6,8,13,15,19,21,23,25,27,28,29]
Fin=pd.DataFrame(columns=['Name', 'SpType', 'RAJ2000', 'DEJ2000', 'distance', 'teff_midi', 'teff_gaia', 'Comp', 'mean_sep', 'mag1', 'mag2', 'diam_midi', 'e_diam_midi', 'diam_cohen', 'e_diam_cohen', 'diam_gaia', 'LDD_meas', 'e_diam_meas', 'UDD_meas', 'band_meas', 'LDD_est', 'e_diam_est', 'UDDL_est', 'UDDM_est', 'UDDN_est', 'Jmag', 'Hmag', 'Kmag', 'W4mag', 'CalFlag', 'IRflag', 'nb_Lflux', 'med_Lflux', 'disp_Lflux', 'nb_Mflux', 'med_Mflux', 'disp_Mflux', 'nb_Nflux', 'med_Nflux', 'disp_Nflux', 'Lcorflux_30', 'Lcorflux_100', 'Lcorflux_130', 'Mcorflux_30', 'Mcorflux_100', 'Mcorflux_130', 'Ncorflux_30', 'Ncorflux_100', 'Ncorflux_130'])
for i in range(len(idxa)):
    Fin.loc[i]=FDF.iloc[idxa[i]]
#L flux = 1 Jy:
            Name HD  95964                         DE -41 RA 11
            Name HD  99616                         DE -56 RA 11
            Name HD  97756                         DE -41 RA 11
            Name HD 103385                         DE -55 RA 11
            Name HD 100898                         DE -54 RA 11
            Name HD 303852                         DE -57 RA 11
            
            Name HD 160355                         DE -47 RA 17
            Name HD 161784                         DE -54 RA 17
            
            Name HD  96272                         DE 7 RA 11
            Name HD  95662                         DE 0 RA 11
            Name HD 102195                         DE 2 RA 11
            Name HD 100314                         DE -5 RA 11
            Name HD  96627                         DE 0 RA 11
            
            Name HD 154145                         DE 0 RA 17
            Name HD 158737                         DE 1 RA 17
            
#L flux = 2 Jy:
            Name HD 103975                         DE -47 RA 11
            Name HD  97702                         DE -43 RA 11
            Name HD  97594                         DE -52 RA 11
            Name HD  98802                         DE -48 RA 11
            
            Name HD 162931                         DE -44 RA 17
            Name HD 157835                         DE -54 RA 17

            Name HD  99166                         DE 2 RA 11
            Name HD  99789                         DE 1 RA 11
            
            Name HD 154929                         DE 8 RA 17
            Name HD 158049                         DE 2 RA 17
#L flux = 5 Jy:
            Name HD 101927                         DE -52 RA 11
            Name HD  97397                         DE -59 RA 11
            
            Name * sig Ara                         DE -46 RA 17
            
            Name HD 101956                         DE -9 RA 11
            
            Name HD 160823                         DE 4 RA 17
            
count  #2
FDF.shape  #2,49
FDF.to_pickle('CalibratorsApril.pk')  #30,49
Fin.shape #12,49 :D
Fin.to_pickle('FinalApril.pk')

NiceFin = Fin[['Name','RAJ2000','DEJ2000','med_Lflux','UDDL_est']]
NiceFin.to_pickle('FinalAprilShort.pk')

Name               RAJ2000      DEJ2000       med_Lflux  UDDL_est Notes
med_Lflux~1Jy
0   HD  95964  11:03:38.8491  -41:06:19.710   1.014922     0.263 South/1st half night
1   HD 160355  17:41:50.3499  -47:30:44.870   0.995836     0.229 South/2nd half night
2   HD  96272  11:06:09.4042  +07:08:23.108   1.024911     0.232 North/1st half night
3   HD 154145  17:03:41.4442  -00:08:45.425   1.049715     0.200 North/2nd half night
med_Lflux~2Jy
4   HD 103975  11:58:15.6077  -47:58:37.559   2.041982     0.338 South/1st half night
5   HD 162931  17:55:44.1975  -44:55:48.355   1.994520     0.398 South/2nd half night
6   HD  99166  11:24:45.1121  +02:05:47.552   2.061082     0.383 North/1st half night
7   HD 154929  17:08:11.7852  +08:09:40.402   1.997267     0.393 North/2nd half night
med_Lflux~5Jy
8   HD 101927  11:43:33.3618  -52:37:10.775   4.541821     0.559 South/1st half night
9   * sig Ara  17:35:39.5896  -46:30:20.462   5.145337     0.376 South/2nd half night
10  HD 101956  11:43:55.0901  -09:07:56.665   5.046773     0.580 North/1st half night
11  HD 160823  17:41:55.8491  +04:22:00.468   4.329684     0.523 North/2nd half night


http://catserver.ing.iac.es/staralt/
Carte du Cel
iObserve
Planetarium for linux
            HD95964  11:03:38.8491  -41:06:19.710
            HD160355  17:41:50.3499  -47:30:44.870
            HD96272  11:06:09.4042  +07:08:23.108
            HD154145  17:03:41.4442  -00:08:45.425
            HD103975  11:58:15.6077  -47:58:37.559
            HD162931  17:55:44.1975  -44:55:48.355
            HD99166  11:24:45.1121  +02:05:47.552
            HD154929  17:08:11.7852  +08:09:40.402
            HD101927  11:43:33.3618  -52:37:10.775
            sigAra  17:35:39.5896  -46:30:20.462
            HD101956  11:43:55.0901  -09:07:56.665
            HD160823  17:41:55.8491  +04:22:00.468
