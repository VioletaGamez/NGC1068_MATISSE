import os
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle
import glob 
import pandas as pd
import numpy as np
import scipy
from mcdb import fitsutil as fu
from mcdb import matutil  as mu
from mcdb import VioletasExp  as ve
from astropy.io  import fits

def mpause(interval):
   figman = plt.get_current_fig_manager()
   canvas = figman.canvas
   if(canvas.figure.stale): canvas.draw()
   canvas.start_event_loop(interval)
   return

def sglob(x):
   return sort(glob.glob(x))
def msave(file, value):
   sf = open(file,'wb')
   p  = pickle.Pickler(sf)
   p.dump(value)
   sf.close()
   del p
   return
def mrestore(file):
   sf = open(file,'rb')
   p  = pickle.Unpickler(sf)
   z  = p.load()
   sf.close()
   del p
   return z
def pwd():
   return os.getcwd()
def cd(file):
   os.chdir(file)
   return

def wtv(a, **kwargs):
   plt.clf()
   b=plt.imshow(a, origin='lower', **kwargs)

def plot(*args):
   plt.clf()
   if (len(args)==1): a=plt.plot(args[0])
   else:a=plt.plot(args[0],args[1])
   return a

def plotm(mask, *args):
   if (len(args)==1):
      a = plt.plot(args[0][mask])
   else:a = plt.plot(args[0][mask], args[1][mask])
   return a
def oplot(*args):
   if (len(args)==1): a=plt.plot(args[0])
   else:a=plt.plot(args[0],args[1])
   return

def max2(data): #  indices of maximum value
   return np.unravel_index(data.argmax(), data.shape)

def min2(data): #  indices of maximum value
   return np.unravel_index(data.argmin(), data.shape)

def bcdsort(data, bcd):
   return data[ve.bcdStdOrder(bcd)]

def tplfn(f, directory=None):
   if directory is not None:
      here = os.getcwd()
      os.chdir(directory)
   pk  = glob.glob('*.tpl.pk')
   out = []
   for p in pk:
      g = mrestore(p)
      for gg in g:
         bcd = gg['bcd']
#        out.append(bcdsort(f(gg), bcd))
         out.append(f(gg))
   out = np.array(out)
   out = out.flatten()
   if (directory is not None):os.chdir(here)
   return out

def tpladdkey(tpd, key, base='../', fill=None, setkey=True):
   values = []
   nfile  = len(tpd['file'])//6
   for i in range(nfile):
      j = 6*i
      v = fu.getkeyword(key, base+tpd['file'][j], fill=fill)
      for k in range(6):values.append(v)
   values = np.array(values)
   if (setkey):
      tpd[key] = values
      return 
   else:return values

def tplkey(keyword, directory=None, base='./',fill=None):
   if directory is not None:
      here = os.getcwd()
      os.chdir(directory)
   pk  = glob.glob('*.tpl.pk')
   out = []
   for p in pk:
      g = mrestore(p)
      for gg in g:
         file = p[0:23]+'/'+gg['file']
         kv   = fu.getkeyword(keyword,base+file,fill=fill)
         for i in range(6):out.append(kv)
   out = np.array(out)
   if (directory is not None):os.chdir(here)
   return out

def tplhkey(keyword, directory=None,newkeys=False):
   if directory is not None:
      here = os.getcwd()
      os.chdir(directory)
   pk  = glob.glob('*.tpl.pk')
   out = []
   for p in pk:
      g = mrestore(p)
      for gg in g:
         if (newkeys):
            print('new keys not implemented in tplhkey)')
            return None
         kv   = gg['header'][keyword]
         for i in range(6):out.append(kv)
   out = np.array(out)
   if (directory is not None):os.chdir(here)
   return out
def lflux(g):
  wave = g['wave']
  a    = np.arange(len(wave))[(wave<3.8) & (wave > 3.2)]
  x1   = np.min(a)
  x2   = np.max(a)
  return np.mean(abs(g['flux'][:,x1:x2]),1)
def lvis(g):
  wave = g['wave']
  a    = np.arange(len(wave))[(wave<3.8) & (wave > 3.2)]
  x1   = np.min(a)
  x2   = np.max(a)
  return np.mean(abs(g['vis'][:,x1:x2]),1)
def lphot(g):
   p = np.mean(g['phot'][:,25:35],1)
   return(p[0:6])
def lbase(g):return(g['pbl'])
def lflag(g):return(g['fflag'])
def lbflag(g):return(g['baseflag'])
def lra(g):
   ras = []
   for z in range(6):ras.append(g['ra'])
   return np.array(ras)
def ldec(g):
   dec = []
   for z in range(6):dec.append(g['dec'])
   return np.array(dec)
def ltau(g):
   tau = []
   for z in range(6):tau.append(g['tau'])
   return np.array(tau)
def lfile(g):
   file = []
   for z in range(6):file.append(g['file'])
   return np.array(file)
def lsky(g):
   sky = []
   for z in range(6):sky.append(g['sky'])
   return np.array(sky)
def lmjd(g):
   mjd=[]
   for z in range(6):mjd.append(g['mjd-obs'])
   return np.array(mjd)
def lchan(g):return np.arange(0,6)
def ropd(g):
   opd = g['opd']
   for b in range(6):opd[b]-= 0.5*(np.roll(opd[b],1)+np.roll(opd[b],-1))
   return np.mean(np.abs(opd),1)
def lbcd(g):
   bcd1   = g['header']['HIERARCH ESO INS BCD1 NAME']
   bcd2   = g['header']['HIERARCH ESO INS BCD2 NAME']
   b      = 0
   if (bcd1 == 'OUT'): b=2
   if (bcd2 == 'OUT'): b+=1
   bcds  = []
   for i in range(6):bcds.append(b)
   return np.array(bcds)
def ltarg(g):
   targ = []
   t = g['header']['HIERARCH ESO OBS TARG NAME']
   for i in range(6):targ.append(t)
   return np.array(targ)

def lnaomi(g):
   gh=g['header']
   ah = 'ESO ISS AT1 NAOMI WFSRMS START'
   out = np.zeros(6)
   if gh.__contains__(ah):
      wfsrms=[gh[ah]]
      for i in range(2,5):wfsrms.append(gh[ah.replace('AT1','AT'+str(i))])
      wfsrms = np.array(wfsrms)
      pb     = mu.photBeam2Int()
      for i in range(6):
         out[i] = np.square(wfsrms[pb[i][0]])+np.square(wfsrms[pb[i][1]])
   return out
def tpldata(d=None):
   out = {}
   out['mjd']   = tplfn(lmjd,directory=d)
   out['chan']  = tplfn(lchan,directory=d)
   out['flux']  = tplfn(lflux,directory=d)
   out['vis']   = tplfn(lvis,directory=d)
   out['phot']  = tplfn(lphot,directory=d)
   out['base']  = tplfn(lbase,directory=d)
   out['tau']   = tplfn(ltau,directory=d)
   out['ropd']  = tplfn(ropd,directory=d)
   out['bcds']  = tplfn(lbcd,directory=d)
   out['file']  = tplfn(lfile,directory=d)
   out['sky']   = tplfn(lsky,directory=d)
   out['targ']  = tplfn(ltarg,directory=d)
   out['ra']    = tplfn(lra,directory=d)
   out['dec']   = tplfn(ldec,directory=d)
   out['naomi'] = tplfn(lnaomi,directory=d)
   out['flag']  = tplfn(lflag,directory=d)
   out['bflag'] = tplfn(lbflag,directory=d)
   out['airy']  = np.ones(len(out['tau']))
   out['jy']    = np.ones(len(out['tau']))
   out['l']     = np.zeros(len(out['tau']))
   out['diam']  = .001*np.ones(len(out['tau']))
   tpls = []
   seqn = []
   valid= []
   if (d is None):pk = glob.glob('*.tpl.pk')
   else:pk = glob.glob(d+'/'+'*.tpl.pk')
   nexp = 0
   for p in pk:
      g = mrestore(p)
      s = 0
      for gg in g:
         for i in range(6):
            seqn.append(s)
            tpls.append(p)
            valid.append(True)
         s+=1
         nexp+=1
   out['tpls'] = np.array(tpls)
   out['seqn'] = np.array(seqn)
   out['valid'] = np.array(valid)
#  out['nexp'] = nexp
   return out

def setflux(tpd, targ, flux):
   tt=(tpd['targ'] == targ)
   tpd['jy'][tt] = flux
def setdiam(tpd, targ, diam):
   tt=(tpd['targ'] == targ)
   tpd['diam'][tt] = diam

def setairy(tpd):
   tpd['tf'] = tpd['flux'].copy()
   tpd['tv'] = tpd['vis'].copy()
   for i in range(len(tpd['targ'])):
      tpd['airy'][i]  = mu.airyBase(tpd['diam'][i], 3.58, tpd['base'][i])
      tpd['tf'][i]    = tpd['flux'][i]/tpd['airy'][i]/tpd['jy'][i]
      tpd['tv'][i]    = tpd['vis'][i]/tpd['airy'][i]

def zcplot(xx,yy,zz,zmin=None, zmax=None, marker='o', over=False, select=None,
   cb = True):
   h24 = cm.get_cmap('hsv',24)
   if (select is not None):
      x = xx[select]
      y = yy[select]
      z = zz[select]
   else:
      x = xx.copy()
      y = yy.copy()
      z = zz.copy()
   
   if (zmin is None): zmin = np.min(z)
   if (zmax is None): zmax = np.max(z)
   if (zmax>zmin):
      zs = 0.7*(z-zmin)/(zmax-zmin)
      zs = np.clip(zs,0,1.)
   else:zs=np.zeros(len(z))
   if False==over: plt.clf()
   for i in range(len(x)):
      a=plt.plot(x[i], y[i], marker=marker, color=h24(zs[i]))
   if (cb):
     smin = str(zmin)[0:5]
     smax = str(zmax)[0:5]
     ymax = max(y)
     ymin = min(y)
     xmin = min(x)
     xmax = max(x)
     a=plt.text(xmin - .05*(xmax-xmin), ymax, smin)
     a=plt.text(xmin + 0.9*(xmax-xmin), ymax, smax)
     if(over==False):
        for i in range(10):
           a=plt.plot(xmin+.1*i*(xmax-xmin), ymax-.05*(ymax-ymin), 
              marker=marker, color=h24(.07*i))
   return

def zsplot(x,y,z,zmin=None, zmax=None, marker='o',over=False):
   if (zmin is None): zmin = np.min(z)
   if (zmax is None): zmax = np.max(z)
   zs = (z-zmin)/(zmax-zmin)
   if False==over: plt.clf()
   for i in range(len(x)):
      plt.plot(x[i], y[i], linestyle='', marker=marker, markersize=1+10*zs[i])
   return

def zmplot(x,y,z,zmin=None, zmax=None,over=False ):
   if (zmin is None): zmin = np.min(z)
   if (zmax is None): zmax = np.max(z)
   zs = 10*(z-zmin)/(zmax-zmin)
   zs = zs.clip(0, 10)
   zs = zs.astype(int)
   if False==over: plt.clf()
   for i in range(len(x)):
      plt.plot(x[i], y[i], linestyle='', marker=zs[i], markersize=6, color='r')
   return
def linkpk(directory):
   pks = glob.glob('*.tpl.pk')
   for p in pks:
      os.link(p,directory+'/'+p)

def meanchan(tpd, tag='tv'):
   lp  = len(tpd['base'])
   cc  = np.zeros((6,2))
   ccc = np.ones(lp)
   ch  = tpd['chan']
   bcd = tpd['bcds']
   tau = tpd['tau']
   tau3= tau > 3
   bc  = [0,3]
   for b in range(2):
      for c in range(6):
         tc = (ch==c) & (bcd==bc[b]) & tau3
         cc[c,b] = np.mean(tpd[tag][tc])
#        elif (v==2):cc[c,b]= np.mean(tpd['tv2'][tc])
   cc = cc/np.mean(cc)
#  print(cc)
   for b in range(2):
      for c in range(6):
         tc = (ch==c) & (bcd==bc[b])
         ccc[tc] = cc[c,b]
   return ccc
      
def chcorr(tpd):
   tpd['tv']/=meanchan(tpd,'tv')
   tpd['tf']/=meanchan(tpd,'tf')
   tpd['flux']/=meanchan(tpd,'flux')
   tpd['vis']/=meanchan(tpd,'vis')
   tpd['phot']/=meanchan(tpd,'phot')

def opdcorr(tpd,fix=7.):
   cc = np.exp(tpd['ropd']/fix)
   tpd['tv']*=cc
   tpd['tf']*=cc
   tpd['flux']*=cc

def readcaltable(file=None):
   if file is None:file='/disks/cool1/jaffe/drs/deccal.txt'
   names=[]
   ras=[]
   decs=[]
   kmag=[]
   lflux=[]
   nflux=[]
   diam=[]
   fin = open(file)
   line='x'
   while(len(line) > 0):
      line = fin.readline()
      if (len(line) > 0):
#        print(line)
         names.append(line[:9])
         ra=line[11:22]
         ra = ra.split(':')
         ra = np.array(ra)
         ra = ra.astype(float)
         ra = ra[0]*15 + ra[1]/4 + ra[2]/240
         ds=line[23]
         dec=line[24:34]
         dec = dec.split(':')
         dec = np.array(dec)
         dec = dec.astype(float)
         dec = dec[0] + dec[1]/60 + dec[2]/3600
         if (ds=='-'):dec = -dec
         k=float(line[37:42])
         l=float(line[45:51])
         n=float(line[53:57])
         d=float(line[61:65])
         ras.append(ra)
         decs.append(dec)
         kmag.append(k)
         lflux.append(l)
         nflux.append(n)
         diam.append(d)
   fin.close()
   out={}
   out['name'] = np.array(names)
   out['ra']   = np.array(ras)
   out['dec']  = np.array(decs)
   out['k']    = np.array(kmag)
   out['l']    = np.array(lflux)
   out['n']    = np.array(nflux)
   out['diam'] = np.array(diam)
   return out

def processAllDir(dirlist, outdir):
   here = os.getcwd()
   for dir in dirlist:
      os.chdir(dir)
      ve.processAllTemplates()
      linkpk(outdir)
      os.chdir(here)
   os.chdir(outdir)
   tpd = tpldata()
   msave('tpldata.pk',tpd)
   os.chdir(here)

def setcaldata(tpd, calfile=None):
   caltable=readcaltable(calfile)
   dec = tpd['dec']
   cdec= caltable['dec']
   ra  = tpd['ra']
   cra = caltable['ra']
   tpd['k'] = tpd['jy'].copy()
   for i in range(len(dec)):
      d = dec[i]
      r = ra[i]
      for j in range(len(cdec)):
         dd = np.square(d-cdec[j]) + np.square(np.cos(d/2/np.pi)*(r-cra[j]))
         if (np.sqrt(dd) < (10/3600.)):
#           tpd['jy'][i]   = caltable['l'][j]
            tpd['jy'][i]   = 253*10.**(-0.4*caltable['k'][j])
            tpd['k'][i]   = caltable['k'][j]
            tpd['diam'][i] = caltable['diam'][j]
   setairy(tpd)
   chcorr(tpd)

def renameKeywords(header):
       """
          If there is an array of FITS headers stored internally in
             self.headerArray, convert the keywords to a more
             easily typed format.  If the database object is known
             as "mt"(for instance), the current keyword list can be
             accessed as kwds = mt.headerArray.columns.

          The ESO keywords are converted by the following algorithm:
             Single (legitimate FITS) keywords are converted to lower case.
             HIERARCH ESO keywords:  
                The HIERARCH ESO characters are dropped.
                If only two words remain these are kept, otherwise one
                   more word is dropped
                The remaining words are concatenated and converted to lower case
       """
       khead = header.keys()
       for key in khead:
          lkey = key.lower()
          c = lkey.split(' ')
          if (len(c) > 1):   
             if(len(c) <= 3): c = ''.join(c[1:])
             else: c = ''.join(c[2:])
#            exception for worthless DET CHOP keywords
             if (len(c)>=3):
                if ((c[1]=='det') & (c[2]=='chop')):
                   c='det'+c
          header.rename_keyword(key, c[:8],force=True)

#        convert violeta's data frames into local tables
def v6(tpv, colname):
   v = tpv[colname]
   vs = len(v)
   vshape = v[0].shape
   vlen = len(vshape)
   wave = tpv['wave'][0]
   lwave= len(wave)
   wrange = (wave < 3.8) & (wave > 3.2)
   out  = []
#  print(colname, vshape)
   if(len(vshape)==0):
      for vv in v:
          for i in range(6):out.append(vv)
      out=np.array(out)
   elif ((vshape[0]==6) | (vshape[0]==10)):
      if (vlen==1):
         for vv in v:out.append(vv)
         out=np.array(out).flatten()
      if (vlen==2):
         if (vshape[1]==lwave):
            for vv in v:
               if vv.dtype==complex:vv=abs(vv)
               for i in range(6):out.append(np.mean(vv[i][wrange]))
            out=np.array(out)
   return(out)

def tpgdata(tpg):  # make summaries of the data in violeta's pickle file
   out = {}
   out['flux']  = v6(tpg,'flux')  # average abs flux over wavelrange
   outlen = len(out['flux'])      # length of summary arrays
   out6   = outlen//6             # # of exposures (6 channels per exposure)
   out['vis']   = v6(tpg,'vis')   # average visibilities (EWS)
   out['vis2']  = v6(tpg,'VIS2')   # average visibilities^2 (DRS)
   out['flux2']=  v6(tpg,'ICFLUX2') # average correlated flux^2 (drs)
   out['wfsrms']= v6(tpg,'wfsrms')  # NAOMI wave front data
   out['valid'] = np.ones(outlen,dtype=bool) # initially all exposures valid
   bcd1   = tpg['BCD1']
   bcd2   = tpg['BCD2']
   out['bcds'] = np.zeros(outlen,dtype=int) # encoded bcd setting
   for i in range(out6):
      b      = 0
      if (bcd1[i] == 'OUT'): b=2
      if (bcd2[i] == 'OUT'): b+=1
      for j in range(6):out['bcds'][6*i+j]=b
   out['tpls'] = np.zeros(outlen,dtype='O')
   for i in range(out6):
      for j in range(6):out['tpls'][6*i+j]=tpg['tplstart'][i]
   out['chan']  = np.zeros(outlen,dtype=int)
   for i in range(out6):out['chan'][6*i:6*i+6]=np.arange(0,6)
   out['phot']  = v6(tpg,'phot')  # averaged photometry over wavelength range
   out['base']  = v6(tpg,'pbl')   # baselines
#                    sort baselines for bcds
   for i in range(out6):
      a = 6*i
      b = a+6
      bb = out['base'][a:b]
      nb = np.zeros(6,dtype=int)
      bc = out['bcds'][a]
      vb = ve.bcdStdOrder(bc)
      for j in range(6):nb[j] = bb[vb[j]]
      out['base'][a:b] = nb
   out['tau']   = v6(tpg,'tau')   # DIMM correlation time
#  out['ropd']  = tplfn(ropd)
   out['ra']    = v6(tpg,'ra')
   out['dec']   = v6(tpg,'dec')
#  out['naomi'] = tplfn(lnaomi)
   out['mjd']   = np.zeros(outlen)  # mjd of initial frame in exposure
   for i in range(out6):
      for j in range(6):out['mjd'][6*i+j] = tpg['header'][i]['MJD-OBS']
   out['targ']  = np.zeros(outlen,dtype='O') # target name
   for i in range(out6):
      for j in range(6):out['targ'][6*i+j] = tpg['targ'][i]
   out['l']     = v6(tpg,'LFlux') # lband flux (Jy) in pierre's catalog
   out['jy']    = out['l'].copy()
   out['diam']  = v6(tpg,'UDDL')  # diameter (mas) in pierre's catalog
   out['airy']  = np.ones(outlen) # loss of visibility due to resolution
#             correct for resolution
   out['tf']    = out['flux'].copy()# corr. flux corrected for resolution
   out['tv']    = out['vis'].copy()  # vis. corrected for resolution
   out['tv2']   = out['vis2'].copy()  # vis^2. corrected for resolution
   for i in range(outlen):
      out['airy'][i]  = mu.airyBase(out['diam'][i], 3.58, out['base'][i])
   out['tf']    /= (out['airy']*out['l'])
   out['tv']    /= out['airy']
   out['tv2']   /= np.square(out['airy'])
   out['flux2']   /= np.square(out['airy'])
#            correct for channel biases
   out['tf']/=meanchan(out,'tf')
   out['tv']/=meanchan(out,'tv')
   out['flux']/=meanchan(out,'flux')
   out['tv2']/=meanchan(out,'tv2')
   out['flux2']/=meanchan(out,'flux2')
   return out

def addNaomi(tpg):
   naomi=[]
   ah = 'ESO ISS AT1 NAOMI WFSRMS START'
   pb = mu.photBeam2Int()
   for h in tpg['header']:
      out = np.zeros(6)
      if h.__contains__(ah):
         wfsrms=[h[ah]]
         for i in range(2,5):wfsrms.append(h[ah.replace('AT1','AT'+str(i))])
         wfsrms = np.array(wfsrms)
         for i in range(6):
            out[i] = np.square(wfsrms[pb[i][0]])+np.square(wfsrms[pb[i][1]])
      naomi.append(out)
      n=tpg.join(pd.Series(naomi,name='wfsrms'))
   return n
def addValid(tpg):
   n = tpg.join(pd.Series(np.ones(len(tpg)), name='valid'))
   return n

def diffhead(h1, h2):
   diff = []
   nokey = ['SENS', 'CLDC', 'INS C',' END', ' ENC']
   keys = h1.keys()
   for k in keys:
      dok = len(k)>0
      for n in nokey:dok = dok & (n not in k)
      if (dok) :
         v1 = h1.get(k)
         v2 = h2.get(k, "NONE2")
         if (v1 != v2):
            print(k, v1, v2)
            diff.append (k+' '+str(v1)+' '+str(v2))
      keys = h2.keys()
   for k in keys:
      dok = len(k)>0
      for n in nokey: dok = dok & (n not in k)
      if (dok):
         v1 = h1.get(k, "NONE1")
         if (v1=="NONE1"):
            v2 = h2[k]
            print(k,v2,v1)
            diff.append (k+' '+str(v1)+' '+str(v2))
   return diff

def setFlag(tpd, key, testvalue, flag=False, test='=='):
   if (test=='=='):tpd['valid'][tpd[key]==testvalue] = flag
   elif (test=='!='):tpd['valid'][tpd[key]!=testvalue] = flag
   elif (test=='>'):tpd['valid'][tpd[key]>testvalue] = flag
   elif (test=='<'):tpd['valid'][tpd[key]<testvalue] = flag
   return

def tplphot(tpl,wt=.25):
   here = pwd()
   cd(tpl)
   print('tpl=',tpl)
   files = np.sort(glob.glob('*.fits'))
   band  = mu.getBand(files[0])
   out   = []
   if (band=='N'):
      for f in files:
         chop = fu.getkeyword('chopfreq', f, fill=0.)
         if (chop > .2):
            data = fu.getData(f, 'DATA5').astype(float)
            tt   = fu.getData(f, 'TARTYP')
            targ = np.mean(data[tt=='T'],0) - np.mean(data[tt=='S'],0)
            sdata = data.shape
            apoim = mu.apoIntImage(sdata[2], sdata[1], band)
            spec  = np.sum(targ*apoim, 1)
            flux  = np.sum(spec)
            bcd   = [fu.getkeyword('bcd1name',f), fu.getkeyword('bcd2name',f)]
            shut  = []
            for i in range(4):shut.append(fu.getkeyword('bsn'+str(i+1)+'st',f))
            sfil  = fu.getkeyword('sfnname',f)
   else:
      i = 0
      windows = ['DATA9','DATA10','DATA12','DATA13']
      for f in files:
         chop = fu.getkeyword('chopfreq', f, fill=0.)
         if (chop > .2):
            i = i+1
            tt   = fu.getData(f, 'TARTYP')
            spec = []
            flux = []
            for w,j in zip(windows, np.arange(4)):
               data  = fu.getData(f, w).astype(float)
               targ  = np.mean(data[tt=='T'],0) - np.mean(data[tt=='S'],0)
               plt.clf()
               plt.imshow(targ)
               plt.pause(wt)
               sdata = data.shape
               apoim = mu.apoimage(sdata[2], sdata[1])
               plt.clf()
               plt.imshow(targ*apoim[j])
               plt.pause(wt)
               s = np.sum(targ*apoim[j],1)
               plt.clf()
               plt.plot(s)
               spec.append(s)
               plt.title(str(i)+' '+str(j))
               plt.pause(wt)
               flux.append(np.sum(spec))
            i +=1
            bcd   = [fu.getkeyword('bcd1name',f), fu.getkeyword('bcd2name',f)]
            sfil  = fu.getkeyword('sflname',f)
            shut  = []
            for j in range(4):shut.append(fu.getkeyword('bsl'+str(j+1)+'st',f))
            stations = []
            for j in range(4):stations.append(fu.getkeyword('confstation'+str(j+1),f))
            airm = fu.getkeyword('airmstart',f)
            tau  = fu.getkeyword('ambitau0start',f)
            out.append({'tpl':tpl, 'file':f, 'bcd':bcd, 'shut':shut, 'sfil':sfil,
               'stations':stations,'spec':spec, 'flux':flux, 'airm':airm, 'tau':tau})
   cd(here)
   return out

def writeOIfits(data, templatefile, outfile, v2err=None, t3err=None):
   pb = mu.photBeam2Int()  # telescope order
   c3 = mu.set3comb()      # triplet order
   outlist = []
   tfile = fits.open(templatefile)
   dfile = fits.open('../'+data[0]['file'])
   wave  = data[0]['wave']
   tfile[0].header = dfile[0].header
   for i in range(2):outlist.append(tfile[i])
   arraynew = tfile['oi_array'].copy()
   arrayold = dfile['array_geometry']
   arraynew.data['TEL_NAME']  = arrayold.data['TEL_NAME']
   arraynew.data['STA_NAME']  = arrayold.data['STA_NAME']
   arraynew.data['STA_INDEX'] = arrayold.data['STA_INDEX']
   arraynew.data['DIAMETER']  = arrayold.data['DIAMETER']
   arraynew.data['STAXYZ']    = arrayold.data['STAXYZ']
   arraynew.data['fov']       = np.ones(len(arraynew.data))
   arraynew.data['fovtype']   = 'RADIUS'
   tel = arraynew.data['TEL_NAME'][0]
   ut  = tel[0]=='U'
   if (ut): stations = {'U1':32, 'U2':33, 'U3':34, 'U4':35}
   else:
      stations = {}
      for n,s in zip(arrayold.data['STA_NAME'],arrayold.data['STA_INDEX']):
         stations[n] = s
   outlist.append(arraynew)

   wnew = fits.BinTableHDU.from_columns(tfile['OI_WAVELENGTH'].columns, nrows=len(wave),
      name='OI_WAVELENGTH')
   wnew.data['EFF_WAVE'] = wave/1.e6
   wnew.data['EFF_BAND'][:] = 3.5e-8
   outlist.append(wnew)

   v2new = fits.BinTableHDU.from_columns(tfile['OI_VIS2'].columns, 
      nrows=6*len(data), name='OI_VIS2')
   v2new.data['target_id'] = 1
   v2new.data['int_time'] = .111
   for i in range(len(data)):
      v2new.data['mjd'][6*i:(6*i+6)]      = data[i]['mjd-obs']
      v2new.data['vis2data'][6*i:(6*i+6)] = np.mean(data[i]['vis2'],0)
#                    fudging vis errors
      v2new.data['vis2err'][6*i:(6*i+6)] = np.ones((6,len(wave)))
      if (v2err is not None):  # ad hoc specification of v2errors
         for b in range(6): v2new.data['vis2err'][6*i+b] = v2err[b]
      v2new.data['ucoord'][6*i:(6*i+6)] = data[i]['uvcoord'][:,0]
      v2new.data['vcoord'][6*i:(6*i+6)] = data[i]['uvcoord'][:,1]
      v2new.data['flag'][6*i:(6*i+6)]   = np.zeros((6,len(wave)))  # flags all false
      for b in range(6):  # station indices
         v2new.data['sta_index'][6*i+b][0] = stations[data[i]['stations'][pb[b,0]]]
         v2new.data['sta_index'][6*i+b][1] = stations[data[i]['stations'][pb[b,1]]]
   t3new = fits.BinTableHDU.from_columns(tfile['OI_T3'].columns, nrows=4*len(data),
      name='OI_T3')
   t3new.data['target_id'] = 1
   t3new.data['int_time'] = .111
   t3new.data['t3amp']    = 1.
   t3new.data['t3amperr'] = .1
   t3new.data['t3phierr'] = 4
   for i in range(len(data)):
      t3new.data['mjd'][4*i:(4*i+4)]   = data[i]['mjd-obs']
      t3new.data['t3phi'][4*i:(4*i+4)] = np.mean(data[i]['t3'],0)
      t3new.data['u1coord'][4*i:(4*i+4)] = data[i]['uvcoord1'][:,0]
      t3new.data['v1coord'][4*i:(4*i+4)] = data[i]['uvcoord1'][:,1]
      t3new.data['u2coord'][4*i:(4*i+4)] = data[i]['uvcoord2'][:,0]
      t3new.data['v2coord'][4*i:(4*i+4)] = data[i]['uvcoord2'][:,1]
      for b in range(4): #         station indices of triplets
         s = np.zeros(3,int)
#        beams = c3[b]   #         matisse beam pairs in triplet b
         tels  = pb[c3[b]]    # telescope pairs in this triplet
#                          order them in ascending telescope numbers
         if (tels[0,0] < tels[1,0]): 
            t1 = tels[0,0]
            t2 = tels[0,1]
            t3 = tels[1,1]
         else:
            t1 = tels[1,0]
            t2 = tels[1,1]
            t3 = tels[0,1]
         s[0] = stations[data[i]['stations'][t1]]
         s[1] = stations[data[i]['stations'][t2]]
         s[2] = stations[data[i]['stations'][t3]]
         t3new.data['sta_index'][4*i+b] = s
      t3new.data['flag'][4*i:(4*i+4)]   = np.zeros((4,len(wave)))  # flags all false
      t3new.data['t3phierr'][4*i:(4*i+4)]   = 2*np.ones((4,len(wave)))  # 2 degree rms error
      if (t3err is not None):  # ad hoc specification of t3errors
         for b in range(4):t3new.data['t3phierr']*=t3err[b]/2
   outlist.append(v2new)
   outlist.append(t3new)
   outfits = fits.HDUList(outlist)
   outfits.writeto(outfile)
   tfile.close()
   dfile.close()

def fsmooth1(data, sigma):
   return scipy.ndimage.gaussian_filter1d(data, sigma, mode='nearest')

def csmooth1(data, sigma):
   return fsmooth1(np.real(data),sigma) + 1j*fsmooth1(np.imag(data),sigma)
class curspos:

   def connect(self):
      return plt.connect('button_press_event',self)
   def __init__(self,tdata):
      self.tdata = tdata
      self.cid   = self.connect()
      self.tpls  = self.tdata['tpls']
      self.seqns = self.tdata['seqn']

   def __call__(self, event):
#     print(event.xdata, event.ydata)
      self.xdata = event.xdata
      self.ydata = event.ydata
#     print(self.xdata,self.ydata)
      xx = self.xd[self.select]
      yy = self.yd[self.select]
      tt = self.tpls[self.select]
      ss = self.seqns[self.select]
      ii = np.arange(0,len(self.select))
      ii = ii[self.select]
      darg = np.argmin(np.square(xx-self.xdata)+np.square(yy-self.ydata))
      self.tpl  = tt[darg]
#     print(self.tpl)
      self.seq  = ss[darg]
      self.darg = ii[darg]
      if (len(glob.glob(self.tpl)) >0): self.data = mrestore(self.tpl)[self.seq]
      else:print('pk file ',self.tpl,' not found')

   def kill(self):
      plt.disconnect(self.cid)

   def plot(self, xd, yd, zd, s=None, marker='o', over=False):
      if(str==type(xd)):self.xd= self.tdata[xd].copy()
      else:self.xd=xd.copy()
      if(str==type(yd)):self.yd= self.tdata[yd].copy()
      else:self.yd=yd.copy()
      if(str==type(zd)):self.zd= self.tdata[zd].copy()
      else:self.zd=zd.copy()
      sel = self.tdata['valid']
      if (s is not None): sel = sel & s
      self.select = sel
      zcplot(self.xd, self.yd, self.zd, select=sel, marker=marker,over=over)

   def getdata(self):
      bp = plt.waitforbuttonpress()
      return self.data

   def gettpl(self):
      bp = plt.waitforbuttonpress()
      return self.tpl

   def getd(self, key=None, key2=None):
      if (key is not None): self.key = key
      else:key = self.key
      data     = self.getdata()
      if (key is None):return data
      if (key2 is None):return self.data[key]
      return self.data[key][key2]

   def gettarg(self):
      return self.getdata()['targ']

   def gettplcontent(self):
      t = self.gettpl()
      return mrestore(t)[self.seq]

   def getheader(self):
      return self.gettplcontent()['header']
   def head2(self):
      h1 = self.getheader()
      h2 = self.getheader()
      return diffhead(h1,h2)

   def get3data(self):
      out={}
      out['targ'] = self.getd('targ')
      out['tpl']  = self.tpl
      out['seq']  = self.seq
      out['bcd']  = self.data['bcd']
      return out

   def gettpddata(key):
      return self.tdata[key][self.darg]
   def findtpl(self):
      zapt = self.tdata['tpls']==self.tpl
      zaps = self.tdata['seqn']==self.seq
      return (zapt&zaps)

   def zapplot(self):self.tdata['valid'][self.findtpl()] = False
   def zaptpl(self):
      tpk = self.tpl
      print('about to delete ',tpk,'. Go ahead [y/n]?')
      a = input()
      if (a!='y'): return
      os.remove(tpk)
      os.chdir(tpk[:23])
      for t in glob.glob('*.fits'):os.remove(t)
      os.chdir('..')
      os.rmdir(tpk[:23])
      self.zaptdata()


class vpos:
   def connect(self):
      return plt.connect('button_press_event',self)
   def __init__(self,vdata):
      if(str==type(vdata)): self.vdata=mrestore(vdata)
      else:self.vdata = vdata # raw violeta style input dataframes
      self.tdata = tpgdata(self.vdata) # reduced summary data
      self.cid   = self.connect()
      self.tpls  = self.tdata['tpls']

   def __call__(self, event):
      self.xdata = event.xdata
      self.ydata = event.ydata
#     print(self.xdata,self.ydata)
      aa = np.arange(0,len(self.xd))
      xx = self.xd[self.select]
      yy = self.yd[self.select]
      ii = aa[self.select]
      darg = np.argmin(np.square(xx-self.xdata)+np.square(yy-self.ydata))
      self.darg  = ii[darg]
      self.index = (self.darg/6).astype(int)
      self.chan  = self.darg-6*self.index
      self.data  = self.vdata.iloc[self.index]
      self.tpl   = self.tpls[self.darg]

   def kill(self):
      plt.disconnect(self.cid)

   def plot(self, xd, yd, zd, s=None, marker='o', over=False):
      if(str==type(xd)):self.xd= self.tdata[xd].copy()
      else:self.xd=xd.copy()
      if(str==type(yd)):self.yd= self.tdata[yd].copy()
      else:self.yd=yd.copy()
      if(str==type(zd)):self.zd= self.tdata[zd].copy()
      else:self.zd=zd.copy()
      sel = self.tdata['valid']
      if (s is not None): sel = sel & s
      self.select = sel
      zcplot(self.xd, self.yd, self.zd, select=sel, marker=marker,over=over)

   def getdata(self):
      bp = plt.waitforbuttonpress()
      return self.data

   def gettpl(self):
      bp = plt.waitforbuttonpress()
      return self.tpl
   
   def gettplcontent(self):
      t = self.gettpl()
      return mrestore(t)[self.seq]

   def getheader(self):
      return self.gettplcontent()['header']

   def getd(self, key=None, key2=None):
      if (key is not None): self.key = key
      else:key = self.key
#     data     = self.getdata()
      bp = plt.waitforbuttonpress()
      if (key is None):return self.data
      if (key2 is None):return self.tdata[key][self.darg]
      return self.tdata[key][self.darg][key2]

   def gettarg(self):
      return self.getd('targ')

   def get3data(self):
      out={}
      out['targ'] = self.getd('targ')
      out['tpl']  = self.tpl
      out['seq']  = self.seq
      out['bcd']  = self.data['bcd']
      return out

   def zapplot(self):self.tdata['valid'][self.darg] = False
