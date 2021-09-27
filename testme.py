import img2vis  as i2
import model_fitter as mf
import fft_models  as fm
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import scipy.optimize as sopt
import time
from matplotlib.backends.backend_pdf import PdfPages
128
from multiprocessing import Pool
import wutil
import matplotlib.patches as patches
from astropy.io import fits


def rphase(a,b):
   return np.angle(np.exp(1j*np.radians(a-b)),True)

def cc(x):    # colors red->green as tuples x is scaled 0->1
   r=np.clip(1-2*x,0,1) 
   b=np.clip(2*x-1,0,1) 
   g=1-b-r 
   return (r,g,b)

def mpause(interval, block=False):
   figman = plt.get_current_fig_manager()
   canvas = figman.canvas
   if(canvas.figure.stale): canvas.draw()
   canvas.start_event_loop(interval)
   if(block): x = input()
   return

def setpltwindow(figure, w=300,h=400, x=700, y=50):
   plt.figure(figure)
   win = plt.get_current_fig_manager().window
   win.move(x,y)
   win.resize(w,h)
   win.raise_()
   win.clearFocus()
   plt.tight_layout()

def tokenize(s):
   ss = s.split(' ')
   out = []
   for t in ss:
      if(len(t)>0) & (t!='\n'):out.append(t.strip())
   return out

def review1(im, tag='', cut=20, ns=None):
   prob = np.load(tag+'probabilities.npy')
   samp = np.load(tag+'samples.npy')
   if (ns is not None):
      prob = prob[-ns:]
      samp = samp[-ns:]
   pcut = prob>(prob.max() - cut)
   labels = im.getlabels()

   setpltwindow(1, 600,400, 700, 50 )
   for l in range(len(labels)):
      plt.clf()
      plt.plot(samp[:,l][pcut], prob[pcut],'+')
      plt.title(labels[l])
      mpause(.1,True)
   return

def review2d(im, tag, lx, ly, cut=20, ns=100000):
   prob = np.load(tag+'probabilities.npy')
   samp = np.load(tag+'samples.npy')
   if (ns is not None):
      prob = prob[-ns:]
      samp = samp[-ns:]
   pcut = prob>(prob.max() - cut)
   labels = im.getlabels()
   plt.close('all')

   win = plt.get_current_fig_manager().window
   for i,l in enumerate(labels):
      if (lx == l):ix = i
      if (ly == l):iy = i
   plt.clf()
   plt.plot(samp[:,ix][pcut], samp[:,iy][pcut],'+')
   mpause(.1)
   win.clearFocus()
   win.move(600,50)
   return

def review2(im, tag='', cut=20,x=700, y=50):
   labels = im.getlabels()
   if (len(labels)> 12):return
   prob   = np.load(tag+'probabilities.npy')
   samp   = np.load(tag+'samples.npy')
   pmax   = prob.max() - cut
   nlabel = len(labels)
   sprob  = prob.shape[0]
   nn     = 150000
   nplot  = sprob//nn
   if (nplot > 8):
      nn     =  500000
      nplot  = sprob//nn
   plt.close('all')
   nx  = 4
   ny  = (nlabel//nx) + ((nlabel%nx) != 0)
   fig, axarr = plt.subplots(ny,nx)
   for l in range(nlabel):
      sl = samp[:,l]
      ax = axarr.flatten()[l]
      ax.set_xlabel(labels[l])
      for i in range(nplot):
         pp = prob[i*nn:(i+1)*nn]
         pa = pp > pmax
         ss = sl[i*nn:(i+1)*nn]
         h  = np.histogram(ss[pa], bins=200)
         ax.plot(h[1][1:], h[0])
      if(l==nx):ax.set_ylabel(tag+f'{pmax+cut:7.1f}')
   plt.tight_layout()
   setpltwindow(1, 500,700, x, y )
   mpause(1)
   return

def runscript(file):
   r = rundata(file)
   r.runscript()
   return r

def check_fit2(tag, param, labels):
#make a plot showing "burn-in" of sample chains
   setpltwindow(1, 600,400, 700, 50)
   x = np.load(tag+'samples.npy')
   naxis = len(labels)
   xvals=np.arange(0,np.shape(x)[0],1)
   plt.plot(xvals,x[:,param], '.-', markersize=0.2, c='k', alpha=0.25)
   plt.plot([0,x.shape[0]],[np.median(x[-100:,param]), np.median(x[-100:,param])], 'r--' )
   plt.xlabel("N iter")
   plt.ylabel(labels[param])
   plt.tight_layout()
   return

def getbandf(file):
   bands = {} # dictionary
   band  = None
   waves = []
   with open(file,'r') as f:
      while True:
         ff = f.readline()
         if(len(ff)==0):break
         line = tokenize(ff)
         if(len(line)==0): continue # empty line
         if(line[0][0]=='#'): continue # comment
         if (len(line)==1): #band name
            if(band is not None): bands[band] = waves # store current definition
            band = line[0]
            waves = []
         elif (len(line)==2):
            waves.append(line)
   bands[band] = waves
   return bands

def getweightf(file):
   weightdata = {} # dictionary
   mode       = None
   with open(file,'r') as f:
      while True:
         ff = f.readline()
         if(len(ff)==0):break
         line = tokenize(ff)
         if(len(line)==0): continue # empty line
         if(line[0][0]=='#'): continue # comment
         if(line[0]=='telescope'):
            mode = 'telescope'
            weightdata['telescope'] = {}
         elif(line[0]=='baseline'):
            mode = 'baseline'
            weightdata['baseline'] = {}
         elif(line[0]=='triangles'):
            mode = 'triangles'
            weightdata['triangles'] = {}
         else:
            if (len(line) != 2):
               print('weightfile syntax error ',line)
               return None
            weightdata[mode][line[0]] = float(line[1])
   return weightdata

def getobsf(file,wavedata=None, raw=False,merge=None, weightdata=None):

   obs = []
   if (raw):
      print('retrieving data described in '+file+' with merge=',merge,
      ' in raw format')
#  else: 
      print('retrieving data described in '+file+' with merge=',merge)
      #' at wavelength ',wave)
   dayn = -1
   with open(file,'r') as f:
      while True:
         ff = f.readline()
         if(len(ff)==0):break #blank line
         if(ff[0]=='#'):continue #comment
         line = tokenize(ff)
         if(line[0][0]=='$'):  #new day
            dayn   +=1
            dayname = line[0][1:]
            obs.append({'dayn':dayn,'dayname':dayname})#information for this day
            obs[dayn]['files'] = []
         else: 
            obs[dayn]['files'].append(line[0]) 
   outobs  = i2.matisseObs(wavedata, obs, raw=raw, merge=merge, 
      weightdata=weightdata)
#             put in my ideas of noise
   md  = outobs.getmatissedata()
   print('data has ',len(md['vis']),' visibilities and ',len(md['cphase']),' closure phases')
   if (not raw):
      md['vis_noise'][:]    = .2
      md['cphase_noise'] = 10*np.ones(len(md['cphase']))
#                 now use error in abs(exp(j *data-model))
      md['cphase_noise'] = 0.2*np.ones(len(md['cphase'])) 
   return outobs
      
def getimf(file):
   with open(file,'r') as f:
#              set up components
      ctype = tokenize(f.readline())
      ncomp = len(ctype)
      comps = []
      for c in range(ncomp):
         if (ctype[c] == 'gauss'):comps.append(fm.gauss())
         elif (ctype[c] == 'disk'):comps.append(fm.disk())
         elif (ctype[c] == 'ring'):comps.append(fm.ring())
         elif (ctype[c] == 'expn'):comps.append(fm.expn())
         elif (ctype[c] == 'ftn2'):comps.append(fm.ftn2())
         elif (ctype[c] == 'point'):comps.append(fm.point())
         else :comps.append(fm.extra(labels=[ctype[c].strip()]))

      im = fm.multicomp(comps)
      while(True):
         vals = tokenize(f.readline())
         if(len(vals) == 0):break
         if(vals[0][0]=='#'):break
         c, label, lval, ival, uval, fix = vals #f.readline().split(' ')
         c = int(c)
         label  = label.strip()
         comp   = comps[c-1]
         comp.setvalue(label,  float(ival))
         comp.setlbound(label, float(lval))
         comp.setubound(label, float(uval))
         if(fix.strip()=='F'):comp.setfixed(label)
   return im

def makescript(prefix=None, ms='',os='a', template=None, 
   ptemplate=None, **kwargs):
   parms = {}
   if (template is not None):
      with open(template,'r') as f:
         while(True):
#              # parse to dictionary
            toks = tokenize(f.readline())
            if (len(toks)==0):break
            parms[toks[0]] = toks[1]
      if (prefix is None): prefix = parms['tag'][0]
      if (ptemplate is None): ptemplate = parms['modelfile']
      for k in kwargs.keys():parms[k] = kwargs[i]
   else:
      parms = kwargs
   if (prefix is None): prefix = 'q'
   swave         = prefix+wave[0]+wave[2]
   parms['tag']  = swave
   parms['modelfile'] = swave+ms
   parms['outmodel'] = swave+os

   f.close()
   f = open('script'+swave, 'w')
   for k in parms.keys():
      f.writelines(f'{k:12}  {parms[k]:12} \n')
   f.close()
   if (ptemplate is not None):
      shutil.copy(ptemplate, parms['modelfile'])
   return

class rundata():
   def __init__(self,file):

      self.script = {}
      self.script['tag'] = None
      with open(file,'r') as f:
         while True:
            ff = f.readline()
            if(len(ff)==0):break
            line = tokenize(ff)
            if(len(line)==0):break
            if(line[0][0]=='#'): continue
            self.script[line[0]] = line[1]


      self.scriptfile = file
      self.tag        = self.script['tag']
   #      copy input files into log
      self.logfile    = self.tag +'.log'
      self.datafile   = self.script['datafile']
      self.modelfile  = self.script['modelfile']
      self.wavefile   = self.script['wavefile']

      print('Processing ',file,': data from ',self.datafile,
         ': model from ',self.modelfile,end=' ')
      self.niter     = int(self.script['niter'])
      if ('cpnoise' in self.script.keys()):
         self.cphase_noise = float(self.script['cpnoise'])
      else:self.cphase_noise = 0.1 # radians
      if ('visnoise' in self.script.keys()):
         self.vis_noise = float(self.script['visnoise'])
      else:self.vis_noise = 0.01 # 
      if (self.script['output']=='1'):
         self.output = self.script['tag'] # create output plots/files
      else:self.output = None
      self.outmodel    = self.script['outmodel']
      if ('merge' in self.script.keys()): 
         self.merge = float(self.script['merge'])
      else:self.merge = None
      self.debug = False
      if ('debug' in self.script.keys()):
         if (self.script['debug'] == 1):self.debug = True
      self.hard  = 0   # default no hard limits on parameters
      if ('hard' in self.script.keys()):
         self.hard = int(self.script['hard'])
      self.update   = self.script['update']
      self.nwalkers =int(self.script['nwalkers'])
      self.uvl      =float(self.script['uvl'])

      self.wavedata = getbandf(self.wavefile)
      if ('weightfile' in self.script.keys()):
         print(': weightfile=',self.script['weightfile'])
         self.weightfile   = self.script['weightfile']
         self.weightdata   = getweightf(self.weightfile)
      else: 
         self.weightdata = None
         self.weightfile = None
         print(': no weight file')


      self.im   = getimf(self.modelfile)
      self.obs  = getobsf(self.datafile, self.wavedata, 
         merge=self.merge, weightdata=self.weightdata)
      self.setuv2(self.uvl)

      self.multitags  = None # not yet processed separate wavelengths
      self.multimodels = None
      self.labels     = np.array(self.im.getlabels())
      self.bestlike   = self.lnlike()
      self.bestvalues = self.im.getvalues()
      print('retrieved rundata from script ' , self.scriptfile)
      print('niter=',self.niter,'nwalkers=',self.nwalkers)
      return


   def setuv2(self,uvl):
      self.uvl = uvl
      uv2 = -0.5*(self.obs.matissedata['u']**2 
         + self.obs.matissedata['v']**2)/self.uvl**2
      self.uv2 = np.exp(uv2)
      
   def dotest(self, dotv=None):
      im  = self.im
      obs = self.obs
      for la,lv,lb,ub in zip(im.getlabels(), im.getvalues(), im.getlbound(), im.getubound()):
         print(f'{la:5s}  {lb:8.2f} {lv:8.2f} {ub:8.2f}')
      self.method = 'mix'
      if(self.debug):  return [obs, im, uv2]
      best_fit = mf.do_mcmc(obs, im,  self, uniform_guess=True, dotv=dotv)
      self.best_fit = best_fit

   def runscript(self):
      file = self.scriptfile
   #      copy input files into log
      islog   = os.path.isfile(self.logfile)
      lf      = open(self.logfile,'a')
      lf.writelines('script = '+file+':\n')
      t       = open(file,'r')
      lf.writelines(t.readlines())
      t.close()
      if (not islog):
         t = open(self.datafile,'r')
         lf.writelines('data input:\n')
         lf.writelines(t.readlines())
         t.close()
      lf.writelines('modelfile='+self.modelfile+':\n')
      t = open(self.modelfile,'r')
      lf.writelines(t.readlines())
      t.close()

      self.dotest(dotv=True)
      values = self.best_fit
      im     = self.im
      if(self.update): self.updatevalues()
      dfree = self.obs.sumweights
      lf.writelines(f'max_log_like= {self.bestlike:7.2f} {self.bestlike/dfree:7.3f}\n')
      lf.close()

      self.review2()
      mpause(2)

   def lnlike(p=None):
      if p is None: return mf.lnprob(self.im.getvalues(),
         'mix', self)
      else: return mf.lnprob(p, 'mix', self)

   def lnlike2(params):
    #log likelihood function
       imagemodel = self.im
       matisseobs = self.obs
       wave       = matisseobs.wave
       nwave      = len(wave)
       debug = rundata.debug
       uvm   = rundata.uv2
       if(debug): print('pp',params)

       md = matisseobs.matissedata #observation data struct
#             update model parameters
       imagemodel.setvaluelist(params)

       labels      = imagemodel.getlabels()  
       if (debug):print(labels)
       y           = 0.1*md['vis'] # 10% of value
       yerr        = self.vis_noise
       yerr        = np.sqrt(y**2 + yerr**2) #  in quadrature

       matisseobs.modelcflux(imagemodel)
       if (not matisseobs.valid): return -np.inf # parameters out of range
       matisseobs.modelpflux(imagemodel)
# update obs.cflux with previous value
       cflux= matisseobs.cflux+self.cflux 
       pflux= matisseobs.photflux+self.photflux # update photflux
       modelvis    = obs.flux2vis(cflux, pflux)
       modelcphase = self.flux2cphase(cflux)
       residuals   = np.zeros(model.shape)
       for i in range(nwave):
          residuals[:,i]   = uvm*md['wt2'][:,i]* \
             ((modelvis[:,i] - md['vis'][:,i])/yerr[:,i])**2

       y           = self.cphase_noise*np.ones(md['cphase'].shape)
       model       = np.append(modelvis,modelcphase)  # append cphases to vis2s
       m           = np.exp(1j*np.radians(m)) # convert to rotor
       data        = np.exp(-1j*np.radians(md['cphase']))
       yerr        = np.append(yerr,y)  # append cphase errs to vis2errs
       residuals   = np.append(residuals, md['wt3']*
          (np.angle(m*data)/y)**2)  # error not in angle but in rotor
#              use rotors to avoid phase wrapping in angles

       out = -0.5*sum(residuals)
       inv_sigma2 = np.absolute(1.0 / (yerr**2))# + model**2 * np.exp(2*lnf)))
       if(debug):
          data = np.append(md['vis'], md['cphase'])
          s = 0
          j = -1
          for d,m,r,y in zip(data,model,residuals,yerr):
             s += r
             j+=1
             print(f'{j:5d} {s:8.4f} {d:7.3f} {m:7.3f} {r:7.3f} {y:7.3f} ')
          print('out',out)
       return out

   def lsq(self, params):
      y =  mf.lnprob(params, 'mix', self)
#     out = f'{y:8.2f}'
#     for p in params:out+=f'{p:6.2f} '
#     print(out)
      return -y

   def findmax(self, doprint=False):
     x0     = self.im.getvalues()
     lb     = self.im.getlbound()
     ub     = self.im.getubound()
     xs     = 5.0*(ub-lb)
     result = sopt.least_squares(self.lsq, x0, x_scale=xs, 
     ftol   = 0.01,
#       loss='soft_l1', 
#       bounds=(lb,ub), 
        verbose=0)
     self.bestvalues = result['x']
     self.bestlike   = mf.lnlike(result['x'], 'mix', self)
     self.im.setvaluelist(self.bestvalues)
     out    = ''
     lout   = '   '
     if (doprint):
        print('best lnlike = ', f'{self.bestlike:8.2f}')
        for l,p in zip(self.labels,self.bestvalues):
           lout+= f'{l:6} '
           out += f'{p:6.2f} '
        print (lout)
        print (out)
     return 

   def getbounds(self, dlog = 2):
      low  = 0*self.bestvalues
      high = 0*self.bestvalues
      bmax = self.bestlike - dlog
      lb   = self.im.getlbound()
      ub   = self.im.getubound()
      xs   = (ub-lb)
      for p in range(len(low)): # loop over params
         best = self.bestvalues.copy()
         ok = False; 
#        dp = .001
         dp = .001*xs[p]
         niter = 0
         while (not ok):
            best[p] = self.bestvalues[p]
            db      = dp*np.abs(best[p])
            for i in range(20):
#              best[p]*= (1+dp)
               best[p] += db
               l = mf.lnlike(best, 'mix', self)
               if (l < bmax):break
            if(i==0):    dp*=.5
            elif(i >=19):dp *=2
            else: ok = True
#           if(self.bebug):print('high' ,p, dp, ok, i, best[p], bmax-l)
            niter += 1
            if (niter>20):break
         high[p] = best[p]

#        print('lowlow', p,mf.lnlike(best, 'mix', self))
         ok = False; 
         dp = .001
         niter = 0
         while (not ok):
            best[p] = self.bestvalues[p]
            db      = -dp*np.abs(best[p])
            for i in range(20):
#              best[p]/= (1+dp)
               best[p] += db
               l = mf.lnlike(best, 'mix', self)
               if (l < bmax):break
            if(i==0):    dp*=.5
            elif(i >=19):dp *=2
            else: ok = True
            niter += 1
            if (niter>20):break
#           if(self.bebug):print('low' ,p, dp, ok, i, best[p], bmax-l)
         low[p] = best[p]
      return np.transpose(np.array([low, self.bestvalues, high]))

   def maxupdate(self, dlog=3, doprint=False, selfupdate=True, fudge=3):
      self.findmax(doprint=doprint)
      bounds = self.getbounds(dlog = dlog)
      self.updatevalues(values = bounds, selfupdate=selfupdate, fudge=fudge)

   def fixpars(self, dofix=['l','m','amp']):
      for p in dofix:
         for c in self.im.comps:
            j = np.where(c.labels==p)[0]
            c.notfixed[j] = False

   def addcomp(self, ab, lb, mb): # bounds for new comp on amp, l, m
      g = fm.gauss()
      g.ubound = self.im.comps[0].ubound  # copy of 1st component
      g.lbound = self.im.comps[0].lbound
      g.values = self.im.comps[0].values.copy()
      g.values[3] = .001 # initial value of amp very small
      g.notfixed[:] = False  # well, mostly false
      g.lbound[3] = ab[0]; g.ubound[3] = ab[1]; g.notfixed[3] = True
      g.lbound[4] = lb[0]; g.ubound[4] = lb[1]; g.notfixed[4] = True
      g.lbound[5] = mb[0]; g.ubound[5] = mb[1]; g.notfixed[5] = True
      self.im.addcomp(g)

   def fmax(self, p):
      self.im.setvaluelist(p)
      self.findmax()
#     print(f'{self.bestlike:7.0f}',end='')
      return [self.bestvalues, self.bestlike]

   def itercomp(self, ab, lb, mb, niter=10, gain=.5,dotv=True ):
#             calculate and store initial model corrflux
      self.obs.modelcflux(self.imagemodel)
      self.obs.modelphotflux(self.imagemodel)
#             add a new component to list with specified limits
      self.addcomp(ab,lb,mb)
      for i in range(niter):
         ta = time.time()
         self.dobest()
         tb = time.time()
         print(f'{i:4d} {self.bestlike:5.1f}  {tb-ta:5.1f}')
         if(dotv):
            m = self.image()
            wutil.wtv(m, vmax=0.3*m.max())
            plt.title(f'{self.bestlike:5.1f}')
            mpause(.1)
         self.fixpars()
         c = self.im.comps[-1] # last added component
         j = np.where(c.labels=='amp')[0]
         c.values[j] *= gain
         self.addcomp(ab,lb,mb)


   def dobest(self):
            
#       sort out input parameters
      nwalkers = self.nwalkers
      ndim     = self.im.nnparams()
      output   = self.output 

      ncpu     = os.cpu_count()  
      with Pool() as pool:
        # uniform initial distribution function using limits from input
#        lbound = self.im.getlbound()
#        ubound = self.im.getubound()
         lbound = self.im.comps[-1].getlbound()
         ubound = self.im.comps[-1].getubound()
         pos = [np.array([np.random.uniform(low=lbound[j],high=ubound[j]) for j in range(len(lbound)) ]) for kk in range(nwalkers)]
         pos = np.array(pos)
#                 recycle current position as starting pos
         pos[0]  = self.im.getvalues()
         samples = []
         probs   = []
# find local max of lnlike from each of the random starting positions
# keep track of positions and likelihoods
         llist = pool.map(self.fmax, pos) 
         for l in llist:
            samples.append(l[0])
            probs.append(l[1])
         samples = np.array(samples)
         probs   = np.array(probs)
#                  find and store most probable of the whole set
         amax            = probs.argmax()
         self.bestlike   = probs[amax]
         self.bestvalues = samples[amax]
         self.im.setvaluelist(self.bestvalues)

         np.save(self.tag+'samples.npy', samples)
         np.save(self.tag+'probabilities.npy', probs)

      return 

   def review1(self, cut=20, ns=None):
      review1(self.im, self.tag, cut=cut, ns=ns)
   def review2d(self, lx, ly, cut=20, ns=100000):
      review2d(self.im, self.tag, lx, ly, cut=20, ns=100000)

   def review2(self, cut=20,x=600, y=50):
      review2(self.im, self.tag, cut=cut,x=x, y=y)

   def bestfit(self, dolist=True):
      prob=np.load(self.tag+'probabilities.npy')
      samp=np.load(self.tag+'samples.npy')
      most = np.where(prob==prob.max())
      most = most[0][0]
      values = samp[most]
      print('max lnlike:',prob[most])
      if dolist:return values
      labels = self.im.getlabels()
      out    = {}
      for l,v in zip(labels,values):out[l] = v
      return out

   def setdvalues(self, params):
#                   set self.im values from a dictionary with labels
#                   can include "fixed" parameters
      for k in params.keys():
         label = k.split('_')
         comp  = int(label[1])
         label = label[0]
         im.setvalue(comp,label, params[k])

   def lnlike(self, params=None):
      method = 'mix'
      im     = self.im
      if (params is not None): im.setvaluelist(params)
      return mf.lnlike(im.getvalues(), method, self)


   def plotmodel(self, onepage=True, dowt = .1, last=2000, usebest=False):
      obs    = self.obs
      md     = obs.matissedata
      im     = self.im
      wave   = self.wave
      tag    = self.tag

   # collect obs data
      wt2    = md['wt2']; ok2 = wt2>dowt
      wt3    = md['wt3']; ok3 = wt3>dowt

      tel2 = md['tel2']; tel3 = md['tel3']
      vis  = md['vis']; cphase = md['cphase']
      bl   = md['bl']

   # collect model estimates
      samples = np.load(tag+'samples.npy')
      visstat,phstat = obs.visstat(im, samples, last=last)
#     phstat  = obs.phasestat(im, samples, last=last)
      plt.close('all')
      if onepage: fig, axarr = plt.subplots(2,3)
#            vis**2 vs. baseline; model
      vv   = visstat[1][ok2]
      verr = np.array([vv-visstat[0][ok2], visstat[2][ok2]-vv])
      if (onepage):
         a = axarr[0,0]
         a.errorbar(bl[ok2], vv, verr, fmt='o')
         a.set_xlabel('baseline(m)')
         a.set_ylabel('vis**2')
         a.set_title('model')
      else:
         plt.errorbar(bl[ok2], vv, verr, fmt='o')
         plt.xlabel('baseline(m)')
         plt.ylabel('vis**2')
         plt.title('model visibilities')
         mpause(.1, True)
#            vis**2 vs. baseline; data
      if (onepage):
         a = axarr[0,1]
         a.plot(bl[ok2], vis[ok2], 'ob')
         a.set_xlabel('baseline(m)')
         a.set_ylabel('vis**2')
         a.set_title('observation')
      else:
         plt.clf()
         plt.plot(bl[ok2], vis[ok2], 'ob')
         plt.xlabel('baseline(m)')
         plt.ylabel('vis**2')
         plt.title('observed')
         mpause(.1, True)
#            vis**2 vs. baseline; residuals
      vv   = vis[ok2] - visstat[1][ok2]
      if (onepage):
         a = axarr[0,2]
         a.errorbar(bl[ok2], vv, verr, fmt='o')
         a.set_xlabel('baseline(m)')
         a.set_ylabel('vis**2')
         a.set_title('residuals')
      else:
         plt.clf()
         plt.errorbar(bl[ok2], vv, verr, fmt='o')
         plt.xlabel('baseline(m)')
         plt.ylabel('vis**2')
         plt.title('residuals')
         mpause(.1, True)
#            cphase; model
      ph      = phstat[1]
      perr    = np.array([ph-phstat[0], phstat[2]-ph])

      if onepage:
         a = axarr[1,0]
         a.errorbar(np.arange(len(ph)), ph, perr, fmt='o')
         a.set_ylabel('phase(deg)')
         a.set_title('residuals')
      else:
         plt.clf()
         plt.errorbar(np.arange(len(ph)), ph, perr, fmt='o')
         plt.ylabel('phase(deg)')
         plt.title('model closure phase')
         mpause(.1, True)
#            cphase; data
      if onepage:
         a = axarr[1,1]
         a.plot(cphase,'bo')
         a.set_ylabel('phase(deg)')
         a.set_title('observations')
      else:
         plt.clf()
         plt.plot(cphase,'bo')
         plt.ylabel('closure phase(deg)')
         plt.title('observed closure phase')
         mpause(.1,True)
#            cphase; residuals
      ph       = np.angle(np.exp(1j*(np.radians(cphase-phstat[1]))),True)
      if onepage:
         a = axarr[1,2]
         a.errorbar(np.arange(len(ph)), ph, perr, fmt='o')
         a.set_ylabel('phase(deg)')
         a.set_title('residuals')
         fig.tight_layout()
      else:
         plt.clf()
         plt.errorbar(np.arange(len(ph)), ph, perr, fmt='o')
         plt.ylabel('closure phase(deg)')
         plt.title('closure phase residuals')
      mpause(.1)
      return

   def printmodel(self, onepage=True, dowt = .1, last=2000, usebest=False):
      obs    = self.obs
      md     = obs.matissedata
      im     = self.im
      wave   = self.wave
      tag    = self.tag

   # collect obs data
      wt2    = md['wt2']; ok2 = wt2>dowt
      wt3    = md['wt3']; ok3 = wt3>dowt

      tel2 = md['tel2'][ok2]; tel3 = md['tel3'][ok3]
      vis  = md['vis'][ok2]; cphase = md['cphase'][ok3]
      bl   = md['bl'][ok2]

   # collect model estimates
      samples = np.load(tag+'samples.npy')
      if (usebest):
         prob = np.load(tag+'probabilities.npy')
         pmax = np.where(prob==prob.max())[0][0]
         smax = samples[pmax]
         im.setvaluelist(smax)
         valid,modelvis,modelcphase = obs.modeldata(im)
      else:
         modelvis,modelcphase= obs.visstat(im, samples, last=last) #median
#        modelcphase  = obs.phasestat(im, samples, last=last)[1]
         modelvis = modelvis[1]
         modelcphase = modelcphase[1]
      for v,m,o in zip(md['vis'], modelvis, ok2):
         if (o):print(f'{v:8.3f} {m:8.3f} {v-m:8.3f}')
      for c,m,o in zip(md['cphase'], modelcphase, ok3):
         r = np.angle(np.exp(1j*(np.radians(c-m))),True)
         if (o):print(f'{c:8.1f} {m:8.1f} {r:8.1f}')


   def multioutput(self):
      labels    = self.im.getlabels()
      waves     = self.obs.getwave()
      allabels  = self.im.getalllabels()
      allvals   = self.im.getallvalues()

      comps = ''
      for c in self.im.comps:
         name = c.name.strip()
         if(name=='gauss'):comps+='gauss '
         elif(name=='disk'):comps+='disk '
         elif(name=='ring'):comps+='ring '
         elif(name=='ftn2'):comps+='ftn2 '
         else:comps+=' lnf '

      f = open(self.outmodel+'.multi','w')
      for w,lnl,tags in zip(waves, self.multilike, self.multitags):
         f.writelines(f'{tags} {w:6.2f} {lnl:6.2f} \n')
         f.writelines(comps+' \n')
         stat = mf.samplestats(tag=tags)
         for label,val in zip(allabels, allvals):
            la = label.split('_')
            la[0] = la[0].strip()
            la[1] = la[1].strip()
            if (label in labels):
               i = labels.index(label)
               v = stat[i,1]
               u = stat[i,2]
               l = stat[i,0]
               fixed = 'V'
            else:
               v = val; l=val; u=val; fixed='F'
            out = f'{la[1]:1} {la[0]:5}  {l:10.4f} {v:10.4f} {u:10.4f}  {fixed:2}'
            print(out)
            f.write(out+'\n')
      f.close()
         
   def updatevalues(self, values=None, fudge = None, selfupdate=False):
      print('update', fudge, selfupdate)
      im       = self.im
      if (values is None):
         stat     = mf.samplestats(tag=self.tag)
      else:
         stat     = values

      if (fudge is not None):  #  increase limits a bit
         stat[:,0] -= fudge*(stat[:,1]-stat[:,0])
         stat[:,2] += fudge*(stat[:,2]-stat[:,1])
      labels   = list(self.labels)
      allabels = im.getalllabels()
      allvals  = im.getallvalues()
      f=open(self.outmodel,'w')
      comps = ''
      for c in im.comps:
         name = c.name.strip()
         if(name=='gauss'):comps+='gauss '
         elif(name=='disk'):comps+='disk '
         elif(name=='ring'):comps+='ring '
         elif(name=='ftn2'):comps+='ftn2 '
         else:comps+=' lnf '
      f.writelines(comps+'\n')
      for label,val in zip(allabels, allvals):
         la = label.split('_')
         la[0] = la[0].strip()
         la[1] = la[1].strip()
         if (label in labels):
            i = labels.index(label)
            v = stat[i,1]
            u = stat[i,2]
            l = stat[i,0]
            fixed = 'V'
         else:
            v = val; l=val; u=val; fixed='F'
         out = f'{la[1]:1} {la[0]:5}  {l:10.4f} {v:10.4f} {u:10.4f}  {fixed:2}'
         print(out)
         f.write(out+'\n')
      f.close()
      if selfupdate:
         im.setvaluelist(stat[:,1])
         im.setlboundlist(stat[:,0])
         im.setuboundlist(stat[:,2])
      return


   def check_fit2(self, param):
      check_fit2(self.tag, param, self.labels)

   def review3(self, param):
      setpltwindow(1, 600,400, 700, 50)
      nbins  = 200
      ilabel = np.where(param==self.labels)
      x      = np.load(self.tag+'samples.npy')[:, ilabel]
      niter  = self.niter
      nwalk  = self.nwalkers
      xvals  = np.arange(niter)
      data   = np.zeros((nbins, self.niter))
      xmin   = x.min()
      xmax   = x.max()
      rdata  = (xmin, xmax)
      for i in range(niter):
         h = np.histogram(x[i*nwalk:(i+1)*nwalk], bins=nbins, range=rdata)
         data[:,i] = h[0]
      y      = np.arange(0, nbins, nbins//5) # y-tick pixel locations
      ylab   = []
      for yy in y:ylab.append(f'{xmin+(xmax-xmin)*yy/nbins:5.2f}')

#  medval = np.median(x[-nwalk:])
      perc   = np.percentile(x[-2*nwalk:],[17,50,83])
      medval = perc[1]
#  medy   = int(nbins*(medval-xmin)/(xmax-xmin))
      dm     = data.max()
      for p in perc:
         medy   = int(nbins*(p-xmin)/(xmax-xmin))
         data[medy,-100:] = 1.5*dm

      plt.imshow(data, origin='lower')
      plt.yticks(y, ylab)
      plt.xlabel("N iter")
      plt.ylabel(param)
      plt.title(param+' '+self.scriptfile+' perc='+f'{perc[0]:6.2f} {perc[1]:6.2f} {perc[2]:6.2f}')
      plt.tight_layout()
      return

   def getstats(self,tag=None):
      if (tag is None):tag = self.tag
      return [np.load(tag+'samples.npy'),np.load(tag+'probabilities.npy')]

   def setmodel(self,model='median',tag=None, cut=10000):
      if (model!='median') & (model!='best'):
         print('dont recognize model '+model+'; nop')
         return
      if (tag is None):tag=self.tag
      samples,probs = self.getstats(tag)
      if (model=='best'): v = samples[np.argmax(probs)]
      else:               v = np.median(samples[-cut:],0)
      self.im.setvaluelist(v)
      return

   def review4(self,output=None):
      im   = self.im
      obs  = self.obs
      md   = obs.matissedata
      waves= obs.getwave()
      multi= (self.multimodels is not None)

      ok2  = md['wt2'][:,0]!=0
      ok3  = md['wt3'][:,0]!=0
      vis  = md['vis'][ok2]
      cphase = md['cphase'][ok3]
      bl   = md['bl'][ok2]
      tel2 = md['tel2'][ok2]
      tel3 = md['tel3'][ok3]
      flip = md['cflip3'][ok3]
      ut3  = np.unique(tel3)
      
      valid,mvis,mcphase = obs.modeldata(im)
      mvis    = mvis[ok2]
      mcphase = mcphase[ok3]

      setpltwindow(1, 600,500, 700, 50)
      plt.clf()
      plt.tight_layout()
      if (output is not None):pdf = PdfPages(output)

      if (not multi):  # single set of source parameters stored in self.im
         valid,mvis,mcphase = obs.modeldata(im) # calculate model vis/cphase
         mvis = mvis[ok2]
         mcphase = mcphase[ok3]
      else:      #  create wavedata sets for individual wavelengths
         mvis    = []
         mcphase = []
         imw     = getimf(self.modelfile)
         for i,k in enumerate(self.wavedata.keys()):
            wd = {k:self.wavedata[k]}
            obw = getobsf(self.datafile, wavedata=wd, 
               weightdata=self.weightdata)
            imw.setvaluelist(self.multimodels[i]) # parameters for this wave
            valid,v,c = obw.modeldata(imw)           
            v = v[ok2,0]
            c = c[ok3,0]
            mvis.append(v)
            mcphase.append(c)
         mvis    = np.transpose(np.array(mvis))
         mcphase = np.transpose(np.array(mcphase))

      deltavis = vis - mvis
      sigvis = np.std(deltavis,axis=0)
      deltaphase = np.angle(np.exp(1j*np.radians(cphase-mcphase)),True)
      sigphase = np.std(deltaphase,axis=0)

      for w in range(len(waves)):
         title = f" {waves[w]:6.2f} $\mu$"
         plt.clf()
         for b,v,m,t in zip(bl,vis[:,w],mvis[:,w],tel2):
            plt.plot(b, v, '+b')
            plt.plot(b, m, 'xb')
            plt.plot(b*np.ones(2),[v,m],linewidth=.2)
            plt.text(b,v,t)
         plt.title('visibilities'+title)
         mpause(.1, True)
         plt.clf()
#        dvis = vis[:,w]-mvis[:,w]
         dvis = deltavis[:,w]
         plt.plot(bl, dvis,'+b'); print(np.sqrt(np.mean((deltavis[:,w])**2)))
         for b,v, t in zip(bl,dvis,tel2):
            plt.text(b,v,t)
         plt.title('vis-modelvis'+title)
         if(output is not None):pdf.savefig()
         mpause(.1, True)
         plt.clf()
#              cphases, flip if necessary
         for t in range(len(ut3)):
            w3 = np.where(tel3==ut3[t])[0]
            w3 = np.sort(w3)
            for ww in w3:
               cp = flip[ww]*cphase[ww,w]
               mp = flip[ww]*mcphase[ww,w]
               plt.plot(t,cp,'+b')
               plt.plot(t,mp,'xb')
               plt.plot(t*np.ones(2),[cp,mp],linewidth=.2)
            plt.text(t,flip[w3[0]]*cphase[w3[0],w],ut3[t])
         plt.title('closure phases'+title)
         if(output is not None):pdf.savefig()
         mpause(.1, True)
         dphase = np.angle(np.exp(1j*np.radians(cphase[:,w]-mcphase[:,w])),True)
         plt.clf()
         acp=[]
         for t in range(len(ut3)):
            w3 = np.where(tel3==ut3[t])[0]
            w3 = np.sort(w3)
            for ww in w3:
               plt.plot(t,flip[ww]*dphase[ww],'+b')
#           plt.text(t+.5,flip[w3[0]]*dphase[w3[0]],ut3[t])
            plt.text(t+.5,np.mean(flip[w3]*dphase[w3]),ut3[t]); acp.append(flip[ww]*dphase[ww])
         print(np.sqrt(np.mean(np.array(acp)**2)))
         plt.title('dphases'+title)
         if(output is not None):pdf.savefig()
         mpause(.1,True)
      if(output is not None):pdf.close()
      return [sigvis,sigphase]

   def getdata(self):
      vis     = self.obs.get2data('vis')
      cp      = self.obs.get3data('cphase')
      flip    = self.obs.get3data('cflip3')
      for i in range(cp.shape[1]):cp[:,i]*=flip
      if (cp.shape[1]==1):
         return [vis.flatten(), cp.flatten()]
      return [vis,cp]

   def getmodeldata(self, parms=None):
      ok2,ok3 = self.obs.getok()
      flip    = self.obs.get3data('cflip3')
      if (parms is not None):self.im.setvaluelist(parms)
      valid,mvis,mcp = self.obs.modeldata(self.im)
      if (not valid):
         print(' no valid model data')
         return None
      mvis = mvis[ok2]
      mcp  = mcp[ok3]
      for i in range(mcp.shape[1]):mcp[:,i]*=flip
      if (mcp.shape[1]==1):
         return [mvis.flatten(),mcp.flatten()]
      return [mvis,mcp]

   def getmultimodels(self):
      mvis    = []
      mcphase = []
      ok2  = self.obs.matissedata['wt2'][:,0]!=0
      ok3  = self.obs.matissedata['wt3'][:,0]!=0
      wd = {'wave':[[3.0,5.0]]}
      flip = self.obs.matissedata['cflip3'][ok3]
      for i,k in enumerate(self.wavedata.keys()):
         wd[k] = self.wavedata[k]
         obsw  = getobsf(self.datafile, wavedata=wd,
            weightdata=self.weightdata)
         self.im.setvaluelist(self.multimodels[i])
         valid,vs,ps = obsw.modeldata(self.im)
         vs  = vs[ok2,0]
         ps  = ps[ok3,0] * flip
         mvis.append(vs)   # model visibility statistics
         mcphase.append(ps)   # model visibility statistics
      mvis    = np.transpose(np.array(mvis))
      mcphase = np.transpose(np.array(mcphase))
      return [mvis, mcphase]

   def getmultistatmodels(self, last=400):
      waves = self.obs.getwave()
      mvis=[]
      mlower=[]
      mupper=[]
      mcphase=[]
      mclower=[]
      mcupper=[]
      ok2   = self.obs.matissedata['wt2'][:,0]!=0
      ok3   = self.obs.matissedata['wt3'][:,0]!=0
      flip  = self.obs.matissedata['cflip3'][ok3]
      wd = {}
      for i,k in enumerate(self.wavedata.keys()):
#        wd[k] = self.wavedata[k]
         wd[k]    = self.wavedata[k]
         obsw  = getobsf(self.datafile, wavedata=wd,
            weightdata=self.weightdata)
         stag    = self.multitags[i]+'samples.npy'
         print('processing ',stag)
         samples = np.load(stag)
         vs,ps = obsw.visstat(self.im, samples, last=last)
         vs  = vs[:,ok2,0]
         ps  = ps[:,ok3,0] * flip
         for i in range(len(flip)): # for flipped signs, flip upper/lower
            if flip[i]:
               x=ps[0][i]
               ps[0][i] = ps[2][i]
               ps[2][i] = x
         mvis.append(vs[1])
         mlower.append(vs[0])
         mupper.append(vs[2])
         mcphase.append(ps[1])
         mclower.append(ps[0])
         mcupper.append(ps[2])
      mvis    = np.transpose(np.array(mvis))
      mlower  = np.transpose(np.array(mlower))
      mupper  = np.transpose(np.array(mupper))
      mcphase = np.transpose(np.array(mcphase))
      mclower = np.transpose(np.array(mclower))
      mcupper = np.transpose(np.array(mcupper))
      return [mvis, mlower, mupper, mcphase, mclower, mcupper]

   def review5(self, dotels=True, dostat=False, last=400, resid=False):
      im   = self.im
      obs  = self.obs
      md   = obs.matissedata
      waves= obs.getwave()
      nwaves=len(waves)
      ok2,ok3 = obs.getok()
#     ok2  = md['wt2'][:,0]!=0
#     ok3  = md['wt3'][:,0]!=0
#     vis  = md['vis'][ok2]
#     cphase = md['cphase'][ok3]
#     tel2 = md['tel2'][ok2]
#     tel3 = md['tel3'][ok3]
#     flip = md['cflip3'][ok3]
      tel2 = obs.get2data('tel2')
      tel3 = obs.get3data('tel3')
      flip = obs.get3data('cflip3')
      ut3  = np.unique(tel3)

      vis,cphase = self.getdata()
      if (dostat):
         mvis, mlower, mupper, mcphase, mclower, mcupper = \
            self.getmultistatmodels()
#        mvis    = []   # model visibilities
#        mupper  = []   # upper bounds
#        mlower  = []
#        mcphase = []   # model closure phases
#        mcupper = []
#        mclower = []

#        samples = np.load(self.tag+'samples.npy')
#        vs,ps = self.obs.visstat(self.im, samples, last=last)
#        vs  = vs[:,ok2]
#        ps  = ps[:,ok3]
#        for i in range(len(waves)):
#           mvis.append(vs[1,ok2,i])
#           mlower.append(vs[0,ok2,i])
#           mupper.append(vs[2,ok2,i])
#           mcphase.append(ps[1,ok3,i])
#           mclower.append(ps[0,ok3,i])
#           mcupper.append(ps[2,ok3,i])
#        mvis    = np.transpose(np.array(mvis))
#        mlower  = np.transpose(np.array(mlower))
#        mupper  = np.transpose(np.array(mupper))
#        mcphase = np.transpose(np.array(mcphase))
#        mclower = np.transpose(np.array(mclower))
#        mcupper = np.transpose(np.array(mcupper))
      else:
         if(self.multimodels is None): # single model
            valid,mvis,mcphase = obs.modeldata(im)
            mvis = mvis[ok2]
            mcphase = mcphase[ok3]
         else:  # multi models
            mvis,mcphase = self.getmultimodels()
      setpltwindow(1, 800,600, 700, 50)
      plt.clf()

      title = 'visibilities '
      colors = ['b','r','g','k','m','c','y']
      for w in range(len(waves)):
         if(nwaves==1):
            vw = vis
            mvw= mvis
         else:
            vw = vis[:,w]
            mvw=mvis[:,w]
         if dostat:
            if(nwaves==1):
               muw = mupper[:,w]
               mlw = mlower[:,w]
            else:
               muw = mupper
               mlw = mlower
         if not dostat:
            for v,m,t in zip(vw,mvw,tel2):
               if resid: y = v-m
               else:     y = m
               plt.plot(v, y, '.'+colors[w])
               if dotels:
                  plt.text(v,y,t,color=colors[w])
         else:
            for v,m,ml,mu,t in zip(vw,mvis[:,w],
               mlw, muw,tel2):
               if resid: y = v-m
               else:     y = m
               plt.plot(v, y, '.'+colors[w])
               plt.plot(v*np.ones(2), [ml,mu],color=colors[w])
               if dotels:plt.text(v,y,t,color=colors[w])
      ymax = mvis.max()
      if(resid): ymax = .1
      for w in enumerate(obs.wavedata.keys()):
         plt.plot(.0,  ymax-.005*w[0], '.'+colors[w[0]])
         plt.text(.002,ymax-.005*w[0], w[1], color='k')
      if resid:plt.plot([0,.15],[0,.00],'k')
      else:plt.plot([0,.15],[0,.15],'k')
      plt.title(title)
      plt.xlabel('measured vis**2')
      plt.ylabel('model vis**2')
      plt.tight_layout()
      mpause(.1, True)
      plt.clf()
#              cphases, 
      cmin  = cphase.min();cmax=cphase.max()
      mcmin = mcphase.min();mcmax=mcphase.min()
      if (mcmin < cmin):cmin = mcmin
      if (mcmax > cmax):cmax = mcmax
      for w in range(len(waves)):
         mp = mcphase[:,w]*flip
         if (nwaves==1):
            cp = cphase
         else:
            cp = cphase[w]
         if resid: y  = rphase(mp,cp)
         else:     y =  mp
         plt.plot(cp,y,'.'+colors[w])
         if (dostat):
            if(nwaves==1):
               lp = mclower
               up = mcupper
            else:
               lp = mclower[:,w]
               up = mcupper[:,w]
                    
#           if (up-mp)>180:up = up-360
#           if (lp-mp)>180:lp = lp-360
#           if (mp-lp)>180:lp = lp+360
#           plt.plot(cp*np.ones(2),[lp,up],color=colors[w])
         if dotels:
            for a,b,c in zip(cp,y,tel3):
               plt.text(a,b,c,color=colors[w])
      if (resid): mcmax=110
      for w in enumerate(obs.wavedata.keys()):
         plt.plot(cmin,  cmax-7*w[0], '.'+colors[w[0]])
         plt.text(cmin,  cmax-7*w[0], w[1], color='k')
         
      plt.title('closure phases')
      if resid:plt.plot([cmin,cmax],[0,0],'k')
      else:plt.plot([cmin,cmax],[cmin,cmax],'k')
      plt.xlabel('measured cphase, degrees')
      plt.ylabel('model cphase, degrees')
      mpause(.1)
      return

   def check_uvvis(self, showmodel=False, scale=150):
      obs = self.obs
      md  = obs.matissedata
      ok2 = md['wt2'][:,0] != 0
      uu  = md['u'][ok2]
      vv  = md['v'][ok2]
      vis = md['vis'][ok2]
      tel = md['tel2'][ok2]

      if (showmodel): mvis = obs.modelvis(self.im)
      else:mvis = vis
      colors = ['b','r','g','k','m','c','y']

      plt.figure(1)
      plt.clf()

      nwave = vis.shape[1]
      for wave in range(nwave):
         c = colors[wave]
         tlist = []
         for u,v,i,m,t in zip(uu,vv,vis[:,wave],mvis[:,wave],tel):
            plt.plot(u,v,'o'+c, fillstyle='none', markersize=scale*i)
            plt.plot(-u,-v,'o'+c, fillstyle='none', markersize=scale*i)

            if (showmodel):
               plt.plot(u,v,'+'+c, markersize=scale*m)
               plt.plot(-u,-v,'+'+c, markersize=scale*m)

            if (t not in tlist):
               plt.text(u+3,v+3,t)
               plt.text(-u+3,-v+3,t)
               tlist.append(t)
      umax = np.max(np.abs(uu))
      plt.xlim((1.2*umax,-1.2*umax))
      return

   def plotraw(self, output=None, title='', modelwaves = None, tags = None,
      last=400, noise=0.03,mmodels=False,nx=3,ny=4,vmax=.2):
#             get raw data, all wavelengths
      rawobs = getobsf(self.datafile, raw=True, weightdata=self.weightdata)
#                  collect observed data
      md          = rawobs.matissedata
      rawwave     = rawobs.wave
      if (rawwave.max()<7): band='L'
      else:band='N'
      print('band',band)
      obsn        = md['obsn']
      obsc        = md['obsc']
      days        = np.array(rawobs.daynames)
      odays       = days[obsn]
      odays3      = days[obsc]
      wt2         = md['wt2']  # weights
      wt3         = md['wt3']  # weights
      tel3        = md['tel3']
      tel2        = md['tel2']
      vis         = md['vis']
      cphase      = md['cphase']
      uu          = md['u']
      vv          = md['v']
      bl          = md['bl']
      pa          = md['pa']
      flip        = md['cflip3']
      ok2 = wt2> 0
      ok3 = wt3> 0

      vis = vis[ok2]
      uu  = uu[ok2]
      vv  = vv[ok2]
      cphase  = cphase[ok3]
      bl    = bl[ok2]
      pa    = pa[ok2]
      tel3  = tel3[ok3]
      tel2  = tel2[ok2]
      obsn  = obsn[ok2]
      obsc  = obsc[ok3]
      odays = odays[ok2]
      odays3= odays3[ok3]
      flip  = flip[ok3]
      for i in range(len(flip)):cphase[i]*=flip[i]

#                       compute and collect model data
      mvis    = []
      mcphase = []
      mupper  = None
      mlower  = None
      mcupper = None
      mclower = None
      multi = (modelwaves is None)  # models computed for many wavelengths
      if (multi):
         waves   = self.obs.getwave()
         mupper  = []   # upper bounds
         mlower  = []
         mcupper = []
         mclower = []
      else:
         waves   = modelwaves

      if (multi & (not mmodels)): #  use statistical models and not best models 
         if (self.multitags is None):# single model for all wavelengths
            stag = self.tag+'samples.npy'
            print('retrieving samples from ',stag)
            samples = np.load(stag)
            vs,ps = self.obs.visstat(self.im, samples, last=last)
            vs  = vs[:,ok2]
            ps  = ps[:,ok3]
            for i in range(len(waves)):
               mvis.append(vs[1,:,i])
               mlower.append(vs[0,:,i])
               mupper.append(vs[2,:,i])
               mcphase.append(ps[1,:,i])
               mclower.append(ps[0,:,i])
               mcupper.append(ps[2,:,i])
            mvis    = np.array(mvis)
            mlower  = np.array(mlower)
            mupper  = np.array(mupper)
            mcphase = np.array(mcphase)
            mclower = np.array(mclower)
            mcupper = np.array(mcupper)
            mvis    = np.transpose(mvis)
            mcphase = np.transpose(mcphase)
            mlower  = np.transpose(mlower)
            mupper  = np.transpose(mupper)
            mclower = np.transpose(mclower)
            mcupper = np.transpose(mcupper)
         else: #  there are stat models run at each wavelegth
            mvis,mlower,mupper,mcphase,mclower,mcupper=\
               self.getmultistatmodels(last=last)
      elif(multi & mmodels):  # don't use stat models but stored best models
         mvis,mcphase = self.getmultimodels()
#        mvis    = np.transpose(mvis)
#        mcphase = np.transpose(mcphase)
      else:
         wd = {'wave':[[3.0,5.0]]}
         for w in waves:
            wd['wave'] = [[w-.05,w+.05]]
            obsw     = getobsf(self.datafile,wavedata=wd,
               weightdata=self.weightdata)
            valid,vs,ps = obsw.modeldata(self.im)
            vs  = vs[ok2,0]
            ps  = ps[ok3,0]
            mvis.append(vs)   # model visibility statistics
            mcphase.append(ps)   # model visibility statistics
#                 combine data at same baselines
      mvis    = np.array(mvis)
      mcphase = np.array(mcphase)
      utel    = np.unique(tel2)
      v  = []  # list of visibilities with same bcds grouped
      bb = []  # baselines
      pp = []  # baselines
      tt = []  #tel number
      dd = []  #date 
      mv = []  #modelvis
      mu = []  #upper lim
      ml = []  #lower lim
      for ut in utel:
         tu = []
         for i in range(len(tel2)):
            if(tel2[i]==ut):tu.append(i)
         v.append(vis[tu])
         mv.append(np.mean(mvis[tu],0))
         bb.append(np.mean(bl[tu]))
         pp.append(np.mean(pa[tu]))
         tt.append(tel2[tu[0]])
         dd.append(odays[tu[0]])
         if (multi & (not mmodels) ):
            mu.append(np.mean(mupper[tu],0))
            ml.append(np.mean(mlower[tu],0))
      vis = v
      nvis = len(vis)
      mv  = np.array(mv);
      bb  = np.array(bb)
      if (multi & (not mmodels)):
         mu=np.array(mu);ml=np.array(ml)
#               model visibilities, upper/lower error estimates
      if(band=='L'):
         wok1 = (rawwave>3.2) & (rawwave<4.02)
         wok2 = (rawwave>4.55) & (rawwave<5)
      else:
         wok1 = (rawwave>8.1) & (rawwave<13)
         wok2 = (rawwave>8.1) & (rawwave<8.0)
      plt.close('all')
      if (output is not None):pdf = PdfPages(output)
      setpltwindow(1, w=700,h=600, x=700, y=50)
      fig = None
#     nx  = 4
#     ny  = 3
      nxy = nx*ny
      for i in range(nvis):
         if(np.mod(i,nxy)==0):
            if (fig is None):
               fig,ax = plt.subplots(ny,nx, sharex=True, sharey=True,num=1)
            else:
               fig.clear()
               ax = fig.subplots(ny,nx, sharex=True, sharey=True)
            for ix in range(nx):ax[2,ix].set_xlabel('$\lambda(\mu$m)')
            for ix in range(nx):
               if band=='L':
                  for iy in range(ny):ax[iy,ix].set_xticks(3+0.5*np.arange(5))
               else:
                  for iy in range(ny):ax[iy,ix].set_xticks(8+1.0*np.arange(6))
            for iy in range(ny):ax[iy,0].set_ylabel('vis**2')
         ix = np.mod(i,nx)
         iy = np.mod(i,(nx*ny))//nx
         if(band=='L'):ax[iy,ix].set_xlim((3.2,5.))
         else:ax[iy,ix].set_xlim((8.0,13.5))
         ax[iy,ix].set_ylim((0,vmax))
         ut = 'UT'+tt[i][0]+'UT'+tt[i][1]
         d  = dd[i]
         uv = f'{ut:7} BL={bb[i]:3.0f} PA={pp[i]:3.0f}'
         if (band=='L'):wmin=3.2
         else:wmin=8.0
         ax[iy,ix].text(wmin,vmax-.08, uv)  # tel/uvcoords
         ax[iy,ix].text(wmin,vmax-.04, dd[i]) # date
         for vvv in vis[i]:
            ax[iy,ix].plot(rawwave[wok1], vvv[wok1],'lightgrey')
            ax[iy,ix].plot(rawwave[wok2], vvv[wok2],'lightgrey')
         vvv = np.mean(vis[i],0)
         ax[iy,ix].plot(rawwave[wok1], vvv[wok1],'b')
         ax[iy,ix].plot(rawwave[wok2], vvv[wok2],'b')
         y    = []
         yerr = []
         n2   = noise**2
         for j in range(len(waves)):
            y.append(mv[i][j])
            if (multi & (not mmodels)):
               u = mu[i][j] - mv[i][j]
               u = np.sqrt(u**2+n2)
               l = ml[i][j] - mv[i][j]
               l = np.sqrt(l**2+n2)
            else:
               u = noise; l = noise
            yerr.append([l,u])
         y = np.array(y)
         yerr = np.transpose(np.array(yerr))
         ax[iy,ix].errorbar(waves, y, yerr, fmt='o')
         if(band=='L'):ax[iy,ix].fill_between([4.0,4.5],0,vmax,color='lightgrey')

         if(np.mod(i,nxy)==(nxy-1)):
            mpause(.1,True)
            if(output is not None):pdf.savefig()
      if(np.mod(i,nxy) != (nxy-1)):
         mpause(.1,True)
         if(output is not None):pdf.savefig()
#                     closure phases
      ut3   = np.unique(tel3)
      cp    = []
      mcp   = []
      mu    = []
      ml    = []
      tt    = []
      dd    = []
      for ut in ut3:
         flip = 1
         pp   = []
         for i,t in enumerate(tel3):
            if(t==ut):
               pp.append(cphase[i])
               if(flip==1):
                  mcp.append(mcphase[i])
                  tt.append(tel3[i])
                  dd.append(odays3[i])
                  if (multi & (not mmodels)):
                     mu.append(mcupper[i])
                     ml.append(mclower[i])
               flip = -1
         pp = np.array(pp)
         cp.append(pp)
      if (multi & (not mmodels)):
         mu = np.array(mu);ml=np.array(ml)
      mcp=np.array(mcp)
#     wu.msave('mcphase.pk',[mcp, mu, ml, bb, tt, dd])
      noise = 5.
      ncp  = len(cp)
#     setpltwindow(1, w=700,h=600, x=700, y=50)
      for i in range(ncp):
         if(np.mod(i,nxy)==0):
            fig.clear()
            ax = fig.subplots(ny,nx, sharex=True, sharey=True)
            for iy in range(ny):
               ax[iy,0].set_ylabel('cphase(deg)')
               ax[iy,0].set_yticks(-180+60*np.arange(7))
            for ix in range(nx):
               ax[1,ix].set_xlabel('$\lambda(\mu$m)')
               if (band=='L'):
                  for iy in range(ny):ax[iy,ix].set_xticks(3+0.5*np.arange(5))
               else:
                  for iy in range(ny):ax[iy,ix].set_xticks(8+1.0*np.arange(6))
         ix = np.mod(i,nx)
         iy = np.mod(i,nxy)//nx
         if (band=='L'):ax[iy,ix].set_xlim((3.2,5))
         else:ax[iy,ix].set_xlim((8,13.5))
         ax[iy,ix].set_ylim((-200,180))
         for pp in cp[i]:
            pp[pp>120] -=360.
            ax[iy,ix].plot(rawwave[wok1], pp[wok1],'lightgrey')
            ax[iy,ix].plot(rawwave[wok2], pp[wok2],'lightgrey')
         r  = np.radians(cp[i])
         r  = np.mean(np.exp(1j*r),0)
         pp = np.angle(r,True)
         pp[pp>120] -=360.
         ax[iy,ix].plot(rawwave[wok1], pp[wok1],'b')
         ax[iy,ix].plot(rawwave[wok2], pp[wok2],'b')
         ut = 'UT'+tt[i][0]+'UT'+tt[i][1]+'UT'+tt[i][2]
         ax[iy,ix].text(wmin, 80, ut)
         ax[iy,ix].text(wmin,130, dd[i])
         y    = []
         yerr = []
         n2   = noise**2
         for j in range(len(waves)):
#           if (tt[i]=='2343'):
#              if (mcp[i][j] >0): mcp[i][j]-=360
#              if (mu[i][j] >0): mu[i][j]-=360
#              if (ml[i][j] >0): ml[i][j]-=360
            y.append(mcp[i][j])
            if (multi & (not mmodels)):
               u = mu[i][j] - mcp[i][j]
               u = np.sqrt(u**2+n2)
               l = ml[i][j] - mcp[i][j]
               l = np.sqrt(l**2+n2)
            else:
               u = noise;l=noise
            yerr.append([l,u])
         y = np.array(y)
         yerr = np.transpose(np.array(yerr))
         ax[iy,ix].errorbar(waves, y, yerr, fmt='o')
         if(band=='L'):ax[iy,ix].fill_between([4.0,4.5],-200,180,color='lightgrey')
         if(np.mod(i,nxy)==(nxy-1)):
            mpause(.1,True)
            if(output is not None):pdf.savefig()
      if(np.mod(i,nxy) != (nxy-1)):
         mpause(.1,True)
         if(output is not None):pdf.savefig()
      if(output is not None):pdf.close()
      return

   def resetfiles(self, wave, prefix, 
      sinput='',soutput=''):
      self.obs = getobsf(self.datafile, wave=[float(wave)], 
         weightdata=self.weightdata)

      ww = wave[0]+wave[2]
      self.tag = prefix+ww
      self.logfile   = self.tag +'.log'
      self.modelfile = prefix+ww+sinput
      self.outmodel  = prefix+ww+soutput
      self.im = getimf(self.modelfile)

   def image(self, s=256, pxscale=.5):#LMbands128
      
       lgrid,mgrid = np.meshgrid(np.arange(-s*pxscale/2,s*pxscale/2,pxscale), 
          np.arange(-s*pxscale/2,s*pxscale/2,pxscale))
       return self.im.image(lgrid, mgrid)

   def multitune(self):
      newwaves = []
      newtags  = []
      newout   = []
      self.multimodels = []
      self.multilike   = []
      oldobs   = self.obs
#       make a new set of wavedata structures for each wavelength separately
      for k in self.wavedata.keys():
         newwaves.append({k:self.wavedata[k]})
         newtags.append(self.tag+k)
         newout.append(self.outmodel+k)
      self.multitags = newtags
#        loop over wavelength: modify tag, outmodel, obs
      for tag,wave,out in zip(newtags,newwaves,newout):
         self.tag      = tag
         self.output   = tag
         self.wavedata = wave
         self.outmodel = out
         self.obs      = getobsf(self.datafile,wavedata=wave,
            weightdata=self.weightdata)
         self.im       = getimf(self.modelfile)
#        for c in self.im.comps:c.setfixed('s',False) #set slope fixed
         print('processing wave ',wave)
         self.runscript()
         self.multimodels.append(self.bestvalues)
         self.multilike.append(mf.lnlike(self.bestvalues,'mix',self))
      self.wavedata = getbandf(self.wavefile)
      self.obs      = getobsf(self.datafile, wavedata=self.wavedata,
                      weightdata = self.weightdata)
      self.colorimage()
      return

   def makemultimodels(self):
      newwaves = []
      newtags  = []
      self.multilike   = []
      self.multimodels = []
      for k in self.wavedata.keys():
         newwaves.append({k:self.wavedata[k]})
         newtags.append(self.tag+k)
      self.multitags = newtags
      for tag in newtags:
         samples,probs = self.getstats(tag)
         a = np.argmax(probs)
         self.multimodels.append(samples[a])
         self.multilike.append(probs[a])


   def colorimage(self, xs=0, ys=0, sat=3, s=64, pxscale=.5):
      lgrid,mgrid = np.meshgrid(np.arange(-s*pxscale/2,s*pxscale/2,pxscale), 
         np.arange(-s*pxscale/2,s*pxscale/2,pxscale))
      sh = lgrid.shape
      waves  = self.obs.getwave()
      output = np.zeros((sh[0], sh[1], 3))  # will contain rgb pixel values
      xx = (waves.max()-waves)/(waves.max()-waves.min())
      for model,x in zip(self.multimodels,xx):
         self.im.setvaluelist(model)
         ii = self.im.image(lgrid+xs, mgrid+ys) # image version of model
         ii/=(ii.max()*len(waves)) # normalize to 1/# of wavelengths
         c = cc(x)  # color triplet for this wavelength
         for i in range(3):output[:,:,i] += c[i]*ii
      plt.clf()
      plt.imshow(sat*output, origin='lower')
      xp = []; xl = []
#     px = 2*(s//8)
      px = (s//8)
      for i in range(8):
         xp.append(px*i) # pixel location for ticks
         xl.append(f'{(i-4)*px*pxscale:3.1f}')  # value in mas
      plt.xticks(xp,xl)
      plt.yticks(xp,xl)
      plt.xlabel('milliarcsec')
      plt.ylabel('milliarcsec')
   

   def test3ellipses(self,l0,m0,last=1000):
      mimgs=np.zeros((3,512,512))
      mat=[43,84,38]
      mis=[11,46,98]
      dxs=[0,-6.0,8.0]
      dys=[0,19.0,-10.0]
      pas=[-43-90,-31-90,-45]
      outflux=[]
      for i in range(3):
         Ei_ell_center = (l0+dxs[i]*2.0, m0+dys[i]*2.0)
         Ei_ell_width = mas[i]*2.
         Ei_ell_height = mis[i]*2. 
         cos_angle = np.cos(np.radians(180.-pas[i]))
         sin_angle = np.sin(np.radians(180.-pas[i]))
         xc=[]; yc=[]
         xct=[]; yct=[]
         rad_cc=[]
         count=0
         for l in range(512):
            for m in range(512):
               xc=l - Ei_ell_center[0]
               yc=m - Ei_ell_center[1]
               xct=xc * cos_angle - yc * sin_angle 
               yct=xc * sin_angle + yc * cos_angle 
               rad_cc=(xct**2/(Ei_ell_width/2.)**2) + (yct**2/(Ei_ell_height/2.)**2)
               if(rad_cc<=1.):  #inside
                  mimgs[i,m,l]=1.0
         
      outflux=(self.relf3ei(mimgs,last=last))
      return outflux  #mimgs

   def test12ellipses(self,l0,m0,last=1000):
      mimgs=np.zeros((14,512,512))  #e1,e2,e3,e4,e5,de1,de2,de3,nc,sc,t,void
      mas=[10,8,9,10,9,68,42,63,28,30,34,9]#
      mis=[7,8,7,8,7,37,20,30,7,8,3,7]
      dxs=[0,-7.5,-6,7,0,-17,4,20.5,-6,-2,3,2]#
      dys=[0,-3,4.5,7,11,17,-19.5,35,6,-17,-5,9]#
      pas=[132-90,170-90,146-90,146-90,146-90,333-90,315-90,10-90,148-90,139-90,130-90,146-90]#
      outflux=[]
      col=['lime','orange','blue','green','brown','red','yellow','lightblue','purple','cyan','magenta','white']
      col2=['cyan','cyan','cyan','cyan','cyan','purple','purple','purple','black','magenta']
      fig,ax = plt.subplots(1)#to plot 5 ell
      tag='rNe10d'#to plot 5 ell
      ax.set_aspect('equal')#to plot 5 ell
      for i in range(12):
         Ei_ell_center = (l0+dxs[i]*2.0, m0+dys[i]*2.0)
         Ei_ell_width = mas[i]*2.
         Ei_ell_height = mis[i]*2.
         Ei_ellipse = patches.Ellipse(Ei_ell_center, Ei_ell_width, Ei_ell_height, angle=pas[i], fill=False, edgecolor=col2[i], linewidth=1)
         cos_angle = np.cos(np.radians(180.-pas[i]))
         sin_angle = np.sin(np.radians(180.-pas[i]))
         xc=[]; yc=[]
         xct=[]; yct=[]
         rad_cc=[]
         count=0
         for l in range(512):
            for m in range(512):
               xc=l - Ei_ell_center[0]
               yc=m - Ei_ell_center[1]
               xct=xc * cos_angle - yc * sin_angle
               yct=xc * sin_angle + yc * cos_angle
               rad_cc=(xct**2/(Ei_ell_width/2.)**2) + (yct**2/(Ei_ell_height/2.)**2)
               if(rad_cc<=1.):  #inside
                  mimgs[i,m,l]=1.0

         ax.add_patch(Ei_ellipse)#to plot 5 ell
      
      #for LM bands go to allegro6/matisse/gamez/rotmodels/LMbands
      hdu2=fits.open('/allegro6/matisse/gamez/UT.11.0-12.0mu.fits')
      hdu=fits.open(tag+'0lg.fits')#to plot the large image of the model with the ellipses
      imgmod2=scipy.ndimage.zoom(hdu2[0].data,2.65624)#to plot 5 ell for img rec
      imgmod=hdu[0].data#to plot 5 ell
      hdu.close()#to plot 5 ell
      plt.imshow(imgmod2,origin='lower',cmap=cm,alpha=1.0)#inferno imgrec
      #plt.imshow(imgmod,origin='lower',cmap='gist_earth')#to plot 5 ell  
      cx=340-172#ctrx (340-174)  for 174 
      cy=170
      #ax.set_aspect('equal')#to plot 5 ell
      plt.xlim(label_locs[2],label_locs[8])
      plt.ylim(label_locs[2],label_locs[10])
      #plt.savefig(str(tag)+'new5ellipses3DE')#to plot 5 ell
      outflux=(self.relfei(mimgs,last=last))
      return outflux  #mimgs



   def relf3ei(self,mimgs,last=1000):
      nimg=mimgs.shape[0]
      pxscale=0.5
      s=512
      lgrid,mgrid = np.meshgrid(np.arange(-s*pxscale/2,s*pxscale/2,pxscale), np.arange(-s*pxscale/2,s*pxscale/2,pxscale))
      samples=np.load(self.tag+'samples.npy')
      samples=samples[-last:]
      im=self.im
      outflux=np.zeros((nimg,last))
      totf=np.zeros((last))
      for n,i in enumerate(samples):
         im.setvaluelist(i)
         image=im.image(lgrid,mgrid)
         totf[n]=np.sum(image)
         for l in range(nimg):
            outflux[l,n]=np.sum(mimgs[l]*image)

      outflux[1]-=outflux[0]
      outflux[2]-=outflux[0]
      outrf=outflux/totf
      outrfmean=np.mean(outrf,1)#sample dimension
      outstdrf=np.std(outrf,1)
      return outrfmean,outstdrf

   def relfei(self,mimgs,last=1000):
      nimg=mimgs.shape[0]
      pxscale=0.5
      s=512
      lgrid,mgrid = np.meshgrid(np.arange(-s*pxscale/2,s*pxscale/2,pxscale), np.arange(-s*pxscale/2,s*pxscale/2,pxscale))
      samples=np.load(self.tag+'samples.npy')
      samples=samples[-last:]
      im=self.im
      outflux=np.zeros((nimg,last))
      totf=np.zeros((last))
      for n,i in enumerate(samples):
         im.setvaluelist(i)
         image=np.rot90(im.image(lgrid,mgrid),2)
         totf[n]=np.sum(image)
         for l in range(nimg):
            outflux[l,n]=np.sum(mimgs[l]*image)

      outflux[12]=outflux[5]-outflux[8]#DE1-DE4
      outflux[13]=outflux[6]-outflux[9]#DE2-DE5
      outrf=outflux/totf
      outrfmean=np.mean(outrf,1)#sample dimension
      outstdrf=np.std(outrf,1)
      return outrfmean,outstdrf

def getrelf3ell(l0s,m0s,last=1000):
    table=[]
    for i in range(np.size(ltags)):
      r=rundata(ltags[i]+'3script')
      meanrf, stdrf= r.test3ellipses(l0s[i],m0s[i],last=last)
      print(ltags[i],meanrf,stdrf)
      table.append([ltags[i],meanrf,stdrf])
    return table

def getrelf12ell(ltags,l0s,m0s,last=1000):
    table=[]
    for i in range(np.size(ltags)):
      r=rundata(ltags[i]+'3script')#for N band go to '/allegro6/matisse/gamez/rotmodels/
      meanrf, stdrf= r.test12ellipses(l0s[i],m0s[i],last=last)
      print(ltags[i],meanrf,stdrf)
      table.append([ltags[i],meanrf,stdrf])
    return table


