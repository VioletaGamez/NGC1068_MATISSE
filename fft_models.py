import numpy as np
import scipy.special as ss
import matplotlib.pyplot as plt
import wutil as wu
from img2vis import lmimage

def airy(x):
   xx = 2*np.clip(np.abs(x),.001,None)
   return 2*ss.jv(1,xx)/xx

class component():
   def __init__(self,name, labels=['a','b','pa','amp','l','m','s']): 
      ''' base class for components
      name = name of component type (e.g. 'gaussian')
      paramnames: list of parameter names assumed float
      '''
      self.labels    = np.array(labels)
                                                    # a,b,l,m in mas
                                                    # l,m positive to W, N
      self.name      = name
      self.values    = np.zeros(len(self.labels))
      self.lbound    = np.zeros(len(self.labels))
      self.ubound    = np.zeros(len(self.labels))
      self.notfixed  = np.ones(len(self.labels),bool)
      self.mas2rad   = 1./2.062468e8
      self.pi2       = 2*np.pi
      self.masm      = self.pi2*1.e6*self.mas2rad  # factor for baseline(m)*size(mas)/wave(mu)
      self.refwave   = 4.0

   def nparams(self):return len(self.labels)
   def nnparams(self):return np.sum(self.notfixed)
   def name(self):return self.name
   def getvalues(self):return self.values[self.notfixed]
   def getallvalues(self):return self.values
   def setallvalues(self, values):self.values = values.copy()
   def getlbound(self):return self.lbound[self.notfixed]
   def getubound(self):return self.ubound[self.notfixed]
   def getalllabels(self):return self.labels
   def getlabels(self):
      ll = []
      for l,nf in zip(self.labels, self.notfixed):
         if(nf):ll.append(l)
      return ll

   def setlbound(self,label, value):
      for i in range(len(self.labels)):
         if(self.labels[i]==label): 
            self.lbound[i] = value
            break
   def setubound(self,label, value):
      for i in range(len(self.labels)):
         if(self.labels[i]==label): 
            self.ubound[i] = value
            break
   def setvalue(self,label, value):
      islabel = False
      for i in range(len(self.labels)):
         if(self.labels[i]==label): 
            self.values[i] = value
            islabel = True
            break
      if(not islabel):print('label ',label,' not found')
      return
   def getvalue(self,label):
      for i in range(len(self.labels)):
         if(self.labels[i]==label): 
            return self.values[i] 
            break

   def setvaluelist(self,values):
      self.values[self.notfixed] = values
   def setuboundlist(self,values):
      self.ubound[self.notfixed] = values
   def setlboundlist(self,values):
      self.lbound[self.notfixed] = values

   def setfixed(self,label, value=False):
      for i in range(len(self.labels)):
         if(self.labels[i]==label): 
            self.notfixed[i] = value
            break

   def ellipsevalues(self,u,v,lam):   #  retrieve and convert values for
                               # standard 2d ellipses
      masm   = self.masm/lam #2pi Br/lambda B(meters);r(mas);lam(mu)
      values = {}
      for l in ['a','b','pa','l','m','amp','s']:
         values[l] = self.values[np.where(self.labels==l)][0]
      a    = values['a']*self.fwhm2sigma # sigma instead of fwhm 
      b    = values['b']*self.fwhm2sigma # sigma instead of fwhm
      pa   = values['pa']
      l    = values['l'] * masm # mas->radians
      m    = values['m'] * masm # mas->radians
      amp  = values['amp'] * (lam/self.refwave)**values['s']
#                check values for valid ranges
      ok   = (a > 0.) & (b>0.)&(a>=b)
      ok   = ok & (amp > 0) & (pa>(-90)) & (pa <100)
#     ok   = ok &  (pa>(-90)) & (pa <100)
      pa   = np.radians(values['pa']) # position angle deg->radians
      c    = np.cos(pa); s = np.sin(pa)
      um   = (u*c - v*s)*masm  # u,v in rotated coordinates
      vm   = (u*s + v*c)*masm # if pa>0 and u>0&v>0, vm>v
      values['um'] = um
      values['vm'] = vm
      values['r2'] = (a*vm)**2 + (b*um)**2   # squared radius in elliptical coord
      values['u.l'] = amp * np.exp(-1j*(-u*l + v*m)) # displacement phase shift

      return [ok,values]

   def cflux(self,u,v,lam):  # correlated flux, lam in microns,u,v, in meters, u,v pos to W,N
      return [True,1+0j]

   def image(self,lgrid, mgrid):
      out = 0*lgrid
      out[(lgrid==0)&(mgrid==0)] = 1.
      return out

class extra(component):
   def __init__(self, labels):
      super().__init__('extra', labels=labels)
   def cflux(self,u,v,lam):return [True,0+0j]
   def image(self, lgrid, mgrid): return 0*lgrid

class gauss(component):
   def __init__(self):
      super().__init__('gauss')
      self.fwhm2sigma = 1./np.sqrt(8*np.log(2))

   def cflux(self,u,v,lam):   # correlated flux, lam in microns, u,v, in meters, u,v pos to W,N
      valid,values = self.ellipsevalues(u,v,lam)
      evalue = .5*values['r2']
#     evalue = np.sqrt(evalue/4)  # slower drop
      return   [valid,values['u.l']*np.exp(-evalue)]

   def image(self,lgrid, mgrid):  # l,m in mas, 
      a    = self.values[0]*self.fwhm2sigma # sigma in radians
      b    = self.values[1]*self.fwhm2sigma # sigma in radians
      pa   = np.radians(self.values[2])
      c    = np.cos(pa); s = np.sin(pa)
      lm   =  (lgrid - self.values[4])*c + (mgrid - self.values[5])*s
      mm   = -(lgrid - self.values[4])*s + (mgrid - self.values[5])*c
      flux = self.values[3]*np.exp(-((lm/b)**2+(mm/a)**2)/2)/(2*np.pi*a*b)
      return flux

class point(gauss):
   def __init__(self):
      super().__init__()
      self.name = 'point'
      self.values[0] = 5. #a
      self.values[1] = 5. #b
      self.values[2] = 0. #pa
      self.labels[0] = 'a' #a
      self.labels[1] = 'b' #b
      self.labels[2] = 'pa' #pa
      self.notfixed[0:3] = False
      self.rvalues = None

   def ellipsevalues(self,u,v,lam):   #  retrieve and convert values for
                               # standard 2d ellipses
      values = {}
      if (self.rvalues is  None):
         masm   = self.masm/lam #2pi Br/lambda B(meters);r(mas);lam(mu)
         for l in ['a','b','pa','l','m','amp','s']:
            values[l] = self.values[np.where(self.labels==l)][0]
         a    = values['a']*self.fwhm2sigma # sigma instead of fwhm 
         b    = values['b']*self.fwhm2sigma # sigma instead of fwhm
         pa   = values['pa']
         l    = values['l'] * masm # mas->radians
         m    = values['m'] * masm # mas->radians
         amp  = values['amp'] * (lam/self.refwave)**values['s']
#                check values for valid ranges
         ok   = (a > 0.) & (b>0.)&(a>=b)
#        ok   = ok & (amp > 0) & (pa>(-90)) & (pa <100)
         ok   = ok  & (pa>(-90)) & (pa <100)
         pa   = np.radians(values['pa']) # position angle deg->radians
         c    = np.cos(pa); s = np.sin(pa)
         um   = (u*c - v*s)*masm  # u,v in rotated coordinates
         vm   = (u*s + v*c)*masm # if pa>0 and u>0&v>0, vm>v
         values['um'] = um
         values['vm'] = vm
         values['r2'] = (a*vm)**2 + (b*um)**2   # squared radius in elliptical coord
         self.rvalues = values
      else: 
         values = self.rvalues
         for l in ['l','m','amp','s']:
            values[l] = self.values[np.where(self.labels==l)][0]
         l = values['l']
         m = values['m']

      amp  = values['amp'] * (lam/self.refwave)**values['s']
      values['u.l'] = amp * np.exp(1j*(-u*l + v*m)) # displacement phase shift
      return [True,values]

class disk(gauss):
   def __init__(self):
      super().__init__()

   def cflux(self,u,v,lam):   # correlated flux, lam in microns, u,v, in meters, u,v pos to W,N
      valid,values = self.ellipsevalues(u,v,lam)
      return   [valid,values['u.l']*airy(np.sqrt(values['r2']))]

   def image(self,lgrid, mgrid):  # l,m in mas, 
      a    = self.values[0]*self.fwhm2sigma # radius in radians
      b    = self.values[1]*self.fwhm2sigma # radius in radians
      pa   = np.radians(self.values[2])
      c    = np.cos(pa); s = np.sin(pa)
      lm   =  (lgrid - self.values[4])*c + (mgrid - self.values[5])*s
      mm   = -(lgrid - self.values[4])*s + (mgrid - self.values[5])*c
      im   = (lm/b)**2+(mm/a)**2 < 1
      flux = self.values[3]* im/np.sum(im)
      return flux

class ring(gauss):
   def __init__(self):
      super().__init__()

   def cflux(self,u,v,lam):   # correlated flux, lam in microns, u,v, in meters, u,v pos to W,N
      valid,values = self.ellipsevalues(u,v,lam)
      return   [valid,values['u.l']*ss.jv(0,np.sqrt(values['r2']))]

   def image(self,lgrid, mgrid):  # l,m in mas, 
      a    = self.values[0]*self.fwhm2sigma # radius in radians
      b    = self.values[1]*self.fwhm2sigma # radius in radians
      pa   = np.radians(self.values[2])
      c    = np.cos(pa); s = np.sin(pa)
      lm   =  (lgrid - self.values[4])*c + (mgrid - self.values[5])*s
      mm   = -(lgrid - self.values[4])*s + (mgrid - self.values[5])*c
      im   = np.abs((lm/b)**2+(mm/a)**2 - 1) < .35  #ring thickness = .2*radius
      flux = self.values[3]* im/np.sum(im)
      return flux

class ftn2(component):
   def __init__(self):
      super().__init__('ftn2',labels=['a','b/a','pa','amp','l','m','n2'])
      self.fwhm2sigma = 1./np.sqrt(8*np.log(2))

   def cflux(self,u,v,lam):   # correlated flux, lam in microns, u,v, in meters, u,v pos to W,N
      values = self.ellipsevalues(u,v,lam)
      farg = 0.5 * values['r2']
      nn   = self.values[np.where(self.labels=='n2')][0] # exponent of visibility powerlaw
      if (nn < 0.1): nn = .1
# inverse power law approximates gaussian for large nn 
      return values['u.l']/(1+farg/nn)**nn  

   def image(self,lgrid, mgrid):  # l,m in mas, 
      a    = self.values[0]*self.fwhm2sigma # sigma in radians
      b    = self.values[1]*self.fwhm2sigma # sigma in radians
      pa   = np.radians(self.values[2])
      c    = np.cos(pa); s = np.sin(pa)
      lm   =  (lgrid - self.values[4])*c + (mgrid - self.values[5])*s
      mm   = -(lgrid - self.values[4])*s + (mgrid - self.values[5])*c
      flux = self.values[3]*np.exp(-((lm/b)**2+(mm/a)**2)/2)/(2*np.pi*a*b)
      return flux

class expn(gauss):
   def __init__(self):
      super().__init__()

   def cflux(self,u,v,lam):   # correlated flux, lam in microns, u,v, in meters, u,v pos to W,N
      valid,values = self.ellipsevalues(u,v,lam)
      return   [valid,values['u.l']/(1+values['r2'])]

   def image(self,lgrid, mgrid):  # l,m in mas, 
      a    = self.values[0]*self.fwhm2sigma # radius in radians
      b    = self.values[1]*self.fwhm2sigma # radius in radians
      pa   = np.radians(self.values[2])
      c    = np.cos(pa); s = np.sin(pa)
      lm   =  (lgrid - self.values[4])*c + (mgrid - self.values[5])*s
      mm   = -(lgrid - self.values[4])*s + (mgrid - self.values[5])*c
      im   = 1.*((lm/b)**2+(mm/a)**2 < 1)
      flux = self.values[3]* im/np.sum(im)
      return flux

class ftn2(component):
   def __init__(self):
      super().__init__('ftn2',labels=['a','b/a','pa','amp','l','m','n2'])
      self.fwhm2sigma = 1./np.sqrt(8*np.log(2))

   def cflux(self,u,v,lam):   # correlated flux, lam in microns, u,v, in meters, u,v pos to W,N
      values = self.ellipsevalues(u,v,lam)
      farg = 0.5 * values['r2']
      nn   = self.values[np.where(self.labels=='n2')][0] # exponent of visibility powerlaw
      if (nn < 0.1): nn = .1
# inverse power law approximates gaussian for large nn 
      return values['u.l']/(1+farg/nn)**nn  

   def image(self,lgrid, mgrid):  # l,m in mas, 
      a    = self.values[0]*self.fwhm2sigma # sigma in radians
      b    = self.values[1]*self.fwhm2sigma # sigma in radians
      pa   = np.radians(self.values[2])
      c    = np.cos(pa); s = np.sin(pa)
      lm   =  (lgrid - self.values[4])*c + (mgrid - self.values[5])*s
      mm   = -(lgrid - self.values[4])*s + (mgrid - self.values[5])*c
      flux = self.values[3]*np.exp(-((lm/b)**2+(mm/a)**2)/2)/(2*np.pi*a*b)
      return flux

class multicomp():
   def __init__(self,comps):
      self.comps = comps

   def nnparams(self):
      n = 0
      for c in self.comps: n += c.nnparams()
      return n
   def ncomps(self):return(len(self.comps))
   def getcomp(self, ncomp):return self.comps[ncomp-1] # one-based
   def addcomp(self, comp):self.comps.append(comp)

   def cflux(self, u, v, lam):
      flux = 0. + 0.j
      valid= True
      for c in self.comps: 
         ok,f  = c.cflux(u,v,lam)
         valid = valid & ok
         flux += f
      return [valid,flux]

   def uvplane(self, ugrid, vgrid, lam):
      flux = 0*ugrid + 0j*ugrid
      for u,v in zip(ugrid, vgrid):
         valid,f = self.cflux(u,v,lam)
         flux   += f
      return flux
      
   def image(self, lgrid, mgrid):
      flux = 0*lgrid
      for c in self.comps: flux+= c.image(lgrid, mgrid)
      return flux

   def makeimage(self, size, dx):
      lmobj = lmimage(size, dx)
      lmobj.setimage(self.image(lmobj.lgrid, lmobj.mgrid))
      return lmobj

   def getvalues(self):
      out = np.zeros(0)
      for c in self.comps: out = np.concatenate([out,c.getvalues()])
      return out
   def getallvalues(self):
      out = np.zeros(0)
      for c in self.comps: out = np.concatenate([out,c.getallvalues()])
      return out
   def getalllabels(self):
      out = []
      for i in range(self.ncomps()):
         suf = '_'+str(i+1)
         c  = self.comps[i]
         la = c.getalllabels()
         for l in la:out.append(l+suf)
      return out
   def getubound(self):
      out = np.zeros(0)
      for c in self.comps: out = np.concatenate([out,c.getubound()])
      return out
   def getallubound(self):
      out = np.zeros(0)
      for c in self.comps: out = np.concatenate([out,c.ubound])
      return out
   def getlbound(self):
      out = np.zeros(0)
      for c in self.comps: out = np.concatenate([out,c.getlbound()])
      return out
   def getalllbound(self):
      out = np.zeros(0)
      for c in self.comps: out = np.concatenate([out,c.lbound])
      return out
   def getlabels(self):
      labels = []
      for i,c in enumerate(self.comps):
        clabels = c.getlabels()
        for cl in clabels:labels.append(cl+'_'+str(i+1))
      return labels
   def setlbound(self,cnum, label, value): self.comps[cnum-1].setlbound(label, value)
   def setubound(self,cnum, label, value): self.comps[cnum-1].setubound(label, value)
   def setvalue(self,cnum, label, value): self.comps[cnum-1].setvalue(label, value)
   def setfixed(self,cnum, label, value): self.comps[cnum-1].setfixed(label, value)
   def setvaluelist(self,vlist):
      i = 0;np=0
      for c in self.comps:
         np += c.nnparams()
         vl = vlist[i:np]
         c.setvaluelist(vl)
         i+=c.nnparams()
   def setallvalues(self,vlist):
      i = 0;np=0
      for c in self.comps:
         np += c.nparams()
         vl = vlist[i:np]
         c.setallvalues(vl)
         i+=c.nparams()
   def setuboundlist(self,vlist):
      i = 0;np=0
      for c in self.comps:
         np += c.nnparams()
         vl = vlist[i:np]
         c.setuboundlist(vl)
         i+=c.nnparams()
   def setlboundlist(self,vlist):
      i = 0;np=0
      for c in self.comps:
         np += c.nnparams()
         vl = vlist[i:np]
         c.setlboundlist(vl)
         i+=c.nnparams()

   def setbestvalues(self, bestlist):
      for k in bestlist.keys():
         label = k.split('_')
         if (label[0] != 'wave'):
            if(label[0] != 'b'):
               comp  = int(label[1])-1
               label = label[0]
               values = bestlist[k]
               self.comps[comp].setvalue(label, values[0])
               self.comps[comp].setubound(label, values[0]+values[1])
               self.comps[comp].setlbound(label, values[0]-values[1])
      return
