from numpy import *
import matplotlib.pyplot as plt
import glob
import testme as tm
import model_fitter as mf
import img2vis as i2
import numpy as np
from mcdb import wutil as wu
from matplotlib.backends.backend_pdf import PdfPages

def tokenize(s):
   ss = s.split(' ')
   out = []
   for t in ss:
      if(len(t)>0) & (t!='\n'):out.append(t.strip())
   return out

def parsefile(file):
   params = []
   data   = None
   for i in range(5):params.append({})
   i = -1
   with open(file,'r') as f:
      while True:
         ff = f.readline()
         if(len(ff)==0):break
         line = tokenize(ff)
         if(len(line)==0):continue
         if (line[0]=='wave'):
            i+=1
            params[i]['wave'] = float(line[1])
         values = []
         for l in line[1:]: values.append(float(l))
         params[i][line[0]] = array(values)
   return params

def getp(param, data):
   lp   = len(data)
   y    = zeros(lp)
   yerr = zeros((2,lp))
   wave = zeros(lp)
   for i in range(lp):
      wave[i] = data[i]['wave']
      y[i]    = data[i][param][0]
      yerr[0,i] = abs(data[i][param][2])
      yerr[1,i] = data[i][param][1]
   pc = param.split('_')
   out = {'label':pc[0], 'comp':pc[1], 'wave':wave, 'y':y, 'yerr':yerr}
   return out

def colorplot(tags, prefix='f', suffix='', ys=0, xs=0, sat=3,size=64, pxscale=.5):
#                         tags should be strings like '38' for 3.8 microns
   lgrid,mgrid = np.meshgrid(np.arange(-size/2,size/2,pxscale), np.arange(-size/2,size/2,pxscale))

   sh = lgrid.shape
   waves  = []  # convert tags to wavelenths
   for t in tags:waves.append(int(t[0])+0.1*int(t[1]))
   waves  = np.array(waves)
   output = np.zeros((sh[0], sh[1], 3))  # will contain rgb pixel values
   xx = (waves.max()-waves)/(waves.max()-waves.min())
   for t,x in zip(tags, xx):
      im = tm.getimf(prefix+t+suffix)  # model data
      ii = im.image(lgrid+xs, mgrid+ys) # image version of model
      ii/=(ii.max()*len(waves)) # normalize to 1/# of wavelengths
      c = i2.cc(x)  # color triplet for this wavelength
      for i in range(3):output[:,:,i] += c[i]*ii

   plt.clf()
   plt.imshow(sat*output, origin='lower')
   xp = []; xl = []
   px = 2*(size//8)
   for i in range(8):
      xp.append(px*i) # pixel location for ticks
      xl.append(f'{(i-4)*px*pxscale:3.1f}')  # value in mas
   plt.xticks(xp,xl)
   plt.yticks(xp,xl)
   plt.xlabel('milliarcsec')
   plt.ylabel('milliarcsec')
#  return output

def plotparm(labels, tags, plotfile = None):
   waves = []
   for t in tags: waves.append(int(t[1])+0.1*int(t[2]))
   win = plt.get_current_fig_manager().window
   titles = {'a':'Major Axis', 'b':'Minor Axis','pa':'Major Axis Position Angle','l':'Relative Position West','m':'Relative Position North','amp':'Relative Flux','b/a':'Axis Ratio'}
   ylabel = {'a':'milliarcsec', 'b':'milliarcsec','l':'milliarcsec','m':'milliarcsec','amp':'relative flux', 'pa':'degrees','b/a':'axis ratio'}
   comps = {'1':'NC', '2':'SC'}
   values = []

   for t in tags:values.append(mf.samplesigma(t, labels=labels.copy(), addb=True))
   newlabels = values[0]['labels']
   if (plotfile is not None):
      pdf = PdfPages(plotfile)
   for i,nl in enumerate(newlabels):
      p,c   = nl.split('_')
      y     = []
      yerr  = []
      for v in values:
         vp = v['perc']
         y.append(vp[2,i])
         yerr.append([vp[2,i]-vp[1,i], vp[3,i]-vp[2,i]])
      yerr = transpose(array(yerr))
      plt.clf()
      plt.errorbar(waves, y, yerr=yerr,fmt='o',capsize=2)
      plt.xlabel('wavelength(microns)')
      plt.ylabel(ylabel[p])
      plt.title(titles[p]+' '+comps[c])
      if (plotfile is None):
         win.move(700,0)
         wu.mpause(.1)
         x = input()
      else: pdf.savefig()
   if(plotfile is not None):pdf.close()

def printparm(imodel, file, tags):
   waves = []
   for t in tags: waves.append(int(t[1])+0.1*int(t[2]))
   labels = imodel.getlabels()
   titles = {'a':'Major Axis', 'b':'Minor Axis','pa':'Major Axis Position Angle','l':'Relative Position West','m':'Relative Position North','amp':'Relative Flux','b/a':'Axis Ratio'}
   ylabel = {'a':'milliarcsec', 'b':'milliarcsec','l':'milliarcsec','m':'milliarcsec','amp':'relative flux', 'pa':'degrees','b/a':'axis ratio'}
   comps = {'1':'NC', '2':'SC'}
   values = []

   for t in tags:values.append(mf.samplesigma(t, labels=labels, addb=True))
   newlabels = values[0]['labels']
   pf = open(file,'w')
   for i,nl in enumerate(newlabels):
      l,c = nl.split('_')
      param = titles[l]+' '+comps[c]+' ('+ylabel[l]+')\n'
      pf.writelines(param)
      pf.writelines(f'   {"lambda":8} {"lower":8} {"median":8} {"upper":8}\n')
      for w,v in zip(waves,values):
         vp = v['perc']
         pf.writelines(f'{w:8.1f} {vp[1,i]:8.4f} {vp[2,i]:8.4f} {vp[3,i]:8.4f} \n')
   pf.close()

def plotparmold(param, data,dolog=False):
   titles = {'a':'Major Axis', 'b':'Minor Axis','pa':'Major Axis Position Angle','l':'Relative Position West','m':'Relative Position North','amp':'Relative Flux','b/a':'Axis Ratio'}
   ylabel = {'a':'milliarcsec', 'b':'milliarcsec','l':'milliarcsec','m':'milliarcsec','amp':'relative flux', 'pa':'degrees','b/a':'axis ratio'}
   comps = {'1':'MINorth', '2':'MISouth'}
   p = getp(param, data)
   plabel = p['label']
   title  = titles[p['label']]+' '+comps[p['comp']]
   label  = ylabel[plabel]

   plt.clf()
   if dolog:
      x = log10(p['wave'])
      y = log10(p['y'])
      yerr = p['yerr']
      for i in range(2):yerr[i] = log10(1+yerr[i]/p['y'])
      plt.errorbar(x, y, yerr=yerr,fmt='o',capsize=2)
      plt.xlabel('log wavelength microns')
   else:
      plt.errorbar(p['wave'], p['y'], yerr=p['yerr'],fmt='o',capsize=2)
      plt.xlabel('wavelength microns')
      plt.ylabel(label)
   plt.title(title)

def plotcomp(models, data='data2'):
   obs = tm.getobsf(data)
   for m in models:
      im = tm.getimf(m)
      wave = float(m[1]+'.'+m[2])
      obs.viscomp(im, showtel=True, output=m, title=str(wave)+' micron ')
   return 

def statimage(tag, ns=1000, simage=64, dy = 0, cut=-5):
   im      = tm.getimf('s473')  # template for model
   pxscale = .5 # mas
   lgrid,mgrid = np.meshgrid(np.arange(-simage/2,simage/2,pxscale), 
      np.arange(-simage/2,simage/2,pxscale)) 
   mgrid +=  dy  # shift y position to center images

   samples = np.load(tag+'samples.npy')[-ns:] # recover samples
   prob    = np.load(tag+'probabilities.npy')[-ns:]
   prob   -= prob.max()
   summ   = 0 * mgrid  # output
   for s,p in zip(samples,prob):
      if(p > cut):
         im.setvaluelist(s) # insert sample values into model
         summ += im.image(lgrid, mgrid)  # add to image
   return summ

def plotlim(bins=50, n0=50000,plotfile=None, tags=None):
   im = tm.getimf('s473')
   labels = im.getlabels()
   if (tags is None):tags=['s320','s340','s360','s380','s470']
   samples = []
   probs   = []
   col     = []
   n       = -n0
   for t in tags:
      samples.append(np.load(t+'samples.npy')[n:])
      probs.append(np.load(t+'probabilities.npy')[n:])
   ntag = len(tags)
   for i in range(ntag): col.append(i2.cc(1.-i/(ntag-1)))
   xlim = {}
   xlim['a_1'] = [5,25.]
   xlim['a_2'] = [0,20.]
   xlim['b_1'] = [5,25.]
   xlim['b_2'] = [5,25.]
   xlim['b/a_1'] = [0.,1.]
   xlim['b/a_2'] = [0.,1.]
   xlim['pa_1'] = [-90.,0]
   xlim['pa_2'] = [-90.,0]
   xlim['amp_2'] = [0.1,4.]
   xlim['l_2'] = [-3,5.]
   xlim['m_2'] = [-16,-10.]
   pp = 2
   if (plotfile is not None):
      pdf = PdfPages(plotfile)
   for l in range(len(labels)):
       plt.clf()
       for i in range(len(tags)):
          p = probs[i]
          p-=p.max()
          s = samples[i][:,l]
          s = s[p>(-4)]
#         h = np.histogram(samples[i][:,l],bins=bins, density=True)
          h = np.histogram(s,bins=bins, density=True)
#         perc = np.percentile(samples[i][:,l],[17,50,83])
          perc = np.percentile(s,[17,50,83])
          x = h[1]; dx=x[1]-x[0];x= (x+.5*dx)[:-1]
          plt.plot(x, h[0]/h[0].max(),color=col[i])
          plt.plot(perc, (0.9-.02*i)*np.ones(3),color=col[i])
          plt.plot(perc, (0.9-.02*i)*np.ones(3),'+',color=col[i])
       plt.title(labels[l])
       plt.xlim(xlim[labels[l]])
       if (plotfile is None): 
          wu.mpause(pp)
          y = input()
       else:pdf.savefig()
   plt.clf()
   for i in range(len(tags)):
       s = samples[i][:,0]*samples[i][:,1]
       h = np.histogram(s,bins=bins, density=True)
       perc = np.percentile(s,[17,50,83])
       x = h[1]; dx=x[1]-x[0];x= (x+.5*dx)[:-1]
       plt.plot(x, h[0]/h[0].max(),color=col[i])
       plt.plot(perc, (0.9-.02*i)*np.ones(3),color=col[i])
       plt.plot(perc, (0.9-.02*i)*np.ones(3),'+',color=col[i])
   plt.title('b_1',pad=0);plt.xlim((0,15))
   if (plotfile is None): wu.mpause(pp)
   else:pdf.savefig()
   wu.mpause(.1)
   plt.clf()
   for i in range(len(tags)):
       s = samples[i][:,3]*samples[i][:,4]
       h = np.histogram(s,bins=bins, density=True)
       perc = np.percentile(s,[17,50,83])
       x = h[1]; dx=x[1]-x[0];x= (x+.5*dx)[:-1]
       plt.plot(x, h[0]/h[0].max(),color=col[i])
       plt.plot(perc, (0.9-.02*i)*np.ones(3),color=col[i])
       plt.plot(perc, (0.9-.02*i)*np.ones(3),'+',color=col[i])
   plt.title('b_2');plt.xlim((0,15))
   for i,t in enumerate(tags):
      wave = (t[1]+'.'+t[2])
#  wave=['3.2','3.4','3.6','3.8','4.7']
      plt.plot([10,12],np.ones(2)*(1-.05*i),color=col[i])
      plt.text(12.1,(1-.05*i-.01),wave,color=col[i])
   if (plotfile is None): wu.mpause(1)
   else:
      pdf.savefig()
      pdf.close()
   wu.mpause(.1)
   return

