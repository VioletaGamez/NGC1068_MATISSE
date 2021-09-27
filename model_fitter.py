import numpy as np
from astropy.io import fits
from glob import glob
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import corner
#from scipy.optimize import curve_fit, least_squares
#from scipy.signal import argrelmax
import matplotlib.cm as cm
import matplotlib as mpl
import os
import time
import matplotlib.gridspec as gridspec
import sys
import gc
import testme as tm


import wutil as wu

os.environ["OMP_NUM_THREADS"] = "1" #set so that numpy doesn't interfere with emcee
#import pandas as pd

from img2vis import matisseObs
#from model_defs import two_gauss_one_centered, three_gauss

def mpause(interval):
   figman = plt.get_current_fig_manager()
   canvas = figman.canvas
   if(canvas.figure.stale): canvas.draw()
   canvas.start_event_loop(interval)
   return
def __version__():
    return 0.21


#---------------------------------------------------------------------------------------
#---------------- ln prob and ln likelihood functions for MCMC fitting------------------
#---------------------------------------------------------------------------------------


def lnlike(params, *argv):
    #log likelihood function
    method, rundata = argv
    imagemodel = rundata.im
    matisseobs = rundata.obs
    wave       = matisseobs.wave
    nwave      = len(wave)
    debug = rundata.debug
    uvm   = rundata.uv2
    if(debug): print('pp',params)

    md = matisseobs.matissedata #observation data struct
#             update model parameters
    imagemodel.setvaluelist(params)

    if (True):  # only consider mixed models
       labels      = imagemodel.getlabels()  # check for lnf parameter
       if (debug):print(labels)
       y           = 0.1*md['vis'] # 10% of value
       yerr        = rundata.vis_noise
       yerr        = np.sqrt(y**2 + yerr**2) #  in quadrature
#      valid,cflux,pflux,mvis = matisseobs.modelvis(imagemodel)
       valid,mvis,mcphase = matisseobs.modeldata(imagemodel)

       if (not valid): return -np.inf # parameters out of range

       residuals   = np.zeros(mvis.shape)
       for i in range(nwave):
          residuals[:,i]   = uvm*md['wt2'][:,i]* \
             ((mvis[:,i] - md['vis'][:,i])/yerr[:,i])**2

       y           = rundata.cphase_noise*np.ones(md['cphase'].shape)
#      mcphase     = matisseobs.modelcphase(cflux,valid)
       model       = np.append(mvis,mcphase)  # append cphases to vis2s
       mcphase     = np.exp(1j*np.radians(mcphase)) # convert to rotor
       data        = np.exp(-1j*np.radians(md['cphase']))
       yerr        = np.append(yerr,y)  # append cphase errs to vis2errs
       residuals   = np.append(residuals, md['wt3']*
          (np.angle(mcphase*data)/y)**2)  # error not in angle but in rotor
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

def lnprior(parms, *argv):
    #compute the prior
    #  currently: only check that values are in bounds
#   m, mo, imagemodel, uv = argv
    m,rundata    = argv
    if (rundata.hard==0): return 0.0 # no hard priors
    imagemodel = rundata.im
#        extract u and  l bounds from model
    lbound = imagemodel.getlbound()
    ubound = imagemodel.getubound()

    if (rundata.hard==1): # hard boundaries
       for l,p,u in zip(lbound, parms, ubound):
          if (p<l) | (p>u):return -np.inf  #  outside bounds
       return 0.0
    else: #gaussian boundaries
       sigma  = 0.5 * (ubound - lbound)*rundata.hard
       middle = 0.5 * (ubound + lbound)
       nparms = len(parms)
       sweight= rundata.obs.sumweights
#             give same relative weight to parms and data
       return -0.5*np.sum(((parms-middle)/sigma)**2)*(sweight/nparms)

def lnprob(theta, *argv):
    #compute the log probability
    lp = lnprior(theta, *argv)
    if not np.isfinite(lp): 
       return -np.inf
    ll = lnlike(theta, *argv)
    if (not np.isfinite(ll)):return -np.inf
    x =  lp + lnlike(theta, *argv)
    return x

def do_mcmc(matisseobs, imagemodel, rundata, 
   x0=None, sigmas=None, uniform_guess=True, dotv=False):
            
    ''' Function to fit data in fname using MCMC sampling.
    matisseobs : object containing matisse observations (img2vis)
    imagemodel: component object containing model parameters,bounds,labels
    x0: the initial guess for the model (None)
    sigmas: sigmas of gaussian ball around x0 (None)
    s:	the length in px of the model image sides (default=64px)
    method: options are 'vis' to fit visibilities alone
    'cphase' to fit closure phase alone (not recommended)
    'bispec' to fit the bispectrum (default, best option)
    uniform_guess: boolean, True starts x0 with uniform random distribution
    inside of bounds. False gives a Gaussian push around x0 with sigmas
    set by weights. Good for intial searching of parameter space with 
    nwalkers ~ 500.

        '''
    import emcee
    from multiprocessing import Pool
    from multiprocessing import cpu_count
#       sort out input parameters
    niter = rundata.niter
    nwalkers = rundata.nwalkers
    output = rundata.output 
    print('noises', rundata.vis_noise, rundata.cphase_noise)

    s = 128
    pxscale = .5
    lgrid,mgrid = \
       np.meshgrid(np.arange(-s*pxscale/2,s*pxscale/2,pxscale), np.arange(-s*pxscale/2,s*pxscale/2,pxscale))
    ncpu = cpu_count()
#   ncpu = 1
#   print("{0} CPUs".format(ncpu))
    with Pool() as pool:
        ndim, nwalkers = imagemodel.nnparams(), nwalkers

        pos = []
        # uniform initial distribution function
        if uniform_guess:
            lbound = imagemodel.getlbound()
            ubound = imagemodel.getubound()
            pos = [np.array([np.random.uniform(low=lbound[j],high=ubound[j]) for j in range(len(lbound)) ]) for kk in range(nwalkers)]
            pos = np.array(pos)
        else:
            # gaussian initial distribution function
            if type(x0) == type(None):
                print('Please supply an ititial guess (x0) and sigmas if uniform_guess=False. Quitting...')
                sys.exit()
            pos = np.array([ np.array(x0) + np.random.randn(ndim)*sigmas for kk in range(nwalkers) ])
        method  = rundata.method
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool,
#       moves = [(emcee.moves.DEMove(),0.8), (emcee.moves.DESnookerMove(),0.2)],
        moves = [(emcee.moves.StretchMove(a=10.0),1.)],
           args=(method, rundata))

        state  = pos
        riter  = niter # remaining iterations
        oldbest= -1.e9
        ok     = True
        while ok:
           it = 100  # number of iterations in this cycle
#          it = riter  # number of iterations in this cycle
           if (it>riter):it= riter  # <= remaining iteratins
           print('niter nwalkers',niter, riter, nwalkers)
           if (it<=0):break  # done
           if (riter==niter):  # startup
              state = sampler.run_mcmc(state, it,progress=True)  # doit
              doplot = None
           else: 
              state= sampler.run_mcmc(None, it, progress=True)
              doplot = dotv
           riter -= it  # update remaining iterations
           print('finished ', niter-riter,' iterations')
           samples = sampler.get_chain(flat=True)
           probs   = sampler.get_log_prob(flat=True)
           amax    = probs.argmax()
           rundata.bestlike = probs[amax]
           rundata.bestvalues= samples[amax]
        
           dfree = rundata.obs.sumweights
           chi2df= rundata.bestlike/dfree
           if (doplot is not None):
              tm.setpltwindow(1, w=600,h=600, x=800, y=10)
              review2(nwalkers, rundata.labels,
                 samples, probs, cut=20,dfree=dfree,
                 title=rundata.tag) 

              imagemodel.setvaluelist(rundata.bestvalues)
              model = imagemodel.image(lgrid, mgrid)
              tm.setpltwindow(2, w=300,h=300, x=400, y=50)
              wu.wtv(model)
              wu.mpause( 1)
              if(output is not None):tag=output
              else:tag = ''

              np.save(tag+'samples.npy', samples)
              np.save(tag+'probabilities.npy', probs)
              dchi2   = chi2df - oldbest
              oldbest = chi2df
              if (dchi2 < .001):
                 print('delta chi2=', f'{dchi2:7.2f}')
                 ok = False
        samples = sampler.get_chain(flat=True)

    #----------------------------------------------------------------------------
    #-------------------------plotting and saving data---------------------------
    #----------------------------------------------------------------------------
    #save the final model
    best_params = samplestats(tag=tag)
    imagemodel.setvaluelist(best_params[:,1])
    if (output is not None):
       model = imagemodel.image(lgrid, mgrid)
       hdu   = fits.PrimaryHDU(model)
       hdul  = fits.HDUList([hdu])
       hdul.writeto(tag+'.fits', overwrite=True)
    #
#      final_params_mp = triangles(rundata.labels, cut=10, tag=tag, output=True)
    return best_params 


def triangles(labels, tag='', cut=10, output=False, ns=100000):
#  stats   =samplesigma(tag)
   samples = np.load(tag+'samples.npy')
#  samples[:,1] = samples[:,1]*samples[:,0]
   prob    = np.load(tag+'probabilities.npy')
   sdim    = samples.shape
   n       = ns
   if (sdim[0] < ns):s=sdim[0]

   samples = samples[-ns:]
   prob    = prob   [-ns:]
   pkeep   = prob> (prob.max() - cut)
   samples = samples[pkeep]
#  labels[1] = 'b_1'
   perc    = np.percentile(samples,[17,50,83],0)
   if (output):plt.ioff()
   fig     = corner.corner(samples, labels=labels, quantiles=[.17,.50,.83], 
      truths=perc[1], levels=[.39,.86])
   if (output):
       fig.savefig(tag+"_triangle.pdf")
#      fig.savefig(tag+"_triangle.pdf", bbinches='tight')
       plt.ion()
   return perc

def newlimits(sig=2, output=None):
   stats=samplestats(tag=output)
   newlow  = []
   newup   = []
   ndim    = stats.shape[0]
   newlow  = stats[:,0]
   newup   = stats[:,2]
   return [newlow, newup]
#-------------------------------------------------------------------------------------------
#---------------------------func to find largest probability -------------------------------
#-------------------------------------------------------------------------------------------
def find_mostprob(labels, samples, probs):
    """Find the samples in the chain with the most probable values"""
    nvals  = samples.shape[0]
    nparam = samples.shape[1]
    if(nvals>200000):
       mprobs   = probs[-200000:]
       goodvals = samples[-200000:]
    else:
       mprobs   = probs
       goodvals = samples
    px = mprobs>mprobs.max()-5
    goodvals = goodvals[px]
    mprobs   = mprobs[px]
    perc = np.transpose(np.percentile(goodvals,[17,50,83],0))
    for (l,p) in zip(labels,perc):
       print(f'{l:6s} {p[1]:5.2f} {p[0]:6.2f} {p[2]:6.2f}')
    best_vals = []
    bv = []
    outfile = 'most_probable_vals'
    outfile+='.npy'
    np.save('most_probable_vals.npy', perc[:,1])
    return perc[:,1]

def review2(nwalkers, labels, samp, prob, cut=20, dfree=1,title=''):
   pmax   = prob.max() - cut
   nlabel = len(labels)
   sprob  = prob.shape[0]
   niter  = sprob//nwalkers
   nn     = 100
   nplot  = niter//nn
   if (nplot<2):return
   if (nplot > 8):
      nn     =  200
      nplot  = niter//nn
   nn  = nn*nwalkers
   if(nlabel>24):nlabel=24
   nx  = 4
   ny  = (nlabel//nx) + ((nlabel%nx) != 0)
   fig = plt.figure(1)
   plt.clf()
   axarr = fig.subplots(ny,nx)
   axf   = axarr.flatten()
   ptext= f'{niter:6} {pmax+cut:7.1f} {(pmax+cut)/dfree:8.3f}'
   for l in range(nlabel):
      axf[l].set_xlabel(labels[l])
      sl = samp[:,l]
      for i in range(1,nplot):
         pp = prob[i*nn:(i+1)*nn]
         pa = pp > pmax
         ss = sl[i*nn:(i+1)*nn]
         h  = np.histogram(ss[pa], bins=100)
         axf[l].plot(h[1][1:], h[0])
#  axf[0].set_ylabel(f'{pmax+cut:7.1f}')
   axf[0].set_ylabel(ptext)
   plt.tight_layout()
   plt.title(title)
   mpause(0.5)
   print(ptext)
   return

def samplesigma(tag, cut=10,labels=None,addb=False,ns=200000):
   sigmas  = [2.5, 17.5, 50., 83., 97.5]
   prob    = np.load(tag+'probabilities.npy')[-ns:]
   pcut    = prob> (prob.max()-cut)
   samples = np.load(tag+'samples.npy')[-ns:][pcut]
   perc    = np.percentile(samples, sigmas, 0)
   if (addb):
      lc     = labels.copy()
      pshape = perc.shape
      nperc  = np.zeros((pshape[0], 2+pshape[1]))
      nperc[:,:-2] = perc
      nperc[:,:-2] = perc
      b1 = samples[:,labels.index('a_1')]*samples[:,labels.index('b/a_1')]
      b2 = samples[:,labels.index('a_2')]*samples[:,labels.index('b/a_2')]
      nperc[:,-2] = np.percentile(b1, sigmas)
      nperc[:,-1] = np.percentile(b2, sigmas)
      lc.append('b_1')
      lc.append('b_2')
      return {'labels':lc, 'perc':nperc}
   return perc

def samplestats(tag='',ns=200000):
    samples = np.load(tag+'samples.npy')
    probs   = np.load(tag+'probabilities.npy').flatten()#note these are ln prob

    ndim   = samples.shape[1]
    nsamp  = samples.shape[0]
    if (nsamp > ns):
       samples = samples[-ns:]
       probs   = probs[-ns:]
    output = np.transpose(np.percentile(samples,[2, 50, 98], 0))
    return output
#-----------------------------------------------------------------------------------------
#--------------------------- function to put wcs coords in fits files -------------------- 
#-----------------------------------------------------------------------------------------
def wcsify(fname, pixelscale, imshape, north_angle=0.,ra=0,dec=0):
    wcs_keys = ['wcsaxes', 'crpix1', 'crpix2', 'crval1', 'crval2', 'ctype1', 'ctype2', 'orientat','cdelt1', 'cdelt2']
    #need pixel scale in degrees
    ra_delt =  -pixelscale / 3600 / 1000.
    dec_delt = pixelscale / 3600 / 1000.
    wcs_vals = [2,	imshape[0]//2, imshape[1]//2, 0,0,'RA','DEC',north_angle,\
                ra_delt,dec_delt ]
    with fits.open(fname, mode='update') as hdul:
        for i,key in enumerate(wcs_keys):
            hdul[0].header[key] = wcs_vals[i]
            hdul.flush()
            hdul.close()

#-------------------------------------------------------------------------------------------
#-------------------------- func to check autocorrelation ----------------------------------
#-------------------------------------------------------------------------------------------
def check_convergence(labels, folder='./'):
    """Compute the autocorrelation time to estimate whether chains are "burnt-in"
    """
    from emcee import autocorr 
    x = np.load(folder+'samples.npy')
    inter_var = []
    fig, axarr = plt.subplots(2, figsize=(10,10))
    ax,bx = axarr.flatten()
    for j in range(len(x[0,0])):
        chain = x[:, :, j].T
        # Compute the estimators for a few different chain lengths
        N = np.exp(np.linspace(np.log(100), np.log(chain.shape[1]), 10)).astype(int)
        gw2010 = np.empty(len(N))
        new = np.empty(len(N))
        test = np.empty(len(N))
        for i, n in enumerate(N):
            test[i] = autocorr.integrated_time(chain[:,:n],c=5)
            ax.loglog(N, test, "o-", label=labels[j])

            #compute the interchain variance
            inter_var.append(np.std(chain))

        # Plot the comparisons
        ax.set_xlabel("number of samples, $N$")
        ax.set_ylabel(r"$\tau$ estimates")
        ax.legend(fontsize=14)

        intra_var = []
        for i in range(x.shape[1]):
            param_var = []
            for j in range(x.shape[2]):
                chain = x[:,i,j].T 
                #compute intra-chain variance
                param_var.append(np.std(chain))
                intra_var.append(param_var)

        #compute the ratios of the intra and interchain variances
        ylim = 10
        for j in range(len(inter_var)):
            all_vals = []
            for var in intra_var[j]:
                ratio = inter_var[j] / var
                all_vals.append(ratio)
                if ratio > ylim:
                    bx.scatter(j,ylim-0.5,marker='^', alpha=0.75, c='dodgerblue')
                else:
                    bx.scatter(j, ratio,c='dodgerblue')
                    bx.scatter(j, np.mean(all_vals), marker='s', s=100, c='k')
                    bx.plot([0,j],[1,1], 'r--', label='Ideal Ratio')
                    bx.legend(fontsize=14)
                    bx.set_xlabel('parameter')
                    bx.set_ylabel(r'${\rm Var}_{\rm inter} / {\rm Var}_{\rm Intra}$ for each chain')
                    bx.set_ylim([0,ylim])
                    plt.show()
                    plt.close()


#-------------------------------------------------------------------------------------------
#---------------------------func to plot the fit results -----------------------------------
#-------------------------------------------------------------------------------------------
def check_fit(matisseobs, imagemodel, pixelscale, labels, save=True, chains=True, mkimage=True, fdir='./',fittypes=['mostprob_model.fits','median_model.fits']):
    ''' Make plots to check the quality of the fit.
    args are matisseobs (=img2vis object), pixelscale (pixelscale in mas/px) and save.
    save defaults to True, but setting to false calls plt.show() instead of
    saving as a pdf
    '''
    wl    = matisseobs.wave
    do_uv = save
    for fittype in fittypes:
        modelfile = fdir+fittype
        namestr = fittype.split('.')[0]

        md = matisseobs.matissedata

        fig, axarr = plt.subplots(2,4,figsize=(12,8.5))
        gs = gridspec.GridSpec(2, 5,figure=fig,width_ratios=[1,0.5,0.2,1,0.5],height_ratios=[1,0.25])
        ax = plt.subplot(gs[0, 0])
        bx = plt.subplot(gs[0,3])
        cx = plt.subplot(gs[1,0])
        dx = plt.subplot(gs[1,3])
        ex = plt.subplot(gs[1,1])
        fx = plt.subplot(gs[1,4])
        zx = plt.subplot(gs[0,1])
        yx = plt.subplot(gs[0,4])
        xx = plt.subplot(gs[0,2])
        wx = plt.subplot(gs[1,2])
        yx.axis('off')
        zx.axis('off')
        wx.axis('off')
        xx.axis('off')
        #bx.axis('off')
        axarr = np.array([[ax,bx],[cx,dx]])


        normvis   = mpl.colors.Normalize(vmin=0,vmax=0.3)
        normphase = mpl.colors.Normalize(-180,180)

        vlow  = 0
        vhi   = np.max( [np.max(md['vis']),np.max(matisseobs.modelvis(imagemodel))]) + 0.15
        ax.set_ylim([vlow, vhi])
        ax.set_xlim([vlow,vhi])
        cx.set_xlim([vlow,vhi])
        rvhi  = np.max( np.absolute(matisseobs.vis_chi2(imagemodel,resid=True)) ) + 0.05
        rvlow = -rvhi
        cx.set_ylim([rvlow,rvhi])
        ex.set_ylim([rvlow,rvhi])
        #cx.grid(); ex.grid()



        clow = -180#np.min([np.min(i.cphase_obs), np.min(i.modelcphase())]) -5
        chi = 180#np.max([np.max(i.cphase_obs), np.max(i.modelcphase())]) + 5
        bx.set_ylim([clow, chi])
        bx.set_xlim([clow, chi])
        rchi = np.max( np.absolute(matisseobs.cphase_chi2(imagemodel,resid=True)) ) + 1
        rclow = -rchi
        dx.set_ylim([rclow,rchi])
        fx.set_ylim([rclow,rchi])



        #vis2
        ax.errorbar(matisseobs.modelvis(imagemodel),md['vis'],yerr=md['vis_noise'], color='firebrick',capsize=0.,ls='none',marker='s',markeredgecolor='k',alpha=0.8)
        ax.plot([vlow, vhi],[vlow, vhi],'k--')
        cx.errorbar(matisseobs.modelvis(imagemodel), matisseobs.vis_chi2(imagemodel,resid=True),yerr=md['vis_noise'],marker='s',ls='none', capsize=0,\
                    color='firebrick',markeredgecolor='k',alpha=0.8)
        cx.plot([vlow,vhi],[0,0],'k--')
        ex.hist(matisseobs.vis_chi2(imagemodel, resid=True), bins=np.linspace(rvlow,rvhi,20),histtype='stepfilled',fc='firebrick',orientation='horizontal')
        ex.set_yticklabels(['' for x in cx.get_yticks()])
        ax.set_xticklabels(['' for x in cx.get_xticks()])

        #closure phase
        bx.errorbar(matisseobs.modelcphase(imagemodel),md['cphase'],yerr=md['cphase_noise'], color='forestgreen',capsize=0., ls='none',marker='s',markeredgecolor='k',alpha=0.8)
        bx.plot([clow, chi],[clow, chi],'k--')
        dx.errorbar(matisseobs.modelcphase(imagemodel), matisseobs.cphase_chi2(imagemodel,resid=True),yerr=md['cphase_noise'],marker='s',ls='none', capsize=0,\
                    color='forestgreen',markeredgecolor='k',alpha=0.8)
        dx.plot([clow,chi],[0,0],'k--')
        fx.hist(matisseobs.cphase_chi2(imagemodel,resid=True),bins=np.linspace(rclow,rchi,20),histtype='stepfilled',fc='forestgreen',orientation='horizontal')
        fx.set_yticklabels(['' for x in dx.get_yticks()])


        #labels
        ax.set_title('Squared Visibilities')
        ax.set_ylabel('Observed')
        cx.set_xlabel('Model')
        bx.set_ylabel('Observed')
        dx.set_xlabel('Model')
        bx.set_title('Closure Phase')
        cx.set_ylabel('Residuals')


        #ex.set_ylim([clow, chi])
        plt.suptitle(r'Model Fit at $%.2f \mu$m'%(wl))
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.0, wspace=0.0)

        if save:
#           plt.savefig(fdir+'fit_results_%s.pdf'%(namestr), bbinches='tight')
            plt.savefig(fdir+'fit_results_%s.pdf'%(namestr))
        else:
            plt.show()
            plt.close()

        if mkimage:
            fig = plt.figure()
            hdu = fits.open(modelfile)
            im = hdu[0].data
            s = im.shape[0]
            w = 50
#           im = np.fliplr(im[s//2-w:s//2+w, s//2-w:s//2+w])
            axis_labels = np.arange(-w*pixelscale, (w+25)*pixelscale, 25*pixelscale)
            label_locs = np.arange(-0.5,2*w+24.5,25)
            plt.imshow(np.sqrt(im), cmap='Greys',origin='lower')
            line_length = 14.3 #mas = 1pc
            plt.plot([5, 5+line_length/pixelscale],[10,10], c='k', lw=2)
            plt.text(5, 5, '1 pc',color='k')
            plt.xticks(label_locs, axis_labels[::-1])
            plt.yticks(label_locs, axis_labels)
            #plt.xticks(label_locs, axis_labels[::-1])
            plt.xlabel('Offset [mas]')
            plt.ylabel('Offset [mas]')
            plt.title('Best Fit Model')
            plt.grid()
#           plt.savefig(fdir+'final_modelfit_%s.pdf'%(namestr), bbinches='tight')
            plt.savefig(fdir+'final_modelfit_%s.pdf'%(namestr))
            plt.close()


#       if do_uv:
        if False:
            fig,axarr = plt.subplots(2,2,figsize=(10,10))
            ax,bx,cx,dx = axarr.flatten()
            vmax = np.min([np.max(i.vis2)+0.1, 1.0])
            vmin = 0
            normvis  = mpl.colors.Normalize(vmin=vmin,vmax=vmax)
            k = i.vis2.shape[0] //2


            im = ax.scatter(md['u'], md['v'], c=matisseobs.modelvis(),cmap='viridis',norm=normvis)
            ax.scatter(-md['u'], -md['v'], c=matisseobs.modelvis(),cmap='viridis',norm=normvis)
            cbar1 = plt.colorbar(im,ax=ax, label='Squared Vis.')
            ax.set_xlabel('u [m]')
            ax.set_ylabel('v [m]')
            bx.imshow(i.vis2[k-32:k+32,k-32:k+32],origin='lower',vmin=vmin, vmax=vmax,cmap='viridis')
            #bx.scatter(i.x-k, i.y-k, marker='o',ec='k',fc='none')
            #bx.scatter(-i.x, -i.y, marker='o',ec='k',fc='none')

            vmin = -180
            vmax = 180
            normphase  = mpl.colors.Normalize(vmin=vmin,vmax=vmax)
            im = cx.scatter(md['u1'], md['v1'], c=i.modelcphase(),cmap='seismic',norm=normphase)
            cx.scatter(-md['u1'], -md['v1'], c=i.modelcphase(),cmap='seismic',norm=normphase)
            im = cx.scatter(md['u2'], md['v2'], c=i.modelcphase(),cmap='seismic',norm=normphase)
            cx.scatter(-md['u2'], -md['v2'], c=i.modelcphase(),cmap='seismic',norm=normphase)
            cbar2 = plt.colorbar(im,ax=cx, label='Closure Phase')
            cx.set_xlabel('u [m]')
            cx.set_ylabel('v [m]')

            dx.imshow(i.cphase_model[k-32:k+32,k-32:k+32],vmin=vmin,origin='lower', vmax=vmax, cmap='seismic')
            #dx.scatter(i.x, i.y, marker='o',ec='k',fc='none')
            #dx.scatter(-i.x, -i.x, marker='o',ec='k',fc='none')
            #dx.set_xticks(n)

            ax.set_title('Observed uv points')
            bx.set_title('Model Visibilities')
            dx.set_title('Model Phase')


#           plt.savefig(fdir+'final_modeluv_%s.pdf'%(namestr), bbinches='tight')
            plt.savefig(fdir+'final_modeluv_%s.pdf'%(namestr))
            plt.close()

    if chains:
        if(len(glob(fdir+'samples.npy'))==0):
           print('skip samples')
           return
        #make a plot showing "burn-in" of sample chains
        x = np.load(fdir+'samples.npy')
        #reshape???
        naxis = len(labels)
        fig,axarr = plt.subplots( int(naxis//2)+naxis%2, 2, figsize=(8,12))

        for i in range(naxis):
            axarr.flatten()[i].plot(x[:,:,i], c='k', alpha=0.25)
            axarr.flatten()[i].plot([0,x.shape[0]],[np.median(x[-100:,:,i]), np.median(x[-100:,:,i])], 'r--' )
            axarr.flatten()[i].set_ylabel(labels[i])
            axarr.flatten()[i].set_xlabel("N Iter")
        plt.tight_layout()
#       plt.savefig(fdir+'chain_check.pdf', bbinches='tight')
        plt.savefig(fdir+'chain_check.pdf')

    return


