import numpy as np
from matplotlib import pyplot as plt
import glob
from astropy.io import fits
from matplotlib.backends.backend_pdf import PdfPages
from mcdb import fitsutil as fu
import wutil as wu
import testme as tm

def mpause(interval):
   figman = plt.get_current_fig_manager()
   canvas = figman.canvas
   if(canvas.figure.stale): canvas.draw()
   canvas.start_event_loop(interval)
   return

def cc(x):    # colors red->green as tuples x is scaled 0->1
   r=np.clip(1-2*x,0,1) 
   b=np.clip(2*x-1,0,1) 
   g=1-b-r 
   return (r,g,b)

def uv2match(uv1, uv2, du): # is (u1,v1) ~= (u2,v2), with possible sign flip
   duv = np.sqrt(np.sum((uv1-uv2)**2))
   if(duv) < du: return 1
   duv = np.sqrt(np.sum((uv1+uv2)**2))
   if(duv) < du: return -1
   return None

def uv3match(uv1a,uv2a, uv3a, uv1b, uv2b,uv3b,du):# are uv1,uv2 a &b the same
#                                 with possible sign flip and permutations
   ua = [uv1a, uv2a, uv3a]
   ub = [uv1b, uv2b, uv3b]
   perms = [[0,1,2], [0,2,1], [1,0,2], [1,2,0], [2,0,1], [2,1,0]]
   ok = False
   for p in perms:
      duv = []
      for i in range(3):
         duv.append(np.sqrt(np.sum((ua[i]-ub[p[i]])**2)))
      if (np.array(duv).max() < du):return [1,p]
   for p in perms:
      duv = []
      for i in range(3):
         duv.append(np.sqrt(np.sum((ua[i]+ub[p[i]])**2)))
      if (np.array(duv).max() < du):return [-1,p]
   return None

def getbcd(files):
   """
       numeric codes for BCD positons
       OO  = 3
       II  = 0
       OI  = 1
       IO  = 2
   """
   nfiles = len(files)
   out    = []
   for i in range(nfiles):
      bcd1 = fu.getkeyword('bcd1name',files[i])
      bcd2 = fu.getkeyword('bcd2name',files[i])
#     out.append(2*(bcd1[0]=='O') + 1*(bcd2[0]=='O'))
      out.append(1*(bcd1[0]=='O') + 2*(bcd2[0]=='O'))
   return np.array(out)

def complex_average(arr):
   '''n,m = arr.shape
   final_arr = []
   for i in range(n):
      running_product = np.exp(1j*np.radians(arr[i,0]) )
      for j in range(1,m):
         running_product *= np.exp(-1j*np.radians(arr[i,j]) )
      final_arr.append(np.angle(running_product, deg=True)/m)
   return np.array(final_arr)'''
   n,m = arr.shape
   final_arr = []
   for i in range(n):
      temp_list = []
      for j in range(m):
         temp_list.append(np.exp(1j*np.radians(arr[i,j])))
      final_arr.append(np.angle(np.mean(temp_list), deg=True))
   return np.array(final_arr)




def expave6(data, mjdave):
#          average vis like data over multiple subexposures if asked
   if(mjdave):
      nbase = data.shape[0]//6
      out   = 0*data[:6]
      for i in range(nbase):out+=data[6*i:(6*i+6)]
      return out/nbase
   else: return data  # nop
def expave6s(data, mjdave):
#          average string data over 6 baselines
   if(mjdave): return data[0:6]
   else: return data  # nop
def expave4(data, mjdave):
#          average vis like data over multiple subexposures if asked
   if(mjdave):
      nbase = data.shape[0]//4
      out   = 0*data[:4]
      for i in range(nbase):out+=data[4*i:(4*i+4)]
      return (out/nbase).astype(data.dtype)
   else: return data  # nop
def expave4c(data, mjdave):
#          average closurephase like data over 4 triangles if requested
   if(mjdave):
      nbase = data.shape[0]//4
      out   = 0*data[:4].astype(complex)
      for i in range(nbase):out+=np.exp(1j*np.radians(data[4*i:(4*i+4)]))
      return np.angle(out/nbase,True)
   else: return data  # nop
def expave4s(data, mjdave):
#          average string data over 4 triangles
   if(mjdave): return data[0:4]
   else: return data  # nop
##
##
##
## FUNCTION read_matisse_oifits
##
## PURPOSE
##    do what it says
##
def read_matisse_oifits_raw(file, mjdave=True):
   return_strs = ['bl', 'pa', 'u', 'v', 'vis', 'wt2', 'wt3', 'sta2', 'tel2',\
      'u1','u2', 'u3', 'v1', 'v2', 'v3', 'cphase', \
      'wt3', 'sta3','tel3','wave']
   return_vals = {x:np.array([]) for x in return_strs}
   hdu = fits.open(file)
   w   = hdu[3].data
   return_vals['wave']  = w["EFF_WAVE"]*1.e6
   
   adata     = hdu['OI_ARRAY'].data
   sta_index = adata['sta_index']
   tel_names = adata['tel_name']
   vdata     = hdu['OI_VIS2'].data
#              only v**2 data currently implemented
   vv       = vdata['vis2data']
   vv_noise = .1*vv

   sta2 = []  # list of telescope names
   for s in vdata['sta_index']:
      tt = []
      for ss in s:
         for a,b in zip(sta_index, tel_names):
            if(ss==a):tt.append(b)
      sta2.append(tt)
#  vis_noise = np.average(vv_noise[:,ix],axis=1)/np.sqrt(np.sum(ix))
   vis_noise = 0.04 + 0.1*vv # test
   u  = vdata["UCOORD"]
   v  = vdata["VCOORD"]
   t3_table = hdu['OI_T3'].data
   u1 = t3_table['u1coord']
   u2 = t3_table['u2coord']
   u3 = -u1 -u2
   v1 = t3_table['v1coord']
   v2 = t3_table['v2coord']
   v3 = -v1 -v2
   cphase = t3_table['t3phi']
   sta3 = []  # list of station names
   for s in t3_table['sta_index']:
      tt = []
      for ss in s:
         for a,b in zip(sta_index, tel_names):
            if(ss==a):tt.append(b)
      sta3.append(tt)
      
   bl = np.sqrt(u**2+v**2)
   pa = np.rad2deg(np.arctan(u/v))
   pairs = [['bl',bl], ['pa',pa], ['u',u], ['v',v], ['vis',vv]]
#                 average over subexposures for each bcd
   for p in pairs:return_vals[p[0]] = expave6(p[1], mjdave)
   return_vals['sta2'] = expave6s(sta2, mjdave)
   return_vals['wt2']  = np.zeros(6)
   return_vals['wt3']  = np.zeros(4)
   pairs = [['u1',u1], ['u2',u2], ['u3',u3], ['v1',v1], \
      ['v2',v2], ['v3',v3]] 
   for p in pairs:return_vals[p[0]] = expave4(p[1], mjdave)
   return_vals['cphase'] = expave4c(cphase, mjdave)
   return_vals['sta3']   = expave4s(sta3, mjdave)

   if (mjdave):return_vals['sta3']=np.reshape(return_vals['sta3'],[return_vals['u1'].shape[0],3])
   else:return_vals['sta3'] = np.reshape(return_vals['sta3'], (return_vals['u1'].shape[0],3) )
   out = {}
   for x in return_strs: out[x] = return_vals[x]
   return out

def read_matisse_oifits(file,wavedata,mjdave=True):
   return_strs = ['bl', 'pa', 'u', 'v', 'vis', 'vis_noise', 'wt2', 'sta2', 'tel2',\
      'u1','u2','u3','v1', 'v2','v3','cphase', 'cphase_noise', 'wt3', 'sta3','tel3', 'wave']
   return_vals = {x:np.array([]) for x in return_strs}
   hdu = fits.open(file)
#do a wavelength selection
   w     = hdu[3].data
   ww    = w["EFF_WAVE"]*1.e6 #  input wavelengths in microns
   bands = []
   waveind  = []
   wavemean = []
   for k in wavedata.keys():
      bands.append(k)
      ii = np.zeros(len(ww),bool)
      for r in wavedata[k]:
         ii = ii | ((ww>=float(r[0])) & (ww<=float(r[1])))
      waveind.append(ii)
      wavemean.append(np.mean(ww[ii]))

   nwave    = len(bands)
   wavemean = np.array(wavemean)
   return_vals['wave'] = wavemean
   adata = hdu['OI_ARRAY'].data
   sta_index = adata['sta_index']
   tel_names = adata['tel_name']
   vdata = hdu['OI_VIS2'].data
#              only v**2 data currently implemented
   vv       = vdata['vis2data']
   vv_noise = .1*vv

   vis = np.zeros((vv.shape[0],nwave))
   for l in range(nwave):
#     ix    = (ww>wave[l]-dlam)&(ww<wave[l]+dlam)
      vis[:,l]   = np.average(vv[:,waveind[l]],axis=1)
   sta2 = []  # list of station names
   for s in vdata['sta_index']:
      tt = []
      for ss in s:
         for a,b in zip(sta_index, tel_names):
            if(ss==a):tt.append(b)
      sta2.append(tt)
#  vis_noise = np.average(vv_noise[:,ix],axis=1)/np.sqrt(np.sum(ix))
   vis_noise = 0.1 + 0.2*vis # test
   u  = vdata["UCOORD"]
   v  = vdata["VCOORD"]
   t3_table = hdu['OI_T3'].data
   u1 = t3_table['u1coord']
   u2 = t3_table['u2coord']
   u3 = -u1 -u2
   v1 = t3_table['v1coord']
   v2 = t3_table['v2coord']
   v3 = -v1 -v2
   sta3 = []  # list of station names
   for s in t3_table['sta_index']:
      tt = []
      for ss in s:
         for a,b in zip(sta_index, tel_names):
            if(ss==a):tt.append(b)
      sta3.append(tt)
   cphase = np.zeros((t3_table['t3phi'].shape[0],nwave))
   for l in range(nwave):
#     ix    = (ww>wave[l]-dlam)&(ww<wave[l]+dlam)
      cphase[:,l]   = complex_average(t3_table['t3phi'][:,waveind[l]])
      
   bl = np.sqrt(u**2+v**2)
   pa = np.rad2deg(np.arctan(u/v))
   pairs = [['bl',bl], ['pa',pa], ['u',u], ['v',v], ['vis',vis],\
      ['vis_noise',vis_noise]]
#                 average over subexposures for each bcd
   for p in pairs:return_vals[p[0]] = expave6(p[1], mjdave)
   return_vals['sta2'] = expave6s(sta2, mjdave)
   return_vals['wt2']  = 0.*return_vals['vis']

   pairs = [['u1',u1], ['u2',u2], ['u3',u3], ['v1',v1], \
      ['v2',v2], ['v3',v3]] 
   for p in pairs:return_vals[p[0]] = expave4(p[1], mjdave)
   return_vals['cphase'] = expave4c(cphase, mjdave)
   return_vals['cphase_noise'] = return_vals['cphase'].copy();
   return_vals['cphase_noise'][:]=5.
   return_vals['sta3'] = expave4s(sta3, mjdave)
   return_vals['wt3'] = 0.*return_vals['cphase']

   if (mjdave):return_vals['sta3']=np.reshape(return_vals['sta3'],[return_vals['u1'].shape[0],3])
   else:return_vals['sta3'] = np.reshape(return_vals['sta3'], (return_vals['u1'].shape[0],3) )
   out = {}
   for x in return_strs: out[x] = return_vals[x]
   return out

def appendFiledata(old, new):
   if(old is None):return new 
   keys = old.keys()
   out  = {}
   for k in keys: out[k] = np.concatenate((old[k], new[k]))
   return out

def waverage(data, weights, pairs, option, flip=False):
   out = []
   for p in pairs:
      if (len(p)==1):d = data[p[0][0]] # only one data element in pairs
      elif (option=='first'):d=data[p[0][0]]
      else:
         d = 0
         w = 0
         for pp in p:
            i = pp[0]
            ww = weights[i]
            if (flip & (pp[1]==-1)):ww = -ww
            d += data[i]*ww
            w += np.abs(ww)
         if((option=='mean') and (w>0.01)): d/= w
      out.append(d)
   return np.array(out)
         
def waverage3(data, weights, pairs, option, flip=False):
   out = []
   for p in pairs:
      if (len(p)==1):d = data[p[0][0]] # only one data element in pairs
      elif (option=='first'):d=data[p[0][0]]
      else:
         d = 0
         w = 0
         for pp in p:
            i = pp[0]
            ww = weights[i]
            if (flip & (pp[1]==-1)):ww = -ww
            d += data[i]*ww
            w += np.abs(ww)
         if((option=='mean') and (w>0.01)):d/=w
      out.append(d)
   return np.array(out)
def waveragec(data, weights, pairs, option):
   out = []
   for p in pairs:
      if (len(p)==1):d = data[p[0][0]] # only one data element in pairs
      elif (option=='first'):d=data[p[0][0]]
      else:
         d = 0
         w = 0
         for pp in p:
            i  = pp[0]
            ww = weights[i]
            a = np.radians(data[i])
            if(pp[1]==1):d += np.exp(1j*a)*ww
            else:        d += np.exp(-1j*a)*ww
            w += ww
         if((option=='mean') & (w>0.01)): d/=w
      out.append(np.angle(d,True))
   return np.array(out)

def pair2(uu,vv, uvmax):
   nuv = len(uu)
   uv2 = []
   for u,v in zip(uu,vv):uv2.append(np.array([u,v]))
   done = np.zeros(nuv,bool)
   paired = []
   for i in range(nuv):  # look for near duplicate uvs
      if(not done[i]):   # not already merged
         p = [] # a baseline always pairs with itself
         for j in range(i,nuv):
            if (not done[j]):
               z = uv2match(uv2[i], uv2[j], uvmax)
               if (z is not None):
                  p.append([j,z])
                  done[j] = True  #  already paired, don't need to do it again
         paired.append(p)
   return paired
def pair3(u1,u2,u3,v1,v2,v3,uvmax):
#  search triplet baselines for matches
   nuv = len(u1) # number of triplets
   uv1 = [] # to store u,v as vectors
   uv2 = [] # to store u,v as vectors
   uv3 = [] # to store u,v as vectors
   for u,v in zip(u1, v1):uv1.append(np.array([u,v])) # create u,v vectors
   for u,v in zip(u2, v2):uv2.append(np.array([u,v])) # create u,v vectors
   for u,v in zip(u3, v3):uv3.append(np.array([u,v])) # create u,v vectors
   done = np.zeros(nuv, bool)  # mark triplets already paired
   paired = []
   for i in range(nuv):  # look for near duplicate uvs
      if(not done[i]):   # not already merged
         p = [] # a triplet always pairs with itself
         for j in range(i,nuv):
            if (not done[j]):
               z = uv3match(uv1[i],uv2[i],uv3[i], uv1[j], uv2[j], uv3[j],uvmax)
               if (z is not None):
                  p.append([j,z])
                  done[j] = True  #  already paired, don't need to do it again
         paired.append(p)
   return paired

def mergeUVs(old, uvmax = 5, reweight=False): # uvmax is maximum du,dv to merge
   keys = old.keys()
   out  = {x:np.array([]) for x in keys}
#           search single baseline u,v sets for near matches
   paired = pair2(old['u'], old['v'], uvmax)
   keys = ['bl','pa','u','v','vis','vis_noise']
   for k in ['bl','pa','vis','vis_noise']:
      out[k] = waverage(old[k], old['wt2'], paired, 'mean')
   for k in ['u','v']:
      out[k] = waverage(old[k], old['wt2'], paired, 'mean',True)
#  out['wt2'] = waverage(np.ones(len(old['wt2'])), old['wt2'], paired, 'sum')
   out['wt2'] = waverage(old['wt2'], old['wt2'], paired, 'sum')
   for k in ['tel2','sta2','obsn']:
      out[k]= waverage(old[k], old['wt2'], paired, 'first')
   paired3 = pair3(old['u1'], old['u2'], old['u3'], old['v1'], old['v2'], old['v3'],
      uvmax)
   for k in ['u1','u2','u3','v1','v2','v3','sta3','tel3','cphase_noise','obsc']:
      out[k] = waverage(old[k], old['wt3'], paired3, 'first')
#  out['wt3'] = waverage(np.ones(len(old['wt3'])), old['wt3'], paired3, 'sum')
   out['wt3'] = waverage(old['wt3'], old['wt3'], paired3, 'sum')
   if (reweight):
      notzero = out['wt2'] > 0.5
      out['wt2'][notzero] = 1.
      notzero = out['wt3'] > 0.5
      out['wt3'][notzero] = 1.
   out['cphase'] = waveragec(old['cphase'], old['wt3'], paired3, 'mean')
   return out

def read_matisse_pk(file,wave,dlam):
   return_strs = ['bl', 'pa', 'u', 'v', 'vis', 'vis_noise', 'wt2', 'sta2', 'tel2',\
      'u1','u2','u3','v1', 'v2','v3','cphase', 'cphase_noise', 'wt3', 'sta3','tel3']
   out = {x:np.array([]) for x in return_strs}
#           get calibrated, mjdaveraged ews data
   pkdata = wu.mrestore(file)
#do a wavelength selection
   ww  = pkdata['wave']
   ix  = (ww>wave-dlam)&(ww<wave+dlam)

#              only v**2 data currently implemented
   vv       = np.abs(pkdata['mvis'])**2
   vis = np.average(vv[:,ix],axis=1) # average over wave
   vis_noise = 0.1 + 0.2*vis # why not

   sta2 = []
   sta3 = []
   tels = pkdata['tels']
   telorder = [[2,3],[0,1],[1,2],[1,3],[0,2],[0,3]]  # standard baseline pairs
   teltriplets = [[0,4,5], [1,2,4], [1,3,5], [0,2,3]] # standard triplet pairs

   for i in range(6):sta2.append([tels[telorder[i][0]], tels[telorder[i][1]]])
   for tt in teltriplets:
      t = np.array(sta2)[tt]
      sta3.append(np.sort(np.unique(t)))
   u  = pkdata["uvcoord"][:,0]
   v  = pkdata["uvcoord"][:,1]
   u1 = pkdata['uvcoord1'][:,0]
   u2 = pkdata['uvcoord2'][:,0]
   u3 = -u1 -u2
   v1 = pkdata['uvcoord1'][:,1]
   v2 = pkdata['uvcoord2'][:,1]
   v3 = -v1 -v2
#                    closure phases, waveaveraged
   cphase = np.angle(np.average(pkdata['mt3'][:,ix],axis=1),True)
   cphase_noise = 5.*np.ones(cphase.shape) # assume 5 deg noise

   wt2 = np.zeros(u.shape)
   wt3 = np.zeros(cphase.shape)
      
   bl = np.sqrt(u**2+v**2)
   pa = np.rad2deg(np.arctan(u/v))
   pairs = [['bl',bl], ['pa',pa], ['u',u], ['v',v], ['vis',vis],\
      ['u1',u1], ['u2',u2], ['u3',u3], ['v1',v1], ['v2',v2],['v3',v3],\
      ['vis_noise',vis_noise], ['cphase',cphase],['cphase_noise',cphase_noise],\
      ['sta2',sta2],['sta3',sta3],['wt2',wt2],['wt3',wt3]]
   for p in pairs:out[p[0]] = p[1]

   return out

def read_matisse_pk_raw(file):
   return_strs = ['wave','bl', 'pa', 'u', 'v', 'vis', 'vis_noise', \
      'wt2', 'sta2', 'tel2',\
      'u1','u2','u3','v1', 'v2','v3','cphase', 'cphase_noise', \
      'wt3', 'sta3','tel3']
   out = {x:np.array([]) for x in return_strs}
#           get calibrated, mjdaveraged ews data
   pkdata = wu.mrestore(file)
#do a wavelength selection
   out['wave']  = pkdata['wave']

#              only v**2 data currently implemented
   vis       = np.abs(pkdata['mvis'])**2
   vis_noise = 0.1 + 0.2*vis # why not

   sta2 = []
   sta3 = []
   tels = pkdata['tels']
   telorder = [[2,3],[0,1],[1,2],[1,3],[0,2],[0,3]]  # standard baseline pairs
   teltriplets = [[0,4,5], [1,2,4], [1,3,5], [0,2,3]] # standard triplet pairs

   for i in range(6):sta2.append([tels[telorder[i][0]], tels[telorder[i][1]]])
   for tt in teltriplets:
      t = np.array(sta2)[tt]
      sta3.append(np.sort(np.unique(t)))
   u  = pkdata["uvcoord"][:,0]
   v  = pkdata["uvcoord"][:,1]
   u1 = pkdata['uvcoord1'][:,0]
   u2 = pkdata['uvcoord2'][:,0]
   u3 = -u1 -u2
   v1 = pkdata['uvcoord1'][:,1]
   v2 = pkdata['uvcoord2'][:,1]
   v3 = -v1 -v2
#                    closure phases, waveaveraged
   cphase = np.angle(pkdata['mt3'],True)
   cphase_noise = 5.*np.ones(cphase.shape) # assume 5 deg noise

   wt2 = np.zeros(u.shape)
   wt3 = np.zeros(cphase.shape[0])
      
   bl = np.sqrt(u**2+v**2)
   pa = np.rad2deg(np.arctan(u/v))
   pairs = [['bl',bl], ['pa',pa], ['u',u], ['v',v], \
      ['u1',u1],['u2',u2], ['u3',u3], ['v1',v1], ['v2',v2], ['v3',v3],\
      ['vis',vis], ['vis_noise',vis_noise], ['cphase',cphase],\
      ['cphase_noise',cphase_noise],\
      ['sta2',sta2],['sta3',sta3],['wt2',wt2],['wt3',wt3]]
   for p in pairs:out[p[0]] = p[1]

   return out
def reorder_bls(sta_indices,uvcoords,data_arr):
   #function to reorder the baselines so that they are in ascending order
   #gives correct sign to the phase
   u1,v1,u2,v2 = uvcoords
   flips = []

   for j in range(len(u1)):
      if u1[j] < 0 or u2[j] < 0: #or v1[j] < 0 or v2 [j] < 0:
         flips.append(j)
   new_data_arr = np.copy(data_arr)
   for k in flips: new_data_arr[k] = -1*new_data_arr[k]
   new_sta = np.array([np.sort(t) for t in sta_indices ])
   

   return new_sta, new_data_arr

def match_bls(sta_index):
   """Function to determine correct sign of the phase for a given baseline pairing. In principle
      when the order flips, the sign flips too.
   """
   try:
      #UTs, default case
      flips = []
      bl_ids = [[32,35], [33,34], [34,35], [32,34], [33,35], [32,33]] #for the UTs
      bl_combos = [ [sta_index[0],sta_index[1]],[sta_index[1],sta_index[2]],[sta_index[0],sta_index[2]]]
      for i,bl in enumerate(bl_combos):
         if bl[0] > bl[1]:
            bl_combos[i] = [bl[1], bl[0]]
            flips.append(i)      
      bls = [bl_ids.index(x) for x in bl_combos]
      
   except:
      try:
         #ASPRO2 (simulated case)
         flips = []
         bl_ids = [[3,4], [1,2], [2,3], [2,4], [1,3], [1,4] ]  #for aspro2
         bl_combos = [ [sta_index[0],sta_index[1]],[sta_index[1],sta_index[2]],[sta_index[0],sta_index[2]]]
         for i,bl in enumerate(bl_combos):
            if bl[0] > bl[1]:
               bl_combos[i] = [bl[1], bl[0]] 
               flips.append(i)   
         bls = [bl_ids.index(x) for x in bl_combos]
      except:
         #ATs (one case)
         flips = []
         bl_ids = [[1,28],[18,23],[23,28],[1,23],[18,28],[1,18] ] #for A0,G1,J2,K0
         bl_combos = [ [sta_index[0],sta_index[1]],[sta_index[1],sta_index[2]],[sta_index[0],sta_index[2]]]
         for i,bl in enumerate(bl_combos):
            if bl[0] > bl[1]:

               bl_combos[i] = [bl[1], bl[0]]    
               flips.append(i)
         bls = [bl_ids.index(x) for x in bl_combos]
   return bls, flips 


def my_medfilter(arr, window=3):
   new_arr = np.copy(arr)
   thresh  = 10
   k       = window-2
   for i in range(k,len(arr)-k):
      if np.absolute(new_arr[i] - new_arr[i-k]) > thresh and np.absolute(new_arr[i] - new_arr[i+k]) > thresh\
         and np.absolute(new_arr[i-k] - new_arr[i+k]) < thresh:
            mean = np.mean([new_arr[i-k], new_arr[i+k]])
            if np.absolute(mean) < np.absolute(new_arr[i]):
            #print(new_arr[i-k:i+k+1])
               new_arr[i] = mean
   return new_arr


class matisseObs():
   """
   A class to convert model surface brightness distributions to visibilities, 
   make nice plots and adjust PA and scale so that the model matches with observed data
   
   NOTES
      PA is defined east of north, i.e. counter-clockwise on sky
      here we treat the image as an image on sky, i.e. pixel coordinates = - RA coordinates
      (the RA axis increases to the left, the pixel coordinates increase to the right)
      Since the same is true for both image and (u,v) plane, we just relabel to RA / (u,v) coordinates
      at the end and keep the image and the fourier transform in pixel space.
   
   PARAMETERS
      f_model     path to FITS file of input model image
      pxscale     pixel scale in milli-arcseconds (mas)
      wave         wavelength in microns
   
   OPTIONAL PARAMETERS
      oifits      path to LIST of OIFITS file(s) containing visibilities for this object

   """
   def __init__(self, wavedata, oifits, raw=False, 
      merge=None, weightdata=None):
      self.wavedata    = wavedata
      self.oifits      = oifits
      self.raw         = raw
      self.daynames    = []
      ##
      ## read observed data
      print('Reading ',len(oifits),' datasets')
      md0   = None
      for oi in oifits:
         obsn  = oi['dayn']   # sequence number of obs
         files = oi['files']
         bcds  = getbcd(files)
         self.daynames.append(oi['dayname'])
         flip3 = -1
         for bcd,file in zip(bcds,files):
            flip3 *= -1
            if(raw): 
               if(file.split('.')[-1]=='fits'):
                  md = read_matisse_oifits_raw(file)
               elif(file.split('.')[-1]=='pk'):
                  md = read_matisse_pk_raw(file)
               else:
                  print('input file format of '+file+' not recognized')
                  return None
               self.wave = md['wave'].copy()
            else: 
               if(file.split('.')[-1]=='fits'):
                  md = read_matisse_oifits(file,wavedata)
                  self.wave = md['wave'].copy()
               elif(file.split('.')[-1]=='pk'):
                  md = read_matisse_pk(file,wavedata)
               else:
                  print('input file format of '+file+' not recognized')
                  return None
               del md['wave']
            md['obsn'] = obsn*np.ones(md['vis'].shape[0],int)
            md['obsc'] = obsn*np.ones(md['cphase'].shape[0],int)
            md['cflip3'] = flip3*np.ones(md['cphase'].shape[0],int)
            md['bcd'] = bcd*np.ones(md['cphase'].shape[0],int)
            md0 = appendFiledata(md0,md)

      tel = []
      for s,o in zip(md0['sta2'],md0['obsn']):
         a = np.sort([s[0][2], s[1][2]])
         tel.append(a[0]+a[1]+str(o))
      md0['tel2'] = np.array(tel)
      tel = []
      for s,o in zip(md0['sta3'],md0['obsc']):
         a = np.sort([s[0][2], s[1][2], s[2][2]])
         tel.append(a[0]+a[1]+a[2]+str(o))
      md0['tel3'] = np.array(tel)
      if(merge is None):self.matissedata = md0
      else:self.matissedata = mergeUVs(md0, merge)
#     if (not raw):
#        self.matissedata['mixed_obs'] = np.append(self.matissedata['vis'], self.matissedata['cphase'])
#        self.matissedata['mixed_obs_noise'] = np.append(self.matissedata['vis_noise'], self.matissedata['cphase_noise'])
      self.setweights(md0, weightdata=weightdata)
      self.setflips()
#     if(not raw):self.sumweights *= len(self.wave)
      return

   def setweights(self, data, weightdata=None):
   ##       set default weights
      data['wt2'][:] = 1.
      data['wt3'][:] = 1.
      if (weightdata is None): return
      nobs = data['obsn'].max()
   ##       single telescopes
      if('telescope' in weightdata.keys()):
         w    = weightdata['telescope']
         keys = w.keys()
         for i in range(len(data['wt2'])):
            wt   = 1.
            tel2 = data['tel2'][i]
            tels = tel2[:2]
            o    = tel2[2]
            for t in tels:
               k = t+o
               if (k in keys):wt*= w[k]
            data['wt2'][i] = np.sqrt(wt)
         for i in range(len(data['wt3'])):
            wt   = 1.
            tel3 = data['tel3'][i]
            tels = tel3[:3]
            o    = tel3[3]
            for t in tels:
               k = t+o
               if (k in keys):
                  wt*= w[k]
            data['wt3'][i] = wt**(1./3.)
   ##      baselines
      if('baseline' in weightdata.keys()):
         w    = weightdata['baseline'] # dictionary keyed by baselines
         kw   = w.keys()
         for i in range(len(data['wt2'])):
            tel2 = data['tel2'][i]
            if (tel2 in kw):data['wt2'][i] = w[tel2]

   ##        triangles
      if('triangles' in weightdata.keys()):
         w    = weightdata['triangles'] # dictionary keyed by triangles
         kw   = w.keys()
         for i in range(len(data['wt3'])):
            tel3 = data['tel3'][i]
            if (tel3 in kw):data['wt3'][i] = w[tel3]
      self.sumweights = \
         np.sum(data['wt2'])+np.sum(data['wt3'])
      return

   def setflips(self):
      md = self.matissedata
      tel3 = md['tel3']
      triangles = np.unique(tel3)
      for t in triangles:
         indices = np.where(tel3==t)[0]
         bcds    = md['bcd'][indices]
         for i3,b in zip(indices, bcds):
            if(b==3):break # select outout
         for i in indices:
            md['cflip3'][i] = self.flip3(i3,i)
      return

   def flip3(self, i1, i2):
      md=self.matissedata
      uv1a = np.array([md['u1'][i1], md['v1'][i1]])
      uv2a = np.array([md['u2'][i1], md['v2'][i1]])
      uv3a = np.array([md['u3'][i1], md['v3'][i1]])
      uv1b = np.array([md['u1'][i2], md['v1'][i2]])
      uv2b = np.array([md['u2'][i2], md['v2'][i2]])
      uv3b = np.array([md['u3'][i2], md['v3'][i2]])
      return uv3match(uv1a,uv2a, uv3a, uv1b, uv2b,uv3b,5)[0]

   def getok(self, cut=0.):
      ok2 = self.matissedata['wt2'] > cut
      ok3 = self.matissedata['wt3'] > cut
      return [ok2[:,0], ok3[:,0]]

   
   def getmatissedata(self):return self.matissedata
   def getwave(self):return self.wave

   def get2data(self, key, cut=0.):
      ok2 = self.matissedata['wt2'] > cut
      return self.matissedata[key][ok2[:,0]]
   def get3data(self, key, cut=0.):
      ok3 = self.matissedata['wt3'] > cut
      return self.matissedata[key][ok3[:,0]]

   def modelcflux(self, imagemodel):
      md       = self.matissedata
      vshape   = md['vis'].shape
      fluxarr  = np.zeros(vshape, complex)
      for l in range(len(self.wave)): # loop over wavelengths
         wave  = self.wave[l]
         for i in range(vshape[0]):
            u = md['u'][i]
            v = md['v'][i]
            valid,cf = imagemodel.cflux(u,v,wave)
            fluxarr[i,l] = cf # array of correlated fluxes
      return [valid, fluxarr]

   def modelpflux(self, imagemodel):
      return np.abs(imagemodel.cflux(0.,0.,self.wave)[1])

   def flux2vis(self, cf, pf ):
      return np.abs(cf/pf)**2

   def modelvis(self, imagemodel):
      valid,cflux    = self.modelcflux(imagemodel)
      photflux       = self.modelpflux(imagemodel)
      return [valid, cflux, photflux, self.flux2vis(cflux,photflux) ]


   def modelcphase(self, cflux, valid):  
      md        = self.matissedata
      if (not valid): return md['cphase']

      u         = md['u']   #  u-coordinates of each baseline
      v         = md['v']

      cshape    = md['cphase'].shape
      phasearr  = np.zeros(cshape)
      for i in range(cshape[0]): # loop over triangles
         u1 = md['u1'][i]
         u2 = md['u2'][i]
         u3 = md['u3'][i]
         v1 = md['v1'][i]
         v2 = md['v2'][i]
         v3 = md['v3'][i]
#                  find matching baselines
         b1   = np.where((abs(u-u1)+abs(v-v1) <.005))[0]
         b2   = np.where((abs(u-u2)+abs(v-v2) <.005))[0]
         b3   = np.where((abs(u+u3)+abs(v+v3) <.005))[0]
         vis3 = cflux[b1] * cflux[b2] * np.conj(cflux[b3])
         phasearr[i] = np.angle(vis3, True)
      return phasearr

   def modeldata(self, imagemodel):
      valid, cflux, photflux, vis2 = self.modelvis(imagemodel)
      return [valid, vis2, self.modelcphase(cflux, valid)] 

   def uvimage(self, du=2, uvl=200):
      plt.clf()
      md = self.matissedata
      if (du is None):
         for u,v,vis in zip(md['u'],md['v'],md['vis']):
            plt.plot( u, v, 'b+')
            plt.plot(-u,-v, 'b+')
            plt.ylim((140,-140))
         return
      ddu = float(du)
      ugrid,vgrid = np.meshgrid(-np.arange(-150,150,ddu), np.arange(-150,150,ddu))
      sig = 1.*ddu
      sig = -0.5/sig/sig
      vv  = 0*ugrid
      ww  = 0*ugrid
      uv2 = np.exp(-0.5*((ugrid/uvl)**2 + (vgrid/uvl)**2))
      for u,v,vis,w in zip(md['u'],md['v'],md['vis'],md['wt2']):
         for i in [-1,1]:
            e  = w*np.exp(sig * ((i*u-ugrid)**2+(i*v-vgrid)**2) )
            vv+= vis * e 
            ww+= e 
      vv/= np.clip(ww,.01,None)
      wu.wtv(vv*uv2)
      x=[];l=[]
      for i in np.arange(-150,175,25):
         x.append((150+i)/ddu)
         l.append(str(-i))
      plt.xticks(x,l);plt.yticks(x,l)
      plt.xlabel('U(m)');plt.ylabel('V(m)')
      return

   def plotraw(self, imagemodel=None, dowt=.1, prefix='s',
      waves=['3.4','3.6','3.8','4.7'],
      output=None, title='', last=400, noise=0.03, pause=1):
      if (not self.raw):
         print('raw data not available')
         return
#                  collect observed data
      md          = self.matissedata
      wave        = self.wave 
      obsn        = md['obsn'] - 1
      obsc        = md['obsc'] - 1
      days        = np.array(['22Sept','25Sept','Nov11','Nov11','Nov11'])
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
      if (dowt):    # remove downweighted observations
         ok  = wt2> dowt
         ok3 = wt3> dowt
         vis = vis[ok]
         uu  = uu[ok]
         vv  = vv[ok]
         cphase  = cphase[ok3]
         bl  = bl[ok]
         pa  = pa[ok]
         tel3  = tel3[ok3]
         tel2  = tel2[ok]
         obsn  = obsn[ok]
         obsc  = obsc[ok3]
      else:
         ok = np.ones(len(vis),bool)
         ok3= np.ones(len(cp),bool)
#                  compute and collect model data
      wavesamples = {}
      modelwaves = []
      for w in waves:
         fw = int(w[0])+0.1*int(w[2])
         modelwaves.append(fw)
         wavesamples[fw] = prefix+w[0]+w[2]+'0samples.npy'

      mvis   = []   # model visibilities
      mupper = []   # upper bounds
      mlower = []
      mcphase = []   # model closure phases
      mcupper = []
      mclower = []

      for w in modelwaves:
         self.wave    = w
         samples     = np.load(wavesamples[w])
         vs,ps = self.visstat(imagemodel, samples, last=last)
#        ps = self.phasestat(imagemodel, samples, last=last)
         if (dowt):  
            vs  = vs[:,ok]
            ps  = ps[:,ok3]
         mvis.append(vs[1])   # model visibility statistics
         mlower.append(vs[0])
         mupper.append(vs[2])
         mcphase.append(ps[1])   # model visibility statistics
         mclower.append(ps[0])
         mcupper.append(ps[2])
         print('got sample data for ',w,' microns')
      mvis   = np.array(mvis)
      mlower = np.array(mlower)
      mupper = np.array(mupper)
      mcphase = np.array(mcphase)
      mclower = np.array(mclower)
      mcupper = np.array(mcupper)
#     return {'cphase':cphase, 'wave':wave}
#     plt.close('all')
#     win = plt.get_current_fig_manager().window
#     win.move(100,0)
#     tel  = tel2
      utel = np.unique(tel2)

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
         mv.append(np.mean(mvis[:,tu],1))
         mu.append(np.mean(mupper[:,tu],1))
         ml.append(np.mean(mlower[:,tu],1))
         bb.append(np.mean(bl[tu]))
         pp.append(np.mean(pa[tu]))
         tt.append(tel2[tu[0]])
         dd.append(odays[tu[0]])
      vis = v
      nvis = len(vis)
      mv  = np.array(mv);mu=np.array(mu);ml=np.array(ml);bb=np.array(bb)
      mout= {'bb':bb,'mv':mv,'mu':mu,'ml':ml,'vis':vis,'wave':wave,
         'tt':tt,'dd':dd}
#               model visibilities, upper/lower error estimates
      wok1 = (wave>3.2) & (wave<4.02)
      wok2 = (wave>4.55) & (wave<5)
      plt.close(1)
      fig = None
      for i in range(nvis):
         if(np.mod(i,9)==0):
            if (fig is None):
               fig,ax = plt.subplots(3,3, sharex=True, sharey=True,num=1)
            else:
               fig.clear()
               ax = fig.subplots(3,3, sharex=True, sharey=True)
            for ix in range(3):ax[2,ix].set_xlabel('$\lambda(\mu$m)')
            for ix in range(3):
               for iy in range(3):ax[iy,ix].set_xticks(3+0.5*np.arange(5))
            for iy in range(3):ax[iy,0].set_ylabel('vis**2')
            fig.set_figheight(6)
            fig.set_figwidth(8)
            fig.tight_layout()
            win=plt.get_current_fig_manager().window
            win.move(400,50)
         ix = np.mod(i,3)
         iy = np.mod(i,9)//3
         ax[iy,ix].set_xlim((3.2,5.))
         ax[iy,ix].set_ylim((0,.15))
         ut = 'UT'+tt[i][0]+'UT'+tt[i][1]
         d  = dd[i]
         uv = f'{ut:7} BL={bb[i]:3.0f} PA={pp[i]:3.0f}'
         ax[iy,ix].text(3.2,0.11, uv)  # tel/uvcoords
         ax[iy,ix].text(3.2,0.13, dd[i]) # date
         for vvv in vis[i]:
            ax[iy,ix].plot(wave[wok1], vvv[wok1],'lightgrey')
            ax[iy,ix].plot(wave[wok2], vvv[wok2],'lightgrey')
         vvv = np.mean(vis[i],0)
         ax[iy,ix].plot(wave[wok1], vvv[wok1],'b')
         ax[iy,ix].plot(wave[wok2], vvv[wok2],'b')
         y    = []
         yerr = []
         n2   = noise**2
         for j in range(len(modelwaves)):
            y.append(mv[i][j])
            u = mu[i][j] - mv[i][j]
            u = np.sqrt(u**2+n2)
            l = ml[i][j] - mv[i][j]
            l = np.sqrt(l**2+n2)
            yerr.append([l,u])
         y = np.array(y)
         yerr = np.transpose(np.array(yerr))
         ax[iy,ix].errorbar(modelwaves, y, yerr, fmt='o')
         ax[iy,ix].fill_between([4.0,4.5],0,.15,color='lightgrey')

         if(np.mod(i,9)==8) & (nvis>9):
            if(output is None) :
               mpause(.1)
               xx=input()
      if (output is not None):plt.savefig(output+'vis2.png')
#     return mout
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
               pp.append(cphase[i]*flip)
               if(flip==1):
                  mcp.append(mcphase[:,i])
                  mu.append(mcupper[:,i])
                  ml.append(mclower[:,i])
                  tt.append(tel3[i])
                  dd.append(odays3[i])
               flip = -1
         pp = np.array(pp)
         cp.append(pp)
      mu = np.array(mu);ml=np.array(ml);mcp=np.array(mcp)
      noise = 5.
      plt.close(2)
      fig2 = None
      ncp  = len(cp)
      for i in range(ncp):
         if(np.mod(i,8)==0):
            if (fig2 is None):
               fig2,ax = plt.subplots(2,4, sharex=True, sharey=True,num=2)
            else:
               fig2.clear()
               ax = fig2.subplots(2,4, sharex=True, sharey=True)
            for iy in range(2):
               ax[iy,0].set_ylabel('closure phase(deg)')
               ax[iy,0].set_yticks(-180+60*np.arange(7))
            for ix in range(4):
               ax[1,ix].set_xlabel('$\lambda(\mu$m)')
               for iy in range(2):ax[iy,ix].set_xticks(3+0.5*np.arange(5))
            fig2.set_figheight(6)
            fig2.set_figwidth(8)
#            win.move(500,50)
            fig2.tight_layout()
         ix = np.mod(i,4)
         iy = np.mod(i,8)//4
         ax[iy,ix].set_xlim((3.2,5))
         ax[iy,ix].set_ylim((-200,180))
         for pp in cp[i]:
            pp[pp>120] -=360.
            ax[iy,ix].plot(wave[wok1], pp[wok1],'lightgrey')
            ax[iy,ix].plot(wave[wok2], pp[wok2],'lightgrey')
         r  = np.radians(cp[i])
         r  = np.mean(np.exp(1j*r),0)
         pp = np.angle(r,True)
         pp[pp>120] -=360.
         ax[iy,ix].plot(wave[wok1], pp[wok1],'b')
         ax[iy,ix].plot(wave[wok2], pp[wok2],'b')
         ut = 'UT'+tt[i][0]+'UT'+tt[i][1]+'UT'+tt[i][2]
         ax[iy,ix].text(3.4,110, ut)
         ax[iy,ix].text(3.4,140, dd[i])
         y    = []
         yerr = []
         n2   = noise**2
         for j in range(len(modelwaves)):
            y.append(mcp[i][j])
            u = mu[i][j] - mcp[i][j]
            u = np.sqrt(u**2+n2)
            l = ml[i][j] - mcp[i][j]
            l = np.sqrt(l**2+n2)
            yerr.append([l,u])
         y = np.array(y)
         yerr = np.transpose(np.array(yerr))
         ax[iy,ix].errorbar(modelwaves, y, yerr, fmt='o')
         ax[iy,ix].fill_between([4.0,4.5],-200,180,color='lightgrey')
         if (np.mod(i,8)==7) & (ncp>8) & (output is None):
            mpause(.1)
            x = input()
      mpause(.1)
      if(output is not None):plt.savefig(output+'cphase.png')
      return 

   def visstat(self, imagemodel, samples, last=4000):
      sa = samples[-last:]
      allvis = []
      allphase = []
      for s in sa:
         imagemodel.setvaluelist(s)
         valid, cflux, photflux,mvis = self.modelvis(imagemodel)
         allvis.append(mvis)
         allphase.append(self.modelcphase(cflux, valid))
      allvis   = np.array(allvis)
      allphase = np.array(allphase)
      return [np.percentile(allvis, [17,50,83],0), np.percentile(allphase, [17,50,83],0)]


   def viscomp(self, imagemodel,dowt=0.1, showtel=False, output=None, title='',
      bcdave=True, doprint=False, showuv=0):
      
      modelvis    = self.modelvis(imagemodel)
      modelcphase = self.modelcphase(imagemodel)

      md          = self.matissedata
      obsn        = md['obsn'] - 1
      obsc        = md['obsc'] - 1
      wt2         = md['wt2']  # weights
      wt3         = md['wt3']  # weights
      vis         = md['vis']
      uu          = md['u']
      vv          = md['v']
      cphase      = md['cphase']
      dphase      = modelcphase - cphase
      modelcphase[dphase > 180] -= 180
      modelcphase[dphase < (-180)] +=180
      s = np.array([1,-1])

      tel = []
      for t,o in zip(md['sta2'], obsn): 
         expno = o//2
         t0 = int(t[0][2]); t1 = int(t[1][2])
         if (t0<t1): tel.append(t[0][2]+t[1][2]+str(expno))
         else: tel.append(t[1][2]+t[0][2]+str(expno))
      tel  = np.array(tel)
      plt.close('all')
      if (dowt):
         ok       = wt2> dowt
         vis      = vis[ok]
         modelvis = modelvis[ok]
         tel      = tel[ok]
         uu       = uu[ok]
         vv       = vv[ok]
      if (bcdave):
         utel = np.unique(tel)
         v = []
         mv= []
         nu= []
         nv= []
         for ut in utel:
            tu = np.where(tel==ut)
            v.append(np.mean(vis[tu]))
            mv.append(np.mean(modelvis[tu]))
            nu.append(uu[tu[0][0]])
            nv.append(vv[tu[0][0]])
         vis = np.array(v)
         modelvis = np.array(mv)
         tel = utel
         uu  = np.array(nu)
         vv  = np.array(nv)
      uv2 = np.sqrt(uu**2 + vv**2)
      win = plt.get_current_fig_manager().window
      win.move(700,0)
      for v,m,t,l in zip(vis, modelvis, tel, uv2):
          c = 'k'  # default color
          if(showtel): sym = t # telescope/date
          else:sym = '+'  # default symbol
          if (showuv==1): sym = f'{l:3.0f}' # display actual value of baseline
          elif (showuv==2): c = cc(l/100.) # code uv as color
          plt.text(v,m,sym,color=c)
#     elif showuv:
#        for v,m,t in zip(vis, modelvis, uv2): plt.text(v,m,f'{t:3.0f}')
#     else:
#        plt.plot(vis, modelvis,'+')
      plt.xlabel('observed');plt.ylabel('model')
      plt.title(title+' Vis**2')
      plt.xlim((0, 1.1*np.max(vis)))
      plt.ylim((0, 1.1*np.max(modelvis)))
      plt.pause(.2)
      if doprint:
         for v,m,t,nu,nv in zip(vis, modelvis, tel,uu,vv): 
            rr = np.sqrt(nu**2+nv**2)
            print(f'{t:5} {v:8.4f} {m:8.4f} {rr:6.0f} {nu:7.0f} {nv:6.0f}')
      if (output is None):x=input()
      else:plt.savefig(output+'vis2fit.png')
#              closure phases
      plt.clf()
      tel = []
      for t,o in zip(md['sta3'], obsc): 
         expno = o//2
         tt    = []
         for i in range(3):tt.append(t[i][2])
         tt = np.sort(np.array(tt))
         tel.append(tt[0]+tt[1]+tt[2]+str(expno))
      tel  = np.array(tel)
      if (dowt):
         ok     = wt3> dowt
         cphase = cphase[ok]
         modelcphase = modelcphase[ok]
         tel  = tel[ok]
         obsc = obsc[ok]
      if (bcdave):
         utel = np.unique(tel)
         cp   = []
         mcp  = []
         for ut in utel:
            tut = tel==ut
            tu  = np.where(tut)
            st  = sum(tut)
            o   = obsc[tu] % 2
            cp.append(np.sum(cphase[tu]*s[o])/st)
            mcp.append(np.sum(modelcphase[tu]*s[o])/st)
         cphase = np.array(cp)
         modelcphase = np.array(mcp)
         tel = utel
      if (showtel): 
         for c,m,t in zip(cphase, modelcphase, tel): plt.text(c, m, t)
         plt.xlim((-180,180));plt.ylim((-180,180))
      else:plt.plot(cphase,modelcphase,'+')
      plt.xlabel('observed');plt.ylabel('model')
      plt.title(title+' Closure Phases (deg)')
      plt.pause(.2)
      if (doprint): 
         for c,m,t in zip(cphase, modelcphase, tel): 
            print(f'{t:5} {c:6.0f} {m:6.0f}')
      if(output is not None):plt.savefig(output+'cphasefit.png')
      return

   def imagetovis(self,image,replace=True):
      vis = []
      md  = self.matissedata
      image.setwave(self.wave)
      for u,v in zip(md['u'], md['v']):vis.append(image.vis(u,v))
      vis  = np.array(vis)
      vis2 = np.abs(vis)**2
      t3 = []
      for u1,u2,v1,v2 in zip(md['u1'],md['u2'],md['v1'],md['v2']):
         u3 = -u1-u2
         v3 = -v1-v2
         t3.append(image.vis(u1,v1) *image.vis(u2,v2) *image.vis(u3,v3))
      if(replace):
         md['vis']    = vis2
         md['cphase'] = np.angle(t3, True)
      else:self.imagedata = {'vis':vis2, 't3':t3, 'cphase':np.angle(t3,True)}
      return

   def fcflux(self, u, v, m, l, s, a, b, pa): 
#              calculate complex correlated flux for 
#              gaussian ellipsoid offset by l,m from center
      apb = .5*(a**2 + b**2)  # a**2 + b**2
      amb = .5*(a**2 - b**2)  # a**2 - b**2
      c2  = np.cos(2*np.radians(pa)) #  cos(2*pa)
      s2  = np.sin(2*np.radians(pa)) #  sin(2*pa)
      rr  = v**2 * (apb + c2 * amb)  #  v**2 projected on tilted axes
      rr += u**2 * (apb - c2 * amb)  #  u**2 on tilted axes
      rr += 2*u*v*amb*s2             #  cross term

      return s*np.exp(-0.5*rr +1j * (m*v - l*u))

#  def fakedata(self, m,l,s,a1,a2,ab1,ab2,pa1, pa2):
   def fakedata(self, im):
      comps = im.comps

      a1 = comps[0].values[0]
      b1 = comps[0].values[1]
      pa1= comps[0].values[2]
      a2 = comps[1].values[0]
      b2 = comps[1].values[1]
      pa2= comps[1].values[2]
      s  = comps[1].values[3]
      l  = comps[1].values[4]
      m  = comps[1].values[5]
      cw = 6.28e6/2.06e8/self.getwave()
      md = self.matissedata
      u  = md['u'];  v = md['v']
#     print(m,l,s,a1,a2,ab1,ab2,pa1,pa2)
      print(m,l,s,a1,a2,b1,b2,pa1,pa2)
      cflux0 = 1.
      uu = [md['u1'], md['u2'], md['u3']]
      vv = [md['v1'], md['v2'], md['v3']]

      for il in range(len(cw)):  # loop over waves
         c = cw[il]  # scaling factor at this wavelength
         a1s  = c*a1/2.35
         a2s  = c*a2/2.35
#        b1s  = a1s * ab1
#        b2s  = a2s * ab2
         b1s  = c*b1/2.35
         b2s  = c*b2/2.35
         cflux1 = self.fcflux(u, v, 0., 0., cflux0, a1s, b1s, pa1)
         cflux2 = self.fcflux(u, v, c*m,c*l,   s,      a2s, b2s, pa2)
         md['vis'][:,il] = np.abs(cflux1+cflux2)**2/(cflux0+s)**2
#        if (l==0):
#           print('l0',c,c*m,c*l,cflux1[3], cflux2[3], md['vis'][3,0])
      
         t3     = 1.
         for i in range(3):
            cflux1 = self.fcflux(uu[i], vv[i], 0., 0., cflux0, a1s, b1s, pa1)
            cflux2 = self.fcflux(uu[i], vv[i], c*m, c*l, s, a2s, b2s, pa2)
            t3 *= (cflux1 + cflux2)
         md['cphase'][:,il] = np.angle(t3, True)

class lmimage():
   def __init__(self, image=None, nx=128, pxscale=0.5, wave=None ):
      self.pxscale = pxscale
      self.maxm    = None
      if (wave is not None):self.setwave(wave)
      if (image is not None):
         nx            = image.shape[0]
         self.image    = image.copy()
         self.photflux = self.phot()
      else: self.image   = np.zeros((nx,nx))
      s            = nx*pxscale
      self.lgrid,self.mgrid = np.meshgrid(np.arange(-s/2,s/2,pxscale), 
         np.arange(-s/2,s/2,pxscale))  # l,m coordinates

   def setwave(self, wave):
      if(type(wave)==float): self.wave    = np.array([wave])
      elif(type(wave)==np.ndarray): self.wave  = wave
      elif (type(wave)==list):self.wave   = np.array(wave)
      else:
         print('wave must be float or array or list of floats')
      self.masm = 2*np.pi*1.e6/2.062468e8/self.wave  # mas to radians
      return

   def setimage(self, image):
      sh = image.shape
      shi= self.image.shape
      if((sh[0]!=shi[0]) | (sh[1]!=shi[1])):
         print('image size is wrong')
         return
      self.image    = image
      self.photflux = self.phot()

   def getimage(self):return self.image

   def cflux(self, u,v):
      if (self.masm is None):
         print('image wavelength not defined')
         return None
      flux = []
      for m in self.masm:
         flux.append(self.pxscale**2*np.sum(self.image*np.exp(-1j*m*(-u*self.lgrid+v*self.mgrid))))
#     return (self.pxscale)**2* \
#        np.sum(self.image*np.exp((self.masm*1j)*(-u*self.lgrid+v*self.mgrid)))
      return np.array(flux)
   def vis(self, u,v):
      return self.cflux(u,v)/self.photflux

   def phot(self): return (self.pxscale)**2*np.sum(self.image)



