#!/usr/bin/python
#------------------------------------------------------------------------------#
# Some notes and scratch code
#------------------------------------------------------------------------------#
# One of the most decent ones so far
# cap = cv2.VideoCapture("../nominal4.MP4")
# startFrame, endFrame = (40, 145)

# One of the most decent ones so far
# cap = cv2.VideoCapture("../nominal2.MP4")
# startFrame, endFrame = (100, 180)

# comparable to the previous one
# cap = cv2.VideoCapture("../nominal.mp4")
# startFrame, endFrame = (25, 180)

# Bad illumination variation in this one...
# cap = cv2.VideoCapture("../2014-04-03-201007.avi")
# startFrame,endFrame = 230,390

# cv2.calcOpticalFlowSF(prvs[roi].reshape(winSize)
#                       , nxt[roi].reshape(winSize)
#                       , params['flow'], 3, 2, 4)
#------------------------------------------------------------------------------#
import cv2,cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import color_flow as cf
import sys, time
import scipy.stats as stats
import brewer2mpl as b2m
from skimage.util import view_as_windows, view_as_blocks

EPS = np.finfo(np.float64).eps
codes = b2m.get_map('YlOrRd', 'sequential', 3)


def generic2dFilter(frame, shape, func, mode='reflect', out=None, padded=False, step=1):
    '''
    generic2dFilter

    Apply a generic function to blocks of a 2d array. Function must accept
    arguments in the form (array, axis, out) where array is a 3d array of
    flattened blocks from the original image, axis is the axis that sequences
    the blocks and out is an optional user input array to be assigned to.

    parameters
    ----------
    mode : string
    Decides how to pad the array. See numpy's pad function for a
    description.

    frame : 2 dimensional array
        2 dimensional array
    step : integer
        Describes the stride used when iterating over the array.
    '''
    bSize= shape[0]//2, shape[1]//2

    if padded:
        paddedFrame = frame
        reshapeSz = [(frame.shape[0]-2*bSize[0])//step
                     , (frame.shape[1]-2*bSize[1])//step
                     , -1]
    else:
        paddedFrame = np.pad(frame, bSize, mode=mode)
        reshapeSz = [frame.shape[0]//step, frame.shape[1]//step, -1]

    return func(view_as_windows(paddedFrame,shape,step=step).reshape(reshapeSz)
                ,axis=2, out=out)


def threshold_local(frame, shape, llim=0.01, ulim=0.99):
    '''
    threshold_local

    Trim a percentile of values from non-overlapping blocks in 2d array.
    '''
    tmp = frame.copy()
    nullmask = (frame != 0) & (frame != np.nan)

    rowblocks = view_as_blocks(tmp, shape).reshape((-1,np.prod(shape)))
    vrange = np.sort(rowblocks, axis=1)
    llimIdx, ulimIdx = round(llim*np.prod(shape)), round(ulim*np.prod(shape))

    thresh_low = vrange[ ..., [llimIdx] ]
    thresh_high = vrange[ ..., [ulimIdx] ]    
    rowblocks[..., (rowblocks < thresh_low) | (rowblocks > thresh_high)] = -1
    
    return nullmask & (tmp != -1)


def threshold_global(frame, llim=0.01, ulim=0.99):
    '''
    threshold_global

    Trim a percentile of values from array. Output should be same size as input.
    '''
    nullmask = (frame != 0) & (frame != np.nan)    
    globrange = np.sort(frame, axis=None)
    h_thresh = globrange[round(ulim*globrange.size)]
    l_thresh = globrange[round(llim*globrange.size)]
    thresh_mask = nullmask & (frame < h_thresh) & (frame > l_thresh)

    return thresh_mask, l_thresh, h_thresh
    

def init():
    # global frames (frames is a global variable here)
    imdisp.set_data(np.zeros_like(frames[0]))
    foedisp.set_data(np.zeros_like(frames[0]))    
    b_latdiv.set_height(0)
    b_verdiv.set_height(0)
    b_ttc.set_height(0)

    update = [imdisp, foedisp, b_ttc, b_verdiv, b_ttc]  
    if opts.vis == "quiver":
        q.set_UVC([],[])
        update.append(q)

    return update


def setup_plot(img):
    fig = plt.figure()
    ax1 = plt.subplot2grid((4,3), (0, 0), colspan=2, rowspan=3)
    ax2 = plt.subplot2grid((4,3), (0, 2))
    ax3 = plt.subplot2grid((4,3), (1, 2))
    ax4 = plt.subplot2grid((4,3), (2, 2))
    ax5 = plt.subplot2grid((4,3), (3, 2))

    for ax in fig.axes: ax.set_xticks([])
    ax1.set_yticks([]); ax2.set_yticks([])
    ax3.grid(); ax4.grid(); ax5.grid()
    ax2.set_title("Lateral Divergence")
    ax3.set_title("Vertical Divergence")
    ax4.set_title("TTC")

    imdisp = ax1.imshow(img,plt.cm.gray,interpolation='none')
    foedisp = ax2.imshow(img,plt.cm.gray,interpolation='none')    
    b_latdiv, = ax3.bar(0.1, 0, 0.8, color='m')
    b_verdiv, = ax4.bar(0.1, 0, 0.8, color='orange')
    b_ttc, = ax5.bar(0.1, 0, 0.8, color='g')

    for ax in (ax3,ax4,ax5):
        ax.set_xlim(0,1); ax.set_ylim(-1,1)
    ax5.set_ylim(-20,20)
    fig.tight_layout()

    return fig, imdisp, foedisp, b_latdiv, b_verdiv, b_ttc


def setup_quiver(axis, Xspan=None, Yspan=None, mask = None, skiplines=20, scale=1):
    startY, stopY = Yspan
    startX, stopX = Xspan
    Y, X = np.mgrid[startY+skiplines//2:stopY+1-skiplines//2:skiplines
                    , startX+skiplines//2:stopX+1-skiplines//2:skiplines]
    q = axis.quiver(X, Y, np.zeros_like(X), np.zeros_like(Y)
                    , np.dstack((X,Y)), units='x', scale=1/float(scale)
                    , angles='uv', alpha=0.5, edgecolor='k', pivot='tail'
                    , linewidth=0.1
                    , cmap=b2m.get_map('RdPu', 'Sequential',9).get_mpl_colormap())
    return X, Y, q


# def rectangleMask(p0, p1, shape, mask=None):
#     startX, startY = p0
#     stopX, stopY = p1

#     maskH, maskW = stopY-startY, stopX-startX
#     midH, midW = maskH//2+startY, maskW//2+startX

#     rmask = np.ones(shape, dtype=np.bool)
#     if mask is not None: rmask &= mask

#     return rmask


def generate2dTemplates(p0, p1, shape, mask=None):
    startX, startY = p0
    stopX, stopY = p1

    maskH, maskW = stopY-startY, stopX-startX
    midH, midW = maskH//2+startY, maskW//2+startX

    if mask is None:
        mask = np.ones((maskH,maskW), dtype=np.bool)

    tmask = np.zeros(shape, dtype=np.bool)
    tmask[startY:midH, startX:stopX] = True
    bmask = np.zeros(shape, dtype=np.bool)
    bmask[midH:stopY, startX:stopX] = True
    lmask = np.zeros(shape, dtype=np.bool)
    lmask[startY:stopY, startX:midW] = True
    rmask = np.zeros(shape, dtype=np.bool)
    rmask[startY:stopY, midW:stopX] = True

    return tmask&mask, bmask&mask, lmask&mask, rmask&mask


def fill_padded_edges(frame,paddedFrame,shape):
    '''
    fill_padded_edges

    Just fills in the borders of a (preallocated) padded frame with the edges of
    the original frame.
    '''
    bH, bW = shape[0]//2, shape[1]//2
    paddedFrame[bH:-bH, bW:-bW] = frame # copy original frame
    bcorners_y, bcorners_x = np.array([0,0,-1,-1]), np.array([0,-1,0,-1])

    # :i allows to maintain column orientation of column i when slicing
    # as opposed to just indexing with 'i'
    paddedFrame[:bH, bW:-bW]              = frame[:1   , :]
    paddedFrame[-bH:, bW:-bW]             = frame[-1:  , :]
    paddedFrame[bH:-bH, :bW]              = frame[:, :1]   
    paddedFrame[bH:-bH, -bW:]             = frame[:, -1:]
    paddedFrame[:bH, :bW]                 = frame[0,0]
    paddedFrame[:bH, -bW:]                = frame[0,-1]
    paddedFrame[-bH:, :bW]                = frame[-1,0]
    paddedFrame[-bH:, -bW:]               = frame[-1,-1]    
        

def generateFoEkernel(width):
    bW = width//2
    y,x = np.mgrid[-bW:bW+1,-bW:bW+1]
    angle = np.arctan2(y,x)
    return angle


def animate(i):
    global opts, framegrabber, frames, params, flowStartFrame
    global roi, flowMask, lmask, rmask, tmask, bmask
    global flowVals, times, history
    global logfile
    global codes

    update = [imdisp, foedisp]    
    t1 = time.time()
    # ------------------------------------------------------------
    # Compute optical flow
    # ------------------------------------------------------------    
    # grab the current frame, update indices
    clrframe = framegrabber.next()
    currFrame = cv2.cvtColor(clrframe, cv2.COLOR_BGR2GRAY)
    framenum = i + flowStartFrame
    times[i] = framenum

    prvs = sum(frames) / float(opts.frameavg)
    nxt = (sum(frames[1:]) + currFrame) / float(opts.frameavg)
    flow = cv2.calcOpticalFlowFarneback(prvs[startY:stopY,startX:stopX]
                                        , nxt[startY:stopY,startX:stopX]
                                        , **params)
    mag, angle = cv2.cartToPolar(flow[...,0], flow[...,1])

    # ------------------------------------------------------------
    # Remove outlier flow
    # ------------------------------------------------------------    
    if opts.nofilt:
        thresh_mask = flowMask
    else:
        # clean up flow estimates, remove outliers
        # thresh_mask = threshold_local(mag, shape=(20,20), llim=0, ulim=0.96)
        # global_mask = threshold_global(mag, llim=0.00, ulim=0.99)[0]
        global_mask = np.ones_like(flowMask)
        lthresh = 1e-6
        thresh_mask = (mag > lthresh) & global_mask
        flow[~thresh_mask] = 0
        mag[~thresh_mask] = 0

    # ------------------------------------------------------------
    # estimate the location of the FoE
    # ------------------------------------------------------------    
    # S = generic2dFilter(angle, (foeW, foeW), matchWin, step=dt, padded=True)
    # participants = generic2dFilter(thresh_mask, (foeW, foeW), np.sum
    #                                , padded=True, step=dt)
    # S /= participants
    # foe_y_subsearch, foe_x_subsearch = np.unravel_index(np.argmin(S), S.shape)
    # foe_y, foe_x = startY + foeW//2 + foe_y_subsearch*dt , startX + foeW//2 + foe_x_subsearch*dt
    foe_x, foe_y = startX + maskW//2, startY + maskW//2
    p0, p1 = (foe_x-foeW//2, foe_y-foeW//2), (foe_x+foeW//2, foe_y+foeW//2)
    # confidence= participants[foe_y_subsearch, foe_x_subsearch] / (foeW**2)
    confidence=0
    divTemplates = generate2dTemplates(p0, p1, thresh_mask.shape, thresh_mask)
    foe_tmask, foe_bmask, foe_lmask, foe_rmask = divTemplates
    foeSlice_y, foeSlice_x = slice(p0[1],p1[1]), slice(p0[0],p1[0])

    # ------------------------------------------------------------
    # estimate divergence parameters and ttc for this frame
    # ------------------------------------------------------------
    xDiv = (np.sum(flow[rmask,0]) - np.sum(flow[lmask,0])) \
           /(np.sum((lmask|rmask) & thresh_mask) + EPS)
    yDiv = (np.sum(flow[tmask,1]) - np.sum(flow[bmask,1])) \
           /(np.sum((tmask|bmask) & thresh_mask) + EPS)
    xDiv_foe = (np.sum(flow[foe_rmask,0]) - np.sum(flow[foe_lmask,0])) \
               /(np.sum(foe_lmask|foe_rmask) + EPS)
    yDiv_foe = (np.sum(flow[foe_tmask, 1]) - np.sum(flow[foe_bmask, 1])) \
               /(np.sum(foe_tmask|foe_bmask) + EPS)
    ttc = 2/(xDiv + yDiv + EPS)
    history[:, :-1] = history[:, 1:]; history[:, -1] = (xDiv,yDiv,ttc)

    # ------------------------------------------------------------    
    # use estimation history to estimate new values
    # ------------------------------------------------------------
    if i > history.shape[1]:
        flowVals[:-1, i] = np.sum(history[:-1]*w_forget, axis=1)/sum(w_forget)
        # flowVals[-1, i] = np.median(history[-1,-3:])

        # m, y0, _, _, std = stats.linregress(np.arange(5)
        #                                     , flowVals[-1,i-4:i+1]*w_forget[:5]/sum(w_forget[:5]))
        # flowVals[2, i] =  m*times[i] + y0
        # flowVals[2, i] = np.median(history[-1, -3:])
        flowVals[2, i] = stats.trim_mean(history[-1, -5:],0.4)
    else:
        flowVals[:, i] = (xDiv,yDiv,ttc)

    # ------------------------------------------------------------            
    # write out results
    # ------------------------------------------------------------
    t2 = time.time()
    out = (framenum, xDiv, yDiv, ttc, 100.*confidence, t2-t1)
    if opts.log:
        print >> logfile, ','.join(map(str,out))
    if not opts.quiet:
        sys.stdout.write("\r%4d %+6.2f %+6.2f %6.2f %6.2f %6.2f" % out)
        sys.stdout.flush()

    # ------------------------------------------------------------
    # update figure
    # ------------------------------------------------------------    
    b_latdiv.set_height(flowVals[0,i])
    b_verdiv.set_height(flowVals[1,i])
    b_ttc.set_height(flowVals[2,i])
    # cv2.rectangle(clrframe, p0, (foe_x, foe_y+foeW//2), color=(255,0,0))
    clrframe[mag <= lthresh, :] = codes.colors[0][::-1]
    clrframe[~global_mask, :] = codes.colors[-1][::-1]
    # cv2.rectangle(clrframe, p0, p1, color=(0,255,0))
    if opts.vis == "color_overlay":
        cf.colorFlow(flow, clrframe[...,::-1]
                     , slice(startX,stopX), slice(startY,stopY), thresh_mask)
        dispim = clrframe[..., ::-1]
    elif opts.vis == "color":
        dispim = cf.flowToColor(flow)
    elif opts.vis == "quiver":
        update.append(q) # add this object to those that are to be updated
        q.set_UVC(flow[flow_strides, flow_strides, 0]
                  , flow[flow_strides, flow_strides, 1]
                  , mag[flow_strides, flow_strides]*255/(np.max(mag)-np.min(mag)+EPS))
        dispim = clrframe[..., ::-1]
    imdisp.set_data(dispim)
    foedisp.set_data(clrframe[foeSlice_y, foeSlice_x, ::-1])

    # shift the frame buffer
    frames[:-1] = frames[1:]; frames[-1] = currFrame

    update.extend([b_latdiv, b_verdiv, b_ttc])
    return update


#------------------------------------------------------------------------------#
# if __name__ == "__main__":
#------------------------------------------------------------------------------#
import optparse

parser = optparse.OptionParser(usage="video_analysis.py [options]")

parser.add_option("--video", dest="video"
                  , type="str", default=None
                  , help="Video file to process.")
parser.add_option("--image", dest="image"
                  , action="store_true", default=False
                  , help="Use a sequence of images to process specified by args.")
parser.add_option("--capture", dest="capture"
                  , type="int", default=None
                  , help="Capture directly camera with given camera ID")
parser.add_option("-o", "--output", dest="output", default=None
                  , help="File to output numerical results from analysis.")
parser.add_option("--p0", dest="p0"
                  , type="str", default="0,0"
                  , help="Rectangular mask to use for cropping image.")
parser.add_option("--p1", dest="p1"
                  , type="str", default="1,1"
                  , help="Rectangular mask to use for cropping image.")
parser.add_option("--start", dest="start"
                  , type="int", default=0
                  , help="Starting frame number for analysis.")
parser.add_option("--stop", dest="stop"
                  , type="int", default=None
                  , help="Stop frame number for analysis.")
parser.add_option("--frame-average", dest="frameavg"
                  , type="int", default=1
                  , help="Use a running average over [frame-average] frames.")
parser.add_option("--decimate", dest="decimate"
                  , type="int", default=1
                  , help="Rate at which to downsample image for display.")    
parser.add_option("--save-animation", dest="save_ani"
                  , type="str", default=''
                  , help="Save animation to video file.")
parser.add_option("--log", dest="log"
                  , type="str", default=''
                  , help="Save frame by frame numerical results to log file.")
parser.add_option("-p", "--plot", dest="plot"
                  , action="store_true", default=True
                  ,help="Show plot of flow estimation over time.")
parser.add_option("--vis", dest="vis"
                  , type="str", default="quiver"
                  , help="Method for flow visualization.")
parser.add_option("--flow-sparsity", dest="flow_sparsity"
                  , type="int", default=15
                  , help="Rate at which to downsample image for display.")    
parser.add_option("-q", "--quiet", dest="quiet"
                  , action="store_true", default=False
                  , help="Suppress printing verbose output.")
parser.add_option("--no-filter", dest="nofilt"
                  , action="store_true", default=False
                  , help="Bypass filtering of optical flow results.")

#------------------------------------------------------------------------------#
# process options and set up defaults
#------------------------------------------------------------------------------#

(opts, args) = parser.parse_args()
if opts.video is not None:
    cap = cv2.VideoCapture(opts.video)
    startFrame = opts.start
    endFrame = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT)) \
               if opts.stop is None else opts.stop
    if startFrame != 0: cap.set(cv.CV_CAP_PROP_POS_FRAMES,startFrame)
    framegrabber= (cv2.GaussianBlur(cap.read()[1],(opts.decimate+1,opts.decimate+1),opts.decimate)[::opts.decimate,::opts.decimate,:].astype(np.uint8)
                   for framenum in range(startFrame, endFrame))
elif opts.capture is not None:
    opts.show = True
    cap = cv2.VideoCapture(opts.capture)
    startFrame = 0
    endFrame = 1000 if opts.stop is None else opts.stop
    framegrabber= (cv2.GaussianBlur(cap.read()[1],(opts.decimate+1,opts.decimate+1),opts.decimate)[::opts.decimate,::opts.decimate,:].astype(np.uint8)
                   for framenum in range(startFrame, endFrame))
elif opts.image:
    exit("No support currently")
    cap = None
    framegrabber = (cv2.GaussianBlur(cv2.imread(imgpath),opts.decimate+1,opts.decimate)[::opts.decimate,::opts.decimate,:]
                    for imgpath in args)
else:
    exit("No input mode selected!")
opts.p0 = map(float,opts.p0.split(','))
opts.p1 = map(float,opts.p1.split(','))

#------------------------------------------------------------------------------#
# set up a frame averager
#------------------------------------------------------------------------------#

frames = []
for i in range(opts.frameavg):
    frames.append(cv2.cvtColor(framegrabber.next(), cv2.COLOR_BGR2GRAY))
flowDataLen = endFrame - startFrame - 1 - (opts.frameavg-1)
flowStartFrame = startFrame + 1 + (opts.frameavg-1)

#------------------------------------------------------------------------------#
# set up parameters for easy indexing into image
#------------------------------------------------------------------------------#

imgH, imgW = frames[0].shape
startX, startY = map(int, (opts.p0[0]*imgW, opts.p0[1]*imgH))
stopX, stopY = map(int,(opts.p1[0]*imgW, opts.p1[1]*imgH))
maskH, maskW = stopY-startY, stopX-startX
print "Frame size:", frames[0].shape[::-1]
print "Using window:", ((startX,startY),(stopX,stopY))
print "Processing %d frames." % (endFrame-startFrame)

#------------------------------------------------------------------------------#
# Set up region of interest and divergence templates
#------------------------------------------------------------------------------#

roi = np.zeros((imgH, imgW), dtype=np.bool)
roi[startY:stopY, startX:stopX] = True
flowMask = np.ones((maskH,maskW), dtype=np.bool)
divTemplates = generate2dTemplates((startX,startY),(stopX,stopY)
                                   ,flowMask.shape,flowMask)
tmask, bmask, lmask, rmask = divTemplates
leftRect = (startX, startY), (startX+int(0.5*maskW), stopY)
rightRect = (startX+int(0.5*maskW), startY), (stopX, stopY)
topRect = (startX, startY), (stopX, startY+int(0.5*maskH))
botRect = (startX, startY+int(0.5*maskH)), (stopX, stopY)

#------------------------------------------------------------------------------#    
# Set up parameters for OF calculation
#------------------------------------------------------------------------------#    

flow = np.zeros((maskH, maskW, 2), np.float32)
flowVals = np.zeros((3,flowDataLen), np.float32)
params = {'pyr_scale': 0.5, 'levels': 4, 'winsize': 25
          , 'iterations': 10, 'poly_n': 7, 'poly_sigma': 1.2
          , 'flags': cv2.OPTFLOW_FARNEBACK_GAUSSIAN}
times = np.zeros(flowDataLen, np.float64)

# create the FoE matched filter
dt=4
foeW = 16
foeKern = generateFoEkernel(foeW).flatten()
matchWin = lambda vec, axis, out: np.sum((vec-foeKern)**2, axis=axis, out=out)

history = np.zeros((3,15))
w_forget = map(lambda x: 1-np.exp(-x/3.), np.arange(1,history.shape[1]+1))

#------------------------------------------------------------------------------#
# Set up figure for interactive plotting
#------------------------------------------------------------------------------#

fig, imdisp, foedisp, b_latdiv, b_verdiv, b_ttc = setup_plot(np.zeros_like(frames[0]))
ax1, ax2, ax3, ax4, ax5 = fig.axes

if opts.vis == 'quiver':
    skiplines = opts.flow_sparsity
    flow_strides = slice(skiplines//2,-skiplines//2+1,skiplines)
                  
    X, Y, q = setup_quiver(ax1
                           , Xspan=(startX, stopX)
                           , Yspan=(startY, stopY)
                           , scale=opts.flow_sparsity/5.
                           , skiplines=skiplines)

#------------------------------------------------------------------------------#
# Run animation
#------------------------------------------------------------------------------#

if opts.log:
    logfile = open(opts.log,'w')
    print >> logfile, '#' + ' '.join(sys.argv)
    print >> logfile, ','.join(['int','float','float','float','float','float'])
    print >> logfile, ','.join(['frame','xdiv','ydiv','ttc','confidence','runtime'])

anim = animation.FuncAnimation(fig, animate, init_func=init
                               , frames=endFrame-flowStartFrame
                               , interval=1/5., blit=True)
if opts.save_ani:
    anim.save(opts.save_ani, fps=15)
else:
    plt.show()
    plt.close()

if opts.log:
    logfile.close()
    
if cap is not None:
    cap.release()
