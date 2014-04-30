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
# flow = calcOpticalFlowHS(prvs[roi].reshape(winSize)
#                          , nxt[roi].reshape(winSize)
#                          , *winSize)
#------------------------------------------------------------------------------#
import cv2,cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import color_flow as cf
import sys
from skimage.util import view_as_windows, view_as_blocks
from scipy import stats
from xbob.optflow.liu import cg_flow as liu_flow


# def pad(frame, shape, step=1):
#     paddedFrame = np.pad(frame, bSize, mode=mode)

# only works with odd kernels currently! (and even steps?)
def generic2dFilter(frame, shape, func, mode='reflect', out=None, padded=False, step=1):
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


def RLS(x,cov,yhat,lam=1):
    '''
    function [th,p] = rolsf(x,y,p,th,lam)
    Recursive ordinary least squares for single output case,
    including the forgetting factor, lambda.
    Enter with x = input, y = output, p = covariance, th = estimate, 
    lam = forgetting factor
    '''
    a=p*x
    g=1/(x.dot(a)+lam)
    k=g*a;
    e=y-x.dot(th)
    th=th+k*e;
    cov = (cov-g*a*a.T)/lam

    return y


def calcOpticalFlowHS(prvs, nxt, lam=100.):
    imgH, imgW = prvs.shape
    u = cv.CreateMat(imgH, imgW, cv.CV_32FC1)
    v = cv.CreateMat(imgH, imgW, cv.CV_32FC1)

    prv_cv = cv.CreateMat(imgH, imgW, cv.CV_8UC1)
    src = cv.fromarray(prvs)
    cv.Convert(src, prv_cv)

    nxt_cv = cv.CreateMat(imgH, imgW, cv.CV_8UC1)
    src = cv.fromarray(nxt)
    cv.Convert(src, nxt_cv)

    term_cond = (cv.CV_TERMCRIT_ITER | cv.CV_TERMCRIT_EPS, 75,0.001)
    cv.CalcOpticalFlowHS(prv_cv, nxt_cv
                         , False
                         , u
                         , v
                         , lam
                         , term_cond)
    return np.dstack((u,v))


# threshold an array of vectors
def trim(vec, axis, llim=0.05, ulim=0.95, fill=0):
    vrange = np.sort(vec,axis=axis)
    thresh_low = vrange[ ..., [round(llim*vrange.shape[axis])] ]
    thresh_high = vrange[ ..., [round(ulim*vrange.shape[axis])] ]
    vec[..., (vec < thresh_low) | (vec > thresh_high)] = fill


def threshold_flow_local(flow, mag, shape, llim=0.01, ulim=0.99):
    tmp = mag.copy()
    nullmask = (mag != 0) & (mag != np.nan)
    trim(view_as_blocks(tmp, shape).reshape((-1,np.prod(shape)))
         , axis=1, llim=llim, ulim=ulim, fill=-1)    

    flowmask = nullmask & (tmp != -1)
    flow[~flowmask, :] = 0

    return flowmask


def threshold_flow_global(flow, mag, llim=0.01, ulim=0.99):
    nullmask = (mag != 0) & (mag != np.nan)    
    # globrange = np.nanmax(mag) - np.nanmin(mag)
    globrange = np.sort(mag, axis=None)
    h_thresh = globrange[round(ulim*globrange.size)]
    l_thresh = globrange[round(llim*globrange.size)]
    flowmask = nullmask & (mag < h_thresh) & (mag > l_thresh)
    flow[~flowmask, :] = 0

    return flowmask

def init():
    imdisp.set_data(np.zeros_like(frames[0]))
    b_latdiv.set_height(0)
    b_verdiv.set_height(0)
    b_ttc.set_height(0)

    update = [imdisp, b_ttc, b_verdiv, b_ttc]  
    if opts.vis == "quiver":
        q.set_UVC([],[])
        update.append(q)

    return update


def setup_plot(img):
    fig = plt.figure()
    ax1 = plt.subplot2grid((3,3), (0, 0), colspan=2, rowspan=3)
    ax2 = plt.subplot2grid((3,3), (0, 2))
    ax3 = plt.subplot2grid((3,3), (1, 2))
    ax4 = plt.subplot2grid((3,3), (2, 2))

    for ax in fig.axes: ax.set_xticks([])
    ax1.set_yticks([])
    ax2.grid(); ax3.grid(); ax4.grid()
    ax2.set_title("Lateral Divergence")
    ax3.set_title("Vertical Divergence")
    ax4.set_title("TTC")

    imdisp = ax1.imshow(img,plt.cm.gray)
    b_latdiv, = ax2.bar(0.1, 0, 0.8, color='m')
    b_verdiv, = ax3.bar(0.1, 0, 0.8, color='orange')
    b_ttc, = ax4.bar(0.1, 0, 0.8, color='g')

    for ax in (ax2,ax3,ax4):
        ax.set_xlim(0,1); ax.set_ylim(-1,1)
    fig.tight_layout()

    return fig, imdisp, b_latdiv, b_verdiv, b_ttc


def setup_quiver(axis, Xspan=None, Yspan=None, mask = None, skiplines=20, scale=1):
    startY, stopY = Yspan
    startX, stopX = Xspan
    Y, X = np.mgrid[startY+skiplines//2:stopY+1-skiplines//2:skiplines
                    , startX+skiplines//2:stopX+1-skiplines//2:skiplines]
    q = axis.quiver(X, Y, np.zeros_like(X), np.zeros_like(Y)
                    , np.dstack((X,Y)), units='x', scale=1/float(scale)
                    , angles='uv', alpha=0.7, edgecolor='k', pivot='tail'
                    , width=1.5, linewidth=0.1, headwidth=3
                    , headlength=5, headaxislength=5
                    , cmap=plt.cm.cool)
    return X, Y, q


def generateTemplates(maskH, maskW):
    flowMask = np.ones((maskH,maskW), dtype=np.bool)
    topMask = np.copy(flowMask); topMask[0.45*maskH:, :] = False
    bottomMask = np.copy(flowMask); bottomMask[:0.55*maskH, :] = False
    leftMask = np.copy(flowMask); leftMask[:, 0.45*maskW:] = False
    rightMask = np.copy(flowMask); rightMask[:, :0.55*maskW] = False
    
    return flowMask, topMask, bottomMask, leftMask, rightMask


def fill_padded_edges(frame,paddedFrame,shape):
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
        
    
def animate(i):
    global opts, framegrabber, frames, params, flowStartFrame
    global roi, flowMask, leftMask, rightMask, topMask, bottomMask
    global flowVals, times, history
    global ax1, ax2
    global xDiv, yDiv, ttc, confidence
    global logfile

    update = [imdisp]    
    framenum = i + flowStartFrame
    times[i] = framenum

    # compute flow
    clrframe = framegrabber.next()
    currFrame = cv2.cvtColor(clrframe, cv2.COLOR_BGR2GRAY)
    prvs = sum(frames) / float(opts.frameavg)
    nxt = (sum(frames[1:]) + currFrame) / float(opts.frameavg)
    flow = cv2.calcOpticalFlowFarneback(prvs[startY:stopY,startX:stopX]
                                        , nxt[startY:stopY,startX:stopX]
                                        , **params)
    # flow = liu_flow(prvs[startY:stopY,startX:stopX]
    #                 , nxt[startY:stopY,startX:stopX])
    # flow = calcOpticalFlowHS(prvs[startY:stopY,startX:stopX]
    #                          , nxt[startY:stopY,startX:stopX]
    #                          , lam=10)
    params['flow'] = flow.copy()
    mag, angle = cv2.cartToPolar(flow[...,0], flow[...,1])
    
    if opts.nofilt:
        threshmask = flowMask
    else:
        # clean up flow estimates, remove outliers
        threshmask = threshold_flow_local(flow, mag, shape=(16,16)
                                          , llim=0, ulim=0.96)
        threshmask |= threshold_flow_global(flow, mag, llim=0.00, ulim=0.99)
        threshmask |= mag > 1e-9
        flow[~threshmask] = 0
        mag[~threshmask] = 0
    params['flow'] = np.copy(flow)  # save current flow values for next call

    # estimate the location of the FoE
    dt=4
    fill_padded_edges(angle, padded, foeKsz)
    S = generic2dFilter(padded, foeKsz, matchWin, padded=True, step=dt)
    participants = generic2dFilter(threshmask, foeKsz, np.sum
                                   , mode='constant', step=dt)
    S /= participants
    foe_y, foe_x = np.unravel_index(np.argmin(S), S.shape)
    confidence = participants[foe_y, foe_x] / np.prod(foeKsz)
    foe_y, foe_x = startY + foe_y*dt , startX + foe_x*dt

    cv2.rectangle(clrframe, (foe_x-foeKsz[1]//2, foe_y-foeKsz[0]//2)
                  , (foe_x+foeKsz[1]//2, foe_y+foeKsz[0]//2)
                  , color=(0,255,0))
    cv2.rectangle(clrframe, (foe_x-foeKsz[1]//2, foe_y-foeKsz[0]//2)
                  , (foe_x, foe_y+foeKsz[0]//2)
                  , color=(255,0,0))

    # estimate divergence parameters and ttc
    xDiv = (np.sum(flow[leftMask,0]) - np.sum(flow[rightMask,0]))/np.sum(leftMask|rightMask)
    yDiv = (np.sum(flow[topMask,1]) - np.sum(flow[bottomMask,1]))/np.sum(topMask|bottomMask)
    xDiv_ttc = (np.sum(flow[foe_y-foeKsz[0]//2:foe_y+foeKsz[0]//2, foe_x-foeKsz[1]//2:foe_x, 0])
                - np.sum(flow[foe_y-foeKsz[0]//2:foe_y+foeKsz[0]//2, foe_x:foe_x+foeKsz[1]//2, 0]))
    yDiv_ttc = (np.sum(flow[foe_y-foeKsz[0]//2:foe_y, foe_x-foeKsz[1]//2:foe_x+foeKsz[1]//2, 1])
                - np.sum(flow[foe_y:foe_y+foeKsz[0]//2, foe_x-foeKsz[1]//2:foe_x+foeKsz[1]//2, 1]))               
    ttc = 2/(xDiv_ttc + yDiv_ttc)
    history[:, :-1] = history[:, 1:]; history[:, -1] = (xDiv,yDiv,ttc)

    if i > history.shape[1]:
        flowVals[:2, i] = np.sum(history[:2, :]*w_forget/sum(w_forget), axis= 1)
        m, y0, _, _, std = stats.linregress(times[i-history.shape[1]+1:i+1], history[-1, :]*w_forget/sum(w_forget))
        flowVals[2, i] =  m*times[i] + y0
    else:
        flowVals[2, i] = ttc

    # write out results
    if opts.log:
        print >> logfile, ','.join(map(str,(framenum, xDiv,yDiv,ttc,100.*confidence)))
    if not opts.quiet:
        out = (framenum, xDiv,yDiv,ttc,100.*confidence)
        sys.stdout.write("\r%04d %+06.2f %+06.2f %+06.2f %+06.2f" % out)
        sys.stdout.flush()

    # update figure
    b_latdiv.set_height(flowVals[0,i] if abs(flowVals[0,i]) > 0.01 else 0)
    b_verdiv.set_height(flowVals[1,i] if abs(flowVals[1,i]) > 0.01 else 0)
    b_ttc.set_height(flowVals[2,i] if np.sum(mag[threshmask])/mag.size > 0.01 else 0)
    if opts.vis == "color_overlay":
        cf.colorFlow(flow, clrframe[...,::-1]
                     , slice(startX,stopX)
                     , slice(startY,stopY)
                     , threshmask)
        imdisp.set_data(clrframe[..., ::-1])
    elif opts.vis == "color":
        imdisp.set_data(cf.flowToColor(flow))
    elif opts.vis == "quiver":
        update.append(q) # add this object to those that are to be updated
        q.set_UVC(flow[skiplines//2:-skiplines//2+1:skiplines
                       , skiplines//2:-skiplines//2+1:skiplines, 0]
                  , flow[skiplines//2:-skiplines//2+1:skiplines
                         , skiplines//2:-skiplines//2+1:skiplines, 1]
                  , mag[skiplines//2:-skiplines//2+1:skiplines
                         , skiplines//2:-skiplines//2+1:skiplines]*255/(np.max(mag)-np.min(mag)))
        imdisp.set_data(clrframe[..., ::-1])

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
    framegrabber= (cap.read()[1][::opts.decimate,::opts.decimate,:].astype(np.uint8)
                   for framenum in range(startFrame, endFrame+1))
elif opts.capture is not None:
    opts.show = True
    cap = cv2.VideoCapture(opts.capture)
    startFrame = 0
    endFrame = 1000 if opts.stop is None else opts.stop
    framegrabber= (cap.read()[1][::opts.decimate,::opts.decimate,:].astype(np.uint8)
                   for framenum in range(startFrame, endFrame+1))
elif opts.image:
    exit("No support currently")
    cap = None
    framegrabber = (cv2.imread(imgpath)[::opts.decimate,::opts.decimate,:]
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
flowMask, topMask, bottomMask, leftMask, rightMask = generateTemplates(maskH,maskW)
leftRect = (startX, startY), (startX+int(0.5*maskW), stopY)
rightRect = (startX+int(0.5*maskW), startY), (stopX, stopY)
topRect = (startX, startY), (stopX, startY+int(0.5*maskH))
botRect = (startX, startY+int(0.5*maskH)), (stopX, stopY)

#------------------------------------------------------------------------------#    
# Set up parameters for OF calculation
#------------------------------------------------------------------------------#    

flow = np.zeros((maskH, maskW, 2), np.float32)
flowVals = np.zeros((3,flowDataLen+1), np.float32)
params = {'pyr_scale': 0.65, 'levels': 3, 'winsize': 15
          , 'iterations': 12, 'poly_n': 7, 'poly_sigma': 1.2
          , 'flags': cv2.OPTFLOW_USE_INITIAL_FLOW #| cv2.OPTFLOW_FARNEBACK_GAUSSIAN
          , 'flow': flow}
times = np.zeros(flowDataLen+1, np.float64)

# create the FoE matched filter
foeKsz = (35, 35)
y,x = np.mgrid[-foeKsz[0]//2:foeKsz[0]//2,-foeKsz[1]//2:foeKsz[1]//2]
y[foeKsz[0]//2,foeKsz[1]//2] = y[foeKsz[0]//2,foeKsz[1]//2] = 0
foeKern = np.arctan2(y,x).flatten()

padded = np.pad(np.zeros((maskH,maskW),dtype=np.float32)
                , (foeKsz[0]//2,foeKsz[1]//2), mode='constant')
history = np.zeros((3,8))
matchWin = lambda vec, axis, out: np.sum((foeKern - vec)**2, axis=axis, out=out)
w_forget = map(lambda x: 1-np.exp(-x/1.5), np.arange(1,history.shape[1]+1))
xDiv=yDiv=ttc=confidence=0

#------------------------------------------------------------------------------#
# Set up figure for interactive plotting
#------------------------------------------------------------------------------#

fig, imdisp, b_latdiv, b_verdiv, b_ttc = setup_plot(np.zeros_like(frames[0]))
ax1, ax2, ax3, ax4 = fig.axes

if opts.vis == 'quiver':
    skiplines = opts.flow_sparsity
    X, Y, q = setup_quiver(ax1
                           , Xspan=(startX, stopX)
                           , Yspan=(startY, stopY)
                           , scale=opts.decimate*2
                           , skiplines=skiplines)

#------------------------------------------------------------------------------#
# Run animation
#------------------------------------------------------------------------------#

if opts.log:
    logfile = open(opts.log,'w')
    print >> logfile, ','.join(['int','float','float','float','float'])

anim = animation.FuncAnimation(fig, animate, init_func=init
                               , frames=endFrame-startFrame
                               , interval=1/5., blit=True)
if opts.save_ani:
    anim.save(opts.save_ani, fps=15, extra_args=['-vcodec', 'libx264'])
else:
    plt.show()
    plt.close()

if opts.log:
    logfile.close()
    
if cap is not None:
    cap.release()
