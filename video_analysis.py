#!/usr/bin/python

import cv2,cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import color_flow as cf
import sys


pause = False


# class OpticalFlowMethod(object):
#     def computeOpticalFlow(self, img0, img1):
#         raise NotImplementedError("Please Implement this method")        

# class HornSchunck(OpticalFlowMethod):
def calcOpticalFlowHS(prvs, nxt, imgH, imgW, smooth=100.):
    imgH, imgW = map(int,(imgH,imgW))
    u = cv.CreateMat(imgH, imgW, cv.CV_32FC1)
    v = cv.CreateMat(imgH, imgW, cv.CV_32FC1)
    uCV = cv.CreateMat(imgH, imgW, cv.CV_8UC1)
    src = cv.fromarray(prvs)
    cv.Convert(src, uCV)
    vCV = cv.CreateMat(imgH, imgW, cv.CV_8UC1)
    src = cv.fromarray(nxt)
    cv.Convert(src, vCV)

    cv.CalcOpticalFlowHS(uCV, vCV
                         , False
                         , u
                         , v
                         , smooth
                         , (cv.CV_TERMCRIT_ITER | cv.CV_TERMCRIT_EPS, 75,0.001))
    return np.dstack((u,v))


def onkey(event):
    global pause
    pause = ~pause
    print "Pause:",pause


def threshold_flow(flow):
    mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
    flowmax, flowmin = np.nanmax(mag), np.nanmin(mag)
    flowthresh_low = 0.01*(flowmax-flowmin)
    flowthresh_high = 0.99*(flowmax-flowmin)    
    flow[(mag < flowthresh_low) | (mag > flowthresh_high), :] = 0

    return flowthresh_low, mag


def init():
    imdisp.set_data(framegrabber.next())
    # l_latdiv, = ax2.plot([],[],'-o', color='m')
    # l_verdiv, = ax3.plot([],[],'-o', color='orange')
    # l_ttc, = ax4.plot([],[],'-o', color='g')
    b_latdiv.set_height(0)
    b_verdiv.set_height(0)
    b_ttc.set_height(0)
    # txt.set_text("Frame %d of %d" % (flowStartFrame, endFrame))
    
    # update = [imdisp,l_latdiv,l_verdiv,l_ttc]
    update = [imdisp, b_ttc, b_verdiv, b_ttc]  

    # for ax in (ax2,ax3,ax4): ax.set_xlim(0,16)

    if opts.vis == "quiver":
        q.set_UVC([],[])
        update.append(q)

    return update


def setup_plot(img):
    fig = plt.figure()
    # ax1 = plt.subplot2grid((3,3), (0, 0), colspan=3, rowspan=2)
    # ax2 = plt.subplot2grid((3,3), (2, 0))
    # ax3 = plt.subplot2grid((3,3), (2, 1))
    # ax4 = plt.subplot2grid((3,3), (2, 2))
    ax1 = plt.subplot2grid((3,3), (0, 0), colspan=2, rowspan=3)
    ax2 = plt.subplot2grid((3,3), (0, 2))
    ax3 = plt.subplot2grid((3,3), (1, 2))
    ax4 = plt.subplot2grid((3,3), (2, 2))

    for ax in fig.axes: ax.set_xticks([])
    ax1.set_yticks([])
    ax2.grid(); ax3.grid(); ax4.grid()
    ax2.set_title("Lateral Divergence (px/frame)")
    ax3.set_title("Vertical Divergence (px/frame)")
    ax4.set_title("TTC (px/frame)")
    # ax4.set_xlabel("Frame Number")

    imdisp = ax1.imshow(img,plt.cm.gray)
    b_latdiv, = ax2.bar(0.1, 0, 0.8, color='m')
    b_verdiv, = ax3.bar(0.1, 0, 0.8, color='orange')
    b_ttc, = ax4.bar(0.1, 0, 0.8, color='g')
    txt = fig.text(0.1, 0.1, "")

    for ax in (ax2,ax3,ax4):
        ax.set_xlim(0,1); ax.set_ylim(-5,5)

    fig.tight_layout()

    # cid = fig.canvas.mpl_connect('key_press_event', onkey)

    # return fig, imdisp, l_latdiv, l_verdiv, l_ttc
    return fig, imdisp, txt, b_latdiv, b_verdiv, b_ttc


def setup_quiver(axis, Xspan=None, Yspan=None, mask = None, skiplines=30, scale=4):
    startY, stopY = Yspan
    startX, stopX = Xspan
    Y, X = np.mgrid[startY:stopY:skiplines, startX:stopX:skiplines]
    q = axis.quiver(X, Y, np.zeros_like(X), np.zeros_like(Y)
                    , np.dstack((X,Y)), scale=1/float(scale), units='x', alpha=0.6
                    , edgecolor='k', pivot='tail', width=1.5
                    , linewidth=0, facecolor='k'
                    , headwidth=3, headlength=5, headaxislength=5)
    return X, Y, q


def generateTemplates(maskH, maskW):
    flowMask = np.ones((maskH,maskW), dtype=np.bool)
    flowMask[0.45*maskH:0.55*maskH, 0.45*maskW:0.55*maskW] = False
    topMask = np.copy(flowMask); topMask[0.3*maskH:, :] = False
    bottomMask = np.copy(flowMask); bottomMask[:0.7*maskH, :] = False
    leftMask = np.copy(flowMask); leftMask[:, 0.45*maskW:] = False
    rightMask = np.copy(flowMask); rightMask[:, :0.55*maskW] = False

    return flowMask, topMask, bottomMask, leftMask, rightMask


def animate(i):
    global opts, framegrabber, frames, params, flowStartFrame
    global roi, flowMask, leftMask, rightMask, topMask, bottomMask
    global flowVals, frameIdx
    global ax1, ax2

    # update = [l_latdiv, l_verdiv, l_ttc, imdisp]
    update = [imdisp]    

    clrframe = framegrabber.next()
    framenum = i + flowStartFrame
    if not opts.quiet:
        sys.stdout.write("\rProcessing frame %d..." % framenum)
        sys.stdout.flush()

    # compute flow
    currFrame = cv2.cvtColor(clrframe, cv2.COLOR_RGB2GRAY)
    prvs = sum(frames) / float(opts.frameavg)
    nxt = (sum(frames[1:]) + currFrame) / float(opts.frameavg)
    flow = cv2.calcOpticalFlowFarneback(prvs[roi].reshape(flowMask.shape)
                                        , nxt[roi].reshape(flowMask.shape)
                                        , **params)
    params['flow'] = np.copy(flow)  # save current flow values for next call

    flowthresh, mag = threshold_flow(flow)
    flowVals[i,0] = computeDivergence(flow, rightMask, leftMask)
    flowVals[i,1] = computeDivergence(flow, topMask, bottomMask)
    flowVals[i,2] = 1/(flowVals[i,0]+flowVals[i,1])

    # update figure
    if opts.vis == "color":
        cf.colorFlow(flow, clrframe
                     , slice(startX,stopX)
                     , slice(startY,stopY)
                     , mag > flowthresh)
        imdisp.set_data(clrframe[::opts.decimate, ::opts.decimate, :])
    elif opts.vis == "quiver":
        update.append(q) # add this object to those that are to be updated

        q.set_UVC(flow[::skiplines,::skiplines,0][::opts.decimate]
                  ,flow[::skiplines,::skiplines,1][::opts.decimate]
                  ,mag[::skiplines,::skiplines])
        # cv2.rectangle(clrframe, *leftRect, color=(32,128,0), thickness=1)
        # cv2.rectangle(clrframe, *rightRect, color=(32,128,0), thickness=1)
        # cv2.rectangle(clrframe, *topRect, color=(128,32,0), thickness=1.5)
        # cv2.rectangle(clrframe, *botRect, color=(128,32,0), thickness=1.5)
        
        imdisp.set_data(clrframe[::opts.decimate, ::opts.decimate, :])
    # l_latdiv.set_data(frameIdx[:i+1]
    #                   , flowVals[:i+1,0])
    # l_verdiv.set_data(frameIdx[:i+1]
    #                   , flowVals[:i+1,1])
    # l_ttc.set_data(frameIdx[:i+1]
    #                   , flowVals[:i+1,2])
    b_latdiv.set_height(flowVals[i,0])
    b_verdiv.set_height(flowVals[i,1])
    b_ttc.set_height(flowVals[i,2])
    # txt.set_text("Frame %d of %d" % (framenum, endFrame))
    

    update.extend([b_latdiv, b_verdiv, b_ttc])
    # if (i % 5) == 0:
    #     for ax in (ax2,ax3,ax4):
    #         ax.set_xlim(framenum-8,framenum+8)
    #         ax.relim()
    #         ax.autoscale_view() # autoscale axes
    # fig.canvas.draw()

    frames[:-1] = frames[1:]  # drop the oldest frame
    frames[-1] = currFrame   # and add this one

    return update


def computeDivergence(flow, mask1, mask2, template='lateral'):
    flow1 = np.mean(flow[mask1, :], 0)
    flow2 = np.mean(flow[mask2, :], 0)

    d = 0 if template == 'lateral' else 1
    return flow1[d] + flow2[d]


# if __name__ == "__main__":
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
parser.add_option("-s", "--show", dest="show"
                  , action="store_true", default=False
                  ,help="Show results of analysis in real time.")
parser.add_option("-p", "--plot", dest="plot"
                  , action="store_true", default=True
                  ,help="Show plot of flow estimation over time.")
parser.add_option("--vis", dest="vis"
                  , type="str", default="color"
                  , help="Method for flow visualization.")
parser.add_option("-q", "--quiet", dest="quiet"
                  , action="store_true", default=False
                  , help="Suppress printing verbose output.")

# process options and set up defaults
#------------------------------------------------------------------------------#
(opts, args) = parser.parse_args()
if opts.video is not None:
    cap = cv2.VideoCapture(opts.video)
    startFrame = opts.start
    endFrame = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT)) \
               if opts.stop is None else opts.stop
    if startFrame != 0: cap.set(cv.CV_CAP_PROP_POS_FRAMES,startFrame)
    framegrabber= (cap.read()[1][...,::-1] for framenum in range(startFrame, endFrame))
elif opts.capture is not None:
    opts.show = True
    cap = cv2.VideoCapture(opts.capture)
    startFrame = 0
    endFrame = 100 if opts.stop is None else opts.stop
    framegrabber= (cap.read()[1][...,::-1] for framenum in range(startFrame, endFrame))
elif opts.image:
    exit("No support currently")
    cap = None
    framegrabber = (cv2.imread(imgpath) for imgpath in args)
else:
    exit("No input mode selected!")
opts.p0 = map(float,opts.p0.split(','))
opts.p1 = map(float,opts.p1.split(','))
#------------------------------------------------------------------------------#
# set up a frame averager

frames = []
for i in range(opts.frameavg):
    frames.append(cv2.cvtColor(framegrabber.next(), cv2.COLOR_BGR2GRAY))
flowDataLen = endFrame - startFrame - 1 - (opts.frameavg-1)
flowStartFrame = startFrame + 1 + (opts.frameavg-1)
#------------------------------------------------------------------------------#
# set up parameters for easy indexing into image

imgH, imgW = frames[0].shape
startX, startY = map(int, (opts.p0[0]*imgW, opts.p0[1]*imgH))
stopX, stopY = map(int,(opts.p1[0]*imgW, opts.p1[1]*imgH))
maskH, maskW = stopY-startY, stopX-startX
print "Using window:", ((startX,startY),(stopX,stopY))
#------------------------------------------------------------------------------#
# Set up region of interest and divergence templates

roi = np.zeros((imgH, imgW), dtype=np.bool)
roi[startY:stopY, startX:stopX] = True
flowMask, topMask, bottomMask, leftMask, rightMask = generateTemplates(maskH,maskW)
leftRect = (startX, startY), (int(0.45*maskW), stopY)
rightRect = (int(0.55*maskW), startY), (maskW, stopY)
# topRect = (int(0.3*maskH), startX), (int(0.45*maskW), stopY)
# botRect = (int(0.55*maskW), startY), (maskW, stopY)

#------------------------------------------------------------------------------#    
# Set up parameters for OF calculation

flow = np.zeros((maskH, maskW, 2))
flowVals = np.zeros((flowDataLen,3))
params = {'pyr_scale': 0.5, 'levels': 2, 'winsize': 15, 'iterations': 30
          ,'poly_n': 5, 'poly_sigma': 1.1  #, 'flags': 0}
          ,'flags': cv2.OPTFLOW_USE_INITIAL_FLOW, 'flow': flow}
frameIdx = np.arange(flowDataLen) + flowStartFrame
#------------------------------------------------------------------------------#
# Set up figure for interactive plotting

# fig, imdisp, l_latdiv, l_verdiv, l_ttc = setup_plot(framegrabber.next())
# ax1, ax2, ax3, ax4 = fig.axes
fig, imdisp, txt, b_latdiv, b_verdiv, b_ttc = setup_plot(framegrabber.next())
ax1, ax2, ax3, ax4 = fig.axes


if opts.vis == 'quiver':
    skiplines = 30
    X, Y, q = setup_quiver(ax1
                           , Xspan=(startX, stopX)
                           , Yspan=(startY, stopY)
                           , scale=opts.decimate*2
                           , skiplines=skiplines)

anim = animation.FuncAnimation(fig, animate , init_func=init
                               , frames=endFrame-flowStartFrame
                               , interval=1/30., blit=True)

plt.show()
if cap is not None: cap.release()
plt.close()
plt.ioff()

# if opts.show:
    # raw_input("Exit?")


# One of the most decent ones so far
# cap = cv2.VideoCapture("../nominal2.MP4")
# startFrame, endFrame = (100, 180)

# comparable to the previous one
# cap = cv2.VideoCapture("../nominal.mp4")
# startFrame, endFrame = (25, 180)

# Bad illumination variation in this one...
# cap = cv2.VideoCapture("../2014-04-03-201007.avi")
# startFrame,endFrame = 230,390


# Create an image of coordinates
# colNums = np.atleast_2d(np.arange(offsetX, imgW-offsetX, dtype=np.float32))
# rowNums = np.atleast_2d(np.arange(offsetY, imgH-offsetY,dtype=np.float32)).T
# imgX = np.tile(colNums, (imgH - 2*offsetY, 1))
# imgY = np.tile(rowNums, (1, imgW - 2*offsetX))
# imgCoords = np.dstack((imgY,imgX)).reshape(((imgH-2*offsetY)*(imgW-2*offsetX),1,2))
# p0 = imgCoords

# lk_params = dict( winSize  = (15,15),
#                   maxLevel = 2,
#                   criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# p1, stat, err = cv2.calcOpticalFlowPyrLK(prvs, nxt, imgCoords, **lk_params)
# flow = p1 - p0
# flow.reshape(list(winSize)+[2])

# for i in range(p1[stat==1].shape[0]):
#     x1, y1 = tuple(p1[i].flat)
#     x0, y0 = tuple(imgCoords[i].flat)

#     cv2.line(nxt,(x0,y0),(x1,y1),(255,0,0),5)

# prvs=prvs[roi].reshape((imgH - 2*offsetY, imgW - 2*offsetX))

# # allocate matrix for visualizing flow as hue and value image
# hsv = np.zeros(list(prvs.shape)+[3],dtype=np.uint8)
# hsv[...,1] = 255

# flow2hsv(flow, hsv, xSlice, ySlice)
# rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
# rgb[~roi, :] = frame2[~roi, :]

# txt.set_text("Frame %5d" % i
#              + "Flow Magnitude Range: (%f, %f)" % (flowmin, flowmax))

# txt = ax1.annotate("Frame %5d" % startFrame,(0.3,0.9)
#                   ,xycoords="figure fraction")


# cv2.calcOpticalFlowSF(prvs[roi].reshape(winSize)
#                       , nxt[roi].reshape(winSize)
#                       , params['flow'], 3, 2, 4)
# flow = calcOpticalFlowHS(prvs[roi].reshape(winSize)
#                          , nxt[roi].reshape(winSize)
#                          , *winSize)
