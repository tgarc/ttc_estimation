#!/usr/bin/python

import cv2,cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani

cap = cv2.VideoCapture("../nominal2.MP4")
# cap = cv2.VideoCapture(1)

startFrame, endFrame = (200, 400)

# cap = cv2.VideoCapture("../2014-04-03-201007.avi")

# startFrame,endFrame = 230,390
# cap.set(cv.CV_CAP_PROP_POS_FRAMES,startFrame)

print "Resolution: %dx%d ,Rate: %f frames/sec" % (cv.CV_CAP_PROP_FRAME_WIDTH,cv.CV_CAP_PROP_FRAME_HEIGHT,cv.CV_CAP_PROP_FPS)

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

## Set up for interactive plotting
plt.ion()
fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.set_xticks([]); ax.set_yticks([])
disp = ax.imshow(prvs,plt.cm.gray)
txt = ax.annotate("Frame %5d" % startFrame,(0.05,0.05),xycoords="figure fraction")

## Set up region of interest
imgH, imgW = np.shape(prvs)
offsetX, offsetY = (0.3*imgW), (0.2*imgH)
startY,stopY = (offsetY, imgH - offsetY)
startX,stopX = (offsetX, imgW - offsetX)
winSize = ((stopY-startY), (stopX-startX))
ySlice = slice(startY,stopY)
xSlice = slice(startX,stopX)

mask = np.zeros_like(prvs,dtype=np.bool)
mask[ySlice, xSlice] = True

# Set up a quiver plot for visualizing flow
# skiplines = 25
# Y, X = np.meshgrid(np.arange(startY,stopY,skiplines), np.arange(startX,stopX,skiplines))
# q = ax.quiver(X, Y, np.zeros_like(X), np.zeros_like(Y)
#               , scale=0.5, units='x', color='cyan', alpha=0.5
#               , pivot='middle')

# Set up parameters for OF calculation
flow = np.zeros(winSize)
params = {'pyr_scale': 0.5, 'levels': 2, 'winsize': 15, 'iterations': 10
          ,'poly_n': 5, 'poly_sigma': 1.1
          ,'flags': cv2.OPTFLOW_USE_INITIAL_FLOW, 'flow': flow}

hsv = np.zeros(list(prvs.shape)+[3],dtype=np.uint8)
hsv[...,1] = 255

# i = 0
# while(1):
    # i += 1
for i in range(startFrame+1,endFrame):
    ret, frame2 = cap.read()
    nxt = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    # nxt = cv2.GaussianBlur(nxt,(15,15),sigmaX=1.5)

    flow = cv2.calcOpticalFlowFarneback(prvs[mask].reshape(winSize)
                                        ,nxt[mask].reshape(winSize),**params)
    params['flow'] = flow # save previous flow values for next call

    # zero out smallest X% of flow values
    mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
    flowmax, flowmin = (np.max(mag), np.min(mag))
    flowthresh = 0.1*(flowmax-flowmin)
    discardIdxs = mag < flowthresh
    flow[discardIdxs,:] = 0

    # represent flow by hsv color values
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[ySlice,xSlice,0] = ang*180/np.pi
    hsv[ySlice,xSlice,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    rgb[~mask,:] = frame2[~mask,:]

    # update figure
    disp.set_data(rgb)
    # q.set_UVC(flow[::skiplines,0],flow[::skiplines,1])
    txt.set_text("Frame %5d\nFlow Magnitude Range: (%f,%f)" % (i,flowmin,flowmax))
    fig.canvas.draw()

    # check for user input
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png',frame2)
        cv2.imwrite('opticalhsv.png',rgb)
        
    prvs = nxt

cap.release()
# cv2.destroyAllWindows()


# p1, stat, err = cv2.calcOpticalFlowPyrLK(prvs, nxt, imgCoords, **lk_params)
# for i in range(p1[stat==1].shape[0]):
#     x1, y1 = tuple(p1[i].flat)
#     x0, y0 = tuple(imgCoords[i].flat)

#     cv2.line(nxt,(x0,y0),(x1,y1),(255,0,0),5)


# imgX = np.tile(colNums, (imgH - 2*offsetY, 1))
# colNums = np.atleast_2d(np.arange(offsetX, imgW-offsetX, dtype=np.float32))
# rowNums = np.atleast_2d(np.arange(offsetY, imgH-offsetY,dtype=np.float32)).T
# imgY = np.tile(rowNums, (1, imgW - 2*offsetX))
# imgCoords = np.dstack((imgY,imgX)).reshape(((imgH-2*offsetY)*(imgW-2*offsetX),1,2))

# prvs=prvs[mask].reshape((imgH - 2*offsetY, imgW - 2*offsetX))

