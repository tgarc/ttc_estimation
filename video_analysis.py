#!/usr/bin/python

import cv2,cv
import numpy as np
import matplotlib.pyplot as plt


def flow2hsv(flow, hsv, xSlice, ySlice):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[ySlice, xSlice, 0] = ang*180/np.pi
    hsv[ySlice, xSlice, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)


cap = cv2.VideoCapture("../nominal2.MP4")
# cap = cv2.VideoCapture(1)

startFrame, endFrame = (250, 400)

# cap = cv2.VideoCapture("../2014-04-03-201007.avi")

# startFrame,endFrame = 230,390
# cap.set(cv.CV_CAP_PROP_POS_FRAMES,startFrame)

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

# Set up figure for interactive plotting
plt.ion()
fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.set_xticks([]); ax.set_yticks([])
disp = ax.imshow(prvs, plt.cm.gray)
txt = ax.annotate("Frame %5d" % startFrame,(0.05,0.05)
                  ,xycoords="figure fraction")

# allocate matrix for visualizing flow as hue and value image
hsv = np.zeros(list(prvs.shape)+[3],dtype=np.uint8)
hsv[...,1] = 255

imgH, imgW = np.shape(prvs)
offsetX, offsetY = (0.3*imgW), (0.2*imgH)
startY,stopY = (offsetY, imgH - offsetY)
startX,stopX = (offsetX, imgW - offsetX)
winSize = ((stopY-startY), (stopX-startX))
ySlice = slice(startY,stopY)
xSlice = slice(startX,stopX)

# Set up region of interest
roi = np.zeros_like(prvs,dtype=np.bool)
roi[ySlice, xSlice] = True

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

# i = 0
# while(1):
    # i += 1
for i in range(startFrame+1,endFrame):
    ret, frame2 = cap.read()
    nxt = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    # nxt = cv2.GaussianBlur(nxt,(15,15),sigmaX=1.5)

    flow = cv2.calcOpticalFlowFarneback(prvs[roi].reshape(winSize)
                                        , nxt[roi].reshape(winSize)
                                        , **params)
    params['flow'] = flow  # save current flow values for next call

    # zero out smallest X% of flow values
    mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    flowmax, flowmin = (np.max(mag), np.min(mag))
    flowthresh = 0.1*(flowmax-flowmin)
    flow[mag < flowthresh, :] = 0

    # represent flow by hsv color values
    flow2hsv(flow, hsv, xSlice, ySlice)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # rgb[~roi, :] = frame2[~roi, :]
    frame2[roi & (hsv[..., 0] != 0), :] = rgb[roi & (hsv[..., 0] != 0), :]

    # update figure
    disp.set_data(frame2)
    # q.set_UVC(flow[::skiplines,0],flow[::skiplines,1])
    txt.set_text("Frame %5d\n" % i
                 + "Flow Magnitude Range: (%f, %f)" % (flowmin, flowmax))
    fig.canvas.draw()

    # check for user input
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb_frame%d.png' % i, frame2)
        cv2.imwrite('opticalhsv_frame%d.png' % i, rgb)
    prvs = nxt

cap.release()
# cv2.destroyAllWindows()


# p1, stat, err = cv2.calcOpticalFlowPyrLK(prvs, nxt, imgCoords, **lk_params)
# for i in range(p1[stat==1].shape[0]):
#     x1, y1 = tuple(p1[i].flat)
#     x0, y0 = tuple(imgCoords[i].flat)

#     cv2.line(nxt,(x0,y0),(x1,y1),(255,0,0),5)

# Create an image of coordinates
# imgX = np.tile(colNums, (imgH - 2*offsetY, 1))
# colNums = np.atleast_2d(np.arange(offsetX, imgW-offsetX, dtype=np.float32))
# rowNums = np.atleast_2d(np.arange(offsetY, imgH-offsetY,dtype=np.float32)).T
# imgY = np.tile(rowNums, (1, imgW - 2*offsetX))
# imgCoords = np.dstack((imgY,imgX)).reshape(((imgH-2*offsetY)*(imgW-2*offsetX),1,2))

# prvs=prvs[roi].reshape((imgH - 2*offsetY, imgW - 2*offsetX))

