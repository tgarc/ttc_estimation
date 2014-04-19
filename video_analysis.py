#!/usr/bin/python

import cv2,cv
import numpy as np
import matplotlib.pyplot as plt
import color_flow as cf
import sys

if __name__ == "__main__":
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
    # parser.add_option("--numframes", dest="numframes"
    #                   ,type="int", default=None
    #                   ,help="Number of frames to process.")
    # parser.add_option("--frameoffset", dest="offset"
    #                   ,type="int", default=0
    #                   ,help="Start processing after [offset] frames.")
    parser.add_option("-c", "--crop", dest="crop"
                      , type=float, default=0.1
                      , help="Fraction of image to crop out of analysis.")
    parser.add_option("--start", dest="start"
                      , type="int", default=0
                      , help="Starting frame number for analysis.")
    parser.add_option("--stop", dest="stop"
                      , type="int", default=None
                      , help="Stop frame number for analysis.")    
    parser.add_option("--decimate", dest="decimate"
                      , type="int", default=1
                      , help="Rate at which to downsample image for display.")    
    parser.add_option("-s", "--show", dest="show"
                      , action="store_true", default=False
                      ,help="Show results of analysis in real time.")
    parser.add_option("--vis", dest="vis"
                      , type="str", default="color"
                      , help="Method for flow visualization.")
    parser.add_option("-q", "--quiet", dest="quiet"
                      , action="store_true", default=False
                      , help="Suppress printing verbose output.")

    (opts, args) = parser.parse_args()
    if opts.video is not None:
        cap = cv2.VideoCapture(opts.video)
        
        startFrame = opts.start
        if opts.stop is None:
            endFrame = cap.get(cv.CV_CAP_PROP_FRAME_COUNT)
        else:
            endFrame = opts.stop
        if startFrame != 0:
            cap.set(cv.CV_CAP_PROP_POS_FRAMES,startFrame)
        ret, frame1 = cap.read()
    elif opts.capture is not None:
        opts.show = True
        cap = cv2.VideoCapture(opts.capture)
        startFrame, endFrame = 0, 100
        ret, frame1 = cap.read()
    elif opts.image:
        exit("No support currently")
        cap = (cv2.imread(imgpath) for imgpath in args)
        frame1 = cap.next()
        # frame1 = cv2.imread("../beyond_pixels_matlab-code_celiu/car1.jpg")
    else:
        exit("No input mode selected!")

# One of the most decent ones so far
# cap = cv2.VideoCapture("../nominal2.MP4")
# startFrame, endFrame = (100, 180)

# comparable to the previous one
# cap = cv2.VideoCapture("../nominal.mp4")
# startFrame, endFrame = (25, 180)

# Bad illumination variation in this one...
# cap = cv2.VideoCapture("../2014-04-03-201007.avi")
# startFrame,endFrame = 230,390


# frame1 = cv2.imread("../beyond_pixels_matlab-code_celiu/car1.jpg")
# startFrame, endFrame = (0, 2)

    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    del(frame1)

    # set up parameters for easy indexing into image
    imgH,   imgW    = np.shape(prvs)
    offsetX,offsetY = (int(opts.crop*imgW), int(opts.crop*imgH))
    startY, stopY   = (offsetY, imgH - offsetY)
    startX, stopX   = (offsetX, imgW - offsetX)
    winSize         = ((stopY-startY), (stopX-startX))
    xSlice, ySlice  = (slice(startX, stopX), slice(startY, stopY))

    # Set up region of interest
    roi = np.zeros_like(prvs, dtype=np.bool)
    roi[ySlice, xSlice] = True

    # Set up figure for interactive plotting
    if opts.show:
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(211); ax1.set_xticks([]); ax1.set_yticks([])
        imdisp = ax1.imshow(prvs[::opts.decimate,::opts.decimate])
        ax2 = fig.add_subplot(212)
        ax2.grid()
        ax2.set_ylabel("Max flow (px/frame)"); ax2.set_xlabel("Frame Number")
        graphdisp, = ax2.plot(startFrame,0,'m-o')
        # txt = ax1.annotate("Frame %5d" % startFrame,(0.3,0.9)
        #                   ,xycoords="figure fraction")
        fig.tight_layout()

        if opts.vis == "quiver":
            skiplines = 30
            Y, X = np.meshgrid(np.arange(startY,stopY,skiplines), np.arange(startX,stopX,skiplines))
            q = ax1.quiver(X, Y, np.zeros_like(X), np.zeros_like(Y)
                           , scale=0.5, units='x', color='cyan', alpha=0.5
                           , pivot='tail', linewidth = 1.5)

    # Set up parameters for OF calculation
    flow = np.zeros(winSize)
    params = {'pyr_scale': 0.5, 'levels': 4, 'winsize': 15, 'iterations': 30
              ,'poly_n': 5, 'poly_sigma': 1.1
              ,'flags': cv2.OPTFLOW_USE_INITIAL_FLOW, 'flow': flow}

    flowVals = np.zeros(endFrame - startFrame)
    frameIdx = np.arange(endFrame - startFrame)+startFrame
    # i = 0
    # while(1):
    #     i += 1
    for i in range(startFrame+1,endFrame):
        if not opts.quiet:
            sys.stdout.write("Processing frame %d...\r" % i)
            sys.stdout.flush()
        # frame2 = cv2.imread("../beyond_pixels_matlab-code_celiu/car2.jpg")    
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
        flowthresh = 0.01*(flowmax-flowmin)
        flow[mag < flowthresh, :] = 0

        if opts.show:  
            # represent flow angle and magnitude by color values
            if opts.vis == "color":
                fimg = cf.flowToColor(flow)
                troi = np.copy(roi)
                troi[ySlice,xSlice] &= mag > flowthresh
                frame2[troi, :] = fimg[mag > flowthresh, :]
                # frame2[ySlice, xSlice, :] = fimg
            elif opts.vis == "quiver":
                q.set_UVC(flow[::skiplines,0],flow[::skiplines,1])

            # update figure
            imdisp.set_data(frame2[::opts.decimate,::opts.decimate,:])
            wind = np.append(flowVals[max(0,i-startFrame-3):i-startFrame], np.max(mag))
            flowVals[i-startFrame-1] = np.median(wind)
            graphdisp.set_data(frameIdx[:i-startFrame], flowVals[:i-startFrame])
            if ~(i % 5):
                ax2.relim(); ax2.autoscale_view() # autoscale axes        
            # txt.set_text("Frame %5d" % i
            #              + "Flow Magnitude Range: (%f, %f)" % (flowmin, flowmax))
            fig.canvas.draw()

            # check for user input
            k = cv2.waitKey(30) & 0xff
            if k == ord('q'):
                break
            elif k == ord('s'):
                cv2.imwrite('opticalfb_frame%d.png' % i, frame2)
                cv2.imwrite('opticalhsv_frame%d.png' % i, rgb)
        prvs = nxt

    if opts.show:
        raw_input("Exit?")

    cap.release()
    plt.close()
    plt.ioff()
    del(prvs, nxt)

# cv2.destroyAllWindows()


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


