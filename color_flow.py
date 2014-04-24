'''
According to the MATLAB source code of Deqing Sun

Author: Thomas Garcia

According to the c++ source code of Daniel Scharstein
Contact: schar@middlebury.edu

Author: Deqing Sun, Department of Computer Science, Brown University
Contact: dqsun@cs.brown.edu
$Date: 2007-10-31 21:20:30 (Wed, 31 Oct 2006) $

Copyright 2007, Deqing Sun.


                        All Rights Reserved

Permission to use, copy, modify, and distribute this software and its
documentation for any purpose other than its incorporation into a commercial
product is hereby granted without fee, provided that the above copyright
notice appear in all copies and that both that copyright notice and this
permission notice appear in supporting documentation, and that the name of
the author and Brown University not be used in advertising or publicity
pertaining to distribution of the software without specific, written prior
permission.

THE AUTHOR AND BROWN UNIVERSITY DISCLAIM ALL WARRANTIES WITH REGARD TO THIS
SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR ANY PARTICULAR PURPOSE.  IN NO EVENT SHALL THE AUTHOR OR BROWN
UNIVERSITY BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR
ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER
IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT
OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
'''
import numpy as np
import cv2


def colorFlow(flow, colorframe, ySlice=None, xSlice=None, flowmask=None):
    fimg = flowToColor(flow)

    if ySlice is None or xSlice is None:
        colorframe = fimg # doesn't actually copy image...how to fix?
    elif flowmask is not None:
        colorframe[ySlice, xSlice, :][flowmask, :] = fimg[flowmask]
    else:
        colorframe[ySlice, xSlice, :] = fimg


def flow2hsv(flow, hsv, xSlice, ySlice):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[ySlice, xSlice, 0] = ang*180/np.pi
    hsv[ySlice, xSlice, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)


def makeColorwheel():
    '''
    color encoding scheme

    adapted from the color circle idea described at
    http://members.shaw.ca/quadibloc/other/colint.htm
    '''
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros((ncols, 3))  # r g b

    col = 0
    #RY
    colorwheel[np.arange(RY), 0] = 255
    colorwheel[np.arange(RY), 1] = np.uint8(255*np.arange(RY)/RY)
    col = col+RY

    #YG
    colorwheel[col+np.arange(YG), 0] = 255 - np.uint8(255*np.arange(YG)/YG)
    colorwheel[col+np.arange(YG), 1] = 255
    col = col+YG

    #GC
    colorwheel[col+np.arange(GC), 1] = 255
    colorwheel[col+np.arange(GC), 2] = np.uint8(255*np.arange(GC)/GC)
    col = col+GC

    #CB
    colorwheel[col+np.arange(CB), 1] = 255 - np.uint8(255*np.arange(CB)/CB)
    colorwheel[col+np.arange(CB), 2] = 255
    col = col+CB

    #BM
    colorwheel[col+np.arange(BM), 2] = 255
    colorwheel[col+np.arange(BM), 0] = np.uint8(255*np.arange(BM)/BM)
    col = col+BM

    #MR
    colorwheel[col+np.arange(MR), 2] = 255 - np.uint8(255*np.arange(MR)/MR)
    colorwheel[col+np.arange(MR), 0] = 255

    return colorwheel


# initialize a global color wheel
Colorwheel = makeColorwheel()


def computeColor(u, v):
    '''
    computeColor color codes flow field U, V
    '''
    global Colorwheel
    
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    ncols = np.size(Colorwheel, 0)

    rad = np.sqrt(u**2 + v**2)

    a = np.arctan2(-v, -u)/np.pi

    fk = (a+1)/2 * (ncols-1)   # -1~1 maped to 1~ncols
    k0 = np.int32(fk)                 # 1, 2, ..., ncols

    k1 = k0+1
    k1[k1 == ncols] = 1

    f = fk - k0

    img = np.zeros(list(np.shape(u))+[3], dtype=np.uint8)
    for i in range(0,3):
        tmp = Colorwheel[:, i]
        col0 = tmp[k0]/255
        col1 = tmp[k1]/255
        col = (1-f)*col0 + f*col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])    # increase saturation with radius
        col[~idx] = col[~idx]*0.75             # out of range

        img[:, :, i] = np.uint8(255*col*(1-nanIdx))

    return img


def flowToColor(flow, maxFlow=None):
    '''
    flowToColor(flow, maxFlow) flowToColor color codes flow field, normalize
    based on specified value, 

    flowToColor(flow) flowToColor color codes flow field, normalize
    based on maximum flow present otherwise 
    '''
    UNKNOWN_FLOW_THRESH = 1e9
    UNKNOWN_FLOW = 1e10

    u = flow[..., 0]
    v = flow[..., 1]

    maxu = -999
    maxv = -999

    minu = 999
    minv = 999
    maxrad = -1

    # % fix unknown flow
    idxUnknown = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknown] = 0
    v[idxUnknown] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u**2 + v**2)
    maxrad = max(maxrad, np.max(rad))

    if maxFlow is not None and maxFlow > 0:
        maxrad = maxFlow

    # %maxrad=14

    u = u/(maxrad+np.finfo(np.float64).eps)
    v = v/(maxrad+np.finfo(np.float64).eps)

    # % compute color
    img = computeColor(u, v)

    # % unknown flow
    img[idxUnknown, :] = 0

    return img
