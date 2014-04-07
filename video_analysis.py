#!/usr/bin/python

import cv2,cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani

# def animate(i):
    

cap = cv2.VideoCapture("../2014-04-03-201007.avi")

startFrame,endFrame = 230,390
cap.set(cv.CV_CAP_PROP_POS_FRAMES,startFrame)

print "Resolution: %dx%d ,Rate: %f frames/sec" % (cv.CV_CAP_PROP_FRAME_WIDTH,cv.CV_CAP_PROP_FRAME_HEIGHT,cv.CV_CAP_PROP_FPS)

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
flow = np.zeros(np.shape(prvs))

fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.set_xticks([]); ax.set_yticks([])
disp = ax.imshow(prvs,cmap=plt.cm.gray)

X,Y = np.meshgrid(np.arange(np.shape(prvs)[1]),np.arange(np.shape(prvs)[0]))
skiplines = 15
X = X[::skiplines,::skiplines]
Y = Y[::skiplines,::skiplines]
q = ax.quiver(X,Y,np.zeros(np.shape(X)),np.zeros(np.shape(Y)),scale=10,units='x')

txt = ax.annotate("Frame %5d" % startFrame,(0.1,0.1),xycoords="figure fraction")

params = {'pyr_scale': 0.5, 'levels': 2, 'winsize': 5, 'iterations': 30
          ,'poly_n': 5, 'poly_sigma': 1.1
          ,'flags': cv2.OPTFLOW_USE_INITIAL_FLOW, 'flow': flow}

fig.show()

for i in range(startFrame+1,endFrame):
    ret, frame2 = cap.read()
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    next = cv2.GaussianBlur(next,(3,3),sigmaX=1)

    flow = cv2.calcOpticalFlowFarneback(prvs,next,**params)
    params['flow'] = flow
    # flow = flow/np.abs(np.max(flow)-np.min(flow))
    # flow = cv2.normalize(flow,None,-0.0001,0.0001,cv2.NORM_MINMAX)
    # cv2.imshow("blerg",next)

    
    disp.set_data(next)
    q.set_UVC(flow[::skiplines,::skiplines,0],flow[::skiplines,::skiplines,1])
    txt.set_text("Frame %5d" % i)
    fig.canvas.draw()
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png',frame2)
        cv2.imwrite('opticalhsv.png',rgb)
        
    prvs = next

cap.release()
# cv2.destroyAllWindows()

# def cvQuiver(img,x,y,u,v,size,thickness){
#   if(u==0)
#       Theta=np.pi/2;
#   else
#       Theta=atan2(float(v),float(u));

#   pt1 = (x,y)
#   pt2 = (x+u,y+v)

#   cv2.line(img,pt1,pt2,Color,thickness,8);  //Draw Line

#   # Size=(int)(Size*0.707);

#   if(Theta==PI/2 && pt1.y > pt2.y){
#       pt1.x=(int)(Size*cos(Theta)-Size*sin(Theta)+pt2.x);
#       pt1.y=(int)(Size*sin(Theta)+Size*cos(Theta)+pt2.y);
#       cv::line(img,pt1,pt2,Color,Thickness,8);  //Draw Line

#       pt1.x=(int)(Size*cos(Theta)+Size*sin(Theta)+pt2.x);
#       pt1.y=(int)(Size*sin(Theta)-Size*cos(Theta)+pt2.y);
#       cv::line(img,pt1,pt2,Color,Thickness,8);  //Draw Line
#   }
#   else{
#       pt1.x=(int)(-Size*cos(Theta)-Size*sin(Theta)+pt2.x);
#       pt1.y=(int)(-Size*sin(Theta)+Size*cos(Theta)+pt2.y);
#       cv::line(img,pt1,pt2,Color,Thickness,8);  //Draw Line

#       pt1.x=(int)(-Size*cos(Theta)+Size*sin(Theta)+pt2.x);
#       pt1.y=(int)(-Size*sin(Theta)-Size*cos(Theta)+pt2.y);
#       cv::line(img,pt1,pt2,Color,Thickness,8);  //Draw Line
# }
