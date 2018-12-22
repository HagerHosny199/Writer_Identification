import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
from sklearn.cluster import MeanShift
from sklearn.cluster import KMeans


class preprocessing_module(object):

  def __init__(self):
      self.WIDTH=2175 #image width
      self.HEIGHT=2304 #image height


  def line_segmentation(self,img):
      lines = []
      kernel = np.ones((5,100), np.uint8)
      img_erosion = cv2.erode(img, kernel, iterations=1)


        #find contours
      im2,ctrs, hier = cv2.findContours(img_erosion.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #sort contours
      sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

      for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)

         # Getting ROI
        roi = img[y:y+h, x:x+w]

        lines.append(roi)


      return lines

#get the window that contains the words
  def crop_image(self,image,y1,y2,x1,x2):
    image=image[y1:y2,x1:x2]
    return image


  #note : use WIDTH and HEIGHT
  def resize_image (self,img):
    img=cv2.resize(img, dsize=(self.HEIGHT, self.WIDTH), interpolation=cv2.INTER_CUBIC)
    return img

  # def detect_lines(self,img):
  #   horizontal_lines=[]
  #   ret_lines=[]
  #   #apply canny
  #   edges = cv2.Canny(img,50,150,apertureSize = 3)
  #   #dilation
  #   kernel = np.ones((5,5), np.uint8)
  #   img_dilation = cv2.dilate(edges, kernel, iterations=1)
  #   #hough line transform
  #   lines = cv2.HoughLinesP(img_dilation, 1, np.pi/180,200, maxLineGap=1,minLineLength=img.shape[1]/2)
  #   #getting the horizontal lines
  #   for i in range(len(lines)):
  #          for line in lines[i]:
  #               pt1 = (line[0],line[1])
  #               pt2 = (line[2],line[3])
  #               if (line[1]-line[3])==0 : #print((line[2]-line[3])//(line[0]-line[1]))
  #                       cv2.line(img, pt1, pt2, (255,255,255), 3)
  #                       horizontal_lines.append(line)
  #   #sort lines to eleminate the closed lines
  #   horizontal_lines=sorted(horizontal_lines,key=lambda x: x[1])
  #   pt=horizontal_lines[0]
  #   ret_lines.append(pt)
  #   #loop to retrive the included lines
  #   for i in range(1,len(horizontal_lines)):
  #       if abs(horizontal_lines[i][1]-pt[1])>100:
  #           #print("included",horizontal_lines[i])
  #           pt=horizontal_lines[i]
  #           ret_lines.append(pt)
  #   return ret_lines

  def detect_lines(self,img):
    horizontal_lines=[]
    ret_lines=[]
    cv2.imshow("img",cv2.resize(img,(700,500)))
    cv2.waitKey(0)
    #apply canny
    # edges = cv2.Canny(img,50,150,apertureSize = 3)
    edges = cv2.Sobel(img,-1,0,1,ksize=3)
    # cv2.imshow("edges",cv2.resize(edges,(500,500)))
    # cv2.waitKey(0)
    #dilation
    kernel = np.ones((5,5), np.uint8)
    img_dilation = cv2.dilate(edges, kernel, iterations=2)
    # cv2.imshow("dilation",cv2.resize(img_dilation,(500,500)))
    # cv2.waitKey(0)
    #hough line transform

    lines = cv2.HoughLinesP(img_dilation, 1, np.pi/180,200, maxLineGap=10,minLineLength=img.shape[1]/2)
    # print(lines)
    #getting the horizontal lines
    for i in range(len(lines)):
           for line in lines[i]:
                pt1 = (line[0],line[1])
                pt2 = (line[2],line[3])
                if (line[1]-line[3])==0 : #print((line[2]-line[3])//(line[0]-line[1]))
                        cv2.line(img, pt1, pt2, (255,255,255), 3)
                        # cv2.imshow("img2",cv2.resize(img,(500,500)))
                        # cv2.waitKey(0)
                        horizontal_lines.append(line)
    #sort lines to eleminate the closed lines
    horizontal_lines=sorted(horizontal_lines,key=lambda x: x[1])
    pt=horizontal_lines[0]
    ret_lines.append(pt)
    #loop to retrive the included lines
    for i in range(1,len(horizontal_lines)):
        if abs(horizontal_lines[i][1]-pt[1])>100:
            #print("included",horizontal_lines[i])
            pt=horizontal_lines[i]
            ret_lines.append(pt)
    return ret_lines


  def get_lines(self,img):

      im = cv2.erode(img,np.ones((3,200), np.uint8))
      im = cv2.medianBlur(im,5)
      im = cv2.bitwise_not(im)
      w= np.size(img,1)
      # cv2.imshow("i",cv2.resize(im,(700,500)))
      # cv2.waitKey(0)

      lines = []
      _,cnt,_ = cv2.findContours(im,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

      for contour in cnt:
        x=[c[0] for c in contour.reshape(-1,2)]
        y=[c[1] for c in contour.reshape(-1,2)]
        minx,miny,maxx,maxy=np.min(x),np.min(y),np.max(x),np.max(y)
        if (maxx-minx)>=(0.8*w):
            lines.append(img[miny:maxy,minx:maxx])
      return lines

  def get_lines2(self,img,hist):
        prev_y=0
        num=0
        lines=[]
        line_num=0
        for i in range(len(hist)):
            if ((i-prev_y)>0 and (num/(i-prev_y))>0.25 and hist[i]==0 and hist[i-1]!=hist[i] and i-prev_y>100) or ((i-prev_y)>0 and (num/(i-prev_y))>0.25 and hist[i]<20 and hist[i-1]<20 and hist[i-1]!=hist[i] and i-prev_y>100) :
                temp=img[prev_y:i+5,0:img.shape[1]]
                prev_y=i+5
                num=0

                line_num+=1
                lines.append(temp)
            elif hist[i]>30:
                num+=1

        return lines


  def get_histogram(self,img):
    histogram=[]
    #loop through the image to calcultae the  horizontal histogram of the image
    for i in range(img.shape[0]): #for each row
        sum=0
        for j in range(img.shape[1]): # for each col
            if img[i][j]==0:
                sum+=1
        histogram.append(sum)
    print(len(histogram),img.shape)
    return histogram


  def get_region2(self,training_data):
    for i in range(len(training_data)):
        img=training_data[i][0]
        lines=self.detect_lines(img)
        # print(lines)
        # print(img.shape)
        # print(len(lines))
        # print(img.shape,lines[1][1]+10,lines[2][1]-3,math.floor(lines[1][0]/4),img.shape[1]-1)
        cropped_img=self.crop_image(img,lines[1][1]+10,lines[2][1]-3,math.floor(lines[1][0]/4),img.shape[1]-1)
        cropped_img=self.resize_image(cropped_img)
        training_data[i][0]=cropped_img


  def get_lines3(self,img,threshold=50,winsize=100):
      hist = np.sum(img==0,axis=1)
      print(len(hist))
      candidateWhiteSpace =np.array( [i for i,value in enumerate(hist) if value<threshold]).reshape(-1,1)
      print(len(candidateWhiteSpace))
      Clustering = MeanShift(bandwidth=winsize).fit(candidateWhiteSpace)
      G=Clustering.cluster_centers_
      ycrop = sorted([int(y[0]) for y in G])
      print(ycrop)


      lines = []
      for i in range(1,len(ycrop)):
          line = img[ycrop[i-1]:ycrop[i],0:]

          if np.sum(sum(line==0))>200:
            lines.append(line)
      return lines



  def get_region(self,training_ex):

        img=training_ex
        lines=self.detect_lines(img)
        print(lines)
        print(img.shape)
        # print(len(lines))
        # print(img.shape,lines[1][1]+10,lines[2][1]-3,math.floor(lines[1][0]/4),img.shape[1]-1)
        cropped_img=self.crop_image(img,lines[1][1]+10,lines[2][1]-3,math.floor(lines[1][0]/4),img.shape[1]-1)
        cv2.imshow("mu",cv2.resize(cropped_img,(500,500)))
        cv2.waitKey(0)
        #cropped_img=self.resize_image(cropped_img)
        return cropped_img


  def get_cropped_image(self,image):
        edges = cv2.Sobel(image,-1,0,1,ksize=3)
        #_,edges = cv2.threshold(image,0,255 ,cv2.THRESH_OTSU)
        #edges = cv2.Canny(image,5,50,apertureSize = 3)
        # cv2.imshow("edges",cv2.resize(edges,(500,700)))
        # cv2.waitKey(0)

        #dilation=cv2.dilate(edges,np.ones((3,3),np.uint8),iterations=2)
        # cv2.imshow("dilation",cv2.resize(dilation,(500,700)))
        # cv2.waitKey(0)

        lines = cv2.HoughLinesP(edges,1,np.pi/180,70,maxLineGap=2,minLineLength=image.shape[1]//3)
        if lines is None or len(lines)<3:
            line_ys = self.use_hints(lines)
        else:



            # print(lines)
            # for line in lines:
            #     im=cv2.line(edges,(line[0][0],line[0][1]),(line[0][2],line[0][3]),(255,255,255),3)

            # cv2.imshow("lines",cv2.resize(im,(500,700)))
            # cv2.waitKey(0)



            lines_y_axis = np.array([[line[0][1]] for line in lines])
            kmeans = KMeans(n_clusters=3,n_init=3).fit(lines_y_axis)

            line_ys =[ int(centroid[0]) for centroid in sorted(kmeans.cluster_centers_,key = lambda x:x[0])]
            if abs(line_ys[0]-line_ys[1])<200 or abs(line_ys[1]-line_ys[2])<200:
                line_ys = self.use_hints(lines)


             # minx = min([line[0][0] for line in lines])
             # maxx = max([line[0][2] for line in lines])
             # maxx = maxx + 50 if (maxx+50<image.shape[1]) else maxx
             # minx = minx - 50 if (minx-50>0) else minx
        # minx = int(0.1 * image.shape[1])
        # maxx = image.shape[1] - minx
        minx = 20
        maxx = image.shape[1]-10
        print(line_ys)

        cropped_image =self.crop_image(image,line_ys[1],line_ys[2],minx,maxx)
        # cv2.imshow("cim",cv2.resize(cropped_image,(700,500)))
        # cv2.waitKey(0)
        return cropped_image





  def use_hints(self,lines):
      hints = [360,660,2800]
      return hints






  def otsu_threshold(selfself,img):
      _,image= cv2.threshold(img,0,255,cv2.THRESH_OTSU)
      return image





if __name__=='__main__':
    pm = preprocessing_module()
    #image = cv2.imread("../version1/data/original/a01-058.png",0)
    #image = cv2.imread("../version1/data/original/a01-003.png",0)
    #image = cv2.imread("../version1/data/original/a01-000u.png",0)
    #image = cv2.imread("../version1/data/original/a01-003x.png",0)
    image = cv2.imread("../version1/data/original/a03-054.png")
    # img = pm.line_segmentation(image)
    # cv2.imshow('ll',img[2])
    # cv2.waitKey(0)
    # img = pm.otsu_threshold(image)
    # img = pm.get_region(img)

   # hist=pm.get_histogram(img)

    # lines = pm.get_lines3(img)
    # for line in lines:
    #     cv2.imshow("l",line)
    #     cv2.waitKey(0)
   # lines=pm.get_lines2(img,hist)

    # for i in range(1,len(lines)+1):
    #     cv2.imshow("ll",lines[i-1])
    #     cv2.waitKey(0)

    # cv2.imshow("img",cv2.resize(image,(500,700)))
    # cv2.waitKey(0)
    #
    #
    #
    # edges = cv2.Sobel(image,-1,0,1,ksize=3)
    # cv2.imshow("edges",cv2.resize(edges,(500,700)))
    # cv2.waitKey(0)

    # dilation=cv2.dilate(edges,np.ones((5,5),np.uint8))
    # cv2.imshow("dilation",cv2.resize(dilation,(500,700)))
    # cv2.waitKey(0)

    # lines = cv2.HoughLinesP(edges,1,np.pi/180,90,maxLineGap=2,minLineLength=image.shape[1]//3)
    # print(lines)
    # for line in lines:
    #     im=cv2.line(edges,(line[0][0],line[0][1]),(line[0][2],line[0][3]),(255,255,255),3)
    #
    # cv2.imshow("lines",cv2.resize(im,(500,700)))
    # cv2.waitKey(0)
    #
    # if len(lines)>=3:
    #     lines_y_axis = np.array([[line[0][1]] for line in lines])
    #     kmeans = KMeans(n_clusters=3,n_init=3).fit(lines_y_axis)
    #     minx = min([line[0][0] for line in lines])
    #     maxx = max([line[0][2] for line in lines])
    #     maxx = maxx + 50 if (maxx+50<image.shape[1]) else maxx
    #     minx = minx - 50 if (minx-50>0) else minx
    #
    #     line_ys =[ int(centroid[0]) for centroid in sorted(kmeans.cluster_centers_,key = lambda x:x[0])]
    #
    # else:
    #     line_ys = pm.use_hints(lines)
    # print(line_ys)
    #
    # cropped_image = pm.crop_image(image,line_ys[1],line_ys[2],minx,maxx)
    #
    # cv2.imshow("cropped",cv2.resize(cropped_image,(700,500)))
    # cv2.waitKey(0)


    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    _,edged = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
    dilation = cv2.dilate(edged,np.ones((7,7),np.uint8),iterations=10)
    cv2.imshow("dilation",dilation)
    _,cnts, _ = cv2.findContours(dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    idx = 0
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w>100:
            idx+=1
            new_img=image[y:y+h,x:x+w]
            # cv2.imwrite(str(idx) + '.png', new_img)

            cv2.imshow("st",cv2.resize(new_img,(700,500)))
            cv2.waitKey(0)
















