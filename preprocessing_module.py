import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

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

  def detect_lines(self,img):
    horizontal_lines=[]
    ret_lines=[]
    #apply canny
    edges = cv2.Canny(img,50,150,apertureSize = 3)
    #dilation
    kernel = np.ones((5,5), np.uint8)
    img_dilation = cv2.dilate(edges, kernel, iterations=1)
    #hough line transform
    lines = cv2.HoughLinesP(img_dilation, 1, np.pi/180,200, maxLineGap=1,minLineLength=img.shape[1]/2)
    #getting the horizontal lines
    for i in range(len(lines)):
           for line in lines[i]:
                pt1 = (line[0],line[1])
                pt2 = (line[2],line[3])
                if (line[1]-line[3])==0 : #print((line[2]-line[3])//(line[0]-line[1]))
                        cv2.line(img, pt1, pt2, (255,255,255), 3)
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






  def get_region2(self,training_data):
    for i in range(len(training_data)):
        img=training_data[i][0]
        lines=self.detect_lines(img)
        # print(len(lines))
        # print(img.shape,lines[1][1]+10,lines[2][1]-3,math.floor(lines[1][0]/4),img.shape[1]-1)
        cropped_img=self.crop_image(img,lines[1][1]+10,lines[2][1]-3,math.floor(lines[1][0]/4),img.shape[1]-1)
        cropped_img=self.resize_image(cropped_img)
        training_data[i][0]=cropped_img

  def get_region(self,training_ex):

        img=training_ex
        lines=self.detect_lines(img)
        # print(len(lines))
        # print(img.shape,lines[1][1]+10,lines[2][1]-3,math.floor(lines[1][0]/4),img.shape[1]-1)
        cropped_img=self.crop_image(img,lines[1][1]+10,lines[2][1]-3,math.floor(lines[1][0]/4),img.shape[1]-1)
        cropped_img=self.resize_image(cropped_img)
        return cropped_img

  def otsu_threshold(selfself,img):
      _,image= cv2.threshold(img,0,255,cv2.THRESH_OTSU)
      return image

if __name__=='__main__':
    pm = preprocessing_module()
    image = cv2.imread("../../version1/data/original/a01-058.png",0)
    # img = pm.line_segmentation(image)
    # cv2.imshow('ll',img[2])
    # cv2.waitKey(0)
    img = pm.get_region(image)
    img2 = pm.otsu_threshold(img)
    lines = pm.get_lines(img2)
    for i in range(1,len(lines)+1):
        plt.subplot(len(lines),1,i)
        plt.imshow(lines[i-1],cmap='gray')

    plt.show()










