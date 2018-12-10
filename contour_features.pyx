import math
import numpy as np
import cv2 as cv



class contour_feautres:
    
    f12=[0]*8
    f13=[0]*8
    f10=[0]*8
    f11=[0]*8
    maxSeg=-1

    def distribute2(self,x,lenth):

        if x>=-math.pi/2 and x < -7*math.pi/16:
            self.f10[0]+=1
            self.f11[0]+=lenth

        elif x>= (-7*math.pi/16) and x< (-5*math.pi)/16:
            self.f10[1]+=1
            self.f11[1]+=lenth

        elif x>= (-5*math.pi/16) and x< (-3*math.pi)/16:
            self.f10[2]+=1
            self.f11[2]+=lenth

        elif x>= (-3*math.pi/16) and x< (-1*math.pi)/16:
            self.f10[3]+=1
            self.f11[3]+=lenth

        elif x>= (-1*math.pi/16) and x< (1*math.pi)/16:
            self.f10[4]+=1
            self.f11[4]+=lenth

        elif x>= (1*math.pi/16) and x< (3*math.pi)/16:
            self.f10[5]+=1
            self.f11[5]+=lenth

        elif x>= (3*math.pi/16) and x< (5*math.pi)/16:
            self.f10[6]+=1
            self.f11[6]+=lenth

        elif x>= (5*math.pi/16) and x< (7*math.pi)/16:
            self.f10[7]+=1
            self.f11[7]+=lenth

        elif x>= (7*math.pi/16) and x<= (math.pi/2):
            self.f10[0]+=1
            self.f11[0]+=lenth

    def distribute(self,x,lenth):

        if x>=0 and x < math.pi/16:
            self.f12[0]+=1
            self.f13[0]+=lenth

        elif x>= (math.pi/16) and x< (3*math.pi)/16:
            self.f12[1]+=1
            self.f13[1]+=lenth

        elif x>= (3*math.pi/16) and x< (5*math.pi)/16:
            self.f12[2]+=1
            self.f13[2]+=lenth

        elif x>= (5*math.pi/16) and x< (7*math.pi)/16:
            self.f12[3]+=1
            self.f13[3]+=lenth

        elif x>= (7*math.pi/16) and x< (9*math.pi)/16:
            self.f12[4]+=1
            self.f13[4]+=lenth

        elif x>= (9*math.pi/16) and x< (11*math.pi)/16:
            self.f12[5]+=1
            self.f13[5]+=lenth

        elif x>= (11*math.pi/16) and x< (13*math.pi)/16:
            self.f12[6]+=1
            self.f13[6]+=lenth

        elif x>= (13*math.pi/16) and x< (15*math.pi)/16:
            self.f12[7]+=1
            self.f13[7]+=lenth

        elif x>= (15*math.pi/16) and x<= (math.pi):
            self.f12[0]+=1
            self.f13[0]+=lenth

    def contour_feautre_extract(self,src,n):

        self.f12=[0]*8
        self.f13=[0]*8
        self.f10=[0]*8
        self.f11=[0]*8
        self.maxSeg=-1

        #print("here")
        #print(self.f10)

        dst = cv.Canny(src, 50, 200, None, 3)
        im2,cnts,hir = cv.findContours(dst.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(src, cnts, -1, (0, 255, 0), 3)
        cv.imwrite('E:/dataset/'+str(n)+'.png',src)

        sum1=0
        sum2=0
        for c in cnts:
            for i in range(2,len(c)):
                pt2=c[i]
                pt1=c[i-1]
                pt0=c[i-2]

                v1=[pt0[0][0]-pt1[0][0],pt0[0][1]-pt1[0][1]]
                v2=[pt1[0][0]-pt2[0][0],pt1[0][1]-pt2[0][1]]

                mag1=np.sqrt(v1[0]*v1[0]+v1[1]*v1[1])
                mag2=np.sqrt(v2[0]*v2[0]+v2[1]*v2[1])

                angle=np.dot(v1,v2)/(mag1*mag2)

                if abs(angle-1)<=0.0000001: angle=1
                if abs(angle+1)<=0.0000001: angle=-1

                angle=math.pi-np.arccos(angle)
                self.distribute(angle,mag1+mag2)

                slope1=np.arctan(v1[1]/v1[0]) if v1[0]!=0 else ((math.pi/2 and v1[1]>0)or (-math.pi/2 ) )
                slope2=np.arctan(v2[1]/v2[0]) if v2[0]!=0 else ((math.pi/2 and v2[1]>0)or (-math.pi/2 ) )

                if i==2:
                    self.distribute2(slope1,mag1)
                    self.distribute2(slope2,mag2)
                    sum2=sum2+mag1+mag2
                else:
                    self.distribute2(slope2,mag2)
                    sum2=sum2+mag2


                sum1=sum1+mag1+mag2

                self.maxSeg=max(self.maxSeg,max(mag1,mag2))

        self.f13=self.f13/sum1
        self.f11=self.f11/sum2

        #print("here2")
        #print( self.f10)

        return self.f10,self.f11,self.f12,self.f13
