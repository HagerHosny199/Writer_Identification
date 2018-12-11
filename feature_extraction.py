import numpy as np
import cv2
import preprocessing_module
import math
from scipy import stats
import statistics as stat
from contour_features import contour_feautres

class feature_extractor(object):
    # n is the length of fragment
    def edge_direction_distribution(self,n,xc,yc,edge_image):
        window_size = (2*n-3)*2 + 2
        total = 0
        hist=np.array([0 for _ in range(window_size)])
        height,width = edge_image.shape
        for y in range(height):
            for x in range(width):
                if edge_image[y][x]:
                    for i in range(window_size):
                        x1,y1 = x+xc[i],y+yc[i]
                        if x1>=0 and x1<width and y1>=0 and y1<height:
                            hist[i] = hist[i]+1 if edge_image[y1][x1] else hist[i]
                            total = total + 1

        normalized_hist = hist/total
        return normalized_hist


    def edge_based_feature(self,img):
        grad_x = cv2.Sobel(img,-1,1,0,ksize=3)
        grad_y = cv2.Sobel(img,-1,0,1,ksize=3)
        edge_image = cv2.addWeighted(grad_x,1,grad_y,1,0)
        # Fragments of length = 2
        xc = [1,1,0,-1]
        yc = [0,1,1,1]
        f4 = self.edge_direction_distribution(2,xc,yc,edge_image)
        # Fragements of length = 3
        xc = [2,2,2,1,0,-1,-2,-2]
        yc = [0,-1,-2,-2,-2,-2,-2,-1]
        f0 = self.edge_direction_distribution(3,xc,yc,edge_image)
        #Fragements of length = 4
        xc = [3,3,3,3,2,1,0,-1,-2,-3,-3,-3]
        yc = [0,-1,-2,-3,-3,-3,-3,-3,-3,-3,-2,-1]
        f1 = self.edge_direction_distribution(4,xc,yc,edge_image)
        #Fragments of length = 5
        xc = [4,4,4,4,4,3,2,1,0,-1,-2,-3,-4,-4,-4,-4]
        yc = [0,-1,-2,-3,-4,-4,-4,-4,-4,-4,-4,-4,-4,-3,-2,-1]
        f2 = self.edge_direction_distribution(5,xc,yc,edge_image)

        return np.concatenate((f0,f1,f2))


    def build_filters(self,start,end,step,size=20,sigma=4,gamma=1,psi=0,lamda=0.25):
        filters = []
        ksize = size

        for theta in np.arange(start, end, step):

            thetarad = np.deg2rad(theta)
            kern = cv2.getGaborKernel((ksize, ksize), sigma, thetarad, lamda, gamma, psi, ktype=cv2.CV_32F)
            #kern /= 1.5*kern.sum() #why this line??
            filters.append(kern)
        return filters

    def Gabor_filter_features(self,img):

         filters = self.build_filters(0,181,10)
         gabor_features=[]

         for kern in filters:
            fimg = cv2.filter2D(img, -1, kern)
            fimg = cv2.normalize(fimg,fimg,0,1,cv2.NORM_MINMAX,dtype=cv2.CV_32FC1)

            img_sum = sum(sum(np.fabs(fimg)))
            gabor_features.append(img_sum)


         return gabor_features


    def find_contour_perimeter(self,im,x,y):

        dx= [1,0,-1,0]
        dy= [0,1,0,-1]
        h = np.size(im,0)
        w = np.size(im,1)
        cnt = 0

        for i in range(4):
            xnew,ynew = x + dx[i],y + dy[i]
            if xnew >= 0 and xnew < w and ynew>=0 and ynew<h:

                if im[ynew][xnew]==255 and (im[y][x]==0 or im[y][x]==100):
                    cnt = cnt+1
                    im[y][x]=200 #border

                else:
                    im[y][x]=100
                    if im[ynew][xnew] == 0:
                        cnt = cnt + self.find_contour_perimeter(im,xnew,ynew)
        return cnt



    def normalize(self,v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def enclosed_regions_features(self,img):

        im = cv2.bitwise_not(img) #invert the image
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea=True
        params.minArea = 5
        detector = cv2.SimpleBlobDetector_create()
        keypoints = detector.detect(im)

        avg_size = 0
        avg_f = 0
        avg_r = 0
        cnt = len(keypoints)
        if cnt!=0:
         for blob in keypoints:
            xc,yc = int(blob.pt[0]),int(blob.pt[1])
            l = self.find_contour_perimeter(img,xc,yc)
            a = blob.size
            l2 = l*l
            r = l2 / a
            f = (4*math.pi*a)/l2
            avg_f = avg_f + f
            avg_size = avg_size + a
            avg_r = avg_r + r

        avg_size = avg_size/cnt if cnt!=0 else 0
        avg_r = avg_r/cnt if cnt!=0 else 0
        avg_f = avg_f/cnt if cnt!=0 else 0
        return self.normalize([avg_size,avg_f,avg_r])



    def minimum_error_line_segments(self,x,y):
        lx = len(x)
        errs = []
        slopes = []


        if lx == 2:
            errs.append(0)
            s = (y[0]-y[1])/(x[0]-x[1])
            slopes.append((0,s,0))

        elif lx == 3:
            errs.append(0)
            s1 = (y[0]-y[1])/(x[0]-x[1])
            s2 = (y[1]-y[2])/(x[1]-x[2])
            slopes.append((0,s1,s2))

        else:
            for i in range(1,lx-1):
                for j in range(i+1,lx-1):


                    if i < j :
                        reg_res_s2 = stats.linregress(x[i:j+1],y[i:j+1])
                        s2 = reg_res_s2[0]
                        err2 = reg_res_s2[4]
                        if(err2<0):
                            print("lessthan 0")
                        reg_res_s1 = stats.linregress(x[:i+1],y[:i+1])
                        s1 = reg_res_s1[0]
                        err1 = reg_res_s1[4]
                        reg_res_s3 = stats.linregress(x[j:],y[j:])
                        s3 = reg_res_s3[0]
                        err3 = reg_res_s3[4]
                        if(err3<0):
                            print("lessthan 0")
                        if(err1<0):
                            print("lessthan 0")
                        err = err1 + err2 +err3
                        slopes.append((s1,s2,s3))
                        errs.append(err)
                    else :

                        break


        min_i_val,min_i= (errs[0],0)
        for i,j in enumerate(errs):
            if(j<min_i_val):
                min_i_val,min_i = (j,i)

        return slopes[min_i]



    def fractal_features(self,line):


        h= np.size(line,0)
        w = np.size(line,1)
        area = h*w

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))

        y=[]
        x=[]
        for i in range(1,10):
            line = cv2.dilate(line,kernel,iterations=1)
            a = area - cv2.countNonZero(line)

            if a<i:
                break
            g = math.log(a/i)
            y.append(g)
            x.append(math.log(i))

        f0,f1,f2 = self.minimum_error_line_segments(x,y)
        return (f0,f1,f2)


    def extract_features_from_lines(self,img,pm):
        '''
        Calculates the mean feature vectors for an image
        :param img: cropped image
        :param pm: preprocessing module
        :return: mean feature vector of lines
        '''
        print("extracting features from lines")
        features = []
        lines = pm.get_lines(img)
        for line in lines:
            # print("Extracting fractal features")
            #f0,f1,f2 = self.fractal_features(line)
            #print("Applying gabor filter")
            #f3 = self.Gabor_filter_features(line)

            #f4 = self.edge_based_feature(line)
            # f6 = self.enclosed_regions_features(line)
       #     f6 = self.connected_comp_feautres2(line)
            #f7 = self.white_space2(line)
            cf = contour_feautres()
            f8 = cf.contour_feautre_extract(line)


           # line_feature = np.concatenate(([f0,f1,f2]))
            features.append(f8)

        mean_feature_vector = np.mean(features,axis=0)
        # std_dev = np.std(features,axis=0)
        # feature_vec =np.concatenate((mean_feature_vector,std_dev))

        return mean_feature_vector


    def extract_features_from_document(self,img):
        return self.Gabor_filter_features(img)

    def extract_features_from_line(self,line):
            # print("Extracting fractal features")
            f0,f1,f2 = self.fractal_features(line)
            #print("Applying gabor filter")
            f3 = self.Gabor_filter_features(line)

            f4 = self.edge_based_feature(line)
            # f6 = self.enclosed_regions_features(line)


            line_feature = np.concatenate(([f0,f1,f2],f3,f4))
            return line_feature






    def  white_space2(self,image):

        # set the variable extension types

        #whiteSpaceList = []
        MaxwhiteSpaceList = []
        feautres=[]

        h = image.shape[0]
        w = image.shape[1]
        max_transitions=-1



        for y in range(0,h):
            #find line with max transitions
            transitions=0
            sumW=0
            last_pixel_color=-1
            whiteSpaceList=[]

            for x in range(0, w):

                if image[y,x]<127 and last_pixel_color==1:
                    transitions+=1
                    whiteSpaceList.append(sumW)
                    last_pixel_color=0
                    sumW=0


                elif image[y,x]>127 and last_pixel_color==0:
                    transitions+=1
                    sumW+=1
                    last_pixel_color=1

                elif image[y,x]>127 and last_pixel_color==1:
                    sumW+=1
                    if x==w-1:
                        whiteSpaceList.append(sumW)

                elif image[y,x]>127 and last_pixel_color==-1:
                    sumW+=1
                    last_pixel_color=1

                elif image[y,x]<127 and last_pixel_color==-1:
                    last_pixel_color=0



            if transitions>max_transitions:
                max_transitions=transitions
                MaxwhiteSpaceList=whiteSpaceList
                row=y
        #you now have max and its corresponding vector find median push it as a feautre

        #MaxwhiteSpaceList.sort()
        #print(MaxwhiteSpaceList)
        feautres.append(stat.median(MaxwhiteSpaceList))


        return feautres




    def connected_comp_feautres2(self,image):


        avg_dist_comp=[]
        avg_w=[]
        std_w=[]
        median_w=[]
        avg_inter_d=[]
        avg_word_d=[]
        formFactor=[]
        roundness=[]
        blobSize=[]
        avg_dist=[]



        blackimg = (image <= 127).astype(np.uint8)

        #whiteimg = (image >= 127).astype(np.uint8)

        imgWidth = image.shape[1]
        threshold= 0.01*imgWidth

        avg=0
        width=[]

        output = cv2.connectedComponentsWithStats(blackimg, 8, cv2.CV_32S)
        output_loops = cv2.connectedComponentsWithStats(image, 8, cv2.CV_32S)


        components=output[2]
        components_loops=output_loops[2]

        components=sorted(components, key = lambda x: x[0])  #AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAT2KDY DRORY
        components_loops=sorted(components_loops, key = lambda x: x[0])  #AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAT2KDY DRORY

        newco=[]   ###############################################################################NEW
        for co in components:
            if not( co[2]==imgWidth  and co[0]==0):
                newco.append(co)
        components=newco

        newco=[]
        for co in components_loops:
            if  not(co[2]>=0.95*imgWidth  and co[0]==0):
                newco.append(co)
        components_loops=newco


        width.append(components[0][2])

        noWords=0
        nointerWords=0
        inter_word_distance=0
        distance_bet_words=0

        for j in range(1,len(components)):
                avg+=(components[j][0]-(components[j-1][0]+components[j-1][2]))
                width.append(components[j][2])

                if (components[j][0]-(components[j-1][0]+components[j-1][2]))< threshold:
                    inter_word_distance+=(components[j][0]-(components[j-1][0]+components[j-1][2]))
                    nointerWords+=1
                else:
                    distance_bet_words += (components[j][0]-(components[j-1][0]+components[j-1][2]))
                    noWords+=1

        inter_word_distance=inter_word_distance/nointerWords if nointerWords!=0 else inter_word_distance
        distance_bet_words=distance_bet_words/noWords if noWords!=0 else distance_bet_words

        avg_inter_d.append(inter_word_distance)
        avg_word_d.append(distance_bet_words)

        if(len(components)):
            avg/=len(components)

        avg_w_c=stat.mean(width)

        # std_w_c=stat.stdev(width)
        std_w_c = np.std(width)

        median_w_c=stat.median(width)

        avg_w.append(avg_w_c)
        std_w.append(std_w_c)
        median_w.append(median_w_c)
        avg_dist.append(avg)


        avgformfac=0
        avgroundess=0
        avgblobsize=0
        h = image.shape[0]
        w = image.shape[1]

        for loopComp in components_loops:

                bound=0
                avgformfac=0
                avgroundess=0
                avgblobsize=0

                for i in range (loopComp[1],loopComp[1]+loopComp[3]):
                    for j in range (loopComp[0],loopComp[0]+loopComp[2]):

                        if (( (i+1<h and image[i+1][j]<127) or (i-1>=0 and image[i-1][j]<127) or (j+1<w and image[i][j+1]<127) or (j-1>=0 and image[i][j-1]<127) or
                           (i+1<h and j+1<w and image[i+1][j+1]<127) or (i+1<h and j-1>=0 and image[i+1][j-1]<127) or (i-1>=0 and j+1<w and image[i-1][j+1]<127) or
						   (i-1>=0 and j-1>=0 and image[i-1][j-1]<127)) and image[i][j]>127):
                                bound+=1

                if(bound==0):bound=loopComp[3]+loopComp[2]


                avgformfac+=((4*math.pi*loopComp[4])/(bound*bound))
                avgroundess+=((bound*bound)/loopComp[4])
                avgblobsize+=loopComp[4]

        formFactor.append(avgformfac/len(components_loops))
        roundness.append(avgroundess/len(components_loops))
        blobSize.append(avgblobsize/len(components_loops))


        return [avg_dist[0],avg_w[0],std_w[0],median_w[0],formFactor[0],roundness[0],blobSize[0]]


    def extract_lines_features(self,img,pm):
        '''
        Calculates the mean feature vectors for an image
        :param img: cropped image
        :param pm: preprocessing module
        :return: mean feature vector of lines
        '''
        print("extracting features from lines")
        features = []
        lines = pm.get_lines(img)
        for line in lines:
            # print("Extracting fractal features")
            #f0,f1,f2 = self.fractal_features(line)
            #print("Applying gabor filter")
            #f3 = self.Gabor_filter_features(line)

            #f4 = self.edge_based_feature(line)
            # f6 = self.enclosed_regions_features(line)
            #f6 = self.connected_comp_feautres2(line)
            #f7 = self.white_space2(line)

            cf = contour_feautres()
            f8 = cf.contour_feautre_extract(line)
           # line_feature = np.concatenate(([f0,f1,f2]))
            features.append(f8) #line features

        return features













if __name__=="__main__":
    ft = feature_extractor()

    # pm = preprocessing_module.preprocessing_module()
    # image = cv2.imread("../../version1/data/original/a01-058.png",0)
    # # img = pm.line_segmentation(image)
    # # cv2.imshow('ll',img[2])
    # # cv2.waitKey(0)
    # img = pm.get_region(image)
    # img2 = pm.otsu_threshold(img)
    #
    # filters = ft.build_filters(5,181,5)
    # for filter in filters:
    #     print(filter)

    # im = cv2.imread('./a01-000u-00.png',0)
    # _,im =cv2.threshold(im,0,255,cv2.THRESH_OTSU)
    # f1= ft.Gabor_filter_features(im)
    #
    # print(f1)

    # #print(ft.enclosed_regions_features(im))
    #
    # im=np.array([[255,255,255,0,255,255,255],[255,0,0,0,0,0,255],[255,0,0,0,0,0,255],[255,255,0,0,0,255,255],[255,255,255,0,255,255,255]])
    # cnt = ft.find_contour_perimeter(im,3,2)
    # print(cnt)


    im = cv2.imread('./testimg.png',0)













