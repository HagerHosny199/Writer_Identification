import xml.etree.ElementTree as ET

import cv2
import os
import glob
import numpy as np
from skimage.io import imread
from matplotlib import pyplot as plt
from os import listdir




class IAM_loader(object):

    def __init__(self,data_path,parse_method="form"):
        self.data_images_path=os.path.join(data_path,'original')
        self.meta_data_xml_path=os.path.join(data_path,'meta','Xml')
        self.meta_data_forms_path=os.path.join(data_path,'meta','forms','forms.txt')
        self.data_path = data_path
        self.parse_method=parse_method
        self.output_path=os.path.join(data_path,'output2')
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)





    def split_data(self):
        '''
        Splits the data set into training, testing and validation sets
        :return: training data array,testing data array and validation data array
        '''
        authors_dict=self.read_forms_meta_data()
        training_data=[]
        test_data=[]
        validation_data=[]

        for author,documents in authors_dict.items():
            if len(documents) == 1:
                doc1,doc2 = self.split_document(documents[0])
                if doc1 is None:
                    continue
                training_data.append([doc1,author])
                if doc2 is not None:
                    test_data.append([doc2,author])
                    print('document '+documents[0]+' was splitted ')
                else:
                    print('Failed to split document %s.png into two regions'%documents[0])
            elif len(documents) >= 4:
                d0,d1,d2,d3 = self.preprocess(documents[0]),self.preprocess(documents[1]),self.preprocess(documents[2]),self.preprocess(documents[3])
                if d0 is not None:
                    print("Adding document %s to the training data"%documents[0])
                    training_data.append([d0,author])
                else:
                    print("Failed to add document %s.png to the training data"%documents[0])
                if d1 is not None:
                    print("Adding document %s to the test data"%documents[1])
                    test_data.append([d1,author])
                else:
                    print("Failed to add document %s.png to the test data"%documents[1])
                if d2 is not None:
                    print("Adding document %s to the validation data"%documents[2])
                    validation_data.append([d2,author])
                else:
                    print("Failed to add document %s.png to the validation data"%documents[2])
                if d3 is not None:
                    print("Adding document %s to the validation data"%documents[3])
                    validation_data.append([d3,author])
                else:
                    print("Failed to add document %s.png to the validation data"%documents[3])
            else:
                d0,d1 = self.preprocess(documents[0]),self.preprocess(documents[1])
                if d0 is not None:
                    print("Adding document %s to the training data"%documents[0])
                    training_data.append([d0,author])
                else:
                    print("Failed to add document %s.png to the training data"%documents[0])
                if d1 is not None:
                    print("Adding document %s to the test data"%documents[1])
                    test_data.append([d1,author])
                else:
                    print("Failed to add document %s.png to the test data"%documents[1])


        return training_data,test_data,validation_data



    def read_forms_meta_data(self):
        '''
        creates a dictionary of authors and the documents they have written
        It uses the ascii meta data in forms.txt to create that dictionary
        :return: {author:[documents]}
        '''

        authors_dict = dict()
        assert os.path.exists(self.meta_data_forms_path)
        with open(self.meta_data_forms_path, 'r') as file:
            for line in file:
                if line[0] != '#':
                    splitline = line.split(" ")
                    key = splitline[0]
                    author = splitline[1]
                    if author not in authors_dict:
                        authors_dict[author] = [key]
                    else:
                        authors_dict[author].append(key)


        return authors_dict


    def preprocess(self,img):
        try:
            image_path = os.path.join(self.data_images_path,img+'.png')
            xml_path = os.path.join(self.meta_data_xml_path,img+'.xml')
            assert os.path.exists(image_path),'{} is not a valid image path'.format(image_path)
            assert os.path.exists(xml_path),'{} is not a valid xml file path '.format(xml_path)
            image = cv2.imread(image_path,0) #reads the image in gray scale
            _,image= cv2.threshold(image,0,255,cv2.THRESH_OTSU) #convert the image to binary
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for item in root.iter(self.parse_method):
                bb = self.get_bounding_box(item)
                cropped_img = self.crop(image,bb)

        except:
            print("Failed to read image at image_path: "+image_path)

            return None
        return cropped_img

    def load_image(self,img):
        try:
            image_path = os.path.join(self.data_images_path,img+'.png')
            assert os.path.exists(image_path),'{} is not a valid image path'.format(image_path)
            image = cv2.imread(image_path,0) #reads the image in gray scale
            _,image= cv2.threshold(image,0,255,cv2.THRESH_OTSU) #convert the image to binary
        except:
            print("Failed to read image at image_path: "+image_path)
            return None
        return image

    def load_image_at_path(self,image_path,THRESHOLD=True):
        '''

        :param image_path: the path of an image
        :return: image after binary thresholding
        '''
        try:

            assert os.path.exists(image_path),'{} is not a valid image path'.format(image_path)
            image = cv2.imread(image_path,0) #reads the image in gray scale
            if THRESHOLD:
                _,image= cv2.threshold(image,0,255,cv2.THRESH_OTSU) #convert the image to binary

        except:
            print("Failed to read image at image_path: "+image_path)
            return None
        return image





    def get_bounding_box(self,item):
        words_meta = [ word_meta for word_meta in item.iter("cmp") ]
        x1 = np.min([int(word_meta.attrib['x']) for word_meta in words_meta])
        x2 = np.max([int(word_meta.attrib['x'])+int(word_meta.attrib['width']) for word_meta in words_meta])
        y1 = np.min([int(word_meta.attrib['y']) for word_meta in words_meta])
        y2 = np.max([int(word_meta.attrib['y'])+int(word_meta.attrib['height']) for word_meta in words_meta])
        return (x1,y1,x2,y2)


    def crop(self,img,bb):
        x1,y1,x2,y2 = bb


        return img[y1:y2,x1:x2]


    def process_data(self):
        xml_files=glob.glob(self.meta_data_xml_path+"/*.xml")
        print("processing data")
        for i,xml_file in enumerate(xml_files):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            height,width = int(root.attrib["height"]),int(root.attrib["width"])
            for item in root.iter(self.parse_method):
                for item2 in item.iter(self.parse_method):
                    list=[a for a in item2.iter("cmp")]


    def get_split_bb(self,lines):
        x1 = np.min([int(word.attrib['x']) for line in lines for word in line.iter("cmp")])
        x2 = np.max([int(word.attrib['x'])+int(word.attrib['width']) for line in lines for word in line.iter('cmp')])
        y1 = np.min([int(word.attrib['y']) for line in lines for word in line.iter("cmp")])
        y2 = np.max([int(word.attrib['y'])+int(word.attrib['height']) for line in lines for word in line.iter('cmp')])

        return (x1,y1,x2,y2)


    def split_document(self,img):
        image_path = os.path.join(self.data_images_path,img+'.png')
        xml_path = os.path.join(self.meta_data_xml_path,img+'.xml')
        assert os.path.exists(image_path),'{} is not a valid image path'.format(image_path)
        assert os.path.exists(xml_path),'{} is not a valid xml file path '.format(xml_path)
        image = cv2.imread(image_path,0)

        if image is None:
            print("Failed to read image %s.png"%img)
            return None,None

        _,image= cv2.threshold(image,0,255,cv2.THRESH_OTSU) #convert the image to binary
        tree = ET.parse(xml_path)
        item = tree.getroot()

        lines = [line for line in item.iter('line')]
        num_lines = len(lines)
        split_line = num_lines//2
        upper_part = lines[:split_line]
        lower_part = lines[split_line:]

        doc1 = self.crop(image,self.get_split_bb(upper_part))
        doc2 = self.crop(image,self.get_split_bb(lower_part)) if (lower_part != None) else None

        return doc1,doc2

    def segment_into_lines(self,img):
        lines = []
        xml_files=glob.glob(self.meta_data_xml_path+"/*.xml")
        print("processing data")
        for i,xml_file in enumerate(xml_files):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            height,width = int(root.attrib["height"]),int(root.attrib["width"])
            for item in root.iter("line"):
              words_meta = [a for a in item.iter("cmp")]
              x1 = np.min([int(word_meta.attrib['x']) for word_meta in words_meta])
              x2 = np.max([int(word_meta.attrib['x'])+int(word_meta.attrib['width']) for word_meta in words_meta])
              y1 = np.min([int(word_meta.attrib['y']) for word_meta in words_meta])
              y2 = np.max([int(word_meta.attrib['y'])+int(word_meta.attrib['height']) for word_meta in words_meta])
            roi = img[x1:x2,y1:y2]
            lines.append(roi)
        return lines


    def split_data_2(self):
        authors_dict=self.read_forms_meta_data()
        test_data = []
        validation_data = []
        training_data = []


        cnt = 0
        for author in sorted(authors_dict.items(),key=lambda a:len(a[1]),reverse=True):
            if cnt == 3:
                break
            print(len(author[1]))
            doc_num = min(len(author[1]),10)
            docs = author[1]
            if doc_num >= 3:
                d0 = self.load_image(docs[0])
                if d0 is not None:
                    test_data.append([d0,author[0]])
                d1 = self.load_image(docs[1])
                if d1 is not None:
                    validation_data.append([d1,author[0]])
                for i in range(2,doc_num):
                    d = self.load_image(docs[2])
                    if d is not None:
                        training_data.append([d,author[0]])

            else:
                pass
            cnt = cnt + 1
        return training_data,test_data,validation_data


    def split_data_3(self):
        authors_dict=self.read_forms_meta_data()
        test_data = []
        validation_data = []
        training_data = []

        f = open("list.txt",'w')

        cnt = 0
        for author in sorted(authors_dict.items(),key=lambda a:len(a[1]),reverse=True):
            if cnt == 3:
                break
            print(len(author[1]))
            doc_num = min(len(author[1]),10)
            docs = author[1]

            for i in range(doc_num):
                f.write("{}.png {}\n".format(docs[i],author[0]))

            cnt = cnt + 1


        f.close()

    def read_test_case(self,dirname):
        path = os.path.join(self.data_path,dirname)
        train_path = os.path.join(path,'train')
        test_files = glob.glob(os.path.join(path,'*.png'))

        test_image = self.load_image_at_path(test_files[0])
        test_label = (test_files[0].split('-')[2]).split('.')[0]
        training = []
        training_labels = []
        training_files = glob.glob(os.path.join(train_path,'*.png'))
        for train_file in training_files:
            train_image = self.load_image_at_path(train_file)
            label =(train_file.split('-')[2]).split('.')[0]
            training.append(train_image)
            training_labels.append(label)
        return test_image,test_label,training,training_labels




    def read_test_case_directory(self,path,threshold=True):
        print("loading test case in directory at path: {}".format(path))
        test_file = os.path.join(path,'test.PNG')
        test_image = self.load_image_at_path(test_file,threshold)
        training = []
        training_labels = []
        for i in range(1,4):
            train_path = os.path.join(path,str(i))
            print("loading image at path: {}".format(train_path))
            training_files = glob.glob(os.path.join(train_path,'*.PNG'))
            print("Training files : {}".format(training_files))
            for train_file in training_files:
                train_image = self.load_image_at_path(train_file,threshold)
                label = i
                training.append(train_image)
                training_labels.append(label)

        return test_image,training,training_labels



    def read_dirs(self,path):
        temp_dirs=os.listdir(path)
        filtered_dirs = [temp_dir for temp_dir in temp_dirs if temp_dir.isnumeric()]
        sorted_dirs = sorted(filtered_dirs,key= lambda dir_name:int(dir_name))
        dirs = (os.path.join(path,dirname) for dirname in sorted_dirs)

        return dirs






























if __name__=="__main__":

    IAM_loader=IAM_loader('../../version1/data')
    #
    # training_data,test_data,validation_data = IAM_loader.split_data()
    #
    #
    #
    # for i in range(100):
    #     path_output = os.path.join(IAM_loader.output_path,str(i)+'.png')
    #     cv2.imwrite(path_output,training_data[i][0])



    # image = cv2.imread("../../version1/data/original/g07-026a.png",0)
    # _,binary_threshold = cv2.threshold(image,0,255,cv2.THRESH_OTSU)
    #
    # edge_image_x_16 = cv2.Sobel(image,cv2.CV_16S,1,0,5)
    # edge_image_y_16 = cv2.Sobel(image,cv2.CV_16S,0,1,5)
    # edge_image_x_16 = np.abs(edge_image_x_16)
    # edge_image_y_16 = np.abs(edge_image_y_16)
    # edge_image_16 = cv2.addWeighted(edge_image_x_16,0.5,edge_image_y_16,0.5,0)
    # edge_image = np.uint8(edge_image_16)
    #
    # plt.subplot(1,2,1)
    # plt.title("binary")
    # plt.imshow(binary_threshold,cmap='gray')
    # plt.subplot(1,2,2)
    # plt.title("edges")
    # plt.imshow(edge_image,cmap='gray')
    # plt.show()
    #
    training_data,test_data,validation_data = IAM_loader.split_data_2()











