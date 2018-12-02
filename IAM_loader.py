import xml.etree.ElementTree as ET

import cv2
import os
import glob
import numpy as np
from skimage.io import imread


class IAM_loader:

    def __init__(self,data_path,parse_method="form"):
        self.data_images_path=os.path.join(data_path,'original')
        self.meta_data_xml_path=os.path.join(data_path,'meta','Xml')
        self.meta_data_forms_path=os.path.join(data_path,'meta','forms','forms.txt')
        self.parse_method=parse_method
        self.output_path=os.path.join(data_path,'output')
        if not os.path.isdir(self.output_path):
            os.mkdir(self.output_path)





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
           # _,image= cv2.threshold(image,0,255,cv2.THRESH_BINARY_INV) #convert the image to binary
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for item in root.iter(self.parse_method):
                bb = self.get_bounding_box(item)
                cropped_img = self.crop(image,bb)
			
        except:
            print("Failed to read image at image_path: "+image_path)

            return None
        return cropped_img



    def get_bounding_box(self,item):
        words_meta = [ word_meta for word_meta in item.iter("cmp") ]
        x1 = np.min([int(word_meta.attrib['x']) for word_meta in words_meta])
        x2 = np.max([int(word_meta.attrib['x'])+int(word_meta.attrib['width']) for word_meta in words_meta])
        y1 = np.min([int(word_meta.attrib['y']) for word_meta in words_meta])
        y2 = np.max([int(word_meta.attrib['y'])+int(word_meta.attrib['height']) for word_meta in words_meta])
        return (x1,y1,x2,y2)


    def crop(self,img,bb):
	
#x1,y1,x2,y2 = bb
        return img#[y1:y2,x1:x2]
		


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






'''
if __name__=="__main__":

    IAM_loader=IAM_loader('../../version1/data')

    training_data,test_data,validation_data = IAM_loader.split_data()



    for i in range(100):
        path_output = os.path.join(IAM_loader.output_path,str(i)+'.png')
        cv2.imwrite(path_output,training_data[i][0])

'''

