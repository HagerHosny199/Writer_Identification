from IAM_loader import IAM_loader
import preprocessing_module
import feature_extraction
import classifier
from sklearn.preprocessing import normalize
import os
from os import listdir
import time
import cv2




loader = IAM_loader("../version1/data")
pm = preprocessing_module.preprocessing_module()
ft = feature_extraction.feature_extractor()
cl = classifier.classifier()

path = "./data"

dirs = loader.read_dirs(path)
results = open("results.txt",'w')
test_time = open("time.txt",'w')




def compare_and_find_accuracy():
    expected=open("expected_output.txt",'r')
    expected_output = [line.strip() for line in expected if line!="\n"]
    print(expected_output)
    expected.close()
    actual = open("results.txt",'r')
    actual_output = [line.strip() for line in actual if line!="\n"]
    print(actual_output)
    actual.close()
    res = sum([int(actual_output[i]==expected_output[i])for i in range(len(expected_output))])
    return (res/len(expected_output))*100



for dir in dirs:

    start_time = time.time()

    test,training_data,training_labels = loader.read_test_case_directory(dir,threshold=False)
    cropped_training_data = [pm.otsu_threshold(pm.get_cropped_image(training_data[i])) for i in range(len(training_data))]
    test = pm.otsu_threshold(pm.get_cropped_image(test))
    feature_vectors = []
    for example in cropped_training_data:
        feature_vector = ft.extract_features_from_lines(example,pm)
        feature_vectors.append(feature_vector)


    test_feature = ft.extract_features_from_lines(test,pm)
    feature_vectors.append(test_feature)
    print(feature_vectors)
    feature_vectors=normalize(feature_vectors,axis=0)
    test_feature = feature_vectors[-1]
    feature_vectors = feature_vectors[:-1]

    test_class = cl.minimum_distance_2(test_feature,feature_vectors,training_labels)


    end_time = time.time()

    iteration_time = end_time - start_time
    test_time.write("%.2f\n"%iteration_time)
    results.write(str(test_class)+"\n")



results.close()
test_time.close()

accuracy = compare_and_find_accuracy()
print("accuracy = {}%".format(accuracy))

