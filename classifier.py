from scipy.spatial import distance

class classifier(object):

 def minimum_distance(self,test_feature,test_label,feature_vectors,training_labels,ft):


    #get features of test vector
    v= test_feature

    min_dist = 1000
    min_i=1
    for i,j in enumerate(feature_vectors):
        dist = distance.euclidean(j,v)
        print(dist)
        if dist<min_dist:
            min_dist = dist
            min_i = i

    test_class = training_labels[min_i]
    success = test_class == test_label
    if(success):
        print("Test passed")
    else:
        print("Failure: expected = {} actual = {} ".format(test_label,test_class))
    return success,test_class
