from IAM_loader import IAM_loader
import preprocessing_module
import feature_extraction
import classifier
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier


test_cnt = 10
successes = 0


loader = IAM_loader('../version1/data')
pm = preprocessing_module.preprocessing_module()
ft = feature_extraction.feature_extractor()
cl = classifier.classifier()


def find_max_label(lines_labels):
    dict = {label:0 for label in lines_labels}
    print(dict)
    for label in lines_labels:
        dict[label] =  dict[label] + 1
    print(dict)
    return sorted(dict.items(),key=lambda a : a[1],reverse=True)[0][0]


for i in range(test_cnt):
    print("testcase {}: ".format(i))
    dir_name = 'dir{}'.format(i)
    test,test_label,training_data,training_labels = loader.read_test_case(dir_name)
    test = pm.get_cropped_image(test)
    print(test_label)
    print(training_labels)
    cropped_training_data = [pm.get_region(training_data[i]) for i in range(len(training_data))]
    feature_vectors = []
    labels = []
    j=0
    for example in cropped_training_data:
        feature_vector = ft.extract_lines_features(example,pm)
        feature_vectors.extend(feature_vector)
        labels.extend([training_labels[j] for _ in range(len(feature_vector))])
        j = j + 1


    test_feature = ft.extract_lines_features(test,pm)
    feature_vectors.extend(test_feature)
    feature_vectors=normalize(feature_vectors,axis=0)
    test_feature = feature_vectors[-len(test_feature):]
    feature_vectors = feature_vectors[:-len(test_feature)]
    print(len(test_feature))
    print(test_feature)
    print(len(feature_vectors))
    print(len(labels))
    print(labels)


    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(feature_vectors,labels)
    lines_labels = neigh.predict(test_feature)
    max_label = find_max_label(lines_labels)
    print("linelables={}".format(lines_labels))
    print(max_label)

    successes = (successes + 1) if max_label==test_label else successes
    msg = "test passed" if test_label==max_label else "failure : expected = {} actual = {}".format(test_label,max_label)
    print(msg)



accuracy = (successes/test_cnt)*100
print(accuracy)
