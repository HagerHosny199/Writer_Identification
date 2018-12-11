from IAM_loader import IAM_loader
import preprocessing_module
import feature_extraction
import classifier
from sklearn.preprocessing import normalize


test_cnt = 10
successes = 0


loader = IAM_loader('../version1/data')
pm = preprocessing_module.preprocessing_module()
ft = feature_extraction.feature_extractor()
cl = classifier.classifier()


def find_max_label(lines_labels):
    dict = {}
    for label in lines_labels:
        dict[label] = dict[label] + 1
    return sorted(dict.items(),key=lambda a : a[1],reverse=True)[0]


for i in range(test_cnt):
    print("testcase {}: ".format(i))
    dir_name = 'dir{}'.format(i)
    test,test_label,training_data,training_labels = loader.read_test_case(dir_name)
    print(test_label)
    print(training_labels)
    cropped_training_data = [pm.get_region(training_data[i]) for i in range(len(training_data))]
    feature_vectors = []
    for example in cropped_training_data:
        feature_vector = ft.extract_features_from_document(example)
        feature_vectors.append(feature_vector)


    test_feature = ft.extract_features_from_document(test)
    feature_vectors.append(test_feature)
    feature_vectors=normalize(feature_vectors,axis=0)
    test_feature = feature_vectors[-1]
    feature_vectors = feature_vectors[:-1]

    success,test_class = cl.minimum_distance(test_feature,test_label,feature_vectors,training_labels,ft)
    successes =(successes + 1 )if success else successes
    print(successes)
    # lines_labels=[]
    # test_lines = pm.get_lines(test)
    # for line in test_lines:
    #     line_feature = ft.extract_features_from_line(line)
    #     _,test_class = cl.minimum_distance(line_feature,test_label,feature_vectors,training_labels,ft)
    #     lines_labels.append(test_class)
    # max_label = find_max_label(lines_labels)
    # successes = (successes + 1) if max_label==test_label else successes




accuracy = (successes/test_cnt)*100
print(accuracy)
