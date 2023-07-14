import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import my_function # function for to make marks

# function for to get data from file
def get_time_and_measuring(file_name: str, columns: int, lines: int):
    ne_data_set = open(file_name)
    LENGTH_OF_NE_DATA_SET = lines
    FEATURE_COUNT = columns
    measuring = np.array([[0] * FEATURE_COUNT] * LENGTH_OF_NE_DATA_SET)
    time_column = np.array([0] * LENGTH_OF_NE_DATA_SET, float)
    for i in range(LENGTH_OF_NE_DATA_SET):
        str1 = ne_data_set.readline().split(" ")
        measuring[i] = list(map(lambda x: int(x), str1))
        time_column[i] = i
    return (time_column, measuring)

time_columns, measuring_columns = get_time_and_measuring("new.dat",16,14741)
measuring_columns = np.array(measuring_columns[:,:-2])
time_columns = np.array(time_columns)
sum_columns = np.array([0]*measuring_columns.shape[0])
for i in range(measuring_columns.shape[0]):
    sum_columns[i] = sum(measuring_columns[i])

# data clean
measuring_columns_first_half = measuring_columns[2124:7607]
time_columns_first_half = time_columns[2124:7607]
sum_columns_first_half = sum_columns[2124:7607]

measuring_columns_second_half = measuring_columns[9300:13096]
time_columns_second_half = time_columns[9300:13096]
sum_columns_second_half = sum_columns[9300:13096]

# get marks and data fitting
time_columns_second_half = time_columns_second_half[:12740 - 9300]
second_target_features = my_function.get_fetures_second(measuring_columns_second_half, sum_columns_second_half)
second_target_features = second_target_features[:12740 - 9300]
measuring_columns_second_half = measuring_columns_second_half[:12740 - 9300]
sum_columns_second_half = sum_columns_second_half[:12740 - 9300]

time_columns_first_half = time_columns_first_half[293:np.shape(time_columns_first_half)[0]-1]
first_target_features = my_function.get_fetures_second(measuring_columns_first_half, sum_columns_first_half)
first_target_features = first_target_features[293:]
measuring_columns_first_half = measuring_columns_first_half[293:np.shape(measuring_columns_first_half)[0]-1]
sum_columns_first_half = sum_columns_first_half[293:np.shape(sum_columns_first_half)[0]-1]

def show_first_half():
    plt.title("first half")
    plt.xlabel("measurement number")
    plt.ylabel("pressure value")
    plt.plot(time_columns_first_half,sum_columns_first_half)
    plt.plot(time_columns_first_half, first_target_features * 500 - 4000)
    plt.plot(time_columns_first_half,measuring_columns_first_half)
    plt.show()

def show_second_half():
    plt.title("second half")
    plt.xlabel("measurement number")
    plt.ylabel("pressure value")
    plt.plot(time_columns_second_half, sum_columns_second_half)
    plt.plot(time_columns_second_half, second_target_features * 500 - 4000)
    plt.plot(time_columns_second_half, measuring_columns_second_half)
    plt.show()

def normaliz_array(array):
    array = np.array(array)
    array /=4095
    return array

def get_mistakes_fraction(y_pr1):
    ans = np.array(second_target_features) - np.array(y_pr1)
    counter=0
    for i in range(np.shape(np.array(second_target_features, float))[0]):
        if second_target_features[i]==y_pr1[i]:
            counter+=1
    print((np.shape(ans)[0]-counter)/np.shape(ans)[0])
    print((np.shape(ans)[0]-counter))
    print(np.shape(ans)[0])
    return (np.shape(ans)[0]-counter)/np.shape(ans)[0]

def logistic_regression_classification():
    import sklearn.linear_model as lm
    clf = lm.LogisticRegression(random_state=0, solver='saga')
    clf.fit(normaliz_array(np.array(measuring_columns_first_half, float)), first_target_features)
    y_pr1 = clf.predict(normaliz_array(np.array(measuring_columns_second_half, float)))
    print(get_mistakes_fraction(y_pr1))
    plt.plot(time_columns_second_half, second_target_features * 500 - 4000)
    plt.plot(time_columns_second_half, y_pr1 * 500 - 4000)
    plt.plot(time_columns_second_half, measuring_columns_second_half + 3000)
    plt.show()
def knn_classification():
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=90, weights='distance')
    clf.fit(normaliz_array(np.array(measuring_columns_first_half, float)), first_target_features)
    y_pr1 = clf.predict(normaliz_array(np.array(measuring_columns_second_half, float)))
    print(get_mistakes_fraction(y_pr1))
    plt.plot(time_columns_second_half, second_target_features * 500 - 4000)
    plt.plot(time_columns_second_half, y_pr1 * 500 - 4000)
    plt.plot(time_columns_second_half, measuring_columns_second_half + 3000)
    plt.show()
def clf_classification():
    clf = svm.SVC(kernel='poly',degree=4)
    clf.fit(normaliz_array(np.array(measuring_columns_first_half,float)), first_target_features)
    y_pr1 = clf.predict(normaliz_array(np.array(measuring_columns_second_half, float)))
    print(get_mistakes_fraction(y_pr1))
    plt.plot(time_columns_second_half, second_target_features*500-4000)
    plt.plot(time_columns_second_half, y_pr1*500-4000)
    plt.plot(time_columns_second_half, measuring_columns_second_half + 3000)
    plt.show()