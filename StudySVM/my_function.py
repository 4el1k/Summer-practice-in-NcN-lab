import numpy as np

def set_markup(array):
    array = np.array(array)
    array = array[1:]  # первая строка в переданном массиве нулевая, ее удаляем
    array_row = array.shape[0]
    k = array_row // 5
    features = np.array([0] * array_row)
    for i in range(2):
        for j in range(k):
            features[i * k + j] = i+1
    for i in range(2, 4):
        for j in range(k):
            features[i * k + j] = 3
    for j in range(4 * k, array_row):
        features[j] = 4
    return features

def set_markup_empty_phase(array):
    array = np.array(array)
    array = array[1:]  # первая строка в переданном массиве нулевая, ее удаляем
    array_row = array.shape[0]
    features = np.array([0] * array_row)
    return features


def get_fetures_second(measuring_, summ):
    measuring_ = np.array(measuring_)
    summ = np.array(summ)
    length = measuring_.shape[0]
    current_measuring = np.array([0] * measuring_.shape[1])
    current_measuring_empty = np.array([0] * measuring_.shape[1])
    features1 = np.array([0])
    flag1 = False
    flag2 = False
    flag3 = False
    flag4 = False
    upper_bound = 28000
    for i in range(length-1):
        if (not flag1):  # начали новый шаг
            bound = summ[i]
            flag1 = True
        if (summ[i] < upper_bound and not flag2): # идем до upper_bound
            current_measuring = np.vstack((current_measuring, measuring_[i]))
        else:
            flag2 = True  #
            if (summ[i] - bound > 1700 and not flag3): # проходим вершину и идем вниз, пока summ[i] - bound > 1700
                current_measuring = np.vstack((current_measuring, measuring_[i]))
            else:
                flag3 = True  #
                if (not (summ[i + 1] - summ[i] > 150) and not flag4): # пока не начали подниматься идем вниз
                    current_measuring = np.vstack((current_measuring, measuring_[i]))

                else:
                    flag4 = True

                    # все, зоны CV1-CV5 пройдены, теперь начинаем идти по зоне CV0
                    if (not (summ[i] - summ[i - 1] >= 800)): # пока не начался сильный подъем идем по зоне CV0
                        current_measuring_empty = np.vstack((current_measuring_empty, measuring_[i]))
                    else: # начался сильный подъем, значит пора начинать записывать новый шаг
                        current_measuring_empty = np.vstack((current_measuring_empty, measuring_[i]))
                        features1 = np.concatenate((features1, set_markup(current_measuring)))  # создаем метки CV1-CV5
                        features1 = np.concatenate((features1, set_markup_empty_phase(current_measuring_empty))) # создаем метку CV0
                        flag1 = False
                        flag2 = False
                        flag3 = False
                        flag4 = False
                        current_measuring = np.array([0] * measuring_.shape[1])
                        current_measuring_empty = np.array([0] * measuring_.shape[1])

    features1 = np.concatenate((features1, set_markup(current_measuring)))
    features1 = np.concatenate((features1, set_markup_empty_phase(current_measuring_empty)))
    features1 = np.delete(features1, 0, axis=0)
    return features1
