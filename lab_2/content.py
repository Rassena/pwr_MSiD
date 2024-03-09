# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 2: k-NN i Naive Bayes
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba, P. Dąbrowski
#  2019
# --------------------------------------------------------------------------

import numpy as np
import itertools
from collections import OrderedDict

def hamming_distance(X, X_train):
    """
    Zwróć odległość Hamminga dla obiektów ze zbioru *X* od obiektów z *X_train*.

    :param X: zbiór porównywanych obiektów N1xD
    :param X_train: zbiór obiektów do których porównujemy N2xD
    :return: macierz odległości pomiędzy obiektami z "X" i "X_train" N1xN2
    """
    x = X.toarray()
    x_train = X_train.toarray()
    a1 = np.array(x,np.int)
    a2 = np.array(x_train,np.int)
    ones_a1 = np.ones_like(a1)
    ones_a2 = np.ones_like(a2)
    sub1 = ones_a1 - a1
    sub2 = ones_a2 - a2
    res = sub1.dot(a2.T) + a1.dot(sub2.T)

    return res


def sort_train_labels_knn(Dist, y):
    """
    Posortuj etykiety klas danych treningowych *y* względem prawdopodobieństw
    zawartych w macierzy *Dist*.

    :param Dist: macierz odległości pomiędzy obiektami z "X" i "X_train" N1xN2
    :param y: wektor etykiet o długości N2
    :return: macierz etykiet klas posortowana względem wartości podobieństw
        odpowiadającego wiersza macierzy Dist N1xN2

    Do sortowania użyj algorytmu mergesort.
    """
    result = []
    for row in Dist:
        values_tup = list(zip(row, y))
        values_tup.sort(key=lambda x: x[0])
        values = list(map(lambda x: x[1], values_tup))
        result.append(values)
    return np.array(result)

def p_y_x_knn(y, k):
    """
    Wyznacz rozkład prawdopodobieństwa p(y|x) każdej z klas dla obiektów
    ze zbioru testowego wykorzystując klasyfikator KNN wyuczony na danych
    treningowych.

    :param y: macierz posortowanych etykiet dla danych treningowych N1xN2
    :param k: liczba najbliższych sasiadow dla KNN
    :return: macierz prawdopodobieństw p(y|x) dla obiektów z "X" N1xM
    """
    result = []
    for row in y:
        counts = [0, 0, 0, 0]
        probabilities = []
        for i in range(k):
            val = row[i]
            counts[val] += 1
        for i in counts:
            probabilities.append(i / k)
        result.append(probabilities)

    return np.array(result)


def classification_error(p_y_x, y_true):
    """
    Wyznacz błąd klasyfikacji.

    :param p_y_x: macierz przewidywanych prawdopodobieństw - każdy wiersz
        macierzy reprezentuje rozkład p(y|x) NxM
    :param y_true: zbiór rzeczywistych etykiet klas 1xN
    :return: błąd klasyfikacji
    """
    classifications = []
    for row in p_y_x:
        classifications.append(max(np.where(row == np.amax(row, axis=0)))[-1])
    differences = 0.0
    for pred, val in zip(classifications, y_true):
        if pred != val:
            differences += 1
    return differences / y_true.shape[0]


def model_selection_knn(X_val, X_train, y_val, y_train, k_values):
    """
    Wylicz bład dla różnych wartości *k*. Dokonaj selekcji modelu KNN
    wyznaczając najlepszą wartość *k*, tj. taką, dla której wartość błędu jest
    najniższa.

    :param X_val: zbiór danych walidacyjnych N1xD
    :param X_train: zbiór danych treningowych N2xD
    :param y_val: etykiety klas dla danych walidacyjnych 1xN1
    :param y_train: etykiety klas dla danych treningowych 1xN2
    :param k_values: wartości parametru k, które mają zostać sprawdzone
    :return: krotka (best_error, best_k, errors), gdzie "best_error" to
        najniższy osiągnięty błąd, "best_k" to "k" dla którego błąd był
        najniższy, a "errors" - lista wartości błędów dla kolejnych
        "k" z "k_values"
    """
    errors = []
    distances = hamming_distance(X_val, X_train)
    labels = sort_train_labels_knn(distances, y_train)
    for k in k_values:
        p_x_y = p_y_x_knn(labels, k)
        error = classification_error(p_x_y, y_val)
        errors.append((error, k))
    best = min(errors, key=lambda x: x[0])
    error_values = list(map(lambda x: x[0], errors))
    return best[0], best[1], error_values


def estimate_a_priori_nb(y_train):
    """
    Wyznacz rozkład a priori p(y) każdej z klas dla obiektów ze zbioru
    treningowego.

    :param y_train: etykiety dla danych treningowych 1xN
    :return: wektor prawdopodobieństw a priori p(y) 1xM
    """
    n = len(y_train)
    local_y = np.array(y_train)
    local_y.sort()
    vals = OrderedDict()
    for val in local_y:
        if val in vals:
            vals[val] += 1
        else:
            vals[val] = 1

    probabilities = []
    for key, val in vals.items():
        probabilities.append(val / n)
    return probabilities


def estimate_p_x_y_nb(X_train, y_train, a, b):
    """
    Wyznacz rozkład prawdopodobieństwa p(x|y) zakładając, że *x* przyjmuje
    wartości binarne i że elementy *x* są od siebie niezależne.

    :param X_train: dane treningowe NxD
    :param y_train: etykiety klas dla danych treningowych 1xN
    :param a: parametr "a" rozkładu Beta
    :param b: parametr "b" rozkładu Beta
    :return: macierz prawdopodobieństw p(x|y) dla obiektów z "X_train" MxD.
    """
    x_train = X_train.toarray()
    D = x_train.shape[1]
    y_mat = np.array([y_train] * D).transpose()
    tmp = x_train * (y_mat + 1)
    res = []

    for k in range(4):
        top = np.count_nonzero(tmp == k + 1, axis=0) + a - 1
        bottom = np.count_nonzero(y_train == k) + a + b - 2
        res.append(top / bottom)
    return np.array(res)


def p_y_x_nb(p_y, p_x_1_y, X):
    """
    Wyznacz rozkład prawdopodobieństwa p(y|x) dla każdej z klas z wykorzystaniem
    klasyfikatora Naiwnego Bayesa.

    :param p_y: wektor prawdopodobieństw a priori 1xM
    :param p_x_1_y: rozkład prawdopodobieństw p(x=1|y) MxD
    :param X: dane dla których beda wyznaczone prawdopodobieństwa, macierz NxD
    :return: macierz prawdopodobieństw p(y|x) dla obiektów z "X" NxM
    """
    x = X.toarray()
    total = []

    for o in x:
        products = []
        products_divided = []
        for k in range(4):
            product = p_y[k]
            for index, val in enumerate(o):
                if val:
                    product *= p_x_1_y[k][index]
                else:
                    product *= (1 - p_x_1_y[k][index])
            products.append(product)

        bottom = sum(products)

        for val in products:
            products_divided.append(val / bottom)
        total.append(products_divided)
    return np.array(total)


def model_selection_nb(X_train, X_val, y_train, y_val, a_values, b_values):
    """
    Wylicz bład dla różnych wartości *a* i *b*. Dokonaj selekcji modelu Naiwnego
    Byesa, wyznaczając najlepszą parę wartości *a* i *b*, tj. taką, dla której
    wartość błędu jest najniższa.
    
    :param X_train: zbiór danych treningowych N2xD
    :param X_val: zbiór danych walidacyjnych N1xD
    :param y_train: etykiety klas dla danych treningowych 1xN2
    :param y_val: etykiety klas dla danych walidacyjnych 1xN1
    :param a_values: lista parametrów "a" do sprawdzenia
    :param b_values: lista parametrów "b" do sprawdzenia
    :return: krotka (best_error, best_a, best_b, errors), gdzie "best_error" to
        najniższy osiągnięty błąd, "best_a" i "best_b" to para parametrów
        "a" i "b" dla której błąd był najniższy, a "errors" - lista wartości
        błędów dla wszystkich kombinacji wartości "a" i "b" (w kolejności
        iterowania najpierw po "a_values" [pętla zewnętrzna], a następnie
        "b_values" [pętla wewnętrzna]).
    """
    pi = estimate_a_priori_nb(y_train)
    errors = []
    for a, b in itertools.product(a_values, b_values):
        theta = estimate_p_x_y_nb(X_train, y_train, a, b)
        probabilities = p_y_x_nb(pi, theta, X_val)
        error = classification_error(probabilities, y_val)
        errors.append((error, a, b))
    min_error = min(errors, key=lambda x: x[0])
    min_err_val = min_error[0]
    min_a = min_error[1]
    min_b = min_error[2]
    err_list = list(map(lambda x: x[0], errors))
    return min_err_val, min_a, min_b, np.reshape(np.array(err_list), (len(a_values), len(b_values)))
