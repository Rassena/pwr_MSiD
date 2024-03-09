# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 3: Regresja logistyczna
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba, P. Dąbrowski
#  2019
# --------------------------------------------------------------------------

import numpy as np
from math import exp
from math import log
import itertools
from functools import partial


def sigmoid(x):
    """
    Wylicz wartość funkcji sigmoidalnej dla punktów *x*.

    :param x: wektor wartości *x* do zaaplikowania funkcji sigmoidalnej Nx1
    :return: wektor wartości funkcji sigmoidalnej dla wartości *x* Nx1
    """
    result = list(map(lambda k: [1 / (1 + exp(-k))], x))
    return np.array(result)


def logistic_cost_function(w, x_train, y_train):
    """
    Wylicz wartość funkcji logistycznej oraz jej gradient po parametrach.

    :param w: wektor parametrów modelu Mx1
    :param x_train: zbiór danych treningowych NxM
    :param y_train: etykiety klas dla danych treningowych Nx1
    :return: krotka (log, grad), gdzie *log* to wartość funkcji logistycznej,
        a *grad* jej gradient po parametrach *w* Mx1
    """
    ln_p = 0.0
    n = y_train.size
    sigArr = sigmoid(x_train @ w)
    vals = []
    for i, row in enumerate(x_train):
        sigma_n = sigmoid(w.T.dot(row))[0][0]
        #    print(sigma_n)
        tmp = y_train[i][0] * log(sigma_n) + (1 - y_train[i][0]) * log(1 - sigma_n)
        ln_p -= tmp
        vals.append(tmp)
    grad = x_train.transpose() @ (sigArr - y_train) / n
    return ln_p / n, grad


def gradient_descent(obj_fun, w0, epochs, eta):
    """
    Dokonaj *epochs* aktualizacji parametrów modelu metodą algorytmu gradientu
    prostego, korzystając z kroku uczenia *eta* i zaczynając od parametrów *w0*.
    Wylicz wartość funkcji celu *obj_fun* w każdej iteracji. Wyznacz wartość
    parametrów modelu w ostatniej epoce.

    :param obj_fun: optymalizowana funkcja celu, przyjmująca jako argument
        wektor parametrów *w* [wywołanie *val, grad = obj_fun(w)*]
    :param w0: początkowy wektor parametrów *w* Mx1
    :param epochs: liczba epok algorytmu gradientu prostego
    :param eta: krok uczenia
    :return: krotka (w, log_values), gdzie *w* to znaleziony optymalny
        punkt *w*, a *log_values* to lista wartości funkcji celu w każdej
        epoce (lista o długości *epochs*)
    """
    w = w0
    values = []
    for i in range(epochs):
        val, grad = obj_fun(w)
        values.append(val)
        w -= grad * eta
    val, _ = obj_fun(w)
    values.append(val)
    del values[0]
    return w, values


def stochastic_gradient_descent(obj_fun, x_train, y_train, w0, epochs, eta, mini_batch):
    """
    Dokonaj *epochs* aktualizacji parametrów modelu metodą stochastycznego
    algorytmu gradientu prostego, korzystając z kroku uczenia *eta*, paczek
    danych o rozmiarze *mini_batch* i zaczynając od parametrów *w0*. Wylicz
    wartość funkcji celu *obj_fun* w każdej iteracji. Wyznacz wartość parametrów
    modelu w ostatniej epoce.

    :param obj_fun: optymalizowana funkcja celu, przyjmująca jako argumenty
        wektor parametrów *w*, paczkę danych składających się z danych
        treningowych *x* i odpowiadających im etykiet *y*
        [wywołanie *val, grad = obj_fun(w, x, y)*]
    :param w0: początkowy wektor parametrów *w* Mx1
    :param epochs: liczba epok stochastycznego algorytmu gradientu prostego
    :param eta: krok uczenia
    :param mini_batch: rozmiar paczki danych / mini-batcha
    :return: krotka (w, log_values), gdzie *w* to znaleziony optymalny
        punkt *w*, a *log_values* to lista wartości funkcji celu dla całego
        zbioru treningowego w każdej epoce (lista o długości *epochs*)
    """
    size = len(x_train)
    batches_x = [x_train[i:i + mini_batch] for i in range(0, size, mini_batch)]
    batches_y = [y_train[i:i + mini_batch] for i in range(0, size, mini_batch)]
    w = np.copy(w0)
    vals = []
    for _ in range(epochs):
        for batch_x, batch_y in zip(batches_x, batches_y):
            val, grad = obj_fun(w, batch_x, batch_y)
            w -= eta * grad
        val, _ = obj_fun(w, x_train, y_train)
        vals.append(val)
    return w, vals


def regularized_logistic_cost_function(w, x_train, y_train, regularization_lambda):
    """
    Wylicz wartość funkcji logistycznej z regularyzacją l2 oraz jej gradient
    po parametrach.

    :param w: wektor parametrów modelu Mx1
    :param x_train: zbiór danych treningowych NxM
    :param y_train: etykiety klas dla danych treningowych Nx1
    :param regularization_lambda: parametr regularyzacji l2
    :return: krotka (log, grad), gdzie *log* to wartość funkcji logistycznej
        z regularyzacją l2, a *grad* jej gradient po parametrach *w* Mx1
    """
    ln_p = 0.0
    n = y_train.size
    sigArr = sigmoid(x_train @ w)
    w0 = np.copy(w)
    w0[0] = 0
    regularizer = w0 * regularization_lambda
    regularizer_val = regularization_lambda * np.linalg.norm(w0) ** 2 / 2
    for i, row in enumerate(x_train):
        sigma_n = sigmoid(w.T.dot(row))[0][0]
        tmp = y_train[i][0] * log(sigma_n) + (1 - y_train[i][0]) * log(1 - sigma_n)
        ln_p -= tmp / n
    ln_p += regularizer_val
    grad = x_train.transpose() @ (sigArr - y_train) / n + regularizer
    return ln_p, grad


def prediction(x, w, theta):
    """
    Wylicz wartości predykowanych etykiet dla obserwacji *x*, korzystając
    z modelu o parametrach *w* i progu klasyfikacji *theta*.

    :param x: macierz obserwacji NxM
    :param w: wektor parametrów modelu Mx1
    :param theta: próg klasyfikacji z przedziału [0,1]
    :return: wektor predykowanych etykiet ze zbioru {0, 1} Nx1
    """
    vals = x * w.T
    sums = list(map(lambda k: sum(k), vals))
    sigmoids = sigmoid(sums)
    etiquettes = list(map(lambda k: [int(k > theta)], sigmoids))
    return np.array(etiquettes)


def f_measure(y_true, y_pred):
    """
    Wylicz wartość miary F (F-measure) dla zadanych rzeczywistych etykiet
    *y_true* i odpowiadających im predykowanych etykiet *y_pred*.

    :param y_true: wektor rzeczywistych etykiet Nx1
    :param y_pred: wektor etykiet predykowanych przed model Nx1
    :return: wartość miary F (F-measure)
    """
    y_true_lst = y_true.T.tolist()[0]
    y_pred_lst = y_pred.T.tolist()[0]
    yt_yp = list(zip(y_true_lst, y_pred_lst))
    tp2 = 2 * sum(map(lambda x: x[0] and x[1], yt_yp))
    fp_fn = sum(map(lambda x: x[0] ^ x[1], yt_yp))
    return tp2 / (tp2 + fp_fn)


def model_selection(x_train, y_train, x_val, y_val, w0, epochs, eta, mini_batch, lambdas, thetas):
    """
    Policz wartość miary F dla wszystkich kombinacji wartości regularyzacji
    *lambda* i progu klasyfikacji *theta. Wyznacz parametry *w* dla modelu
    z regularyzacją l2, który najlepiej generalizuje dane, tj. daje najmniejszy
    błąd na ciągu walidacyjnym.

    :param x_train: zbiór danych treningowych NxM
    :param y_train: etykiety klas dla danych treningowych Nx1
    :param x_val: zbiór danych walidacyjnych NxM
    :param y_val: etykiety klas dla danych walidacyjnych Nx1
    :param w0: początkowy wektor parametrów *w* Mx1
    :param epochs: liczba epok stochastycznego algorytmu gradientu prostego
    :param eta: krok uczenia
    :param mini_batch: rozmiar paczki danych / mini-batcha
    :param lambdas: lista wartości parametru regularyzacji l2 *lambda*,
        które mają być sprawdzone
    :param thetas: lista wartości progów klasyfikacji *theta*,
        które mają być sprawdzone
    :return: krotka (regularization_lambda, theta, w, F), gdzie
        *regularization_lambda* to wartość regularyzacji *lambda* dla
        najlepszego modelu, *theta* to najlepszy próg klasyfikacji,
        *w* to parametry najlepszego modelu, a *F* to macierz wartości miary F
        dla wszystkich par *(lambda, theta)* #lambda x #theta
    """
    results = []
    fs = []
    for l in lambdas:
        fn_curried = partial(regularized_logistic_cost_function, regularization_lambda=l)
        w, _ = stochastic_gradient_descent(fn_curried, x_train, y_train, w0, epochs, eta, mini_batch)
        tmp = []
        for theta in thetas:
            y_pred = prediction(x_val, w, theta)
            f = f_measure(y_val, y_pred)
            results.append((l, theta, w, f))
            tmp.append(f)
        fs.append(tmp)
    best = max(results, key=lambda x: x[3])
    return best[0], best[1], best[2], fs
