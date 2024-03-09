# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 1: Regresja liniowa
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import numpy as np

from utils import polynomial


def mean_squared_error(x, y, w):
    """
    :param x: ciąg wejściowy Nx1
    :param y: ciąg wyjsciowy Nx1
    :param w: parametry modelu (M+1)x1
    :return: błąd średniokwadratowy pomiędzy wyjściami y oraz wyjściami
     uzyskanymi z wielowamiu o parametrach w dla wejść x
    """
    sum = 0
    for n in range(np.shape(x)[0]):
        y2 = 0
        for m in range(np.shape(w)[0]): y2 += w[m] * (x[n] ** m)
        sum += (y[n] - y2) ** 2
    return np.sum(sum / (np.shape(x)[0]), axis=0)


def design_matrix(x_train, M):
    """
    :param x_train: ciąg treningowy Nx1
    :param M: stopień wielomianu 0,1,2,...
    :return: funkcja wylicza Design Matrix Nx(M+1) dla wielomianu rzędu M
    """
    result = np.zeros(shape=(np.shape(x_train)[0], M + 1))
    for n in range(result.shape[0]):
        for m in range(M + 1):
            result[n][m] = x_train[n] ** m
    return result


def least_squares(x_train, y_train, M):
    """
    :param x_train: ciąg treningowy wejśćia Nx1
    :param y_train: ciąg treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :return: funkcja zwraca krotkę (w,err), gdzie w są parametrami dopasowanego 
    wielomianu, a err to błąd średniokwadratowy dopasowania
    """
    designMatrix = np.array(design_matrix(x_train, M));
    designMatrixTransposed = np.transpose(designMatrix)

    result = np.linalg.inv(designMatrixTransposed.dot(designMatrix))
    result = result.dot(designMatrixTransposed)
    result = np.array(result).dot(y_train)

    return result, mean_squared_error(x_train, y_train, result)


def regularized_least_squares(x_train, y_train, M, regularization_lambda):
    """
    :param x_train: ciąg treningowy wejśćia Nx1
    :param y_train: ciąg treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :param regularization_lambda: parametr regularyzacji
    :return: funkcja zwraca krotkę (w,err), gdzie w są parametrami dopasowanego
    wielomianu zgodnie z kryterium z regularyzacją l2, a err to błąd 
    średniokwadratowy dopasowania
    """
    designMatrix = np.array(design_matrix(x_train, M));
    designMatrixTransposed = np.transpose(designMatrix)

    result = designMatrixTransposed.dot(designMatrix) + regularization_lambda * np.eye(designMatrixTransposed.dot(designMatrix).shape[0])
    result = np.linalg.inv(result).dot(designMatrixTransposed)
    result = np.array(result).dot(y_train)

    return result, mean_squared_error(x_train, y_train, result)


def model_selection(x_train, y_train, x_val, y_val, M_values):
    """
    :param x_train: ciąg treningowy wejśćia Nx1
    :param y_train: ciąg treningowy wyjscia Nx1
    :param x_val: ciąg walidacyjny wejśćia Nx1
    :param y_val: ciąg walidacyjny wyjscia Nx1
    :param M_values: tablica stopni wielomianu, które mają byc sprawdzone
    :return: funkcja zwraca krotkę (w,train_err,val_err), gdzie w są parametrami
    modelu, ktory najlepiej generalizuje dane, tj. daje najmniejszy błąd na 
    ciągu walidacyjnym, train_err i val_err to błędy na sredniokwadratowe na 
    ciągach treningowym i walidacyjnym
    """
    result = (0, 0, 0);
    current = least_squares(x_train, y_train, M_values[0]);
    best = mean_squared_error(x_val, y_val, current[0])
    for m in range(len(M_values)):
        current = least_squares(x_train, y_train, M_values[m])
        err = mean_squared_error(x_val, y_val, current[0])
        if best > err: best = err; result = (current[0], current[1], best)
    return result


def regularized_model_selection(x_train, y_train, x_val, y_val, M, lambda_values):
    """
    :param x_train: ciąg treningowy wejśćia Nx1
    :param y_train: ciąg treningowy wyjscia Nx1
    :param x_val: ciąg walidacyjny wejśćia Nx1
    :param y_val: ciąg walidacyjny wyjscia Nx1
    :param M: stopień wielomianu
    :param lambda_values: lista z wartościami różnych parametrów regularyzacji
    :return: funkcja zwraca krotkę (w,train_err,val_err,regularization_lambda),
    gdzie w są parametrami modelu, ktory najlepiej generalizuje dane, tj. daje
    najmniejszy błąd na ciągu walidacyjnym. Wielomian dopasowany jest wg
    kryterium z regularyzacją. train_err i val_err to błędy średniokwadratowe
    na ciągach treningowym i walidacyjnym. regularization_lambda to najlepsza
    wartość parametru regularyzacji
    """
    result = (0, 0, 0);
    current = regularized_least_squares(x_train, y_train, M, lambda_values[0]);
    best = mean_squared_error(x_val, y_val, current[0])
    for l in range(len(lambda_values)):
        current = regularized_least_squares(x_train, y_train, M, lambda_values[l])
        err = mean_squared_error(x_val, y_val, current[0])
        if best > err: best = err; result = (current[0], current[1], best, lambda_values[l])
    return result
