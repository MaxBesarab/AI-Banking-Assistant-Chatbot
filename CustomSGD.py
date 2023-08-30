import numpy as np
import matplotlib.pyplot as plt
import copy
import time

if __name__ == '__main__':
    np.random.seed(1)
    lmd = 0.0001  # Parametr regularyzacji lambda
    lr = 0.01  # Współczynnik nauki
    num_iter = 200  # Liczba iteracji treningowych
    rho = 0.90  # Parametr momentum

    # Wczytanie danych treningowych
    trainX = np.load('train_x_test.npy')
    trainY = np.load('train_y_test.npy')

    print('trainX shape:', trainX.shape)
    print('trainY shape:', trainY.shape)

    num_features = trainX.shape[1]  # Liczba cech w danych treningowych
    W = np.random.normal(loc=0.0, scale=1.0, size=[trainX.shape[1], 1])  # Inicjalizacja macierzy wag W
    W_old = copy.deepcopy(W)  # Skopiowanie W do obliczeń momentum

    train_losses_sgdn = []  # Lista do przechowywania wartości straty treningowej
    i = 0  # Licznik do obliczania malejącego współczynnika nauki

    for iter in range(num_iter):
        train_loss_sum = 0  # Suma straty treningowej dla danej iteracji
        batch_count = 0  # Licznik liczby instancji przetworzonych w danej iteracji

        for x, y in zip(trainX, trainY):
            i += 1
            lr = 0.001 / i  # Malejący współczynnik nauki wraz z przetwarzanymi instancjami
            momentum = (1 + rho) * W - rho * W_old  # Obliczenie momentum na podstawie poprzednich i aktualnych wag
            n_exp = np.exp(-y * np.matmul(x, momentum))  # Wyrażenie wykładnicze używane do obliczeń gradientowych
            exp = np.exp(-y * np.matmul(x, W))  # Wyrażenie wykładnicze używane do obliczeń straty
            train_grad = -(np.expand_dims(x, axis=1).dot(np.expand_dims(y * n_exp / (1 + n_exp), axis=0))) / \
                         trainX.shape[0] + lmd * momentum  # Obliczenie gradientu z uwzględnieniem regularyzacji

            W_old = copy.deepcopy(W)  # Aktualizacja poprzednich wag przed kolejną iteracją
            W = momentum - lr * train_grad  # Aktualizacja wag na podstawie momentum i współczynnika nauki
            train_loss = np.mean(np.log(1 + exp), axis=0) + (lmd * (np.sum(W ** 2)) / 2)  # Obliczenie straty treningowej
            train_loss_sum += train_loss
            batch_count += 1

        train_losses_sgdn.append(
            train_loss_sum / batch_count)  # Obliczenie i zapisanie średniej straty treningowej dla danej iteracji
        print(iter, train_loss_sum / batch_count)

    # Wykres straty treningowej
    plt.plot(np.log(train_losses_sgdn))
    plt.xlabel('Numer epoki')
    plt.ylabel('Strata/Dokładność')
    plt.title('Wartość funkcji straty i dokładności treningowej')
    plt.show()
