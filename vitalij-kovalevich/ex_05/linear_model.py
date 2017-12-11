import numpy as np

class LogisticRegression():
    '''Логистическая регрессия
    Параметры (все необязательные)
        alpha     : float, скорость обучения
        epsilon   : float, точность
        max_iter  : int, максимальное число итераций спуск
        normalize : boolean, если True, то нормализовывать
                    данные, False подразумевает, что данные
                    уже нормализованы 
    '''

    def __init__(self, alpha=0.001, epsilon=0.000001, max_iter=100000, normalize=True):
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.normalize = normalize

    @staticmethod
    def normalization(x):
        '''Метод выполняет стандартизацию данных 
        по формуле (x - среднее)/(макс-мин)
        это отличается от классического Z-score
        В классическом Z-score используется стандартное 
        отклонение = sqrt(mean(|x - x.mean()|^2)).
        Чтобы вычислить реальный std -> x.std(axis=0) 
        Аргументы
            x  :  ndarray, массив признаков
        '''

        '''Замените код на свой, для того, чтобы функция верно 
            выполняла нормализацию 
            для проверки: массив [[1,2,3],
                                  [4,5,6],
                                  [7,8,9]]
            после нормализации будет равен = [[-0.5, -0.5, -0.5],
                                              [ 0. ,  0. ,  0. ],
                                              [ 0.5,  0.5,  0.5]])
        '''
        x_min = np.min(x, axis=0)
        x_max = np.max(x, axis=0)
        x_nx = x_max - x_min
        x_sr = np.mean(x, axis = 0)
        x_norm = (x-x_sr)/x_nx
        return x_norm

    def __weight_init(self, n_features):
        '''Приватный метод инициализации весов
           
           Аргументы
                self       : объект класса LinearRegression
                n_features : int, число признаков, кол-во
                             столбцов X
        '''
        # TODO исправьте код так, чтобы
        # self.W был равен вектору НУЛЕЙ
        self.W = np.zeros(n_features)

    def predict(self, x):
        '''Метод выполняющий прогноз, согласно гипотезе
           hw(x) = w0+w1*x1+w2*x2
           X0 = 1, это значит, что в данных уже должен
                   быть первый единичный столбец
           Аргументы
                self  : объект класса LinearRegression
                X     : ndarray, исходный массив
        '''
        if not hasattr(self, 'W'):
            self.__weight_init(x.shape[1])
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        # TODO нужно исправить return, чтобы он возвращал результат 
        # матричного перемножения x и параметров модели
        # TODO (new) не забудьте применить sigmoid к результату 
        return sigmoid(self.W.dot(x.T))

    
    def cost(self, X, y):
        '''
        ### TODO (new) напишите функцию стоимости log_loss
        sum(-y * log(y_pred) - (1-y)*log(1- y_pred))/len(X)
        
        Это самая важная часть задания, смотрите внимательно на формулу
        и проверяйте какой размерности получается результат 
        '''
        return (-y.dot(np.log(self.predict(X)))-(1-y).dot(np.log(1 - self.predict(X))))/len(X)
                   
    def __gradient(self, x, y):
        ''' TODO Метод, вычисляющий градиенты на каждом шагу
        grad = 1/m*(sum(h(x)-y)*x)
        '''
        # допишите код для вычисления градиентов 
        # по каждому параметру 

        grad =(self.predict(x) - y).dot(x)/len(y)
        return grad

    def fit(self, X, y, verbose=True):
        '''Метод выполняющий обучение, подбор параметров 
           гипотезы. Реализует алгоритм градиентного спуска
           пока iter < max_iter или cost[i] - cost[i-1] < epsilon:
                1) найти градиенты
                2) обновить все веса
           Аргументы
                X       : ndarray, исходный массив признаков
                y       : ndarray, целевая переменная, target
                verbose : bool, если True, то выводит логи обучения
        '''
        if self.normalize:
            # Нормализация данных
            X_norm = self.normalization(X)
        # инициализация весов
        self.__weight_init(X.shape[1])
        # градиентный спуск
        prev_cost = self.cost(X, y) # начальное значение ошибки
        if verbose:
            print('Начальное значение ошибки: {}'.format(prev_cost))

        i_epoch   = 0
        while i_epoch < self.max_iter:
            # вычисление градиента
            grad   = self.__gradient(X, y)
            # обновление весов
            self.W = self.W - self.alpha*grad
            # вычисление новой ошибки
            curr_cost = self.cost(X, y)
            if verbose:
                print('Итерация {:05d}: Ошибка={}'.format(i_epoch, curr_cost))
            # проверка условия сходимости
            if abs(prev_cost-curr_cost) < self.epsilon:
                break
            else:
                prev_cost = curr_cost
            i_epoch += 1


