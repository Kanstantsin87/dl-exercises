import random
import operator
import sys

class MatrixError(BaseException):
    """ Êëàññ èñêëþ÷åíèÿ äëÿ ìàòðèö """
    pass

class Matrix(object):
    """Ïðîñòîé êëàññ ìàòðèöû â Python
    Îñíîâíûå îïåðàöèè ëèíåéíîé àëãåáðû ðåàëèçîâàíû
    ïóòåì ïåðåãðóçêè îïåðàòîðîâ 
    """
    def __init__(self, n, m, init=True):
        """Êîíñòðóêòîð
        #Àðãóìåíòû:
            n    :  int, ÷èñëî ñòðîê
            m    :  int, ÷èñëî ñòîëáöîâ 
            init :  (íåîáÿçàòåëüíûé ïàðàìåòð), ëîãè÷åñêèé.
                    åñëè False, òî ñîçäàåòñÿ ïóñòîé ìàññèâ
        """
        if init:
            # ñîçäàåì ìàññèâ íóëåé
            self.array = [[0]*m for x in range(n)]
        else:
            self.array = []

        self.n = n
        self.m = m

    def __getitem__(self, idx):
        """Ïåðåãðóçêà îïåðàòîðà ïîëó÷åíèÿ ýëåìåíòà ìàññèâà 
        """
        # ïðîâåðÿåì, åñëè èíäåêñ - ýòî ñïèñîê èíäåêñîâ
        if isinstance(idx, tuple): 
            if len(idx) == 2: 
                return self.array[idx[0]][idx[1]]
            else:
                # ó ìàòðèöû åñòü òîëüêî ñòðîêè è ñòîëáöû
                raise MatrixError("Matrix has only two shapes!")
        else:
            return self.array[idx]

    def __setitem__(self, idx, item):
        """Ïåðåãðóçêà îïåðàòîðà ïðèñâàèâàíèÿ 
        """
        # ïðîâåðÿåì, åñëè èíäåêñ - ýòî ñïèñîê èíäåêñîâ
        if isinstance(idx, tuple):
            if len(idx) == 2: 
                self.array[idx[0]][idx[1]] = item
            else:
                # ó ìàòðèöû åñòü òîëüêî ñòðîêè è ñòîëáöû
                raise MatrixError("Matrix has only two shapes!")
        else:
            self.array[idx] = item

    def __str__(self):
        """Ïåðåîïðåäåëÿåì ìåòîä âûâîäà ìàòðèöû â êîíñîëü
        """
        s='\n'.join([' '.join([str(item) for item in row]) for row in self.array])
        return s + '\n'

    def getRank(self):
        """Ïîëó÷èòü ÷èñëî ñòðîê è ñòîëáöîâ
        """
        return (self.n, self.m)


    def __eq__(self, mat):
        """ Ïðîâåðêà íà ðàâåíñòâî """

        return (mat.array == self.array)

    def transpose(self):
        """ Òðàíñïîíèðîâàííîå ïðåäñòàâëåíèå ìàòðèöû 
    
            Çäåñü íóæíî âïèñàòü ñâîé êîä äëÿ ñîçäàíèÿ 
            ôóíêöèè, âîçâðàùàþùåé òðàíñïîíèðîâàííóþ
            ìàòðèöó
            Aij = Aji
        """
        transmat = Matrix(self.m, self.n)
        for i in range(self.n):
            for j in range(self.m):
                transmat[j][i] = self[i][j]
        return transmat

    def __add__(self, mat):
        """ Ïåðåîïðåäåëåíèå îïåðàöèè ñëîæåíèÿ "+"
        äëÿ ìàòðèö
        """
        if self.getRank() != mat.getRank():
            raise MatrixError("Trying to add matrixes of varying rank!")

        '''
        Ñþäà âïèñàòü êîä ôóíêöèè, âûïîëíÿþùåé
        îïåðàöèþ ñëîæåíèÿ
        Cij = Aij+Bij
        '''
        for i in range(self.n):
                for j in range(self.m):
                    self[i][j] = self[i][j]+mat[i][j]
        
        return self
        

    def __sub__(self, mat):
        """ Ïåðåîïðåäåëåíèå îïåðàöèè âû÷èòàíèÿ "-"
        äëÿ ìàòðèö
        """
        if self.getRank() != mat.getRank():
            raise MatrixError("Trying to add matrixes of varying rank!")

        '''
        Ñþäà âïèñàòü êîä ôóíêöèè, âûïîëíÿþùåé
        îïåðàöèþ âû÷èòàíèÿ
        Cij = Aij-Bij
        '''
        for i in range(self.n):
                for j in range(self.m):
                    self[i][j] = self.array[i][j]-mat[i][j]
        return self

    def __mul__(self, mat):
        """Ïðîèçâåäåíèå Àäàìàðà èëè ïîòî÷å÷íîå óìíîæåíèå"""
        mulmat = Matrix(self.n, self.m) # ðåçóëüòèðóþùàÿ ìàòðèöà

        # åñëè âòîðîé àðãóìåíò - ÷èñëî, òî 
        # ïðîñòî óìíîæèòü êàæäûé ýëåìåíò íà ýòî ÷èñëî
        if isinstance(mat, int) or isinstance(mat, float):
            for i in range(self.n):
                for j in range(self.m):
                    mulmat[i][j] = self.array[i][j]*mat
            return mulmat
        else:
            # äëÿ ïîòî÷å÷íîãî ïåðåìíîæåíèÿ ìàòðèö  
            # èõ ðàçìåðíîñòè äîëæíû áûòü îäèíàêîâûìè
            if (self.n != mat.n or self.m != mat.m):
                raise MatrixError("Matrices cannot be multipled!")
                
            for i in range(self.n):
                for j in range(self.m):
                    mulmat[i][j] = self.array[i][j]*mat[i][j]
            return mulmat

    def dot(self, mat):
        """ Ìàòðè÷íîå óìíîæåíèå """
        
        matn, matm = mat.getRank()
        # äëÿ ïåðåìíîæåíèÿ ìàòðèö ÷èñëî ñòîëáöîâ îäíîé 
        # äîëæíî ðàâíÿòüñÿ ÷èñëó ñòðîê â äðóãîé
        if (self.m != matn):
            raise MatrixError("Matrices cannot be multipled!")
        '''
        Çäåñü íàïèñàòü êîä, âûïîëíÿþùèé 
        ìàòðè÷íîå ïåðåìíîæåíèå è âîçâðàùàþøèé
        íîâóþ ìàòðèöó
        Cij = sum(Aik*Bkj)
        '''
        dotmat = Matrix(self.n, matm)
        for i in range(self.n):
            for j in range(mat.m):
                for k in range(self.m):
                    dotmat[i][j] = dotmat[i][j] + self.array[i][k]*mat.array[k][j]
        return dotmat

    @classmethod
    def _makeMatrix(cls, array):
        """Ïåðåîïðåäåëåíèå êîíñòðóêòîðà
        """
        n = len(array)
        m = len(array[0])
        # Validity check
        if any([len(row) != m for row in array[1:]]):
            raise MatrixError("inconsistent row length")
        mat = Matrix(n,m, init=False)
        mat.array = array

        return mat

    @classmethod
    def fromList(cls, listoflists):
        """ Ñîçäàíèå ìàòðèöû íàïðÿìóþ èç ñïèñêà """

        # E.g: Matrix.fromList([[1 2 3], [4,5,6], [7,8,9]])

        array = listoflists[:]
        return cls._makeMatrix(array)

    @classmethod
    def makeId(cls, n):
        """ Ñîçäàòü åäèíè÷íóþ ìàòðèöó ðàçìåðà (nxn) """

        array = [[0]*n for x in range(n)]
        idx = 0
        for row in array:
            row[idx] = 1
            idx += 1

        return cls.fromList(array)


if __name__ == "__main__":
    a = Matrix.fromList([[1,2], [3, 4]])
    print(a)
    print(a*2)
    print('Identity:')
    print(Matrix.makeId(3))
    print(a*a)
    pass