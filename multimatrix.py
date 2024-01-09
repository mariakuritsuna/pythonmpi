import mpi4py
from mpi4py import MPI
from random import randint
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
workers = comm.Get_size() - 1

mtrx1 = []
mtrx2 = []
mtrx3 = []

N = 1000

def init():

    global X 
    X = [[randint(0, 9) for i in range(N)] for j in range(N)]

    global Y 
    Y = [[randint(0, 9) for i in range(N)] for j in range(N)]

def matrix():

    Z = [[sum(a * b for a, b in zip(X_row, Y_col)) for Y_col in zip(*Y)]
            for X_row in X]

    return Z

def distribute_matrix_data():

    def split_matrix(seq, p):

        row = []
        n = len(seq) / p
        r = len(seq) % p
        b, e = 0, n + min(1, r)
        for i in range(p):
            row.append(seq[b:e])
            r = max(0, r - 1)
            b, e = e, e + n + min(1, r)

        return row

    rows = split_matrix(X, workers)

    pid = 1
    for rows in rows:
        comm.send(rows, dest=pid, tag=1)
        comm.send(Y, dest=pid, tag=2)
        pid = pid + 1


def assemble_matrix_data():

    global Z

    pid = 1
    for n in range(workers):
        row = comm.recv(source=pid, tag=pid)
        Z = Z + row
        pid = pid + 1


def master_operation():

    distribute_matrix_data()
    assemble_matrix_data()


def slave_operation():
    x = comm.recv(source=0, tag=1)
    y = comm.recv(source=0, tag=2)
    z = matrix(x, y)
    comm.send(z, dest=0, tag=rank)

if __name__ == '__main__':

    if rank == 0:

        init()
        # start time
        t1 = time.time()
        print('--------------------------------------------------------------')
        print('Start time', t1)
        print('--------------------------------------------------------------')
        print('\n')

        master_operation()

        # end time
        t2 = time.time()

        print('--------------------------------------------------------------')
        print(mtrx1)
        print('--------------------------------------------------------------')
        print('\n')

        print('--------------------------------------------------------------')
        print(mtrx2)
        print('--------------------------------------------------------------')
        print('\n')

        print('--------------------------------------------------------------')
        print(mtrx3)
        print('--------------------------------------------------------------')
        print('\n')

        print('--------------------------------------------------------------')
        print('Start time', t1)
        print('--------------------------------------------------------------')
        print('\n')

        print('--------------------------------------------------------------')
        print('End time', t1)
        print('--------------------------------------------------------------')
        print('\n')

        print('--------------------------------------------------------------')
        print('Time taken in seconds', int(t2 - t1))
        print('--------------------------------------------------------------')
        print('\n')
    else:
        slave_operation()