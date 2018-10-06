# -*- coding:utf-8 -*-


def print_mul():
    for i in range(1, 10):
        for j in range(1, i + 1):
            print "%d*%d=%d" % (i, j, i*j),
            print '\t',
        print


if __name__ == '__main__':
    print_mul()
