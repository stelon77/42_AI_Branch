#!/usr/bin/env python3

import time


def ft_progress(lst):
    start = time.time()
    nb = 0
    max = len(lst)
    if max == 0:
        print("the list is empty")
        exit()
    for i in lst:
        nb += 1
        percent = int(nb * 100 / max)
        elapsedTime = time.time() - start
        eta = elapsedTime * max / nb
        bar = "=" * int(percent * 24 / 100) + ">"
        print("\rETA: {:.2f}s [{:3d}%] [{: <25}] {}/{} | elapsed time {:.2f}"
              .format(eta, percent, bar, nb, max, elapsedTime), end="")
        yield i


if __name__ == "__main__":
    listy = range(1000)
    ret = 0
    for elem in ft_progress(listy):
        ret += (elem + 3) % 5
        time.sleep(0.01)
    print()
    print(ret)
    # listy = range(3333)
    # ret = 0
    # for elem in ft_progress(listy):
    #     ret += elem
    #     time.sleep(0.005)
    # print()
    # print(ret)
