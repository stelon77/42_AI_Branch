from my_minipack.progressbar import progressbar
import time

if __name__ == "__main__":
    listy = range(1000)
    ret = 0
    for elem in progressbar(listy):
        ret += (elem + 3) % 5
        time.sleep(0.01)
    print()
    print(ret)
    # listy = range(3333)
    # ret = 0
    # for elem in progressbar(listy):
    #     ret += elem
    #     time.sleep(0.005)
    # print()
    # print(ret)
