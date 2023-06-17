import time
from random import randint
import os


def logger(fonction):
    def logging(*args, **kwargs):
        str2 =fonction.__name__.replace('_', ' ').title()
        timeBefore = time.time()
        ret = fonction(*args, **kwargs)
        timeAfter = time.time()
        execTime = timeAfter - timeBefore
        if execTime < 0.001:
            timeStr = "{:.3f}".format(execTime * 1000) + " ms ]"
        else:
            timeStr = "{:.3f}".format(execTime) + " s ]"
        logLine = "(" + os.environ["USER"] + ")Running: " + \
                  "{: <17}".format(str2) + "[ exec-time = " + timeStr
        # print(logLine)
        with open("machine.log", "a") as logger:
            logger.write(logLine + "\n")
        return ret
    return logging
