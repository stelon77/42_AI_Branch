import time
from random import randint
import os


def log(fonction):
    def logging(*args, **kwargs):
        str2 = fonction.__name__.replace('_', ' ').title()
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


class CoffeeMachine():
    water_level = 100

    @log
    def start_machine(self):
        if self.water_level > 20:
            return True
        else:
            print("Please add water!")
            return False

    @log
    def boil_water(self):
        return "boiling..."

    @log
    def make_coffee(self):
        if self.start_machine():
            for _ in range(20):
                time.sleep(0.1)
                self.water_level -= 1
            print(self.boil_water())
            print("Coffee is ready!")

    @log
    def add_water(self, water_level):
        time.sleep(randint(1, 5))
        self.water_level += water_level
        print("Blub blub blub...")


if __name__ == "__main__":
    machine = CoffeeMachine()
    for i in range(0, 5):
        machine.make_coffee()
    machine.make_coffee()
    machine.add_water(70)
