#!/usr/bin/env python3
from random import randint

guess = randint(1, 99)
cont = True
numberOfTry = 0
print("This is an interactive guessing game!")
print("You have to enter a number between 1 and 99 to find \
out the secret number.")
print("Type 'exit' to end the game.")
print("Good luck!\n")
while cont:
    print("What's your guess between 1 and 99?")
    nb = input(">> ")
    if nb == "exit":
        print("Goodbye!")
        exit()
    try:
        num = int(nb)
    except ValueError:
        print("That's not a number.")
        numberOfTry += 1
        continue
    if num < 1 or num > 99:
        print("Your number is not in the range.")
        numberOfTry += 1
        continue
    numberOfTry += 1
    if num < guess:
        print("Too low!")
    elif num > guess:
        print("Too high!")
    else:
        if guess == 42:
            print("The answer to the ultimate question of life, \
the universe and everything is 42.")
        if numberOfTry == 1:
            print("Congratulations! You got it on your first try!")
            exit()
        else:
            print("Congratulations, you've got it!")
            print("You won in {} attempts!".format(numberOfTry))
            exit()
