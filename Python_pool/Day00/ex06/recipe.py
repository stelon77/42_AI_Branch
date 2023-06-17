#!/usr/bin/env python3
import re

cookbook = {"sandwich":
            {"ingredients": ["ham", "bread", "cheese", "tomatoes"],
             "meal": "lunch",
             "prep_time": 10},
            "cake":
            {"ingredients": ["flour", "sugar", "eggs"],
             "meal": "dessert",
             "prep_time": 60},
            "salad":
            {"ingredients": ["avocado", "arugula", "tomatoes", "spinach"],
             "meal": "lunch",
             "prep_time": 15}}


def montrer(cookbook):
    print("voici les cles")
    for key in cookbook.keys():
        print(key)
    print("\nvoici les valeurs")
    for value in cookbook.values():
        print(value)
    print("\net voici la totalite des items")
    for key, value in cookbook.items():
        print("La clé {} contient la valeur {}.".format(key, value))


def printRecipe(name):
    """
    Print a recipe

    param: name of the recipe
    """
    try:
        recipe = cookbook[name]
    except KeyError:
        print("\nthis recipe doesn't exist in the cookbook")
        print("...")
        return
    print("\nRecipe for {}:". format(name))
    print("Ingredients list: {}". format(recipe["ingredients"]))
    print("To be eaten for {}". format(recipe["meal"]))
    print("Takes {} minutes of cooking.". format(recipe["prep_time"]))
    print("...")


def deleteRecipe(name):
    """
    Delete a recipe

    param: name of the recipe
    """
    if len(cookbook) == 0:
        print("no more recipe to delete in cookbook")
        return
    try:
        del cookbook[name]
    except KeyError:
        print("\nthis recipe doesn't exist in the cookbook")
        print("...")
        return
    print("\nrecipe of {} has been removed". format(name))
    print("...")


def addNewRecipe(name, ingredients, meal, prep_time):
    """
    Add a new recipe

    param: name of recipe, list of ingredients, meal, prep_time in minutes
    """

    recipe = dict()
    recipe["ingredients"] = ingredients
    recipe["meal"] = meal
    recipe["prep_time"] = prep_time
    cookbook[name] = recipe
    print("\nrecipe {} has been included in cookbook". format(name))
    print("...")


def showCookbook():
    """
    Show the name of recipes in the cookbook
    """

    if len(cookbook) == 0:
        print("\nAaarrrrgh : I have no recipe in my cookbook ! What a shame !")
    else:
        print("\nvoici les recettes :")
        for key in cookbook.keys():
            print(key)
    print("...")


run = 1
while run:
    print("Please select an option by typing the corresponding number:")
    print("1: Add a recipe")
    print("2: Delete a recipe")
    print("3: Print a recipe")
    print("4: Print the cookbook")
    print("5: Quit")
    nb = input(">> ")
    if re.match("^[1-5]$", nb) is None:
        print("""This option does not exist,
please type the corresponding number.""")
        print("To exit, enter 5.")
        print("...")
        continue
    elif nb == '5':
        print("\nCookbook closed.")
        run = 0
    elif nb == '1':
        print("\nEnter the name of the recipe you want to add")
        name = input(">> ")
        ingredients = []
        ingredient = "n"
        while ingredient != "":
            print(("\nEnter the name one ingredient then enter, \
just type enter alone when at the end of the list"))
            ingredient = input(">> ")
            if ingredient != "":
                ingredients.append(ingredient)
        print("\nEnter the name of the meal for this recipe")
        meal = input(">> ")
        a = 1
        while a:
            print("\nEnter the time (in minutes) to cook this recipe")
            time = input(">> ")
            try:
                prep_time = int(time)
                if prep_time <= 0:
                    raise ValueError
                a = 0
            except ValueError:
                print("\nyou have to type a positive number")
        addNewRecipe(name, ingredients, meal, prep_time)
    elif nb == '2':
        print("\nEnter the name of the recipe you want to delete")
        name = input(">> ")
        deleteRecipe(name)
    elif nb == '3':
        print("\nPlease enter the recipe's name to get its details:")
        name = input(">> ")
        printRecipe(name)
    elif nb == '4':
        showCookbook()
