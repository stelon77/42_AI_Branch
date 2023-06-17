from datetime import datetime
from recipe import Recipe


class Book:
    """a fabulous cooking book with good recipes"""

    def __init__(self, nameOfBook):
        self.name = nameOfBook
        self.last_update = datetime.now()
        self.creation_date = self.last_update
        self.recipes_list = {"starter": [], "lunch": [], "dessert": []}

    def get_recipe_by_name(self, name):
        """Prints a recipe with the name and returns the instance"""
        for meal in self.recipes_list.values():
            for recipe in meal:
                if recipe.name == name:
                    print(str(recipe))
                    return recipe
        print("there is no recipe with this name in this book")
        return(None)

    def get_recipes_by_types(self, recipe_type):
        """Get all recipe names for a given recipe_type """
        for meal, recipes in self.recipes_list.items():
            if meal == recipe_type:
                if len(recipes) == 0:
                    print("No available recipe for", recipe_type)
                    return []
                print("This is the list of available recipe for",
                      recipe_type, ":\n")
                for recipe in recipes:
                    print("-", recipe.name, "\n")
                return recipes
        print("This is not a recipe type of the book")
        return []

    def add_recipe(self, recipe):
        """Add a recipe to the book and update last_update"""
        if not isinstance(recipe, Recipe):
            print("This is not a recipe, please provide a recipe")
            return
        for meal, recipes in self.recipes_list.items():
            if meal == recipe.recipe_type:
                for check in recipes:
                    if check.name == recipe.name:
                        print("this recipe already exists in the cookbook")
                        return
                recipes.append(recipe)
                self.last_update = datetime.now()
                return
        print("this recipe_type is not allowed in the book")
        return
