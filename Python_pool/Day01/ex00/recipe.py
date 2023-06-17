r_type = ("starter", "lunch", "dessert")


class Recipe:
    """ all you want to know about a cooking recipe"""
    def __init__(self, name, cooking_lvl, cooking_time,
                 ingredients, description="", recipe_type=None):
        """constructor of object 'recipe'

        params : name(str), cooking_lvl(1-5), cooking_time(int),
                ingredients(list),recipe_type(str), description(str)
        """
        if type(name) is not str:
            print("the name has to be a chain of characters")
            exit()
        if (type(cooking_lvl) is not int) or cooking_lvl < 1 \
           or cooking_lvl > 5:
            print("the cooking lvl is a number between 1 and 5")
            exit()
        if (type(cooking_time) is not int) or cooking_time < 0:
            print("the cooking_time is a positive number")
            exit()
        if type(ingredients) is not list:
            print("ingredients has to be a list of strings")
            exit()
        if len(ingredients) == 0:
            print("ingredients can't be empty")
            exit()
        for ingredient in ingredients:
            if type(ingredient) is not str:
                print("ingredients has to be a list of strings")
                exit()
        if r_type.count(recipe_type) == 0:
            print("recipe type has to  be one of these : ",
                  ", ".join(i for i in r_type))
            exit()
        self.name = name
        self.cooking_lvl = cooking_lvl
        self.cooking_time = cooking_time
        self.ingredients = ingredients
        self.recipe_type = recipe_type
        self.description = str(description)

    def __str__(self):
        """Return the string to print with the recipe info"""
        txt = "Recipe of : " + self.name \
            + "\n\nCooking level : " + str(self.cooking_lvl) \
            + "\nCooking time : " + str(self.cooking_time) + " minutes" \
            + "\nRecipe Type : " + self.recipe_type \
            + "\n\nIngredients : " \
            + ", ".join(ingredient for ingredient in self.ingredients) \
            + "\n\nDescription :\n" + self.description
        return txt


if __name__ == "__main__":
    tourte = Recipe("lolo", 2, 5, ["chou", "hibou", "genou"], "", "lunch")
    print(str(tourte))
