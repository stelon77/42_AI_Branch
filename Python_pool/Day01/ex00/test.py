from recipe import Recipe
from book import Book

print("creer recette et l'afficher\n")

# Voici des recettes falacieuses
# tarte = Recipe(4, 2, 30, ["flour", "butter", "apple"],
#                "A delicious apple pie", "dessert")
# tarte = Recipe("pie", 9, 30, ["flour", "butter", "apple"],
#                "A delicious apple pie", "dessert")
# tarte = Recipe("pie", 2, -30, ["flour", "butter", "apple"],
#                "A delicious apple pie", "dessert")
# tarte = Recipe("pie", 2, 3.0, ["flour", "butter", "apple"],
#                "A delicious apple pie", "dessert")
# tarte = Recipe("pie", 2, 30, [4, "butter", "apple"],
#                "A delicious apple pie", "dessert")
# tarte = Recipe("pie", 2, 30, ["flour", "butter", "apple"],
#                "A delicious apple pie", "desert")
tarte = Recipe("pie", 2, 30, ["flour", "butter", "apple"],
               "A delicious apple pie", "dessert")
to_print = str(tarte)
print(to_print)

myCookBook = Book("me")
print("\nOh lala grosse erreur !!")
myCookBook.add_recipe(to_print)

print("\nla ca va mieux !!")
# ajout de recettes
myCookBook.add_recipe(tarte)
myCookBook.add_recipe(Recipe("cake", 5, 60,
                      ["eggs", "chocolate", "flour"], "", "dessert"))
myCookBook.add_recipe(Recipe("salted cake", 5, 60,
                      ["eggs", "salt", "flour"], "", "lunch"))
# on met une recette qui a le meme nom
myCookBook.add_recipe(Recipe("salted cake", 5, 60,
                      ["egg", "salt", "flour"], "", "lunch"))
myCookBook.get_recipe_by_name("cake")

print("\nOh lala grosse erreur !!")
myCookBook.get_recipe_by_name(2)
myCookBook.get_recipe_by_name("cake2")

print("\nOh lala grosse erreur !!")
myCookBook.get_recipes_by_types("cakes")

print("\nla ca va mieux !!")
myCookBook.get_recipes_by_types("dessert")
myCookBook.get_recipes_by_types("lunch")
myCookBook.get_recipes_by_types("starter")
for recipe in myCookBook.get_recipes_by_types("dessert"):
    print(recipe)
print("\n\n")
print(myCookBook.get_recipes_by_types("starter"))
