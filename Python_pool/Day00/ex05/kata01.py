#!/usr/bin/env python3
"""
outputs keys and items of a dictionnary
"""

languages = {
    'Python': 'Guido van Rossum',
    'Ruby': 'Yukihiro Matsumoto',
    'PHP': 'Rasmus Lerdorf',
}
if len(languages) == 0:
    print("Nothing in dictionnary")
else:
    for cle, value in languages.items():
        print("{} was created by {}".format(cle, value))
