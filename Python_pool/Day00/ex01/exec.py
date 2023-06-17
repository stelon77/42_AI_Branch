from sys import argv

if len(argv) < 2:
    exit()
print((' '.join(argv[1:])).swapcase()[::-1])
