#!/usr/bin/env python3
from sys import argv

morse = {'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.',
         'F': '..-.', 'G': '--.', 'H': '....', 'I': '..', 'J': '.---',
         'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---',
         'P': '.--.', 'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-',
         'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-', 'Y': '-.--',
         'Z': '--..', ' ': '/', '1': '.----', '2': '..---', '3': '...--',
         '4': '....-', '5': '.....', '6': '-....', '7': '--...', '8': '---..',
         '9': '----.', '0': '-----'}

if len(argv) < 2:
    exit(0)
msg = (" ".join(argv[1:])).upper()
morseMsg = ""
for letter in msg:
    morseLetter = morse.get(letter)
    if morseLetter is None:
        print("ERROR")
        exit(0)
    morseMsg += (morseLetter + " ")
morseMsg = morseMsg.strip()
print(morseMsg)
