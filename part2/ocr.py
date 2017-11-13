#!/usr/bin/python
#
# ./ocr.py : Perform optical character recognition, usage:
#     ./ocr.py train-image-file.png train-text.txt test-image-file.png
#
# Authors: (insert names here)
# (based on skeleton code by D. Crandall, Oct 2017)
#

from PIL import Image, ImageDraw, ImageFont
import sys
from __future__ import division

CHARACTER_WIDTH = 14
CHARACTER_HEIGHT = 25


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    print im.size
    print int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

def load_training_letters(fname):
    TRAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }

def read_data_part1(fname="bc.train"):
    exemplars = []
    file = open(fname, 'r');
    for line in file:
        data = tuple([w for w in line.split()])
        exemplars += [ " ".join(data[0::2]) ] # fix space before fullstop
    return exemplars



def train(data):
    VALID_CHAR = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    P_char = {}
    initial = {}
    transition = {ch:{} for ch in VALID_CHAR}
    emission = {}
#    with open(fname, 'r') as fhand:
#        data = fhand.read()
#        unicode_data = data.decode("utf-8")
#        data = unicode_data.encode("ascii", "ignore")

    # Total count of a character
    for line in data:
        for letter in line:
            if letter in VALID_CHAR:
                P_char[letter] = P_char.get(letter, 0) + 1
    # Convert to prob
    S_total = sum(P_char.values())
    for l in P_char:
        P_char[l] /= S_total

    # Count of character at the start of a line
    for line in data:
        initial[line[0]] = initial.get(line[0], 0) + 1
    # Convert to prob
    S_total = len(data)
    for l in initial:
        initial[l] /= S_total

    # Transition Probability
    for line in data:
        for l, l_n in zip(line, line[1:]):
            if l in VALID_CHAR and l_n in VALID_CHAR:
                transition[l][l_n] = transition[l].get(l_n, 0) + 1
    # Convert to probability
    for l in transition:
        S_total = sum(transition[l].values())
        for l_n in transition[l]:
            transition[l][l_n] /= S_total

    return P_char, initial, transition


#####
# main program
(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)

P_char, initial, transition = train(data = read_data_part1())
## Below is just some sample code to show you how the functions above work.
# You can delete them and put your own code here!


# Each training letter is now stored as a list of characters, where black
#  dots are represented by *'s and white dots are spaces. For example,
#  here's what "a" looks like:
print "\n".join([ r for r in train_letters['a'] ])

# Same with test letters. Here's what the third letter of the test data
#  looks like:
print "\n".join([ r for r in test_letters[2] ])



