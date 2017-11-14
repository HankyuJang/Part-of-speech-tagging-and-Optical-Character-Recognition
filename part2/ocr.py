#!/usr/bin/python
#
# ./ocr.py : Perform optical character recognition, usage:
#     ./ocr.py train-image-file.png train-text.txt test-image-file.png
#
# Authors: (insert names here)
# (based on skeleton code by D. Crandall, Oct 2017)
#

from __future__ import division
from PIL import Image, ImageDraw, ImageFont
import sys
import numpy as np
from math import log
from scipy.misc import comb
CHARACTER_WIDTH = 14
CHARACTER_HEIGHT = 25
SMALL_PROB = 1/10**4
SMALL_PROB2 = 1/10**3
VALID_CHAR = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
states = list(VALID_CHAR)
## Delete this ###
f = 0
b = 0
v = []
vi = []
##################

def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
#    print im.size
#    print int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH
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

    P_char = {}
    initial = {}
    transition = {ch:{} for ch in VALID_CHAR}

    ##### Delete this ################
#    with open("sh.txt", 'r') as fhand:
#        data = fhand.read()
#        unicode_data = data.decode("utf-8")
#        data = unicode_data.encode("ascii", "ignore")
    ##################################

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
    valid_lines = 0
    for line in data:
        if line[0] in VALID_CHAR:
            valid_lines += 1
            initial[line[0]] = initial.get(line[0], 0) + 1

    # Convert to prob
    for l in initial:
#        initial[l] = 1/len(states)   # uncomment to make const 1/72
        initial[l] /= valid_lines

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

# Increase coeffcient of fp or tn to handle ' ' better, but this may end up in many empty spaces.
def emission(st, obs):
    """
    obs: list of list representing the character
    st: character

    tp: True positive  - obs:'*', st:'*'
    fp: False positive - obs:'*', st:' '
    tn: True negative  - obs:' ', st:'*'
    fn: False negative - obs:' ', st:' '
    """
<<<<<<< HEAD
    m, n, star = 0, 0, 0
    for line_train, line_obs in zip(train_letters[st], obs):
        for p1, p2 in zip(line_train, line_obs):
            if p1 == "*":
                star += 1
            if p1 == p2:
                m += 1
            else:
                n += 1
    return (m + 1) / ( (m+n) + 72)
#    return 0.8*m/(m+n)
=======
    tp, fn, tn, fp = 0, 0, 0, 0
    for line_train, line_obs in zip(train_letters[st], obs):
        for p1, p2 in zip(line_train, line_obs):
            if p1 == '*' and p2 == '*':
                tp += 1
            elif p1 == ' ' and p2 == ' ':
                fn += 1
            elif p1 == '*' and p2 == ' ':
                tn += 1
            elif p1 == ' ' and p2 == '*':
                fp += 1
    
    return (0.95**tp)*(0.6**fn)*(0.4**tn)*(0.2**fp)
>>>>>>> 68223437f9f072529e48fa40e134ca6a276b9260

# Functions for each algorithm.
#
def simplified(sentence):
    ##### P(S | W) = P(W | S) * P(S) / P(W)
    # using 1/72 for p_initial

    predicted_states = []
    observed = sentence
    for obs in observed:
        most_prob_state = max([ (st, emission(st, obs) * 1/len(states)) \
                                    for st in states], key = lambda x: x[1])
        predicted_states.append(most_prob_state[0])
    return predicted_states

# def upscale(number):
    # factor = 0
    # while number < 1:
        # number *= 10
        # factor += 1
    # return factor

def hmm_ve(sentence):
    global f
    global b

    observed = sentence
    forward = np.zeros([len(states), len(observed)])
    backward = np.zeros([len(states), len(observed)])
    predicted_states = []
    factor_list = np.zeros([len(states), len(observed)])

    for i, obs in enumerate(observed):
        for j, st in enumerate(states):
            if i == 0:
                p = initial.get(st, SMALL_PROB)     # initial
            else:
                p = sum( [forward[k][i-1] * transition[key].get(st, SMALL_PROB) \
                            for k, key in enumerate(states)] )
<<<<<<< HEAD
            forward[j][i] = p * emission(st, obs) #* pow(10,50)
=======
            # factor = upscale(p*emission(st,obs))
            # factor_list[j][i] = factor
            forward[j][i] = p * emission(st, obs) * pow(10,80)
>>>>>>> 68223437f9f072529e48fa40e134ca6a276b9260

    for i, obs in zip(range(len(observed)-1, -1, -1), observed[::-1]):
        for j, st in enumerate(states):
            if i == len(observed) - 1:
                p = 1
            else:
                p = sum( [ backward[k][i+1] * transition[st].get(key, SMALL_PROB) * emission(key, observed[i+1]) \
                            for k, key in enumerate(states)] )
            # factor = upscale(p*emission(st,obs))
            # backward[j][i] = p * pow(10, factor_list[j][i])
            backward[j][i] = p * pow(10, 80)

    f = forward
    b = backward

    ve = np.multiply(forward, backward)

    for i in range(len(observed)):
        z = np.argmax(ve[:, i])
        predicted_states.append(states[z])
    global v
    v = ve
    return predicted_states

def hmm_viterbi( sentence):
    states = list(VALID_CHAR)
    observed = sentence

<<<<<<< HEAD
=======
    # observed = [word for word in sentence if word in self.words_in_training] # ignore unseen words
>>>>>>> 68223437f9f072529e48fa40e134ca6a276b9260
    viterbi = np.zeros([len(states), len(observed)])
    trace = np.zeros([len(states), len(observed)], dtype=int)

    for i, obs in enumerate(observed):
        for j, st in enumerate(states):
            if i == 0:
                viterbi[j][i], trace[j][i] = log(initial.get(st, SMALL_PROB)) + log(emission(st, obs)), 0
            else:
                max_k, max_p = max([ (k, viterbi[k][i-1] + log(transition[key].get(st, SMALL_PROB))) \
                                       for k, key in enumerate(states)], key = lambda x: x[1])
                viterbi[j][i], trace[j][i] = max_p + log(emission(st, obs)), max_k

    # trace back
    z = np.argmax(viterbi[:,-1])
    hidden = [states[z]]
    for i in range(len(observed)-1, 0, -1):
        z = trace[z,i]
        hidden.append(states[z])

    global vi
    vi = viterbi
    # return REVERSED traceback sequence
    return hidden[::-1]

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
#print "\n".join([ r for r in train_letters['a'] ])

# Same with test letters. Here's what the third letter of the test data
#  looks like:
#print "\n".join([ r for r in test_letters[2] ])

# print " Simple:", "".join(simplified(test_letters))
print " HMM VE:", "".join(hmm_ve(test_letters))
# print "HMM MAP:", "".join(hmm_viterbi(test_letters))
