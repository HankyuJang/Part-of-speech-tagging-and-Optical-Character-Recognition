#!/usr/bin/python
#
# ./ocr.py : Perform optical character recognition, usage:
#     ./ocr.py train-image-file.png train-text.txt test-image-file.png
#
# Authors: Hankyu Jang, Pulkit Maloo, Shyam Narasimhan
# UserID: hankjang-maloop-shynaras
# (based on skeleton code by D. Crandall, Oct 2017)
#
# (1) Description
#
# [Creating initial, transition, emission probability] First we calculated initial
#  and transition probabilities. This procedure was done similarly as the part1.
#  Calculating the emission probability was different from the part1.
#  I explained this in detain in (3).
#
# [Simplified]
#  For Simplified, for every word, we maximized over P(POS)*P(word | POS)
#  In case the test data set had new unseen word, a very small value of probability was
#  taken as emission
#
# [Variable Elimination]
# For variable elimination, we used Dynamic Programming to calculate forward propagation
# and backward propagation values. To handle the underflow, we scaled the probability
# to get over 1, and saved the scaling factors. Using this information, we kept track
# of the log of the sum of probabilities. In the last step of combining the forward
# and backward probabilities, we added the log probabilities,
#
# [Viterbi] we used `score` and `trace` matrices. `score` matrix contains the scores
#  calculated during the Viterbi algorithm. `trace` is used to trace back the
#  hidden states. During the traceback, we appended the states in the list `hidden`,
#  then returned the reverse order of `hidden` that returns the list of predicted
#  hidden states from the beginning of the given sentence. To handle the underflow,
#  we used log probabilities.
#
##############################################################
# (2) Description of how the program works
#
# The program `ocr.py` takes in a training image file, test text file, and test image file.
# The Initial and transition probabilities are calculated from the test text file.
# The training image file and the test image file  is used to calculate the
# emission probability. Using those probabilities, the program outputs the possible
# hidden states for three different algorithms.
#
# We made a python script `test-ocr.py` to test our three algorithm to test images.
# You can run the script in the Server. Following is the output of running the script.
#
# [hankjang@silo part2]$ python test-ocr.py
# ####################################
# Correct: SUPREME COURT OF THE UNITED STATES
# Simple: SUPREME COURT OF THE UNITED STATES
# HMM VE: SUPREME COURT OF THE UN1TED STATES
# HMM MAP: SUPREME COURT OE THE UNITED STATES
# ####################################
# Correct: Certiorari to the United States Court of Appeals for the Sixth Circuit
# Simple: C rt1crar1 to tn  Un1ted  tat-s  cur  o  -pp-a1a   .  he S1xth C1rcu1t
# HMM VE: C rt1orar1 to tn  Un1teu  tate   our  o  -pp-a1    r  he Sirth Circui
# HMM MAP: C rticrarl to the United States Court of -ppeala ' . the Sixth Circuit
# ####################################
# Correct: Nos. 14-556. Argued April 28, 2015 - Decided June 26, 2015
# Simple: Nos. 14-556. Argued Apri1 28, 2015 - Decided June 26, 2015
# HMM VE: Nos. 14-556. Argued Apri1 28, 2015 - Decided June 26, 2015
# HMM MAP: Nos. 14-556. Argued April 28, 2015 - Decided June 26, 2015
# ####################################
# Correct: Together with No. 14-562, Tanco et al. v. Haslam, Governor of
# Simple: Together with No. 14-562, Tanco et al. v. Haslam, Governor of
# HMM VE: Together with No. 14-562, Tanoo et al. v. Haslam, Governor of
# HMM MAP: Together with No. 14-562, Tanco et al. v. Haslam, Governor of
# ####################################
# Correct: Tennessee, et al., also on centiorari to the same court.
# Simple: Tennessee, et a1., also on centiorari to the same court.
# HMM VE: Tennessee, et a1., also on centiorari to the same court.
# HMM MAP: Tennessee, et al , also on centiorari to the same court.
# ####################################
# Correct: Opinion of the Court
# Simple: Opinion"ofIthekCourt
# HMM VE: Opihion oflthe Court
# HMM MAP: Opinion ofithe Court
# ####################################
# Correct: As some of the petitioners in these cases demonstrate, marriage
# Simple: As some of the petitioners in these cases demonstrate, marr1age
# HMM VE: As some of the petitioners in these cases demonstrate, marriage
# HMM MAP: As some of the petitioners in these cases demonstrate. marriage
# ####################################
# Correct: embodies a love that may endure even past death.
# Simple: embodies a love that may endure even past death.
# HMM VE: embodies a love that may endure even past death.
# HMM MAP: embodies a love that may endure even past death.
# ####################################
# Correct: It would misunderstand these men and women to say they disrespect
# Simple: 1  w u1u m1su  -r tanc th- - m-  an    m n t   ay  h , u1 r-s -ct
# HMM VE: 1    u1u m1su  er tanc the   me  an    m n t   ay  h y u1 r-   ct
# HMM MAP: It would misu cer tand the e men and . m n to cay th , ol r s -ct
# ####################################
# Correct: the idea of marriage.
# Simple: the idea of marriage.
# HMM VE: the idea of marriage.
# HMM MAP: the idea of marriage.
# ####################################
# Correct: Their plea is that they do respect it, respect it so deeply that
# Simple: Their p1ea is that they do respect it, respect it so deep1y that
# HMM VE: Their p1ea is that they do respect it, respect it so deeply that
# HMM MAP: Their plea is that they do respect it, respect it so deeply that
# ####################################
# Correct: they seek to find its fulfillment for themselves.
# Simple: they seek to f1nd 1ts fu1f111ment for themse1ves.
# HMM VE: they seek to find 1ts fu1fi11ment for themselves.
# HMM MAP: they seek to find its fulfillment for themselves.
# ####################################
# Correct: Their hope is not to be condemned to live in loneliness,
# Simple: Their hope is not to be condemned to 1ive in lone1iness,
# HMM VE: Their hope is not to be condemned to 1ive in lone1iness,
# HMM MAP: Their hope is not to be condemned to live in loneliness,
# ####################################
# Correct: excluded from one of civilization's oldest institutions.
# Simple: excluded from one of civilization's oldest institutions.
# HMM VE: excluded from one of civilization's oldest institutions.
# HMM MAP: excluded from one of civilization's oldest institutions.
# ####################################
# Correct: They ask for equal dignity in the eyes of the law.
# Simple: They ask for equal dignity in the eyes of the law.
# HMM VE: They ask for equal dignity in the eyes of the law.
# HMM MAP: They ask for equal dignity in the eyes of the law.
# ####################################
# Correct: The Constitution grants them that right.
# Simple: The"Constitution-grants them that right.
# HMM VE: The Constitution grants them that right.
# HMM MAP: The Constitution grants them that right.
# ####################################
# Correct: The judgement of the Court of Appeals for the Sixth Circuit is reversed.
# Simple: The judgement of the Court of nppea1s for the Sixth Circu1t 1s reversed.
# HMM VE: The judgement of the Court of hppea1s for the Sixth Circuit 1s reversed.
# HMM MAP: The judgement of the Court of sppeals for the Sixth Circuit is reversed.
# ####################################
# Correct: It is so ordered.
# Simple: It is so ordered.
# HMM VE: It is so ordered.
# HMM MAP: It is so ordered.
# ####################################
# Correct: KENNEDY, J., delivered the opinion of the Court, in which
# Simple: KENNEDY, J., de11vered the op1n1on cf the Court, 1n which
# HMM VE: KENNEDY, J., de11vered the opin1on of the Court, in which
# HMM MAP: NEUNEDI. J., delivered the opinion of the Court. in which
#
##############################################################
# (3) Disscussion of problems, assumptions, simplification, and design decisions we made
#
#  We implemented a simple Naive Bayes classifier for the emission probability.
#  Here, hidden states are set of valid characters in `TRAIN_LETTERS`.
#  Obervation are 350 binary characters (pixel) that are either '*' or ' '.
#  We compared each observed pixel with the pixel of the hidden state.
#  We differentiated the four possible combinations, and named each case as the following:
#
#  tp: True positive  - obs:'*', st:'*'
#  fp: False positive - obs:'*', st:' '
#  tn: True negative  - obs:' ', st:'*'
#  fn: False negative - obs:' ', st:' '
#
#  This means that for a given pixel in position (i,j),
#  if the O_{ij} == '*' and l_{ij} == '*', we regard this case as `tp`.
#  Intuitively, if this is the case, the probability of the hidden state to be the
#  actual hidden state would be high.
#  However, if O_{ij} == '*' and l_{ij} == ' ', the probability would be low,
#  because it is unlikely to have a pixel in observed letter is black, but actual
#  character (hidden state) being white.
#  We used this information, and experimented with the different probabilities per case.
#
#  As the result, we fixed 0.95 for `tp`, 0.6 for `fn`, 0.4 for `tn` and 0.2 for `fp`.
#  This means that if obs = "**  *", and st = "*****", the emission would be
#  0.95 * 0.95 * 0.4 * 0.4 * 0.95
#  When finalizing the probabilites for each case, we had to make a decision.
#  if we want to change the coefficients to make the images with noise work better,
#  the performance goes down for the images that are blurred, and vice versa.
#  Hence, we settled at some point that works pretty well for both of the cases.
#  If we can have the idea of whether the image noisy or blurry, then we can feed in two
#  different sets of probabilities that could make better recognition of the words.
#
#  In practice, we are multiplying 350 probabilities to get the emission probability.
#  This lead to problems of underflow in the Variable Elimination algorithm. We could not
#  simply take log probabilities because it contained calculations such as sum of
#  probabilities. Hence, we tried a different approach.
#
#  We used scaling method in Variable Elimination, then kept track of the sum of the
#  log probabilities separately. We got this idea from the equation (23) in the paper
#  <Hidden Markov Models> by Phil Blunsom.
#
####

from __future__ import division
from PIL import Image, ImageDraw, ImageFont
import sys
import numpy as np
from math import log

#==============================================================================
# Change Flag to True to train on train_txt_fname
#  as currently the models are training on "bc.train"
flag = False
#==============================================================================


CHARACTER_WIDTH = 14
CHARACTER_HEIGHT = 25
SMALL_PROB = 1/10**6
SCALE_FACTOR = 10
VALID_CHAR = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
states = list(VALID_CHAR)
emission_dict = {st: {} for st in states}


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


def read_data(fname, flag):
    exemplars = []

    if flag:
        with open(fname, 'r') as file:
            for line in file:
                data = tuple([w for w in line.split()])
                exemplars += [ " ".join(data) ]

    else:
        with open("bc.train", 'r') as file:
            for line in file:
                data = tuple([w for w in line.split()])
                exemplars += [ " ".join(data[0::2]) ] # fix space before fullstop

    return exemplars


def train(data):
    P_char = {}
    initial = {}
    transition = {ch:{} for ch in VALID_CHAR}

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
        #initial[l] = 1/len(states)   # uncomment to make const 1/72
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


# Increase coeffcient of fp or tn to handle ' ' better,
# but this may end up in many empty spaces.
def emission(st, i):
    """
    obs: list of list representing the character
    st: character

    tp: True positive  - obs:'*', st:'*'
    fp: False positive - obs:'*', st:' '
    tn: True negative  - obs:' ', st:'*'
    fn: False negative - obs:' ', st:' '
    """
    try:
        return emission_dict[st][i]
    except:
        obs = test_letters[i]
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

        emission_dict[st][i] = (0.95**tp)*(0.6**fn)*(0.4**tn)*(0.2**fp)
        return emission_dict[st][i]


def upscale(number):
    factor = 0
    while number < 1:
        number *= SCALE_FACTOR
        factor += 1
    return factor


# Functions for each algorithm.
#
def simplified(sentence):
    ##### P(S | W) = P(W | S) * P(S) / P(W)
    # using 1/72 for p_initial

    predicted_states = []
    observed = sentence
    for i, obs in enumerate(observed):
        most_prob_state = max( [ (st, emission(st, i) * 1/len(states)) \
                                    for st in states ], key = lambda x: x[1] )
        predicted_states.append(most_prob_state[0])
    return predicted_states


def hmm_ve(sentence):
    observed = sentence
    forward = np.zeros([len(states), len(observed)])
    backward = np.zeros([len(states), len(observed)])
    forward_log = np.zeros([len(states), len(observed)])
    backward_log = np.zeros([len(states), len(observed)])
    forward_scale = np.zeros([len(states), len(observed)])
    backward_scale = np.zeros([len(states), len(observed)])
    predicted_states = []

    for i, obs in enumerate(observed):
        for j, st in enumerate(states):
            if i == 0:
                p = initial.get(st, SMALL_PROB)     # P_char
                # p = P_char.get(st, SMALL_PROB)     # P_char
                # p = 1/len(states)                  # const - 1/72
            else:
                p = sum([forward[k][i-1] * transition[key].get(st, SMALL_PROB) \
                            for k, key in enumerate(states)])
            factor = upscale(p * emission(st, i))
            forward_scale[j][i] = factor
            forward[j][i] = p * emission(st, i) * pow(SCALE_FACTOR, factor)
            forward_log[j][i] = log(p * emission(st, i))

    for i, obs in zip(range(len(observed)-1, -1, -1), observed[::-1]):
        for j, st in enumerate(states):
            if i == len(observed) - 1:
                p = 1
            else:
                p = sum( [ backward[k][i+1] * transition[st].get(key, SMALL_PROB) * emission(key, i+1) \
                            for k, key in enumerate(states)] )
            factor = upscale(p * emission(st, i))
            backward_scale[j][i] = factor
            backward[j][i] = p * pow(SCALE_FACTOR, factor)
            backward_log[j][i] = log(p)

    ve = forward_log + backward_log

    for i in range(len(observed)):
        z = np.argmax(ve[:, i])
        predicted_states.append(states[z])
    return predicted_states


def hmm_viterbi( sentence):
    observed = sentence

    viterbi = np.zeros([len(states), len(observed)])
    trace = np.zeros([len(states), len(observed)], dtype=int)

    for i, obs in enumerate(observed):
        for j, st in enumerate(states):
            if i == 0:
                viterbi[j][i], trace[j][i] = log(initial.get(st, SMALL_PROB)) + log(emission(st, i)), 0
                # viterbi[j][i], trace[j][i] = log(P_char.get(st, SMALL_PROB)) + log(emission(st, i)), 0
                # viterbi[j][i], trace[j][i] = log(1/len(states)) + log(emission(st, i)), 0
            else:
                max_k, max_p = max( [( k, viterbi[k][i-1] + log(transition[key].get(st, SMALL_PROB)) ) \
                                       for k, key in enumerate(states)], key = lambda x: x[1] )
                viterbi[j][i], trace[j][i] = max_p + log(emission(st, i)), max_k

    # trace back
    z = np.argmax(viterbi[:,-1])
    hidden = [states[z]]
    for i in range(len(observed)-1, 0, -1):
        z = trace[z,i]
        hidden.append(states[z])

    # return REVERSED traceback sequence
    return hidden[::-1]


#####
# main program
(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)

P_char, initial, transition = train(data = read_data(train_txt_fname, flag))

print " Simple:", "".join(simplified(test_letters))
print " HMM VE:", "".join(hmm_ve(test_letters))
print "HMM MAP:", "".join(hmm_viterbi(test_letters))
