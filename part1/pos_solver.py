###################################
# CS B551 Fall 2017, Assignment #3
#
# Name: Hankyu Jang, Pulkit Maloo, Shyam Narasimhan
# UserID: hankjang-maloop-shynaras
#
# (Based on skeleton code by D. Crandall)
#
#
####
# Put your report here!!
#
# (1) Description
#
# [Creating initial, transition, emission probability] First I calculated initial,
#  transition, and emission probabilities. In this procedure, I first calculated
#  initial probability. Now I have the total different types of the hidden states.
#  Using this information, I next initialized transition probabilities with all
#  possible combinations and set it to 0. Then I calculated transition probabilities.
#  Lastly, when creating emission probability, whenever a new word appears, I added
#  that word as the emission probability for all the hidden states and set it to 0.
#  The reason for the initialization is because, in the calculation of Viterbi algorithm,
#  I need all the information of the emission probabilities for all the words appeared
#  in training set.
#
# [Viterbi] I used `score` and `trace` matrices. `score` matrix contains the scores
#  calculated during the Viterbi algorithm. `trace` is used to trace back the
#  hidden states. During the traceback, I appended the states in the list `hidden`,
#  then returned the reverse order of `hidden` that returns the list of predicted
#  hidden states from the beginning of the given sentence.
#
##############################################################
# (2) Description of how the program works
#
# The program `label.py` takes in a training file and a test file, then applies
# three different algorithms to mark every word in a sentence with its part of speech.
# In `solver.train`, function `train` in class `Solver` is called. As described above,
# the train function creates `initial`, `transition`, and `emission` probabilities.
# Then using these information, the program tests the test data on three algorithm
# we implemented in the class `Solver`.
#
#
#
##############################################################
# (3) Disscussion of problems, assumptions, simplification, and design decisions we made
#
# There were words in test file that are not trained in training file. In this case,
# there's no emission probabilities, hence the Viterbi algorithm raised error.
# I tried two different approach on this problem. First approach was simply set
# the score to 0 whenever unknown word appeared. Following is the result of the
# approach for the test set bc.test:
#
#==> So far scored 2000 sentences with 29442 words.
#                   Words correct:     Sentences correct:
#   0. Ground truth:      100.00%              100.00%
#     1. Simplified:       18.60%                0.00%
#         2. HMM VE:       18.60%                0.00%
#        3. HMM MAP:       62.21%               30.15%
#
# The result was poor. Hence, I tried another approach: reproduce the algorithm
# by only removing the emission probability in calculation (Using the previous
# score and transition probability only). Following is the result of the approach
# for the test set bc.test:
#
#==> So far scored 2000 sentences with 29442 words.
#                   Words correct:     Sentences correct:
#   0. Ground truth:      100.00%              100.00%
#     1. Simplified:       18.60%                0.00%
#         2. HMM VE:       18.60%                0.00%
#        3. HMM MAP:       89.78%               31.55%
#
# There was tremendous improvement on the accuracy.
#
##############################################################
# (4) Answers to any questions asked below in the assignment
#
#
#
####

from __future__ import division
import random
import math
import numpy as np

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:

    SMALL_PROB = 1/float(10**6)
    initial_all = {}
    initial = {}
    transition = {}
    emission = {}
    words_in_training = []

    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
    def posterior(self, sentence, label):
        return 0

    # Do the training!
    #
    def train(self, data):
        ##############################################################
        # Initial Probability
        ##############################################################
        for line in data:
            ##### considering only first word
            self.initial[line[1][0]] = self.initial.get(line[1][0], 0) + 1
            ##### considering all words
            for S in line[1]:
                 self.initial_all[S] = self.initial_all.get(S, 0) + 1

        ##############################################################
        # Transition Probability
        ##############################################################
        states = self.initial.keys()

        for line in data:
            for S, S_prime in zip(line[1], line[1][1:]):
                try:
                    self.transition[S][S_prime] = self.transition[S].get(S_prime, 0) + 1
                except:
                    self.transition[S] = {}
                    self.transition[S][S_prime] = self.transition[S].get(S_prime, 0) + 1

        ##############################################################
        # Emission Probability
        ##############################################################
        for S in states:
            self.emission[S] = {}

        for line in data:
            for W, S in zip(line[0], line[1]):
                self.emission[S][W] = self.emission[S].get(W, 0) + 1
                # Initialize emission probabilities with 0 for all possible W
#                for S_other in states:
#                    if W not in self.emission[S_other]:
#                        self.emission[S_other][W] = 0
                self.words_in_training.append(W)

#        self.words_in_training = self.emission[S].keys()

        # Convert Counts to probabilities
        S_total = sum(self.initial_all.values())
        for S in self.initial_all:
            self.initial_all[S] /= S_total
        S_total = sum(self.initial.values())
        for S in self.initial:
            self.initial[S] /= S_total
        for S in self.transition:
            S_total = sum(self.transition[S].values())
            for S_prime in self.transition[S]:
                self.transition[S][S_prime] /= S_total
        for S in self.emission:
            S_total = sum(self.emission[S].values())
            for W in self.emission[S]:
                self.emission[S][W] /= S_total

    # Functions for each algorithm.
    #
    def simplified(self, sentence):
        ##### P(S | W) = P(W | S) * P(S) / P(W)
        states = self.initial_all.keys()
        predicted_states = []
        for word in sentence:
            most_prob_state = max([ (st, self.emission[st].get(word, self.SMALL_PROB) * self.initial_all[st]) \
                                        for st in states ], key = lambda x: x[1])
            predicted_states.append(most_prob_state[0])
#            max_prob, most_prob_state = 0, ''
#            for state in states:
#                P_state_given_word = self.emission[state].get(word, 1) * self.initial[state]
#                if P_state_given_word > max_prob:
#                    max_prob, most_prob_state = P_state_given_word, state
#            sentence_states.append(most_prob_state)
        return predicted_states

    def hmm_ve(self, sentence):
        states = self.initial.keys()
        observed = sentence
        # observed = [word for word in sentence if word in self.words_in_training]
        score = np.zeros([len(states), len(observed)])
        predicted_states = []

        for i, obs in enumerate(observed):

            max_value, max_state = 0, ''

            for j, st in enumerate(states):
                if i == 0:
                    score[j][i] = self.initial[st] * self.emission[st].get(obs, self.SMALL_PROB)
                else:
                    p = sum( [score[k][i-1] * self.transition[key].get(st, self.SMALL_PROB) \
                                for k, key in enumerate(self.transition)] )
                    score[j][i] = p * self.emission[st].get(obs, self.SMALL_PROB)

                if score[j][i] > max_value:
                    max_value, max_state = score[j][i], st
            predicted_states.append(max_state)

        return predicted_states

    def hmm_viterbi(self, sentence):
        states = self.initial.keys()
        observed = sentence
        # observed = [word for word in sentence if word in self.words_in_training] # ignore unseen words
        score = np.zeros([len(states), len(observed)])
        trace = np.zeros([len(states), len(observed)], dtype=int)

        for i, obs in enumerate(observed):
            for j, st in enumerate(states):
                if i == 0:
                    score[j][i] = self.initial[st] * self.emission[st].get(obs, self.SMALL_PROB)
                    trace[j][i] = 0
                    #print score[j][i]
                else:
                    max_k, maximum = max([ (k, score[k][i-1] * self.transition[key].get(st, self.SMALL_PROB)) \
                                           for k, key in enumerate(self.transition)], key = lambda x: x[1])
                    score[j][i], trace[j][i] = maximum * self.emission[st].get(obs, self.SMALL_PROB), max_k
        # trace back
        z = np.argmax(score[:,-1])
        hidden = [states[z]]
        for i in range(len(observed)-1, 0, -1):
            z = trace[z,i]
            hidden.append(states[z])

        # return REVERSED traceback sequence
        return hidden[::-1]

    # This solve() method is called by label.py, so you should keep the interface
    #  the same, but you can change the code itself.
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, algo, sentence):
        if algo == "Simplified":
            return self.simplified(sentence)
        elif algo == "HMM VE":
            return self.hmm_ve(sentence)
        elif algo == "HMM MAP":
            return self.hmm_viterbi(sentence)
        else:
            print "Unknown algo!"



