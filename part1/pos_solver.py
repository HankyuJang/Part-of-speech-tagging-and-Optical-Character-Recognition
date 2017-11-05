###################################
# CS B551 Fall 2017, Assignment #3
#
# Your names and user ids:
#
# (Based on skeleton code by D. Crandall)
#
#
####
# Put your report here!!
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
            for index, S in enumerate(line[1]):
                # Initial Probability
                if S in self.initial:
                    self.initial[S] += 1
                else:
                    self.initial[S] = 1
        ##############################################################
        # Transition Probability
        ##############################################################
        states = self.initial.keys()
        # Initialize transition probability with zeros.
        for S in states:
            self.transition[S] = {}
            for S_prime in states:
                self.transition[S][S_prime] = 0

        for line in data:
            n = len(line[1]) - 1 # n is for transitional probability
            for index, S in enumerate(line[1]):
                if index < n:
                    S_prime = line[1][index+1]
                    self.transition[S][S_prime] += 1

        ##############################################################
        # Transition Probability
        ##############################################################
        for S in states:
            self.emission[S] = {}

        for line in data:
            for index, S in enumerate(line[1]):
                # Emission Probability
                W = line[0][index]
                if S not in self.emission:
                    self.emission[S] = {}
                # Initialize emission probabilities with 0 for all possible W
                for S_other in states:
                    if W not in self.emission[S_other]:
                        self.emission[S_other][W] = 0
                self.emission[S][W] += 1

        self.words_in_training = self.emission[S].keys()

        # Convert Counts to probabilities
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
        return [ "noun" ] * len(sentence)

    def hmm_ve(self, sentence):
        return [ "noun" ] * len(sentence)

    def hmm_viterbi(self, sentence):
        states = self.initial.keys()
        observed = sentence
        # observed = [word for word in sentence if word in self.words_in_training]

        score = np.zeros([len(states), len(observed)])
        trace = np.zeros([len(states), len(observed)], dtype=int)

        for i, obs in enumerate (observed) :
            for j, st in enumerate (states) :
                if i == 0:
                    if obs in self.words_in_training:
                        score[j][i] = self.initial[st] * self.emission[st][obs]
                        trace[j][i] = 0
                    else:
                        score[j][i] = 0
                        trace[j][i] = 0
                    #print score[j][i]
                else:
                    if obs in self.words_in_training:
                        maximum = score[j][i-1] * self.transition[st][st] * self.emission[st][obs]
                        max_k = j
                        for k, key in enumerate(self.transition.keys()):
                            if score[k][i-1] * self.transition[key][st] * self.emission[st][obs] > maximum:
                                maximum = score[k][i-1] * self.transition[key][st] * self.emission[st][obs]
                                max_k = k
                        score[j][i] = maximum
                        trace[j][i] = max_k
                    else:
                        score[j][i] = 0
                        trace[j][i] = 0

        # trace back
        z = np.argmax(score[:,-1])
        hidden = [states[z]]

        for i in range(1,len(observed))[::-1] :
            z = trace[z,i]
            hidden.append(states[z])

        # return REVERSED traceback sequence
        print hidden
        return hidden[::-1]
        # return hidden
        

        # return [ "noun" ] * len(sentence)


    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
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

