#!/usr/bin/python
#
# Perform optical character recognition, usage:
#     python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors: (insert names here)
# (based on skeleton code by D. Crandall, Oct 2020)
#

from PIL import Image, ImageDraw, ImageFont
import sys,copy,math

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [['*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg + CHARACTER_WIDTH) for y in range(0, CHARACTER_HEIGHT)], ]
    return result

def load_training_letters(fname):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }

class Image2Text: 

    def __init__(self, train_letters, test_letters, train_txt_fname):
        self.train_letters = train_letters
        self.test_letters = test_letters 
        self.train_txt_fname = train_txt_fname
        self.initial_probabilities, self.transition_probabilities, self.emission_probabilities = self.calculate_probabilities() 

    # return initial probability, transition probability and emission probabilty
    def calculate_probabilities(self):

        initial_probabilities, transition_probabilities, emission_probabilities = {},{},{}

        with open(self.train_txt_fname, 'r') as f:
            for i in f:

                char_list = list(" ".join([j for j in i.split()]))
                #Splitting into words from the test data(bc.train)

                if char_list:
                    # Calculating number of times each character is present at the beginning(for initial prob.)
                    initial_probabilities.setdefault(char_list[0], 0)
                    initial_probabilities[char_list[0]] += 1

                    # If it's the first time we are encountering a character 
                    for itr in range(1, len(char_list)):
                        # the transition probability is a dictionary that has charactera as the key and the value f each character is a dictionary containing the number of that how many times each character comes after this soecifi character

                        #For calculating number of times a character has appeared after the character to which it's being compared
                        transition_probabilities.setdefault(char_list[itr - 1], {char_list[itr]: 0})
                        #Since transition prob. depends on the previous state value
                        transition_probabilities[char_list[itr - 1]].setdefault(char_list[itr], 0)
                        transition_probabilities[char_list[itr - 1]][char_list[itr]] += 1
   
        # Calculating Initial Probability
        for i in initial_probabilities:
            initial_probabilities[i] /= sum(initial_probabilities[letter] for letter in initial_probabilities)

        # Calculation of Transition Probability
        for itr in transition_probabilities:

            for next_letter in transition_probabilities[itr]:

                transition_probabilities[itr][next_letter] /= sum(transition_probabilities[itr][next_letter] for next_letter in transition_probabilities[itr])

        # calculate emission probability
        for test_letter in range(len(self.test_letters)):

            emission_probabilities[test_letter] = {}

            for train_letter in self.train_letters:
                star_count,nstar_count = 0,0
                word_count,nword_count = 0,0

                # update star_count, word_count, nstar_count, nword_count
                for letter in range(len(self.test_letters[test_letter])):

                    if self.train_letters[train_letter][letter] == '*' and self.test_letters[test_letter][letter] == self.train_letters[train_letter][letter]:
                        star_count += 1

                    elif self.train_letters[train_letter][letter] == ' ' and self.test_letters[test_letter][letter] == self.train_letters[train_letter][letter] :
                        word_count += 1

                    elif self.train_letters[train_letter][letter] == '*':
                        nstar_count += 1

                    elif self.train_letters[train_letter][letter] == ' ':
                        nword_count += 1
                    
                    # multiply weighted exponents of the counts
                    emission_probabilities[test_letter][train_letter] = (0.99 ** star_count) * (0.7 ** word_count) * (0.3 ** nstar_count) * (0.01 ** nword_count)

        return initial_probabilities, transition_probabilities, emission_probabilities

    def hmm(self):
        # Using viterbi algorithm here to find the correct order of words 
        for i in range(len(self.test_letters)):

            current_state = [None] * 256

            for current_letter in (self.train_letters):
                # add a small value to probability to avoid math errors
                if self.emission_probabilities[i][current_letter] == 0:
                        self.emission_probabilities[i][current_letter] += 1e-10
                if i == 0:
                    result = -math.log(self.emission_probabilities[0][current_letter]) - math.log(self.initial_probabilities.get(current_letter, 1e-10))
                    current_state[ord(current_letter)] = [result, [current_letter]]
                else:
                    _ = sys.maxsize
                    max_previous_probability = []

                    for previous_letter in (self.train_letters):
                        previous_transition_probability= -math.log(self.transition_probabilities.get(previous_letter, {}).get(current_letter, 1e-10)) + previous_state[ord(previous_letter)][0]

                        if previous_transition_probability < _:

                            _ = previous_transition_probability
                            max_previous_probability = [previous_transition_probability, previous_state[ord(previous_letter)][1] + [current_letter]]

                    result = max_previous_probability[0] - math.log(self.emission_probabilities[i][current_letter])
                    current_state[ord(current_letter)] = [result, max_previous_probability[1]]
            previous_state = copy.deepcopy(current_state)

        _ = sys.maxsize
        result = []
        for i in previous_state:
            # find the minimum value to find the maximum a posteriori
            if i and _ > i[0]:
                _, result = i[0], i

        return ''.join(result[1])

    def image2text(self):
        simple_text = ""
        for letter in self.emission_probabilities:
            simple_text += "".join(max(self.emission_probabilities[letter], key=lambda x: self.emission_probabilities[letter][x]))

        hmm_text = self.hmm()
        return simple_text, hmm_text


#####
# main program
if len(sys.argv) != 4:
    raise Exception("Usage: python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png")

(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)

## Below is just some sample code to show you how the functions above work. 
# You can delete this and put your own code here!

image2text = Image2Text(train_letters, test_letters, train_txt_fname)
simple_text, hmm_text = image2text.image2text()

# The final two lines of your output should look something like this:
print("Simple: " + simple_text)
print("   HMM: " + hmm_text) 