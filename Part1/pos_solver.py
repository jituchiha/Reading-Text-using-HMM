###################################
# CS B551 Fall 2022, Assignment #3
#
# Your names and user ids:
#
# (Based on skeleton code by D. Crandall)
#


import random
import math
import numpy as np

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
# 

class Solver:

    def __init__(self):
        self.pos_probabilities = {}
        self.transition_probabilities_hmm, self.transition_probabilities_complex = {}, {} 
        self.emission_probabilities = {}
        self.start_probabilities = {}
        self.parts_of_speech = []

    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def posterior(self, model, sentence, label):
        if model == "Simple":
            for i in range(len(sentence)):
                return math.log(self.pos_probabilities[label[i]]) + math.log(self.emission_probabilities[label[i]].get(sentence[i], 1e-8))
        elif model == "HMM":
            p = 0
            for i in range(len(sentence)):
                p += math.log(self.emission_probabilities[label[i]].get(sentence[i], 1e-8))
                if i == 0:
                    p += math.log(1e-8)
                else:
                    p += math.log(self.transition_probabilities_hmm[label[i-1]].get(label[i], 1e-8))
            return p
        elif model == "Complex":
            p = math.log(self.start_probabilities[label[0]])

            if (len(label) >= 2):
                for i in range(len(sentence) - 1):
                    p += math.log(self.transition_probabilities_hmm[label[i]].get(label[i+1], 1e-8))

            for i in range(len(sentence)):    
                p += math.log(self.emission_probabilities[label[i]].get(sentence[i], 1e-8))         

            for i in range(len(sentence) - 2):
                p += math.log(self.transition_probabilities_complex.get(label[i+2]+"->"+label[i+1]+label[i], 1e-8))

            return p 
        else:
            print("Unknown algo!")

    # Do the training!
    #
    def train(self, data):
        t = 0 
        with open('bc.train', 'r') as f:
            for sentence in f:
                i = 0
                words = sentence.lower().split(' ')
                for word in words:
                    if i == 1:
                        self.start_probabilities.setdefault(word, 0)
                        self.start_probabilities[word] += 1 
                        t += 1 
                    if i % 2 == 0:
                        w = word 
                    else:
                        if word not in self.emission_probabilities:
                            self.parts_of_speech.append(word)
                        self.emission_probabilities.setdefault(word, {})
                        self.emission_probabilities[word].setdefault(w, 0)
                        self.emission_probabilities[word][w] += 1
                    i += 1 
                
        for key, value in self.start_probabilities.items():
            self.start_probabilities[key] = value / t 

        wi_si = []
        for key, value in self.emission_probabilities.items():
            total = 0
            for k, v in value.items():
                total += v
            wi_si.append(total)

        for i, (key, value) in enumerate(self.emission_probabilities.items()):
            for val in value:
                self.emission_probabilities[key][val] /= wi_si[i]
		
        total = 0
        for index, i in enumerate(data):
            for j in i:
                prev_k = None
                if index % 2==0:
                    for k in j:
                        self.pos_probabilities.setdefault(k, 0)
                        self.transition_probabilities_hmm.setdefault(k, {})
                        self.pos_probabilities[k] += 1
                        total += 1

                        if prev_k:
                            self.transition_probabilities_hmm[prev_k].setdefault(k, 0)
                            self.transition_probabilities_hmm[prev_k][k] += 1

                        prev_k = k 

        for k in self.pos_probabilities:
            self.pos_probabilities[k] /= total

        si = []
        for i in self.transition_probabilities_hmm:
            total = 0
            for k in self.transition_probabilities_hmm[i]:
                total += self.transition_probabilities_hmm[i][k]
            si.append(total)

        for index, i in enumerate(self.transition_probabilities_hmm):
            for k in self.transition_probabilities_hmm[i]:
                self.transition_probabilities_hmm[i][k] /= si[index]

        # transition probabilities
        for line in data:
          for i in range(len(line[1])-2):
                key = line[1][i+2] +"->"+ line[1][i+1] + line[1][i]
                self.transition_probabilities_complex.setdefault(key, 0)
                self.transition_probabilities_complex[key] += 1

        denominator = {}
        for line in data:
            for i in range(len(line[1])-1):
                key = line[1][i+1] + line[1][i]
                denominator.setdefault(key, 0)
                denominator[key] += 1

        for key in self.transition_probabilities_complex.keys():
            _, k = key.split("->")
            self.transition_probabilities_complex[key] /= denominator[k]

        self.parts_of_speech.remove('')

    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence):
        parts_of_speech = [] 
        for word in sentence:
            max_i = 0 
            pos = 'noun'
            for i in self.emission_probabilities:
                if word in self.emission_probabilities[i] and max_i < self.emission_probabilities[i][word]:
                    max_i = self.emission_probabilities[i][word]
                    pos = i 
            parts_of_speech.append(pos) 
        return parts_of_speech 

    def hmm_viterbi(self, sentence):
        # ---
        # reference: https://en.wikipedia.org/wiki/Viterbi_algorithm
        V = [{}]
        for pos in self.parts_of_speech:
            V[0][pos] = {"prob": self.start_probabilities[pos] * self.emission_probabilities[pos].get(sentence[0], 0), "prev":None}
        
        flag = True
        for i in V[0]:
            if V[0][i]['prob']:
                flag = False
        if flag:
            V[0]['noun']['prob'] = 1

        for x in range(1, len(sentence)):
            V.append({})
            flag = True
            for pos in self.parts_of_speech:
                max_pos_probability = V[x-1][self.parts_of_speech[0]]["prob"] * self.transition_probabilities_hmm[self.parts_of_speech[0]][pos]
                prev_pos_selected = self.parts_of_speech[0]
                for prev_pos in self.parts_of_speech[1:]:
                    prob = V[x-1][prev_pos]["prob"] * self.transition_probabilities_hmm[prev_pos].get(pos, 0)

                    if prob > max_pos_probability:
                        max_pos_probability, prev_pos_selected = prob, prev_pos
				
                max_probability = max_pos_probability * self.emission_probabilities[pos].get(sentence[x], 0)
                V[x][pos] = {"prob": max_probability, "prev": prev_pos_selected}

            for i in V[x]:
                if V[x][i]['prob']:
                    flag = False

            if flag:
                V[x]['noun']['prob'] = 1

        opt = []
        max_probability = max(value["prob"] for value in V[-1].values())
        previous = None

        for st, data in V[-1].items():
            if data["prob"] == max_probability:
                opt.append(st)
                previous = st
                break

        for t in range(len(V) - 2, -1, -1):
            opt.insert(0, V[t+1][previous]["prev"])
            previous = V[t+1][previous]["prev"]
        # ---
        return opt
            
    def complex_mcmc(self, sentence):
        burnin_period = 100 
        iterations = 500 
        samples = [] 
        samples_to_check = [] 
        counter = 0 
        sample = ["noun"] * len(sentence)

        for i in range(iterations):
            final_sample = [] 
            for j in range(len(sentence)):
                word = sentence[j]
                probability = [0] * len(self.parts_of_speech) 

                for k in range(len(self.parts_of_speech)):
                    current_probability = self.parts_of_speech[k]
                    start_probability = self.start_probabilities[self.parts_of_speech[k]]
                    
                    value = self.emission_probabilities.get(self.parts_of_speech[k], None)
                    if value:
                        probability_1 = value.get(word, 1e-8)
                    else:
                        probability_1 = 1e-8 
                    
                    if j != 0:
                        value = self.transition_probabilities_hmm.get(sample[j-1], None)
                        if value:
                            probability_2 = value.get(current_probability, 1e-8)
                        else:
                            probability_2 = 1e-8 

                        key = self.parts_of_speech[k] + "->" + sample[j-1] + sample[j-2]
                        probability_3 = self.transition_probabilities_complex.get(key, 1e-8)

                        if j != len(sentence) - 1: 
                            value = self.transition_probabilities_hmm.get(sample[j+1], None)
                            if key: 
                                probability_4 = value.get(self.parts_of_speech[k], 1e-8)
                            else:
                                probability_4 = 1e-8 

                            if j != len(sentence) - 2:
                                key = sample[j+2] + "->" + sample[j+1] + self.parts_of_speech[k]
                                probability_5 = self.transition_probabilities_complex.get(key, 1e-8)
                
                    if j == 0:
                        probability[k] = start_probability * probability_1 
                    elif j == len(sentence) - 1:
                        probability[k] = probability_1 * probability_2 * probability_3 
                    elif j == len(sentence) - 2:
                        probability[k] = probability_1 * probability_2 * probability_3 * probability_4 
                    else:
                        probability[k] = probability_1 * probability_2 * probability_3 * probability_4 * probability_5 

                probability_sum = sum(probability)
                for l in range(len(probability)):
                    probability[l] /= probability_sum
                final_sample.append(np.random.choice(self.parts_of_speech, 1, p=probability)[0])
            sample = final_sample

            if i > burnin_period:
                samples.append(sample)
                samples_to_check.append(sample)
                counter += 1
                if counter > 5: 
                    samples_to_check.pop(0)
                    flag = True
                    for i in range(1, len(samples_to_check)):
                        if samples_to_check[0] != samples_to_check[i]:
                            flag = False
                            break
                    if flag:
                        break

        return samples[len(samples)-1]

    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        else:
            print("Unknown algo!")

