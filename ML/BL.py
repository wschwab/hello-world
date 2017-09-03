from numpy.random.mtrand import dirichlet
from numpy import *
from sys import *

numstates = 5
alphabet = 5
ll_bound = 10.0

if len(argv) != 3:
    print "required input: trainfile testfile"
    assert(False)

train_file = argv[1]
test_file = argv[2]

def number(num):
    ​return float(num)
    # Normalize the values and make sure they sum to 1.0

def normalize(arr):
    sumarr = number(sum(arr))
    if sumarr != 0.0:
        for i in range(len(arr)):
            arr[i] = arr[i] / sumarr

# A probable and finite automaton model's state (I,F,S,T)
# I - an array of all the initial probabilities of the states
# F - an array of the final probabilities of the states
# S - this is the matrix of each state's symbols probabilities
# T - this is the 3d matrix of the transition probabilities

# This creates a model that has 0 probabilities
def emptymodel(numstates,alphabet):
    I = array([number(0.0)] * numstates)
    F = array([number(0.0)] * numstates)
    S = []
    for i in range(numstates):
        newrow = array([number(0.0)] * alphabet)
        S.append(newrow)
    T = []
    for i in range(alphabet):
        T.append([])
        for j in range(numstates):
            newrow = array([number(0.0)] * numstates)
            T[i].append(newrow)

    return (I,F,S,T)

# This creates a model that is fully connected and has random probabilities
def randommodel(numstates, alphabet):
    I = array(dirichlet([1] * numstates))
    F = array([0.0] * numstates)
    S = []
    # here F is the end of the string symbol
    for i in range(numstates):
        probs = dirichlet([1] * (alphabet + 1))
        newrow = array(probs[0:alphabet])
        normalize(newrow)
        S.append(newrow)
        F[i] = probs[alphabet]

    T = []
    for i in range(alphabet):
        T.append([])
        for j in range(numstates):
            newrow = array(dirichlet([1] * numstates))
            T[i].append(newrow)

    return (I,F,S,T)

# This calculates the string probabilities using recursion
def computeprobabilityrecursion((I,F,S,T),sequence,index,state,DPdict):
    # Probability = P (final)
    if index == len(sequence):
         DPdict[tuple([state])] = F[state]
         return F[state]

    # Return the hashed result
    if DPdict.has_key(tuple([state] + sequence[index:len(sequence)])):
        return DPdict[tuple([state] + sequence[index:len(sequence)])]

    # For each next state s possible:
    # Probability = P(symbol) * P(transition to s) * P(future)
    symb_prob  = S[state][sequence[index]]
    final_prob = F[state]
    prob  = number(0.0)
    for nextstate in range(len(T[sequence[index]][state])):
        if T[sequence[index]][state][nextstate] > 0.0:
            trans_prob = T[sequence[index]][state][nextstate]
            future_prob = computeprobabilityrecursion((I,F,S,T),sequence,index+1,nextstate,DPdict)
            prob = prob + (number(1.0)-final_prob) * symb_prob * trans_prob * future_prob

        # Hash the result
        DPdict[tuple([state] + sequence[index:len(sequence)])] = prob
        return prob

# Calculates the string probabilities forwards using recursion
def computeprobability((I,F,S,T),sequence,DPdict):
    result = number(0.0)
    for state in range(len(I)):
        if I[state] > 0.0:
            result = result + I[state] * computeprobabilityrecursion((I,F,S,T),sequence,0,state,DPdict)
            return result

# Calculates all of the probabilities from an example list
def computeprobabilities((I,F,S,T),sett):
    probs = []
    DPdict = dict()
    for sequence in sett:
        probs.append(computeprobability((I,F,S,T),sequence,DPdict))

    return probs

# Calculates the string probabilities backward using recursion
def computeprobabilityrecursionreverse((I,F,S,T),sequence,index,state,DPdict):
    # Probability = P(initial)
    if index == 0:
        DPdict[tuple([state])] = I[state]
        return I[state]

    # Return the hashed result
    if DPdict.has_key(tuple([state] + sequence[0:index])):
        return DPdict[tuple([state] + sequence[0:index])] # For each possible previous state s: # Probability += P(symbol) * P(transition from s) * P(past) prob = number(0.0) for prevstate in range(len(I)): if T[sequence[index-1]][prevstate][state] > 0.0: final_prob = F[prevstate] symb_prob  = S[prevstate][sequence[index-1]] trans_prob = T[sequence[index-1]][prevstate][state] past_prob  = computeprobabilityrecursionreverse((I,F,S,T),sequence,index-1,prevstate,DPdict) prob = prob + ((number(1.0)-final_prob) * symb_prob * trans_prob * past_prob) # Hash the result DPdict[tuple([state] + sequence[0:index])] = prob return prob # Computes string probabilities backwards using a recursion def computeprobabilityreverse((I,F,S,T),sequence,DPdict): result = number(0.0) # For every final state f: # Probability += P(end in f) * P(past) for state in range(len(I)): result = result + F[state] * computeprobabilityrecursionreverse((I,F,S,T),sequence,len(sequence),state,DPdict) return result # Computes all probabilities from an example list def computeprobabilitiesreverse((I,F,S,T),sett): probs = [] DPdict = dict() for sequence in sett: probs.append(computeprobabilityreverse((I,F,S,T),sequence,DPdict)) return probs def iterateEM((I,F,S,T),sett): backward = dict() probs = [] for sequence in sett: probs.append(computeprobability((I,F,S,T),sequence,backward)) # backward = P(s|start(q)) forward = dict() for sequence in sett: computeprobabilityreverse((I,F,S,T),sequence,forward) # forward = P(s,end(q)) (Inew,Fnew,Snew,Tnew) = emptymodel(numstates,alphabet) # P(I(q)|s) =  P(I(q),s)/P(s) # P(I(q)|s) =  P(I(q))*P(s|start(q))/P(s) for state in range(len(I)): for seq in range(len(sett)): sequence = sett[seq] prob = probs[seq] key = tuple([state] + sequence) if backward.has_key(key): Inew[state] = Inew[state] + ((I[state] * backward[key]) / prob) normalize(Inew) # P(F(q)|s) =  P(F(q),s)/P(s) # P(F(q)|s) =  P(end(q),s)*P(F(q))/P(s) for state in range(len(I)): for seq in range(len(sett)): sequence = sett[seq] prob = probs[seq] key = tuple([state] + sequence) if forward.has_key(key): Fnew[state] = Fnew[state] + ((F[state] * forward[key]) / prob) # P(S(q,a)|s) =  P(S(q,a),s)/P(s) # P(S(q,a)|s) =  P(end(q),S(q,a),tail(q))/P(s) # P(S(q,a)|s) =  P(end(q),head(s))*P(tail(s)|start(q))/P(s) Stotal = number(0.0) for seq in range(len(sett)): sequence = sett[seq] prob = probs[seq] for index in range(len(sequence)): key = tuple([state] + sequence[0:index]) if forward.has_key(key): key2 = tuple([state] + sequence[index:len(sequence)]) if backward.has_key(key2): symprob = forward[key] * backward[key2] Snew[state][sequence[index]] = Snew[state][sequence[index]] + (symprob / prob) if Fnew[state] != 0.0: Fnew[state] = Fnew[state] / (Fnew[state] + sum(Snew[state])) normalize(Snew[state]) for state in range(len(I)): for seq in range(len(sett)): sequence = sett[seq] prob = probs[seq] for index in range(len(sequence)): key1 = tuple([state] + sequence[0:index]) if forward.has_key(key1): for state2 in range(len(I)): key2 = tuple([state2] + sequence[(index+1):len(sequence)]) if backward.has_key(key2): transprob = (number(1.0) - F[state]) * S[state][sequence[index]] * T[sequence[index]][state][state2] transprob = forward[key1] * transprob * backward[key2] Tnew[sequence[index]][state][state2] = Tnew[sequence[index]][state][state2] + (transprob / prob) for a in range(alphabet): for state in range(len(I)): normalize(Tnew[a][state]) return (Inew,Fnew,Snew,Tnew) def loglikelihood(probs): sumt = number(0.0) log2 = log10(number(2.0)) for index in range(len(probs)): term = log10(probs[index]) / log2 sumt = sumt + term return sumt def readset(f): sett = [] line = f.readline() l = line.split(" ") num_strings = int(l[0]) alphabet_size = int(l[1]) for n in range(num_strings): line = f.readline() l = line.split(" ") sett = sett + [[int(i) for i in l[1:len(l)]]] return alphabet_size, sett def writeprobs(probs,f): f.write(str(len(probs)) + "\n") for i in range(len(probs)): f.write(str(probs[i]) + "\n") alphabet, train = readset(open(train_file,"r")) alphabet, test = readset(open(test_file,"r")) model = randommodel(numstates,alphabet) print "loglikelihood:", loglikelihood(computeprobabilities(model,train+test)) prev = -1.0 ll = -1.0 while prev == -1.0 or ll - prev > ll_bound: prev = ll m = iterateEM(model,train+test) probs = computeprobabilities(m,train+test) ll = loglikelihood(probs) print "loglikelihood:", ll model = m writeprobs(computeprobabilities(m,test),open(test_file+".bm","w")) Now that we know more about unsupervised learning with the help of Markov models in Python, let’s test our knowledge with some real applications and exercises that can help us master data science. Applications
