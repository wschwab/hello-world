def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    for st in states:
        V[0][st] = {"prob": start_p[st] * emit_p[st][obs[0]], "prev": None}

    # When t > 0, run Viterbi
    for t in range(1, len(obs)):
        V.append({})
        for st in states:
            max_tr_prob = max(V[t-1][prev_st]["prob"] * trans_p[prev_st][st] for prev_st in states)
            for prev_st in states:
                if V[t-1][prev_st]["prob"] * trans_p[prev_st][st] == max_tr_prob:
                    max_prob = max_tr_prob * emit_p[st][obs[t]]
                    V[t][st] = {"prob": max_prob, "prev": prev_st}
                    break

    for line in dptable(V):
        print(line)
    opt = []

    # The highest likelihood
    max_prob = max(value["prob"] for value in V[-1].values)
    previous = None

    #Get the most likely state, as well as its backtrack
    for st, data in V[-1].items():
        if data["prob"] == max_prob:
            opt.append(st)
            previous = st
            break

    # Follow the backtrack until the first object
    for t in range(len(V) -2, -1, -1):
        opt.insert(0, V[t + 1][previous]["prev"])
        previous = V[t +1][previous]["prev"]

    print("The steps of states are" + ''.join(opt) + f"with the highest probability of {max_prob}")

def dptable(V):
    # Printing table of steps from the dict
    yield "".join(("%12d" %i) for i in range(len(V)))
    for state in V[0]:
        yield "%.7s:" %state + "".join("%.7s" %("%f" %v[state]["prob"] for v in V))
