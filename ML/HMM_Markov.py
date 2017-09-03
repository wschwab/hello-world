# This is a HMM Markov algorithm

def fwd_bkw(observations, states, start_prob, trans_prob, emm_prob, end_st):

    #This is the forward part of the algorithm
    print("Running Hidded Markov Machine")
    print("Initializing forward")
    fwd = []
    f_prev = {}
    for i, observation_i in enumerate(observations):
        f_curr = {}
        for st in states:
            if i == 0:
                prev_f_sum = start_prob[st]
            else:
                prev_f_sum = sum(f_prev[k] * trans_prob[k][st] for k in states)

            f_curr[st] = emm_prob[st][observation_i] * prev_f_sum

        fwd.append(f_curr)
        print(f"Adding {f_curr} to fwd")
        f_prev = f_curr
        print(f"{f_prev} is now f_prev")

    p_fwd = sum(f_curr[k] * trans_prob[k][end_st] for k in states)
    print(f"At the end of the forward loops, {fwd} is the forward list")
    print(f"And {p_fwd} is p_fwd")

    # This is the backward part of the algorithm
    print("Initializing backwards")
    bkw = []
    b_prev = {}
    for i, observation_i_plus in enumerate(reversed(observations[1:] + (None,))):
        b_curr = {}
        for st in states:
            if i == 0:
                b_curr[st] = trans_prob[st][end_st]
            else:
                b_curr[st] = sum(trans_prob[st][l] * emm_prob[l][observation_i_plus] * b_prev[l] for l in states)
        bkw.insert(0, b_curr)
        print(f"Inserting 0 and {b_curr} to bkw")
        b_prev = b_curr
        print(f"{b_prev} is now b_prev")

    p_bkw = sum(start_prob[l] * emm_prob[l][observations[0]] * b_curr[l] for l in states)
    print(f"At the end of the backward loops, {bkw} is the backwards list")
    print(f"And {p_bkw} is p_bkw")


    # Now the merge
    print("Initializing merge")
    posterior = []
    for i in range(len(observations)):
        posterior.append({st: fwd[i][st] * bkw[i][st] / p_fwd for st in states})

    assert p_fwd == p_bkw
    return fwd, bkw, posterior
