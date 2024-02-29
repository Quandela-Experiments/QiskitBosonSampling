from perceval import BSDistribution


def TVD(bsd1, bsd2):

    s = 0

    bsd1.normalize()
    bsd2.normalize()
    assert isinstance(bsd1, BSDistribution) and isinstance(bsd2, BSDistribution)
    for state in bsd1.keys():
        s += abs(bsd1[state]-bsd2[state])

    for state in bsd2.keys():
        if state not in bsd1.keys():
            s += bsd2[state]
    return s
