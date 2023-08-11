from g2p_en import G2p
number=['1','2','3','4','5','6','7','8','9','0']


def countsyl(text):
    g2p = G2p()
    out = g2p(text)
    # print(out)
    phm=0
    for i in out:
        if i[-1] in number:
            phm=phm+1
    #print(phm)
    return phm

# if __name__ == '__main__':
#     text="Hello, how are you?"
#     print(countsyl(text))