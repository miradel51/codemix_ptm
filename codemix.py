import random
random.seed(42)

from nltk.tokenize import word_tokenize

S = []
T = []

fsrc = open('raw_data/Tatoeba.de-en.en', 'r', encoding='utf-8')
ftgt = open('raw_data/Tatoeba.de-en.de', 'r', encoding='utf-8')

for i, (__s, __t) in enumerate(zip(fsrc, ftgt)):
    s = __s.strip()
    t = __t.strip()
    if (len(s) == 0) or (len(t) == 0):
        continue
    S.append(word_tokenize(s))
    T.append(word_tokenize(t))
    if i % 1000 == 0:
        print(i)

fsrc.close()
ftgt.close()

dict_ende = {}

fdict = open('raw_data/en-de.txt', 'r', encoding='utf-8')

for __x in fdict:
    s, t = __x.strip().split(' ')
    if not (s in dict_ende):
        dict_ende[s] = []
    dict_ende[s].append(t)

fdict.close()

dict_deen = {}

fdict = open('raw_data/de-en.txt', 'r', encoding='utf-8')

for __x in fdict:
    s, t = __x.strip().split(' ')
    if not (s in dict_deen):
        dict_deen[s] = []
    dict_deen[s].append(t)

fdict.close()

foutsrc = open('codemix_data/Tatoeba.en', 'w', encoding='utf-8')
fouttgt = open('codemix_data/Tatoeba.de', 'w', encoding='utf-8')

for s, t in zip(S, T):
    len_s = len(s)
    n = random.randint(0, (len_s - 1) // 2)
    A = random.sample(list(range(0, len_s)), n)
    for i in A:
        ss = s[i].lower()
        if ss in dict_ende:
            tt = random.choice(dict_ende[ss])
            if i == 0:
                tt = tt[0].upper() + tt[1 : ]
            s[i] = tt
    len_t = len(t)
    m = random.randint(0, (len_t - 1) // 2)
    B = random.sample(list(range(0, len_t)), m)
    for i in B:
        tt = t[i].lower()
        if tt in dict_deen:
            ss = random.choice(dict_deen[tt])
            if i == 0:
                ss = ss[0].upper() + ss[1 : ]
            t[i] = ss
    foutsrc.write(' '.join(s) + '\n')
    fouttgt.write(' '.join(t) + '\n')

foutsrc.close()
fouttgt.close()
