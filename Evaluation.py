import sys

from nltk.tokenize import RegexpTokenizer
from nltk.translate import bleu_score
import numpy as np

splitter = RegexpTokenizer('\w+') #\w seleziona tutto cio che è un carattere "di parola"" . il + concatena alla selezione più "char parola" consecutivi

if len(sys.argv) != 3:
    print("Missing arguments")
    exit(-1)
ref_path = sys.argv[1]
gen_path = sys.argv[2]

#Create reference tokens
ref = []
ref_file = open(ref_path,"r")
for l in ref_file:
    token = splitter.tokenize(l)
    if len(token)>1:  ref.append(token)
ref_file.close()
print(ref)

#Create generation tokens

gen = []
gen_file = open(gen_path,"r")
for l in gen_file:
    token = splitter.tokenize(l)
    if len(token)>1:  gen.append(token)
gen_file.close()
print(gen)

bleu = []
self_bleu = []
self_bleu_metodo2 = []
i = 0
for g in gen:
    bleu.append(bleu_score.sentence_bleu(ref,g, smoothing_function=bleu_score.SmoothingFunction().method1))

    self_bleu.append(bleu_score.sentence_bleu(np.delete(np.array(gen, dtype=object),i),g, smoothing_function=bleu_score.SmoothingFunction().method1))

    gen_x = gen[:]
    gen_x.remove(g)
    self_bleu_metodo2.append(bleu_score.sentence_bleu(gen_x,g, smoothing_function=bleu_score.SmoothingFunction().method1))
    i=i+1

print("Bleu score: %f\nSelf_Bleu score: %f(%f)"%(np.array(bleu).mean(), np.array(self_bleu).mean(), np.array(self_bleu_metodo2).mean()))
