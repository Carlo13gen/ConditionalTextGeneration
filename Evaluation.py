import sys
import os
import nltk
nltk.download('averaged_perceptron_tagger')

from nltk.tokenize import RegexpTokenizer
from nltk.translate import bleu_score
from nltk import pos_tag
import numpy as np

splitter = RegexpTokenizer('\w+') #\w seleziona tutto cio che è un carattere "di parola"" . il + concatena alla selezione più "char parola" consecutivi

if len(sys.argv) < 3:
    print("Missing arguments")
    exit(-1)
ref_path = sys.argv[1]
gen_path = sys.argv[2]

out_file = ""
if len(sys.argv) == 9:
    out_file = sys.argv[3]
    lr = sys.argv[4]
    iter = sys.argv[5]
    temp = sys.argv[6]
    topk = sys.argv[7]
    pen = sys.argv[8]
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


ref_pos = []
for r in ref:
    ref_pos.append(pos_tag(r))

bleu = []
self_bleu = []
self_bleu_metodo2 = []
pos_bleu = []
  
i = 0
for g in gen:
    bleu.append(bleu_score.sentence_bleu(ref,g, smoothing_function=bleu_score.SmoothingFunction().method1))

    self_bleu.append(bleu_score.sentence_bleu(np.delete(np.array(gen, dtype=object),i),g, smoothing_function=bleu_score.SmoothingFunction().method1))

    gen_x = gen[:]
    gen_x.remove(g)
    self_bleu_metodo2.append(bleu_score.sentence_bleu(gen_x,g, smoothing_function=bleu_score.SmoothingFunction().method1))
    
    g_pos = pos_tag(g)
    pos_bleu.append(bleu_score.sentence_bleu(ref_pos,g_pos, smoothing_function=bleu_score.SmoothingFunction().method1))
    
    i=i+1

print("Bleu score: %f\nSelf_Bleu score: %f(%f) Pos_bleu score:%f"%(np.array(bleu).mean(), np.array(self_bleu).mean(), np.array(self_bleu_metodo2).mean(), np.array(pos_bleu).mean()))

if out_file != "":
    if os.path.exists(out_file):
        append_write = "a"
    else:
        append_write = "w"
    f_out = open(out_file, append_write)
    f_out.write("PARAMS: lr: %s, iterations: %s, temperature: %s, topk: %s, penalty: %s\n"%(lr,iter,temp,topk,pen))
    f_out.write("Bleu score: %f Self_Bleu score: %f(%f) Pos_bleu score:%f\n\n"%(np.array(bleu).mean(), np.array(self_bleu).mean(), np.array(self_bleu_metodo2).mean(), np.array(pos_bleu).mean()))
    f_out.close()
