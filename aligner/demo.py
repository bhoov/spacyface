import torch
from aligner import RobertaAligner
from transformers import AutoModel
import matplotlib.pyplot as plt
import seaborn as sn

model_name = "roberta-base"
sentence = "Failure is an option here. If things are not failing, you are not innovating enough"
layer = 8
heads = [7]

alnr = RobertaAligner.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_attentions=True)
model.eval() # Remove DropOut effect

model_input, meta_info = alnr.sentence_to_input(sentence)

_, _, atts = model(**model_input)

to_show = atts[layer][0][heads].mean(0)[1:-1, 1:-1] # Don't show special tokens for Roberta Model

deps = [t.dep for t in meta_info[1:-1]]
poss = [t.pos for t in meta_info[1:-1]]

plt.figure()
sn.set(font_scale=1.5
sn.heatmap(to_show.detach().numpy(), xticklabels=deps, yticklabels=deps)
