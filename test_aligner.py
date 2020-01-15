from aligner import (
    BertAligner,
    RobertaAligner,
    GPT2Aligner,
    DistilBertAligner
)
import pytest

def load_sample_en_sents():
    s = ['the LIFE',
        'the LIFEST',
        'the LIFESTPHSESDF',
        'the LI FE ST',
        "I can't understand for the LIFE of me why we Aren't going home",
         "There is nothing I can say or do... that will <MAKE> me do what YOU want!!",
         "This ain't going to mess me up, Ain't it?",
         "It's tonsa fun in the whatve whatve-you-done U.K."
    ]

    return s

sentences = load_sample_en_sents()

@pytest.mark.paramtrize(["model_name", "alnr_class"],
                        [('bert-base-uncased', BertAligner),
                        ('bert-base-cased', BertAligner),
                         ('gpt2', GPT2Aligner),
                        ('roberta-base', RobertaAligner),
                        ('distilbert-base-uncased', DistilBertAligner)])
def test_aligner(model_name, alnr_class):
    """NOTE: Will be obsolete when the aligner is able to work with transformer auto model"""
    a = alnr_class.from_pretrained(model_name)

    for s in sentences:
        mtokens = [m['token'] for m in a.meta_tokenize(s)]
        tokens = a.tokenize(s)
        #model_id_tokens = a.convert_ids_to_tokens(a.encode(s))
        #print(model_id_tokens)
        assert tokens == mtokens, f"{tokens} \n {mtokens}"
