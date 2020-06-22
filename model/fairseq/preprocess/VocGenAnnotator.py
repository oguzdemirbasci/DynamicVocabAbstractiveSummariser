import torch 
from models.voc_gen_models import VocGenerator
from data.voc_gen_data import Corpus

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'could', 'who', 'would', 'may'])#'go','come', 'become', 'be'
stop_words.remove('own')

class VocGenAnnotator:

    def __init__(self, vocGenModel, K, hidden_dimension = None, out_features = 0, voc_features = 0):
        assert model_dimension is not None
        assert out_features != 0
        assert voc_features != 0

        self.loadVocGen(vocGenModel, model_dimension, out_features, voc_features)
        self.K = K

    def loadVocGen(self, vocGenFile, hidden_dimension, out_features, voc_features):
        self.vocGen = VocGenerator(hidden_dimension, out_features, voc_features)
        self.vocGen.load_state_dict(torch.load(vocGenFile))
        self.vocGen.cuda()
        self.vocGen.eval()