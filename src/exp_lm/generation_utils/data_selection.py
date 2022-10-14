import math
import logging
import difflib
import Levenshtein
import collections
import pandas as pd

from scipy import spatial
from scipy.special import softmax
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Consistency():
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def filter(self, input_phrase, para_phrases, consistency_threshold=0, device='cpu'):
        self.model = self.model.to(device)

        inputs = self.tokenizer(input_phrase, return_tensors='pt')
        input_ids, attention_mask = inputs.input_ids.to(device), inputs.attention_mask.to(device)
        encoder_outputs = self.model.t5.encoder(input_ids, attention_mask)
        prediction = self.model.style_classification(encoder_outputs[0], attention_mask)
        probs = prediction['probs'][0].detach().cpu().numpy()
        input_tag = probs.argmax(-1)

        top_consistency_phrases = []
        for para_phrase in para_phrases:
            inputs = self.tokenizer(para_phrase, return_tensors='pt')
            input_ids, attention_mask = inputs.input_ids.to(device), inputs.attention_mask.to(device)
            encoder_outputs = self.model.t5.encoder(input_ids, attention_mask)
            prediction = self.model.style_classification(encoder_outputs[0], attention_mask)
            probs = prediction['probs'][0].detach().cpu().numpy()
            consistency_score = probs[1 - input_tag] # LABEL_0 = Formal, LABEL_1 = Informal
            if consistency_score >= consistency_threshold:
                top_consistency_phrases.append(para_phrase)
 
        return top_consistency_phrases
    
    def score(self, input_phrase, para_phrases, consistency_threshold=0, device='cpu'):
        self.model = self.model.to(device)

        inputs = self.tokenizer(input_phrase, return_tensors='pt')
        input_ids, attention_mask = inputs.input_ids.to(device), inputs.attention_mask.to(device)
        encoder_outputs = self.model.t5.encoder(input_ids, attention_mask)
        prediction = self.model.style_classification(encoder_outputs[0], attention_mask)
        probs = prediction['probs'][0].detach().cpu().numpy()
        input_tag = probs.argmax(-1)

        consistency_scores = collections.Counter()
        for para_phrase in para_phrases:
            inputs = self.tokenizer(para_phrase, return_tensors='pt')
            input_ids, attention_mask = inputs.input_ids.to(device), inputs.attention_mask.to(device)
            encoder_outputs = self.model.t5.encoder(input_ids, attention_mask)
            prediction = self.model.style_classification(encoder_outputs[0], attention_mask)
            probs = prediction['probs'][0].detach().cpu().numpy()
            consistency_score = probs[1 - input_tag] # LABEL_0 = Formal, LABEL_1 = Informal
            if consistency_score >= consistency_threshold:
                consistency_scores[para_phrase] = consistency_score

        return consistency_scores


class Adequacy():
  
    def __init__(self, model_tag='prithivida/parrot_adequacy_model'):
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(model_tag)
        self.tokenizer = AutoTokenizer.from_pretrained(model_tag)

    def filter(self, input_phrase, para_phrases, adequacy_threshold=0, device='cpu'):
        self.nli_model = self.nli_model.to(device)
        top_adequacy_phrases = []
        for para_phrase in para_phrases:
            x = self.tokenizer.encode(input_phrase, para_phrase, return_tensors='pt',truncation='only_first')
            logits = self.nli_model(x.to(device))[0]
            # we throw away "neutral" (dim 1) and take the probability of "entailment" (2) as the adequacy score
            entail_contradiction_logits = logits[:,[0,2]]
            probs = entail_contradiction_logits.softmax(dim=1)
            prob_label_is_true = probs[:,1]
            adequacy_score = prob_label_is_true[0].item()
            if adequacy_score >= adequacy_threshold:
                top_adequacy_phrases.append(para_phrase)
        return top_adequacy_phrases

    def score(self, input_phrase, para_phrases, adequacy_threshold=0, device='cpu'):
        self.nli_model = self.nli_model.to(device)
        adequacy_scores = collections.Counter()
        for para_phrase in para_phrases:
            x = self.tokenizer.encode(input_phrase, para_phrase, return_tensors='pt',truncation='only_first')
            logits = self.nli_model(x.to(device))[0]
            # we throw away "neutral" (dim 1) and take the probability of "entailment" (2) as the adequacy score
            entail_contradiction_logits = logits[:,[0,2]]
            probs = entail_contradiction_logits.softmax(dim=1)
            prob_label_is_true = probs[:,1]
            adequacy_score = prob_label_is_true[0].item()
            if adequacy_score >= adequacy_threshold:
                adequacy_scores[para_phrase] = adequacy_score
        return adequacy_scores


class Fluency():
    
    def __init__(self, model_tag='prithivida/parrot_fluency_model'):
        self.cola_model = AutoModelForSequenceClassification.from_pretrained(model_tag, num_labels=2)
        self.cola_tokenizer = AutoTokenizer.from_pretrained(model_tag)

    def filter(self, para_phrases, fluency_threshold=0, device='cpu'):
        self.cola_model = self.cola_model.to(device)
        top_fluent_phrases = []
        for para_phrase in para_phrases:
            input_ids = self.cola_tokenizer("Sentence: " + para_phrase, return_tensors='pt', truncation=True).to(device)
            prediction = self.cola_model(**input_ids)
            scores = prediction[0][0].detach().cpu().numpy()
            scores = softmax(scores)
            fluency_score = scores[1] # LABEL_0 = Bad Fluency, LABEL_1 = Good Fluency
            if fluency_score >= fluency_threshold:
                top_fluent_phrases.append(para_phrase)
        return top_fluent_phrases

    def score(self, para_phrases, fluency_threshold=0, device='cpu'):
        self.cola_model = self.cola_model.to(device)
        fluency_scores = collections.Counter()
        for para_phrase in para_phrases:
            input_ids = self.cola_tokenizer("Sentence: " + para_phrase, return_tensors='pt', truncation=True).to(device)
            prediction = self.cola_model(**input_ids)
            scores = prediction[0][0].detach().cpu().numpy()
            scores = softmax(scores)
            fluency_score = scores[1] # LABEL_0 = Bad Fluency, LABEL_1 = Good Fluency
            if fluency_score >= fluency_threshold:
                fluency_scores[para_phrase] = fluency_score
        return fluency_scores


class Diversity():

    def __init__(self, model_tag='paraphrase-distilroberta-base-v2'):
        logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
        self.diversity_model = SentenceTransformer(model_tag)

    def rank(self, input_phrase, para_phrases, diversity_ranker='levenshtein', device='cpu'):
        if diversity_ranker == "levenshtein":
            return self.levenshtein_ranker(input_phrase, para_phrases)
        elif diversity_ranker == "euclidean":
            return self.euclidean_ranker(input_phrase, para_phrases, device)
        elif diversity_ranker == "diff":
            return self.diff_ranker(input_phrase, para_phrases)

    def euclidean_ranker(self, input_phrase, para_phrases, device='cpu'):
        self.diversity_model = self.diversity_model.to(device)
        diversity_scores = collections.Counter()
        outputs = []
        input_enc = self.diversity_model.encode(input_phrase.lower().to(device))
        for para_phrase in para_phrases:              
            paraphrase_enc = self.diversity_model.encode(para_phrase.lower().to(device))
            euclidean_distance = (spatial.distance.euclidean(input_enc, paraphrase_enc))
            outputs.append((para_phrase, euclidean_distance))
        df = pd.DataFrame(outputs, columns=['paraphrase', 'scores'])
        fields = []
        for col in df.columns:
            if col == "scores":
                tup = ([col], MinMaxScaler())
            else:  
                tup = ([col], None)
            fields.append(tup) 

        mapper = DataFrameMapper(fields, df_out=True)
        for _, row in mapper.fit_transform(df.copy()).iterrows():
            diversity_scores[row['paraphrase']] = row['scores'] / len(input_enc)
        return diversity_scores

    def levenshtein_ranker(self, input_phrase, para_phrases):
        diversity_scores = collections.Counter()
        for para_phrase in para_phrases:
            distance = Levenshtein.distance(input_phrase.lower(), para_phrase.lower())
            diversity_scores[para_phrase] = distance / len(para_phrase) if len(para_phrase) else -math.inf
        return diversity_scores
  
    def diff_ranker(self, input_phrase, para_phrases):
        differ = difflib.Differ()
        diversity_scores = collections.Counter()
        for para_phrase in para_phrases:
            diff = differ.compare(input_phrase.split(), para_phrase.split())
            count = 0
            for d in diff:
                if "+" in d or "-" in d:
                    count += 1
            diversity_scores[para_phrase] = count / len(input_phrase.split())
        return diversity_scores
