from __future__ import print_function

import sys
import numpy as np
import json
from scipy.stats.mstats import gmean


try:
    from nltk.util import ngrams
except:
    print('fail to import nltk package')



# need coco caption api
coco_caption_api_path = '/data2/ynli/datasets/coco-caption/pycocoevalcap'

try:
    sys.path.append(coco_caption_api_path)

    from tokenizer.ptbtokenizer import PTBTokenizer
    from bleu.bleu import Bleu
    from meteor.meteor import Meteor
    from rouge.rouge import Rouge
    from cider.cider import Cider

except Exception as error:
    print(error)
    print('fail to import coco-caption api from %s' % coco_caption_api_path)



class VDQG():
    '''
    This is a class to wrap VDQG annotation data and provide evaluation tools.

    -version: Version of dataset
    -annotation: (dict) all annotations. Each tuple (s_id, anno) is a sample's annotation, where
        -s_id: (str) unique sample id
        -anno: (dict) annotation containing:
            -id: (str) s_id
            -object: (list) object list (2 objects)
                -VG_image_id: (str) visual genome image id
                -VG_object_id: (str) visual genome object id
                -bbox: (list) object bounding box [x1, y1, x2, y2]
            -question: (list of str) question annotations
            -question_label: (list of int) question labels, where "-1" means negative, "1" means weak positive and "2" means strong positive
    -hard_set_ids: (list of str) the id list of hard samples
    '''

    def __init__(self, fn = './annotations/annotation.json'):
        '''
        load VDQG annotations 

        '''
        try:
            with open(fn, 'r') as f:
                data = json.load(f)

                self.version = data['version']
                self.annotation = data['annotation']
                self.hard_set_ids = data['hard_set_ids']
        except:
            print('fail to load annotation from %s' % fn)


    def load_human_result(self, fn = './annotations/human_result.json', random = False):
        '''
        load human written questions. This is for evaluation of human performance.

        input:
            fn      : human result file.
            random  : each sample has 2-3 human written questions. if random = False, use top-1 questions,
                      if random = True, choose random questions.

        output:
            rst     : a dict. Each key is a sample id, and each value is a string, which is a generated (by human) question.
                      When evaluating the result of a model using eval_delta_bleu or eval_coco, the rst should have the same format.
        '''

        try:
            with open(fn, 'r') as f:
                ques_human = json.load(f)
        except:
            print('fail to load human written questions from %s' % fn)


        if not random:
            rst_human = {s_id: v['question'][0] for s_id, v in ques_human.iteritems()}
        else:
            rst_human = {s_id: v['question'][v['random_id']] for s_id, v in ques_human.iteritems()}

        return rst_human



    def eval_delta_bleu(self, rst, ngram = 4, hard_set = False):
        
        '''
        evaluate generated questions using Delta-BLEU.

        input:
            rst      : see load_human_result
            ngram    : max gram length
            hard_set : if True, only use hard set

        output:
            d_bleu   : numpy.array with length "ngram", the n-th element is the deltaBLEU-n
        '''

        if hard_set:
            id_lst = self.hard_set_ids
        else:
            id_lst = self.annotation.keys()


        hypo_lst = [rst[s_id] for s_id in id_lst]


        # ref_lst[i] is a list of questions(str)
        ref_lst = [self.annotation[s_id]['question'] for s_id in id_lst]

        # score_lst[i] is a list of score(float), one for each reference.
        label2score = {
            2   : 1.0,  # strong positive
            1   : 0.5,  # weak positive
            -1  : -0.5,  # negative
        }

        score_lst = [[label2score[l] for l in self.annotation[s_id]['question_label']] for s_id in id_lst]

        
        # compute delta bleu
        small = 1e-8
        tiny = 1e-15


        num = len(hypo_lst)
        assert(len(ref_lst) == num)
        assert(len(score_lst) == num)

        up = np.zeros(ngram, dtype = np.float)
        down = np.zeros(ngram, dtype = np.float)
        len_r = 0
        len_h = 0
        for idx, (hypo, refs, scores) in enumerate(zip(hypo_lst, ref_lst, score_lst)):
            # match ngram for one sample
            hypo = hypo if isinstance(hypo, list) else hypo.lower().split()
            refs = [ref if isinstance(ref, list) else ref.lower().split() for ref in refs]
            for n in xrange(1, ngram + 1):
                max_score = max(scores)
                hypo_ngram = list(ngrams(hypo, n))
                refs_ngram = [list(ngrams(ref, n)) for ref in refs]

                for g in set(hypo_ngram):
                    c = hypo_ngram.count(g)
                    down[n-1] += max_score * c
                    match = [min(c, ref.count(g)) * s for s, ref in zip(scores, refs_ngram) if g in ref]
                    if match:
                        up[n-1] += max(match)
            len_h += len(hypo)
            len_rs = np.array([len(ref) for ref in refs])
            len_r += len_rs[np.argmin(np.abs(len_rs - len(hypo)))]

        up = np.maximum(up, tiny)
        pn = (up) / (down+small)
        BP = 1.0 if len_h > len_r else np.exp(1 - len_r/len_h)
        d_bleu = np.zeros(ngram, dtype = np.float)
        for n in xrange(0,ngram):
            d_bleu[n] = gmean(pn[0:n+1])

        d_bleu = BP * d_bleu

        return d_bleu



    def eval_coco(self, rst, hard_set = False):
        '''
        evaluate generated questions using multiple metrics provided by coco-caption api

        input:
            rst      : see load_human_result
            hard_set : if True, only use hard set
            
        '''

        if hard_set:
            id_lst = self.hard_set_ids
        else:
            id_lst = self.annotation.keys()


        # hypo_lst = [rst[s_id] for s_id in id_lst]


        # # ref_lst[i] is a list of questions(str)
        # ref_lst = [self.annotation[s_id]['question'] for s_id in id_lst]


        res = {idx: [{'image_id': idx, 'caption': rst[s_id]}] for idx, s_id in enumerate(id_lst)}

        gts = {}
        
        for idx, s_id in enumerate(id_lst):
            anno = self.annotation[s_id]
            gts[idx] = [{'image_id': idx, 'caption': q} for q, l in zip(anno['question'], anno['question_label']) if l in {1, 2}]

        tokenizer = PTBTokenizer()
        res = tokenizer.tokenize(res)
        gts = tokenizer.tokenize(gts)
        
        scorers = [(Bleu(4), 'bleu'), (Meteor(), 'meteor'), (Rouge(), 'rough')]
        
        for scorer, name in scorers:
            score, scores = scorer.compute_score(gts, res)
            print('==> %s' % name)
            print(score)

