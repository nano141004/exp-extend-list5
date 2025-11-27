import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from transformers import T5Tokenizer, T5ForConditionalGeneration, LlamaTokenizer
import pickle
import itertools
import time
import argparse
from tqdm import tqdm
import jsonlines
import json
from pprint import pprint
import pandas as pd
import torch
import math
import glob
import copy
import numpy as np
import pickle
from transformers import pipeline
import sys
from pathlib import Path
from beir_eval import run_rerank_eval
from FiDT5 import FiDT5
from deepspeed.profiling.flops_profiler import FlopsProfiler
import random

sys.setrecursionlimit(10**7)


def read_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def read_jsonl(path):
    data = []
    with jsonlines.open(path, 'r') as reader:
        for instance in reader:
            data.append(instance)
    return data

class Runner():
    def __init__(self, args):
        self.idx = 0
        self.imsi = []
        self.args = args
        try:
            self.tok = T5Tokenizer.from_pretrained(self.args.model_path, legacy=False)
        except:
            print(f"No tokenizer found for {self.args.model_path}. Backoffing from t5-base")
            self.tok = T5Tokenizer.from_pretrained('t5-base', legacy=False)
        self.test_file = read_jsonl(self.args.test_path)
        if self.args.shuffle:
            new = []
            for instance in self.test_file:
                bm25_res = instance['bm25_results']
                random.shuffle(bm25_res)
                instance['bm25_results'] = bm25_res
                new.append(instance)
            self.test_file = new
        #if self.args.debug:
        print(self.args.test_path)
        self.idx2tokid = self.tok.encode(' '.join([str(x) for x in range(1, self.args.listwise_k+1)]))[:-1]
        self.model = self.load_model()
        self.num_forward = 0

    def write_json_file(self, path, data):
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Writing to {path} done!")

    def write_jsonl_file(self, path, data):
        if self.args.measure_flops:
            self.prof.stop_profile()
            self.flops = self.prof.get_total_flops()
        else:
            self.flops = 0
        print(f"Flops: {self.flops}!")
        with jsonlines.open(path, 'w') as writer:
            writer.write_all(data)
        print(f"Writing to {path} done!")

    def load_model(self):
        start = time.time()
        print("Loading model..")
        if self.args.sub_mode in ['monot5', 'rankt5']:
            print(f"Loading baseline {self.args.sub_mode} model from {self.args.model_path}")
            model = T5ForConditionalGeneration.from_pretrained(self.args.model_path).to('cuda')#, torch_dtype=torch.float32)
        else:
            print(f"Loading fid model from {self.args.model_path}")
            model = FiDT5.from_pretrained(self.args.model_path, use_auth_token=True).to('cuda')
        end = time.time()
        print(f"Done! took {end-start} second")
        model.eval()
        if self.args.measure_flops:
            self.prof = FlopsProfiler(model)
            self.prof.start_profile()
        return model

    def make_input_tensors(self, texts):
        raw = self.tok(texts, return_tensors='pt',
                padding=self.args.padding, max_length=self.args.max_length,
                truncation=True).to('cuda')
        input_tensors = {'input_ids': raw['input_ids'].unsqueeze(0),
                'attention_mask': raw['attention_mask'].unsqueeze(0)}
        return input_tensors

    def make_listwise_text(self, question, ctxs):
        out = []
        for i in range(len(ctxs)):
            if self.args.sub_mode == 'monot5':
                text = f"Query: {question} Document: {ctxs[i]} Relevant:"
            elif self.args.sub_mode == 'rankt5':
                text = f"Query: {question} Document: {ctxs[i]}"
            elif self.args.sub_mode == 'duot5':
                text = f"Query: {question} Document0: {ctxs[0]} Document1: {ctxs[1]} Relevant:"
            else: # ours
                text = f"Query: {question}, Index: {i+1}, Context: {ctxs[i]}"
            out.append(text)
        return out

    def run_inference(self, input_tensors):
        if self.args.beam_size == -1:
            output = self.model.generate(**input_tensors,
                max_length = self.args.max_gen_length,
                return_dict_in_generate=True, output_scores=True)
        else:
            output = self.model.generate(**input_tensors, num_return_sequences=1,
                    num_beams=self.args.beam_size,
                max_length = self.args.max_gen_length,
                return_dict_in_generate=True, output_scores=True)
        self.num_forward += 1
        return output

    def remove_duplicates(self, indexes):
        out = []
        for x in indexes:
            if x not in out:
                out.append(x)
        return out

    def group2chunks(self, l, n=5):
        for i in range(0, len(l), n):
            yield l[i:i+n]

    def get_order_batch(self, question_batch, ctxs_batch):
        full_input_texts_batchwise = [self.make_listwise_text(q,c) for q,c in zip(question_batch, ctxs_batch)]
        raw_tensors_batchwise = [self.tok(x, padding=self.args.padding, return_tensors='pt',
            max_length = self.args.max_length, truncation=True) for x in full_input_texts_batchwise]
        batch_inputids = torch.stack([x['input_ids'] for x in raw_tensors_batchwise]).to('cuda')
        batch_attnmasks = torch.stack([x['attention_mask'] for x in raw_tensors_batchwise]).to('cuda')
        output = self.run_inference({'input_ids': batch_inputids, 'attention_mask': batch_attnmasks})
        del batch_inputids
        del batch_attnmasks
        gen_out = self.tok.batch_decode(output.sequences, skip_special_tokens=True)
        gen_out = [reversed(x.strip().split(' ')) for x in gen_out]
        processed = []
        for y in gen_out:
            try:
                processed.append([int(x) - 1 for x in y])
            except Exception as e:
                print(f"Error in gen_out. : {gen_out} / {e}")
                processed.append(list(range(self.args.listwise_k)))
        return processed

    def get_order(self, question, ctxs):
        full_input_texts = self.make_listwise_text(question, ctxs)
        input_tensors = self.make_input_tensors(full_input_texts)
        output = self.run_inference(input_tensors)
        gen_out = self.tok.batch_decode(output.sequences, skip_special_tokens=True)
        gen_out = gen_out[0].strip().split(' ')
        gen_out = reversed(gen_out)
        try:
            gen_out = [int(x)-1 for x in gen_out]
        except Exception as e:
            print(f"Error in gen_out.: {gen_out} / {e}")
            gen_out = [0,1,2,3,4]
        return gen_out

    def run_one_loop_batchwise(self, questions_batch, topk_ctxs_batch, full_list_idx_batch):
        bsize = len(questions_batch)
        assert len(questions_batch) == len(topk_ctxs_batch) == len(full_list_idx_batch)
        cur_idx_batch = [x[:] for x in full_list_idx_batch]
        for start_idx in tqdm(list(range(self.args.topk - self.args.listwise_k, 0, -1 * self.args.stride))):
            if self.args.verbose:
                print(f"In run_one_loop_batchwise, start_idx: {start_idx}")
            orig_idxs_batch = [x[start_idx:start_idx + self.args.listwise_k] for x in cur_idx_batch]
            cur_idx = cur_idx_batch[0]
            if self.args.verbose:
                print('before')
                print(cur_idx[:start_idx], '|||', cur_idx[start_idx:start_idx+5], '|||', cur_idx[start_idx+5:])
            ordered_relidxs_batch = self.get_order_batch(questions_batch, [[topk_ctxs_batch[b_i][i] for i in orig_idxs_batch[b_i]] for b_i in range(bsize)])
            ordered_idxs_batch = [[orig_idxs_batch[b_i][i] for i in ordered_relidxs_batch[b_i]] for b_i in range(bsize)]
            for b_i in range(bsize):
                for i in range(self.args.listwise_k):
                    cur_idx_batch[b_i][start_idx + i] = ordered_idxs_batch[b_i][i]
            if self.args.verbose:
                print(f"{orig_idxs_batch[0]} => {ordered_relidxs_batch[0]} ({ordered_idxs_batch[0]})")
                print('after')
                print(cur_idx_batch[0][:start_idx], '|||', cur_idx_batch[0][start_idx:start_idx+5], '|||', cur_idx_batch[0][start_idx+5:])
        if start_idx != 0:
            orig_idxs_batch = [x[:self.args.listwise_k] for x in cur_idx_batch]
            ordered_relidxs_batch = self.get_order_batch(questions_batch, [[topk_ctxs_batch[b_i][i] for i in orig_idxs_batch[b_i]] for b_i in range(bsize)])
            ordered_idxs_batch = [[orig_idxs_batch[b_i][i] for i in ordered_relidxs_batch[b_i]] for b_i in range(bsize)]
            for b_i in range(bsize):
                for i in range(self.args.listwise_k):
                    cur_idx_batch[b_i][i] = ordered_idxs_batch[b_i][i]
        return cur_idx_batch

    def run_one_loop(self, question, topk_ctxs, full_list_idx):
        cur_idx = full_list_idx[:]
        if self.args.verbose:
            print('&&&&&&&&&&&&&&&&  START  &&&&&&&&&&&&&&&&&&')
        for start_idx in list(range(len(full_list_idx)-5, 0, -1 * self.args.stride)):
            orig_idxs = cur_idx[start_idx:start_idx + 5]
            ordered_relidxs = self.get_order(question, [topk_ctxs[i] for i in orig_idxs])
            ordered_idxs = [orig_idxs[i] for i in ordered_relidxs]
            if self.args.verbose:
                print(cur_idx[:start_idx], '|||', cur_idx[start_idx:start_idx+5], '|||', cur_idx[start_idx+5:])
                print(f"{orig_idxs} => {ordered_idxs} ({ordered_relidxs})")
            for i in range(5):
                cur_idx[start_idx + i] = ordered_idxs[i]
            if self.args.verbose:
                print(cur_idx[:start_idx], '|||', cur_idx[start_idx:start_idx+5], '|||', cur_idx[start_idx+5:])
                print(f'-'*100)
            #print(f"changed after: {cur_idx[start_idx:]}")
        try:
            if start_idx == 0:
                a = 1
        except:
            import pdb; pdb.set_trace()
        if start_idx != 0:
            orig_idxs = cur_idx[:5]
            ordered_relidxs = self.get_order(question, [topk_ctxs[i] for i in orig_idxs])
            ordered_idxs = [orig_idxs[i] for i in ordered_relidxs]
            if self.args.verbose:
                print(f"Running last")
                print('|||', cur_idx[:5], '|||', cur_idx[start_idx+5:])
            for i in range(5):
                cur_idx[i] = ordered_idxs[i]
            if self.args.verbose:
                print(f"{orig_idxs} => {ordered_idxs} ({ordered_relidxs})")
                print('|||', cur_idx[:5], '|||', cur_idx[start_idx+5:])
                print("Done.")
            #print(f"start_idx: {start_idx}, orig_idxs: {orig_idxs}, changed: {cur_idx}")
        return cur_idx

    def get_full_order_in_one_loop(self, question, topk_ctxs, full_list_idx):
        ordered_relidxs = self.get_order(question, topk_ctxs)
        return ordered_relidxs

    def check_valid_list(self, full_list):
        for exc in self.global_exclude:
            while exc in full_list:
                exc_idx = full_list.index(exc)
                new_val = (exc + 1) % len(full_list)
                while new_val in self.global_exclude:
                    new_val = (new_val + 1) % len(full_list)
                full_list[exc_idx] = new_val
        return full_list

    def run_subeval(self, i, output):
        if i % 500 == 0:
            ndcg_10, string = run_rerank_eval(output, combined=True)
            print(f"Iter: {i}, ndcg@10: {ndcg_10}")
        return

    def get_top100_goldidx(self, instance):
        top100_pids = [x['pid'] for x in instance['bm25_results'][:self.args.topk]]
        top100_goldidx = []
        gold_pids = [x for x in instance['qrels'] if instance['qrels'][x] != 0]
        for pid in gold_pids:
            try:
                top100_goldidx.append(top100_pids.index(pid))
            except ValueError:
                continue
        return top100_goldidx

    def run_eval_ours(self):
        skip_idx = 0
        short_idx = 0
        normal_idx = 0
        cached_output = []
        print(f"Running first batchwise iteration..")
        if self.args.num_iter == -1:
            num_iter = int(np.ceil(self.args.rerank_topk / (self.args.listwise_k - self.args.stride)))
        else:
            num_iter = self.args.num_iter

        print(f"Num_iter: {num_iter}, listwise_k: {self.args.listwise_k}, stride: {self.args.stride}, bsize: {self.args.bsize}")

        exceptions = []
        for i, instance in tqdm(enumerate(self.test_file), total=len(self.test_file)):
            question = instance['q_text']
            topk_ctxs = [f"{x['title']} {x['text']}".strip() for x in instance['bm25_results']][:self.args.topk]
            # handling exceptions
            # (2) prepare for skipping those that don't have gold in topk(100)
            top100_goldidx = self.get_top100_goldidx(instance)
            if len(top100_goldidx) == 0 and self.args.skip_no_candidate:
                if self.args.verbose:
                    print('No gold in bm25 top100. skip this instance')
                exceptions.append({'i': i, 'question': question, 'topk_ctxs': topk_ctxs, 'goldidx': [], 'reranked_result': []})
                skip_idx += 1
                # (3) don't batch calculate those that have shorter n than 100
            elif len(topk_ctxs) < self.args.topk:
                exceptions.append({'i': i, 'question': question, 'topk_ctxs': topk_ctxs, 'goldidx': top100_goldidx, 'reranked_result': []})
                short_idx += 1
            else:
                normal_idx += 1
                temp = {'i': i, 'question': question, 'topk_ctxs': topk_ctxs[:self.args.topk], 'goldidx': top100_goldidx, 'reranked_result': []}
                cached_output.append(temp)
        print(f"[STATS] skip idx was {skip_idx}/{len(self.test_file)}, instance that has shorter ctx than {self.args.topk} was {short_idx}/{len(self.test_file)}, normal: {normal_idx}/{len(self.test_file)}")
        print(f"Running exceptions (len {len(exceptions)}) start.")
        for cache in tqdm(exceptions):
            top100_goldidx = cache['goldidx']
            topk_ctxs = cache['topk_ctxs']
            question = cache['question']
            if len(top100_goldidx) == 0 and self.args.skip_no_candidate:
                cache['reranked_result'] = list(range(len(topk_ctxs)))
            elif len(topk_ctxs) == 0:
                continue
            elif len(topk_ctxs) == 1:
                cache['reranked_result'] = [0]
            elif len(topk_ctxs) <= self.args.listwise_k:
                duplicated_list = list(range(len(topk_ctxs))) * 5
                duplicated_list = duplicated_list[:5]
                full_rank = self.get_full_order_in_one_loop(question, topk_ctxs, duplicated_list)
                # replace duplicated idx to orig idx
                replace_dict = {i: duplicated_list[i] for i in range(5)}
                full_rank = [replace_dict[i] for i in full_rank]
                shortened_list = []
                for x in full_rank:
                    if x not in shortened_list:
                        shortened_list.append(x)
                cache['reranked_result'] = shortened_list
            elif len(topk_ctxs) < self.args.topk: # bigger than listwise_k, but shorter than topk (100) - calculate them in non-batchwise manner.
                iter_idx = list(range(len(topk_ctxs)))
                for i in range(num_iter):
                    #print(f"iter: {i}")
                    iter_idx = self.run_one_loop(question, topk_ctxs, iter_idx)
                cache['reranked_result'] = iter_idx
            else: # should not come her
                raise Exception
        # to do: assert cache reranked result added in the original pointer value
        new_cached_output = []
        batch = []
        if len(cached_output) < self.args.bsize:
            print(f"total length {len(cached_output)} is smaller than bsize {self.args.bsize}! running all examples at once!:)")
        for i, cache in tqdm(enumerate(cached_output), total=len(cached_output)): # do batchwise caching
            assert len(cache['topk_ctxs']) == self.args.topk
            batch.append(cache)
            if len(batch) == self.args.bsize or i == (len(cached_output) - 1):
                batch_full_list_idx = [list(range(self.args.topk)) for _ in range(len(batch))]
                batch_questions = [x['question'] for x in batch]
                batch_topk_ctxs = [x['topk_ctxs'] for x in batch]
                for topk_iter in range(num_iter):
                    if self.args.verbose:
                        print(f"Running iter: {topk_iter}")
                    batch_full_list_idx = self.run_one_loop_batchwise(batch_questions, batch_topk_ctxs, batch_full_list_idx)
                for i in range(len(batch)):
                    batch[i]['reranked_result'] = batch_full_list_idx[i]
                new_cached_output += batch
                batch = []
        if len(new_cached_output) != len(cached_output):
            print(f"Sth wrong!")
            import pdb; pdb.set_trace()
        cached_output = exceptions + new_cached_output
        cached_output = sorted(cached_output, key=lambda x: x['i']) # rearrange cache in orig order
        temp = []
        if len(cached_output) != len(self.test_file):
            import pdb; pdb.set_trace()
        for instance, cache in tqdm(zip(self.test_file, cached_output), total=len(cached_output)):
            if instance['q_text'] != cache['question']:
                import pdb; pdb.set_trace()
            top100_goldidx = cache['goldidx']
            topk_ctxs = cache['topk_ctxs']
            question = cache['question']
            if len(top100_goldidx) == 0 and self.args.skip_no_candidate:
                if self.args.verbose:
                    print(f"No gold in candidate list. skipping.")
                temp.append(instance)
            elif len(topk_ctxs) == 0:
                temp.append(instance)
            elif len(topk_ctxs) == 1 or ('skip' in self.args.sub_mode and len(topk_ctxs) <= 10):
                if self.args.verbose:
                    print(f"Length of topk ctxs is 1 (or less than 10 in skip option). skipping.")
                temp.append(instance)
            else:
                full_rank = cache['reranked_result']
                reranked_instances = []
                for i, rank_id in enumerate(full_rank):
                    template = instance['bm25_results'][rank_id]
                    template['orig_bm25_score'] = template['bm25_score']
                    template['bm25_score'] = 100000 - i
                    reranked_instances.append(template)
                instance['bm25_results'] = reranked_instances
                temp.append(instance)
                #self.run_subeval(cache['i'], temp)
        print(f'%%%%%%%%%DONE%%%%%%%%%%%')
        self.write_jsonl_file(self.args.output_path, temp)

def run_reranker(args):
    module = Runner(args)
    if args.sub_mode in ['monot5', 'rankt5']:
        module.run_baseline()
    else:
        module.run_eval_ours()
    flops = module.flops
    num_forward = module.num_forward
    ndcg_10, string = run_rerank_eval(args.output_path)
    return ndcg_10, flops, num_forward

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='outputs/ans-large-cot-highlr/tfmr_20', type=str)
    parser.add_argument('--topk', default=100, type=int) #or 1000
    parser.add_argument('--beam_size', default=-1, type=int)
    parser.add_argument('--gold_dir', default='./', type=str)
    parser.add_argument('--dataname', required=True, type=str)
    parser.add_argument('--outname', required=True, type=str)
    parser.add_argument('--max_gen_length', default=256, type=int)
    parser.add_argument('--num_iter', default=-1, type=int)
    parser.add_argument('--bsize', default=128, type=int)
    parser.add_argument('--padding', default='max_length', type=str) # longest is recommended
    parser.add_argument('--listwise_k', default=20, type=int)
    parser.add_argument('--rerank_topk', default=10, type=int)
    parser.add_argument('--out_k', default=1, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--sub_mode', default='', type=str)
    parser.add_argument('--measure_flops', action='store_true')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--stride', default=2, type=int)
    parser.add_argument('--shuffle_local', action='store_true')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--top1000', action='store_true')
    parser.add_argument('--custom_bsize', action='store_true')
    parser.add_argument('--skip_no_candidate', action='store_true', help='skip instances with no gold qrels included at first-stage retrieval for faster inference')
    parser.add_argument('--skip_issubset', action='store_true', help='skip the rest of reranking when the gold qrels is a subset of reranked output')
    parser.add_argument('--size_3b', action='store_true', help='management option to reduce batch size for 3b models')
    args = parser.parse_args()
    res = {}
    random.seed(args.seed)
    if args.top1000:
        args.topk = 1000
        args.gold_dir = '../dataset/beir_bm25/final_fromindex_top1000'
        args.outname = args.outname + '_top1000'
    if args.shuffle:
        print(f"Shuffle input data with seed : {args.seed}")
    args.test_path = f"{args.gold_dir}/{args.dataname}.jsonl"
    # adjusting max length adaptively
    #data2len = {'msmarco': 256, 'dl19': 256, 'dl20': 256, 'trec-covid': 1024, 'nfcorpus': 1024,
    #        'nq': 512, 'hotpotqa': 512, 'fiqa': 1024, 'signal': 256, 'news': 1024, 'robust04': 1024,
    #        'arguana': 1024, 'webis-touche2020': 1024, 'quora': 256, 'dbpedia-entity': 256,
    #        'scidocs': 1024, 'fever': 512, 'climate-fever': 1024, 'msmarco_top1000': 256, 'cqadupstack': 512}
    data2len = {'msmarco': 256, 'dl19': 256, 'dl20': 256, 'trec-covid': 512, 'nfcorpus': 512,
            'bioasq': 512,
            'nq': 256, 'hotpotqa': 256, 'fiqa': 512, 'signal': 256, 'news': 1024, 'robust04': 1024,
            'arguana': 1024, 'touche': 1024, 'cqadupstack': 512, 'quora': 256, 'dbpedia-entity': 256,
            'scidocs': 512, 'fever': 256, 'climate-fever': 256, 'msmarco_top1000': 256,
            'scifact': 512}
    if 'cqadupstack' in args.dataname:
        max_length = data2len.get('cqadupstack')
    else:
        max_length = data2len.get(args.dataname)
    if args.sub_mode in ['monot5', 'rankt5']:
        args.output_path = f"./baseline_model_outputs/{args.sub_mode}/{args.model_path}/{args.outname}/{args.dataname}_output.jsonl"
        args.bsize = {256: 384*3, 512: 160*3, 1024: 40*3, 1280: 24*3}[max_length]
    elif 'Soyoung97' in args.model_path or 't5-base' == args.model_path:
        args.output_path = f'./outputs/{args.model_path}/{args.outname}/{args.dataname}_output.jsonl'
        args.max_gen_length = args.listwise_k + 3
    else:
        args.output_path = f"{args.model_path}/{args.outname}/{args.dataname}_output.jsonl"
        args.max_gen_length = args.listwise_k + 3

    if max_length == None:
        print(f"No mapping for dataname {args.dataname}!!!!!")
        raise Exception
    else:
        print(f"Max length: {max_length} for dataname: {args.dataname}")

    if 'predefined_5sort' in args.sub_mode:
        args.max_gen_length = args.listwise_k + 3
        if not args.custom_bsize:
            args.bsize = {256: 384, 512: 160, 1024: 40, 1280: 24}[max_length]
            #args.bsize = args.bsize * 2 # for H100 80G
            if args.size_3b:
                args.bsize = args.bsize // 12
            if args.listwise_k == 10:
                args.bsize = args.bsize // 2
            elif args.listwise_k != 5 and not args.measure_flops:
                print(f"bsize option for this {args.listwise_k} listwise_k is not implemented!")
                import pdb; pdb.set_trace()
        print(f"Adjusted batch size to {args.bsize}!")
    pprint(args)
    args.max_length = max_length
    Path(args.output_path).parent.mkdir(exist_ok=True, parents=True)
    start_time = time.time()
    scores, flops, num_forwards = run_reranker(args)
    res['flops'] = flops
    res['num_forwards'] = num_forwards
    res[args.output_path] = scores
    ndcg_at_10 = scores
    res['ndcg@10'] = ndcg_at_10
    end_time = time.time()
    res['time_duration'] = end_time - start_time
    if args.shuffle or args.shuffle_local:
        res['shuffle_true_and_seed_is'] = args.seed
    print(res)
    return res



if __name__ == '__main__':
    main()
