# %% [code] {"execution":{"iopub.status.busy":"2022-05-22T04:58:31.957919Z","iopub.execute_input":"2022-05-22T04:58:31.958757Z","iopub.status.idle":"2022-05-22T04:58:51.222565Z","shell.execute_reply.started":"2022-05-22T04:58:31.958714Z","shell.execute_reply":"2022-05-22T04:58:51.221633Z"}}
!pip install transformers
!pip install sentence_transformers

# %% [code] {"execution":{"iopub.status.busy":"2022-05-22T05:01:14.977844Z","iopub.execute_input":"2022-05-22T05:01:14.978142Z","iopub.status.idle":"2022-05-22T05:01:15.066126Z","shell.execute_reply.started":"2022-05-22T05:01:14.978109Z","shell.execute_reply":"2022-05-22T05:01:15.065263Z"}}
# util.py

import os
import fnmatch


def file_names(directory, wildcard):
    return [os.path.join(directory, fn) for fn in fnmatch.filter(os.listdir(directory), wildcard)]

def train_test_split(dataset, train_test_ratio=10):
    train = []
    test = []
    for i, example in enumerate(dataset):
        if i % train_test_ratio == 0:
            test.append(example)
        else:
            train.append(example)
    return train, test

# datasets.py

'''
pairs==True: batch consisting of sentence pairs (a_1, p_1), (a_2, p_2)…, (a_n, p_n) where we assume that (a_i, p_i) are a positive pair and (a_i, p_j) for i!=j a negative pair.
train_examples = [InputExample(texts=['Anchor 1', 'Positive 1']),
    InputExample(texts=['Anchor 2', 'Positive 2'])]
To use with:
- MultipleNegativesRankingLoss
- MultipleNegativesSymmetricRankingLoss

pairs==False: batch with (label, sentence) pairs and computes the loss for all possible, valid triplets, i.e., anchor and positive must have the same label, anchor and negative a different label. The labels must be integers, with same label indicating sentences from the same class. You train dataset must contain at least 2 examples per label class.
train_examples = [InputExample(texts=['Sentence from class 0'], label=0), InputExample(texts=['Another sentence from class 0'], label=0),
    InputExample(texts=['Sentence from class 1'], label=1), InputExample(texts=['Sentence from class 2'], label=2)]
To use with:
- BatchHardTripletLoss
- BatchHardSoftMarginTripletLoss
- BatchSemiHardTripletLoss
- BatchAllTripletLoss

See
https://www.sbert.net/docs/package_reference/losses.html
https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/paraphrases/MultiDatasetDataLoader.py
https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/other/training_batch_hard_trec.py
'''

import random
import gzip
from sentence_transformers.readers import InputExample

ABSTRACT_MAX_SIZE = 500 # characters

def parsed_pubmed_dataset(file_path):
    dataset = []
    with gzip.open(file_path, 'rt', encoding='utf8') as fd:
        for line in fd:
            try:
                splits = line.strip().split("\t")
                pmid = int(splits[0])
                title = splits[1][:ABSTRACT_MAX_SIZE]
                abstract = splits[2][:ABSTRACT_MAX_SIZE]
                mesh_terms = splits[3:]
                dataset.append((pmid, title, abstract, mesh_terms))
            except:
                pass
    return dataset
    
def mesh_terms_from_dataset(dataset):
    meshs = set()
    for pmid, title, abstract, mesh_terms in dataset:
        meshs |= set(mesh_terms)
    return meshs

def selected_test_dataset(file_path, train_test_ratio=10, pairs=True, with_abstracts=True):
    test_dataset = []
    all_mesh_terms = set()
    i = size_train_examples = 0
    with open(file_path, encoding='utf8') as fd:
        for line in fd:
            try:
                if '\t\t\t' in line:
                    print(line, i)
                splits = line.strip().split("\t")
                pmid = int(splits[0])
                title = splits[1][:ABSTRACT_MAX_SIZE]
                abstract = splits[2][:ABSTRACT_MAX_SIZE]
                mesh_terms = splits[3:]
                all_mesh_terms |= set(mesh_terms)
                if i % train_test_ratio == 0: # test
                    test_dataset.append((pmid, title, abstract, mesh_terms))
                else: # train
                    if not pairs: # triplet
                        size_train_examples += 1
                        if with_abstracts:
                            size_train_examples += 1
                    for mesh in mesh_terms:
                        if pairs:
                            size_train_examples += 1
                            if with_abstracts:
                                size_train_examples += 1
                        else: # triplet
                            size_train_examples += 1
            except:
                pass
            i += 1
    return test_dataset, all_mesh_terms, size_train_examples

def selected_train_str_example_generator(file_path, pairs=True, shuffle=True, train_test_ratio=10, example_ratio=1., with_abstracts=True, max_cache_size=100000):
    count = i = 0
    dataset = []
    with open(file_path, encoding='utf8') as fd:
        for line in fd:
            try:
                splits = line.strip().split("\t")
                pmid = int(splits[0])
                title = splits[1][:ABSTRACT_MAX_SIZE]
                abstract = splits[2][:ABSTRACT_MAX_SIZE]
                mesh_terms = splits[3:]
                if i % train_test_ratio != 0: # train
                    if pairs:
                        meshs = '\t'.join(mesh_terms)
                    else: # triplet
                        dataset.append(f'{pmid}\t{title}\n')
                        if with_abstracts:
                            dataset.append(f'{pmid}\t{abstract}\n')
                    for mesh in mesh_terms:
                        if pairs:
                            dataset.append(f'{pmid}\t{title}\t{mesh}\t{meshs}\n')
                            if with_abstracts:
                                dataset.append(f'{pmid}\t{abstract}\t{mesh}\t{meshs}\n')
                        else: # triplet
                            dataset.append(f'{pmid}\t{mesh}\n')
                    if len(dataset) >= max_cache_size or not shuffle:
                        if shuffle:
                            random.shuffle(dataset)
                        if example_ratio < 1:
                            dataset = dataset[:int(example_ratio*len(dataset))]
                        for example in dataset:
                            yield example
                        count += len(dataset)
                        dataset = []
                        print('Synchronized cache.', count, 'added examples')
            except:
                pass
            i += 1
    if shuffle:
        random.shuffle(dataset)
    if example_ratio < 1:
        dataset = dataset[:int(example_ratio*len(dataset))]
    for example in dataset:
        yield example

def write_examples(file_path, examples, size):
    with open(file_path, 'w', encoding='utf-8') as fd:
        fd.write(f'{size}\n')
        count = 0
        for example in examples:
            fd.write(example)
            count += 1
        print(count, 'examples written in file', file_path)

def pubmed_dataset_to_str_examples(raw_dataset, pairs=True, shuffle=True, with_abstracts=False):
    dataset = []
    for pmid, title, abstract, mesh_terms in raw_dataset:
        if not pairs: # triplet
            dataset.append(f'{pmid}\t{title}\n')
            if with_abstracts:
                dataset.append(f'{pmid}\t{abstract}\n')
        for mesh in mesh_terms:
            if pairs:
                mesh_terms = '\t'.join(mesh_terms)
                dataset.append(f'{pmid}\t{title}\t{mesh}\t{mesh_terms}\n')
                if with_abstracts:
                    dataset.append(f'{pmid}\t{abstract}\t{mesh}\t{mesh_terms}\n')
            else: # triplet
                dataset.append(f'{pmid}\t{mesh}\n')
    if shuffle:
        random.shuffle(dataset)
    return dataset

def pubmed_dataset_to_sbert_examples(raw_dataset, pairs=True, shuffle=True, with_abstracts=False):
    dataset = []
    for pmid, title, abstract, mesh_terms in raw_dataset:
        if not pairs: # triplet
            dataset.append(InputExample(texts=[title], label=pmid, guid=pmid))
            if with_abstracts:
                dataset.append(InputExample(texts=[abstract], label=pmid, guid=pmid))
        for mesh in mesh_terms:
            if pairs:
                dataset.append(InputExample(texts=[title, mesh], label=mesh_terms, guid=pmid))
                if with_abstracts:
                    dataset.append(InputExample(texts=[abstract, mesh], label=mesh_terms, guid=pmid))
            else: # triplet
                dataset.append(InputExample(texts=[mesh], label=pmid, guid=pmid))
    if shuffle:
        random.shuffle(dataset)
    return dataset


from collections.abc import Iterable, Sized
class IterableSizedExamples(Iterable, Sized):
    def __init__(self, file_path, max_random_skip_size=0):
        self.file_path = file_path
        self.max_random_skip_size = max_random_skip_size
        with gzip.open(self.file_path, 'rt', encoding='utf8') as fd:
            self.size = int(fd.readline())
    def __iter__(self):
        while True:
            with gzip.open(self.file_path, 'rt', encoding='utf8') as fd:
                self.size = int(fd.readline())
                skip_size = random.randint(0, self.max_random_skip_size)
                for _ in range(skip_size):
                    fd.readline()
                for line in fd:
                    splits = line.strip().split("\t")
                    if len(splits) > 2: # pairs
                        pmid = int(splits[0])
                        text = splits[1]
                        mesh = splits[2]
                        mesh_terms = splits[3:]
                        yield InputExample(texts=[text, mesh], label=mesh_terms, guid=pmid)
                    else: # triplet
                        pmid = int(splits[0])
                        text = splits[1]
                        yield InputExample(texts=[text], label=pmid, guid=pmid)
    def __len__(self):
        return self.size

class PubmedPairLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True, allow_swap=False):
        self.dataset = dataset
        if shuffle:
            random.shuffle(self.dataset)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.allow_swap = allow_swap
        self.pointer = 0
    def __iter__(self):
        # iterate on batches
        # Interdit de prendre deux fois le même article (guid) dans le même batch
        # Interdit les mots MeSH associés à un article pris dans le batch
        # Interdit les articles associés à un mot Mesh pris dans le batch
        for _ in range(len(self)):
            batch = []
            guid_in_batch = set()
            mesh_terms_in_batch_articles = set()
            mesh_terms_in_batch = set()
            while len(batch) < self.batch_size:
                example = self.dataset[self.pointer]
                valid_example = True
                # If the example has a guid, check if guid is in batch
                if example.guid is not None:
                    valid_example = example.guid not in guid_in_batch # Interdit de prendre deux fois le même article (guid) dans le même batch
                    guid_in_batch.add(example.guid)
                # Interdit les mots MeSH associés à un article pris dans le batch 
                mesh = example.texts[1]
                if mesh in mesh_terms_in_batch_articles:
                    valid_example = False
                # Interdit les articles associés à un mot MeSH pris dans le batch
                mesh_terms = set(example.label)
                if len(mesh_terms & mesh_terms_in_batch) > 0:
                    valid_example = False
                if valid_example:
                    if self.allow_swap and random.random() > 0.5:
                        example.texts[0], example.texts[1] = example.texts[1], example.texts[0]
                    batch.append(InputExample(texts=[example.texts[0], example.texts[1]]))
                    mesh_terms_in_batch_articles |= mesh_terms
                    mesh_terms_in_batch.add(mesh)
                self.pointer += 1
                if self.pointer >= len(self.dataset):
                    self.pointer = 0
                    if self.shuffle:
                        random.shuffle(self.dataset)
            yield self.collate_fn(batch) if hasattr(self, 'collate_fn') and self.collate_fn is not None else batch
    def __len__(self):
        # number of batches
        return int(len(self.dataset) / self.batch_size)

class PubmedLowMemoryPairLoader:
    def __init__(self, file_path, batch_size=32, max_random_skip_size=100, allow_swap=False):
        self.file_path = file_path
        self.batch_size = batch_size
        self.allow_swap = allow_swap
        self.max_random_skip_size = max_random_skip_size
        self.dataset = IterableSizedExamples(self.file_path, self.max_random_skip_size)
    def __iter__(self):
        # iterate on batches
        # Interdit de prendre deux fois le même article (guid) dans le même batch
        # Interdit les mots MeSH associés à un article pris dans le batch
        # Interdit les articles associés à un mot Mesh pris dans le batch
        batch = []
        guid_in_batch = set()
        mesh_terms_in_batch_articles = set()
        mesh_terms_in_batch = set()
        batch_count = 0
        while True:
            for example in self.dataset:
                if len(batch) < self.batch_size:
                    valid_example = True
                    # If the example has a guid, check if guid is in batch
                    if example.guid is not None:
                        valid_example = example.guid not in guid_in_batch # Interdit de prendre deux fois le même article (guid) dans le même batch
                        guid_in_batch.add(example.guid)
                    # Interdit les mots MeSH associés à un article pris dans le batch 
                    mesh = example.texts[1]
                    if mesh in mesh_terms_in_batch_articles:
                        valid_example = False
                    # Interdit les articles associés à un mot MeSH pris dans le batch
                    mesh_terms = set(example.label)
                    if len(mesh_terms & mesh_terms_in_batch) > 0:
                        valid_example = False
                    if valid_example:
                        if self.allow_swap and random.random() > 0.5:
                            example.texts[0], example.texts[1] = example.texts[1], example.texts[0]
                        batch.append(InputExample(texts=[example.texts[0], example.texts[1]]))
                        mesh_terms_in_batch_articles |= mesh_terms
                        mesh_terms_in_batch.add(mesh)
                else:
                    yield self.collate_fn(batch) if hasattr(self, 'collate_fn') and self.collate_fn is not None else batch
                    batch_count += 1
                    if batch_count == len(self):
                        return
                    else:
                        batch = []
                        guid_in_batch = set()
                        mesh_terms_in_batch_articles = set()
                        mesh_terms_in_batch = set()
            self.dataset = IterableSizedExamples(self.file_path, self.max_random_skip_size)
    @property
    def size(self):
        # dataset size
        return len(self.dataset)
    def __len__(self):
        # number of batches
        return int(len(self.dataset) / self.batch_size)


# evaluation.py

'''
See:
https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/evaluation/RerankingEvaluator.py
https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/evaluation/TripletEvaluator.py
'''

import time
import os
import sys
import csv
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sentence_transformers.evaluation import SentenceEvaluator


class PubmedTruePositiveEvaluator(SentenceEvaluator):
    def __init__(self, val_dataset, meshs_terms, batch_size=16, 
                top_ks=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000], 
                name='true_positives', with_abstracts=False):
        self.val_dataset = val_dataset
        self.meshs = sorted(list(meshs_terms))
        self.batch_size = batch_size
        self.top_ks = top_ks
        self.name = name
        if with_abstracts:
            self.texts = [abstract for pmid, title, abstract, mesh_terms in val_dataset]
        else: # with titles
            self.texts = [title for pmid, title, abstract, mesh_terms in val_dataset]
        self.text_meshs = [mesh_terms for pmid, title, abstract, mesh_terms in val_dataset]
        self.mesh_code = {mesh:code for code, mesh in enumerate(self.meshs)}
        # self.text_mesh_codes[text_code] contains the mesh codes of the text
        self.text_mesh_codes = []
        mesh_text_codes = {}
        for text_code, text_meshs in enumerate(self.text_meshs):
            mesh_codes = set([self.mesh_code[mesh] for mesh in text_meshs])
            self.text_mesh_codes.append(mesh_codes)
            for mesh_code in mesh_codes:
                if mesh_code in mesh_text_codes:
                    mesh_text_codes[mesh_code].append(text_code)
                else:
                    mesh_text_codes[mesh_code] = []
        # self.mesh_text_codes[mesh_code] contains the text codes where the mesh term belongs
        self.mesh_text_codes = [set()] * len(self.meshs)
        for mesh_code in range(len(self.meshs)):
            if mesh_code in mesh_text_codes:
                self.mesh_text_codes[mesh_code] = set(mesh_text_codes[mesh_code])
        self.csv_headers = ['Epoch', 'Step'] + [str(top_k) for top_k in top_ks]
        self.stdout = csv.writer(sys.stdout)
        self.classification_csv_path = self.name + '.classification_true_positives.csv'
        self.retrieval_csv_path = self.name + '.retrieval_true_positives.csv'
        self.start = 0
        self.classification_results = [self.csv_headers]
        self.retrieval_results = [self.csv_headers]
    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if self.start != 0:
            self.end = time.time()
            print(f'{(self.end - self.start):.1f} seconds for training steps\n')
        start = time.time()
        # Compute cosine distances
        text_embeddings = model.encode(self.texts, batch_size=self.batch_size, show_progress_bar=False, convert_to_numpy=True)
        mesh_embeddings = model.encode(self.meshs, batch_size=self.batch_size, show_progress_bar=False, convert_to_numpy=True)
        cos_distances = cosine_distances(text_embeddings, mesh_embeddings)
        # Compute classification true positives for each top_k
        classification_true_positives = self.compute_true_positives(cos_distances, self.text_mesh_codes)
        print('Classification true positives')
        result_row = self.write_results(self.classification_csv_path, classification_true_positives, epoch, steps)
        self.classification_results.append(result_row)
        # Compute retrieval true positives for each top_k
        retrieval_true_positives = self.compute_true_positives(cos_distances.T, self.mesh_text_codes)
        print('Retrieval true positives')
        result_row = self.write_results(self.retrieval_csv_path, retrieval_true_positives, epoch, steps)
        self.retrieval_results.append(result_row)
        end = time.time()
        print(f'{(end - start):.1f} seconds for evaluation\n')
        self.start = time.time()
        return classification_true_positives[self.top_ks[-1]]
    def compute_true_positives(self, cos_distances, related):
        # Compute true positives for each top_k
        true_positives = {top_k:0 for top_k in self.top_ks}
        ranks = np.argsort(cos_distances, axis=1)
        for i in range(len(cos_distances)):
            for top_k in self.top_ks:
                predicted = set(ranks[i][:top_k])
                real = related[i]
                found = len(real & predicted)
                true_positives[top_k] += found
        return true_positives
    def write_results(self, csv_path, true_positives, epoch, steps):
        row = [epoch, steps] + [true_positives[top_k] for top_k in true_positives]
        # Write results in the standard output
        self.stdout.writerow(self.csv_headers)
        self.stdout.writerow(row)
        # Write results in a csv file
        output_file_exists = os.path.isfile(csv_path)
        with open(csv_path, newline='', mode='a' if output_file_exists else 'w', encoding="utf-8") as fd:
            writer = csv.writer(fd)
            if not output_file_exists:
                writer.writerow(self.csv_headers)
            writer.writerow(row)
        return row
    def print_result_tables(self):
        print()
        print('Classification true positives')
        for row in self.classification_results:
            self.stdout.writerow(row)
        print()
        print('Retrieval true positives')
        for row in self.retrieval_results:
            self.stdout.writerow(row)


# %% [code] {"execution":{"iopub.status.busy":"2022-05-22T05:01:32.822663Z","iopub.execute_input":"2022-05-22T05:01:32.822952Z"}}
# train.py

'''
Losses:
MultipleNegativesRankingLoss
MultipleNegativesSymmetricRankingLoss
BatchAllTripletLoss
BatchHardTripletLoss
BatchHardSoftMarginTripletLoss
BatchSemiHardTripletLoss

See:
https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/paraphrases/training.py
https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/other/training_batch_hard_trec.py
https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/ms_marco/train_bi-encoder_mnrl.py

max_seq_length:
https://www.sbert.net/docs/pretrained_models.html
>>> model = SentenceTransformer('all-MiniLM-L6-v2')
>>> model.max_seq_length
256
>>> model = SentenceTransformer('all-mpnet-base-v2')
>>> model.max_seq_length
384
'''

import time
import os
import gc
import argparse
from datetime import datetime
from torch.utils.data import DataLoader
from sentence_transformers import models, losses
from sentence_transformers import SentenceTransformer
from sentence_transformers.datasets import SentenceLabelDataset
from sentence_transformers.losses.BatchHardTripletLoss import BatchHardTripletLossDistanceFunction
cosine_distance = BatchHardTripletLossDistanceFunction.cosine_distance
eucledian_distance = BatchHardTripletLossDistanceFunction.eucledian_distance


DATA_FILE_PREFIX = 'all_examples_r.015'
DATA_DIRECTORY_PATH = '/kaggle/input/pubmed-all-mesh-r015/'
MODELS_DIRECTORY_PATH = 'models/'
WITH_ABSTRACTS = True

parser = argparse.ArgumentParser()
parser.add_argument('--data_file_prefix', default=DATA_FILE_PREFIX)
parser.add_argument('--model_name', default='all-MiniLM-L6-v2') # Use --create_new_sbert_model option if not SBERT model
parser.add_argument('--create_new_sbert_model', default=False, action='store_true')
parser.add_argument('--loss', default='MultipleNegativesRankingLoss') 
parser.add_argument('--max_seq_length', default=128, type=int)
parser.add_argument('--train_test_ratio', default=200, type=int)
parser.add_argument('--batch_size', default=384, type=int)
parser.add_argument('--epochs', default=1, type=int)
parser.add_argument('--pooling', default='mean')
parser.add_argument('--evaluation_steps', default=1000, type=int)
parser.add_argument('--warmup_steps', default=500, type=int)
parser.add_argument('--lr', default=2e-5, type=float)
parser.add_argument('--use_amp', default=False, action='store_true') # Set to False, if you use a CPU or your GPU does not support FP16 operations

args, unknown = parser.parse_known_args()

args.use_amp = True # Set to False, if you use a CPU or your GPU does not support FP16 operations

pairs = args.loss == 'MultipleNegativesRankingLoss' or args.loss=='MultipleNegativesSymmetricRankingLoss'
pairs_or_triplets = '.pairs' if pairs else '.triplets'

# Data loader
file_path = DATA_DIRECTORY_PATH + args.data_file_prefix + pairs_or_triplets + '.train_examples'

if pairs:
    train_dataloader = PubmedLowMemoryPairLoader(file_path, batch_size=args.batch_size)
else:
    train_examples = IterableSizedExamples(file_path, max_random_skip_size=100)
    train_examples_ = SentenceLabelDataset(train_examples) # TODO replace by a PubmedLowMemoryTripletLoader to write
    train_dataloader = DataLoader(train_examples_, batch_size=args.batch_size, shuffle=False, drop_last=True)
    train_examples = None

file_path = DATA_DIRECTORY_PATH + args.data_file_prefix + '.val_dataset'
val_dataset = parsed_pubmed_dataset(file_path)

file_path = DATA_DIRECTORY_PATH + args.data_file_prefix + '.mesh_terms'
mesh_terms = [mesh.strip() for mesh in open(file_path, encoding='utf-8')]

val_mesh_terms = mesh_terms_from_dataset(val_dataset)

print(train_dataloader.size, 'train examples')
print(len(val_dataset), 'validation articles')
print(len(mesh_terms), 'mesh terms')
print(len(val_mesh_terms), 'validation mesh terms')
print(sum([len(mesh_terms) for pmid, title, abstract, mesh_terms in val_dataset]), 'validation article mesh terms')

# Free unused memory

gc.collect()

# Model

os.makedirs(MODELS_DIRECTORY_PATH, exist_ok=True)

if args.create_new_sbert_model:
    model_file_name = 'sbert-'+args.model_name.replace('/', '-')+'-pubmed-'+args.loss+'-'+datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_save_path = MODELS_DIRECTORY_PATH + model_file_name
    print('Create new SBERT model with', args.model_name)
    word_embedding_model = models.Transformer(args.model_name, max_seq_length=args.max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), args.pooling)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
else:
    model_file_name = args.model_name.replace('/', '-')+'-pubmed-'+args.loss+'-'+datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_save_path = MODELS_DIRECTORY_PATH + model_file_name
    print('Use pretrained SBERT model', args.model_name)
    model = SentenceTransformer(args.model_name)
    model.max_seq_length = args.max_seq_length

# Loss function
if pairs:
    train_loss = eval('losses.'+args.loss+'(model)') 
else: # triplets
    train_loss = eval('losses.'+args.loss+'(model=model, distance_metric=cosine_distance)')

# Free unused memory

gc.collect()

# Train the model
val_evaluator = PubmedTruePositiveEvaluator(val_dataset, mesh_terms, name=model_save_path, with_abstracts=WITH_ABSTRACTS)

print('Train with', args.loss)
print('Before fine-tuning:')
model.evaluate(val_evaluator)

start = time.time()

model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=val_evaluator,
          epochs=args.epochs,
          evaluation_steps=args.evaluation_steps,
          warmup_steps=args.warmup_steps,
          output_path=model_save_path,
          use_amp=args.use_amp,
          #checkpoint_path=model_save_path,
          #checkpoint_save_steps=1000,
)

end = time.time()
print(f'{(end - start):.1f} seconds for training')

val_evaluator.print_result_tables()

model.save(model_save_path)
print(model_save_path, 'saved')
