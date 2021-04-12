import pickle
import os
import random
import numpy
import copy

class Dataset(object):
    
    def __init__(self, xs=[], ys=[], raws=None, norms=None, ids=None, pos=None, cands=None, targets=None,
                 idx2txt=[], txt2idx={}, max_len=250, vocab_size=30000, dtype=None):
        
        self.__dtype = dtype
        self.__vocab_size = vocab_size
        self.__idx2txt = idx2txt
        self.__txt2idx = txt2idx
        self.__max_len = max_len
        self.__xs = []
        self.__raws = []
        self.__norms = []
        self.__ys = []
        self.__ls = []
        self.__ids = []
        self.__pos = []
        self.__cands = []
        self.__targets = []
        if raws is None:
            assert len(xs) == len(ys)
            raws = [None for _ in ys]
        else:
            assert len(xs) == len(ys) and len(ys) == len(raws)
        if norms is None:
            norms = [None for _ in ys]
        if pos is None:
            pos = [None for _ in ys]
        else:
            assert len(xs) == len(pos)
        if cands is None:
            cands = [None for _ in ys]
        else:
            assert(len(xs) == len(cands))
        if targets is None:
            targets = [None for _ in ys]
        else:
            assert(len(cands) == len(targets))
        if ids is None:
            ids = list(range(len(xs)))
        else:
            assert len(xs) == len(ids)
        for x, y, r, n, i, p, cd, tg in zip(xs, ys, raws, norms, ids, pos, cands, targets):
            self.__raws.append(r)
            self.__norms.append(n)
            self.__ys.append(y)
            self.__ids.append(i)
            self.__pos.append(p)
            self.__cands.append(cd)
            self.__targets.append(tg)
            if len(x) > self.__max_len:
                self.__ls.append(self.__max_len)
            else:
                self.__ls.append(len(x))
            self.__xs.append([])
            for t in x[:self.__max_len]:
                if t >= self.__vocab_size:
                    self.__xs[-1].append(self.__txt2idx['<unk>'])
                else:
                    self.__xs[-1].append(t)
            while len(self.__xs[-1]) < self.__max_len:
                self.__xs[-1].append(self.__txt2idx['<pad>'])
        self.__xs = numpy.asarray(self.__xs, dtype=self.__dtype['int'])
        self.__ys = numpy.asarray(self.__ys, dtype=self.__dtype['int'])
        self.__ls = numpy.asarray(self.__ls, dtype=self.__dtype['int'])
        self.__ids = numpy.asarray(self.__ids, dtype=self.__dtype['int'])
        self.__size = len(self.__raws)
        
        assert self.__size == len(self.__raws)      \
            and len(self.__raws) == len(self.__norms) \
            and len(self.__norms) == len(self.__pos) \
            and len(self.__pos) == len(self.__xs)  \
            and len(self.__xs) == len(self.__ys) \
            and len(self.__ys) == len(self.__ls) \
            and len(self.__ls) == len(self.__ids) \
            and len(self.__ids) == len(self.__cands) \
            and len(self.__cands) == len(self.__targets)
        
        self.__epoch = None
        self.reset_epoch()

    def reset_epoch(self):
        
        self.__epoch = random.sample(range(self.__size), self.__size)
        
    def next_batch(self, batch_size=32):
        
        batch = {"x": [], "y": [], "l": [], "raw": [], "norm": [], "id": [], "pos": [],
                 "candidates": [], "targets": [], "new_epoch": False}
        assert batch_size <= self.__size
        if len(self.__epoch) < batch_size:
            batch['new_epoch'] = True
            self.reset_epoch()
        idxs = self.__epoch[:batch_size]
        self.__epoch = self.__epoch[batch_size:]
        batch['x'] = numpy.take(self.__xs, indices=idxs, axis=0)
        batch['y'] = numpy.take(self.__ys, indices=idxs, axis=0)
        batch['l'] = numpy.take(self.__ls, indices=idxs, axis=0)
        batch['id'] = numpy.take(self.__ids, indices=idxs, axis=0)
        for i in idxs:
            batch['raw'].append(self.__raws[i])
        batch['raw'] = copy.deepcopy(batch['raw'])
        for i in idxs:
            batch['norm'].append(self.__norms[i])
        batch['norm'] = copy.deepcopy(batch['norm'])
        for i in idxs:
            batch['pos'].append(self.__pos[i])
        batch['pos'] = copy.deepcopy(batch['pos'])
        for i in idxs:
            batch['candidates'].append(self.__cands[i])
        batch['candidates'] = copy.deepcopy(batch['candidates'])
        for i in idxs:
            batch['targets'].append(self.__targets[i])
        batch['targets'] = copy.deepcopy(batch['targets'])
        return batch
        
    def idxs2raw(self, xs, ls):
        
        seq = []
        for x, l in zip(xs, ls):
            seq.append([])
            for t in x[:l]:
                seq[-1].append(self.__idx2txt[t])
        return seq
        
    def get_size(self):
        
        return self.__size
        
    def get_rest_epoch_size(self):
        
        return len(self.__epoch)


class Py150(object):
    
    def __init__(self, path, max_len=250, vocab_size=30000, dtype='32'):
        
        self.__dtypes = self.__dtype(dtype)
        self.__max_len = max_len
        self.__vocab_size = vocab_size
        
        self.__train_path = os.path.join(path, "train.pkl")
        self.__valid_path = os.path.join(path, "valid.pkl")
        self.__test_path = os.path.join(path, "test.pkl")
        
        with open(self.__train_path, "rb") as f:
            self.train = pickle.load(f)
        with open(self.__valid_path, "rb") as f:
            self.valid = pickle.load(f)
        with open(self.__test_path, "rb") as f:
            self.test = pickle.load(f)
        
        self.__idx2txt = None
        self.__txt2idx = None
        self.build_vocab(self.train)

        self.train = self.build_dataset(self.train)
        self.dev = self.build_dataset(self.valid)
        self.test = self.build_dataset(self.test)
        
        return
    

    def build_dataset(self, data):
        raw, norm, x, y, ids, pos, cands, targets = [], [], [], [], [], [], [], []
        for i in range(len(data["label"])):
            raw.append(data["raw"][i])
            norm.append(data["norm"][i])
            x.append(self.raw2idxs(norm[-1]))
            y.append(data["label"][i])
            ids.append(i)
            pos.append(data["error"][i])
            cands.append(data["candidates"][i])
            targets.append(data["targets"][i])
        return Dataset(xs=x, ys=y, raws=raw, norms=norm, ids=ids, pos=pos, cands=cands, targets=targets,
                        idx2txt=self.__idx2txt,
                        txt2idx=self.__txt2idx,
                        max_len=self.__max_len,
                        vocab_size=self.__vocab_size,
                        dtype=self.__dtypes)


    def __dtype(self, dtype='32'):
    
        assert dtype in ['16', '32', '64']
        if dtype == '16':
            return {'fp': numpy.float16, 'int': numpy.int16}
        elif dtype == '32':
            return {'fp': numpy.float32, 'int': numpy.int32}
        elif dtype == '64':
            return {'fp': numpy.float64, 'int': numpy.int64}

    def get_dtype(self):
        
        return self.__dtypes
    
    def get_max_len(self):
        
        return self.__max_len
        
    def get_vocab_size(self):
        
        return self.__vocab_size
        
    def get_idx2txt(self):
        
        return copy.deepcopy(self.__idx2txt)
    
    def get_txt2idx(self):
        
        return copy.deepcopy(self.__txt2idx)
        
    def vocab2idx(self, vocab):
        
        if vocab in self.__txt2idx.keys():
            return self.__txt2idx[vocab]
        else:
            return self.__txt2idx['<unk>']

    def idx2vocab(self, idx):
        
        if idx >= 0 and idx < len(self.__idx2txt):
            return self.__idx2txt[idx]
        else:
            return '<unk>'
            
    def idxs2raw(self, xs, ls):
        
        seq = []
        for x, l in zip(xs, ls):
            seq.append([])
            for t in x[:l]:
                seq[-1].append(self.__idx2txt[t])
        return seq

    def raw2idxs(self, raw):
        return [self.__txt2idx.get(tok, self.__txt2idx["<unk>"]) for tok in raw]
    
    def build_vocab(self, traindata):
        counter = dict()
        for seq in traindata["norm"]:
            for token in seq:
                counter[token] = counter.get(token, 0) + 1
        dic_items = list(counter.items())
        dic_items.sort(key=lambda x: x[1], reverse=True)
        self.__idx2txt = ["<pad>", "<unk>"] + [item[0] for item in dic_items][:self.__vocab_size]
        self.__txt2idx = {}
        for i in range(self.__vocab_size + 2):
            self.__txt2idx[self.__idx2txt[i]] = i

if __name__ == "__main__":
    
    import time
    start_time = time.time()
    java = Py150(path="../../great")
    print ("time cost = " + str(time.time()-start_time)+" sec")
    with open("../../great/Py150.pkl", "wb") as f:
        pickle.dump(java, f)

    start_time = time.time()
    b = java.train.next_batch(1)
    print ("time cost = " + str(time.time()-start_time)+" sec")
    for t in b['raw'][0]:
        print (t, end=" ")
    print ()
    for t in java.idxs2raw(b['x'], b['l'])[0]:
        print (t, end=" ")
    print ("\n")
    start_time = time.time()
    b = java.dev.next_batch(1)
    print ("time cost = "+str(time.time()-start_time)+" sec")
    for t in b['raw'][0]:
        print (t, end=" ")
    print ()
    for t in java.idxs2raw(b['x'], b['l'])[0]:
        print (t, end=" ")
    print ("\n")
    start_time = time.time()
    b = java.test.next_batch(1)
    print ("time cost = "+str(time.time()-start_time)+" sec")
    for t in b['raw'][0]:
        print (t, end=" ")
    print ()
    for t in java.idxs2raw(b['x'], b['l'])[0]:
        print (t, end=" ")
    print ("\n")