import numpy as np
import pandas as pd
import glob
import string
import torch


class DataPreparation:
    def __init__(self, datapath):
        self.train_path = glob.glob(datapath + '/*')[1]
        self.val_path = glob.glob(datapath + '/*')[2]
        self.test_path = glob.glob(datapath + '/*')[0]

        self.train_df = pd.read_csv(self.train_path, names=['English', 'Hindi'])
        self.val_df = pd.read_csv(self.val_path, names=['English', 'Hindi'])
        self.test_df = pd.read_csv(self.test_path, names=['English', 'Hindi'])

    def dictionary_lookup(self, vocab):
        char2int = dict([(char, i) for i, char in enumerate(vocab)])
        int2char = dict([(i, char) for char, i in char2int.items()])
        return char2int, int2char

    def encode(self, source, target, source_chars, target_chars, source_char2int=None, target_char2int=None):
        num_encoder_tokens = len(source_chars)
        num_decoder_tokens = len(target_chars)
        max_source_length = max([len(txt) for txt in source])
        max_target_length = max([len(txt) for txt in target])

        encoder_input_data = np.zeros((len(source), max_source_length, num_encoder_tokens), dtype="float32")
        decoder_input_data = np.zeros((len(target), max_target_length, num_decoder_tokens), dtype="float32")
        decoder_target_data = np.zeros((len(target), max_target_length, num_decoder_tokens), dtype="float32")

        source_vocab, target_vocab = None, None
        if source_char2int is None and target_char2int is None:
            source_char2int, source_int2char = self.dictionary_lookup(source_chars)
            target_char2int, target_int2char = self.dictionary_lookup(target_chars)
            source_vocab = (source_char2int, source_int2char)
            target_vocab = (target_char2int, target_int2char)

        for i, (input_text, target_text) in enumerate(zip(source, target)):
            for t, char in enumerate(input_text):
                encoder_input_data[i, t, source_char2int[char]] = 1.0
            encoder_input_data[i, t + 1:, source_char2int["-PAD-"]] = 1.0
            for t, char in enumerate(target_text):
                decoder_input_data[i, t, target_char2int[char]] = 1.0
                if t > 0:
                    decoder_target_data[i, t - 1, target_char2int[char]] = 1.0
            decoder_input_data[i, t + 1:, target_char2int["-PAD-"]] = 1.0
            decoder_target_data[i, t:, target_char2int["-PAD-"]] = 1.0

        if source_vocab is not None and target_vocab is not None:
            return encoder_input_data, decoder_input_data, decoder_target_data, source_vocab, target_vocab
        else:
            return encoder_input_data, decoder_input_data, decoder_target_data

    def preprocess(self, source, target):
        source_chars = set(list(string.ascii_lowercase))
        target_chars = set([chr(alpha) for alpha in range(2304, 2432)])

        source = [str(x) for x in source]
        target = [str(x) for x in target]

        source_words = []
        target_words = []
        for src, tgt in zip(source, target):
            tgt = "\t" + tgt + "\n"
            source_words.append(src)
            target_words.append(tgt)
            for char in src:
                if char not in source_chars:
                    source_chars.add(char)
            for char in tgt:
                if char not in target_chars:
                    target_chars.add(char)

        source_chars = sorted(list(source_chars))
        target_chars = sorted(list(target_chars))

        source_chars.append('-PAD-')
        target_chars.append('-PAD-')

        self.num_encoder_tokens = len(source_chars)
        self.num_decoder_tokens = len(target_chars)
        self.max_source_length = max([len(txt) for txt in source_words])
        self.max_target_length = max([len(txt) for txt in target_words])

        print(f"\nNumber of samples: {len(source)}")
        print(f"Source Vocab length: {self.num_encoder_tokens}")
        print(f"Target Vocab length: {self.num_decoder_tokens}")
        print(f"Max sequence length for inputs: {self.max_source_length}")
        print(f"Max sequence length for outputs: {self.max_target_length}")

        return source_words, target_words, source_chars, target_chars

    def create_dataloaders(self, batch_size):
        train_source_words, train_target_words, train_source_chars, train_target_chars = self.preprocess(
            self.train_df["English"].to_list(), self.train_df["Hindi"].to_list())
        self.train_data = self.encode(train_source_words, train_target_words, train_source_chars, train_target_chars)
        (self.train_encoder_input, self.train_decoder_input, self.train_decoder_target, 
         self.source_vocab, self.target_vocab) = self.train_data
        self.source_char2int, self.source_int2char = self.source_vocab
        self.target_char2int, self.target_int2char = self.target_vocab

        val_source_words, val_target_words, val_source_chars, val_target_chars = self.preprocess(
            self.val_df["English"].to_list(), self.val_df["Hindi"].to_list())
        self.val_data = self.encode(val_source_words, val_target_words, list(self.source_char2int.keys()),
                                   list(self.target_char2int.keys()), source_char2int=self.source_char2int,
                                   target_char2int=self.target_char2int)
        self.val_encoder_input, self.val_decoder_input, self.val_decoder_target = self.val_data

        test_source_words, test_target_words, test_source_chars, test_target_chars = self.preprocess(
            self.test_df["English"].to_list(), self.test_df["Hindi"].to_list())
        self.test_data = self.encode(test_source_words, test_target_words, list(self.source_char2int.keys()),
                                    list(self.target_char2int.keys()), source_char2int=self.source_char2int,
                                    target_char2int=self.target_char2int)
        self.test_encoder_input, self.test_decoder_input, self.test_decoder_target = self.test_data

        encoder_input_data_train = torch.stack([torch.from_numpy(np.array(i)) for i in self.train_encoder_input])
        decoder_input_data_train = torch.stack([torch.from_numpy(np.array(i)) for i in self.train_decoder_input])

        encoder_input_data_val = torch.stack([torch.from_numpy(np.array(i)) for i in self.val_encoder_input])
        decoder_input_data_val = torch.stack([torch.from_numpy(np.array(i)) for i in self.val_decoder_input])

        encoder_input_data_test = torch.stack([torch.from_numpy(np.array(i)).float() for i in self.test_encoder_input])
        decoder_input_data_test = torch.stack([torch.from_numpy(np.array(i)).float() for i in self.test_decoder_input])

        train_dataset = torch.utils.data.TensorDataset(encoder_input_data_train, decoder_input_data_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = torch.utils.data.TensorDataset(encoder_input_data_val, decoder_input_data_val)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        test_dataset = torch.utils.data.TensorDataset(encoder_input_data_test, decoder_input_data_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        return train_loader, val_loader, test_loader
