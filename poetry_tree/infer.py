import math
import pickle
import random
import sys

import dvc.api
import hydra
import spacy
import torch
import torch.nn as nn
import torchdata.datapipes as dp
import torchtext.transforms as T
from hydra.core.config_store import ConfigStore
from spacy_download import load_spacy
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Sampler
from torchtext.data.metrics import bleu_score

from poetry_tree.config import Params
from poetry_tree.transformer import Decoder, Encoder, Seq2Seq


def remove_attribution(row: tuple) -> tuple:
    """
    Function to keep only the first two elements in a tuple
    and convert strings to lower case
    """
    return (row[0].lower(), row[1].lower())


def get_transform(vocab):
    """
    Create transforms based on given vocabulary. The returned transform is applied to sequence
    of tokens.
    """
    text_tranform = T.Sequential(
        # converts the sentences to indices based on given vocabulary
        T.VocabTransform(vocab=vocab),
        # Add <sos> at beginning of each sentence. 1 because the index for <sos> in vocabulary is
        # 1 as seen in previous section
        T.AddToken(vocab["<sos>"], begin=True),
        # Add <eos> at beginning of each sentence. 2 because the index for <eos> in vocabulary is
        # 2 as seen in previous section
        T.AddToken(vocab["<eos>"], begin=False),
    )
    return text_tranform


class BatchSamplerSimilarLength(Sampler):
    def __init__(
        self, dataset, batch_size, spacy_tokenizer, indices=None, shuffle=True
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        # get the indices and length
        self.indices = []
        for i, s in enumerate(dataset):
            length_of_s = len(
                [token.text for token in spacy_tokenizer.tokenizer(s[1])]
            )
            self.indices.append(
                (
                    i,
                    length_of_s,
                )
            )
        # if indices are passed, then use only the ones passed (for ddp)
        if indices is not None:
            self.indices = torch.tensor(self.indices)[indices].tolist()

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)

        pooled_indices = []
        # create pool of indices with similar lengths
        for i in range(0, len(self.indices), self.batch_size * 100):
            pooled_indices.extend(
                sorted(
                    self.indices[i : i + self.batch_size * 100],
                    key=lambda x: x[1],
                )
            )
        self.pooled_indices = [x[0] for x in pooled_indices]

        batches = [
            self.pooled_indices[i : i + self.batch_size]
            for i in range(0, len(self.pooled_indices), self.batch_size)
        ]

        if self.shuffle:
            random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        return len(self.pooled_indices) // self.batch_size


class collate_batch(object):
    def __init__(
        self,
        # *params
        eng_vocab,
        spacy_tokenizer_eng,
        rus_vocab,
        spacy_tokenizer_rus,
        device,
    ):
        self.eng_vocab = eng_vocab
        self.spacy_tokenizer_eng = spacy_tokenizer_eng
        self.rus_vocab = rus_vocab
        self.spacy_tokenizer_rus = spacy_tokenizer_rus
        self.device = device

    def __call__(self, batch):
        eng_vocab = self.eng_vocab
        spacy_tokenizer_eng = self.spacy_tokenizer_eng
        rus_vocab = self.rus_vocab
        spacy_tokenizer_rus = self.spacy_tokenizer_rus
        device = self.device
        eng_list, rus_list = [], []
        for eng, rus in batch:
            eng_tensor = torch.tensor(
                get_transform(eng_vocab)(
                    [token.text for token in spacy_tokenizer_eng.tokenizer(eng)]
                ),
                device=device,
            )
            eng_list.append(eng_tensor)
            rus_tensor = torch.tensor(
                get_transform(rus_vocab)(
                    [token.text for token in spacy_tokenizer_rus.tokenizer(rus)]
                ),
                device=device,
            )
            rus_list.append(rus_tensor)

        return pad_sequence(
            eng_list, padding_value=eng_vocab["<pad>"]
        ), pad_sequence(rus_list, padding_value=rus_vocab["<pad>"])


def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch[1].T
            trg = batch[0].T

            output = model(src, trg[:, :-1])

            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            # output = [batch size * trg len - 1, output dim]
            # trg = [batch size * trg len - 1]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def translate_sentence_vectorized(
    src_tensor, src_field, trg_field, model, device, max_len=50
):
    assert isinstance(src_tensor, torch.Tensor)

    model.eval()
    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)
    # enc_src = [batch_sz, src_len, hid_dim]

    trg_indexes = [[trg_field["<sos>"]] for _ in range(len(src_tensor))]
    # Even though some examples might have been completed by producing a <eos> token
    # we still need to feed them through the model because other are not yet finished
    # and all examples act as a batch. Once every single sentence prediction encounters
    # <eos> token, then we can stop predicting.
    translations_done = [0] * len(src_tensor)
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)
        with torch.no_grad():
            output, attention = model.decoder(
                trg_tensor, enc_src, trg_mask, src_mask
            )
        pred_tokens = output.argmax(2)[:, -1]
        for i, pred_token_i in enumerate(pred_tokens):
            trg_indexes[i].append(pred_token_i)
            if pred_token_i == trg_field["<eos>"]:
                translations_done[i] = 1
        if all(translations_done):
            break

    # Iterate through each predicted example one by one;
    # Cut-off the portion including the after the <eos> token
    pred_sentences = []
    for trg_sentence in trg_indexes:
        pred_sentence = []
        for i in range(1, len(trg_sentence)):
            if trg_sentence[i] == trg_field["<eos>"]:
                break
            pred_sentence.append(trg_field.get_itos()[trg_sentence[i]])
        pred_sentences.append(pred_sentence)

    return pred_sentences, attention


def calculate_bleu(iterator, src_field, trg_field, model, device, max_len=50):
    trgs = []
    pred_trgs = []
    with torch.no_grad():
        for batch in iterator:
            src = batch[1].T
            trg = batch[0].T
            _trgs = []
            for sentence in trg:
                tmp = []
                # Start from the first token which skips the <start> token
                for i in sentence[1:]:
                    # Targets are padded. So stop appending as soon as a padding or eos token is encountered
                    if i == trg_field["<eos>"] or i == trg_field["<pad>"]:
                        break
                    tmp.append(trg_field.get_itos()[i.cpu().item()])
                _trgs.append([tmp])
            trgs += _trgs
            pred_trg, _ = translate_sentence_vectorized(
                src, src_field, trg_field, model, device
            )
            pred_trgs += pred_trg
    return pred_trgs, trgs, bleu_score(pred_trgs, trgs) * 100


cs = ConfigStore.instance()
cs.store(name="params", node=Params)


@hydra.main(version_base="1.3.2", config_path="../config", config_name="config")
def main(cfg: Params):
    if cfg.train.random_state is not None:
        RANDOM_STATE = cfg.train.random_state
    else:
        # fix randomstate for logging
        RANDOM_STATE = random.randrange(sys.maxsize)
    random.seed(RANDOM_STATE)
    # np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    torch.cuda.manual_seed(RANDOM_STATE)
    torch.backends.cudnn.deterministic = cfg.train.random_state is not None

    spacy.prefer_gpu()
    eng = load_spacy("en_core_web_sm")
    rus = load_spacy("ru_core_news_sm")

    FILE_PATH = cfg.train.path_to_dataset

    data_pipe = dp.iter.IterableWrapper([FILE_PATH])
    data_pipe = dp.iter.FileOpener(data_pipe, mode="rb")
    data_pipe = data_pipe.parse_csv(skip_lines=0, delimiter="\t", as_tuple=True)

    data_pipe = data_pipe.map(remove_attribution)

    list_data_pipe = list(data_pipe)
    total_samples = len(list_data_pipe)
    train_size = int(0.8 * total_samples)
    val_size = int(0.15 * total_samples)
    test_size = total_samples - train_size - val_size
    _, _, test_dataset = torch.utils.data.random_split(
        list_data_pipe,
        [train_size, val_size, test_size],
    )

    with dvc.api.open(cfg.model.path_to_eng_vocab, mode="rb") as f:
        eng_vocab = pickle.load(f)
    with dvc.api.open(cfg.model.path_to_rus_vocab, mode="rb") as f:
        rus_vocab = pickle.load(f)

    BATCH_SIZE = cfg.train.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    my_collator = collate_batch(
        eng_vocab=eng_vocab,
        spacy_tokenizer_eng=eng,
        rus_vocab=rus_vocab,
        spacy_tokenizer_rus=rus,
        device=device,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_sampler=BatchSamplerSimilarLength(
            dataset=test_dataset,
            batch_size=BATCH_SIZE,
            spacy_tokenizer=rus,
            shuffle=False,
        ),
        collate_fn=my_collator,
    )

    INPUT_DIM = len(rus_vocab)
    OUTPUT_DIM = len(eng_vocab)
    HID_DIM = cfg.model.hid_dim
    ENC_LAYERS = cfg.model.enc_layers
    DEC_LAYERS = cfg.model.dec_layers
    ENC_HEADS = cfg.model.enc_heads
    DEC_HEADS = cfg.model.dec_heads
    ENC_PF_DIM = cfg.model.enc_pf_dim
    DEC_PF_DIM = cfg.model.dec_pf_dim
    ENC_DROPOUT = cfg.model.enc_dropout
    DEC_DROPOUT = cfg.model.dec_dropout

    enc = Encoder(
        INPUT_DIM,
        HID_DIM,
        ENC_LAYERS,
        ENC_HEADS,
        ENC_PF_DIM,
        ENC_DROPOUT,
        device,
    )

    dec = Decoder(
        OUTPUT_DIM,
        HID_DIM,
        DEC_LAYERS,
        DEC_HEADS,
        DEC_PF_DIM,
        DEC_DROPOUT,
        device,
    )

    model = Seq2Seq(
        enc, dec, rus_vocab["<pad>"], eng_vocab["<pad>"], device
    ).to(device)

    with dvc.api.open(cfg.model.path, mode="rb") as f:
        model.load_state_dict(torch.load(f))

    criterion = nn.CrossEntropyLoss(ignore_index=eng_vocab["<pad>"])
    test_loss = evaluate(model, test_dataloader, criterion)

    print(f"Test Loss: {test_loss:.3f}    Test PPL: {math.exp(test_loss):7.3f}")

    pred_trgs, trgs, bleu_s = calculate_bleu(
        test_dataloader, rus_vocab, eng_vocab, model, device
    )
    print(f"BLEU score on test data: {bleu_s}")

    with open("results/sources.txt", "w", encoding="utf-8") as f:
        f.writelines(f"{' '.join(sentence[0])}\n" for sentence in trgs)

    with open("results/translations.txt", "w", encoding="utf-8") as f:
        f.writelines(f"{' '.join(sentence)}\n" for sentence in pred_trgs)


if __name__ == "__main__":
    main()
