import math
import pickle
import random
import sys
import time
from datetime import datetime

import git
import hydra
import mlflow
import onnx
import spacy
import torch
import torch.nn as nn
import torch.onnx
import torchdata.datapipes as dp
import torchtext.transforms as T
from hydra.core.config_store import ConfigStore
from spacy_download import load_spacy
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Sampler
from torchtext.vocab import build_vocab_from_iterator

from poetry_tree.config import Params
from poetry_tree.transformer import Decoder, Encoder, Seq2Seq


def remove_attribution(row: tuple) -> tuple:
    """
    Function to keep only the first two elements in a tuple
    and convert strings to lower case
    """
    return (row[0].lower(), row[1].lower())


def get_tokens(data_iter, spacy_tokenizer):
    """
    Function to yield tokens from an iterator. Since, our iterator contains
    tuple of sentences (source and target), `place` parameters defines for which
    index to return the tokens for. `place=0` for source and `place=1` for target
    """
    for words in data_iter:
        yield [token.text for token in spacy_tokenizer.tokenizer(words)]


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


def showSomeTransformedSentences(eng, rus, eng_vocab, rus_vocab):
    """
    Function to show how the sentences look like after applying all transforms.
    Here we try to print actual words instead of corresponding index
    """
    source_index_to_string = rus_vocab.get_itos()
    target_index_to_string = eng_vocab.get_itos()
    len_eng = len(eng)
    len_rus = len(rus)
    source, target = "", ""
    print()
    for i in range(min(len_eng, len_rus)):
        target += " " + target_index_to_string[eng[i]]
        source += " " + source_index_to_string[rus[i]]
    print(f"Source: {source}")
    print(f"Traget: {target}")


def initialize_weights(m):
    if hasattr(m, "weight") and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0
    history_loss = 0
    for i, batch in enumerate(iterator):
        src = batch[1].T
        trg = batch[0].T

        optimizer.zero_grad()

        output, _ = model(src, trg[:, :-1])

        # output = [batch size, trg len - 1, output dim]
        # trg = [batch size, trg len]

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

        history_loss += loss.cpu().data.numpy().item()
        if (i + 1) % 10 == 0:
            mlflow.log_metric("Train Loss", history_loss / 10)
            history_loss = 0

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch[1].T
            trg = batch[0].T

            output, _ = model(src, trg[:, :-1])

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


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


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
    train_size = int(0.8 * total_samples)  # 80%
    val_size = int(0.15 * total_samples)  # 15%
    test_size = total_samples - train_size - val_size  # остальные 5%
    train_dataset, val_dataset, _ = torch.utils.data.random_split(
        list_data_pipe,
        [train_size, val_size, test_size],
    )

    eng_vocab = build_vocab_from_iterator(
        get_tokens([string[0] for string in train_dataset], eng),
        min_freq=3,
        specials=["<unk>", "<pad>", "<sos>", "<eos>"],
        special_first=True,
    )
    eng_vocab.set_default_index(eng_vocab["<unk>"])

    rus_vocab = build_vocab_from_iterator(
        get_tokens([string[1] for string in train_dataset], rus),
        min_freq=3,
        specials=["<unk>", "<pad>", "<sos>", "<eos>"],
        special_first=True,
    )
    rus_vocab.set_default_index(rus_vocab["<unk>"])

    with open(cfg.model.path_to_eng_vocab, "wb") as f:
        pickle.dump(eng_vocab, f)
    with open(cfg.model.path_to_rus_vocab, "wb") as f:
        pickle.dump(rus_vocab, f)

    BATCH_SIZE = cfg.train.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    my_collator = collate_batch(
        eng_vocab=eng_vocab,
        spacy_tokenizer_eng=eng,
        rus_vocab=rus_vocab,
        spacy_tokenizer_rus=rus,
        device=device,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=BatchSamplerSimilarLength(
            dataset=train_dataset,
            batch_size=BATCH_SIZE,
            spacy_tokenizer=rus,
            shuffle=True,
        ),
        collate_fn=my_collator,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_sampler=BatchSamplerSimilarLength(
            dataset=val_dataset,
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

    num_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"The model has {num_parameters:,} trainable parameters")

    model.apply(initialize_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=eng_vocab["<pad>"])

    N_EPOCHS = cfg.train.n_epochs
    CLIP = cfg.train.clip_grad

    mlflow.set_tracking_uri(uri=cfg.train.mlflow_uri)
    mlflow.set_experiment(
        "Train model " + str(datetime.now()).replace(":", "-")
    )

    with mlflow.start_run():
        mlflow.log_params(
            {
                "input_dim": INPUT_DIM,
                "output_dim": OUTPUT_DIM,
                "random_seed_of_session": RANDOM_STATE,
            }
        )
        mlflow.log_params(cfg)
        repo = git.Repo(search_parent_directories=True)
        mlflow.set_tag("Git commit ID", str(repo.head.object.hexsha))

        best_valid_loss = float("inf")
        for epoch in range(N_EPOCHS):
            start_time = time.time()

            train_loss = train(
                model, train_dataloader, optimizer, criterion, CLIP
            )
            valid_loss = evaluate(model, val_dataloader, criterion)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), cfg.model.path)

            print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")
            print(
                f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}"
            )
            print(
                f"\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}"
            )

            mlflow.log_metric("Validation Loss", valid_loss)
            mlflow.log_metric("Train PPL", math.exp(train_loss))
            mlflow.log_metric("Validation PPL", math.exp(valid_loss))

        model.load_state_dict(torch.load(cfg.model.path))
        model.eval()
        input_onnx_src = torch.randint(
            0, INPUT_DIM, (1, 50), device=device, dtype=torch.int
        )
        input_onnx_trg = torch.randint(
            0, OUTPUT_DIM, (1, 50), device=device, dtype=torch.int
        )
        example_output, example_attention = model(
            input_onnx_src, input_onnx_trg
        )
        torch.onnx.export(
            model,
            args=(input_onnx_src, input_onnx_trg),
            f=cfg.model.path_onnx,
            verbose=True,
            input_names=["source", "target"],
            output_names=["output", "attention"],
        )
        onnx_model = onnx.load(cfg.model.path_onnx)
        mlflow.onnx.log_model(onnx_model=onnx_model, artifact_path="models")


if __name__ == "__main__":
    main()
