import pickle

import numpy as np
import spacy
import torchtext.transforms as T
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        with open("/assets/eng_vocab.pickle", mode="rb") as f:
            self.eng_vocab = pickle.load(f)
        with open("/assets/rus_vocab.pickle", mode="rb") as f:
            self.rus_vocab = pickle.load(f)
        self.eng = spacy.blank("en").from_disk("/assets/eng_tokenizer")
        self.rus = spacy.blank("ru").from_disk("/assets/rus_tokenizer")

    def get_transform(self, vocab):
        text_transform = T.Sequential(
            T.VocabTransform(vocab=vocab),
            T.AddToken(vocab["<sos>"], begin=True),
            T.AddToken(vocab["<eos>"], begin=False),
        )
        return text_transform

    def execute(self, requests):
        responses = []
        for request in requests:
            texts_source = pb_utils.get_input_tensor_by_name(
                request, "TEXTS_SOURCE"
            ).as_numpy()
            texts_target = pb_utils.get_input_tensor_by_name(
                request, "TEXTS_TARGET"
            ).as_numpy()
            texts_source = [el.decode() for el in texts_source][0].lower()
            texts_target = [el.decode() for el in texts_target][0].lower()
            # print(type(texts_source))  # FIXME
            words_source = [
                token.text for token in self.rus.tokenizer(texts_source)
            ]
            words_target = [
                token.text for token in self.eng.tokenizer(texts_target)
            ]
            tokens_rus = self.get_transform(self.rus_vocab)(words_source)
            if len(words_target) == 0:
                tokens_eng = [
                    self.eng_vocab["<sos>"],
                ]
            else:
                tokens_eng = self.get_transform(self.eng_vocab)(words_target)[
                    :-1
                ]
            tokens_rus = tokens_rus[:50] + [
                self.rus_vocab["<pad>"]
                for _ in range(50 - len(tokens_rus[:50]))
            ]
            tokens_eng = tokens_eng[:50] + [
                self.eng_vocab["<pad>"]
                for _ in range(50 - len(tokens_eng[:50]))
            ]
            output_tensor_tokens_rus = pb_utils.Tensor(
                "source", np.array(tokens_rus, dtype=np.int32)
            )
            output_tensor_tokens_eng = pb_utils.Tensor(
                "target", np.array(tokens_eng, dtype=np.int32)
            )

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    output_tensor_tokens_rus,
                    output_tensor_tokens_eng,
                ]
            )
            responses.append(inference_response)
        return responses
