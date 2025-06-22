import time

from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq

from .general import Timer


class DataCollatorForSeq2SeqTimed(DataCollatorForSeq2Seq):
    timer = Timer()

    def __call__(self, features, return_tensors=None):
        start_time = time.time()
        r = super().__call__(features, return_tensors)
        self.timer.append(time.time() - start_time)
        return r


class DataCollatorForLanguageModelingTimed(DataCollatorForLanguageModeling):
    timer = Timer()

    def __call__(self, features, return_tensors=None):
        start_time = time.time()
        r = super().__call__(features, return_tensors)
        self.timer.append(time.time() - start_time)
        return r
