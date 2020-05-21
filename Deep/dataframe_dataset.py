#Based on https://gist.github.com/nissan/ccb0553edb6abafd20c3dec34ee8099d

from torchtext import data
from tqdm import tqdm

class DataFrameDataset(data.Dataset):
    def __init__(self, df, text_field, label_field, **kwargs):
        fields = [('text', text_field), ('label', label_field)]
        examples = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            label = row.overall - 1
            text = row.reviewText
            if len(text) <= 0:
                continue
            examples.append(data.Example.fromlist([text, label], fields))

        super().__init__(examples, fields, **kwargs)

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    @classmethod
    def splits(cls, text_field, label_field, train_df, val_df=None, test_df=None, **kwargs):
        train_data, val_data, test_data = (None, None, None)

        if train_df is not None:
            train_data = cls(train_df.copy(), text_field, label_field, **kwargs)
        if val_df is not None:
            val_data = cls(val_df.copy(), text_field, label_field, **kwargs)
        if test_df is not None:
            test_data = cls(test_df.copy(), text_field, label_field, **kwargs)

        return tuple(d for d in (train_data, val_data, test_data) if d is not None)
