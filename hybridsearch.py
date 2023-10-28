import pinecone
from datasets import load_dataset
import requests
from transformers import BertTokenizerFast
import pinecone_text
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from IPython.core.display import HTML
from io import BytesIO
import numpy as np
from typing import List, Callable, Optional, Dict

from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import HashingVectorizer
from base64 import b64encode
SparseVector = Dict[str, List]


class BM25(BaseEstimator):

    def __init__(self, tokenizer: Callable[[str], List[str]], n_features=2 ** 16, b=0.75, k1=1.6):
        """OKapi BM25 with HashingVectorizer

        Args:
            tokenizer: A function to converts text to a list of tokens
            n_features: The number of features to hash to
            b: BM25 parameters
            k1: BM25 parameters
        """
        # Fixed params
        self.n_features: int = n_features
        self.b: float = b
        self.k1: float = k1

        self._tokenizer: Callable[[str], List[str]] = tokenizer
        self._vectorizer = HashingVectorizer(
            n_features=self.n_features,
            token_pattern=None,
            tokenizer=tokenizer, norm=None,
            alternate_sign=False, binary=True)
        # Learned Params
        self.doc_freq: Optional[np.ndarray] = None
        self.n_docs: Optional[int] = None
        self.avgdl: Optional[float] = None

    def fit(self, X: List[str], y=None) -> "BM25":
        """Fit BM25 by calculating document frequency over the corpus"""
        X = self._vectorizer.transform(X)
        self.avgdl = X.sum(1).mean()
        self.n_docs = X.shape[0]
        self.doc_freq = X.sum(axis=0).A1
        return self

    def vectorize(self, text) -> SparseVector:
        sparse_array = self._vectorizer.transform(text)
        return {'indices': [int(x) for x in sparse_array.indices], 'values': sparse_array.data.tolist()}

    def get_params(self, deep=True):
        return {
            'avgdl': self.avgdl,
            'ndocs': self.n_docs,
            'doc_freq': list(self.doc_freq),
            'b': self.b,
            'k1': self.k1,
            'n_features': self.n_features
        }

    def set_params(self, **params):
        self.avgdl = params['avgdl']
        self.n_docs = params['ndocs']
        self.doc_freq = np.array(params['doc_freq'])
        self.b = params['b']
        self.k1 = params['k1']
        self.n_features = params['n_features']

    def transform_doc(self, doc: str) -> SparseVector:
        """Normalize document for BM25 scoring"""
        doc_tf = self._vectorizer.transform([doc])
        norm_doc_tf = self._norm_doc_tf(doc_tf)
        return {'indices': [int(x) for x in doc_tf.indices], 'values': norm_doc_tf.tolist()}

    def transform_query(self, query: str) -> SparseVector:
        """Normalize query for BM25 scoring"""
        query_tf = self._vectorizer.transform([query])
        indices, values = self._norm_query_tf(query_tf)
        return {'indices': [int(x) for x in indices], 'values': values.tolist()}

    def _norm_doc_tf(self, doc_tf) -> np.ndarray:
        """Calculate BM25 normalized document term-frequencies"""
        b, k1, avgdl = self.b, self.k1, self.avgdl
        tf = doc_tf.data
        norm_tf = tf / (k1 * (1.0 - b + b * (tf.sum() / avgdl)) + tf)
        return norm_tf

    def _norm_query_tf(self, query_tf):
        """Calculate BM25 normalized query term-frequencies"""
        idf = np.log((self.n_docs + 1) / (self.doc_freq[query_tf.indices] + 0.5))
        return query_tf.indices, idf / idf.sum()

def display_result(image_batch):
    figures = []
    for img in image_batch:
        b = BytesIO()
        img.save(b, format='png')
        figures.append(f'''
            <figure style="margin: 5px !important;">
              <img src="data:image/png;base64,{b64encode(b.getvalue()).decode('utf-8')}" style="width: 90px; height: 120px" >
            </figure>
        ''')
    return HTML(data=f'''
        <div style="display: flex; flex-flow: row wrap; text-align: center;">
        {''.join(figures)}
        </div>
    ''')

# init connection to pinecone
pinecone.init(
    api_key='YOUR_API_KEY',  # app.pinecone.io
    environment='gcp-starter'  # find next to api key
)
index_name = "hybrid-image-search"

if index_name not in pinecone.list_indexes():
    # create the index
    pinecone.create_index(
      index_name,
      dimension=512,
      metric="dotproduct",
      pod_type="s1"
    )
index = pinecone.Index(index_name)
fashion = load_dataset(
    "ashraq/fashion-product-images-small",
    split="train"
)
images = fashion["image"]
metadata = fashion.remove_columns("image")
images[9000]

# convert metadata into a pandas dataframe
metadata = metadata.to_pandas()
metadata.head()

def hybrid_scale(dense, sparse, alpha: float):
    """Hybrid vector scaling using a convex combination

    alpha * dense + (1 - alpha) * sparse

    Args:
        dense: Array of floats representing
        sparse: a dict of `indices` and `values`
        alpha: float between 0 and 1 where 0 == sparse only
               and 1 == dense only
    """
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    # scale sparse and dense vectors to create hybrid search vecs
    hsparse = {
        'indices': sparse['indices'],
        'values':  [v * (1 - alpha) for v in sparse['values']]
    }
    hdense = [v * alpha for v in dense]
    return hdense, hsparse

# load bert tokenizer from huggingface
tokenizer = BertTokenizerFast.from_pretrained(
    'bert-base-uncased'
)
def tokenize_func(text):
    token_ids = tokenizer(
        text,
        add_special_tokens=False
    )['input_ids']
    return tokenizer.convert_ids_to_tokens(token_ids)

bm25 = BM25(tokenize_func)
tokenize_func('Turtle Check Men Navy Blue Shirt')
bm25.fit(metadata['productDisplayName'])
bm25.transform_query(metadata['productDisplayName'][0])
bm25.transform_doc(metadata['productDisplayName'][0])
model = SentenceTransformer(
    'sentence-transformers/clip-ViT-B-32',
    device='cpu'
)
#insert the dataset in pinecone database
# batch_size = 200

# for i in tqdm(range(0, len(fashion), batch_size)):
#     # find end of batch
#     i_end = min(i+batch_size, len(fashion))
#     # extract metadata batch
#     meta_batch = metadata.iloc[i:i_end]
#     meta_dict = meta_batch.to_dict(orient="records")
#     # concatinate all metadata field except for id and year to form a single string
#     meta_batch = [" ".join(x) for x in meta_batch.loc[:, ~meta_batch.columns.isin(['id', 'year'])].values.tolist()]
#     # extract image batch
#     img_batch = images[i:i_end]
#     # create sparse BM25 vectors
#     sparse_embeds = [bm25.transform_doc(text) for text in meta_batch]
#     # create dense vectors
#     dense_embeds = model.encode(img_batch).tolist()
#     # create unique IDs
#     ids = [str(x) for x in range(i, i_end)]

#     upserts = []
#     # loop through the data and create dictionaries for uploading documents to pinecone index
#     for _id, sparse, dense, meta in zip(ids, sparse_embeds, dense_embeds, meta_dict):
#         upserts.append({
#             'id': _id,
#             'sparse_values': sparse,
#             'values': dense,
#             'metadata': meta
#         })
#     # upload the documents to the new hybrid index
#     index.upsert(upserts)

# show index description after uploading the documents
index.describe_index_stats()

# display the images
query = "blue tshirt for mens"
# create sparse and dense vectors
sparse = bm25.transform_query(query)
dense = model.encode(query).tolist()
# scale sparse and dense vectors - keyword search first
hdense, hsparse = hybrid_scale(dense, sparse, alpha=0.05)
# search
result = index.query(
    top_k=14,
    vector=hdense,
    sparse_vector=hsparse,
    include_metadata=True
)
# used returned product ids to get images
imgs = [images[int(r["id"])] for r in result["matches"]]
# display the images
display_result(imgs)




