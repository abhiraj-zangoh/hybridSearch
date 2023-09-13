import pinecone
from datasets import load_dataset
import requests
from transformers import BertTokenizerFast
import pinecone_text
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from IPython.core.display import HTML
from io import BytesIO
from base64 import b64encode
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
    api_key='f4fa8f28-2f64-4e5b-a0c1-36c247b1fab6',  # app.pinecone.io
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
with open('pinecone_text.py' ,'w') as fb:
    fb.write(requests.get('https://storage.googleapis.com/gareth-pinecone-datasets/pinecone_text.py').text)
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

bm25 = pinecone_text.BM25(tokenize_func)    
tokenize_func('Turtle Check Men Navy Blue Shirt')
bm25.fit(metadata['productDisplayName'])
bm25.transform_query(metadata['productDisplayName'][0])
bm25.transform_doc(metadata['productDisplayName'][0])
model = SentenceTransformer(
    'sentence-transformers/clip-ViT-B-32',
    device='cpu'
)
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
query = "dark blue french connection jeans for men"

# create sparse and dense vectors
sparse = bm25.transform_query(query)
dense = model.encode(query).tolist()
# search
result = index.query(
    top_k=14,
    vector=dense,
    sparse_vector=sparse,
    include_metadata=True
)
# used returned product ids to get images
imgs = [images[int(r["id"])] for r in result["matches"]]
print(imgs)
print(display_result(imgs))
