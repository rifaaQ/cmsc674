# LLM Arrtribution

This project contains code for testing BM25 related stuff and the best embeddigng method used for llm attribution.


## Demo
A demo can be found [here](https://huggingface.co/spaces/hongjos/llm-attribution) 🤗. Just enter some text into the query to get the documents it probably came from (right now it only contains some wikipedia articles).

## Some Notable Files
- `bm25_init.py` and `bm25_intro` contains code for the implementation and testing of BM25.
- `embedding_examples` folder contains some examples of the embeddings of different sentence transformer models used for testing (the rest is stored elsewher).
- `test_data` contains all the test data from Wikipedia, TreCovid, and NFCorpus.
- `generate_embeddings` - creates the different embeddings to compare using some given model from Huggingface.
- `dim_reduce` - tests dimensionlaity reduction using PCA on some embeddings to see how accuracy changes 

