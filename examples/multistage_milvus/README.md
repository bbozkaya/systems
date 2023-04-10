The two examples here show how to build a custom Merlin Systems inference operator to build and query a Milvus vector database for item embeddings.

The first notebook needs to run first to create the retrieval and ranking models, and to export the resulting item and user embedding vectors to disk.

The second notebook will start and connect to a Milvus server, set up a milvus vector database collection using item embeddings and create an index
on it. It will then create a Triton ensemble workflow and package to make nearest neighbor queries (using L2 distance) against this index. Triton
will return the top-k ordered list of items relevant to the user id being queried.

Commands needed to install and start a milvus server (before creating a vector index and querying it) are included in the second notebook.

Note: This version of milvus (2.2.X) used in the notebooks does not support GPU acceleration.
