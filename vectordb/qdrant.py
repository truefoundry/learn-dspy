from collections import defaultdict
from typing import List, Optional, Union

import dspy
from dsp.utils import dotdict

from embeddings.interface import Embeddings

try:
    from qdrant_client import QdrantClient
except ImportError:
    raise ImportError(
        "The 'qdrant' extra is required to use QdrantRM. Install it with `pip install dspy-ai[qdrant]`",
    )


class CustomQdrantRetriever(dspy.Retrieve):
    """
    A retrieval module that uses Qdrant to return the top passages for a given query.

    Assumes that a Qdrant collection has been created and populated with the following payload:
        - document: The text of the passage

    Args:
        qdrant_collection_name (str): The name of the Qdrant collection.
        qdrant_client (QdrantClient): A QdrantClient instance.
        k (int, optional): The default number of top passages to retrieve. Defaults to 3.

    Examples:
        Below is a code snippet that shows how to use Qdrant as the default retriver:
        ```python
        from qdrant_client import QdrantClient

        llm = dspy.OpenAI(model="gpt-3.5-turbo")
        qdrant_client = QdrantClient()
        retriever_model = QdrantRM("my_collection_name", qdrant_client=qdrant_client)
        dspy.settings.configure(lm=llm, rm=retriever_model)
        ```

        Below is a code snippet that shows how to use Qdrant in the forward() function of a module
        ```python
        self.retrieve = QdrantRM("my_collection_name", qdrant_client=qdrant_client, k=num_passages)
        ```
    """

    def __init__(
        self,
        qdrant_collection_name: str,
        qdrant_client: QdrantClient,
        embedding_model: Embeddings,
        k: int = 3,
    ):
        self._qdrant_collection_name = qdrant_collection_name
        self._qdrant_client = qdrant_client
        self.embedding_model = embedding_model
        self.k = k

        super().__init__(k=k)

    def forward(self, query_or_queries: Union[str, List[str]], k: Optional[int] = None, filter: dict = {}) -> dspy.Prediction:
        """Search with Qdrant for self.k top passages for query

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            k (Optional[int]): The number of top passages to retrieve. Defaults to self.k.
            filter (dict): The filter to apply to the search query. Defaults to {}.
        Returns:
            dspy.Prediction: An object containing the retrieved passages.
        """
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        queries = [q for q in queries if q]  # Filter empty queries

        k = k if k is not None else self.k

        batch_embeddings = [self.embedding_model.embed_query(query) for query in queries]

        batch_results = [self._qdrant_client.search(
            collection_name=self._qdrant_collection_name,
            query_vector=embedding,
            query_filter=filter if filter else {},
            limit=k,
            with_payload=True, 
            
        ) for embedding in batch_embeddings]
        
        # Accumulate scores for each passage
        passages_scores = defaultdict(float)
        # Store the metadata and page_content from payload dict
        passages_content = defaultdict(dict)

        for batch in batch_results:
            for result in batch:
                # If a passage is returned multiple times, the score is accumulated.
                passages_scores[result.id] += result.score
                passages_content[result.id] = result.payload

        # Sort passages by their accumulated scores in descending order
        sorted_passages = sorted(passages_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        passages = []
        metadata = []
        probs = []
        for id, score in sorted_passages:
            passages_content[id]['metadata']['score'] = score
            
            passages.append(passages_content[id]['page_content'])
            metadata.append(passages_content[id]['metadata'])
            probs.append(score)

        return dspy.Prediction(passages=passages, metadata=metadata, probs=probs)

        