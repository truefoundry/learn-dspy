
from typing import List, Optional, Union
from sentence_transformers import CrossEncoder
import dspy
from dsp.utils import dotdict

# More about why re-ranking is essential: https://www.mixedbread.ai/blog/mxbai-rerank-v1
class MxBaiReranker(dspy.Retrieve):
    """
    Document Reranking using mixedbread-ai reranker.
    """

    def __init__(self, model_name: str = "mixedbread-ai/mxbai-rerank-xsmall-v1", k=5) -> None:
        self.model_name = model_name
        self.model = CrossEncoder(self.model_name)
        self.k = k

        super().__init__(k=k)

    def forward(
        self,
        query_or_queries: Union[str, List[str]], 
        k: Optional[int] = None,
        *,
        documents: Union[str, List[str]]
    ) -> dspy.Prediction:
        """Compress retrieved documents given the query context."""

        queries = (
            query_or_queries
            if isinstance(query_or_queries, str)
            else [q for q in queries if q]
        )

        k = k if k is not None else self.k
        if isinstance(queries, str):
            reranked_docs = self.model.rank(queries, documents, return_documents=True, top_k=self.k)  
        elif isinstance(queries, List[str]):
            reranked_docs = [self.model.rank(q, documents, return_documents=True, top_k=self.k) for q in queries]
            reranked_docs = reranked_docs[0]
        passages = [dotdict({"long_text": f"{doc['text']}"}) for doc in reranked_docs]
        return passages
