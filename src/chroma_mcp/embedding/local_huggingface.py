"""
Local huggingface embedding function.
Model should be download to local to run offline first. Follow: https://huggingface.co/docs/hub/en/models-downloading
"""

from chromadb.api.types import Embeddings, Documents, EmbeddingFunction, Space
from typing import List, Dict, Any, Optional
import os
import numpy as np
from chromadb.utils.embedding_functions.schemas import validate_config_schema
from sentence_transformers import SentenceTransformer
from typing import cast


class HuggingFaceLocalEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    This class is used to get embeddings for a list of texts using the HuggingFace API.
    It requires an API key and a model name. The default model name is "sentence-transformers/all-MiniLM-L6-v2".
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Initialize the HuggingFaceEmbeddingFunction.

        Args:
            api_key_env_var (str, optional): Environment variable name that contains your API key for the HuggingFace API.
                Defaults to "CHROMA_HUGGINGFACE_API_KEY".
            model_name (str, optional): The name of the model to use for text embeddings.
                Defaults to "sentence-transformers/all-MiniLM-L6-v2".
        """

        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        """
        Get the embeddings for a list of texts.

        Args:
            input (Documents): A list of texts to get embeddings for.

        Returns:
            Embeddings: The embeddings for the texts.

        Example:
            >>> hugging_face = HuggingFaceEmbeddingFunction(api_key_env_var="CHROMA_HUGGINGFACE_API_KEY")
            >>> texts = ["Hello, world!", "How are you?"]
            >>> embeddings = hugging_face(texts)
        """
        response = self.model.encode(
            input,
        )
        # Convert to numpy arrays
        return cast(Embeddings, response.tolist())

    @staticmethod
    def name() -> str:
        return "huggingface"

    def default_space(self) -> Space:
        return "cosine"

    def supported_spaces(self) -> List[Space]:
        return ["cosine", "l2", "ip"]

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "EmbeddingFunction[Documents]":
        model_name = config.get("model_name")
        assert model_name is not None, "Model name must be provided in the config"

        return HuggingFaceLocalEmbeddingFunction(model_name=model_name)

    def get_config(self) -> Dict[str, Any]:
        return {"model_name": self.model_name}

    def validate_config_update(
        self, old_config: Dict[str, Any], new_config: Dict[str, Any]
    ) -> None:
        if "model_name" in new_config:
            raise ValueError(
                "The model name cannot be changed after the embedding function has been initialized."
            )

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """
        Validate the configuration using the JSON schema.

        Args:
            config: Configuration to validate

        Raises:
            ValidationError: If the configuration does not match the schema
        """
        validate_config_schema(config, "huggingface")
