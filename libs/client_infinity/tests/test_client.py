from infinity_client import Client
from infinity_client.models import OpenAIEmbeddingInputText, OpenAIModelInfo, SparseEmbeddingInput
from infinity_client.api.default import embeddings, models, sparse_embeddings
from infinity_client.types import Response
import pytest


def test_model_info(server_available):
    client = Client(base_url=pytest.URL)

    with client as client:
        model_info: OpenAIModelInfo = models.sync(client=client)
        # or if you need more info (e.g. status_code)
        response: Response[OpenAIModelInfo] = models.sync_detailed(client=client)


def _first_embed_model_id(client: Client) -> str:
    model_info = models.sync(client=client)
    assert model_info is not None
    assert model_info.data
    for model in model_info.data:
        capabilities = model.capabilities if model.capabilities is not None else []
        if "embed" in capabilities:
            return model.id
    pytest.skip("No deployed model with capability `embed`")


def _first_sparse_embed_model_id(client: Client) -> str:
    model_info = models.sync(client=client)
    assert model_info is not None
    assert model_info.data
    for model in model_info.data:
        capabilities = model.capabilities if model.capabilities is not None else []
        if "sparse_embed" in capabilities:
            return model.id
    pytest.skip("No deployed model with capability `sparse_embed`")


def test_sparse_embeddings(server_available):
    client = Client(base_url=pytest.URL)

    with client as client:
        model_id = _first_sparse_embed_model_id(client)
        body = SparseEmbeddingInput(
            input_=["hello sparse"],
            model=model_id,
            prune_ratio=0.0,
            task="document",
        )
        response = sparse_embeddings.sync_detailed(client=client, body=body)
        assert response.status_code == 200
        assert response.parsed is not None
        assert response.parsed.data
        assert response.parsed.data[0].embedding.indices
        assert response.parsed.data[0].embedding.values


def test_gzip_embeddings(server_available):
    client = Client(base_url=pytest.URL)

    with client as client:
        model_id = _first_embed_model_id(client)
        body = OpenAIEmbeddingInputText(
            input_=["x" * 3000],
            model=model_id,
        )
        response = embeddings.sync_detailed(client=client, body=body, use_gzip=True)
        assert response.status_code == 200
        assert response.parsed is not None


def test_cyrillic_embeddings(server_available):
    client = Client(base_url=pytest.URL)

    with client as client:
        model_id = _first_embed_model_id(client)
        body = OpenAIEmbeddingInputText(
            input_=["Привет, как дела? Ёжик любит чай."],
            model=model_id,
        )
        response = embeddings.sync_detailed(client=client, body=body)
        assert response.status_code == 200
        assert response.parsed is not None
