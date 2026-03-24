import pytest
from asgi_lifespan import LifespanManager  # type: ignore[import-untyped]
from httpx import ASGITransport, AsyncClient
from sentence_transformers.sparse_encoder import SparseEncoder  # type: ignore[import-untyped]

from infinity_emb import create_server
from infinity_emb.args import EngineArgs
from infinity_emb.primitives import Device, InferenceEngine

MODEL = "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1"
PREFIX = "/v1_sparse_embeddings"
QUERY = "What's the weather in ny now?"
DOCUMENT = "Currently New York is rainy."

app = create_server(
    url_prefix=PREFIX,
    engine_args_list=[
        EngineArgs(
            model_name_or_path=MODEL,
            batch_size=2,
            model_warmup=False,
            engine=InferenceEngine.torch,
            device=Device.cpu,
        )
    ],
)


@pytest.fixture()
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test", timeout=120
    ) as client, LifespanManager(app):
        yield client


def _sparse_dot(lhs: dict, rhs: dict) -> float:
    rhs_map = {idx: val for idx, val in zip(rhs["indices"], rhs["values"])}
    return sum(val * rhs_map.get(idx, 0.0) for idx, val in zip(lhs["indices"], lhs["values"]))


async def _embed_sparse(client: AsyncClient, text: str, task: str, prune_ratio: float = 0.0) -> dict:
    response = await client.post(
        f"{PREFIX}/sparse_embeddings",
        json={"model": MODEL, "input": [text], "prune_ratio": prune_ratio, "task": task},
    )
    response.raise_for_status()
    data = response.json()
    return data["data"][0]["embedding"]


@pytest.mark.anyio
async def test_sparse_embeddings_similarity_matches_sparseencoder_snippet(client):
    # API similarity from sparse vectors
    query_sparse = await _embed_sparse(client, QUERY, task="query")
    doc_sparse = await _embed_sparse(client, DOCUMENT, task="document")
    api_similarity = _sparse_dot(query_sparse, doc_sparse)

    # Reference similarity from the official SparseEncoder snippet behavior
    model = SparseEncoder(MODEL)
    query_embed = model.encode_query(QUERY)
    document_embed = model.encode_document(DOCUMENT)
    reference_similarity = float(model.similarity(query_embed, document_embed).item())

    assert api_similarity > 0
    assert abs(api_similarity - reference_similarity) <= 5e-2


@pytest.mark.anyio
async def test_sparse_embeddings_with_prune_ratio(client):
    query_sparse = await _embed_sparse(client, QUERY, task="query", prune_ratio=0.0)
    doc_sparse = await _embed_sparse(client, DOCUMENT, task="document", prune_ratio=0.0)
    sim_no_prune = _sparse_dot(query_sparse, doc_sparse)

    query_sparse_pruned = await _embed_sparse(client, QUERY, task="query", prune_ratio=0.1)
    doc_sparse_pruned = await _embed_sparse(client, DOCUMENT, task="document", prune_ratio=0.1)
    sim_pruned = _sparse_dot(query_sparse_pruned, doc_sparse_pruned)

    assert len(query_sparse_pruned["indices"]) <= len(query_sparse["indices"])
    assert len(doc_sparse_pruned["indices"]) <= len(doc_sparse["indices"])
    query_threshold = max(query_sparse["values"]) * 0.1
    doc_threshold = max(doc_sparse["values"]) * 0.1
    assert all(v > query_threshold for v in query_sparse_pruned["values"])
    assert all(v > doc_threshold for v in doc_sparse_pruned["values"])
    assert sim_no_prune > 0
    assert sim_pruned > 0
