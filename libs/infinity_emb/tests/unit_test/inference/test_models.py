"""
Tests that the pretrained models produce the correct scores on the STSbenchmark dataset
"""

import copy
import sys

import pytest
from sentence_transformers import InputExample  # type: ignore[import-untyped]
from sentence_transformers.evaluation import (  # type: ignore[import-untyped]
    EmbeddingSimilarityEvaluator,
)

from infinity_emb.args import EngineArgs
from infinity_emb.transformer.embedder.sentence_transformer import (
    SentenceTransformerPatched,
)


def _pretrained_model_score(
    dataset: list[InputExample],
    model_name,
    expected_score,
):
    test_samples = dataset[::3]
    model = SentenceTransformerPatched(
        engine_args=EngineArgs(
            model_name_or_path=model_name,
        )
    )
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name="sts-test")

    score = model.evaluate(evaluator)["sts-test_spearman_cosine"] * 100  # type: ignore
    print(model_name, "{:.2f} vs. exp: {:.2f}".format(score, expected_score))
    assert score > expected_score or abs(score - expected_score) < 0.01


@pytest.mark.parametrize(
    "model,score",
    [
        ("sentence-transformers/bert-base-nli-mean-tokens", 76.46),
        ("sentence-transformers/all-MiniLM-L6-v2", 81.03),
        ("michaelfeil/bge-small-en-v1.5", 84.90),
    ],
)
@pytest.mark.skipif(sys.platform == "darwin", reason="does not run on mac")
def test_bert(get_sts_bechmark_dataset, model, score):
    samples = copy.deepcopy(get_sts_bechmark_dataset[2])
    _pretrained_model_score(samples, model, score)
