import csv

from final_project.backend.config import BackendSettings, active_llm_model
from final_project.evaluation.collect_results import collect
from final_project.evaluation.dataset import load_evaluation_questions
from final_project.evaluation.judge_results import parse_judge_json, summarize
from final_project.evaluation.ragas_results import make_llm, record_to_sample_payload
from final_project.evaluation.metrics import retrieval_metrics, source_matches
from final_project.evaluation.strategies import (
    _fuse_ranked_chunks,
    _parse_agent_decision,
    _parse_query_list,
)


def test_load_evaluation_questions_groups_blank_rows(tmp_path):
    dataset = tmp_path / "qa.csv"
    with dataset.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["Question Nr", "Question", "Type (narrow/broad)", "Answer", "Source"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "Question Nr": "1.0",
                "Question": "What do dogs look at?",
                "Type (narrow/broad)": "broad",
                "Answer": "They look at eyes.",
                "Source": "Somppi et al. 2016",
            }
        )
        writer.writerow(
            {
                "Question Nr": "",
                "Question": "",
                "Type (narrow/broad)": "",
                "Answer": "They also scan mouths.",
                "Source": "Barber et al. 2016",
            }
        )

    questions = load_evaluation_questions(dataset)

    assert len(questions) == 1
    assert questions[0].question_id == "1"
    assert questions[0].reference_answers == ["They look at eyes.", "They also scan mouths."]
    assert questions[0].expected_sources == ["Somppi et al. 2016", "Barber et al. 2016"]


def test_source_matching_handles_normalized_titles():
    assert source_matches("Somppi et al. 2016", "Somppi et al. 2016")
    assert source_matches("Somppi_et_al._2016", "Somppi et al. 2016")


def test_retrieval_metrics_are_paper_level():
    chunks = [
        {"title": "Wrong paper"},
        {"title": "Somppi et al. 2016"},
        {"title": "Somppi et al. 2016"},
    ]

    metrics = retrieval_metrics(chunks, ["Somppi et al. 2016", "Barber et al. 2016"])

    assert metrics["matched_source_count"] == 1
    assert metrics["retrieval_recall"] == 0.5
    assert metrics["mrr"] == 0.5


def test_parse_agent_decision_extracts_json_from_text():
    decision = _parse_agent_decision(
        'Here is JSON: {"sufficient": false, "reason": "missing", "improved_query": "dog face gaze eyes"}',
        fallback_query="original",
    )

    assert decision["sufficient"] is False
    assert decision["improved_query"] == "dog face gaze eyes"


def test_backend_settings_can_hold_evaluation_defaults():
    settings = BackendSettings(llm_provider="mock", embedding_provider="hash")

    assert settings.llm_provider == "mock"
    assert settings.embedding_provider == "hash"


def test_active_llm_model_uses_provider_specific_model():
    openai_settings = BackendSettings(
        llm_provider="openai",
        llm_model_name="gpt-5.6-terra",
        openai_compatible_model="gpt-oss-120b",
    )
    compatible_settings = BackendSettings(
        llm_provider="openai_compatible",
        llm_model_name="gpt-5.6-terra",
        openai_compatible_model="gpt-oss-120b",
    )

    assert active_llm_model(openai_settings) == "gpt-5.6-terra"
    assert active_llm_model(compatible_settings) == "gpt-oss-120b"


def test_parse_query_list_keeps_original_and_deduplicates():
    queries = _parse_query_list(
        '{"queries": ["dog gaze human faces", "dog gaze human faces", "canine facial attention"]}',
        fallback_query="Where do dogs look?",
        max_queries=3,
    )

    assert queries == [
        "Where do dogs look?",
        "dog gaze human faces",
        "canine facial attention",
    ]


def test_fuse_ranked_chunks_prefers_repeated_high_ranked_chunks():
    fused = _fuse_ranked_chunks(
        [
            [{"chunk_id": "a", "text": "A"}, {"chunk_id": "b", "text": "B"}],
            [{"chunk_id": "b", "text": "B"}, {"chunk_id": "c", "text": "C"}],
        ],
        top_n=2,
    )

    assert [chunk["chunk_id"] for chunk in fused] == ["b", "a"]
    assert fused[0]["fusion_score"] > fused[1]["fusion_score"]


def test_parse_judge_json_handles_fenced_json():
    parsed = parse_judge_json(
        """```json
        {
          "answer_correctness": 4,
          "faithfulness": 5,
          "answer_relevance": 4,
          "context_usefulness": 3,
          "completeness": 4,
          "overall_quality": 4,
          "reason": "Grounded answer."
        }
        ```"""
    )

    assert parsed["faithfulness"] == 5.0
    assert parsed["reason"] == "Grounded answer."


def test_judge_summary_groups_by_strategy():
    rows = [
        {
            "strategy": "classic",
            "answer_correctness": 3,
            "faithfulness": 5,
            "answer_relevance": 4,
            "context_usefulness": 3,
            "completeness": 2,
            "overall_quality": 3,
        },
        {
            "strategy": "classic",
            "answer_correctness": 5,
            "faithfulness": 3,
            "answer_relevance": 4,
            "context_usefulness": 5,
            "completeness": 4,
            "overall_quality": 5,
        },
    ]

    summary = summarize(rows)

    assert summary[0]["strategy"] == "classic"
    assert summary[0]["n_questions"] == 2
    assert summary[0]["mean_answer_correctness"] == 4


def test_collect_results_merges_available_summaries(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    (evaluation_dir / "llm_judge").mkdir(parents=True)
    (evaluation_dir / "ragas").mkdir(parents=True)

    with (evaluation_dir / "rag_strategy_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["strategy", "n_questions", "mean_retrieval_recall", "mean_mrr"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "strategy": "classic",
                "n_questions": "7",
                "mean_retrieval_recall": "0.5",
                "mean_mrr": "1.0",
            }
        )

    with (evaluation_dir / "llm_judge" / "llm_judge_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["strategy", "n_questions", "mean_overall_quality"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "strategy": "classic",
                "n_questions": "7",
                "mean_overall_quality": "4.25",
            }
        )

    rows = collect(evaluation_dir)

    assert rows == [
        {
            "strategy": "classic",
            "custom_n_questions": 7.0,
            "custom_mean_retrieval_recall": 0.5,
            "custom_mean_mrr": 1.0,
            "judge_n_questions": 7.0,
            "judge_mean_overall_quality": 4.25,
        }
    ]


def test_openai_ragas_wrapper_exposes_generate_method():
    settings = BackendSettings(
        llm_provider="openai",
        llm_model_name="test-model",
        openai_api_key="dummy",
    )

    llm = make_llm(settings, max_tokens=100)

    assert hasattr(llm, "generate")
    assert hasattr(llm, "generate_text")
    assert hasattr(llm, "agenerate_text")


def test_ragas_payload_truncates_long_reference():
    record = {
        "question": "What matters?",
        "answer": "Context matters.",
        "chunks": [{"text": "A useful retrieved context."}],
        "benchmark": {"reference_answer": "word " * 1000},
    }

    payload = record_to_sample_payload(
        record,
        max_context_chars=100,
        max_reference_chars=50,
    )

    assert payload["reference"].endswith("[truncated]")
    assert len(payload["reference"]) < 80
