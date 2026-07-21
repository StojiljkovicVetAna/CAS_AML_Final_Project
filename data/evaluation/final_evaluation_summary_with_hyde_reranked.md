# Final Evaluation Summary

| strategy | custom_mean_retrieval_recall | custom_mean_retrieval_precision | custom_mean_mrr | custom_mean_matched_source_count | judge_mean_answer_correctness | judge_mean_faithfulness | judge_mean_context_usefulness | judge_mean_overall_quality | ragas_mean_faithfulness | ragas_mean_llm_context_precision_with_reference | ragas_mean_context_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classic | 0.481 | 0.8167 | 0.9286 | 2.5714 | 4.2857 | 4.0 | 4.7143 | 4.1429 | 0.8255 | 0.85 | 0.4123 |
| hyde | 0.5095 | 0.8452 | 1.0 | 2.4286 | 4.5714 | 4.4286 | 5.0 | 4.5714 | 0.8874 | 0.8524 | 0.4136 |
| hyde_reranked | 0.4714 | 0.8952 | 1.0 | 2.4286 | 4.5714 | 5.0 | 5.0 | 4.5714 | 0.9128 | 0.8571 | 0.3461 |
| multi_query | 0.4714 | 0.7857 | 0.8571 | 2.4286 | 4.2857 | 4.1429 | 4.8571 | 4.2857 | 0.8404 | 0.8222 | 0.3019 |
| reranked | 0.5286 | 0.8 | 0.9286 | 2.8571 | 4.5714 | 4.4286 | 5.0 | 4.4286 | 0.9017 | 0.8419 | 0.3749 |

Notes:
- `custom_*` metrics are deterministic paper-level retrieval metrics from the benchmark source labels.
- `judge_*` metrics are 1-5 LLM-as-judge scores.
- Empty `ragas_*` values mean RAGAS did not return valid scores for that metric.
