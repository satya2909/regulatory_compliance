from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from app import ragas_data

dataset = Dataset.from_dict(ragas_data)

print("Columns:", dataset.column_names)

results = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision
    ]
)

print(results)
