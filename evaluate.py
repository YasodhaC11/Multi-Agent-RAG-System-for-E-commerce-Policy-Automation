import json
from main import graph

def evaluate():
    with open("evaluation_dataset.json") as f:
        data = json.load(f)

    total = len(data)
    citation_count = 0
    escalation_correct = 0
    unsupported_claims = 0

    for sample in data:
        result = graph.invoke(sample)

        #  Citation coverage
        if result.get("citations"):
            citation_count += 1

        #  Escalation correctness
        if sample["type"] in ["conflict", "not_in_policy"]:
            if result["decision"] == "needs_escalation":
                escalation_correct += 1

        #  Unsupported claim
        if not result.get("citations"):
            unsupported_claims += 1

    print("\n Evaluation Results:\n")

    print("Citation Coverage:", citation_count / total)
    print("Escalation Accuracy:", escalation_correct / 6)  # 3 conflict + 3 not_in_policy
    print("Unsupported Claim Rate:", unsupported_claims / total)


if __name__ == "__main__":
    evaluate()