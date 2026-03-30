import json
from main import graph


def evaluate():
    with open("evaluation_dataset.json") as f:
        data = json.load(f)

    total = len(data)
    citation_count = 0
    unsupported_claims = 0
    escalation_total = 0
    escalation_correct = 0
    failed_samples = 0

    print("\n Running Evaluation...\n")
    print("=" * 60)

    for i, sample in enumerate(data):
        sample_type = sample.get("type", "unknown")
        label = sample.get("ticket_text", "")[:60]

        try:
            # Only pass what graph expects
            input_data = {
                "ticket_text": sample["ticket_text"],
                "order_context": sample["order_context"]
            }
            result = graph.invoke(input_data)

            has_citations = bool(result.get("citations"))
            decision = result.get("decision", "")

            # Citation coverage
            if has_citations:
                citation_count += 1
            else:
                unsupported_claims += 1

            # Escalation accuracy — only for escalation-type samples
            if sample_type in ["conflict", "not_in_policy"]:
                escalation_total += 1
                correct = decision == "needs_escalation"
                if correct:
                    escalation_correct += 1
                status = "✅" if correct else "❌"
            else:
                status = "✅" if has_citations else "⚠️"

            print(f"[{i+1}] {status} | type={sample_type} | decision={decision} | citations={has_citations}")
            print(f"     ticket: {label}...")

        except Exception as e:
            failed_samples += 1
            print(f"[{i+1}] ❌ ERROR | type={sample_type} | {str(e)}")

        print()

    # ========================
    # Final Metrics
    # ========================
    print("=" * 60)
    print("\n📊 Evaluation Results:\n")
    print(f"Total Samples         : {total}")
    print(f"Failed (Errors)       : {failed_samples}")
    print(f"Evaluated             : {total - failed_samples}")
    print()
    print(f"Citation Coverage     : {citation_count}/{total} "
          f"= {citation_count / total:.0%}")
    print(f"Unsupported Claim Rate: {unsupported_claims}/{total} "
          f"= {unsupported_claims / total:.0%}")

    if escalation_total > 0:
        print(f"Escalation Accuracy   : {escalation_correct}/{escalation_total} "
              f"= {escalation_correct / escalation_total:.0%}")
    else:
        print("Escalation Accuracy   : N/A (no escalation samples found)")


if __name__ == "__main__":
    evaluate()