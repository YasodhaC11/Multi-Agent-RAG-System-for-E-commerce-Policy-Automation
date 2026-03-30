import os
import json
from typing import TypedDict, List
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END


# =========================
# STATE
# =========================
class State(TypedDict):
    ticket_text: str
    order_context: dict
    classification: dict
    clarifying_questions: List[str]
    retrieved_docs: List[dict]
    decision: str
    rationale: str
    citations: List[dict]
    customer_response: str
    internal_notes: str
    retry_count: int


# =========================
# MODELS
# =========================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    response_format={"type": "json_object"}
)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectordb = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 10}
)


# =========================
# NODE 1: TRIAGE
# =========================

def triage_node(state: State):
    prompt = f"""
You are a strict JSON generator.

Return ONLY valid JSON. No explanation. No text before or after.

Schema:
{{
  "issue_type": "string (e.g. refund, return, replacement, damaged, wrong_item, shipping, other)",
  "confidence": float between 0 and 1,
  "clarifying_questions": list of strings (empty list if no clarification needed)
}}

Ticket: {state['ticket_text']}
Order: {state['order_context']}
"""
    response = llm.invoke(prompt)

    try:
        result = json.loads(response.content)
    except Exception:
        result = {
            "issue_type": "unknown",
            "confidence": 0.5,
            "clarifying_questions": []
        }

    return {
        "classification": result,
        "clarifying_questions": result.get("clarifying_questions", []),
        "retry_count": 0  }


# =========================
# NODE 2: RETRIEVER
# =========================

def retriever_node(state: State):
    issue_type = state.get("classification", {}).get("issue_type", "")
    ticket = state["ticket_text"]
    category = state.get("order_context", {}).get("item_category", "")
    query = f"{category} {issue_type} {ticket}".strip()

    docs = retriever.invoke(query)

    retrieved = []
    for d in docs:
        retrieved.append({
            "text": d.page_content,
            "document": d.metadata.get("document", "unknown"),
            "chunk_id": d.metadata.get("chunk_id", "N/A")
        })

    return {"retrieved_docs": retrieved}


# =========================
# NODE 3: GENERATOR
# =========================

def generator_node(state: State):
    context = "\n\n".join(
        f"[Source: {d['document']} | Chunk: {d['chunk_id']}]\n{d['text']}"
        for d in state["retrieved_docs"]
    )

    prompt = f"""
You are a STRICT policy-based e-commerce support agent.

RULES:
- Use ONLY the provided policy context below.
- Every claim in your rationale MUST reference a document and chunk_id from context.
- If context is insufficient to make a decision → use "needs_escalation".
- Output ONLY valid JSON. No explanation outside JSON.

Policy Context:
{context}

Customer Ticket:
{state['ticket_text']}

Order Context:
{json.dumps(state['order_context'])}

Return this exact JSON structure:
{{
  "decision": "approve | deny | partial | needs_escalation",
  "rationale": "Detailed explanation referencing policy context",
  "citations": [
    {{"document": "document_name", "chunk_id": "chunk_id_value"}}
  ],
  "customer_response": "Polite response to send to the customer",
  "internal_notes": "Notes for internal support team"
}}
"""

    response = llm.invoke(prompt)


    try:
        result = json.loads(response.content)
    except Exception:
        result = {}

    return {
        "decision": result.get("decision", "needs_escalation"),
        "rationale": result.get("rationale", ""),
        "citations": result.get("citations", []),
        "customer_response": result.get("customer_response", ""),
        "internal_notes": result.get("internal_notes", "")
    }


# =========================
# NODE 4: VERIFIER
# =========================

def verifier_node(state: State):
    retry_count = state.get("retry_count", 0)

    # 1. Clarifying questions needed → stop and ask customer
    if state.get("clarifying_questions"):
        return {"decision": "needs_clarification"}

    # 2. Seller policy conflict → escalate
    if state.get("order_context", {}).get("seller_policy_override"):
        return {"decision": "needs_escalation"}

    # 3. Not enough retrieved docs → escalate
    if len(state.get("retrieved_docs", [])) < 2:
        return {"decision": "needs_escalation"}

    # 4. Missing citations → retry
    citations = state.get("citations", [])
    if not citations:
        return {"decision": "retry", "retry_count": retry_count + 1}

    # 5. Invalid citation fields → retry
    for c in citations:
        if not c.get("document") or not c.get("chunk_id"):
            return {"decision": "retry", "retry_count": retry_count + 1}

    # 6. Weak rationale → retry
    if len(state.get("rationale", "")) < 20:
        return {"decision": "retry", "retry_count": retry_count + 1}

    # 7. Retry limit reached → escalate
    if retry_count >= 2:
        return {"decision": "needs_escalation"}

    return {"decision": "valid"}


# =========================
# ROUTING
# =========================

def route_after_verifier(state: State):
    decision = state.get("decision")
    retry_count = state.get("retry_count", 0)

    print(f"[Router] Decision: {decision} | Retries: {retry_count}")

    if retry_count >= 2:
        return END

    if decision == "retry":
        return "generate"

    # All terminal states → END
    return END


# =========================
# BUILD GRAPH
# =========================

builder = StateGraph(State)

builder.add_node("triage", triage_node)
builder.add_node("retrieve", retriever_node)
builder.add_node("generate", generator_node)
builder.add_node("verify", verifier_node)

builder.set_entry_point("triage")

builder.add_edge("triage", "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", "verify")

builder.add_conditional_edges("verify", route_after_verifier)

graph = builder.compile()


# =========================
# RUN SAMPLE
# =========================

if __name__ == "__main__":
    input_data = {
        "ticket_text": "My order arrived late and cookies are melted. I want full refund.",
        "order_context": {
            "item_category": "perishable",
            "order_status": "delivered",
            "seller_policy_override": False
        }
    }

    print("\n Running pipeline...\n")
    result = graph.invoke(input_data)

    print("\n FINAL OUTPUT:\n")
    print(json.dumps(result, indent=2))