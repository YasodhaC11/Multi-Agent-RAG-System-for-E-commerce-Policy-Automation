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
#  MODELS
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

retriever = vectordb.as_retriever(search_kwargs={"k": 4})

# =========================
#  NODE 1: TRIAGE
# =========================

def triage_node(state: State):
    prompt = f"""
    You are a strict JSON generator.

    Return ONLY valid JSON. No explanation. No text before or after.

    Schema:
    {{
     "issue_type": "string",
     "confidence": float (0 to 1),
     "clarifying_questions": list of strings
    }}

    Ticket: {state['ticket_text']}
    Order: {state['order_context']}
    """
    response = llm.invoke(prompt)

    try:
        result = json.loads(response.content)
    except:
        result = {
            "issue_type": "unknown",
            "confidence": 0.5,
            "clarifying_questions": []
        }

    return {
        "classification": result,
        "clarifying_questions": result.get("clarifying_questions", []),
        "retry_count": 0
    }


# =========================
# NODE 2: RETRIEVER
# =========================

def retriever_node(state: State):
    docs = retriever.invoke(state["ticket_text"])

    retrieved = []
    for d in docs:
        retrieved.append({
            "text": d.page_content,
            "document": d.metadata.get("document"),
            "chunk_id": d.metadata.get("chunk_id")
        })

    return {"retrieved_docs": retrieved}


# =========================
# NODE 3: GENERATOR
# =========================

def generator_node(state: State):
    context = "\n\n".join([d["text"] for d in state["retrieved_docs"]])

    prompt = f"""
You are a STRICT policy-based support agent.

RULES:
- Use ONLY provided context
- Every claim MUST include citation
- If insufficient info → needs_escalation
- Output ONLY JSON

Context:
{context}

Ticket:
{state['ticket_text']}

Order Context:
{state['order_context']}

JSON:
{{
 "decision": "approve | deny | partial | needs_escalation",
 "rationale": "...",
 "citations": [
    {{"document": "...", "chunk_id": "..."}}
 ],
 "customer_response": "...",
 "internal_notes": "..."
}}
"""

    response = llm.invoke(prompt)

    try:
        return json.loads(response.content)
    except:
        return {"decision": "needs_escalation"}


# =========================
#  NODE 4: VERIFIER
# =========================

def verifier_node(state: State):
    retry_count = state.get("retry_count", 0)

    #  1. Clarifying questions → stop
    if state.get("clarifying_questions"):
        return {"decision": "needs_clarification"}

    #  2. Conflict handling
    if state["order_context"].get("seller_policy_override"):
        return {"decision": "needs_escalation"}

    #  3. Minimum evidence check
    if len(state.get("retrieved_docs", [])) < 2:
        return {"decision": "needs_escalation"}

    #  4. Missing citations
    if not state.get("citations"):
        return {"decision": "retry", "retry_count": retry_count + 1}

    #  5. Validate citation fields
    for c in state["citations"]:
        if not c.get("document") or not c.get("chunk_id"):
            return {"decision": "retry", "retry_count": retry_count + 1}

    #  6. Weak rationale
    if len(state.get("rationale", "")) < 20:
        return {"decision": "retry", "retry_count": retry_count + 1}

    #  7. Retry limit
    if retry_count >= 2:
        return {"decision": "needs_escalation"}

    return {"decision": "valid"}


# =========================
#  ROUTING
# =========================

def route_after_verifier(state: State):
    decision = state.get("decision")
    retry_count = state.get("retry_count", 0)

    print(f"Routing decision: {decision}, retries: {retry_count}")

    # Safety: prevent infinite loop
    if retry_count >= 2:
        return END

    if decision == "retry":
        return "generate"

    if decision == "needs_clarification":
        return END

    if decision == "needs_escalation":
        return END

    if decision == "valid":
        return END

    # fallback
    return END
# =========================
#  BUILD GRAPH
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
#  RUN SAMPLE
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

    result = graph.invoke(input_data)

    print("\n✅ FINAL OUTPUT:\n")
    print(json.dumps(result, indent=2))