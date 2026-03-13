import os
from typing import TypedDict, Literal
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

load_dotenv()

# 1. Setup the "Committee"
# Using temperature=0 for the Auditor to ensure consistent compliance checks
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# 2. Define the "Shared Memory" (State)
class CommitteeState(TypedDict):
    user_request: str
    proposed_transaction: str
    compliance_report: str
    status: Literal["approved", "rejected"]
    retry_count: int

# --- NODE 1: The Trader (Planner) ---
def trader_node(state: CommitteeState):
    print("--- TRADER: PROPOSING TRANSACTION ---")
    prompt = f"""
    You are a Financial Trader. User Request: {state['user_request']}
    Propose a transaction in JSON format: {{'amount': X, 'from': 'Y', 'to': 'Z'}}.
    """
    response = llm.invoke(prompt)
    return {"proposed_transaction": response.content, "retry_count": state.get('retry_count', 0) + 1}

# --- NODE 2: The Compliance Officer (Auditor) ---
def compliance_node(state: CommitteeState):
    print("--- COMPLIANCE: AUDITING TRANSACTION ---")
    # This represents the "Safety & Guardrails" section of your JD
    policy = """
    POLICY RULES:
    1. No transaction shall exceed $10,000.
    2. Transfers to 'Savings' are always allowed.
    3. Transfers to 'Crypto' are STRICTLY PROHIBITED.
    """
    
    prompt = f"""
    Proposed Transaction: {state['proposed_transaction']}
    Policy: {policy}
    
    If it violates policy, start with 'REJECTED' and explain why.
    If it passes, start with 'APPROVED'.
    """
    response = llm.invoke(prompt)
    
    status = "approved" if "APPROVED" in response.content.upper() else "rejected"
    return {"compliance_report": response.content, "status": status}

# --- ROUTER: The "Gatekeeper" ---
def route_approval(state: CommitteeState):
    if state["status"] == "approved":
        return "end"
    else:
        # If it's a small rejection, we could send it back to the trader, 
        # but for now, we terminate the flow for safety.
        return "terminate"

# 3. Build the Graph
workflow = StateGraph(CommitteeState)

workflow.add_node("trader", trader_node)
workflow.add_node("compliance", compliance_node)

workflow.set_entry_point("trader")
workflow.add_edge("trader", "compliance")

workflow.add_conditional_edges(
    "compliance",
    route_approval,
    {
        "end": END,
        "terminate": END
    }
)

app = workflow.compile()

# --- 4. TEST IT ---
if __name__ == "__main__":
    # Test 1: The Failure Path (Prohibited Destination)
    print("\n--- TEST: CRYPTO TRANSFER ---")
    inputs = {"user_request": "Move $5,000 from Checking to my Crypto wallet"}
    for output in app.stream(inputs):
        print(output)

    # Test 2: The Failure Path (Limit Exceeded)
    print("\n--- TEST: LIMIT EXCEEDED ---")
    inputs = {"user_request": "Move $50,000 from Checking to Savings"}
    for output in app.stream(inputs):
        print(output)
        
    # Test 3: The Success Path (Allowed Transaction)
    print("\n--- TEST: ALLOWED TRANSACTION ---")
    inputs = {"user_request": "Move $5,000 from Checking to Savings"}
    for output in app.stream(inputs):
        print(output)