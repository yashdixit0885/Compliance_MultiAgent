import os
from typing import TypedDict, Literal
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# ... (Keep your existing State and Node definitions)

load_dotenv()
memory = MemorySaver()
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

app = workflow.compile(
    checkpointer=memory,
    interrupt_after=["compliance"]
)

# --- 4. TEST IT ---
if __name__ == "__main__":
    # A 'thread_id' represents a specific conversation or transaction ID
    config = {"configurable": {"thread_id": "tx_999"}}
    
    inputs = {"user_request": "Move $8,000 from Checking to Savings"}
    
    print("\n--- STARTING TRANSACTION ---")
    # We run the graph until it hits the breakpoint
    for event in app.stream(inputs, config):
        for node, state in event.items():
            print(f"\nNode: {node}")
            if node == "compliance":
                print(f"Auditor Report: {state['compliance_report']}")

    # --- THE GRAPH IS NOW PAUSED ---
    print("\n--- SYSTEM PAUSED: WAITING FOR HUMAN APPROVAL ---")
    snapshot = app.get_state(config)
    print(f"Current Status in Memory: {snapshot.values['status']}")
    
    # Simulate a human checking the dashboard and typing 'yes'
    confirm = input("\nType 'YES' to finalize this transaction or 'NO' to cancel: ")
    
    if confirm.upper() == "YES":
        print("\n--- RESUMING WORKFLOW ---")
        # Passing 'None' as input tells LangGraph to just pick up where it left off
        for event in app.stream(None, config):
            print(event)
        print("\n--- TRANSACTION COMPLETE ---")
    else:
        print("\n--- TRANSACTION ABORTED BY HUMAN ---")