from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import json
import pandas as pd
from datetime import datetime
from google.cloud import bigquery
import os

# ==================== STATE DEFINITION ====================

class PolicyLookupState(TypedDict):
    trigger_codes: List[Dict[str, Any]]  # Input from Module 2
    policy_instructions: List[Dict[str, Any]]  # Retrieved instructions
    access_method_used: str  # "json", "bigquery", "rag"
    processing_metadata: Dict[str, Any]
    errors: List[str]

# ==================== METHOD 1: JSON FILE ACCESS ====================

@tool
def load_policies_json(file_path: str = "policies.json") -> Dict[str, Any]:
    """Loads complete policies from JSON file with comprehensive trigger code instructions."""
    
    # Mock comprehensive policies JSON - in real implementation, load from actual file
    mock_policies_data = {
        "TRG-001": {
            "trigger_code": "TRG-001",
            "priority": "HIGH",
            "category": "high_priority_review",
            "estimated_hours": 2.0,
            "instructions": {
                "step_1": {
                    "action": "Immediate priority review for Form-A, ID-Copy",
                    "details": "Flag claim for high-priority processing within 2 hours. Notify supervisor immediately and update claim status to URGENT.",
                    "required_documents": ["Form-A", "ID-Copy", "Medical-Records"],
                    "systems": ["ClaimDB", "NotificationSystem"],
                    "checkpoints": ["Supervisor notified", "Status updated to URGENT"],
                    "time_estimate_minutes": 15
                },
                "step_2": {
                    "action": "Comprehensive document validation",
                    "details": "Verify all required documentation is present, properly signed, and meets compliance standards.",
                    "required_documents": ["Form-A", "ID-Copy", "Medical-Records", "Supporting-Evidence"],
                    "systems": ["DocumentDB", "ComplianceChecker"],
                    "checkpoints": ["All docs verified", "Compliance check passed"],
                    "time_estimate_minutes": 45
                },
                "step_3": {
                    "action": "Supervisor review and approval",
                    "details": "Schedule immediate supervisor review for final approval before processing.",
                    "systems": ["SchedulingSystem", "ApprovalWorkflow"],
                    "checkpoints": ["Supervisor meeting scheduled", "Review completed"],
                    "time_estimate_minutes": 60
                }
            },
            "escalation_triggers": ["Missing critical documents", "Compliance failure"],
            "required_roles": ["Claim Processor", "Supervisor"],
            "policy_version": "v2.1",
            "last_updated": "2025-06-01"
        },
        "CLM-456": {
            "trigger_code": "CLM-456", 
            "priority": "MEDIUM",
            "category": "standard_processing",
            "estimated_hours": 0.8,
            "instructions": {
                "step_1": {
                    "action": "Standard claim processing initialization",
                    "details": "Open claim file, verify claimant identity, and validate policy status.",
                    "required_documents": ["Claim-Form", "ID-Verification"],
                    "systems": ["ClaimDB", "PolicyDB"],
                    "checkpoints": ["Identity verified", "Policy active"],
                    "time_estimate_minutes": 20
                },
                "step_2": {
                    "action": "Coverage assessment and calculation",
                    "details": "Review policy coverage limits and calculate claim eligibility based on terms.",
                    "systems": ["PolicyDB", "CoverageCalculator", "PaymentSystem"],
                    "checkpoints": ["Coverage confirmed", "Amount calculated"],
                    "time_estimate_minutes": 30
                }
            },
            "escalation_triggers": ["Coverage disputes", "Calculation errors"],
            "required_roles": ["Claim Processor"],
            "policy_version": "v1.8",
            "last_updated": "2025-03-15"
        },
        "TRG-789": {
            "trigger_code": "TRG-789",
            "priority": "LOW",
            "category": "documentation_request",
            "estimated_hours": 0.6,
            "instructions": {
                "step_1": {
                    "action": "Additional documentation request",
                    "details": "Contact claimant via phone/email to request missing documentation required for claim processing.",
                    "required_documents": ["Contact-Information"],
                    "systems": ["CommunicationSystem", "DocumentTracker"],
                    "checkpoints": ["Contact attempted", "Request sent", "Tracking initiated"],
                    "time_estimate_minutes": 25
                },
                "step_2": {
                    "action": "Set pending status with follow-up schedule",
                    "details": "Update claim status to 'Pending Documentation' and schedule follow-up in 5 business days.",
                    "systems": ["ClaimDB", "TaskScheduler"],
                    "checkpoints": ["Status updated", "Follow-up scheduled"],
                    "time_estimate_minutes": 10
                }
            },
            "escalation_triggers": ["No response after 10 days"],
            "required_roles": ["Claim Processor"],
            "policy_version": "v1.5",
            "last_updated": "2024-12-01"
        },
        "CLAIM-123": {
            "trigger_code": "CLAIM-123",
            "priority": "HIGH",
            "category": "fraud_investigation",
            "estimated_hours": 4.0,
            "instructions": {
                "step_1": {
                    "action": "Fraud investigation initiation",
                    "details": "Flag claim for potential fraud and initiate investigation protocol.",
                    "systems": ["FraudDB", "InvestigationSystem"],
                    "checkpoints": ["Fraud flag set", "Investigation opened"],
                    "time_estimate_minutes": 30
                },
                "step_2": {
                    "action": "Evidence collection and analysis",
                    "details": "Gather all available evidence and perform detailed analysis.",
                    "required_documents": ["All-Supporting-Evidence", "Investigation-Forms"],
                    "systems": ["EvidenceDB", "AnalysisTools"],
                    "checkpoints": ["Evidence collected", "Analysis completed"],
                    "time_estimate_minutes": 180
                }
            },
            "escalation_triggers": ["Confirmed fraud indicators"],
            "required_roles": ["Senior Investigator", "Supervisor"],
            "policy_version": "v3.0",
            "last_updated": "2025-05-15"
        }
    }
    
    try:
        # In real implementation:
        # with open(file_path, 'r') as f:
        #     return json.load(f)
        
        return {
            "status": "success",
            "data": mock_policies_data,
            "total_policies": len(mock_policies_data),
            "source": "json_file",
            "loaded_at": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "data": {},
            "source": "json_file"
        }

@tool
def batch_extract_from_json(policies_data: Dict[str, Any], trigger_codes: List[str]) -> Dict[str, Any]:
    """Extracts multiple trigger codes from JSON data in batch operation."""
    
    if policies_data.get("status") != "success":
        return {
            "results": {},
            "found_codes": [],
            "missing_codes": trigger_codes,
            "error": "Failed to load JSON policies data",
            "source": "json_file"
        }
    
    policies = policies_data.get("data", {})
    results = {}
    found_codes = []
    missing_codes = []
    
    for code in trigger_codes:
        # Try exact match first
        if code in policies:
            results[code] = policies[code]
            found_codes.append(code)
        # Try case-insensitive match
        else:
            found_match = False
            for policy_code, policy_data in policies.items():
                if policy_code.upper() == code.upper():
                    results[code] = policy_data
                    found_codes.append(code)
                    found_match = True
                    break
            
            if not found_match:
                missing_codes.append(code)
    
    return {
        "results": results,
        "found_codes": found_codes,
        "missing_codes": missing_codes,
        "total_requested": len(trigger_codes),
        "found_count": len(found_codes),
        "success_rate": len(found_codes) / len(trigger_codes) if trigger_codes else 0,
        "source": "json_file"
    }

# ==================== METHOD 2: BIGQUERY DATABASE ACCESS ====================

@tool
def query_bigquery_policies(trigger_code: str) -> Dict[str, Any]:
    """Queries BigQuery for specific trigger code instructions."""
    
    # Mock BigQuery response - in real implementation, use actual BigQuery client
    mock_bigquery_data = {
        "TRG-001": {
            "trigger_code": "TRG-001",
            "instruction_json": json.dumps({
                "steps": [
                    {
                        "step_number": 1,
                        "action": "Immediate priority review",
                        "details": "Flag for high-priority processing",
                        "time_minutes": 15,
                        "systems": ["ClaimDB", "NotificationSystem"]
                    },
                    {
                        "step_number": 2,
                        "action": "Document validation",
                        "details": "Verify all required documentation",
                        "time_minutes": 45,
                        "systems": ["DocumentDB"]
                    }
                ]
            }),
            "priority": "HIGH",
            "estimated_hours": 2.0,
            "required_documents": "Form-A,ID-Copy,Medical-Records",
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-06-01T00:00:00Z"
        },
        "CLM-456": {
            "trigger_code": "CLM-456",
            "instruction_json": json.dumps({
                "steps": [
                    {
                        "step_number": 1,
                        "action": "Standard processing initialization",
                        "details": "Open claim file and verify basic information",
                        "time_minutes": 20,
                        "systems": ["ClaimDB"]
                    },
                    {
                        "step_number": 2,
                        "action": "Coverage assessment",
                        "details": "Review policy coverage limits",
                        "time_minutes": 30,
                        "systems": ["PolicyDB", "CoverageCalculator"]
                    }
                ]
            }),
            "priority": "MEDIUM",
            "estimated_hours": 0.8,
            "required_documents": "Claim-Form,Supporting-Evidence",
            "created_at": "2024-06-01T00:00:00Z",
            "updated_at": "2025-03-15T00:00:00Z"
        }
    }
    
    try:
        if trigger_code in mock_bigquery_data:
            raw_data = mock_bigquery_data[trigger_code]
            
            # Parse JSON instruction field
            instructions = json.loads(raw_data["instruction_json"])
            
            # Parse required documents
            docs = raw_data["required_documents"].split(",") if raw_data["required_documents"] else []
            
            return {
                "found": True,
                "trigger_code": trigger_code,
                "instruction_data": {
                    "trigger_code": trigger_code,
                    "priority": raw_data["priority"],
                    "estimated_hours": raw_data["estimated_hours"],
                    "instructions": instructions,
                    "required_documents": docs,
                    "metadata": {
                        "created_at": raw_data["created_at"],
                        "updated_at": raw_data["updated_at"],
                        "source": "bigquery"
                    }
                },
                "source": "bigquery"
            }
        else:
            return {
                "found": False,
                "trigger_code": trigger_code,
                "error": f"Trigger code {trigger_code} not found in BigQuery",
                "source": "bigquery"
            }
            
    except Exception as e:
        return {
            "found": False,
            "trigger_code": trigger_code,
            "error": f"BigQuery query failed: {str(e)}",
            "source": "bigquery"
        }

@tool
def batch_query_bigquery(trigger_codes: List[str]) -> Dict[str, Any]:
    """Batch queries BigQuery for multiple trigger codes efficiently."""
    
    # In real implementation, use single query with IN clause:
    # query = f"SELECT * FROM policies.instructions WHERE trigger_code IN ({','.join(['%s'] * len(trigger_codes))})"
    
    results = {}
    found_codes = []
    missing_codes = []
    errors = []
    
    for code in trigger_codes:
        try:
            result = query_bigquery_policies(code)
            if result and result.get("found"):
                results[code] = result["instruction_data"]
                found_codes.append(code)
            else:
                missing_codes.append(code)
                if result.get("error"):
                    errors.append(f"{code}: {result['error']}")
        except Exception as e:
            missing_codes.append(code)
            errors.append(f"{code}: Query failed - {str(e)}")
    
    return {
        "results": results,
        "found_codes": found_codes,
        "missing_codes": missing_codes,
        "total_requested": len(trigger_codes),
        "found_count": len(found_codes),
        "success_rate": len(found_codes) / len(trigger_codes) if trigger_codes else 0,
        "errors": errors,
        "source": "bigquery"
    }

# ==================== METHOD 3: RAG/VECTOR STORE ====================

class PolicyRAGSystem:
    """RAG system for semantic policy instruction retrieval."""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize vector store with policy documents."""
        
        # Create documents from policy knowledge base
        policy_documents = [
            {
                "code": "TRG-001",
                "content": "TRG-001: High priority immediate review process for urgent claims requiring supervisor notification and rapid processing within 2 hours. Includes document validation for Form-A, ID-Copy, and Medical-Records.",
                "metadata": {"priority": "HIGH", "category": "urgent_review"}
            },
            {
                "code": "CLM-456", 
                "content": "CLM-456: Standard claim processing workflow for routine claims including identity verification, policy validation, and coverage assessment using ClaimDB and PolicyDB systems.",
                "metadata": {"priority": "MEDIUM", "category": "standard_processing"}
            },
            {
                "code": "TRG-789",
                "content": "TRG-789: Additional documentation request process for incomplete claims requiring claimant contact and follow-up scheduling with pending status update.",
                "metadata": {"priority": "LOW", "category": "documentation_request"}
            },
            {
                "code": "CLAIM-123",
                "content": "CLAIM-123: Fraud investigation protocol for suspicious claims requiring evidence collection, analysis, and senior investigator involvement with comprehensive documentation.",
                "metadata": {"priority": "HIGH", "category": "fraud_investigation"}
            }
        ]
        
        documents = []
        for policy in policy_documents:
            doc = Document(
                page_content=policy["content"],
                metadata={
                    "trigger_code": policy["code"],
                    "type": "policy_instruction",
                    **policy["metadata"]
                }
            )
            documents.append(doc)
        
        # Create vector store
        try:
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name="policy_instructions"
            )
        except Exception as e:
            print(f"Vector store initialization failed: {e}")
            self.vector_store = None

@tool
def search_rag_policies(trigger_codes: List[str]) -> Dict[str, Any]:
    """Semantic search for policy instructions using RAG/vector store."""
    
    try:
        rag_system = PolicyRAGSystem()
        
        if not rag_system.vector_store:
            return {
                "results": {},
                "found_codes": [],
                "missing_codes": trigger_codes,
                "error": "Vector store not available",
                "source": "rag",
                "status": "failed"
            }
        
        results = {}
        found_codes = []
        missing_codes = []
        
        for code in trigger_codes:
            # Create search query
            query = f"Policy instructions for trigger code {code} processing steps requirements"
            
            # Perform semantic search
            search_results = rag_system.vector_store.similarity_search_with_score(
                query, k=3
            )
            
            # Process results
            if search_results:
                best_match = search_results[0]
                doc, score = best_match
                
                if score < 0.8:  # Good similarity threshold
                    results[code] = {
                        "trigger_code": code,
                        "instructions": {
                            "semantic_match": doc.page_content,
                            "confidence": 1.0 - score,  # Convert distance to confidence
                            "matched_code": doc.metadata.get("trigger_code", ""),
                            "category": doc.metadata.get("category", ""),
                            "priority": doc.metadata.get("priority", "MEDIUM")
                        },
                        "source": "rag_semantic",
                        "similarity_score": score
                    }
                    found_codes.append(code)
                else:
                    missing_codes.append(code)
            else:
                missing_codes.append(code)
        
        return {
            "results": results,
            "found_codes": found_codes,
            "missing_codes": missing_codes,
            "total_requested": len(trigger_codes),
            "found_count": len(found_codes),
            "success_rate": len(found_codes) / len(trigger_codes) if trigger_codes else 0,
            "source": "rag",
            "status": "success"
        }
        
    except Exception as e:
        return {
            "results": {},
            "found_codes": [],
            "missing_codes": trigger_codes,
            "error": f"RAG search failed: {str(e)}",
            "source": "rag",
            "status": "failed"
        }

# ==================== REACT AGENT FOR POLICY ACCESS ====================

def create_policy_access_agent():
    """Creates ReAct agent that intelligently selects the best data access method."""
    
    tools = [
        load_policies_json,
        batch_extract_from_json,
        query_bigquery_policies,
        batch_query_bigquery,
        search_rag_policies
    ]
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    agent = create_react_agent(
        llm,
        tools,
        state_modifier="""
        You are an expert Policy Access Agent for retrieving trigger code instructions.
        
        Your goal is to retrieve complete policy instructions for all trigger codes using the best available method.
        
        ACCESS METHOD STRATEGY:
        1. **JSON File Access (FIRST CHOICE)** - Use load_policies_json + batch_extract_from_json
           - Fastest and most structured
           - Contains comprehensive instruction details
           - Best for known trigger codes
        
        2. **BigQuery Database (FALLBACK)** - Use batch_query_bigquery
           - Good for enterprise-scale data
           - Reliable for large datasets
           - Use if JSON is incomplete or unavailable
        
        3. **RAG/Vector Store (SEMANTIC SEARCH)** - Use search_rag_policies
           - Best for unknown or similar codes
           - Semantic matching capabilities
           - Use for codes not found by other methods
        
        DECISION LOGIC:
        - Always try JSON first for maximum detail
        - If JSON missing codes, try BigQuery for those specific codes
        - If still missing codes, try RAG for semantic matching
        - Combine results from multiple sources for complete coverage
        
        QUALITY STANDARDS:
        - Prioritize detailed instructions over basic matches
        - Ensure all requested codes are addressed
        - Provide source tracking for transparency
        - Report success rates and missing codes
        """
    )
    
    return agent

# ==================== WORKFLOW IMPLEMENTATION ====================

def policy_lookup_node(state: PolicyLookupState) -> PolicyLookupState:
    """Main workflow node for policy lookup using ReAct agent."""
    
    try:
        # Extract trigger codes from previous module
        trigger_codes = [item["code"] for item in state["trigger_codes"]]
        
        if not trigger_codes:
            state["errors"].append("No trigger codes provided for policy lookup")
            state["policy_instructions"] = []
            return state
        
        # Create and use agent
        agent = create_policy_access_agent()
        
        agent_query = f"""
        Retrieve policy instructions for these trigger codes: {trigger_codes}
        
        Use the optimal access strategy:
        1. Try JSON file access first for detailed instructions
        2. Use BigQuery for any codes not found in JSON
        3. Use RAG semantic search for remaining unknown codes
        
        Provide comprehensive instructions for all codes with source tracking.
        """
        
        # Execute agent
        result = agent.invoke({"messages": [("user", agent_query)]})
        
        # Also execute the strategy directly for demonstration
        final_instructions = execute_access_strategy(trigger_codes)
        
        # Update state
        state["policy_instructions"] = final_instructions["instructions"]
        state["access_method_used"] = final_instructions["methods_used"]
        state["processing_metadata"] = {
            "processed_at": datetime.now().isoformat(),
            "agent_response": result['messages'][-1].content,
            "codes_requested": len(trigger_codes),
            "instructions_found": len(final_instructions["instructions"]),
            "success_rate": final_instructions["success_rate"],
            "methods_used": final_instructions["methods_used"],
            "source_breakdown": final_instructions["source_breakdown"],
            "success": True
        }
        
    except Exception as e:
        state["errors"].append(f"Policy lookup failed: {str(e)}")
        state["policy_instructions"] = []
        state["processing_metadata"] = {
            "processed_at": datetime.now().isoformat(),
            "success": False,
            "error": str(e)
        }
    
    return state

def execute_access_strategy(trigger_codes: List[str]) -> Dict[str, Any]:
    """Executes the intelligent access strategy for policy retrieval."""
    
    all_instructions = []
    methods_used = []
    source_breakdown = {}
    remaining_codes = trigger_codes.copy()
    
    # Step 1: Try JSON first
    try:
        json_data = load_policies_json()
        if json_data["status"] == "success":
            json_result = batch_extract_from_json(json_data, remaining_codes)
            
            # Convert JSON results to standard format
            for code, policy_data in json_result["results"].items():
                all_instructions.append({
                    "trigger_code": code,
                    "found": True,
                    "instructions": policy_data,
                    "source": "json_file"
                })
            
            remaining_codes = json_result["missing_codes"]
            methods_used.append("json_file")
            source_breakdown["json_file"] = len(json_result["found_codes"])
    except Exception as e:
        pass  # Continue to next method
    
    # Step 2: Try BigQuery for remaining codes
    if remaining_codes:
        try:
            bq_result = batch_query_bigquery(remaining_codes)
            
            # Convert BigQuery results to standard format
            for code, policy_data in bq_result["results"].items():
                all_instructions.append({
                    "trigger_code": code,
                    "found": True,
                    "instructions": policy_data,
                    "source": "bigquery"
                })
            
            remaining_codes = bq_result["missing_codes"]
            methods_used.append("bigquery")
            source_breakdown["bigquery"] = len(bq_result["found_codes"])
        except Exception as e:
            pass  # Continue to next method
    
    # Step 3: Try RAG for remaining codes
    if remaining_codes:
        try:
            rag_result = search_rag_policies(remaining_codes)
            
            # Convert RAG results to standard format
            for code, policy_data in rag_result["results"].items():
                all_instructions.append({
                    "trigger_code": code,
                    "found": True,
                    "instructions": policy_data,
                    "source": "rag"
                })
            
            remaining_codes = rag_result["missing_codes"]
            methods_used.append("rag")
            source_breakdown["rag"] = len(rag_result["found_codes"])
        except Exception as e:
            pass  # RAG failed
    
    # Add missing codes to results
    for code in remaining_codes:
        all_instructions.append({
            "trigger_code": code,
            "found": False,
            "instructions": {},
            "source": "none",
            "error": "Code not found in any data source"
        })
    
    success_rate = (len(trigger_codes) - len(remaining_codes)) / len(trigger_codes) if trigger_codes else 0
    
    return {
        "instructions": all_instructions,
        "methods_used": methods_used,
        "source_breakdown": source_breakdown,
        "success_rate": success_rate,
        "total_codes": len(trigger_codes),
        "found_codes": len(trigger_codes) - len(remaining_codes),
        "missing_codes": remaining_codes
    }

# ==================== LANGGRAPH WORKFLOW ====================

def create_policy_lookup_workflow():
    """Creates the LangGraph workflow for policy lookup."""
    
    # Create workflow
    workflow = StateGraph(PolicyLookupState)
    
    # Add lookup node
    workflow.add_node("policy_lookup", policy_lookup_node)
    
    # Define flow
    workflow.add_edge(START, "policy_lookup")
    workflow.add_edge("policy_lookup", END)
    
    return workflow.compile()

# ==================== INTEGRATION INTERFACE ====================

def lookup_policy_instructions(trigger_codes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Main interface for Module 3 - looks up policy instructions for trigger codes."""
    
    # Initialize state
    initial_state = PolicyLookupState(
        trigger_codes=trigger_codes,
        policy_instructions=[],
        access_method_used="",
        processing_metadata={},
        errors=[]
    )
    
    # Run workflow
    workflow = create_policy_lookup_workflow()
    result = workflow.invoke(initial_state)
    
    return result

# ==================== MODULE INTEGRATION ====================

def integrate_with_module2(module2_output: Dict[str, Any]) -> Dict[str, Any]:
    """Integrates Module 3 with Module 2 output."""
    
    return lookup_policy_instructions(module2_output["trigger_codes"])

# ==================== USAGE EXAMPLES ====================

if __name__ == "__main__":
    print("=== MODULE 3: POLICY LOOKUP & INSTRUCTION RETRIEVAL TESTING ===")
    
    # Test cases with different trigger code scenarios
    test_cases = [
        {
            "name": "Known Codes (JSON Available)",
            "trigger_codes": [
                {"code": "TRG-001", "confidence": 0.95, "source": "llm_primary"},
                {"code": "CLM-456", "confidence": 0.87, "source": "regex_only"}
            ]
        },
        {
            "name": "Mixed Known/Unknown Codes",
            "trigger_codes": [
                {"code": "TRG-001", "confidence": 0.95, "source": "llm_primary"},
                {"code": "UNKNOWN-999", "confidence": 0.6, "source": "regex_only"},
                {"code": "CLAIM-123", "confidence": 0.8, "source": "llm_and_regex"}
            ]
        },
        {
            "name": "All Unknown Codes",
            "trigger_codes": [
                {"code": "XYZ-999", "confidence": 0.5, "source": "regex_only"},
                {"code": "ABC-888", "confidence": 0.4, "source": "llm_primary"}
            ]
        },
        {
            "name": "No Codes",
            "trigger_codes": []
        }
    ]
    
    for test_case in test_cases:
        print(f"\n--- Testing: {test_case['name']} ---")
        
        result = lookup_policy_instructions(test_case['trigger_codes'])
        
        print(f"Success: {result['processing_metadata'].get('success', False)}")
        print(f"Methods Used: {result['processing_metadata'].get('methods_used', [])}")
        print(f"Success Rate: {result['processing_metadata'].get('success_rate', 0):.2%}")
        print(f"Instructions Found: {len(result['policy_instructions'])}")
        
        for instruction in result['policy_instructions']:
            status = "✓" if instruction['found'] else "✗"
            print(f"  {status} {instruction['trigger_code']} (source: {instruction['source']})")
        
        if result['errors']:
            print(f"Errors: {result['errors']}")
        
        print("-" * 50)
    
    print("\n=== MODULE 3 TESTING COMPLETE ===")