from typing import TypedDict, List, Dict, Any, Optional, Annotated
from langgraph.graph import StateGraph, START, END, add_messages
from datetime import datetime

# ==================== UNIFIED PIPELINE STATE ====================

class ClaimProcessingPipelineState(TypedDict):
    # Input
    user_input: str
    
    # Module 1: Input Processing outputs
    raw_content: str
    input_type: str
    
    # Module 2: Trigger Extraction outputs  
    trigger_codes: List[Dict[str, Any]]
    
    # Module 3: Policy Lookup outputs
    policy_instructions: List[Dict[str, Any]]
    
    # Module 4: Action Planning outputs
    action_plan: Dict[str, Any]
    formatted_output: str
    
    # Pipeline tracking
    current_module: str
    pipeline_metadata: Dict[str, Any]
    errors: List[str]

# ==================== MODULE NODE FUNCTIONS ====================

def module1_input_processing_node(state: ClaimProcessingPipelineState) -> ClaimProcessingPipelineState:
    """Module 1: Input Processing Node - processes user input into clean text."""
    
    try:
        state["current_module"] = "module1_input_processing"
        
        # Call Module 1 logic (from your previous implementation)
        # result = process_input(state["user_input"])
        
        # Mock simple implementation
        user_input = state["user_input"]
        if "email" in user_input.lower():
            state["raw_content"] = f"Email content: {user_input} with trigger codes TRG-001, CLM-456"
            state["input_type"] = "email_query"
        else:
            state["raw_content"] = user_input
            state["input_type"] = "raw_text"
        
        # Update pipeline metadata
        state["pipeline_metadata"]["module1"] = {
            "completed_at": datetime.now().isoformat(),
            "success": True,
            "input_type": state["input_type"],
            "content_length": len(state["raw_content"])
        }
        
    except Exception as e:
        state["errors"].append(f"Module 1 failed: {str(e)}")
        state["raw_content"] = state["user_input"]  # Fallback
        state["input_type"] = "raw_text"
    
    return state

def module2_trigger_extraction_node(state: ClaimProcessingPipelineState) -> ClaimProcessingPipelineState:
    """Module 2: Trigger Extraction Node - extracts trigger codes from raw content."""
    
    try:
        state["current_module"] = "module2_trigger_extraction"
        
        # Call Module 2 logic (from your previous implementation)
        # result = extract_trigger_codes(state["raw_content"], state["input_type"])
        
        # Mock simple implementation
        import re
        patterns = [r'TRG-\d+', r'CLM-\d+', r'CLAIM-\d+']
        codes = []
        
        for pattern in patterns:
            matches = re.findall(pattern, state["raw_content"], re.IGNORECASE)
            for match in matches:
                codes.append({
                    "code": match.upper(),
                    "confidence": 0.9,
                    "source": "regex"
                })
        
        state["trigger_codes"] = codes
        
        # Update pipeline metadata
        state["pipeline_metadata"]["module2"] = {
            "completed_at": datetime.now().isoformat(),
            "success": True,
            "codes_found": len(codes),
            "codes": [c["code"] for c in codes]
        }
        
    except Exception as e:
        state["errors"].append(f"Module 2 failed: {str(e)}")
        state["trigger_codes"] = []
    
    return state

def module3_policy_lookup_node(state: ClaimProcessingPipelineState) -> ClaimProcessingPipelineState:
    """Module 3: Policy Lookup Node - retrieves instructions for trigger codes."""
    
    try:
        state["current_module"] = "module3_policy_lookup"
        
        # Call Module 3 logic (from your previous implementation)
        # result = lookup_policy_instructions(state["trigger_codes"])
        
        # Mock simple implementation
        mock_policies = {
            "TRG-001": {"priority": "HIGH", "steps": ["Immediate review", "Supervisor approval"]},
            "CLM-456": {"priority": "MEDIUM", "steps": ["Standard processing", "Coverage check"]},
            "CLAIM-123": {"priority": "HIGH", "steps": ["Fraud investigation"]}
        }
        
        instructions = []
        for code_obj in state["trigger_codes"]:
            code = code_obj["code"]
            if code in mock_policies:
                instructions.append({
                    "trigger_code": code,
                    "found": True,
                    "instructions": mock_policies[code],
                    "source": "json_file"
                })
            else:
                instructions.append({
                    "trigger_code": code,
                    "found": False,
                    "instructions": {},
                    "source": "none"
                })
        
        state["policy_instructions"] = instructions
        
        # Update pipeline metadata
        state["pipeline_metadata"]["module3"] = {
            "completed_at": datetime.now().isoformat(),
            "success": True,
            "instructions_found": len([i for i in instructions if i["found"]]),
            "total_requested": len(state["trigger_codes"])
        }
        
    except Exception as e:
        state["errors"].append(f"Module 3 failed: {str(e)}")
        state["policy_instructions"] = []
    
    return state

def module4_action_planning_node(state: ClaimProcessingPipelineState) -> ClaimProcessingPipelineState:
    """Module 4: Action Planning Node - generates final action plan."""
    
    try:
        state["current_module"] = "module4_action_planning"
        
        # Call Module 4 logic (from your previous implementation)
        # result = generate_action_plan(state["policy_instructions"], state["raw_content"])
        
        # Mock simple implementation
        codes = [inst["trigger_code"] for inst in state["policy_instructions"] if inst["found"]]
        
        summary = {
            "request": f"Process claim with {len(codes)} trigger codes",
            "actions": f"Follow procedures for: {', '.join(codes)}",
            "trigger_codes": codes,
            "total_codes": len(codes)
        }
        
        state["action_plan"] = {
            "summary": summary,
            "created_at": datetime.now().isoformat(),
            "status": "ready"
        }
        
        # Generate formatted output
        state["formatted_output"] = format_final_output(state)
        
        # Update pipeline metadata
        state["pipeline_metadata"]["module4"] = {
            "completed_at": datetime.now().isoformat(),
            "success": True,
            "plan_generated": True
        }
        
    except Exception as e:
        state["errors"].append(f"Module 4 failed: {str(e)}")
        state["action_plan"] = {}
        state["formatted_output"] = "Action plan generation failed"
    
    return state

def format_final_output(state: ClaimProcessingPipelineState) -> str:
    """Formats the final output for claim processor."""
    
    action_plan = state.get("action_plan", {})
    summary = action_plan.get("summary", {})
    
    output = f"""
CLAIM PROCESSING ACTION PLAN
Generated: {action_plan.get('created_at', 'Unknown')}

SUMMARY
Request: {summary.get('request', 'N/A')}
Actions: {summary.get('actions', 'N/A')}

TRIGGER CODES FOUND
"""
    
    codes = summary.get('trigger_codes', [])
    for i, code in enumerate(codes, 1):
        output += f"{i}. {code}\n"
    
    if not codes:
        output += "No trigger codes found\n"
    
    return output.strip()

# ==================== PIPELINE WORKFLOW CREATION ====================

def create_claim_processing_pipeline():
    """Creates the complete end-to-end claim processing pipeline."""
    
    # Create the main pipeline graph
    workflow = StateGraph(ClaimProcessingPipelineState)
    
    # Add all module nodes
    workflow.add_node("module1_input_processing", module1_input_processing_node)
    workflow.add_node("module2_trigger_extraction", module2_trigger_extraction_node)
    workflow.add_node("module3_policy_lookup", module3_policy_lookup_node)
    workflow.add_node("module4_action_planning", module4_action_planning_node)
    
    # Define linear pipeline flow
    workflow.add_edge(START, "module1_input_processing")
    workflow.add_edge("module1_input_processing", "module2_trigger_extraction")
    workflow.add_edge("module2_trigger_extraction", "module3_policy_lookup")
    workflow.add_edge("module3_policy_lookup", "module4_action_planning")
    workflow.add_edge("module4_action_planning", END)
    
    return workflow.compile()

# ==================== PIPELINE EXECUTION INTERFACE ====================

def run_claim_processing_pipeline(user_input: str) -> Dict[str, Any]:
    """Main interface to run the complete claim processing pipeline."""
    
    # Initialize pipeline state
    initial_state = ClaimProcessingPipelineState(
        user_input=user_input,
        raw_content="",
        input_type="",
        trigger_codes=[],
        policy_instructions=[],
        action_plan={},
        formatted_output="",
        current_module="",
        pipeline_metadata={
            "started_at": datetime.now().isoformat(),
            "pipeline_id": f"pipeline_{int(datetime.now().timestamp())}",
            "total_modules": 4
        },
        errors=[]
    )
    
    # Create and run pipeline
    pipeline = create_claim_processing_pipeline()
    final_state = pipeline.invoke(initial_state)
    
    # Add completion metadata
    final_state["pipeline_metadata"]["completed_at"] = datetime.now().isoformat()
    final_state["pipeline_metadata"]["success"] = len(final_state["errors"]) == 0
    final_state["pipeline_metadata"]["total_errors"] = len(final_state["errors"])
    
    return final_state

# ==================== CONDITIONAL ROUTING EXAMPLE ====================

def create_pipeline_with_conditional_routing():
    """Example of pipeline with conditional routing based on results."""
    
    def should_skip_module3(state: ClaimProcessingPipelineState) -> str:
        """Conditional routing: skip policy lookup if no trigger codes found."""
        if not state["trigger_codes"]:
            return "module4_action_planning"  # Skip module 3
        else:
            return "module3_policy_lookup"    # Continue normally
    
    workflow = StateGraph(ClaimProcessingPipelineState)
    
    # Add nodes
    workflow.add_node("module1_input_processing", module1_input_processing_node)
    workflow.add_node("module2_trigger_extraction", module2_trigger_extraction_node)
    workflow.add_node("module3_policy_lookup", module3_policy_lookup_node)
    workflow.add_node("module4_action_planning", module4_action_planning_node)
    
    # Linear flow for first two modules
    workflow.add_edge(START, "module1_input_processing")
    workflow.add_edge("module1_input_processing", "module2_trigger_extraction")
    
    # Conditional routing after trigger extraction
    workflow.add_conditional_edges(
        "module2_trigger_extraction",
        should_skip_module3,
        {
            "module3_policy_lookup": "module3_policy_lookup",
            "module4_action_planning": "module4_action_planning"
        }
    )
    
    # Final connections
    workflow.add_edge("module3_policy_lookup", "module4_action_planning")
    workflow.add_edge("module4_action_planning", END)
    
    return workflow.compile()

# ==================== USAGE EXAMPLES ====================

if __name__ == "__main__":
    print("=== END-TO-END CLAIM PROCESSING PIPELINE TESTING ===")
    
    test_cases = [
        "Process email from claims@company.com with TRG-001 and CLM-456",
        "Review claim document containing CLAIM-123 fraud investigation",
        "Handle standard claim with no specific trigger codes mentioned"
    ]
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {test_input[:50]}... ---")
        
        # Run complete pipeline
        result = run_claim_processing_pipeline(test_input)
        
        # Display results
        print(f"Pipeline Success: {result['pipeline_metadata']['success']}")
        print(f"Modules Completed: {len([k for k in result['pipeline_metadata'].keys() if k.startswith('module')])}")
        print(f"Trigger Codes Found: {len(result['trigger_codes'])}")
        print(f"Instructions Retrieved: {len([i for i in result['policy_instructions'] if i.get('found')])}")
        
        if result['errors']:
            print(f"Errors: {result['errors']}")
        
        print("\nFinal Output:")
        print(result['formatted_output'])
        print("-" * 60)
    
    print("\n=== PIPELINE TESTING COMPLETE ===")

# ==================== STATE MANAGEMENT BEST PRACTICES ====================

"""
STATE MANAGEMENT KEY POINTS:

1. **Unified State Structure**
   - Single TypedDict for entire pipeline
   - Each module adds its outputs to shared state
   - Clear data flow between modules

2. **State Updates**
   - Each module node updates specific state fields
   - Preserves previous module outputs
   - Adds metadata and error tracking

3. **Error Handling**
   - Errors accumulated in shared errors list
   - Modules can fallback gracefully
   - Pipeline continues even with partial failures

4. **Metadata Tracking**
   - Each module records completion status
   - Pipeline-level metadata for monitoring
   - Timing and success tracking

5. **Conditional Routing**
   - Can skip modules based on state
   - Dynamic pipeline adaptation
   - Efficient processing paths

6. **Module Integration**
   - Each module is a single node function
   - Clear input/output contracts via state
   - Easy to test and maintain individually
"""