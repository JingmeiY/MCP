from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import re
import json
from datetime import datetime
from collections import Counter

# ==================== STATE DEFINITION ====================

class TriggerExtractionState(TypedDict):
    raw_content: str  # Input from Module 1
    input_type: str  # From Module 1 metadata
    trigger_codes: List[Dict[str, Any]]  # Final extracted codes
    processing_metadata: Dict[str, Any]
    errors: List[str]

# ==================== CORE TOOLS/FUNCTIONS ====================

@tool
def llm_extract_triggers(text_content: str) -> List[Dict[str, Any]]:
    """Primary LLM-based trigger code extraction with contextual understanding."""
    
    if not text_content or len(text_content.strip()) < 10:
        return []
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # Specialized prompt for trigger code extraction
    extraction_prompt = f"""
    You are an expert at extracting trigger codes from claim processing documents.
    
    TRIGGER CODE PATTERNS TO LOOK FOR:
    - Codes starting with TRG- followed by digits (e.g., TRG-001, TRG-123)
    - Codes starting with CLM- followed by digits (e.g., CLM-456, CLM-789)
    - Codes starting with CLAIM- followed by digits (e.g., CLAIM-001)
    - Pure numeric codes that appear to be trigger codes (e.g., 001, 456, 789)
    - Alphanumeric combinations that could be trigger codes (e.g., A123, BC456)
    - Any code pattern mentioned as "trigger", "code", "reference", or similar
    
    CONTEXT CLUES:
    - Look for phrases like "Trigger Code:", "Code:", "Reference:", "TRG:", "CLM:"
    - Check for codes mentioned in context of processing, review, or action required
    - Consider codes that appear in structured formats (lists, tables, headers)
    
    ANALYZE THIS TEXT:
    {text_content}
    
    INSTRUCTIONS:
    1. Find ALL possible trigger codes in the text
    2. Assign confidence scores (0.0 to 1.0) based on:
       - Clear formatting/labeling (high confidence)
       - Context relevance (medium confidence) 
       - Pattern matching only (lower confidence)
    3. Extract surrounding context for each code
    4. Return ONLY valid JSON - no other text
    
    REQUIRED OUTPUT FORMAT (JSON only):
    [
        {{
            "code": "TRG-001",
            "confidence": 0.95,
            "context": "Trigger Code: TRG-001 requires immediate attention",
            "source": "llm_contextual",
            "location": "paragraph 2"
        }}
    ]
    
    If no trigger codes found, return: []
    """
    
    try:
        # Get LLM response
        response = llm.invoke(extraction_prompt)
        response_text = response.content.strip()
        
        # Parse JSON response
        try:
            trigger_codes = json.loads(response_text)
            
            # Validate structure
            if isinstance(trigger_codes, list):
                validated_codes = []
                for code_obj in trigger_codes:
                    if isinstance(code_obj, dict) and "code" in code_obj:
                        # Ensure required fields
                        validated_code = {
                            "code": str(code_obj.get("code", "")).strip().upper(),
                            "confidence": float(code_obj.get("confidence", 0.5)),
                            "context": str(code_obj.get("context", "")).strip(),
                            "source": "llm_contextual",
                            "location": str(code_obj.get("location", "")).strip()
                        }
                        
                        # Only add if code is not empty
                        if validated_code["code"]:
                            validated_codes.append(validated_code)
                
                return validated_codes
            else:
                return []
                
        except json.JSONDecodeError as e:
            # Try to extract codes from malformed response
            return extract_codes_from_text_response(response_text)
    
    except Exception as e:
        # Return empty list on any error
        return []

def extract_codes_from_text_response(response_text: str) -> List[Dict[str, Any]]:
    """Helper function to extract codes from malformed LLM response."""
    
    codes = []
    
    # Look for common patterns in text response
    patterns = [
        r'TRG-\d+',
        r'CLM-\d+', 
        r'CLAIM-\d+',
        r'\b\d{3,4}\b',  # 3-4 digit numbers
        r'\b[A-Z]{1,3}\d{2,4}\b'  # Letter-digit combinations
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, response_text, re.IGNORECASE)
        for match in matches:
            code = match.group().upper()
            codes.append({
                "code": code,
                "confidence": 0.6,  # Medium confidence for fallback extraction
                "context": "Extracted from LLM response",
                "source": "llm_fallback",
                "location": "response parsing"
            })
    
    # Remove duplicates
    seen_codes = set()
    unique_codes = []
    for code_obj in codes:
        if code_obj["code"] not in seen_codes:
            seen_codes.add(code_obj["code"])
            unique_codes.append(code_obj)
    
    return unique_codes

@tool
def regex_pattern_match(text_content: str) -> List[str]:
    """Backup regex pattern matching for reliable trigger code detection."""
    
    if not text_content:
        return []
    
    # Comprehensive regex patterns for trigger codes
    patterns = [
        # Standard prefixed codes
        r'\bTRG-\d{1,4}\b',           # TRG-001, TRG-1234
        r'\bCLM-\d{1,4}\b',           # CLM-456, CLM-789
        r'\bCLAIM-\d{1,4}\b',         # CLAIM-001, CLAIM-999
        
        # Contextual codes (with trigger/code context)
        r'(?:trigger\s*code|code|ref|reference)[:=\s]+([A-Z]{1,4}-?\d{1,4})',
        r'(?:trigger|code|ref)[:=\s]+(\d{3,4})',
        
        # Standalone numeric codes (with context)
        r'\b(?:code|trigger|ref|reference)\s*[:=]?\s*(\d{3,4})\b',
        
        # Alphanumeric patterns
        r'\b[A-Z]{1,3}\d{2,4}\b',     # A123, BC456, ABC1234
        r'\b\d{1,2}[A-Z]{1,3}\d{1,3}\b',  # 1A23, 12BC456
        
        # Special formats
        r'\b[A-Z]+\d+-\d+\b',         # ABC123-456
        r'\b\d{3,4}-[A-Z]{1,3}\b',    # 123-ABC, 1234-AB
    ]
    
    found_codes = []
    
    for pattern in patterns:
        matches = re.finditer(pattern, text_content, re.IGNORECASE)
        for match in matches:
            if match.groups():
                # If pattern has groups, use the first group
                code = match.group(1).upper().strip()
            else:
                # Use the full match
                code = match.group(0).upper().strip()
            
            # Clean up the code
            code = code.replace(" ", "").replace(":", "").replace("=", "")
            
            if code and len(code) >= 2:  # Minimum length filter
                found_codes.append(code)
    
    # Remove duplicates while preserving order
    unique_codes = []
    seen = set()
    for code in found_codes:
        if code not in seen:
            seen.add(code)
            unique_codes.append(code)
    
    return unique_codes

@tool
def validate_trigger_codes(codes: List[str]) -> List[Dict[str, Any]]:
    """Validates and scores potential trigger codes based on patterns and context."""
    
    validated_codes = []
    
    for code in codes:
        if not code or len(code) < 2:
            continue
        
        code = code.strip().upper()
        confidence = 0.5  # Base confidence
        
        # Scoring based on patterns
        if re.match(r'^TRG-\d+$', code):
            confidence = 0.9  # High confidence for TRG- pattern
        elif re.match(r'^CLM-\d+$', code):
            confidence = 0.9  # High confidence for CLM- pattern
        elif re.match(r'^CLAIM-\d+$', code):
            confidence = 0.85  # High confidence for CLAIM- pattern
        elif re.match(r'^\d{3,4}$', code):
            confidence = 0.7  # Medium-high for 3-4 digits
        elif re.match(r'^[A-Z]{1,3}\d{2,4}$', code):
            confidence = 0.6  # Medium for alphanumeric
        elif '-' in code:
            confidence = 0.75  # Higher for hyphenated codes
        else:
            confidence = 0.4  # Lower for other patterns
        
        # Length-based adjustment
        if len(code) < 3:
            confidence *= 0.8
        elif len(code) > 10:
            confidence *= 0.7
        
        validated_codes.append({
            "code": code,
            "confidence": min(confidence, 1.0),
            "source": "regex_validation",
            "pattern_type": determine_pattern_type(code)
        })
    
    return validated_codes

def determine_pattern_type(code: str) -> str:
    """Determines the pattern type of a trigger code."""
    
    if re.match(r'^TRG-\d+$', code):
        return "standard_trg"
    elif re.match(r'^CLM-\d+$', code):
        return "standard_clm"
    elif re.match(r'^CLAIM-\d+$', code):
        return "standard_claim"
    elif re.match(r'^\d+$', code):
        return "numeric_only"
    elif re.match(r'^[A-Z]+\d+$', code):
        return "alpha_numeric"
    elif '-' in code:
        return "hyphenated"
    else:
        return "other"

@tool
def merge_and_deduplicate_codes(llm_codes: List[Dict[str, Any]], 
                               regex_codes: List[str]) -> List[Dict[str, Any]]:
    """Merges LLM and regex results, removing duplicates and optimizing confidence scores."""
    
    # Convert regex codes to structured format
    regex_structured = validate_trigger_codes(regex_codes)
    
    # Create a dictionary to merge codes
    merged_codes = {}
    
    # Add LLM codes first (higher priority)
    for code_obj in llm_codes:
        code = code_obj["code"]
        merged_codes[code] = {
            "code": code,
            "confidence": code_obj["confidence"],
            "context": code_obj.get("context", ""),
            "source": "llm_primary",
            "location": code_obj.get("location", ""),
            "pattern_type": determine_pattern_type(code)
        }
    
    # Add regex codes, boosting confidence if also found by LLM
    for code_obj in regex_structured:
        code = code_obj["code"]
        
        if code in merged_codes:
            # Code found by both methods - boost confidence
            merged_codes[code]["confidence"] = min(
                merged_codes[code]["confidence"] + 0.1, 
                1.0
            )
            merged_codes[code]["source"] = "llm_and_regex"
        else:
            # Code only found by regex
            merged_codes[code] = {
                "code": code,
                "confidence": code_obj["confidence"],
                "context": "Pattern matched",
                "source": "regex_only",
                "location": "text pattern",
                "pattern_type": code_obj["pattern_type"]
            }
    
    # Convert back to list and sort by confidence
    final_codes = list(merged_codes.values())
    final_codes.sort(key=lambda x: x["confidence"], reverse=True)
    
    return final_codes

@tool
def analyze_extraction_quality(trigger_codes: List[Dict[str, Any]], 
                             text_content: str) -> Dict[str, Any]:
    """Analyzes the quality and completeness of trigger code extraction."""
    
    analysis = {
        "total_codes_found": len(trigger_codes),
        "confidence_distribution": {},
        "source_distribution": {},
        "pattern_distribution": {},
        "quality_score": 0.0,
        "potential_missing_codes": [],
        "extraction_confidence": "low"
    }
    
    if not trigger_codes:
        analysis["extraction_confidence"] = "low"
        analysis["potential_missing_codes"] = check_for_missed_patterns(text_content)
        return analysis
    
    # Confidence distribution
    confidence_ranges = {"high": 0, "medium": 0, "low": 0}
    for code in trigger_codes:
        conf = code["confidence"]
        if conf >= 0.8:
            confidence_ranges["high"] += 1
        elif conf >= 0.5:
            confidence_ranges["medium"] += 1
        else:
            confidence_ranges["low"] += 1
    
    analysis["confidence_distribution"] = confidence_ranges
    
    # Source distribution
    sources = [code["source"] for code in trigger_codes]
    analysis["source_distribution"] = dict(Counter(sources))
    
    # Pattern distribution
    patterns = [code.get("pattern_type", "unknown") for code in trigger_codes]
    analysis["pattern_distribution"] = dict(Counter(patterns))
    
    # Overall quality score
    avg_confidence = sum(code["confidence"] for code in trigger_codes) / len(trigger_codes)
    multi_source_bonus = 0.1 if len(analysis["source_distribution"]) > 1 else 0
    analysis["quality_score"] = min(avg_confidence + multi_source_bonus, 1.0)
    
    # Extraction confidence assessment
    if analysis["quality_score"] >= 0.8:
        analysis["extraction_confidence"] = "high"
    elif analysis["quality_score"] >= 0.6:
        analysis["extraction_confidence"] = "medium"
    else:
        analysis["extraction_confidence"] = "low"
    
    return analysis

def check_for_missed_patterns(text_content: str) -> List[str]:
    """Checks for potential trigger code patterns that might have been missed."""
    
    # Look for potential indicators that codes might be present
    indicators = [
        r'\btrigger\b',
        r'\bcode\b',
        r'\breference\b',
        r'\bprocess\b',
        r'\bclaim\b',
        r'\breview\b'
    ]
    
    potential_issues = []
    
    for indicator in indicators:
        if re.search(indicator, text_content, re.IGNORECASE):
            potential_issues.append(f"Found '{indicator}' - may indicate missed codes")
    
    return potential_issues

# ==================== REACT AGENT CREATION ====================

def create_trigger_extraction_agent():
    """Creates the ReAct agent for trigger code extraction."""
    
    # Define all available tools
    tools = [
        llm_extract_triggers,
        regex_pattern_match,
        validate_trigger_codes,
        merge_and_deduplicate_codes,
        analyze_extraction_quality
    ]
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # Create ReAct agent
    agent = create_react_agent(
        llm,
        tools,
        state_modifier="""
        You are an expert Trigger Code Extraction Agent for claim processing.
        
        Your goal is to identify and extract ALL trigger codes from text content with high accuracy.
        
        EXTRACTION STRATEGY:
        1. Use llm_extract_triggers() as PRIMARY method for contextual understanding
        2. Use regex_pattern_match() as BACKUP method for pattern reliability  
        3. Use merge_and_deduplicate_codes() to combine results intelligently
        4. Use analyze_extraction_quality() to assess completeness
        
        TRIGGER CODE TYPES TO FIND:
        - Standard prefixed: TRG-001, CLM-456, CLAIM-123
        - Numeric codes: 001, 456, 789 (when in context)
        - Alphanumeric: A123, BC456, ABC1234
        - Any code referenced as "trigger", "code", "reference"
        
        QUALITY STANDARDS:
        - Prioritize high-confidence extractions
        - Always run both LLM and regex methods
        - Merge results to maximize coverage
        - Analyze quality to ensure completeness
        
        Return comprehensive results with confidence scores and source tracking.
        """
    )
    
    return agent

# ==================== WORKFLOW IMPLEMENTATION ====================

def trigger_extraction_node(state: TriggerExtractionState) -> TriggerExtractionState:
    """Main workflow node for trigger code extraction."""
    
    try:
        text_content = state["raw_content"]
        
        if not text_content or len(text_content.strip()) < 10:
            state["errors"].append("Insufficient text content for trigger extraction")
            state["trigger_codes"] = []
            return state
        
        # Create and use agent
        agent = create_trigger_extraction_agent()
        
        agent_query = f"""
        Extract all trigger codes from this text content:
        
        TEXT CONTENT:
        {text_content}
        
        Use both LLM and regex methods, then merge results for maximum accuracy.
        Analyze the extraction quality and provide comprehensive results.
        """
        
        # Execute agent
        result = agent.invoke({"messages": [("user", agent_query)]})
        
        # Also run extraction directly for demonstration
        llm_codes = llm_extract_triggers(text_content)
        regex_codes = regex_pattern_match(text_content)
        merged_codes = merge_and_deduplicate_codes(llm_codes, regex_codes)
        quality_analysis = analyze_extraction_quality(merged_codes, text_content)
        
        # Update state
        state["trigger_codes"] = merged_codes
        state["processing_metadata"] = {
            "processed_at": datetime.now().isoformat(),
            "agent_response": result['messages'][-1].content,
            "llm_codes_found": len(llm_codes),
            "regex_codes_found": len(regex_codes),
            "final_codes_count": len(merged_codes),
            "quality_analysis": quality_analysis,
            "text_length": len(text_content),
            "success": True
        }
        
    except Exception as e:
        state["errors"].append(f"Trigger extraction failed: {str(e)}")
        state["trigger_codes"] = []
        state["processing_metadata"] = {
            "processed_at": datetime.now().isoformat(),
            "success": False,
            "error": str(e)
        }
    
    return state

# ==================== LANGGRAPH WORKFLOW ====================

def create_trigger_extraction_workflow():
    """Creates the LangGraph workflow for trigger extraction."""
    
    # Create workflow
    workflow = StateGraph(TriggerExtractionState)
    
    # Add extraction node
    workflow.add_node("trigger_extraction", trigger_extraction_node)
    
    # Define flow
    workflow.add_edge(START, "trigger_extraction")
    workflow.add_edge("trigger_extraction", END)
    
    return workflow.compile()

# ==================== INTEGRATION INTERFACE ====================

def extract_trigger_codes(raw_content: str, input_type: str = "raw_text") -> Dict[str, Any]:
    """Main interface for Module 2 - extracts trigger codes from processed text."""
    
    # Initialize state
    initial_state = TriggerExtractionState(
        raw_content=raw_content,
        input_type=input_type,
        trigger_codes=[],
        processing_metadata={},
        errors=[]
    )
    
    # Run workflow
    workflow = create_trigger_extraction_workflow()
    result = workflow.invoke(initial_state)
    
    return result

# ==================== MODULE INTEGRATION ====================

def integrate_with_module1(module1_output: Dict[str, Any]) -> Dict[str, Any]:
    """Integrates Module 2 with Module 1 output."""
    
    return extract_trigger_codes(
        raw_content=module1_output["raw_content"],
        input_type=module1_output["input_type"]
    )

# ==================== USAGE EXAMPLES ====================

if __name__ == "__main__":
    print("=== MODULE 2: TRIGGER CODE EXTRACTION TESTING ===")
    
    # Test cases with different trigger code patterns
    test_cases = [
        {
            "name": "Standard Prefixed Codes",
            "content": """
            CLAIM PROCESSING ALERT
            
            Trigger Code: TRG-001 requires immediate attention
            Standard processing for CLM-456 is required
            Additional review needed for CLAIM-789
            
            Please process according to established procedures.
            """
        },
        {
            "name": "Mixed Format Codes",
            "content": """
            Subject: Processing Request - Multiple Codes
            
            The following codes require processing:
            - Code 123 (high priority)
            - Reference A456
            - Trigger BC789
            - Process code 001
            
            Contact supervisor for guidance on XYZ-999.
            """
        },
        {
            "name": "Email Content",
            "content": """
            FROM: claims@company.com
            SUBJECT: Urgent - TRG-001 Review Required
            
            Dear Processor,
            
            Please review claim TRG-001 immediately.
            Standard workflow applies to CLM-456.
            Request additional documentation for reference 789.
            
            Codes: TRG-001, CLM-456, 789, ABC123
            """
        },
        {
            "name": "No Clear Codes",
            "content": """
            This is a general message about claim processing.
            Please review the standard procedures.
            Contact the team if you have questions.
            No specific trigger codes mentioned here.
            """
        }
    ]
    
    for test_case in test_cases:
        print(f"\n--- Testing: {test_case['name']} ---")
        
        result = extract_trigger_codes(test_case['content'])
        
        print(f"Success: {result['processing_metadata'].get('success', False)}")
        print(f"Codes Found: {len(result['trigger_codes'])}")
        
        for code in result['trigger_codes']:
            print(f"  - {code['code']} (confidence: {code['confidence']:.2f}, source: {code['source']})")
        
        quality = result['processing_metadata'].get('quality_analysis', {})
        print(f"Quality Score: {quality.get('quality_score', 0):.2f}")
        print(f"Extraction Confidence: {quality.get('extraction_confidence', 'unknown')}")
        
        if result['errors']:
            print(f"Errors: {result['errors']}")
        
        print("-" * 50)
    
    print("\n=== MODULE 2 TESTING COMPLETE ===")