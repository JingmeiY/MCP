from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import re
import PyPDF2
import pytesseract
from PIL import Image
import io
import requests
from datetime import datetime, timedelta
import base64

# ==================== STATE DEFINITION ====================

class InputProcessingState(TypedDict):
    user_input: str  # Raw user query
    input_type: str  # "raw_text", "email_query", "pdf_upload"
    raw_content: str  # Final extracted plain text
    processing_metadata: Dict[str, Any]
    errors: List[str]

# ==================== CORE TOOLS/FUNCTIONS ====================

# --- Text Input Processing Tools ---

@tool
def validate_text_input(text: str) -> Dict[str, Any]:
    """Validates raw text input quality and returns validation results."""
    
    if not text or not isinstance(text, str):
        return {
            "valid": False, 
            "error": "Invalid or empty text input",
            "char_count": 0,
            "word_count": 0
        }
    
    text = text.strip()
    
    if len(text) < 10:
        return {
            "valid": False,
            "error": "Text too short (minimum 10 characters required)",
            "char_count": len(text),
            "word_count": len(text.split())
        }
    
    # Check for potential trigger code patterns
    trigger_patterns = [r'TRG-\d+', r'CLM-\d+', r'CLAIM-\d+', r'[A-Z]{2,4}-\d{3,4}']
    has_potential_codes = any(re.search(pattern, text, re.IGNORECASE) for pattern in trigger_patterns)
    
    return {
        "valid": True,
        "char_count": len(text),
        "word_count": len(text.split()),
        "line_count": len(text.split('\n')),
        "has_potential_trigger_codes": has_potential_codes,
        "has_email_format": "@" in text and "subject:" in text.lower(),
        "quality_score": min(1.0, len(text) / 500)  # Quality based on length
    }

@tool
def sanitize_text(text: str) -> str:
    """Cleans and normalizes text formatting for processing."""
    
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Normalize line breaks
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove non-printable characters except newlines and tabs
    text = re.sub(r'[^\x20-\x7E\n\t]', '', text)
    
    # Clean up multiple consecutive newlines
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

# --- Email Access Tools ---

@tool
def authenticate_outlook_system() -> Dict[str, Any]:
    """Handles OAuth2 authentication for Outlook via Microsoft Graph API."""
    
    # Mock authentication for demonstration
    # In real implementation, this would:
    # 1. Use Microsoft Graph API OAuth2 flow
    # 2. Handle token refresh
    # 3. Store credentials securely
    
    try:
        # Simulate OAuth2 flow
        auth_result = {
            "authenticated": True,
            "access_token": "mock_access_token_" + str(datetime.now().timestamp()),
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": "https://graph.microsoft.com/Mail.Read",
            "refresh_token": "mock_refresh_token",
            "tenant_id": "mock_tenant_id"
        }
        
        return {
            "success": True,
            "auth_data": auth_result,
            "authenticated_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(seconds=3600)).isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Authentication failed: {str(e)}",
            "auth_data": None
        }

@tool
def search_outlook_emails(sender: str = None, subject: str = None, days_back: int = 1) -> List[Dict[str, Any]]:
    """Searches Outlook emails by sender, subject, and date criteria."""
    
    # Mock email search - replace with Microsoft Graph API calls
    # Real implementation would use:
    # GET https://graph.microsoft.com/v1.0/me/messages?$filter=...
    
    try:
        # Simulate search results
        mock_emails = [
            {
                "id": "email_001",
                "subject": "Claim Processing Alert - TRG-001 Review Required",
                "sender": {
                    "name": "Claims System",
                    "address": "claims@company.com"
                },
                "received_datetime": (datetime.now() - timedelta(hours=2)).isoformat(),
                "has_attachments": False,
                "importance": "high",
                "is_read": False,
                "body_preview": "Urgent: Please review claim TRG-001 requiring immediate attention..."
            },
            {
                "id": "email_002", 
                "subject": "Standard Claim Processing - CLM-456",
                "sender": {
                    "name": "Processing Team",
                    "address": "processing@company.com"
                },
                "received_datetime": (datetime.now() - timedelta(hours=6)).isoformat(),
                "has_attachments": True,
                "importance": "normal",
                "is_read": True,
                "body_preview": "Standard processing required for claim CLM-456..."
            }
        ]
        
        # Apply filters
        filtered_emails = []
        for email in mock_emails:
            include_email = True
            
            # Filter by sender
            if sender and sender.lower() not in email["sender"]["address"].lower():
                include_email = False
            
            # Filter by subject
            if subject and subject.lower() not in email["subject"].lower():
                include_email = False
            
            # Filter by date
            email_date = datetime.fromisoformat(email["received_datetime"].replace('Z', '+00:00'))
            cutoff_date = datetime.now() - timedelta(days=days_back)
            if email_date < cutoff_date:
                include_email = False
            
            if include_email:
                filtered_emails.append(email)
        
        return filtered_emails
        
    except Exception as e:
        return [{"error": f"Email search failed: {str(e)}"}]

@tool
def extract_outlook_email_content(email_id: str) -> str:
    """Extracts email subject, headers, and body as plain text."""
    
    # Mock email content extraction - replace with Microsoft Graph API
    # Real implementation would use:
    # GET https://graph.microsoft.com/v1.0/me/messages/{email_id}
    
    try:
        # Simulate email content based on email_id
        mock_email_contents = {
            "email_001": """
SUBJECT: Claim Processing Alert - TRG-001 Review Required
FROM: Claims System <claims@company.com>
TO: Claim Processor <processor@company.com>
DATE: 2025-06-19T08:30:00Z
PRIORITY: High

BODY:
Dear Claim Processor,

This is an automated notification regarding a high-priority claim that requires immediate attention.

TRIGGER CODE: TRG-001
CLAIM REFERENCE: CL-2025-0619-001
PRIORITY LEVEL: HIGH

DETAILS:
- Claimant: John Smith
- Policy Number: POL-123456
- Incident Date: 2025-06-15
- Trigger: High-value claim requiring supervisor review

REQUIRED ACTIONS:
Please review this claim according to TRG-001 procedures within 2 hours.
All required documentation must be validated before processing.

Additional trigger codes may apply: CLM-456 for standard documentation review.

Best regards,
Automated Claims Processing System
            """,
            "email_002": """
SUBJECT: Standard Claim Processing - CLM-456
FROM: Processing Team <processing@company.com>
TO: Claim Processor <processor@company.com>
DATE: 2025-06-19T04:30:00Z
PRIORITY: Normal

BODY:
Hello,

Please process the following standard claim according to normal procedures.

TRIGGER CODE: CLM-456
CLAIM REFERENCE: CL-2025-0619-002

DETAILS:
- Claimant: Jane Doe
- Policy Number: POL-789012
- Incident Date: 2025-06-18
- Type: Standard property damage claim

Please follow CLM-456 standard processing workflow.

Thanks,
Processing Team
            """
        }
        
        content = mock_email_contents.get(email_id, "Email content not found")
        return content.strip()
        
    except Exception as e:
        return f"Failed to extract email content: {str(e)}"

# --- PDF Processing Tools ---

@tool
def extract_pdf_text(pdf_content: bytes) -> str:
    """Extracts text from PDF using PyPDF2."""
    
    try:
        # Use PyPDF2 to extract text
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        text_content = ""
        
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                text_content += f"--- Page {page_num + 1} ---\n"
                text_content += page_text + "\n\n"
            except Exception as page_error:
                text_content += f"--- Page {page_num + 1} (extraction failed) ---\n"
                text_content += f"Error: {str(page_error)}\n\n"
        
        # If minimal text extracted, suggest OCR
        if len(text_content.strip()) < 100:
            return f"Minimal text extracted ({len(text_content)} chars). Document may be scanned - OCR recommended.\n\n{text_content}"
        
        return text_content.strip()
        
    except Exception as e:
        return f"PDF text extraction failed: {str(e)}. Try OCR for scanned documents."

@tool 
def ocr_pdf_images(pdf_content: bytes) -> str:
    """OCR for scanned PDF documents using Tesseract."""
    
    try:
        # Mock OCR implementation - in real implementation:
        # 1. Convert PDF to images using pdf2image
        # 2. Apply pytesseract OCR to each image
        # 3. Combine results
        
        # Simulated OCR result for demonstration
        mock_ocr_result = """
--- OCR EXTRACTED CONTENT ---

CLAIM PROCESSING DOCUMENT

Date: June 19, 2025
Document Type: Claim Review Notice

TRIGGER CODES IDENTIFIED:
- TRG-001: High Priority Review Required
- CLM-456: Standard Processing Required  
- TRG-789: Additional Documentation Needed

CLAIM DETAILS:
Claimant: Michael Johnson
Policy: POL-555888
Incident Date: June 17, 2025

PROCESSING INSTRUCTIONS:
1. Immediate review required for TRG-001
2. Standard workflow for CLM-456
3. Request additional docs for TRG-789

REQUIRED DOCUMENTS:
- Form A-123
- Identity verification
- Medical records (if applicable)
- Supporting evidence

STATUS: Pending Processor Review
PRIORITY: HIGH

--- END OCR CONTENT ---
        """
        
        return mock_ocr_result.strip()
        
    except Exception as e:
        return f"OCR processing failed: {str(e)}"

# ==================== REACT AGENT CREATION ====================

def create_input_processing_agent():
    """Creates the ReAct agent for input processing with all available tools."""
    
    # Define all available tools
    tools = [
        validate_text_input,
        sanitize_text,
        authenticate_outlook_system,
        search_outlook_emails,
        extract_outlook_email_content,
        extract_pdf_text,
        ocr_pdf_images
    ]
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # Create ReAct agent with comprehensive system prompt
    agent = create_react_agent(
        llm,
        tools,
        state_modifier="""
        You are an expert Input Processing Agent for a claim processing system.
        
        Your primary goal is to extract clean, plain text content from various input formats.
        
        SUPPORTED INPUT TYPES:
        1. RAW TEXT (copy/paste) - Use validate_text_input() and sanitize_text()
        2. EMAIL QUERY (Outlook access) - Use authenticate_outlook_system(), search_outlook_emails(), extract_outlook_email_content()
        3. PDF UPLOAD (document processing) - Use extract_pdf_text(), fallback to ocr_pdf_images()
        
        PROCESSING STRATEGY:
        1. Analyze the user's input to determine the input type
        2. Select appropriate tool chain for that input type
        3. Extract and clean the content
        4. Return plain text ready for trigger code analysis
        
        TOOL SELECTION LOGIC:
        - For text input: validate → sanitize → return clean text
        - For email queries: authenticate → search → extract content → sanitize
        - For PDF upload: extract text → if minimal content, try OCR → sanitize
        
        Always prioritize getting clean, readable text that preserves important information
        like trigger codes, claim references, and processing instructions.
        
        If one method fails, try alternatives. For example, if PDF text extraction yields
        minimal content, automatically try OCR.
        """
    )
    
    return agent

# ==================== WORKFLOW IMPLEMENTATION ====================

def input_processing_node(state: InputProcessingState) -> InputProcessingState:
    """Main workflow node that processes user input using ReAct agent."""
    
    try:
        # Create agent
        agent = create_input_processing_agent()
        
        # Analyze user input to determine processing approach
        user_input = state["user_input"]
        
        # Create agent query
        agent_query = f"""
        Process this user input and extract clean text content:
        
        USER INPUT: {user_input}
        
        Steps:
        1. Determine the input type (raw_text, email_query, or pdf_upload)
        2. Use appropriate tools to process the input
        3. Extract clean, plain text content
        4. Ensure trigger codes and important information are preserved
        
        Return the final clean text content ready for trigger code extraction.
        """
        
        # Execute agent
        result = agent.invoke({"messages": [("user", agent_query)]})
        
        # Extract agent response
        agent_response = result['messages'][-1].content
        
        # For demonstration, also determine input type and process directly
        input_type = determine_input_type(user_input)
        processed_content = process_based_on_type(user_input, input_type)
        
        # Update state
        state["raw_content"] = processed_content
        state["input_type"] = input_type
        state["processing_metadata"] = {
            "processed_at": datetime.now().isoformat(),
            "agent_response": agent_response,
            "input_type_detected": input_type,
            "content_length": len(processed_content),
            "success": True
        }
        
    except Exception as e:
        state["errors"].append(f"Input processing failed: {str(e)}")
        state["raw_content"] = state["user_input"]  # Fallback to original
        state["input_type"] = "raw_text"
        state["processing_metadata"] = {
            "processed_at": datetime.now().isoformat(),
            "success": False,
            "error": str(e)
        }
    
    return state

def determine_input_type(user_input: str) -> str:
    """Helper function to determine input type from user query."""
    
    user_input_lower = user_input.lower()
    
    # Check for email-related keywords
    email_keywords = ["email", "outlook", "search", "sender", "subject", "latest"]
    if any(keyword in user_input_lower for keyword in email_keywords):
        return "email_query"
    
    # Check for PDF-related keywords
    pdf_keywords = ["pdf", "document", "file", "upload", "scan"]
    if any(keyword in user_input_lower for keyword in pdf_keywords):
        return "pdf_upload"
    
    # Default to raw text
    return "raw_text"

def process_based_on_type(user_input: str, input_type: str) -> str:
    """Helper function to process input based on detected type."""
    
    try:
        if input_type == "raw_text":
            # Direct text processing
            validation = validate_text_input(user_input)
            if validation["valid"]:
                return sanitize_text(user_input)
            else:
                return f"Text validation failed: {validation['error']}"
        
        elif input_type == "email_query":
            # Email processing simulation
            auth_result = authenticate_outlook_system()
            if auth_result["success"]:
                emails = search_outlook_emails(days_back=1)
                if emails and "error" not in emails[0]:
                    email_content = extract_outlook_email_content(emails[0]["id"])
                    return sanitize_text(email_content)
                else:
                    return "No recent emails found or search failed"
            else:
                return f"Email authentication failed: {auth_result['error']}"
        
        elif input_type == "pdf_upload":
            # PDF processing simulation (would need actual PDF bytes)
            return "PDF processing would require actual file upload. Mock content: TRG-001 claim processing required."
        
        else:
            return sanitize_text(user_input)  # Fallback
            
    except Exception as e:
        return f"Processing failed: {str(e)}. Fallback: {user_input}"

# ==================== LANGGRAPH WORKFLOW ====================

def create_input_processing_workflow():
    """Creates the LangGraph workflow for input processing."""
    
    # Create workflow
    workflow = StateGraph(InputProcessingState)
    
    # Add single processing node
    workflow.add_node("input_processing", input_processing_node)
    
    # Define flow
    workflow.add_edge(START, "input_processing")
    workflow.add_edge("input_processing", END)
    
    return workflow.compile()

# ==================== INTEGRATION INTERFACE ====================

def process_input(user_input: str) -> Dict[str, Any]:
    """Main interface for Module 1 - processes any input type and returns clean text."""
    
    # Initialize state
    initial_state = InputProcessingState(
        user_input=user_input,
        input_type="",
        raw_content="",
        processing_metadata={},
        errors=[]
    )
    
    # Run workflow
    workflow = create_input_processing_workflow()
    result = workflow.invoke(initial_state)
    
    return result

# ==================== USAGE EXAMPLES ====================

if __name__ == "__main__":
    print("=== MODULE 1: INPUT PROCESSING TESTING ===")
    
    # Test cases for different input types
    test_cases = [
        {
            "name": "Raw Text Input",
            "input": """
            Claim Processing Alert: TRG-001 requires immediate attention.
            Claimant: John Smith, Policy: POL-123456
            Additional codes: CLM-456, TRG-789
            Please process according to high priority procedures.
            """
        },
        {
            "name": "Email Query",
            "input": "Search for the latest email from claims@company.com about TRG-001"
        },
        {
            "name": "PDF Processing", 
            "input": "Process the uploaded PDF document containing claim trigger codes"
        }
    ]
    
    for test_case in test_cases:
        print(f"\n--- Testing: {test_case['name']} ---")
        
        result = process_input(test_case['input'])
        
        print(f"Input Type: {result['input_type']}")
        print(f"Success: {result['processing_metadata'].get('success', False)}")
        print(f"Content Length: {len(result['raw_content'])}")
        print(f"Raw Content Preview: {result['raw_content'][:200]}...")
        
        if result['errors']:
            print(f"Errors: {result['errors']}")
        
        print("-" * 50)
    
    print("\n=== MODULE 1 TESTING COMPLETE ===")