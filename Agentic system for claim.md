# Claim Processing Agentic System - Blueprint

## 🎯 System Overview

This agentic system assists claim processors by automatically analyzing communications, extracting trigger codes, retrieving policy instructions, and generating actionable plans. The system uses **LangGraph** for workflow orchestration, **Model Context Protocol (MCP)** for data access and tools, and **ReAct agents** for intelligent decision-making.

### High-Level Architecture
```
Input Processing → Trigger Code Extraction → Policy Retrieval → Action Planning → Human-Readable Output
```

---

## 📋 Core Modules

### **Module 1: Input Processing & Content Extraction**

**Purpose**: Convert various input formats into clean, raw text content ready for trigger code analysis.

#### **Supported Input Methods**
1. **Raw Text Input** (Copy & Paste)
2. **Email Access** (Outlook integration via MCP)
3. **PDF Document Upload** (Text extraction + OCR)

#### **Core Functions**

| Function | Description | Input | Output |
|----------|-------------|--------|--------|
| `validate_text_input()` | Validates raw text input quality | `str: text` | `Dict: validation_result` |
| `sanitize_text()` | Cleans and normalizes text formatting | `str: raw_text` | `str: clean_text` |
| `authenticate_outlook_system()` | Handles OAuth2 authentication for Outlook | `None` | `Dict: auth_tokens` |
| `search_outlook_emails()` | Searches emails by sender, subject, date | `str: sender, str: subject, int: days_back` | `List[Dict]: email_metadata` |
| `extract_outlook_email_content()` | Extracts email subject + body as plain text | `str: email_id` | `str: plain_text_content` |
| `extract_pdf_text()` | Extracts text from PDF using PyPDF2 | `bytes: pdf_content` | `str: extracted_text` |
| `ocr_pdf_images()` | OCR for scanned PDFs using Tesseract | `bytes: pdf_content` | `str: ocr_text` |

#### **ReAct Agent Strategy**
The **Input Processing Agent** dynamically selects the appropriate tool chain:
1. **Text Input**: `validate_text_input()` → `sanitize_text()`
2. **Email Query**: `authenticate_outlook_system()` → `search_outlook_emails()` → `extract_outlook_email_content()`
3. **PDF Upload**: `extract_pdf_text()` → `ocr_pdf_images()` (fallback)

#### **Output Format**
```python
{
    "raw_content": "Plain text content containing trigger codes...",
    "input_type": "email_query|pdf_upload|raw_text",
    "processing_metadata": {
        "processed_at": "2025-06-19T10:30:00Z",
        "success": True
    }
}
```

---

### **Module 2: Trigger Code Extraction**

**Purpose**: Identify and extract all trigger codes from the processed text content using both AI and pattern matching.

#### **Core Functions**

| Function | Description | Input | Output |
|----------|-------------|--------|--------|
| `llm_extract_triggers()` | Primary LLM-based trigger code extraction | `str: text_content` | `List[Dict]: trigger_codes_with_confidence` |
| `regex_pattern_match()` | Backup regex pattern matching | `str: text_content` | `List[str]: matched_codes` |

#### **Trigger Code Patterns**
- Can be digits only or a combination of digits and letters.

#### **Extraction Strategy**
1. **Primary**: LLM with specialized prompt for contextual understanding
2. **Backup**: Regex pattern matching for reliability
3. **Combination**: Merge results 

#### **Output Format**
```python
{
    "trigger_codes": [
            "code": "TRG-001",
    ]
}
```

---

### **Module 3: Policy Lookup & Instruction Retrieval**

**Purpose**: Retrieve step-by-step instructions for each trigger code using multiple data sources via intelligent tool selection.

#### **Data Access Methods**

##### **Method 1: JSON File Access**
| Function | Description | Input | Output |
|----------|-------------|--------|--------|
| `load_policies_json()` | Loads complete policies from JSON file | `str: file_path` | `Dict: policies_data` |
| `batch_extract_from_json()` | Extracts multiple trigger codes from JSON | `Dict: policies_data, List[str]: codes` | `Dict: extraction_results` |

**JSON Structure:**
```json
{
  "TRG-001": {
    "instructions": {
      "step_1": {
        "action": "Immediate priority review for Form-A, ID-Copy",
      }
    },
  }
}
```

##### **Method 2: BigQuery Database Access**
| Function | Description | Input | Output |
|----------|-------------|--------|--------|
| `query_bigquery_policies()` | Queries BigQuery for trigger code instructions | `str: trigger_code` | `Dict: policy_data` |
| `batch_query_bigquery()` | Batch queries for multiple codes | `List[str]: trigger_codes` | `Dict: batch_results` |



##### **Method 4: RAG/Vector Store (will integrate later)**
| Function | Description | Input | Output |
|----------|-------------|--------|--------|
| `search_rag_policies()` | Semantic search for policy instructions | `List[str]: trigger_codes` | `Dict: rag_results` |

#### **ReAct Agent Strategy**
The **Policy Access Agent** intelligently selects data sources:
1. **Try JSON first** (fastest, most structured)
2. **Fallback to BigQuery** (if JSON incomplete)
4. **RAG** (semantic search capabilities)

#### **Output Format**
```python
{
    "policy_instructions": [
        {
            "trigger_code": "TRG-001",
            "found": True,
            "instructions": {...},
            "source": "json_file|bigquery|csv_file|rag"
        }
    ]
}
```

---

### **Module 4: Action Planning & Summary Generation**

**Purpose**: Generate concise, human-readable action plans with clear summaries and next steps.


#### **ReAct Agent**
The **Action Planning Agent** focuses solely on generating clear, actionable summaries for claim processors.


#### **Human-Readable Output**
```markdown
CLAIM PROCESSING ACTION PLAN

SUMMARY
Request: Process high priority claim requiring immediate attention and documentation review
Actions: Follow high priority procedures for TRG-001, complete standard processing for CLM-456

TRIGGER CODES FOUND
1. TRG-001
2. CLM-456
```

---

## 🔧 Technical Implementation

### **Technology Stack**
- **Framework**: LangGraph for workflow orchestration
- **LLM Provider**: OpenAI GPT-4o / Google Gemini (switchable)
- **Agent Pattern**: ReAct (Reasoning + Acting)
- **Data Access and Tools**: Model Context Protocol (MCP)
- **Authentication**: OAuth2 (Outlook/Microsoft Graph)
- **Document Processing**: PyPDF2, Tesseract OCR
- **Database**: BigQuery (enterprise) / Local files 

### **MCP Integration Points**

#### **Custom MCP Servers Needed**
1. **Outlook MCP Server**
   - OAuth2 authentication
   - Email search and retrieval
   - Content extraction

2. **Database MCP Server** 
   - BigQuery connection
   - Policy table access
   - Batch query support

3. **File System MCP Server**
   - JSON file access
   - Document upload handling
   - Local file management

### **LangGraph Workflow Architecture**

```python
# Main Pipeline
Input Processing → Trigger Extraction → Policy Retrieval → Action Planning

# Each module uses StateGraph with ReAct agents
StateGraph(ModuleState) → ReAct Agent → Tool Selection → Output
```




---



