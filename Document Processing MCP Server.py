#!/usr/bin/env python3
"""
Document Processing MCP Server for Claim Processing System
Handles PDF processing, OCR, text extraction, and trigger code detection
"""
import asyncio
import io
import re
import json
import base64
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
import tempfile
import os

from mcp.server.fastmcp import FastMCP
import PyPDF2
import pytesseract
from PIL import Image
import fitz  # PyMuPDF for better PDF handling
import pandas as pd
from pdf2image import convert_from_bytes

@dataclass
class DocumentMetadata:
    filename: str
    file_size: int
    file_type: str
    page_count: Optional[int]
    processing_timestamp: str
    has_text: bool
    has_images: bool

@dataclass
class TextExtractionResult:
    success: bool
    extracted_text: str
    method_used: str
    confidence_score: Optional[float]
    page_count: int
    error_message: Optional[str]

@dataclass
class TriggerCodeDetection:
    potential_codes: List[Dict[str, Any]]
    confidence_scores: List[float]
    extraction_method: str
    context_snippets: List[str]

class DocumentProcessor:
    """Handles various document processing operations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.temp_dir = Path(config.get("temp_directory", "/tmp"))
        self.temp_dir.mkdir(exist_ok=True)
        
        # Trigger code patterns
        self.trigger_patterns = [
            r'\b[A-Z]{3}-\d{3}\b',  # TRG-001 format
            r'\b[A-Z]{2,4}\d{2,4}\b',  # CLM123 format
            r'\bTRG\d{3,4}\b',  # TRG001 format
            r'\bCLM[-_]?\d{3,4}\b',  # CLM-001 or CLM_001
            r'\bPOL[-_]?\d{3,4}\b',  # POL-001 format
            r'\bESC[-_]?\d{3,4}\b',  # ESC-001 format
            r'\bPRI[-_]?\d{3,4}\b',  # PRI-001 format
        ]
    
    def validate_text_input(self, text: str) -> Dict[str, Any]:
        """Validate raw text input quality"""
        if not text or not text.strip():
            return {
                "valid": False,
                "issues": ["Empty text input"],
                "word_count": 0,
                "char_count": 0
            }
        
        word_count = len(text.split())
        char_count = len(text.strip())
        issues = []
        
        # Check for minimum content
        if word_count < 5:
            issues.append("Text too short (minimum 5 words)")
        
        # Check for encoding issues
        if any(ord(char) > 127 for char in text):
            # Contains non-ASCII characters - might be encoding issues
            non_ascii_count = sum(1 for char in text if ord(char) > 127)
            if non_ascii_count > len(text) * 0.1:  # More than 10% non-ASCII
                issues.append("Possible encoding issues detected")
        
        # Check for excessive special characters
        special_char_count = sum(1 for char in text if not char.isalnum() and not char.isspace())
        if special_char_count > len(text) * 0.3:  # More than 30% special chars
            issues.append("High percentage of special characters")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "word_count": word_count,
            "char_count": char_count,
            "quality_score": max(0, 100 - len(issues) * 20)
        }
    
    def sanitize_text(self, raw_text: str) -> str:
        """Clean and normalize text formatting"""
        if not raw_text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', raw_text.strip())
        
        # Remove common OCR artifacts
        text = re.sub(r'[^\w\s\-.,;:!?()\[\]{}@#$%^&*+=<>/\\|`~"\'']', '', text)
        
        # Fix common OCR mistakes
        replacements = {
            r'\b0\b': 'O',  # Zero to O
            r'\bl\b': 'I',  # lowercase l to I
            r'rn': 'm',     # rn to m
            r'vv': 'w',     # vv to w
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        
        # Normalize line breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    async def extract_pdf_text(self, pdf_content: bytes) -> TextExtractionResult:
        """Extract text from PDF using PyPDF2"""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
            page_count = len(pdf_reader.pages)
            
            extracted_text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    extracted_text += page_text + "\n"
            
            has_text = bool(extracted_text.strip())
            
            return TextExtractionResult(
                success=True,
                extracted_text=extracted_text,
                method_used="PyPDF2",
                confidence_score=0.9 if has_text else 0.1,
                page_count=page_count,
                error_message=None
            )
            
        except Exception as e:
            return TextExtractionResult(
                success=False,
                extracted_text="",
                method_used="PyPDF2",
                confidence_score=0.0,
                page_count=0,
                error_message=str(e)
            )
    
    async def extract_pdf_text_advanced(self, pdf_content: bytes) -> TextExtractionResult:
        """Extract text using PyMuPDF (better for complex PDFs)"""
        try:
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            page_count = len(doc)
            
            extracted_text = ""
            for page_num in range(page_count):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                if page_text:
                    extracted_text += page_text + "\n"
            
            doc.close()
            
            has_text = bool(extracted_text.strip())
            
            return TextExtractionResult(
                success=True,
                extracted_text=extracted_text,
                method_used="PyMuPDF",
                confidence_score=0.95 if has_text else 0.1,
                page_count=page_count,
                error_message=None
            )
            
        except Exception as e:
            return TextExtractionResult(
                success=False,
                extracted_text="",
                method_used="PyMuPDF",
                confidence_score=0.0,
                page_count=0,
                error_message=str(e)
            )
    
    async def ocr_pdf_images(self, pdf_content: bytes) -> TextExtractionResult:
        """OCR for scanned PDFs using Tesseract"""
        try:
            # Convert PDF to images
            images = convert_from_bytes(pdf_content, dpi=300, fmt='jpeg')
            page_count = len(images)
            
            extracted_text = ""
            total_confidence = 0
            
            for i, image in enumerate(images):
                # OCR each page
                custom_config = r'--oem 3 --psm 6'  # OCR Engine Mode 3, Page Segmentation Mode 6
                
                # Get OCR result with confidence
                ocr_data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)
                
                # Extract text and calculate confidence
                page_text = ""
                confidences = []
                
                for j in range(len(ocr_data['text'])):
                    if int(ocr_data['conf'][j]) > 30:  # Filter low confidence words
                        word = ocr_data['text'][j].strip()
                        if word:
                            page_text += word + " "
                            confidences.append(int(ocr_data['conf'][j]))
                
                if page_text:
                    extracted_text += f"[Page {i+1}]\n{page_text}\n\n"
                    if confidences:
                        total_confidence += sum(confidences) / len(confidences)
            
            avg_confidence = total_confidence / page_count if page_count > 0 else 0
            
            return TextExtractionResult(
                success=True,
                extracted_text=extracted_text,
                method_used="Tesseract OCR",
                confidence_score=avg_confidence / 100,  # Convert to 0-1 scale
                page_count=page_count,
                error_message=None
            )
            
        except Exception as e:
            return TextExtractionResult(
                success=False,
                extracted_text="",
                method_used="Tesseract OCR",
                confidence_score=0.0,
                page_count=0,
                error_message=str(e)
            )
    
    def regex_extract_trigger_codes(self, text: str) -> TriggerCodeDetection:
        """Extract trigger codes using regex patterns"""
        potential_codes = []
        confidence_scores = []
        context_snippets = []
        
        for pattern in self.trigger_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                code = match.group().upper()
                start_pos = max(0, match.start() - 50)
                end_pos = min(len(text), match.end() + 50)
                context = text[start_pos:end_pos].strip()
                
                # Calculate confidence based on context
                confidence = 0.7  # Base confidence for regex match
                
                # Boost confidence for claim-related context
                claim_keywords = ['claim', 'policy', 'trigger', 'process', 'action', 'review']
                context_lower = context.lower()
                
                for keyword in claim_keywords:
                    if keyword in context_lower:
                        confidence += 0.05
                
                confidence = min(confidence, 1.0)
                
                potential_codes.append({
                    "code": code,
                    "pattern": pattern,
                    "position": match.start()
                })
                confidence_scores.append(confidence)
                context_snippets.append(context)
        
        return TriggerCodeDetection(
            potential_codes=potential_codes,
            confidence_scores=confidence_scores,
            extraction_method="regex_patterns",
            context_snippets=context_snippets
        )

@dataclass
class AppContext:
    processor: DocumentProcessor
    config: Dict[str, Any]

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle"""
    config = {
        "temp_directory": "/tmp/claim_processing",
        "max_file_size": 50 * 1024 * 1024,  # 50MB
        "supported_formats": [".pdf", ".txt", ".docx"],
        "ocr_dpi": 300,
        "tesseract_config": "--oem 3 --psm 6"
    }
    
    processor = DocumentProcessor(config)
    
    try:
        yield AppContext(processor=processor, config=config)
    finally:
        # Cleanup temp files
        pass

# Initialize FastMCP server
mcp = FastMCP(
    "Document Processing Server",
    dependencies=["PyPDF2", "pytesseract", "PIL", "PyMuPDF", "pdf2image"],
    lifespan=app_lifespan
)

# Resources
@mcp.resource("documents://processing-capabilities", title="Document Processing Capabilities")
def get_processing_capabilities() -> str:
    """Get information about document processing capabilities"""
    ctx = mcp.get_context()
    config = ctx.lifespan_context.config
    
    return f"""DOCUMENT PROCESSING CAPABILITIES
===============================

Supported Formats: {', '.join(config['supported_formats'])}
Maximum File Size: {config['max_file_size'] / (1024*1024):.0f}MB
OCR DPI Setting: {config['ocr_dpi']}

TEXT EXTRACTION METHODS:
✅ PyPDF2 - Fast extraction for text-based PDFs
✅ PyMuPDF - Advanced extraction for complex PDFs  
✅ Tesseract OCR - Image-based PDF processing
✅ Text validation and sanitization

TRIGGER CODE DETECTION:
✅ Regex pattern matching for common formats
✅ Context-aware confidence scoring
✅ Multi-pattern support (TRG-001, CLM123, etc.)

PROCESSING FEATURES:
✅ Batch document processing
✅ Quality validation
✅ Text sanitization
✅ Error handling and fallbacks
"""

@mcp.resource("documents://trigger-patterns", title="Trigger Code Patterns")
def get_trigger_patterns() -> str:
    """Get information about supported trigger code patterns"""
    ctx = mcp.get_context()
    processor = ctx.lifespan_context.processor
    
    patterns_info = """SUPPORTED TRIGGER CODE PATTERNS
==============================

1. TRG-001 format: [A-Z]{3}-\\d{3}
   Examples: TRG-001, CLM-456, POL-789

2. CLM123 format: [A-Z]{2,4}\\d{2,4}  
   Examples: CLM123, TRIG456, POL789

3. TRG001 format: TRG\\d{3,4}
   Examples: TRG001, TRG1234

4. CLM-001 format: CLM[-_]?\\d{3,4}
   Examples: CLM-001, CLM_123, CLM456

5. POL-001 format: POL[-_]?\\d{3,4}
   Examples: POL-001, POL_456

6. ESC-001 format: ESC[-_]?\\d{3,4}
   Examples: ESC-001, ESC_789

7. PRI-001 format: PRI[-_]?\\d{3,4}
   Examples: PRI-001, PRI_123

CONFIDENCE BOOSTING KEYWORDS:
- claim, policy, trigger, process, action, review
"""
    
    return patterns_info

# Tools
@mcp.tool()
def validate_text_input(text: str) -> str:
    """
    Validate raw text input quality for trigger code processing
    
    Args:
        text: Raw text content to validate
    """
    ctx = mcp.get_context()
    processor = ctx.lifespan_context.processor
    
    validation_result = processor.validate_text_input(text)
    
    output = f"""TEXT VALIDATION RESULTS
=====================

Valid: {'✅ Yes' if validation_result['valid'] else '❌ No'}
Word Count: {validation_result['word_count']}
Character Count: {validation_result['char_count']}
Quality Score: {validation_result['quality_score']}/100

"""
    
    if validation_result['issues']:
        output += "ISSUES DETECTED:\n"
        for issue in validation_result['issues']:
            output += f"⚠️  {issue}\n"
        output += "\nRECOMMENDATIONS:\n"
        output += "- Ensure text contains complete sentences\n"
        output += "- Check for encoding problems\n"
        output += "- Remove excessive special characters\n"
    else:
        output += "✅ Text quality is good - ready for processing"
    
    return output

@mcp.tool()
def sanitize_text(raw_text: str) -> str:
    """
    Clean and normalize text formatting for better processing
    
    Args:
        raw_text: Raw text content to clean
    """
    ctx = mcp.get_context()
    processor = ctx.lifespan_context.processor
    
    if not raw_text or not raw_text.strip():
        return "❌ Empty text provided"
    
    original_length = len(raw_text)
    sanitized_text = processor.sanitize_text(raw_text)
    new_length = len(sanitized_text)
    
    return f"""TEXT SANITIZATION COMPLETE
========================

Original Length: {original_length} characters
Cleaned Length: {new_length} characters
Reduction: {original_length - new_length} characters removed

CLEANED TEXT:
{sanitized_text}
"""

@mcp.tool()
async def extract_pdf_text(pdf_base64: str, filename: str = "document.pdf") -> str:
    """
    Extract text from PDF document using multiple methods
    
    Args:
        pdf_base64: Base64 encoded PDF content
        filename: Original filename for reference
    """
    ctx = mcp.get_context()
    processor = ctx.lifespan_context.processor
    
    try:
        # Decode base64 PDF content
        pdf_content = base64.b64decode(pdf_base64)
        
        # Try advanced extraction first
        result = await processor.extract_pdf_text_advanced(pdf_content)
        
        # If advanced fails or low confidence, try basic method
        if not result.success or result.confidence_score < 0.5:
            basic_result = await processor.extract_pdf_text(pdf_content)
            if basic_result.success and basic_result.confidence_score > result.confidence_score:
                result = basic_result
        
        # If still no good text, try OCR
        if not result.success or result.confidence_score < 0.3:
            ocr_result = await processor.ocr_pdf_images(pdf_content)
            if ocr_result.success and ocr_result.confidence_score > result.confidence_score:
                result = ocr_result
        
        if result.success:
            output = f"""PDF TEXT EXTRACTION SUCCESSFUL
============================

Filename: {filename}
Method Used: {result.method_used}
Pages Processed: {result.page_count}
Confidence Score: {result.confidence_score:.2f}
Text Length: {len(result.extracted_text)} characters

EXTRACTED CONTENT:
{result.extracted_text[:2000]}{'...' if len(result.extracted_text) > 2000 else ''}
"""
        else:
            output = f"""PDF TEXT EXTRACTION FAILED
=========================

Filename: {filename}
Error: {result.error_message}

Suggestions:
- Try a different PDF file
- Ensure PDF is not corrupted
- Check if PDF has text content or images
"""
        
        return output
        
    except Exception as e:
        return f"❌ Error processing PDF: {str(e)}"

@mcp.tool()
def extract_trigger_codes_regex(text: str) -> str:
    """
    Extract potential trigger codes from text using regex patterns
    
    Args:
        text: Text content to analyze for trigger codes
    """
    ctx = mcp.get_context()
    processor = ctx.lifespan_context.processor
    
    if not text or not text.strip():
        return "❌ Empty text provided"
    
    detection = processor.regex_extract_trigger_codes(text)
    
    if not detection.potential_codes:
        return f"""NO TRIGGER CODES DETECTED
========================

Text Length: {len(text)} characters
Patterns Checked: {len(processor.trigger_patterns)}

The text does not contain any recognizable trigger code patterns.

SUPPORTED PATTERNS:
- TRG-001, CLM-456 (letter-number format)
- CLM123, TRIG456 (alphanumeric format)
- TRG001, POL789 (concatenated format)
"""
    
    output = f"""TRIGGER CODES DETECTED
====================

Total Codes Found: {len(detection.potential_codes)}
Extraction Method: {detection.extraction_method}

DETECTED CODES:
"""
    
    for i, (code_info, confidence, context) in enumerate(zip(
        detection.potential_codes, 
        detection.confidence_scores, 
        detection.context_snippets
    ), 1):
        output += f"\n{i}. {code_info['code']}\n"
        output += f"   Confidence: {confidence:.2f}\n"
        output += f"   Pattern: {code_info['pattern']}\n"
        output += f"   Context: ...{context}...\n"
    
    # Extract unique codes
    unique_codes = list(set(code['code'] for code in detection.potential_codes))
    output += f"\nUNIQUE CODES FOR POLICY LOOKUP:\n"
    for code in sorted(unique_codes):
        output += f"- {code}\n"
    
    return output

@mcp.tool()
async def process_document_complete(pdf_base64: str, filename: str = "document.pdf") -> str:
    """
    Complete document processing: extract text and detect trigger codes
    
    Args:
        pdf_base64: Base64 encoded PDF content
        filename: Original filename for reference
    """
    ctx = mcp.get_context()
    processor = ctx.lifespan_context.processor
    
    try:
        # Step 1: Extract text
        pdf_content = base64.b64decode(pdf_base64)
        extraction_result = await processor.extract_pdf_text_advanced(pdf_content)
        
        if not extraction_result.success:
            # Try OCR as fallback
            extraction_result = await processor.ocr_pdf_images(pdf_content)
        
        if not extraction_result.success:
            return f"❌ Failed to extract text from {filename}: {extraction_result.error_message}"
        
        # Step 2: Sanitize text
        clean_text = processor.sanitize_text(extraction_result.extracted_text)
        
        # Step 3: Detect trigger codes
        detection = processor.regex_extract_trigger_codes(clean_text)
        
        # Step 4: Generate comprehensive report
        unique_codes = list(set(code['code'] for code in detection.potential_codes))
        
        output = f"""COMPLETE DOCUMENT PROCESSING REPORT
=================================

DOCUMENT INFO:
- Filename: {filename}
- Pages: {extraction_result.page_count}
- Extraction Method: {extraction_result.method_used}
- Text Quality: {extraction_result.confidence_score:.2f}

TEXT PROCESSING:
- Original Length: {len(extraction_result.extracted_text)} characters
- Cleaned Length: {len(clean_text)} characters

TRIGGER CODE ANALYSIS:
- Total Codes Found: {len(detection.potential_codes)}
- Unique Codes: {len(unique_codes)}

DETECTED TRIGGER CODES:
"""
        
        if unique_codes:
            for code in sorted(unique_codes):
                # Find the best confidence score for this code
                code_confidences = [conf for code_info, conf in zip(detection.potential_codes, detection.confidence_scores) 
                                   if code_info['code'] == code]
                max_confidence = max(code_confidences) if code_confidences else 0
                output += f"✅ {code} (confidence: {max_confidence:.2f})\n"
            
            output += f"\nREADY FOR POLICY LOOKUP:\n"
            output += f"Use these codes with the Policy Database MCP Server:\n"
            for code in sorted(unique_codes):
                output += f"- {code}\n"
        else:
            output += "❌ No trigger codes detected in document\n"
            output += "\nTROUBLESHOOTING:\n"
            output += "- Check if document contains claim-related content\n"
            output += "- Verify text extraction quality\n"
            output += "- Try manual text input if OCR quality is poor\n"
        
        return output
        
    except Exception as e:
        return f"❌ Error in complete document processing: {str(e)}"

@mcp.tool()
async def batch_process_documents(documents: List[Dict[str, str]]) -> str:
    """
    Process multiple documents in batch for trigger code extraction
    
    Args:
        documents: List of documents with 'pdf_base64' and 'filename' keys
    """
    ctx = mcp.get_context()
    processor = ctx.lifespan_context.processor
    config = ctx.lifespan_context.config
    
    if len(documents) > 10:
        return "❌ Maximum 10 documents allowed per batch"
    
    results = []
    all_trigger_codes = set()
    
    for i, doc in enumerate(documents, 1):
        try:
            filename = doc.get('filename', f'document_{i}.pdf')
            pdf_content = base64.b64decode(doc['pdf_base64'])
            
            # Extract text
            extraction_result = await processor.extract_pdf_text_advanced(pdf_content)
            if not extraction_result.success:
                extraction_result = await processor.ocr_pdf_images(pdf_content)
            
            if extraction_result.success:
                # Clean and detect codes
                clean_text = processor.sanitize_text(extraction_result.extracted_text)
                detection = processor.regex_extract_trigger_codes(clean_text)
                unique_codes = list(set(code['code'] for code in detection.potential_codes))
                all_trigger_codes.update(unique_codes)
                
                results.append({
                    'filename': filename,
                    'success': True,
                    'codes_found': len(unique_codes),
                    'codes': unique_codes,
                    'method': extraction_result.method_used,
                    'confidence': extraction_result.confidence_score
                })
            else:
                results.append({
                    'filename': filename,
                    'success': False,
                    'error': extraction_result.error_message
                })
                
        except Exception as e:
            results.append({
                'filename': doc.get('filename', f'document_{i}.pdf'),
                'success': False,
                'error': str(e)
            })
    
    # Generate batch report
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    output = f"""BATCH DOCUMENT PROCESSING RESULTS
===============================

Total Documents: {len(documents)}
Successful: {len(successful)}
Failed: {len(failed)}
Total Unique Codes: {len(all_trigger_codes)}

PROCESSING RESULTS:
"""
    
    for result in results:
        if result['success']:
            output += f"✅ {result['filename']}\n"
            output += f"   Method: {result['method']}\n"
            output += f"   Codes Found: {result['codes_found']}\n"
            output += f"   Confidence: {result['confidence']:.2f}\n"
            if result['codes']:
                output += f"   Codes: {', '.join(result['codes'])}\n"
            output += "\n"
        else:
            output += f"❌ {result['filename']}\n"
            output += f"   Error: {result['error']}\n\n"
    
    if all_trigger_codes:
        output += f"ALL UNIQUE TRIGGER CODES FOUND:\n"
        for code in sorted(all_trigger_codes):
            output += f"- {code}\n"
        output += f"\n💡 Use batch_lookup_trigger_codes in Policy Database MCP Server with these codes"
    
    return output

@mcp.tool()
def extract_codes_with_context(text: str, context_window: int = 100) -> str:
    """
    Extract trigger codes with surrounding context for better analysis
    
    Args:
        text: Text to analyze
        context_window: Number of characters before/after each code for context
    """
    ctx = mcp.get_context()
    processor = ctx.lifespan_context.processor
    
    if not text or not text.strip():
        return "❌ Empty text provided"
    
    detection = processor.regex_extract_trigger_codes(text)
    
    if not detection.potential_codes:
        return "No trigger codes found in the provided text"
    
    output = f"""TRIGGER CODES WITH CONTEXT
=========================

Text Length: {len(text)} characters
Codes Found: {len(detection.potential_codes)}
Context Window: ±{context_window} characters

DETAILED ANALYSIS:
"""
    
    for i, (code_info, confidence) in enumerate(zip(detection.potential_codes, detection.confidence_scores), 1):
        # Find position and extract context
        position = code_info['position']
        start_pos = max(0, position - context_window)
        end_pos = min(len(text), position + len(code_info['code']) + context_window)
        
        context = text[start_pos:end_pos]
        
        # Highlight the code in context
        highlighted_context = context.replace(
            code_info['code'], 
            f"**{code_info['code']}**"
        )
        
        output += f"\n{i}. CODE: {code_info['code']}\n"
        output += f"   Position: {position}\n"
        output += f"   Confidence: {confidence:.2f}\n"
        output += f"   Pattern: {code_info['pattern']}\n"
        output += f"   Context: ...{highlighted_context}...\n"
    
    return output

# Prompts for LangGraph agents
@mcp.prompt("document-processing-strategy")
def document_processing_strategy() -> str:
    """Strategy for effective document processing"""
    return """
DOCUMENT PROCESSING STRATEGY FOR CLAIM PROCESSING

PROCESSING PIPELINE:
1. Input Validation → 2. Text Extraction → 3. Text Cleaning → 4. Code Detection

TEXT EXTRACTION PRIORITY:
1. PyMuPDF (best for complex layouts)
2. PyPDF2 (fastest for simple PDFs)
3. Tesseract OCR (for scanned documents)

QUALITY ASSURANCE:
- Validate extraction confidence scores
- Use multiple methods for low-confidence results
- Apply text sanitization to improve accuracy
- Cross-validate detected codes

TRIGGER CODE DETECTION:
- Use comprehensive regex patterns
- Apply context-based confidence scoring
- Remove duplicates and normalize formats
- Validate codes against known patterns

ERROR HANDLING:
- Graceful fallback between extraction methods
- Clear error messaging for failed extractions
- Suggestions for manual processing when needed
- Batch processing limits to prevent timeouts

OPTIMIZATION TIPS:
- Process documents in batches when possible
- Cache processing results for repeated documents
- Use appropriate DPI settings for OCR (300 recommended)
- Pre-filter documents by type and size
"""

@mcp.prompt("trigger-code-patterns")
def trigger_code_patterns() -> str:
    """Comprehensive guide to trigger code patterns"""
    return """
TRIGGER CODE PATTERN RECOGNITION GUIDE

COMMON FORMATS:
1. Standard Format: TRG-001, CLM-456, POL-789
   - 3 letters, hyphen, 3 digits
   - Most reliable pattern

2. Compact Format: CLM123, TRIG456, ESC789
   - 2-4 letters followed by 2-4 digits
   - No separators

3. Underscore Format: TRG_001, CLM_456
   - Alternative separator style
   - Common in system exports

4. Zero-padded: TRG0001, CLM0456
   - Extended digit sequences
   - Database-generated codes

CONTEXT INDICATORS:
- "Trigger Code:", "Reference:", "Code:"
- "Process according to", "Follow procedure"
- "Priority:", "Action Required:", "Escalate"
- Near claim numbers, policy references

CONFIDENCE FACTORS:
- High: Explicit labeling (90-100%)
- Medium: Claim-related context (70-85%)
- Low: Pattern match only (50-70%)

VALIDATION RULES:
- Must match established patterns
- Should have claim-processing context
- Avoid false positives (phone numbers, dates)
- Cross-reference with policy database
"""

@mcp.prompt("ocr-optimization")
def ocr_optimization() -> str:
    """Guide for optimizing OCR results"""
    return """
OCR OPTIMIZATION FOR CLAIM DOCUMENTS

PRE-PROCESSING:
- Use 300 DPI minimum for scanning
- Ensure good contrast and lighting
- Remove noise and artifacts
- Straighten skewed documents

TESSERACT CONFIGURATION:
- OEM 3: Default LSTM OCR Engine
- PSM 6: Uniform block of text
- Language: eng (English)
- Custom config for claim docs

QUALITY IMPROVEMENT:
- Image preprocessing (contrast, sharpening)
- Multiple resolution attempts
- Character-level confidence filtering
- Post-processing text correction

VALIDATION TECHNIQUES:
- Compare multiple OCR attempts
- Cross-validate with pattern matching
- Manual review for critical documents
- Confidence score thresholds

COMMON ISSUES:
- Poor scan quality → Use image enhancement
- Mixed fonts → Try different PSM modes
- Tables/forms → Use structured OCR
- Handwritten text → May require manual entry

PERFORMANCE OPTIMIZATION:
- Batch process similar documents
- Cache OCR results
- Use appropriate image formats
- Limit processing time per document
"""

if __name__ == "__main__":
    mcp.run()