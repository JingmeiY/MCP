#!/usr/bin/env python3
"""
Outlook MCP Server for Claim Processing System
Handles email authentication, search, and content extraction
"""
import asyncio
import json
import base64
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from mcp.server.fastmcp import FastMCP
import msal
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

@dataclass
class EmailMetadata:
    email_id: str
    subject: str
    sender: str
    received_date: str
    has_attachments: bool
    importance: str

@dataclass
class EmailContent:
    email_id: str
    subject: str
    sender: str
    body_text: str
    body_html: str
    attachments: List[Dict[str, Any]]

class OutlookClient:
    """Microsoft Graph API client for Outlook operations"""
    
    def __init__(self, client_id: str, client_secret: str, tenant_id: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.tenant_id = tenant_id
        self.access_token = None
        self.app = None
        
    async def authenticate(self) -> Dict[str, Any]:
        """Authenticate with Microsoft Graph API"""
        try:
            authority = f"https://login.microsoftonline.com/{self.tenant_id}"
            self.app = msal.ConfidentialClientApplication(
                self.client_id,
                authority=authority,
                client_credential=self.client_secret
            )
            
            # Get token for application permissions
            scopes = ["https://graph.microsoft.com/.default"]
            result = self.app.acquire_token_for_client(scopes=scopes)
            
            if "access_token" in result:
                self.access_token = result["access_token"]
                return {"success": True, "expires_in": result.get("expires_in")}
            else:
                return {"success": False, "error": result.get("error_description")}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def search_emails(
        self, 
        sender: Optional[str] = None,
        subject: Optional[str] = None,
        days_back: int = 7,
        limit: int = 50
    ) -> List[EmailMetadata]:
        """Search emails with filters"""
        if not self.access_token:
            raise ValueError("Not authenticated. Call authenticate() first.")
        
        # Build OData filter
        filters = []
        
        if sender:
            filters.append(f"from/emailAddress/address eq '{sender}'")
        
        if subject:
            filters.append(f"contains(subject, '{subject}')")
        
        # Date filter
        since_date = (datetime.now() - timedelta(days=days_back)).isoformat() + 'Z'
        filters.append(f"receivedDateTime ge {since_date}")
        
        filter_query = " and ".join(filters) if filters else ""
        
        # Graph API endpoint
        url = "https://graph.microsoft.com/v1.0/me/messages"
        params = {
            "$select": "id,subject,from,receivedDateTime,hasAttachments,importance",
            "$top": limit,
            "$orderby": "receivedDateTime desc"
        }
        
        if filter_query:
            params["$filter"] = filter_query
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            emails = []
            
            for email in data.get("value", []):
                emails.append(EmailMetadata(
                    email_id=email["id"],
                    subject=email["subject"],
                    sender=email["from"]["emailAddress"]["address"],
                    received_date=email["receivedDateTime"],
                    has_attachments=email["hasAttachments"],
                    importance=email["importance"]
                ))
            
            return emails
            
        except requests.RequestException as e:
            raise ValueError(f"Failed to search emails: {str(e)}")
    
    async def get_email_content(self, email_id: str) -> EmailContent:
        """Extract full email content including body and attachments"""
        if not self.access_token:
            raise ValueError("Not authenticated. Call authenticate() first.")
        
        url = f"https://graph.microsoft.com/v1.0/me/messages/{email_id}"
        params = {
            "$select": "id,subject,from,body,attachments"
        }
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            email_data = response.json()
            
            # Get attachments info
            attachments = []
            if email_data.get("attachments"):
                for attachment in email_data["attachments"]:
                    attachments.append({
                        "name": attachment.get("name"),
                        "content_type": attachment.get("contentType"),
                        "size": attachment.get("size"),
                        "id": attachment.get("id")
                    })
            
            return EmailContent(
                email_id=email_id,
                subject=email_data["subject"],
                sender=email_data["from"]["emailAddress"]["address"],
                body_text=email_data["body"].get("content", ""),
                body_html=email_data["body"].get("content", "") if email_data["body"]["contentType"] == "html" else "",
                attachments=attachments
            )
            
        except requests.RequestException as e:
            raise ValueError(f"Failed to get email content: {str(e)}")

@dataclass
class AppContext:
    outlook_client: OutlookClient
    config: Dict[str, Any]

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle"""
    # Initialize Outlook client
    outlook_client = OutlookClient(
        client_id="your-client-id",  # Load from environment
        client_secret="your-client-secret",
        tenant_id="your-tenant-id"
    )
    
    config = {
        "max_emails_per_search": 100,
        "default_search_days": 7,
        "supported_attachments": [".pdf", ".docx", ".txt", ".jpg", ".png"]
    }
    
    try:
        yield AppContext(outlook_client=outlook_client, config=config)
    finally:
        # Cleanup if needed
        pass

# Initialize FastMCP server
mcp = FastMCP(
    "Outlook Claim Processing Server",
    dependencies=["msal", "requests"],
    lifespan=app_lifespan
)

# Resources
@mcp.resource("outlook://auth-status", title="Outlook Authentication Status")
def get_auth_status() -> str:
    """Get current authentication status"""
    ctx = mcp.get_context()
    client = ctx.lifespan_context.outlook_client
    
    if client.access_token:
        return "✅ Authenticated with Microsoft Graph API\nReady to access Outlook emails"
    else:
        return "❌ Not authenticated\nRun authenticate_outlook tool first"

@mcp.resource("outlook://config", title="Outlook Server Configuration")
def get_outlook_config() -> str:
    """Get server configuration"""
    ctx = mcp.get_context()
    config = ctx.lifespan_context.config
    
    return f"""Outlook MCP Server Configuration:
Max Emails per Search: {config['max_emails_per_search']}
Default Search Days: {config['default_search_days']}
Supported Attachments: {', '.join(config['supported_attachments'])}"""

# Tools
@mcp.tool()
async def authenticate_outlook() -> str:
    """
    Authenticate with Outlook/Microsoft Graph API
    Required before using any other email tools
    """
    ctx = mcp.get_context()
    client = ctx.lifespan_context.outlook_client
    
    result = await client.authenticate()
    
    if result["success"]:
        return f"✅ Successfully authenticated with Outlook\nToken expires in: {result.get('expires_in', 'Unknown')} seconds"
    else:
        return f"❌ Authentication failed: {result.get('error', 'Unknown error')}"

@mcp.tool()
async def search_claim_emails(
    sender: Optional[str] = None,
    subject_keywords: Optional[str] = None,
    days_back: int = 7,
    limit: int = 20
) -> str:
    """
    Search for claim-related emails in Outlook
    
    Args:
        sender: Filter by sender email address
        subject_keywords: Keywords to search in email subject
        days_back: Number of days to search back (default: 7)
        limit: Maximum number of emails to return (default: 20)
    """
    ctx = mcp.get_context()
    client = ctx.lifespan_context.outlook_client
    config = ctx.lifespan_context.config
    
    if not client.access_token:
        return "❌ Not authenticated. Run authenticate_outlook tool first."
    
    # Limit the search to prevent overload
    limit = min(limit, config["max_emails_per_search"])
    
    try:
        emails = await client.search_emails(
            sender=sender,
            subject=subject_keywords,
            days_back=days_back,
            limit=limit
        )
        
        if not emails:
            return f"No emails found matching criteria:\n- Sender: {sender or 'Any'}\n- Subject: {subject_keywords or 'Any'}\n- Days back: {days_back}"
        
        result = f"Found {len(emails)} emails:\n\n"
        
        for i, email in enumerate(emails, 1):
            result += f"{i}. ID: {email.email_id}\n"
            result += f"   Subject: {email.subject}\n"
            result += f"   From: {email.sender}\n"
            result += f"   Date: {email.received_date}\n"
            result += f"   Attachments: {'Yes' if email.has_attachments else 'No'}\n"
            result += f"   Priority: {email.importance}\n\n"
        
        return result
        
    except Exception as e:
        return f"❌ Error searching emails: {str(e)}"

@mcp.tool()
async def extract_email_content(email_id: str) -> str:
    """
    Extract full content from a specific email for trigger code analysis
    
    Args:
        email_id: The email ID from search results
    """
    ctx = mcp.get_context()
    client = ctx.lifespan_context.outlook_client
    
    if not client.access_token:
        return "❌ Not authenticated. Run authenticate_outlook tool first."
    
    try:
        email_content = await client.get_email_content(email_id)
        
        # Clean and format the content for processing
        result = f"""EMAIL CONTENT EXTRACTED
========================

Subject: {email_content.subject}
From: {email_content.sender}
Email ID: {email_content.email_id}

CONTENT FOR TRIGGER CODE ANALYSIS:
{email_content.body_text}

ATTACHMENTS:
"""
        
        if email_content.attachments:
            for att in email_content.attachments:
                result += f"- {att['name']} ({att['content_type']}, {att['size']} bytes)\n"
        else:
            result += "No attachments\n"
        
        return result
        
    except Exception as e:
        return f"❌ Error extracting email content: {str(e)}"

@mcp.tool()
async def batch_extract_emails(email_ids: List[str]) -> str:
    """
    Extract content from multiple emails for batch processing
    
    Args:
        email_ids: List of email IDs to extract content from
    """
    ctx = mcp.get_context()
    client = ctx.lifespan_context.outlook_client
    
    if not client.access_token:
        return "❌ Not authenticated. Run authenticate_outlook tool first."
    
    if len(email_ids) > 10:
        return "❌ Maximum 10 emails can be processed in batch mode"
    
    results = []
    
    for email_id in email_ids:
        try:
            email_content = await client.get_email_content(email_id)
            results.append({
                "email_id": email_id,
                "subject": email_content.subject,
                "sender": email_content.sender,
                "content": email_content.body_text,
                "success": True
            })
        except Exception as e:
            results.append({
                "email_id": email_id,
                "error": str(e),
                "success": False
            })
    
    # Format batch results
    output = "BATCH EMAIL EXTRACTION RESULTS\n" + "="*40 + "\n\n"
    
    for i, result in enumerate(results, 1):
        if result["success"]:
            output += f"{i}. ✅ {result['subject']}\n"
            output += f"   From: {result['sender']}\n"
            output += f"   Content Length: {len(result['content'])} characters\n\n"
        else:
            output += f"{i}. ❌ Failed: {result['email_id']}\n"
            output += f"   Error: {result['error']}\n\n"
    
    return output

# Prompts for LangGraph agents
@mcp.prompt("outlook-search-guidance")
def outlook_search_guidance() -> str:
    """Guidance for searching claim-related emails effectively"""
    return """
OUTLOOK EMAIL SEARCH GUIDANCE FOR CLAIM PROCESSING

EFFECTIVE SEARCH STRATEGIES:
1. Use specific sender addresses for known claim sources
2. Include keywords like: "claim", "policy", "trigger", "processing"
3. Search recent emails first (7-14 days) for active claims
4. Check high-priority emails for urgent trigger codes

COMMON CLAIM EMAIL PATTERNS:
- Subjects containing: "Claim #", "Policy Update", "Trigger Code", "Action Required"
- Senders from: insurance companies, claim departments, policy systems
- Attachments: Often contain policy documents or claim forms

SEARCH TIPS:
- Start broad, then narrow with specific filters
- Check both subject and sender filters
- Look for emails with attachments (often contain trigger codes)
- Sort by importance/priority first
"""

@mcp.prompt("email-content-analysis")
def email_content_analysis() -> str:
    """Prompt for analyzing email content for trigger codes"""
    return """
EMAIL CONTENT ANALYSIS FOR TRIGGER CODE EXTRACTION

ANALYSIS OBJECTIVES:
1. Identify all potential trigger codes in email content
2. Extract relevant claim processing information
3. Determine urgency and priority levels
4. Identify required documentation or forms

TRIGGER CODE PATTERNS TO LOOK FOR:
- Alphanumeric codes (e.g., TRG-001, CLM-456, POL-789)
- Reference numbers in claim context
- Policy violation codes
- Processing instruction codes
- Escalation triggers

CONTENT AREAS TO EXAMINE:
- Email subject line (often contains primary codes)
- Email body text (detailed trigger information)
- Attachment names (may contain code references)
- Signature blocks (system-generated codes)

OUTPUT FORMAT:
- List all identified codes with confidence levels
- Note context for each code found
- Flag urgent or high-priority items
- Summarize key claim processing requirements
"""

if __name__ == "__main__":
    mcp.run()