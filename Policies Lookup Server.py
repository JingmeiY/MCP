#!/usr/bin/env python3
"""
Policy Database MCP Server for Claim Processing System
Handles policy lookup, instruction retrieval from multiple data sources
"""
import asyncio
import json
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
import sqlite3

from mcp.server.fastmcp import FastMCP
from google.cloud import bigquery
import csv

@dataclass
class PolicyInstruction:
    trigger_code: str
    title: str
    priority: str
    steps: List[Dict[str, Any]]
    required_documents: List[str]
    deadline: Optional[str]
    escalation_rules: Dict[str, Any]
    last_updated: str

@dataclass
class PolicyLookupResult:
    trigger_code: str
    found: bool
    source: str
    instruction: Optional[PolicyInstruction]
    error: Optional[str]

class PolicyDataManager:
    """Manages access to multiple policy data sources"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.json_data = None
        self.bigquery_client = None
        self.csv_data = None
        
    async def initialize(self):
        """Initialize all data sources"""
        await self._load_json_policies()
        await self._initialize_bigquery()
        await self._load_csv_policies()
    
    async def _load_json_policies(self):
        """Load policies from JSON file"""
        try:
            json_path = Path(self.config["policy_json_path"])
            if json_path.exists():
                with open(json_path, 'r') as f:
                    self.json_data = json.load(f)
                print(f"Loaded {len(self.json_data)} policies from JSON")
            else:
                print(f"JSON policy file not found: {json_path}")
                self.json_data = {}
        except Exception as e:
            print(f"Error loading JSON policies: {e}")
            self.json_data = {}
    
    async def _initialize_bigquery(self):
        """Initialize BigQuery client"""
        try:
            if self.config.get("bigquery_enabled"):
                self.bigquery_client = bigquery.Client(
                    project=self.config["bigquery_project"]
                )
                print("BigQuery client initialized")
        except Exception as e:
            print(f"Error initializing BigQuery: {e}")
            self.bigquery_client = None
    
    async def _load_csv_policies(self):
        """Load policies from CSV file"""
        try:
            csv_path = Path(self.config["policy_csv_path"])
            if csv_path.exists():
                self.csv_data = pd.read_csv(csv_path)
                print(f"Loaded {len(self.csv_data)} policies from CSV")
            else:
                print(f"CSV policy file not found: {csv_path}")
                self.csv_data = None
        except Exception as e:
            print(f"Error loading CSV policies: {e}")
            self.csv_data = None
    
    async def lookup_from_json(self, trigger_code: str) -> Optional[PolicyInstruction]:
        """Lookup policy from JSON data"""
        if not self.json_data or trigger_code not in self.json_data:
            return None
        
        policy_data = self.json_data[trigger_code]
        
        return PolicyInstruction(
            trigger_code=trigger_code,
            title=policy_data.get("title", ""),
            priority=policy_data.get("priority", "medium"),
            steps=policy_data.get("instructions", {}).get("steps", []),
            required_documents=policy_data.get("required_documents", []),
            deadline=policy_data.get("deadline"),
            escalation_rules=policy_data.get("escalation_rules", {}),
            last_updated=policy_data.get("last_updated", "")
        )
    
    async def lookup_from_bigquery(self, trigger_code: str) -> Optional[PolicyInstruction]:
        """Lookup policy from BigQuery"""
        if not self.bigquery_client:
            return None
        
        try:
            query = f"""
            SELECT 
                trigger_code,
                title,
                priority,
                instructions_json,
                required_documents_json,
                deadline,
                escalation_rules_json,
                last_updated
            FROM `{self.config['bigquery_dataset']}.{self.config['bigquery_table']}`
            WHERE trigger_code = @trigger_code
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("trigger_code", "STRING", trigger_code)
                ]
            )
            
            result = self.bigquery_client.query(query, job_config=job_config)
            
            for row in result:
                return PolicyInstruction(
                    trigger_code=row.trigger_code,
                    title=row.title,
                    priority=row.priority,
                    steps=json.loads(row.instructions_json) if row.instructions_json else [],
                    required_documents=json.loads(row.required_documents_json) if row.required_documents_json else [],
                    deadline=row.deadline,
                    escalation_rules=json.loads(row.escalation_rules_json) if row.escalation_rules_json else {},
                    last_updated=row.last_updated.isoformat() if row.last_updated else ""
                )
            
            return None
            
        except Exception as e:
            print(f"BigQuery lookup error: {e}")
            return None
    
    async def lookup_from_csv(self, trigger_code: str) -> Optional[PolicyInstruction]:
        """Lookup policy from CSV data"""
        if self.csv_data is None:
            return None
        
        try:
            matching_rows = self.csv_data[self.csv_data['trigger_code'] == trigger_code]
            
            if matching_rows.empty:
                return None
            
            row = matching_rows.iloc[0]
            
            # Parse JSON fields if they exist
            steps = []
            if pd.notna(row.get('instructions_json')):
                try:
                    steps = json.loads(row['instructions_json'])
                except:
                    steps = [{"step_1": {"action": row.get('instructions', '')}}]
            
            required_docs = []
            if pd.notna(row.get('required_documents')):
                required_docs = row['required_documents'].split(',') if isinstance(row['required_documents'], str) else []
            
            return PolicyInstruction(
                trigger_code=trigger_code,
                title=row.get('title', ''),
                priority=row.get('priority', 'medium'),
                steps=steps,
                required_documents=required_docs,
                deadline=row.get('deadline'),
                escalation_rules={},
                last_updated=row.get('last_updated', '')
            )
            
        except Exception as e:
            print(f"CSV lookup error: {e}")
            return None
    
    async def batch_lookup(self, trigger_codes: List[str]) -> List[PolicyLookupResult]:
        """Lookup multiple trigger codes with fallback strategy"""
        results = []
        
        for code in trigger_codes:
            result = await self.lookup_single_with_fallback(code)
            results.append(result)
        
        return results
    
    async def lookup_single_with_fallback(self, trigger_code: str) -> PolicyLookupResult:
        """Lookup single trigger code with fallback strategy"""
        # Try JSON first (fastest)
        instruction = await self.lookup_from_json(trigger_code)
        if instruction:
            return PolicyLookupResult(
                trigger_code=trigger_code,
                found=True,
                source="json_file",
                instruction=instruction,
                error=None
            )
        
        # Try BigQuery
        instruction = await self.lookup_from_bigquery(trigger_code)
        if instruction:
            return PolicyLookupResult(
                trigger_code=trigger_code,
                found=True,
                source="bigquery",
                instruction=instruction,
                error=None
            )
        
        # Try CSV
        instruction = await self.lookup_from_csv(trigger_code)
        if instruction:
            return PolicyLookupResult(
                trigger_code=trigger_code,
                found=True,
                source="csv_file",
                instruction=instruction,
                error=None
            )
        
        # Not found in any source
        return PolicyLookupResult(
            trigger_code=trigger_code,
            found=False,
            source="none",
            instruction=None,
            error=f"Trigger code {trigger_code} not found in any data source"
        )

@dataclass
class AppContext:
    policy_manager: PolicyDataManager
    config: Dict[str, Any]

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle"""
    config = {
        "policy_json_path": "policies/trigger_codes.json",
        "policy_csv_path": "policies/trigger_codes.csv",
        "bigquery_enabled": False,  # Set to True in production
        "bigquery_project": "your-project-id",
        "bigquery_dataset": "claim_processing",
        "bigquery_table": "policy_instructions",
        "max_batch_size": 50
    }
    
    policy_manager = PolicyDataManager(config)
    await policy_manager.initialize()
    
    try:
        yield AppContext(policy_manager=policy_manager, config=config)
    finally:
        # Cleanup if needed
        pass

# Initialize FastMCP server
mcp = FastMCP(
    "Policy Database Server",
    dependencies=["pandas", "google-cloud-bigquery"],
    lifespan=app_lifespan
)

# Resources
@mcp.resource("policies://data-sources", title="Available Policy Data Sources")
def get_data_sources() -> str:
    """Get information about available policy data sources"""
    ctx = mcp.get_context()
    manager = ctx.lifespan_context.policy_manager
    
    sources = []
    
    if manager.json_data:
        sources.append(f"✅ JSON File: {len(manager.json_data)} policies loaded")
    else:
        sources.append("❌ JSON File: Not available")
    
    if manager.bigquery_client:
        sources.append("✅ BigQuery: Connected")
    else:
        sources.append("❌ BigQuery: Not available")
    
    if manager.csv_data is not None:
        sources.append(f"✅ CSV File: {len(manager.csv_data)} policies loaded")
    else:
        sources.append("❌ CSV File: Not available")
    
    return "POLICY DATA SOURCES STATUS:\n" + "\n".join(sources)

@mcp.resource("policies://sample-codes", title="Sample Trigger Codes")
def get_sample_codes() -> str:
    """Get sample trigger codes for testing"""
    ctx = mcp.get_context()
    manager = ctx.lifespan_context.policy_manager
    
    sample_codes = []
    
    if manager.json_data:
        sample_codes.extend(list(manager.json_data.keys())[:10])
    
    if sample_codes:
        return f"Sample Trigger Codes Available:\n" + "\n".join(f"- {code}" for code in sample_codes)
    else:
        return "No sample codes available. Load policy data first."

@mcp.resource("policies://statistics", title="Policy Database Statistics")
def get_policy_statistics() -> str:
    """Get statistics about the policy database"""
    ctx = mcp.get_context()
    manager = ctx.lifespan_context.policy_manager
    
    stats = {
        "json_policies": len(manager.json_data) if manager.json_data else 0,
        "csv_policies": len(manager.csv_data) if manager.csv_data is not None else 0,
        "bigquery_status": "Connected" if manager.bigquery_client else "Disconnected"
    }
    
    return f"""POLICY DATABASE STATISTICS:
JSON Policies: {stats['json_policies']}
CSV Policies: {stats['csv_policies']}
BigQuery Status: {stats['bigquery_status']}

Total Available Sources: {sum(1 for v in [manager.json_data, manager.csv_data, manager.bigquery_client] if v)}
"""

# Tools
@mcp.tool()
async def lookup_single_trigger_code(trigger_code: str) -> str:
    """
    Lookup instructions for a single trigger code
    
    Args:
        trigger_code: The trigger code to lookup (e.g., TRG-001, CLM-456)
    """
    ctx = mcp.get_context()
    manager = ctx.lifespan_context.policy_manager
    
    result = await manager.lookup_single_with_fallback(trigger_code)
    
    if not result.found:
        return f"❌ Trigger code '{trigger_code}' not found in any data source"
    
    instruction = result.instruction
    
    output = f"""✅ POLICY INSTRUCTIONS FOR {trigger_code}
Source: {result.source.upper()}

TITLE: {instruction.title}
PRIORITY: {instruction.priority.upper()}

PROCESSING STEPS:
"""
    
    for i, step in enumerate(instruction.steps, 1):
        if isinstance(step, dict):
            for step_key, step_value in step.items():
                if isinstance(step_value, dict):
                    output += f"{i}. {step_value.get('action', str(step_value))}\n"
                else:
                    output += f"{i}. {step_value}\n"
        else:
            output += f"{i}. {step}\n"
    
    if instruction.required_documents:
        output += f"\nREQUIRED DOCUMENTS:\n"
        for doc in instruction.required_documents:
            output += f"- {doc}\n"
    
    if instruction.deadline:
        output += f"\nDEADLINE: {instruction.deadline}\n"
    
    if instruction.escalation_rules:
        output += f"\nESCALATION RULES:\n"
        for rule, details in instruction.escalation_rules.items():
            output += f"- {rule}: {details}\n"
    
    output += f"\nLast Updated: {instruction.last_updated}"
    
    return output

@mcp.tool()
async def batch_lookup_trigger_codes(trigger_codes: List[str]) -> str:
    """
    Lookup instructions for multiple trigger codes at once
    
    Args:
        trigger_codes: List of trigger codes to lookup
    """
    ctx = mcp.get_context()
    manager = ctx.lifespan_context.policy_manager
    config = ctx.lifespan_context.config
    
    if len(trigger_codes) > config["max_batch_size"]:
        return f"❌ Maximum {config['max_batch_size']} codes allowed per batch. Received {len(trigger_codes)} codes."
    
    results = await manager.batch_lookup(trigger_codes)
    
    # Summary statistics
    found_count = sum(1 for r in results if r.found)
    not_found = [r.trigger_code for r in results if not r.found]
    
    output = f"""BATCH LOOKUP RESULTS
==================
Total Codes: {len(trigger_codes)}
Found: {found_count}
Not Found: {len(not_found)}

"""
    
    # List found policies with brief details
    for result in results:
        if result.found:
            instruction = result.instruction
            output += f"✅ {result.trigger_code} ({result.source})\n"
            output += f"   Title: {instruction.title}\n"
            output += f"   Priority: {instruction.priority}\n"
            output += f"   Steps: {len(instruction.steps)}\n\n"
        else:
            output += f"❌ {result.trigger_code} - Not found\n\n"
    
    if not_found:
        output += f"CODES NOT FOUND:\n"
        for code in not_found:
            output += f"- {code}\n"
    
    return output

@mcp.tool()
async def search_policies_by_priority(priority: str) -> str:
    """
    Search for policies by priority level
    
    Args:
        priority: Priority level (high, medium, low)
    """
    ctx = mcp.get_context()
    manager = ctx.lifespan_context.policy_manager
    
    priority = priority.lower()
    matching_codes = []
    
    # Search in JSON data
    if manager.json_data:
        for code, policy in manager.json_data.items():
            if policy.get("priority", "medium").lower() == priority:
                matching_codes.append({
                    "code": code,
                    "title": policy.get("title", ""),
                    "source": "json"
                })
    
    # Search in CSV data
    if manager.csv_data is not None:
        try:
            priority_matches = manager.csv_data[manager.csv_data['priority'].str.lower() == priority]
            for _, row in priority_matches.iterrows():
                matching_codes.append({
                    "code": row['trigger_code'],
                    "title": row.get('title', ''),
                    "source": "csv"
                })
        except Exception as e:
            pass  # Continue if CSV doesn't have priority column
    
    if not matching_codes:
        return f"No policies found with priority level: {priority}"
    
    output = f"POLICIES WITH {priority.upper()} PRIORITY:\n"
    output += "="*40 + "\n\n"
    
    for policy in matching_codes:
        output += f"• {policy['code']} ({policy['source']})\n"
        output += f"  {policy['title']}\n\n"
    
    return output

@mcp.tool()
async def get_policy_summary(trigger_code: str) -> str:
    """
    Get a brief summary of a policy without full details
    
    Args:
        trigger_code: The trigger code to summarize
    """
    ctx = mcp.get_context()
    manager = ctx.lifespan_context.policy_manager
    
    result = await manager.lookup_single_with_fallback(trigger_code)
    
    if not result.found:
        return f"❌ Trigger code '{trigger_code}' not found"
    
    instruction = result.instruction
    
    return f"""POLICY SUMMARY: {trigger_code}
Title: {instruction.title}
Priority: {instruction.priority}
Steps: {len(instruction.steps)} processing steps
Documents Required: {len(instruction.required_documents)}
Deadline: {instruction.deadline or 'Not specified'}
Source: {result.source}"""

@mcp.tool()
async def validate_trigger_codes(trigger_codes: List[str]) -> str:
    """
    Validate if trigger codes exist in the policy database
    
    Args:
        trigger_codes: List of codes to validate
    """
    ctx = mcp.get_context()
    manager = ctx.lifespan_context.policy_manager
    
    results = await manager.batch_lookup(trigger_codes)
    
    valid_codes = [r.trigger_code for r in results if r.found]
    invalid_codes = [r.trigger_code for r in results if not r.found]
    
    output = f"""TRIGGER CODE VALIDATION RESULTS
===============================
Total Codes Checked: {len(trigger_codes)}
Valid Codes: {len(valid_codes)}
Invalid Codes: {len(invalid_codes)}

"""
    
    if valid_codes:
        output += "✅ VALID CODES:\n"
        for code in valid_codes:
            output += f"- {code}\n"
        output += "\n"
    
    if invalid_codes:
        output += "❌ INVALID CODES:\n"
        for code in invalid_codes:
            output += f"- {code}\n"
    
    return output

# Prompts for LangGraph agents
@mcp.prompt("policy-lookup-strategy")
def policy_lookup_strategy() -> str:
    """Strategy for efficient policy lookups"""
    return """
POLICY LOOKUP STRATEGY FOR CLAIM PROCESSING

LOOKUP PRIORITY ORDER:
1. JSON File - Fastest, most structured data
2. BigQuery - Enterprise database, real-time updates
3. CSV File - Backup/legacy data source

BATCH PROCESSING GUIDELINES:
- Use batch_lookup for multiple codes (more efficient)
- Maximum 50 codes per batch to prevent timeouts
- Validate codes first for large sets

PRIORITY-BASED PROCESSING:
- High priority codes: Process immediately
- Medium priority: Standard workflow
- Low priority: Can be batched/deferred

ERROR HANDLING:
- Always check if code exists before processing
- Have fallback procedures for missing policies
- Log failed lookups for policy team review

OPTIMIZATION TIPS:
- Cache frequently accessed policies
- Pre-validate trigger codes from emails
- Use summary view for quick decisions
"""

@mcp.prompt("policy-instruction-formatting")
def policy_instruction_formatting() -> str:
    """Guide for formatting policy instructions for claim processors"""
    return """
POLICY INSTRUCTION FORMATTING FOR CLAIM PROCESSORS

CLEAR PRESENTATION STRUCTURE:
1. Trigger Code & Title (identification)
2. Priority Level (urgency indicator)
3. Step-by-step Instructions (actionable items)
4. Required Documents (compliance checklist)
5. Deadlines (time constraints)
6. Escalation Rules (when to escalate)

INSTRUCTION CLARITY:
- Use action verbs (Review, Verify, Process, Escalate)
- Include specific timeframes
- Reference exact document names/forms
- Provide clear success criteria

PRIORITY INDICATORS:
- HIGH: Immediate action required (within hours)
- MEDIUM: Standard processing (within days)
- LOW: Routine processing (within weeks)

FORMATTING BEST PRACTICES:
- Number steps clearly (1, 2, 3...)
- Use bullet points for document lists
- Highlight deadlines and urgent items
- Include source information for verification
"""

if __name__ == "__main__":
    mcp.run()