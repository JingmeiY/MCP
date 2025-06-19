#!/usr/bin/env python3
"""
Pattern 2: How Specialized Agents Connect, Share State, and Orchestrate
Shows 3 different orchestration approaches
"""
import asyncio
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

from mcp import ClientSession
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

# Shared State Object
@dataclass
class ClaimProcessingState:
    """Shared state that flows between agents"""
    # Input
    input_text: str = ""
    input_type: str = ""  # "text", "email", "pdf"
    
    # Document Agent Results
    extracted_codes: List[str] = field(default_factory=list)
    document_quality: str = ""
    codes_with_context: str = ""
    
    # Policy Agent Results  
    policy_details: Dict[str, str] = field(default_factory=dict)
    priority_codes: List[str] = field(default_factory=list)
    deadlines: Dict[str, str] = field(default_factory=dict)
    
    # Email Agent Results
    related_emails: List[str] = field(default_factory=list)
    urgent_emails: List[str] = field(default_factory=list)
    
    # Final Output
    action_plan: str = ""
    processing_complete: bool = False
    
    # Agent Communication
    agent_messages: List[Dict[str, Any]] = field(default_factory=list)
    current_step: str = "input_processing"
    errors: List[str] = field(default_factory=list)

class ProcessingStep(Enum):
    INPUT_PROCESSING = "input_processing"
    DOCUMENT_ANALYSIS = "document_analysis"  
    POLICY_LOOKUP = "policy_lookup"
    EMAIL_SEARCH = "email_search"
    ACTION_PLANNING = "action_planning"
    COMPLETE = "complete"

# Simple MCP Servers (same as before but abbreviated)
POLICY_SERVER = '''
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("Policy Server")

POLICIES = {
    "TRG-001": {"title": "High Priority", "priority": "high", "deadline": "2 hours"},
    "CLM-456": {"title": "Standard Processing", "priority": "medium", "deadline": "5 days"}
}

@mcp.tool()
def lookup_policy(code: str) -> str:
    if code in POLICIES:
        p = POLICIES[code]
        return f"{code}: {p['title']} (Priority: {p['priority']}, Deadline: {p['deadline']})"
    return f"{code}: Not found"

@mcp.tool()
def get_high_priority_codes() -> str:
    high_codes = [c for c, p in POLICIES.items() if p['priority'] == 'high']
    return f"High priority: {', '.join(high_codes)}"

if __name__ == "__main__": mcp.run(transport="stdio")
'''

DOCUMENT_SERVER = '''
from mcp.server.fastmcp import FastMCP
import re
mcp = FastMCP("Document Server")

@mcp.tool()
def extract_trigger_codes(text: str) -> str:
    codes = re.findall(r'\\b[A-Z]{3}-\\d{3}\\b', text)
    return ', '.join(set(codes)) if codes else "No codes found"

@mcp.tool() 
def assess_quality(text: str) -> str:
    words = len(text.split())
    if words < 10: return "POOR"
    elif words < 50: return "FAIR" 
    else: return "GOOD"

if __name__ == "__main__": mcp.run(transport="stdio")
'''

# =============================================================================
# APPROACH 1: Sequential Orchestration (Simple)
# =============================================================================

class SequentialOrchestrator:
    """Simple sequential processing - one agent after another"""
    
    def __init__(self):
        self.document_agent = None
        self.policy_agent = None
        self.state = ClaimProcessingState()
    
    async def setup_agents(self):
        """Initialize all agents"""
        # Write server files
        Path("policy_server.py").write_text(POLICY_SERVER)
        Path("document_server.py").write_text(DOCUMENT_SERVER)
        
        # Setup Document Agent
        async with stdio_client("python", "document_server.py") as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                doc_tools = await load_mcp_tools(session)
                self.document_agent = create_react_agent("openai:gpt-4o-mini", doc_tools)
        
        # Setup Policy Agent
        async with stdio_client("python", "policy_server.py") as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                policy_tools = await load_mcp_tools(session)
                self.policy_agent = create_react_agent("openai:gpt-4o-mini", policy_tools)
        
        print("✅ Sequential orchestrator ready")
    
    async def process_claim(self, input_text: str) -> ClaimProcessingState:
        """Process claim sequentially: Document → Policy → Plan"""
        
        self.state.input_text = input_text
        self.state.current_step = "document_analysis"
        
        # Step 1: Document Agent extracts codes
        print("🔍 Step 1: Document Analysis")
        doc_response = await self.document_agent.ainvoke({
            "messages": [HumanMessage(content=f"Extract trigger codes and assess quality: {input_text}")]
        })
        
        # Parse document agent response
        doc_result = doc_response["messages"][-1].content
        self.state.agent_messages.append({"agent": "document", "response": doc_result})
        
        # Extract codes from response (simplified parsing)
        import re
        codes = re.findall(r'[A-Z]{3}-\d{3}', doc_result)
        self.state.extracted_codes = codes
        self.state.current_step = "policy_lookup"
        
        # Step 2: Policy Agent looks up policies
        print(f"📋 Step 2: Policy Lookup for codes: {codes}")
        if codes:
            policy_prompt = f"Look up policies for these trigger codes: {', '.join(codes)}"
            policy_response = await self.policy_agent.ainvoke({
                "messages": [HumanMessage(content=policy_prompt)]
            })
            
            policy_result = policy_response["messages"][-1].content
            self.state.agent_messages.append({"agent": "policy", "response": policy_result})
        
        # Step 3: Generate final plan
        self.state.current_step = "action_planning"
        self.state.action_plan = self._generate_action_plan()
        self.state.processing_complete = True
        
        return self.state
    
    def _generate_action_plan(self) -> str:
        """Combine results into action plan"""
        plan = "CLAIM PROCESSING ACTION PLAN\\n" + "="*30 + "\\n\\n"
        
        if self.state.extracted_codes:
            plan += f"TRIGGER CODES FOUND: {', '.join(self.state.extracted_codes)}\\n\\n"
            
        plan += "AGENT ANALYSIS:\\n"
        for msg in self.state.agent_messages:
            plan += f"- {msg['agent'].title()}: {msg['response'][:100]}...\\n"
        
        plan += "\\nNEXT STEPS:\\n"
        plan += "1. Review agent recommendations above\\n"
        plan += "2. Process according to trigger code priorities\\n"
        plan += "3. Follow up within specified deadlines\\n"
        
        return plan

# =============================================================================
# APPROACH 2: LangGraph State Management (Recommended)
# =============================================================================

class LangGraphOrchestrator:
    """LangGraph-based orchestration with proper state flow"""
    
    def __init__(self):
        self.document_agent = None
        self.policy_agent = None
        self.workflow = None
    
    async def setup_agents(self):
        """Setup agents and workflow"""
        Path("policy_server.py").write_text(POLICY_SERVER)
        Path("document_server.py").write_text(DOCUMENT_SERVER)
        
        # Setup agents (same as before)
        async with stdio_client("python", "document_server.py") as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                doc_tools = await load_mcp_tools(session)
                self.document_agent = create_react_agent("openai:gpt-4o-mini", doc_tools)
        
        async with stdio_client("python", "policy_server.py") as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                policy_tools = await load_mcp_tools(session)
                self.policy_agent = create_react_agent("openai:gpt-4o-mini", policy_tools)
        
        # Create LangGraph workflow
        self.workflow = self._create_workflow()
        print("✅ LangGraph orchestrator ready")
    
    def _create_workflow(self) -> StateGraph:
        """Create LangGraph workflow with state management"""
        
        workflow = StateGraph(ClaimProcessingState)
        
        # Add nodes (each node modifies the shared state)
        workflow.add_node("document_analysis", self._document_node)
        workflow.add_node("policy_lookup", self._policy_node)
        workflow.add_node("action_planning", self._planning_node)
        
        # Define flow
        workflow.set_entry_point("document_analysis")
        workflow.add_edge("document_analysis", "policy_lookup")
        workflow.add_edge("policy_lookup", "action_planning")
        workflow.add_edge("action_planning", END)
        
        return workflow.compile()
    
    async def _document_node(self, state: ClaimProcessingState) -> ClaimProcessingState:
        """Document analysis node - modifies state"""
        print("🔍 LangGraph: Document Analysis Node")
        
        response = await self.document_agent.ainvoke({
            "messages": [HumanMessage(content=f"Extract codes and assess quality: {state.input_text}")]
        })
        
        result = response["messages"][-1].content
        state.agent_messages.append({"agent": "document", "response": result})
        
        # Extract codes
        import re
        codes = re.findall(r'[A-Z]{3}-\d{3}', result)
        state.extracted_codes = codes
        state.current_step = "policy_lookup"
        
        return state
    
    async def _policy_node(self, state: ClaimProcessingState) -> ClaimProcessingState:
        """Policy lookup node - modifies state"""
        print(f"📋 LangGraph: Policy Lookup Node for {state.extracted_codes}")
        
        if state.extracted_codes:
            codes_str = ', '.join(state.extracted_codes)
            response = await self.policy_agent.ainvoke({
                "messages": [HumanMessage(content=f"Look up policies for: {codes_str}")]
            })
            
            result = response["messages"][-1].content
            state.agent_messages.append({"agent": "policy", "response": result})
        
        state.current_step = "action_planning"
        return state
    
    async def _planning_node(self, state: ClaimProcessingState) -> ClaimProcessingState:
        """Final planning node"""
        print("📝 LangGraph: Action Planning Node")
        
        # Generate comprehensive plan
        plan = "LANGGRAPH CLAIM PROCESSING PLAN\\n" + "="*35 + "\\n\\n"
        
        if state.extracted_codes:
            plan += f"CODES EXTRACTED: {', '.join(state.extracted_codes)}\\n\\n"
        
        plan += "WORKFLOW RESULTS:\\n"
        for msg in state.agent_messages:
            plan += f"📍 {msg['agent'].title()} Agent:\\n   {msg['response'][:150]}...\\n\\n"
        
        plan += "STATUS: Processing complete via LangGraph workflow"
        
        state.action_plan = plan
        state.processing_complete = True
        state.current_step = "complete"
        
        return state
    
    async def process_claim(self, input_text: str) -> ClaimProcessingState:
        """Process claim using LangGraph workflow"""
        
        initial_state = ClaimProcessingState(
            input_text=input_text,
            current_step="document_analysis"
        )
        
        # Run the workflow
        final_state = await self.workflow.ainvoke(initial_state)
        return final_state

# =============================================================================
# APPROACH 3: Event-Driven Coordination (Advanced)
# =============================================================================

class EventDrivenOrchestrator:
    """Event-driven coordination between agents"""
    
    def __init__(self):
        self.document_agent = None
        self.policy_agent = None
        self.state = ClaimProcessingState()
        self.event_queue = asyncio.Queue()
    
    async def setup_agents(self):
        """Setup agents"""
        Path("policy_server.py").write_text(POLICY_SERVER)
        Path("document_server.py").write_text(DOCUMENT_SERVER)
        
        # Setup agents (same pattern)
        async with stdio_client("python", "document_server.py") as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                doc_tools = await load_mcp_tools(session)
                self.document_agent = create_react_agent("openai:gpt-4o-mini", doc_tools)
        
        async with stdio_client("python", "policy_server.py") as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                policy_tools = await load_mcp_tools(session)
                self.policy_agent = create_react_agent("openai:gpt-4o-mini", policy_tools)
        
        print("✅ Event-driven orchestrator ready")
    
    async def process_claim(self, input_text: str) -> ClaimProcessingState:
        """Process using event-driven coordination"""
        
        self.state.input_text = input_text
        
        # Start event processing
        await self.event_queue.put({"type": "start_processing", "data": input_text})
        
        # Process events until completion
        while not self.state.processing_complete:
            event = await self.event_queue.get()
            await self._handle_event(event)
        
        return self.state
    
    async def _handle_event(self, event: Dict[str, Any]):
        """Handle different types of events"""
        
        if event["type"] == "start_processing":
            print("🎬 Event: Starting processing")
            await self.event_queue.put({"type": "analyze_document", "data": event["data"]})
        
        elif event["type"] == "analyze_document":
            print("🔍 Event: Analyzing document")
            response = await self.document_agent.ainvoke({
                "messages": [HumanMessage(content=f"Extract codes: {event['data']}")]
            })
            
            result = response["messages"][-1].content
            self.state.agent_messages.append({"agent": "document", "response": result})
            
            # Extract codes and trigger next event
            import re
            codes = re.findall(r'[A-Z]{3}-\d{3}', result)
            self.state.extracted_codes = codes
            
            await self.event_queue.put({"type": "lookup_policies", "data": codes})
        
        elif event["type"] == "lookup_policies":
            print(f"📋 Event: Looking up policies for {event['data']}")
            
            if event["data"]:
                codes_str = ', '.join(event["data"])
                response = await self.policy_agent.ainvoke({
                    "messages": [HumanMessage(content=f"Look up: {codes_str}")]
                })
                
                result = response["messages"][-1].content
                self.state.agent_messages.append({"agent": "policy", "response": result})
            
            await self.event_queue.put({"type": "finalize_plan", "data": None})
        
        elif event["type"] == "finalize_plan":
            print("📝 Event: Finalizing action plan")
            
            plan = "EVENT-DRIVEN PROCESSING COMPLETE\\n" + "="*35 + "\\n\\n"
            plan += f"CODES: {', '.join(self.state.extracted_codes)}\\n\\n"
            
            for msg in self.state.agent_messages:
                plan += f"🎯 {msg['agent'].title()}: {msg['response'][:100]}...\\n"
            
            self.state.action_plan = plan
            self.state.processing_complete = True

async def main():
    """Test all three orchestration approaches"""
    print("🎭 Testing Agent Orchestration Patterns")
    print("="*50)
    
    test_input = """
    URGENT: Process claim immediately
    Trigger codes: TRG-001, CLM-456
    High priority review required
    """
    
    try:
        # Test Approach 1: Sequential
        print("\\n🥇 APPROACH 1: Sequential Orchestration")
        print("-"*40)
        seq_orch = SequentialOrchestrator()
        await seq_orch.setup_agents()
        seq_result = await seq_orch.process_claim(test_input)
        print("Result:", seq_result.action_plan[:200] + "...")
        
        # Test Approach 2: LangGraph (Recommended)
        print("\\n🥇 APPROACH 2: LangGraph State Management")
        print("-"*40)
        lg_orch = LangGraphOrchestrator()
        await lg_orch.setup_agents()
        lg_result = await lg_orch.process_claim(test_input)
        print("Result:", lg_result.action_plan[:200] + "...")
        
        # Test Approach 3: Event-Driven
        print("\\n🥇 APPROACH 3: Event-Driven Coordination")
        print("-"*40)
        event_orch = EventDrivenOrchestrator()
        await event_orch.setup_agents()
        event_result = await event_orch.process_claim(test_input)
        print("Result:", event_result.action_plan[:200] + "...")
        
        print("\\n✅ All approaches completed!")
        print("\\n🎯 RECOMMENDATION: Use LangGraph (Approach 2)")
        print("   - Built-in state management")
        print("   - Visual workflow representation") 
        print("   - Easy debugging and monitoring")
        print("   - Excellent error handling")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure: export OPENAI_API_KEY=your_key")
    
    finally:
        # Cleanup
        for file in ["policy_server.py", "document_server.py"]:
            if Path(file).exists():
                Path(file).unlink()

if __name__ == "__main__":
    asyncio.run(main())