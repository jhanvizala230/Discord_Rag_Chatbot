# Agent Architecture - Modular Design

## Overview
The agent system follows a modular architecture with clean separation of concerns, making it easy to add new agents and routing logic.

## File Structure

```
backend_langchain/agents/
├── __init__.py              # Module exports and singleton orchestrator
├── doc_agent.py             # Document Q&A agent (RAG workflow)
├── routing_agent.py         # Query classification (placeholder)
├── orchestrator.py          # Agent coordination and routing
└── agents_backup.py         # Old corrupted file (safe to delete)
```

## Components

### 1. DocAgent (`doc_agent.py`)
**Purpose**: Document Q&A using retrieval-augmented generation (RAG)

**Workflow**:
1. **Query Refinement Chain**: Uses `query_refinement_prompt.txt` to generate 2-3 optimized search queries
2. **Document Retrieval**: Embeds queries, retrieves from ChromaDB, deduplicates, reranks
3. **Answer Synthesis Chain**: Uses `answer_synthesis_prompt.txt` to generate final answer from context
4. **Memory**: Saves conversation to SQLite using ConversationBufferWindowMemory

**LangChain Components**:
- `ChatPromptTemplate`: System + MessagesPlaceholder + Human messages
- `RunnablePassthrough.assign()`: Loads chat history into chain context
- `OllamaLLM`: LLM interface (qwen2.5:7b-instruct-q4_K_M)
- `StrOutputParser`: Parses LLM string output

**Key Methods**:
- `invoke({"input": query})`: Main entry point
- `_build_query_refinement_chain()`: Creates LangChain chain for query refinement
- `_build_answer_synthesis_chain()`: Creates LangChain chain for answer synthesis
- `_retrieve_documents(queries)`: Retrieves and reranks documents

### 2. RoutingAgent (`routing_agent.py`)
**Purpose**: Classify user queries to determine appropriate agent

**Current**: Placeholder implementation (always returns "doc")

**Future**:
- LLM-based classification
- Route to doc/advisor/other agents based on query intent
- Consider keywords, conversation history, user preferences

**Key Methods**:
- `decide(query, session_id)`: Returns agent type ("doc", "advisor", "other")

### 3. AgentOrchestrator (`orchestrator.py`)
**Purpose**: Coordinate agent routing and execution

**Configuration**:
- `use_routing=False` (default): Always use doc agent
- `use_routing=True`: Enable RoutingAgent classification

**Workflow**:
1. Route query to appropriate agent type
2. Get or create agent instance (cached per session_id)
3. Execute agent and return result

**Key Methods**:
- `run(query, session_id)`: Main entry point, returns (agent_type, answer)
- `_route(query, session_id)`: Determine which agent to use
- `_get_doc_agent(session_id)`: Get or create DocAgent instance

### 4. Module Interface (`__init__.py`)
**Purpose**: Provide clean API for other modules

**Exports**:
- `get_agent_orchestrator()`: Returns singleton AgentOrchestrator instance
- `AgentOrchestrator`: Class for direct instantiation

## Prompt Templates

### query_refinement_prompt.txt
**Variables**: None (query passed via ChatPromptTemplate human message)
**Input**: User query from conversation
**Output**: Numbered list of 2-3 refined search queries

### answer_synthesis_prompt.txt
**Variables**: None (context and query passed via ChatPromptTemplate human message)
**Input**: Retrieved context + user query
**Output**: Natural language answer based on context

## Memory System

- **Storage**: SQLite database (`conversation_history` table)
- **Type**: ConversationBufferWindowMemory (k=6, stores 12 messages)
- **Per Session**: Each session_id has independent conversation history
- **Integration**: Loaded via `RunnablePassthrough.assign()` in chains

## Adding New Agents

### Step 1: Create Agent File
```python
# agents/advisor_agent.py
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaLLM

class AdvisorAgent:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.memory = get_agent_memory(session_id)
        self.llm = OllamaLLM(...)
        self.chain = self._build_chain()
    
    def invoke(self, inputs):
        result = self.chain.invoke(inputs)
        self.memory.save_context(inputs, {"output": result})
        return {"output": result}
```

### Step 2: Update RoutingAgent
```python
# routing_agent.py
def decide(self, query: str, session_id: str) -> AgentType:
    # Add classification logic
    if "document" in query.lower():
        return "doc"
    elif "advice" in query.lower():
        return "advisor"
    # Or use LLM classification
```

### Step 3: Update Orchestrator
```python
# orchestrator.py
def __init__(self):
    self._doc_agents = {}
    self._advisor_agents = {}  # Add new cache
    
def _get_advisor_agent(self, session_id: str):
    if session_id not in self._advisor_agents:
        self._advisor_agents[session_id] = AdvisorAgent(session_id)
    return self._advisor_agents[session_id]

def run(self, query: str, session_id: str):
    route = self._route(query, session_id)
    if route == "doc":
        agent = self._get_doc_agent(session_id)
    elif route == "advisor":
        agent = self._get_advisor_agent(session_id)  # Add routing
    response = agent.invoke({"input": query})
    return route, response.get("output")
```

## Configuration

### Environment Variables (`.env`)
- `OLLAMA_BASE_URL`: LLM endpoint
- `LLM_MODEL`: Model name (qwen2.5:7b-instruct-q4_K_M)
- `LLM_MAX_TOKENS`: Max output length
- `EMBEDDING_MODEL`: Embedding model
- `RERANKER_MODEL`: Reranker model

### Agent Parameters
- `session_id`: Conversation identifier (required)
- `window_size`: Conversation memory window (default: 6 = 12 messages)
- `top_k`: Number of chunks to retrieve (default: 15)

## API Integration

The orchestrator is used via the `/routed_query` endpoint:

```python
# api/agentic.py
from backend_langchain.agents import get_agent_orchestrator

orchestrator = get_agent_orchestrator()

@router.post("/routed_query")
def routed_query(payload):
    agent_type, answer = orchestrator.run(
        query=payload.query,
        session_id=payload.session_id
    )
    return {"answer": answer, "metadata": {"agent_type": agent_type}}
```

## Key Design Principles

1. **Modularity**: Each agent in separate file
2. **Single Responsibility**: Each agent handles one task type
3. **Caching**: Agent instances cached per session_id
4. **LangChain Native**: Uses ChatPromptTemplate, RunnablePassthrough, chains
5. **No Tool Selection**: Fixed workflow (no ReAct complexity)
6. **Memory Integration**: ConversationBufferWindowMemory with SQLite
7. **Extensibility**: Easy to add new agents without modifying existing code

## Troubleshooting

### Import Errors (langchain_core)
- These are Pylance resolution issues
- Packages are installed in venv
- Code will run correctly despite IDE warnings
- Verify: `pip list | grep langchain`

### Memory Not Persisting
- Check SQLite database exists: `output/sql/conversation_history.db`
- Verify session_id consistency across requests
- Check ConversationBufferWindowMemory initialization

### Agent Not Routing
- Verify `use_routing=True` in orchestrator initialization
- Check RoutingAgent.decide() logic
- Enable debug logging: `logger.debug()`
