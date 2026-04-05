# Introduction to LLM Agents

## Overview

This project is designed for **Marquette University's AI Development and Value Creation (ADV) program**. It provides a comprehensive, hands-on introduction to building production-ready LLM (Large Language Model) agents, including Retrieval-Augmented Generation (RAG) systems, structured data agents, and multi-agent architectures.

### Learning Objectives

This project has three primary educational goals:

1. **RAG Pipeline Development**: Teach students how to build end-to-end RAG systems, including:
   - Text extraction from various document formats (PDF, Word, PowerPoint, plain text)
   - Text chunking strategies (character-based, sentence-based, semantic)
   - Embedding generation and vector database management
   - Context retrieval and response generation

2. **LLM Agent Architecture**: Teach students how to build different types of LLM agents, including:
   - RAG agents for querying document collections
   - Structured data agents for SQL and Excel queries
   - Multi-agent systems that route queries to specialized agents
   - Structured output generation using Pydantic models

3. **Production-Ready Agent Systems**: Teach students how to package agent systems for real-world applications, including:
   - Modular, reusable components following strategy patterns
   - Token counting and context window management
   - Error handling and confidence thresholds
   - Agent orchestration and routing

## Project Structure

```
intro_to_agents/
├── intro_to_agents/              # Main package
│   ├── agents/                    # Agent implementations
│   │   ├── agents.py             # RAG, SQL, Excel, and Multi-agent classes
│   │   ├── llms.py               # LLM interface and OpenAI implementation
│   │   └── token_counters.py     # Token counting utilities
│   ├── rag/                       # RAG pipeline components
│   │   ├── text_extractors.py    # Document text extraction
│   │   ├── chunkers.py           # Text chunking strategies
│   │   ├── embedders.py          # Text embedding generation
│   │   └── vector_databases.py   # Vector database interface and ChromaDB
│   ├── 1 - load_vectordb.ipynb   # Notebook 1: Building a vector database
│   ├── 2 - structured_outputs.ipynb # Notebook 2: Structured LLM outputs
│   ├── 3 - rag_agent.ipynb        # Notebook 3: RAG agent implementation
│   ├── 4 - sql_agent.ipynb       # Notebook 4: SQL agent implementation
│   └── 5 - composite_agent.ipynb # Notebook 5: Multi-agent system
├── data/                          # Data files
│   ├── text_corpus/              # Sample text documents
│   ├── supply_chain.csv          # Sample structured data
│   └── TGNSI_Trade.db            # Sample SQLite database
├── pyproject.toml                # Poetry dependencies
└── README.md                      # This file
```

## Workflow Overview

The project is organized into five sequential notebooks that guide students through building increasingly sophisticated agent systems:

### Notebook 1: Load Vector Database

**Purpose**: Build and populate a vector database for RAG

**Key Steps**:
1. **Text Extraction**
   - Extract text from various document formats (PDF, Word, PowerPoint, text files)
   - Use `TextExtractor` or `MarkitdownExtractor` for different file types

2. **Text Chunking**
   - Choose a chunking strategy:
     - `SplitCharChunker`: Split by character sequences (e.g., paragraphs)
     - `CharLenChunker`: Split by character length with overlap
     - `SentenceChunker`: Split into sentence windows
     - `SemanticChunker`: Use LLM for semantic chunking

3. **Embedding Generation**
   - Generate embeddings using `SentenceTransformerEmbedder`
   - Embeddings convert text chunks into vector representations

4. **Vector Database Creation**
   - Create a ChromaDB vector database
   - Store documents with embeddings
   - Create collections for organized document storage

### Notebook 2: Structured Outputs

**Purpose**: Learn to generate structured outputs from LLMs using Pydantic models

**Key Steps**:
1. **Define Structured Output Schema**
   - Create Pydantic `BaseModel` classes
   - Define fields with types and descriptions

2. **Generate Structured Responses**
   - Use `llm.structured_query()` to get structured outputs
   - LLM returns validated Pydantic models instead of free-form text

3. **Use Cases**
   - Data extraction from unstructured text
   - API response formatting
   - Consistent data structures for downstream processing

### Notebook 3: RAG Agent

**Purpose**: Build a RAG agent that queries document collections

**Key Steps**:
1. **Load Vector Database**
   - Connect to existing vector database
   - Load collection with embedded documents

2. **Create RAG Agent**
   - Initialize `ChromaAgent` with LLM and vector database
   - Agent combines retrieval and generation

3. **Query the Agent**
   - Ask questions about the document collection
   - Agent retrieves relevant context and generates answers
   - Configure retrieval parameters (k, max_distance, citations)

### Notebook 4: SQL Agent

**Purpose**: Build agents that query structured data (SQLite and Excel)

**Key Steps**:
1. **Database Connection**
   - Connect to SQLite database
   - Provide database schema and context to the agent

2. **Create SQL Agent**
   - Initialize `SQLiteAgent` with LLM and database connection
   - Agent generates SQL queries from natural language

3. **Query Structured Data**
   - Ask questions that require database queries
   - Agent generates SQL, executes it, and interprets results
   - View generated SQL for transparency

4. **Excel Agent** (optional)
   - Similar workflow for Excel files using `ExcelAgent`

### Notebook 5: Composite Agent

**Purpose**: Build a multi-agent system that routes queries to specialized agents

**Key Steps**:
1. **Initialize Multiple Agents**
   - Create RAG agent for document queries
   - Create SQL agent for database queries
   - Create additional specialized agents as needed

2. **Create Multi-Agent System**
   - Initialize `MultiAgent` with all agents and descriptions
   - System uses LLM to route queries to appropriate agent

3. **Query the System**
   - Ask questions that may require different agents
   - System automatically selects and uses the right agent
   - View routing logic for transparency

## Key Concepts

### Strategy Pattern

All components follow the strategy pattern with abstract base classes:
- `BaseTextExtractor`: Interface for text extraction
- `BaseChunker`: Interface for text chunking
- `BaseEmbedder`: Interface for embedding generation
- `BaseVectorDB`: Interface for vector databases
- `BaseLLM`: Interface for language models
- `BaseRAGAgent`: Interface for RAG agents
- `BaseStructuredDataAgent`: Interface for SQL/Excel agents

This ensures:
- **Modularity**: Easy to swap implementations
- **Extensibility**: Add new strategies without changing existing code
- **Testability**: Mock interfaces for testing

### RAG Pipeline

The RAG pipeline consists of four main stages:

1. **Extraction**: Convert documents to text
2. **Chunking**: Split text into manageable pieces
3. **Embedding**: Convert chunks to vector representations
4. **Storage**: Store embeddings in vector database

At query time:
1. **Query Embedding**: Convert query to vector
2. **Retrieval**: Find similar document chunks
3. **Context Assembly**: Combine retrieved chunks
4. **Generation**: Use LLM to generate answer from context

### Token Management

The system includes token counting and management:
- `OpenAITokenCounter`: Count tokens for OpenAI models
- `OPENAI_TOKEN_LIMITS`: Model-specific token limits
- Automatic validation to prevent exceeding context windows
- Safety margins for system prompts and metadata

### Agent Confidence

Agents include confidence mechanisms:
- Distance thresholds for retrieved documents
- Confidence scoring for generated responses
- Optional rejection of low-confidence responses

## Setup Instructions

### Prerequisites

- **Python 3.13+**
- **C++ Build Tools** (Required for some dependencies like `sentence-transformers` and `chromadb`):
  - Windows: Download and install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
  - During installation, select "C++ build tools" workload
  - This is required for compiling Python packages with C extensions
- **Jupyter Notebook** or **JupyterLab**
- **OpenAI API Key** (for LLM functionality)

### Installation

1. **Clone or download this repository**

2. **Install C++ Build Tools** (Windows only):
   - Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Run the installer
   - Select "C++ build tools" workload
   - Complete installation and restart your computer if prompted

3. **Install dependencies using Poetry**:
   ```
   poetry install
   ```

4. **Create a `.env` file** in the project root:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
   **Important**: Never commit your `.env` file to version control!

5. **Verify installation**:
   ```bash
   poetry run python -c "from intro_to_agents.agents.llms import OpenAILLM; print('Installation successful!')"
   ```

### Running the Notebooks

1. **Start Jupyter**:

2. **Execute notebooks in order**:
   - **First**: Run all cells in `1 - load_vectordb.ipynb`
   - **Second**: Run all cells in `2 - structured_outputs.ipynb`
   - **Third**: Run all cells in `3 - rag_agent.ipynb`
   - **Fourth**: Run all cells in `4 - sql_agent.ipynb`
   - **Fifth**: Run all cells in `5 - composite_agent.ipynb`

## Custom Modules

### `intro_to_agents.agents`

**LLM Interface**:
- `BaseLLM`: Abstract base class for LLM implementations
- `OpenAILLM`: OpenAI GPT model implementation
- `OPENAI_TOKEN_LIMITS`: Token limits for different models

**Agents**:
- `ChromaAgent`: RAG agent using ChromaDB vector database
- `SQLiteAgent`: Agent for querying SQLite databases
- `ExcelAgent`: Agent for querying Excel files
- `MultiAgent`: Multi-agent system with automatic routing

**Token Counting**:
- `OpenAITokenCounter`: Token counting for OpenAI models

### `intro_to_agents.rag`

**Text Extraction**:
- `TextExtractor`: Extract text from PDF, Word, PowerPoint, and text files
- `MarkitdownExtractor`: Extract text using MarkItDown (supports more formats)

**Chunking**:
- `SplitCharChunker`: Split by character sequences
- `CharLenChunker`: Split by character length with overlap
- `SentenceChunker`: Split into sentence windows
- `SemanticChunker`: Use LLM for semantic chunking

**Embedding**:
- `SentenceTransformerEmbedder`: Generate embeddings using sentence transformers

**Vector Databases**:
- `ChromaDBVectorDB`: ChromaDB implementation for vector storage and retrieval

## Best Practices Demonstrated

1. **Separation of Concerns**: Each component has a single, well-defined responsibility
2. **Interface-Based Design**: Abstract base classes enable flexibility and testing
3. **Error Handling**: Comprehensive error handling with informative messages
4. **Documentation**: Extensive docstrings and type hints
5. **Modularity**: Reusable components that can be combined in different ways
6. **Token Management**: Proper handling of context windows and token limits
7. **Confidence Thresholds**: Agents can reject low-confidence responses

## Notes for Students

- **Start with Notebook 1**: The vector database must be created before using RAG agents
- **Understand the Pipeline**: RAG requires extraction → chunking → embedding → storage
- **Experiment with Chunking**: Different chunking strategies work better for different document types
- **Monitor Token Usage**: Be aware of token limits and costs when using LLMs
- **Test Agent Routing**: In multi-agent systems, verify queries are routed correctly
- **Review Generated SQL**: Always review SQL queries before execution in production

## Troubleshooting

### C++ Build Tools Error

If you encounter errors about missing C++ compilers:
- Ensure C++ Build Tools are installed (see Prerequisites)
- Restart your terminal/IDE after installation
- On Windows, you may need to restart your computer

### Import Errors

If you get `ModuleNotFoundError: No module named 'intro_to_agents'`:
- Ensure you've run `poetry install` or `pip install -e .`
- Activate the virtual environment: `poetry shell`
- Verify the package is installed: `poetry run python -c "import intro_to_agents"`

### OpenAI API Errors

If you get API errors:
- Verify your API key is set in `.env` file
- Check your OpenAI account has available credits
- Ensure you're using a valid model name (check `OPENAI_TOKEN_LIMITS`)

## Credits

**Source Code Developer**: Prof Sandman  
**Course**: AI Development and Value Creation (ADV Program)  
**Institution**: Marquette University

---

## Additional Resources

- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Sentence Transformers Documentation](https://www.sbert.net/)
