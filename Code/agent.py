import os
import json
from typing import TypedDict, Dict, Any, List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langgraph.graph import StateGraph, END
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache

# ---------------------------
# STEP 1: Enable Cache for Speed
# ---------------------------
set_llm_cache(InMemoryCache())

# ---------------------------
# STEP 2: API Key & Models
# ---------------------------
os.environ["OPENAI_API_KEY"] = ""


# Fast model for intent detection
llm_fast = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
# High-quality model for final answer
llm_reasoning = ChatOpenAI(model="gpt-4", temperature=0)

embedding_model = OpenAIEmbeddings()

#


# STEP 3: Load and Chunk PDFs
# ---------------------------
data_folder = "data"
docs = []
for file in os.listdir(data_folder):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(data_folder, file))
        docs.extend(loader.load())

# Split documents into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(docs)

# ---------------------------
# STEP 4: Vector Store for Retrieval
# ---------------------------
vectorstore = FAISS.from_documents(chunks, embedding_model)

# ---------------------------
# STEP 5: Define State for LangGraph
# ---------------------------
class GraphState(TypedDict):
    query: str
    sub_queries: List[str]
    intent_info: Dict
    context: str
    answer: str


# ---------------------------
# STEP 6: Intent Classification Node
# ---------------------------
intent_prompt = PromptTemplate(
    input_variables=["query"],
    template="""
Classify the user query into JSON:
- intent: [overview | competitor_analysis | sales_saas | sales_fmcg | multi_doc_analysis]
- reasoning: short reason
- sectors: [SaaS, FMCG, Both]
- quarters: [Q1, Q2, Both, NA]

Query: "{query}"
Output only JSON.
"""
)
intent_chain = intent_prompt | llm_fast | StrOutputParser()

 #---------------------------
# STEP 7: Multi-hop Query Decomposition
# ---------------------------
decompose_prompt = PromptTemplate(
    input_variables=["query"],
    template="""
If the query is complex, break it into smaller sub-queries for analysis.
Return as JSON:
{{
  "sub_queries": ["question1", "question2", ...]
}}
Query: "{query}"
"""
)
decompose_chain = decompose_prompt | llm_fast | StrOutputParser()

# ---------------------------
# STEP 8: Weighted Retrieval
# ---------------------------
def weighted_retrieval(query: str, intent_info: Dict) -> str:
    results = vectorstore.similarity_search_with_score(query, k=12)
    boosted = []
    for doc, score in results:
        meta_path = doc.metadata.get("source", "").lower()
        if intent_info.get("sectors") == "SaaS" and "saas" in meta_path:
            score *= 0.7
        if intent_info.get("sectors") == "FMCG" and "fmcg" in meta_path:
            score *= 0.7
        if intent_info.get("quarters") == "Q1" and "q1" in meta_path:
            score *= 0.8
        if intent_info.get("quarters") == "Q2" and "q2" in meta_path:
            score *= 0.8
        boosted.append((doc.page_content, score))
    boosted.sort(key=lambda x: x[1])
    return "\n\n".join([text for text, _ in boosted[:5]])

# ---------------------------
# STEP 9: Answer Generation Node
# ---------------------------
response_prompt = ChatPromptTemplate.from_template("""
You are a Strategic Business Analyst.
Answer using ONLY the given context.

USER QUERY: {query}
CONTEXT:
{context}

Rules:
- Do NOT invent facts.
- If info missing, say "Not found in documents."
- Use this format:

### EXECUTIVE SUMMARY
- 3-4 bullets

### DETAILED INSIGHTS
- Explain with supporting facts

### STRATEGIC IMPLICATIONS
- Recommendations
""")
response_chain = response_prompt | llm_reasoning | StrOutputParser()


# ---------------------------
# STEP 10: LangGraph Nodes
# ---------------------------
def classify_node(state: GraphState) -> GraphState:
    query = state["query"]
    raw_intent = intent_chain.invoke({"query": query})
    try:
        intent_info = json.loads(raw_intent)
    except:
        intent_info = {"intent": "multi_doc_analysis", "sectors": "Both", "quarters": "Both"}
    state["intent_info"] = intent_info
    return state

def decompose_node(state: GraphState) -> GraphState:
    query = state["query"]
    raw_decomp = decompose_chain.invoke({"query": query})
    try:
        sub_queries = json.loads(raw_decomp).get("sub_queries", [])
    except:
        sub_queries = []
    state["sub_queries"] = sub_queries if sub_queries else [query]
    return state

def retrieve_node(state: GraphState) -> GraphState:
    context = ""
    retrieved_docs = set()
    retrieved_chunks = []

    for q in state["sub_queries"]:
        results = vectorstore.similarity_search_with_score(q, k=6)
        boosted = []
        for doc, score in results:
            meta_path = doc.metadata.get("source", "")
            retrieved_docs.add(os.path.basename(meta_path))
            boosted.append((doc.page_content, score))
        boosted.sort(key=lambda x: x[1])
        top_chunks = [text for text, _ in boosted[:3]]
        retrieved_chunks.extend(top_chunks)
        context += "\n\n".join(top_chunks) + "\n\n"

    state["context"] = context
    state["retrieved_docs"] = list(retrieved_docs)
    state["retrieved_chunks"] = retrieved_chunks
    return state


def answer_node(state: GraphState) -> GraphState:
    query = state["query"]
    context = state["context"]
    answer = response_chain.invoke({"query": query, "context": context})
    state["answer"] = answer
    return state


# ---------------------------
# STEP 11: Build LangGraph Workflow
# ---------------------------
graph = StateGraph(GraphState)
graph.add_node("classify", classify_node)
graph.add_node("decompose", decompose_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("answer", answer_node)

graph.set_entry_point("classify")
graph.add_edge("classify", "decompose")
graph.add_edge("decompose", "retrieve")
graph.add_edge("retrieve", "answer")
graph.add_edge("answer", END)

app = graph.compile()


def run_agent(query: str):
    initial_state = {"query": query, "sub_queries": [], "intent_info": {}, "context": "", "answer": ""}
    result = app.invoke(initial_state)
    return {
        "answer": result["answer"],
        "intent_info": result.get("intent_info", {}),
        "sub_queries": result.get("sub_queries", []),
        "retrieved_docs": result.get("retrieved_docs", []),
        "retrieved_chunks": result.get("retrieved_chunks", [])
    }

