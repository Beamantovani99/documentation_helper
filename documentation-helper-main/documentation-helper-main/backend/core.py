from typing import Any, Dict

from dotenv import load_dotenv
from langchain.tools import tool
from langchain.messages import ToolMessage

from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

# -----------------------
# EMBEDDINGS (NO OPENAI)
# -----------------------
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# -----------------------
# VECTOR STORE
# -----------------------
vectorstore = PineconeVectorStore(
    index_name="langchain-doc-index",
    embedding=embeddings
)

# -----------------------
# LLM (GROQ)
# -----------------------
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=None  # oppure os.environ["GROQ_API_KEY"]
)

# -----------------------
# TOOL RETRIEVAL
# -----------------------
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant documentation to help answer user queries about LangChain."""

    retrieved_docs = vectorstore.as_retriever().invoke(query)

    serialized = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )

    return serialized, retrieved_docs


# -----------------------
# MAIN RAG FUNCTION
# -----------------------
def run_llm(query: str) -> Dict[str, Any]:

    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "Use the retrieval tool before answering. "
        "Always cite sources. "
        "If you don't know, say so."
    )

    # NOTE: keep agent abstraction (Groq works fine here)
    from langchain.agents import create_agent

    agent = create_agent(
        model,
        tools=[retrieve_context],
        system_prompt=system_prompt
    )

    messages = [{"role": "user", "content": query}]

    response = agent.invoke({"messages": messages})

    answer = response["messages"][-1].content

    context_docs = []
    for message in response["messages"]:
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)

    return {
        "answer": answer,
        "context": context_docs
    }


if __name__ == "__main__":
    result = run_llm("what are deep agents?")
    print(result)