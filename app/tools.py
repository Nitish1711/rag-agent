from langchain_core.tools import tool

from app.config import TOP_K
from app.retriever import get_vectorstore


vectorstore = get_vectorstore()


@tool
def retrieve_document(query: str) -> str:
    """
    Retrieve relevant information from the financial document.

    Use this tool when the user asks a question that requires
    information from the knowledge base.
    """

    try:

        documents = vectorstore.similarity_search(
            query,
            k=TOP_K
        )

        if not documents:
            return "No relevant information was found."

        results = []

        for index, document in enumerate(documents, start=1):

            page_number = document.metadata.get("page")

            if page_number is not None:
                page_number += 1

            source = document.metadata.get(
                "source",
                "Unknown source"
            )

            results.append(
                f"""
Document {index}
Source: {source}
Page: {page_number}

Content:
{document.page_content}
"""
            )

        return "\n".join(results)

    except Exception as exc:

        return (
            f"Retrieval failed. "
            f"Please try again. Error: {str(exc)}"
        )