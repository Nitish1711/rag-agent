from app.retriever import create_vectorstore


if __name__ == "__main__":

    print("Starting document ingestion...")

    vectorstore = create_vectorstore()

    print("Document ingestion completed.")
    print("Vector database created successfully.")