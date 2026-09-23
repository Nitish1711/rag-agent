from langchain_core.messages import HumanMessage

from app.graph import rag_agent


def run_agent():

    print("=" * 60)
    print("Agentic RAG Financial Assistant")
    print("=" * 60)

    print("Type 'exit' to quit.\n")

    conversation = []

    while True:

        question = input("You: ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            continue

        conversation.append(
            HumanMessage(content=question)
        )

        try:

            result = rag_agent.invoke(
                {
                    "messages": conversation
                }
            )

            conversation = list(
                result["messages"]
            )

            answer = conversation[-1].content

            print("\nAssistant:")
            print(answer)
            print()

        except Exception as exc:

            print(
                f"\nError: {exc}\n"
            )


if __name__ == "__main__":
    run_agent()