# orchestration
from langchain.chat_models import ChatOpenAI

class RiskAgent:

    def __init__(self, retriever):
        self.llm = ChatOpenAI()
        self.retriever = retriever

    def explain(self, risk_score, features):

        context = self.retriever.similarity_search(
            "financial risk factors"
        )

        prompt = f"""
        Risk Score: {risk_score}

        Features:
        {features}

        Context:
        {context}

        Explain the risk and suggest actions.
        """

        return self.llm.predict(prompt)

# RiskAGent is the main orchestrator that ties everything together. It uses a retriever to get relevant context and a language model to generate explanations based on the risk score and features.
# class RiskAgent:

#     def __init__(self, llm, retriever):
#         self.llm = llm
#         self.retriever = retriever

#     def run(self, risk_score, features):

#         rules = risk_explanation_tool(features)
#         docs = self.retriever.similarity_search("credit risk advice")

#         prompt = f"""
#         Risk Score: {risk_score}

#         Feature Signals:
#         {features}

#         Rule-based insights:
#         {rules}

#         Knowledge:
#         {docs}

#         Provide:
#         1. Clear explanation
#         2. Top 3 actions
#         3. Risk level (low/medium/high)
#         """

#         return self.llm.invoke(prompt)