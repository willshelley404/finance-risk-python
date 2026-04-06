from groq import Groq
from app.config import GROQ_API_KEY, MODEL_NAME
import json

class FinanceAgent:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = MODEL_NAME
        
        # Add system prompt (improves responses a LOT)
        self.conversation_history = [
            {
                "role": "system",
                "content": "You are a financial risk advisor. Provide clear, practical, and actionable recommendations based on user risk profiles."
            }
        ]
    
    # ✅ NEW: helper to fix numpy serialization
    def _to_serializable(self, obj):
        if isinstance(obj, dict):
            return {k: self._to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._to_serializable(v) for v in obj]
        elif hasattr(obj, "tolist"):  # numpy arrays
            return obj.tolist()
        elif hasattr(obj, "item"):  # numpy scalars
            return obj.item()
        else:
            return obj
    
    def add_context(self, risk_score, explanation, user_profile):
        safe_explanation = self._to_serializable(explanation)
        safe_profile = self._to_serializable(user_profile)

        context = f"""
        Risk Assessment Context:
        - Default Risk Score: {risk_score:.2%}
        - User Profile: {json.dumps(safe_profile)}
        - Key Risk Factors: {json.dumps(safe_explanation)}
        """
        
        self.conversation_history.append({
            "role": "user",
            "content": context
        })
    
    def ask(self, question):
        self.conversation_history.append({
            "role": "user",
            "content": question
        })
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.conversation_history,
            temperature=0.7,
            max_tokens=500
        )
        
        answer = response.choices[0].message.content
        self.conversation_history.append({
            "role": "assistant",
            "content": answer
        })
        return answer
    
    def get_recommendations(self, risk_score, risk_factors):
        # ✅ Fix numpy issue
        safe_risk_factors = self._to_serializable(risk_factors)

        factors_text = []

        for k, v in safe_risk_factors.items():
            # Handle list/array cases
            if isinstance(v, list):
                v = v[0] if len(v) > 0 else 0

            try:
                v = float(v)
            except:
                v = 0.0  # fallback safety

            factors_text.append(f"- {k}: {round(v, 3)}")

        factors_text = "\n".join(factors_text)

        risk_level = "HIGH" if risk_score > 0.7 else "MODERATE" if risk_score > 0.4 else "LOW"

        prompt = f"""
        Based on this person's financial risk profile:

        Risk Score: {risk_score:.2%}
        Risk Level: {risk_level}

        Key Risk Drivers:
        {factors_text}

        Provide 3-5 specific, actionable recommendations to reduce financial risk.
        Be concise and prioritize the most impactful actions.
        """
        
        return self.ask(prompt)