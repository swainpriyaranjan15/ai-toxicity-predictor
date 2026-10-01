"""
AI Toxicity Predictor
6-Label Jigsaw Model + Moderation Engine
"""

import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================
# MODEL SETTINGS
# ============================================================

MODEL_PATH = "./jigsaw_6label_model_v4"

LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]


# ============================================================
# TEXT PREPROCESSOR
# ============================================================

class TextPreprocessor:

    @staticmethod
    def preprocess(text):
        text = str(text).lower()

        # Remove URLs
        text = re.sub(
            r"http\S+|www\S+|https\S+",
            "",
            text,
            flags=re.MULTILINE
        )

        # Remove extra spaces
        text = re.sub(r"\s+", " ", text).strip()

        return text


# ============================================================
# TOXICITY PREDICTOR
# ============================================================

class ToxicityPredictor:

    def __init__(self, model_path=MODEL_PATH):

        print("Loading toxicity model...")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path
        )

        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded successfully on {self.device}")

    def predict(self, text):

        # Preprocess
        processed_text = TextPreprocessor.preprocess(text)

        # Tokenize
        inputs = self.tokenizer(
            processed_text,
            max_length=128,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        # Prediction
        with torch.no_grad():

            outputs = self.model(**inputs)

        # Multi-label classification
        probabilities = torch.sigmoid(outputs.logits)[0]

        predictions = {}

        for label, probability in zip(LABELS, probabilities):

            percentage = probability.item() * 100

            predictions[label] = {
                "probability": probability.item(),
                "percentage": round(percentage, 2),
                "detected": percentage >= 50
            }

        # Find highest probability label
        highest_label = max(
            LABELS,
            key=lambda label: predictions[label]["probability"]
        )

        highest_probability = predictions[highest_label]["probability"]

        # Determine overall result
        detected_labels = [
            label
            for label in LABELS
            if predictions[label]["detected"]
        ]

        if detected_labels:
            overall_prediction = "TOXIC"
        else:
            overall_prediction = "NORMAL"

        # Toxicity score = highest detected probability
        toxicity_score = highest_probability * 100

        # Severity
        if overall_prediction == "NORMAL":
            severity = "Low"

        elif "threat" in detected_labels:
            severity = "Severe"

        elif "identity_hate" in detected_labels:
            severity = "Severe"

        elif "severe_toxic" in detected_labels:
            severity = "High"

        elif "toxic" in detected_labels or "insult" in detected_labels:
            severity = "Medium"

        else:
            severity = "Low"

        result = {
            "text": text,
            "preprocessed_text": processed_text,
            "prediction": overall_prediction,
            "toxicity_score": round(toxicity_score, 2),
            "toxicity_score_percent": f"{toxicity_score:.2f}%",
            "highest_label": highest_label,
            "confidence": highest_probability,
            "confidence_percent": f"{highest_probability * 100:.2f}%",
            "severity": severity,
            "detected_labels": detected_labels,
            "class_probabilities": predictions
        }

        return result


# ============================================================
# MODERATION ENGINE
# ============================================================

class ModerationEngine:

    @staticmethod
    def get_moderation_action(result):

        prediction = result["prediction"]
        severity = result["severity"]
        confidence = result["confidence"]

        if prediction == "NORMAL":

            action = "Allow"
            reason = "No toxic content detected."

        elif confidence < 0.60:

            action = "Human Review"
            reason = "Prediction confidence is low."

        elif severity == "Severe":

            action = "Block & Report"
            reason = "Severe toxic content detected."

        elif severity == "High":

            action = "Hide Comment"
            reason = "Highly toxic content detected."

        elif severity == "Medium":

            action = "Warn User"
            reason = "Toxic or insulting content detected."

        else:

            action = "Allow"
            reason = "No significant toxicity detected."

        return {
            "action": action,
            "reason": reason,
            "moderation_message": f"[{action.upper()}] {reason}"
        }


# ============================================================
# COMPLETE PIPELINE
# ============================================================

class ToxicityModerationPipeline:

    def __init__(self, model_path=MODEL_PATH):

        self.predictor = ToxicityPredictor(model_path)
        self.moderator = ModerationEngine()

    def analyze(self, text):

        prediction = self.predictor.predict(text)

        moderation = self.moderator.get_moderation_action(
            prediction
        )

        result = {
            **prediction,
            **moderation
        }

        return result

    def batch_analyze(self, texts):

        results = []

        for text in texts:

            results.append(
                self.analyze(text)
            )

        return results


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    pipeline = ToxicityModerationPipeline()

    test_text = "You are stupid and I hate you."

    result = pipeline.analyze(test_text)

    print("\n" + "=" * 60)
    print("AI TOXICITY PREDICTOR")
    print("=" * 60)

    print("\nText:")
    print(result["text"])

    print("\nOverall Prediction:")
    print(result["prediction"])

    print("\nToxicity Score:")
    print(result["toxicity_score_percent"])

    print("\nHighest Label:")
    print(result["highest_label"])

    print("\nConfidence:")
    print(result["confidence_percent"])

    print("\nSeverity:")
    print(result["severity"])

    print("\nDetected Labels:")
    print(result["detected_labels"])

    print("\nModeration Action:")
    print(result["action"])

    print("\nReason:")
    print(result["reason"])

    print("\n" + "=" * 60)