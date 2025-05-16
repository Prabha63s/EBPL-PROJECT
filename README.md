from flask import Flask, request, jsonify
from transformers import pipeline

# Initialize the app and NLP model
app = Flask(__name__)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Predefined intents and responses
intents = {
    "reset password": "You can reset your password using the 'Forgot Password' option on the login page.",
    "order status": "Please provide your order ID. We’ll fetch the current status for you.",
    "return policy": "Our return policy allows returns within 30 days of purchase.",
    "technical support": "Please describe the issue you're facing, and we’ll assist you shortly.",
    "greetings": "Hello! How can I assist you today?",
    "goodbye": "Thank you for contacting us. Have a great day!"
}

intent_labels = list(intents.keys())

@app.route('/chat', methods=['POST'])
def chatbot():
    user_message = request.json.get("message", "")
    
    if not user_message:
        return jsonify({"response": "Please enter a valid query."}), 400
    
    # Predict intent
    result = classifier(user_message, intent_labels)
    best_intent = result['labels'][0]

    # Fetch response
    response = intents.get(best_intent, "I'm not sure how to help with that. Let me connect you to a human agent.")
    
    return jsonify({
        "intent": best_intent,
        "response": response
    })

if __name__ == "__main__":
    app.run(debug=True)
