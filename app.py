from flask import Flask, request, jsonify
import torch
from transformers import MT5ForConditionalGeneration, MT5Tokenizer

app = Flask(__name__)

tokenizer = MT5Tokenizer.from_pretrained("LocalDoc/azerbaijani_spell_corrector")
model = MT5ForConditionalGeneration.from_pretrained("LocalDoc/azerbaijani_spell_corrector")

def correct_sentence(sentence):
    input_text = "correct: " + sentence
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=128, truncation=True)
    outputs = model.generate(input_ids=input_ids, max_length=128, num_beams=5, early_stopping=True)
    corrected_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_sentence

@app.route("/correct", methods=["POST"])
def correct():
    data = request.get_json()
    if "sentence" not in data:
        return jsonify({"error": "Missing 'sentence' in request"}), 400
    
    corrected_text = correct_sentence(data["sentence"])
    return jsonify({"corrected_sentence": corrected_text})

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Azerbaijani Grammar Checker API"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
