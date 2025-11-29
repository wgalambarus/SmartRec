from flask import Flask, request, jsonify
from python import parse_cv_file, score_candidate, load_embedding_model  # from your file
import os

app = Flask(__name__)

model = load_embedding_model()   # load once (fast)

@app.route("/score", methods=["POST"])
def score():
    data = request.json

    job_desc = data.get("job_description")
    cv_path = data.get("cv_path")

    if not os.path.exists(cv_path):
        return jsonify({"error": "CV file not found"}), 400

    parsed = parse_cv_file(cv_path)
    score, breakdown = score_candidate(parsed, job_desc, model)

    return jsonify({
        "name": parsed["name"],
        "email": parsed["email"],
        "skills": parsed["skills"],
        "score": score,
        "breakdown": breakdown
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001)
