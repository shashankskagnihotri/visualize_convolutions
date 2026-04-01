from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from convolution_visualizer.convolution import parse_request, solve_convolution


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/api/compute")
    def compute():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Expected a JSON object payload."}), 400

        try:
            parsed = parse_request(payload)
            result = solve_convolution(parsed)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        return jsonify(result)

    return app
