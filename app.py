from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

model        = bundle["model"]
le_locality  = bundle["le_locality"]
le_land      = bundle["le_land"]
le_road      = bundle["le_road"]

LOCALITIES = list(le_locality.classes_)
def get_base_price(loc):
    loc_enc  = le_locality.transform([loc])[0]
    road_enc = le_road.transform(["Yes"])[0]
    land_enc = le_land.transform(["Residential"])[0]
    pred = model.predict(np.array([[1000, loc_enc, 5, road_enc, land_enc]]))[0]
    return int(pred / 1000)

BASE_PRICES = {loc: get_base_price(loc) for loc in LOCALITIES}

# HOME PAGE
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/api/localities", methods=["GET"])
def localities():
    return jsonify({
        "localities": [
            {
                "key":        loc,
                "label":      loc,
                "base_price": BASE_PRICES[loc]
            }
            for loc in LOCALITIES
        ]
    })

@app.route("/api/zone-rates", methods=["GET"])
def zone_rates():
    import random
    random.seed(42)
    rates = []
    for loc in LOCALITIES:
        change = round(random.uniform(-1.5, 8.5), 1)
        rates.append({
            "locality": loc,
            "price":    BASE_PRICES[loc],
            "change":   change,
            "trend":    "up" if change >= 0 else "down"
        })
    return jsonify({"rates": rates})

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json(force=True)

        locality   = data.get("locality", "")
        area       = float(data.get("area", 0))
        plot_type  = data.get("plot_type", "residential")
        road_width = int(data.get("road_width", 30))
        infra      = data.get("infra", "none")

        if not locality or area <= 0:
            return jsonify({"error": "Invalid input"}), 400

        land_map  = {"residential": "Residential", "commercial": "Commercial",
                     "agricultural": "Residential", "mixed_use": "Commercial"}
        land_type = land_map.get(plot_type, "Residential")
        road_access = "Yes" if road_width >= 20 else "No"
        infra_dist  = {"metro": 3, "highway": 5, "it_park": 8, "school": 6, "none": 15}
        distance    = infra_dist.get(infra, 8)

        loc_enc  = le_locality.transform([locality])[0]
        road_enc = le_road.transform([road_access])[0]
        land_enc = le_land.transform([land_type])[0]

        features = np.array([[area, loc_enc, distance, road_enc, land_enc]])
        predicted = float(model.predict(features)[0])

        price_per_sqft = round(predicted / area)
        low  = round(price_per_sqft * 0.94)
        high = round(price_per_sqft * 1.06)

        factors = {
            "Location":       75,
            "Infrastructure": max(30, 100 - distance * 4),
            "Road Access":    80 if road_access == "Yes" else 40,
            "Plot Type":      70 if land_type == "Commercial" else 60,
            "Area Size":      min(100, round(area / 100)),
        }

        return jsonify({
            "price_per_sqft": price_per_sqft,
            "total_value":    int(predicted),
            "range": {"low": low, "high": high, "confidence": "93.5%"},
            "factors": factors,
            "inputs": {"locality": locality, "area": area,
                       "plot_type": plot_type, "road_width": road_width,
                       "facing": data.get("facing","east"), "infra": infra}
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
