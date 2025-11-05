# app.py
import re
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request, redirect, url_for
import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from aura import analyze_aura
from forecasting import forecast_mood  # NEW

# --- CONFIG: Replace with your Reddit credentials ---
REDDIT_CLIENT_ID = "M1I1jvc74xBb2xr-QuK2zQ"
REDDIT_CLIENT_SECRET = "qSFZqF4lbrk5kZ9q3UP44n0RMBHrig"
REDDIT_USER_AGENT = "wellness-tracker-demo"

# --- APP INIT ---
app = Flask(__name__)
analyzer = SentimentIntensityAnalyzer()
user_posts = {}  # in-memory store

# --- Reddit Init ---
def init_reddit():
    return praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )

# --- Preprocess ---
def preprocess_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9\s.,!?']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --- Fetch posts ---
def fetch_user_submissions(username, limit=100):
    reddit = init_reddit()
    ruser = reddit.redditor(username)
    posts = []
    for s in ruser.submissions.new(limit=limit):
        text = (s.title or "") + " " + (s.selftext or "")
        clean_text = preprocess_text(text)
        posts.append({
            "id": s.id,
            "title": s.title,
            "created": datetime.utcfromtimestamp(s.created_utc),
            "text": clean_text,
        })
    user_posts[username] = posts
    return posts

# --- Sentiment Analysis ---
def analyze_posts(posts):
    results = []
    for p in posts:
        scores = analyzer.polarity_scores(p["text"])
        results.append({
            "id": p["id"],
            "title": p["title"],
            "created": p["created"],
            "compound": scores["compound"],
        })
    return results

# --- Daily Mood ---
def get_daily_mood(username, days=60):
    posts = user_posts.get(username, [])
    if not posts:
        return []
    analyzed = analyze_posts(posts)
    cutoff = datetime.utcnow() - timedelta(days=days)
    buckets = {}
    for a in analyzed:
        if a["created"] < cutoff:
            continue
        d = a["created"].date().isoformat()
        buckets.setdefault(d, []).append(a["compound"])
    trend = [{"date": d, "avg_compound": sum(vals)/len(vals)} for d, vals in sorted(buckets.items())]
    return trend

# --- Alerts ---
def get_alerts(username, threshold=-0.5):
    trend = get_daily_mood(username, days=60)
    return [{"date": t["date"], "message": "Significant drop in mood"} for t in trend if t["avg_compound"] < threshold]

# --- ROUTES ---
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/dashboard")
def dashboard():
    username = request.args.get("username")
    if not username:
        return redirect(url_for("home"))
    return render_template("dashboard.html", username=username)

@app.route("/api/fetch/<username>")
def api_fetch(username):
    try:
        posts = fetch_user_submissions(username, limit=100)
        return jsonify({"fetched": len(posts)})
    except Exception as e:
        return jsonify({"fetched": 0, "error": str(e)})

@app.route("/api/mood_trend/<username>")
def api_mood(username):
    days = int(request.args.get("days", 60))
    daily = get_daily_mood(username, days)
    alerts = get_alerts(username)
    return jsonify({"trend": daily, "alerts": alerts})

@app.route("/api/aura/<username>")
def api_aura(username):
    posts = user_posts.get(username, [])
    aura_info = analyze_aura(posts)
    if not aura_info:
        return jsonify({"error": "Not enough data"})
    return jsonify(aura_info)

# --- NEW: Forecast Route ---
@app.route("/api/forecast/<username>")
def api_forecast(username):
    daily_mood = get_daily_mood(username, days=60)
    forecast_data = forecast_mood(daily_mood)
    return jsonify(forecast_data)

# --- RUN ---
if __name__ == "__main__":
    app.run(debug=True)
