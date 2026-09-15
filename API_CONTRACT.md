# FakeNews_AI — AI API Contract

Version: 1.0.0

## 1. Purpose

FakeNews_AI exposes the hybrid fake-news detection pipeline through
a REST API.

The AI service accepts a news claim and returns:

- ML classification
- Real-time evidence information
- Evidence stance
- NLI summary
- Retrieved articles
- LIME explanation
- SHAP explanation
- Final hybrid decision

The AI service uses three possible final decisions:

- REAL
- FAKE
- UNCERTAIN

The system does not claim absolute truth.

---

# 2. Base URL

Local development:

http://127.0.0.1:8000

---

# 3. Endpoints

## GET /

Health/basic service information.

### Response

```json
{
  "service": "FakeNews_AI",
  "status": "running",
  "version": "1.0.0"
}