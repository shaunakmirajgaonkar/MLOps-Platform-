# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| v1.x (current) | ✅ |

## Reporting a Vulnerability

Please **do not open a public GitHub issue** for security vulnerabilities.

Report privately via GitHub Security Advisory tab.

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

You will receive a response within 72 hours.

---

## Security Model

- All data stored locally in SQLite
- No external API calls
- No user authentication by default (single-user local tool)
- Model artifacts stored in local `models/` folder
- No patient or sensitive data transmitted anywhere

## Recommendations for Shared Deployment

- Place behind a reverse proxy with authentication (nginx + basic auth)
- Bind Streamlit to `127.0.0.1` only
- Do not expose port 8501 publicly
