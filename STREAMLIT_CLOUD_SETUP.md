# Streamlit Cloud Setup Guide

This guide covers deploying the GeoClimate Intelligence Platform on Streamlit Cloud, including the **Quick Access** service account setup.

## Basic Deployment

1. Fork or push the repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app" → select your repository
4. Set main file path to `app.py`
5. Deploy

## Service Account Setup (for Quick Access feature)

This enables the "Quick Access" mode so users can use the platform **instantly without any setup** — no credentials upload, no project ID, no terminal commands.

### Cost

- Google Earth Engine is **FREE** for research and education
- This does **NOT** consume GCP billing credits ($300 trial or otherwise)
- Creating a service account is free
- GEE has rate limits (not billing limits) — if too many concurrent users, requests queue but you are **never charged**
- The $300 GCP free credit is for OTHER services (Compute Engine, BigQuery, etc.) — GEE does not touch it
- **Total cost of this setup: $0**

### 1. Create a GCP Project (Free, One-Time)

1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Create a new project (or select an existing one)
3. Note down your **Project ID** (e.g., `geoclimate-platform-12345`)

### 2. Enable the Earth Engine API

1. In Google Cloud Console, go to **APIs & Services → Library**
2. Search for **"Earth Engine"**
3. Click **"Google Earth Engine API"** → **Enable**

### 3. Create a Service Account

1. Go to **IAM & Admin → Service Accounts**
2. Click **"Create Service Account"**
3. Name: `geoclimate-platform` (or any name you like)
4. No roles needed (GEE auth is handled separately)
5. Click **"Create and Continue"** → **"Done"**

### 4. Generate a JSON Key

1. Click on the new service account in the list
2. Go to the **"Keys"** tab
3. Click **"Add Key"** → **"Create new key"** → **JSON**
4. Download the JSON file — **keep this safe, don't commit to Git**

### 5. Register the Service Account with Earth Engine

1. Go to [code.earthengine.google.com](https://code.earthengine.google.com)
2. Click the user icon → **"Register a new project"**
3. Select **"Use with a Cloud Project"**
4. Enter your project ID and register the service account email
   (it looks like `geoclimate-platform@your-project.iam.gserviceaccount.com`)

### 6. Add Secrets in Streamlit Cloud

Go to your app's **Settings → Secrets** and paste:

```toml
[gee]
client_email = "geoclimate-platform@YOUR-PROJECT.iam.gserviceaccount.com"
project_id = "YOUR-GCP-PROJECT-ID"
key_json = '{"type":"service_account","project_id":"...","private_key_id":"...","private_key":"-----BEGIN RSA PRIVATE KEY-----\\n...\\n-----END RSA PRIVATE KEY-----\\n","client_email":"...","client_id":"...","auth_uri":"https://accounts.google.com/o/oauth2/auth","token_uri":"https://oauth2.googleapis.com/token","auth_provider_x509_cert_url":"https://www.googleapis.com/oauth2/v1/certs","client_x509_cert_url":"..."}'
```

**Important:**
- The `key_json` value is the **entire contents** of the downloaded JSON key file, on a single line, enclosed in single quotes
- Replace `\n` inside the private key with `\\n` (escaped newlines)
- **NEVER commit this to Git** — `.streamlit/secrets.toml` is already in `.gitignore`

### 7. Verify

1. Redeploy (or reboot) your Streamlit app
2. Visit the app — "Quick Access" should auto-authenticate
3. Test all modules: GeoData Explorer, Climate Analytics, Hydrology, Product Selector, Data Visualizer

## Local Development with Secrets

For local testing with the service account, create `.streamlit/secrets.toml` in the project root with the same format as above. This file is gitignored.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Quick Access unavailable" | Secrets not configured — check Settings → Secrets in Streamlit Cloud |
| "Service account not registered" | Register at code.earthengine.google.com with the service account email |
| "Earth Engine API not enabled" | Enable it in Google Cloud Console → APIs & Services → Library |
| "Permission denied" | Ensure the service account email is registered with Earth Engine |
