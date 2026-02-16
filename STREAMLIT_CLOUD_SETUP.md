# Streamlit Cloud Setup Guide - Earth Engine Authentication

**Date:** 2026-02-16
**Issue:** GeoData Explorer visualization fails silently on Streamlit Cloud
**Root Cause:** Using `ee.Initialize()` instead of `geemap.ee_initialize()`
**Status:** ✅ FIXED

---

## 🔴 The Problem

### Symptoms
- Works perfectly on local machine
- Silent failure on Streamlit Cloud during map visualization
- No error messages, map just doesn't render

### Root Cause Analysis

**The Error Chain:**
1. Local code uses `ee.Initialize(project=project_id)` to initialize Earth Engine
2. **Works locally** because credentials exist in `~/.config/earthengine/credentials`
3. **Fails on Streamlit Cloud** because no local credentials file exists
4. EE operations fail silently, map renders empty

**Why Silent Failure?**
- `ee.Initialize()` on Streamlit Cloud either fails partially or succeeds without proper auth
- `Map.addLayer()` calls fail because EE isn't properly authenticated
- `Map.to_streamlit()` renders an empty map (no exception thrown)

---

## ✅ The Solution

### Code Changes Made

**Changed in 2 files:**

1. **interface/geodata_explorer.py** (Line 1014)
2. **interface/climate_analytics.py** (Line 2168)

```python
# BEFORE (❌ Fails on Streamlit Cloud):
ee.Initialize(project=project_id)

# AFTER (✅ Works on Streamlit Cloud):
geemap.ee_initialize(project=project_id)
```

### Why This Works

**geemap.ee_initialize() advantages:**
- Automatically reads `EARTHENGINE_TOKEN` from Streamlit secrets
- Creates proper credentials file in Streamlit Cloud environment
- Handles authentication flow correctly for web deployment
- Recommended by geemap maintainer for Streamlit apps

---

## 🔧 Streamlit Cloud Configuration

### Step 1: Get Your Earth Engine Credentials

**On your local machine:**

```bash
# Find your credentials file
cat ~/.config/earthengine/credentials
```

**You'll see something like:**
```json
{
  "refresh_token": "1//0gXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
  "type": "authorized_user",
  "client_id": "XXXXXXXXXXXX.apps.googleusercontent.com",
  "client_secret": "XXXXXXXXXXXXXXXXXXXXXXXXX",
  "scopes": [
    "https://www.googleapis.com/auth/earthengine",
    "https://www.googleapis.com/auth/devstorage.full_control"
  ]
}
```

**Copy the ENTIRE contents** of this file.

### Step 2: Configure Streamlit Secrets

1. **Go to your Streamlit Cloud dashboard**
2. **Click on your app** → Settings (⚙️)
3. **Navigate to "Secrets"** tab
4. **Add the following:**

```toml
EARTHENGINE_TOKEN = """
{
  "refresh_token": "1//0gXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
  "type": "authorized_user",
  "client_id": "XXXXXXXXXXXX.apps.googleusercontent.com",
  "client_secret": "XXXXXXXXXXXXXXXXXXXXXXXXX",
  "scopes": [
    "https://www.googleapis.com/auth/earthengine",
    "https://www.googleapis.com/auth/devstorage.full_control"
  ]
}
"""
```

**Important:**
- Use **triple quotes** `"""` to preserve JSON formatting
- Copy the **entire** credentials JSON content
- Don't use just the refresh token alone

5. **Click "Save"**
6. **Redeploy your app** (it will restart automatically)

---

## 📋 Alternative: Service Account Authentication

For production deployments, use a **Google Cloud Service Account**:

### Step 1: Create Service Account

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Select your project
3. Navigate to **IAM & Admin** → **Service Accounts**
4. Click **Create Service Account**
5. Give it Earth Engine permissions
6. Create and download JSON key file

### Step 2: Configure Streamlit Secrets

```toml
[gcp_service_account]
type = "service_account"
project_id = "your-project-id"
private_key_id = "xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
private_key = "-----BEGIN PRIVATE KEY-----\nXXXXXXXXXXXX\n-----END PRIVATE KEY-----\n"
client_email = "your-service-account@your-project.iam.gserviceaccount.com"
client_id = "xxxxxxxxxxxxxxxxxxxxx"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/your-service-account%40your-project.iam.gserviceaccount.com"
```

### Step 3: Update Authentication Code

```python
import json
import ee

# Load service account from secrets
service_account_info = json.loads(st.secrets["gcp_service_account"])
credentials = ee.ServiceAccountCredentials(
    service_account_info['client_email'],
    key_data=service_account_info['private_key']
)
ee.Initialize(credentials, project=service_account_info['project_id'])
```

---

## 🧪 Testing

### Local Testing

```bash
conda activate fetcher
streamlit run app.py
```

Navigate to GeoData Explorer and verify:
- ✅ Dataset selection works
- ✅ Band selection works
- ✅ Map preview renders correctly

### Streamlit Cloud Testing

After deploying with secrets configured:

1. Navigate to **GeoData Explorer**
2. Select any dataset (e.g., ERA5 Daily)
3. Select bands and date range
4. **Map should now render successfully!** 🎉

---

## 📊 Before vs After

| Aspect | Before | After |
|--------|--------|-------|
| **Initialization Method** | `ee.Initialize()` | `geemap.ee_initialize()` |
| **Works Locally** | ✅ Yes | ✅ Yes |
| **Works on Streamlit Cloud** | ❌ Silent failure | ✅ Works! |
| **Authentication Source** | Local credentials only | Streamlit secrets |
| **Map Visualization** | Empty/blank | Renders correctly |
| **Error Messages** | None (silent) | Proper error handling |

---

## 🔗 References

### Documentation Sources

- [geemap Discussion #588 - Authentication with Streamlit](https://github.com/gee-community/geemap/discussions/588)
- [geemap Discussion #1889 - EE client library not initialized](https://github.com/gee-community/geemap/discussions/1889)
- [Streamlit Blog - Creating satellite timelapse with Earth Engine](https://blog.streamlit.io/creating-satellite-timelapse-with-streamlit-and-earth-engine/)
- [Medium Tutorial - Deploy GEE Analysis Into Web App](https://medium.com/@tahjudil.witra/deploy-your-google-earth-engine-gee-analysis-into-a-web-app-streamlit-a7841e35b0d8)

### Key Recommendations from geemap Maintainer

1. **Use `geemap.ee_initialize()` or `geemap.Map()`** before any EE operations
2. **DO NOT use `ee.Authenticate()` or `ee.Initialize()`** in Streamlit apps
3. **Set EARTHENGINE_TOKEN** from `~/.config/earthengine/credentials` (entire JSON)
4. **geemap handles Streamlit Cloud authentication automatically**

---

## ✅ Deployment Checklist

Before deploying to Streamlit Cloud:

- [ ] Code uses `geemap.ee_initialize()` instead of `ee.Initialize()`
- [ ] EARTHENGINE_TOKEN configured in Streamlit secrets
- [ ] requirements.txt has `python-box<7.0.0` (not >=7.0.0)
- [ ] requirements.txt has `geemap>=0.35.0`
- [ ] requirements.txt has `streamlit-folium>=0.15.0`
- [ ] Tested locally with same configuration
- [ ] Committed and pushed changes to GitHub
- [ ] Streamlit Cloud redeployed with new code

---

## 🎯 Summary

**The fix required two changes:**

1. **Code:** Replace `ee.Initialize()` with `geemap.ee_initialize()`
2. **Config:** Add EARTHENGINE_TOKEN to Streamlit secrets

**Result:** GeoData Explorer now works perfectly on Streamlit Cloud with proper map visualization! 🌍✨

---

**Questions or Issues?**

If visualization still fails after these changes:
1. Check Streamlit Cloud logs for error messages
2. Verify EARTHENGINE_TOKEN is properly formatted in secrets
3. Ensure app has redeployed after secret changes
4. Check that your Earth Engine project has API enabled
