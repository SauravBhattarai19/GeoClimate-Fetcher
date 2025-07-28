# GitHub Pages Deployment Guide

This guide will help you publish your GeoClimate Intelligence Platform landing page using GitHub Pages for free hosting.

## Prerequisites

- GitHub account
- Git installed on your computer
- Your landing page files ready (index.html, styles.css, script.js, photos/, etc.)

## Method 1: Direct GitHub Upload (Easiest)

### Step 1: Create a New Repository
1. Go to [GitHub.com](https://github.com) and sign in
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Name it: `geoclimate-landing-page` (or any name you prefer)
5. Make it **Public** (required for free GitHub Pages)
6. Check "Add a README file"
7. Click "Create repository"

### Step 2: Upload Your Files
1. In your new repository, click "uploading an existing file"
2. Drag and drop ALL your landing page files:
   - `index.html`
   - `styles.css`
   - `script.js`
   - `photos/` folder (with all images)
   - Any other files except server files
3. Add a commit message like "Add GeoClimate landing page"
4. Click "Commit changes"

### Step 3: Enable GitHub Pages
1. In your repository, go to **Settings** tab
2. Scroll down to **Pages** section (left sidebar)
3. Under "Source", select **"Deploy from a branch"**
4. Choose **"main"** branch
5. Select **"/ (root)"** folder
6. Click **"Save"**

### Step 4: Access Your Website
- GitHub will provide a URL like: `https://yourusername.github.io/geoclimate-landing-page/`
- It may take 5-10 minutes to be live
- Your site will auto-update whenever you push changes

## Method 2: Using Git Command Line

### Step 1: Initialize Git in Your Landing Page Folder
```bash
cd "C:\Users\J01013381\OneDrive - Jackson State University\Research Projects\2025\1 Microsoft Manual\GeoClimate-Fetcher\landing-page"
git init
git add .
git commit -m "Initial commit: GeoClimate landing page"
```

### Step 2: Connect to GitHub Repository
```bash
git remote add origin https://github.com/yourusername/geoclimate-landing-page.git
git branch -M main
git push -u origin main
```

### Step 3: Enable GitHub Pages (same as Method 1, Step 3)

## Method 3: Using GitHub Desktop (User-Friendly)

### Step 1: Install GitHub Desktop
- Download from [desktop.github.com](https://desktop.github.com)
- Install and sign in with your GitHub account

### Step 2: Create Repository
- Click "Create a New Repository on your hard drive"
- Name: `geoclimate-landing-page`
- Local path: Choose your landing page folder
- Check "Publish to GitHub.com"
- Make it Public
- Click "Create Repository"

### Step 3: Publish
- Click "Publish repository"
- Enable GitHub Pages from repository settings (same as above)

## Custom Domain (Optional)

If you want to use your own domain (e.g., geoclimate.yourdomain.com):

1. In your repository, create a file named `CNAME`
2. Add your domain name (e.g., `geoclimate.yourdomain.com`)
3. Configure your domain's DNS settings to point to GitHub Pages

## Files to Include/Exclude

### ✅ Include These Files:
- `index.html` (main page)
- `styles.css` (styling)
- `script.js` (functionality)
- `photos/` folder (all images)
- `README.md` (optional project description)

### ❌ Exclude These Files:
- `server.py` (not needed for GitHub Pages)
- `simple-server.py` (not needed)
- `start-server.bat` (not needed)
- `__pycache__/` folders
- `.py` files (Python server files)

## Tips for Success

### 1. File Structure Should Look Like:
```
your-repo/
├── index.html          (your main page)
├── styles.css          (your CSS)
├── script.js           (your JavaScript)
├── photos/             (image folder)
│   ├── rocky.png
│   ├── saurav.jpeg
│   ├── workshop1.jpeg
│   └── ...
└── README.md           (optional)
```

### 2. Update Links
- All file paths should be relative (no `C:\` paths)
- Use forward slashes: `photos/rocky.png`
- GitHub Pages serves index.html automatically

### 3. Test Locally First
- Run your local server to ensure everything works
- Check all images load correctly
- Verify all links work

### 4. Common Issues
- **Images not loading**: Check file paths and case sensitivity
- **Site not updating**: Wait 10 minutes, clear browser cache
- **404 errors**: Ensure file names match exactly (case sensitive)

## Example Repository URLs

After deployment, your site will be available at:
- `https://yourusername.github.io/geoclimate-landing-page/`
- Or with custom domain: `https://geoclimate.yourdomain.com`

## Automatic Updates

Once set up, any changes you make to the repository will automatically update your live website within minutes!

## Need Help?

- [GitHub Pages Documentation](https://pages.github.com/)
- [GitHub Desktop Help](https://docs.github.com/en/desktop)
- Check repository settings if site doesn't load

---

## Quick Start Commands

```bash
# If using command line:
git init
git add .
git commit -m "Add GeoClimate landing page"
git remote add origin https://github.com/yourusername/your-repo-name.git
git push -u origin main
```

Your professional GeoClimate Intelligence Platform landing page will be live on the web! 🚀
