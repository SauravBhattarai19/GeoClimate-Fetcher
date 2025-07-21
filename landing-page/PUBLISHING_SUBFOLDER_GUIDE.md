# Publishing the Landing Page Folder - Step by Step Guide

Since you want to publish only the `landing-page` folder from your existing repository, here are the best approaches:

## Option 1: Create a Separate Repository (Recommended)

### Step 1: Create New Repository
1. Go to GitHub.com and create a new repository
2. Name it: `geoclimate-landing-page`
3. Make it **Public**
4. Don't initialize with README (we'll add our files)

### Step 2: Copy Files to New Location
1. Create a new folder on your computer (outside the current project)
2. Copy ONLY these files from your `landing-page` folder:
   ```
   NEW_FOLDER/
   ├── index.html
   ├── styles.css
   ├── script.js
   ├── photos/ (entire folder)
   ├── README.md
   └── .gitignore
   ```

### Step 3: Initialize Git in New Folder
Open terminal in your NEW folder and run:
```bash
git init
git add .
git commit -m "Initial commit: GeoClimate landing page"
git remote add origin https://github.com/yourusername/geoclimate-landing-page.git
git branch -M main
git push -u origin main
```

### Step 4: Enable GitHub Pages
1. Go to your new repository on GitHub
2. Settings → Pages
3. Source: "Deploy from a branch"
4. Branch: "main", Folder: "/ (root)"
5. Save

## Option 2: Use Subdirectory Deployment

If you want to keep everything in one repository, you can use GitHub Actions to deploy just the landing-page folder.

### Step 1: Create GitHub Actions Workflow
In your main repository, create `.github/workflows/deploy.yml`:

```yaml
name: Deploy Landing Page to GitHub Pages

on:
  push:
    branches: [ main ]
    paths: [ 'landing-page/**' ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to GitHub Pages
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./landing-page
        exclude_assets: '*.py,*.bat,server.py,simple-server.py'
```

### Step 2: Enable GitHub Pages
1. Repository Settings → Pages
2. Source: "GitHub Actions"

## Option 3: Use Git Subtree (Advanced)

Push only the landing-page folder to a separate repository:

```bash
# From your main repository root
git subtree push --prefix=landing-page origin gh-pages
```

## Option 4: Manual Copy Method (Simplest)

### Step 1: Download/Copy Files
1. Navigate to your `landing-page` folder
2. Select all files EXCEPT:
   - `server.py`
   - `simple-server.py` 
   - `start-server.bat`
   - Any `.pyc` files

### Step 2: Create New Repository
1. Create new GitHub repository
2. Upload files via web interface (drag & drop)
3. Enable GitHub Pages

## Recommended Approach

**Use Option 1** - Create a separate repository because:
- ✅ Cleaner separation of concerns
- ✅ Easier to manage
- ✅ No complex workflows needed
- ✅ Direct GitHub Pages deployment
- ✅ Independent version control for landing page

## Files to Include in New Repository

Copy ONLY these files:
```
geoclimate-landing-page/
├── index.html              ← Main page
├── styles.css              ← Styling
├── script.js               ← JavaScript
├── photos/                 ← All images
│   ├── rocky.png
│   ├── saurav.jpeg
│   ├── sunil.JPG
│   ├── subash.JPG
│   ├── samuel.jpeg
│   ├── douglas.jpeg
│   ├── workshop1.jpeg
│   ├── workshop2.jpeg
│   └── workshop3.jpeg
├── README.md               ← Documentation
└── .gitignore              ← Git ignore file
```

## DO NOT Include These Files:
- ❌ `server.py`
- ❌ `simple-server.py`
- ❌ `start-server.bat`
- ❌ `__pycache__/` folders
- ❌ Any `.py` files

## Your Final URL
After deployment: `https://yourusername.github.io/geoclimate-landing-page/`

## Quick Commands for Option 1

```bash
# 1. Create new folder and copy files
mkdir geoclimate-landing-page
# Copy your landing page files to this new folder

# 2. Initialize git
cd geoclimate-landing-page
git init
git add .
git commit -m "Initial commit: GeoClimate landing page"

# 3. Connect to new GitHub repository
git remote add origin https://github.com/yourusername/geoclimate-landing-page.git
git push -u origin main

# 4. Enable GitHub Pages in repository settings
```

This approach gives you a clean, dedicated repository just for your landing page! 🚀
