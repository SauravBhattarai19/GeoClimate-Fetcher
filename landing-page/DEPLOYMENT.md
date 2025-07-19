# Deployment Guide for GeoClimate Intelligence Platform Landing Page

## Quick Start

To view the landing page locally:

```bash
# Navigate to the landing-page directory
cd landing-page

# Option 1: Using Python's built-in server
python server.py

# Option 2: Using Python's simple HTTP server
python -m http.server 8000

# Option 3: Using Node.js (if you have it installed)
npx serve .
```

Open your browser and navigate to:
- Demo page: http://localhost:8000/demo.html
- Main landing page: http://localhost:8000/index.html
- Microsoft partnership page: http://localhost:8000/microsoft-partnership.html

## Deployment Options

### 1. GitHub Pages (Free)

1. Push the `landing-page` folder to a GitHub repository
2. Go to repository Settings > Pages
3. Select source branch (usually `main`)
4. Set folder to `/landing-page` or move files to root
5. Your site will be available at `https://username.github.io/repository-name`

**Setup Commands:**
```bash
# If landing-page is in the root of your repo
git add landing-page/
git commit -m "Add landing page"
git push origin main

# Then enable GitHub Pages in repository settings
```

### 2. Netlify (Free tier available)

1. Drag and drop the `landing-page` folder to [netlify.com](https://netlify.com)
2. Or connect your GitHub repository
3. Set build directory to `landing-page`
4. Deploy automatically

**Netlify Deploy:**
```bash
# Install Netlify CLI
npm install -g netlify-cli

# Deploy from landing-page directory
cd landing-page
netlify deploy --prod
```

### 3. Vercel (Free tier available)

1. Install Vercel CLI: `npm install -g vercel`
2. In the landing-page directory, run: `vercel`
3. Follow the prompts

**Vercel Deploy:**
```bash
# Install Vercel CLI
npm install -g vercel

# Deploy from landing-page directory
cd landing-page
vercel --prod
```

### 4. Azure Static Web Apps (Microsoft Integration)

Since this is a Microsoft AUC project, Azure Static Web Apps is a great choice:

1. Go to [Azure Portal](https://portal.azure.com)
2. Create a new "Static Web App" resource
3. Connect your GitHub repository
4. Set app location to `/landing-page`
5. Leave build location empty (static site)

**Azure Configuration:**
```yaml
# staticwebapp.config.json
{
  "routes": [
    {
      "route": "/",
      "serve": "/index.html"
    },
    {
      "route": "/*",
      "serve": "/404.html",
      "statusCode": 404
    }
  ],
  "navigationFallback": {
    "rewrite": "/index.html"
  }
}
```

### 5. Firebase Hosting (Google Integration)

Good choice since you're using Google Earth Engine:

1. Install Firebase CLI: `npm install -g firebase-tools`
2. Run `firebase init hosting` in the landing-page directory
3. Set public directory to `.` (current directory)
4. Deploy with `firebase deploy`

**Firebase Setup:**
```bash
# Install Firebase CLI
npm install -g firebase-tools

# Initialize and deploy
cd landing-page
firebase login
firebase init hosting
firebase deploy
```

## Custom Domain Setup

After deploying to any platform, you can add a custom domain:

### Suggested Domains:
- `geoclimate-platform.com`
- `geoclimate-intelligence.org`
- `jsu-geoclimate.edu` (if available through your university)

### DNS Configuration:
1. Purchase domain from registrar (Namecheap, GoDaddy, etc.)
2. Add CNAME record pointing to your hosting provider
3. Configure SSL certificate (usually automatic)

## Performance Optimization

The landing page is already optimized, but for production:

### 1. Minification (Optional)
```bash
# Install build tools
npm install -g html-minifier clean-css-cli terser

# Minify files
html-minifier --remove-comments --collapse-whitespace index.html > index.min.html
cleancss styles.css > styles.min.css
terser script.js > script.min.js
```

### 2. CDN Configuration
- Use a CDN for faster global delivery
- Cloudflare (free tier available) works with any hosting provider
- Azure CDN if using Azure Static Web Apps

### 3. Image Optimization
- The current site uses text/emoji icons for fast loading
- If you add images later, use WebP format and lazy loading

## Monitoring and Analytics

Add tracking to understand usage:

### Google Analytics
Add to `<head>` section:
```html
<!-- Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'GA_MEASUREMENT_ID');
</script>
```

### Microsoft Clarity (Free)
```html
<!-- Microsoft Clarity -->
<script type="text/javascript">
    (function(c,l,a,r,i,t,y){
        c[a]=c[a]||function(){(c[a].q=c[a].q||[]).push(arguments)};
        t=l.createElement(r);t.async=1;t.src="https://www.clarity.ms/tag/"+i;
        y=l.getElementsByTagName(r)[0];y.parentNode.insertBefore(t,y);
    })(window, document, "clarity", "script", "CLARITY_ID");
</script>
```

## SEO Optimization

The landing page includes basic SEO, but you can enhance it:

### Add to `<head>`:
```html
<!-- Open Graph for social media -->
<meta property="og:title" content="GeoClimate Intelligence Platform">
<meta property="og:description" content="Advanced Earth Engine climate data analysis platform">
<meta property="og:image" content="https://your-domain.com/og-image.jpg">
<meta property="og:url" content="https://your-domain.com">

<!-- Twitter Card -->
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="GeoClimate Intelligence Platform">
<meta name="twitter:description" content="Microsoft AUC Partnership Project">

<!-- Canonical URL -->
<link rel="canonical" href="https://your-domain.com">

<!-- Structured Data -->
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "SoftwareApplication",
  "name": "GeoClimate Intelligence Platform",
  "applicationCategory": "Climate Analysis Software",
  "operatingSystem": "Web Browser",
  "offers": {
    "@type": "Offer",
    "price": "0",
    "priceCurrency": "USD"
  }
}
</script>
```

## Security Headers

Add security headers (example for Netlify):

Create `_headers` file:
```
/*
  X-Frame-Options: DENY
  X-Content-Type-Options: nosniff
  X-XSS-Protection: 1; mode=block
  Referrer-Policy: strict-origin-when-cross-origin
  Content-Security-Policy: default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdnjs.cloudflare.com; font-src https://fonts.gstatic.com; img-src 'self' data: https:;
```

## Maintenance

### Regular Updates:
1. Update CDN library versions quarterly
2. Check all external links monthly
3. Monitor site performance with PageSpeed Insights
4. Update content as the platform evolves

### Backup Strategy:
- Keep source code in version control (Git)
- Export hosting provider configurations
- Document any custom domain/DNS settings

## Troubleshooting

### Common Issues:

1. **CORS errors**: Use the provided `server.py` for local development
2. **Missing fonts**: Ensure Google Fonts CDN is accessible
3. **QR code not generating**: Check that QRCode library is loaded
4. **Mobile layout issues**: Test on multiple devices

### Support:
- Create issues in the GitHub repository
- Contact the development team
- Check browser console for JavaScript errors

---

**Recommended Deployment for Microsoft AUC Project:**
**Azure Static Web Apps** - Best integration with Microsoft ecosystem, free tier available, automatic SSL, and custom domains supported.
