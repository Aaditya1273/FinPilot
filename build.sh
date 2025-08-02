#!/bin/bash

# Create public directory for Netlify
mkdir -p public

# Copy static files to public directory
cp landing.html public/index.html
cp about.html public/about.html
cp -r calculative public/
cp -r assets public/
cp -r svg public/

# Copy favicon and other assets
if [ -f "favicon.ico" ]; then
    cp favicon.ico public/
fi

echo "✅ Build completed! Files ready for Netlify deployment."
echo "📁 Static files copied to public/ directory"
echo "🔧 Serverless functions ready in netlify/functions/"
