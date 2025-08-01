# 🚀 FinPilot - AI-Powered Financial Modeling Platform

![FinPilot Banner](./svg/sdk.png)

[![FinPilot](https://img.shields.io/badge/FinPilot-AI--Powered-blueviolet?style=for-the-badge&logo=openai)](https://github.com/Aaditya1273/FinPilot)
[![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Chart.js](https://img.shields.io/badge/Chart.js-FF6384?style=for-the-badge&logo=chartdotjs&logoColor=white)](https://www.chartjs.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)

> **Navigate your financial future with AI-powered insights and stunning visualizations**

FinPilot is a cutting-edge financial modeling platform that combines advanced analytics, predictive modeling, and an intuitive glassmorphism interface to empower strategic business decisions and drive growth.

---

## ✨ Key Features:

### 🎨 **Stunning Colorful Glassmorphism Design**
- **Vibrant Color Palette**: Beautiful multi-color gradient charts with transparency effects
- **Premium Glass Effects**: Advanced backdrop-filter blur (20px) for professional glass look
- **Rainbow Gradients**: Colorful gradient backgrounds, borders, and text effects
- **Enhanced Shadows**: Multi-layered colored glow effects and drop shadows
- **Shimmer Animations**: Rainbow shimmer effects on interactive elements

### 🎬 **Advanced Animation System**
- **Staggered Entrance**: Cascading slide-in animations with 0.1s delays
- **Dynamic Number Counting**: Smooth counting from 0 to target values (1.5s duration)
- **Chart Animations**: Professional easeOutQuart animations for all visualizations
- **Hover Effects**: Scale transforms, glows, and enhanced shadows
- **Loading States**: Elegant shimmer effects during data processing

### 📊 **Dynamic Financial Charts**
All charts are **fully dynamic** and respond to real user inputs:

#### **Chart Types:**
1. **📈 Unit Economics (LTV vs CAC)** - Pie Chart with bright green & coral red
2. **💰 Revenue Breakdown** - Multi-color pie (blue, yellow, purple) with realistic SaaS splits
3. **⏰ Runway & Capital Projection** - Vibrant orange line chart with gradient fills
4. **📊 MRR Growth Projection** - Fresh green line chart with 12-month projections
5. **💸 Expense Distribution** - Colorful doughnut (pink, indigo, cyan, lime, orange)
6. **⚖️ Risk vs Growth Analysis** - Multi-gradient scatter plot with intelligent scoring

#### **Smart Features:**
- **Real-Time Calculations**: All data computed from actual form inputs
- **Intelligent Tooltips**: Comprehensive explanations with glassmorphism styling
- **Trend Indicators**: Color-coded performance metrics (Green/Orange/Red)
- **Interactive Legends**: Enhanced hover states and animations
- **Mobile Responsive**: Perfect rendering across all devices

### 🎯 **Professional UX Enhancements**
- **Smart Metric Classification**: 
  - LTV:CAC ratio: Good ≥3:1, Warning 1.5-3:1, Poor <1.5:1
  - Runway: Good ≥18 months, Warning 6-18 months, Critical <6 months
  - Risk Score: Intelligent scoring with color-coded indicators
- **Info Tooltips**: Informative (ℹ️) icons with detailed explanations
- **Pulsing Animations**: Visual feedback for important metrics
- **Enhanced Navigation**: Professional Flask routing structure

### 🏗️ **Professional Architecture**
- **Flask Backend**: Robust server with proper routing (`/`, `/calculator`, `/about`)
- **Landing Page Flow**: Professional user journey from landing → calculator
- **Asset Management**: Organized static file serving for optimal performance
- **Responsive Design**: Mobile-first approach with glassmorphism effects

---

## 🚀 Tech Stack:

### **Frontend**
- **HTML5/CSS3**: Modern semantic markup with advanced CSS features
- **JavaScript ES6+**: Dynamic interactions and animations
- **Chart.js**: Professional data visualizations
- **Font Awesome**: Premium icon library
- **Google Fonts**: Inter typography for modern aesthetics

### **Backend**
- **Python 3.8+**: Core application logic
- **Flask**: Lightweight web framework with routing
- **Financial Engine**: Custom calculation modules
- **AI Integration**: LLaMA 3 model support (offline)

### **Design System**
- **Glassmorphism**: Advanced backdrop-filter effects
- **Color Theory**: Vibrant multi-color palette with accessibility
- **Animation Framework**: CSS3 + JavaScript for 60fps performance
- **Responsive Grid**: Mobile-first responsive design

---

## 📸 Screenshots & Demo

### **Colorful Glassmorphism Interface**
```
🎨 Features vibrant charts with:
✅ Multi-color gradients (green, blue, orange, pink, purple)
✅ Glass transparency effects with backdrop blur
✅ Rainbow shimmer animations
✅ Enhanced shadows and glows
✅ Professional color-coded metrics
```

### **Dynamic Chart System**
```
📊 Real-time calculations:
✅ Revenue projections based on actual ARPU × users
✅ MRR growth with churn rate modeling
✅ Capital runway with burn rate analysis
✅ Risk scoring with multiple factors
✅ Interactive tooltips with explanations
```

### **Professional Animations**
```
🎬 Smooth 60fps animations:
✅ Staggered card entrances (0.1s delays)
✅ Number counting with easeOutQuart
✅ Chart rendering with 1.5s duration
✅ Hover effects with scale & glow
✅ Loading shimmer states
```

---

## 🛠️ Installation & Setup

### **Quick Start**
```bash
# Clone the repository
git clone https://github.com/Aaditya1273/FinPilot.git
cd FinPilot

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

### **Access the Application**
1. **Landing Page**: `http://127.0.0.1:5000/` - Professional welcome experience
2. **Calculator**: `http://127.0.0.1:5000/calculator` - Main financial modeling interface
3. **About**: `http://127.0.0.1:5000/about` - Project information

---

## 🎯 Usage Guide

### **Step 1: Input Financial Data**
- Enter your business metrics (users, ARPU, growth rate, etc.)
- All fields are validated and provide real-time feedback

### **Step 2: View Dynamic Charts**
- Charts automatically update based on your inputs
- Hover over elements for detailed tooltips
- Observe color-coded trend indicators

### **Step 3: Analyze Insights**
- Review calculated metrics with animated counters
- Understand risk vs growth positioning
- Plan based on runway and capital projections

---

## 🌟 Advanced Features

### **Financial Calculations**
- **MRR/ARR Projections**: 12-month growth modeling with churn
- **Burn Rate Analysis**: Capital depletion tracking
- **LTV/CAC Optimization**: Customer acquisition efficiency
- **Risk Scoring**: Multi-factor risk assessment
- **Revenue Distribution**: Realistic SaaS revenue splits

### **Visual Intelligence**
- **Trend Classification**: Automated performance categorization
- **Color Psychology**: Strategic use of colors for data interpretation
- **Animation Timing**: Carefully crafted timing for optimal UX
- **Responsive Breakpoints**: Perfect rendering across devices

### **Technical Excellence**
- **Performance Optimized**: 60fps animations with requestAnimationFrame
- **Memory Efficient**: Optimized chart rendering and data handling
- **Accessibility**: WCAG compliant with proper contrast ratios
- **SEO Ready**: Semantic HTML with proper meta tags

---



## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### **Development Setup**
```bash
# Fork the repository
# Clone your fork
git clone https://github.com/YOUR_USERNAME/FinPilot.git

# Create feature branch
git checkout -b feature/amazing-feature

# Make your changes
# Commit and push
git commit -m "Add amazing feature"
git push origin feature/amazing-feature

# Create Pull Request
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---






## 🙏 Acknowledgments

- **Chart.js** - For excellent charting capabilities
- **Flask** - For robust web framework
- **Font Awesome** - For beautiful icons
- **Google Fonts** - For typography excellence
- **CSS Glassmorphism** - For modern design inspiration

---

## ⭐ Show Your Support

If you found FinPilot helpful, please consider:
- ⭐ **Starring** this repository
- 🍴 **Forking** for your own projects
- 📢 **Sharing** with your network
- 🐛 **Reporting** issues or suggestions

---

<div align="center">

**Built with ❤️ and cutting-edge technology**

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Powered by Flask](https://img.shields.io/badge/Powered%20by-Flask-green.svg)](https://flask.palletsprojects.com/)
[![Charts by Chart.js](https://img.shields.io/badge/Charts%20by-Chart.js-ff6384.svg)](https://www.chartjs.org/)

</div>
