# CPU Usage Prediction

## Deployed Application

https://cpuusageprediction-reakzeyme3w8sysy3gin4k.streamlit.app

## Dashboard

https://cpuusageprediction-mxaalx88d2co7ffvgkqkjv.streamlit.app

## Project Overview

This project provides a CPU usage prediction model and a dashboard to visualize the predictions. The application is built using Streamlit, making it easy to deploy and interact with.

## Installation Instructions

To set up this project locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repo/cpu_usage_prediction.git
   cd cpu_usage_prediction
   ```

2. Create a virtual environment and activate it:

   ```bash
   python -m venv venv
   ./venv/Scripts/activate # On Windows
   source venv/bin/activate # On macOS/Linux
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt # Assuming a requirements.txt file exists
   ```

## Usage Guide

To run the Streamlit applications, use the following commands:

- For the main application:

  ```bash
  streamlit run dashboard/app.py
  ```

- For the metrics dashboard:
  ```bash
  streamlit run dashboard/metrics_app.py
  ```

## Configuration Details

This project uses a `.streamlit/config.toml` file for application-wide configurations, including theme settings. This file should be located in the root directory of the project.

## Customization Options

### Theme Customization

To change the theme of the Streamlit applications, you can modify the `.streamlit/config.toml` file. By default, the applications are configured to use a light theme.

Example `config.toml` for a light theme:

```toml
[theme]
base = "light"
# primaryColor = "#1E90FF"
# backgroundColor = "#FFFFFF"
# secondaryBackgroundColor = "#F0F0F0"
# textColor = "#000000"
# font = "sans serif"
```

You can uncomment and modify the color codes to further customize the look and feel of your applications.
