# 🏠 California House Value Prediction App

This Streamlit app predicts the **median house value** for California neighborhoods based on housing features. Users can input housing data and get an estimated house value using a pre-trained machine learning model.

## Features

- Predict California house prices using 8 key features:
  - House Age (`housing_median_age`)
  - Average Rooms (`AveRooms`)
  - Average Bedrooms (`AveBedrms`)
  - Population (`population`)
  - Average Occupancy (`AveOccup`)
  - Latitude (`latitude`)
  - Longitude (`longitude`)
  - Median Income (`median_income`)
- Simple interactive web interface built with Streamlit.
- Loads a pre-trained model for fast predictions.

## Getting Started

### Prerequisites

- Python 3.9+
- `pip` package manager

### Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd apps/house_price
