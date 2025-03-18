# Weather Forecasting with LSTM

## 📌 Project Overview
This project aims to develop a deep learning model using **Long Short-Term Memory (LSTM)** networks for weather forecasting. The model is trained on the **Jena Climate Dataset**, which contains weather data collected from the Max Planck Institute for Biogeochemistry in Jena, Germany.

## 📊 Dataset Information
- **Source**: Max Planck Institute for Biogeochemistry
- **Location**: Weather Station, Jena, Germany
- **Time Frame Considered**: January 10, 2009 - December 31, 2016
- **Features**: The dataset consists of 14 weather-related features recorded every **10 minutes**, including:
  - Temperature (°C)
  - Pressure (hPa)
  - Humidity (%)
  - Wind speed (m/s)
  - Wind direction (°)
  - Other atmospheric conditions

## 📓 Notebook 
A compelete Tutorial with extensive data analysis is available in the Notebook folder.

## 🔥 Model Architecture
The weather forecasting model is built using **PyTorch** and is based on an **LSTM neural network** to capture temporal dependencies in time-series weather data.

## 🔧 Model Parameters
| Sequence Length = 24 
| Prediction Length = 1 
| Batch Size = 32 
| Hidden Size = 128 
| Number of LSTM Layers = 2 
| Dropout = 0.1 
| Learning Rate = 0.001 
| Training Epochs = 30 

## 📁 Project Structure
Project tree is available in project_structure.txt

## 🛠 Setup & Installation
### 1️⃣ **Clone the Repository**
```bash
git clone https://github.com/your-username/weather_forecasting.git
cd weather_forecasting
```

### 2️⃣ **Create and Activate Virtual Environment**
```bash
python -m venv myenv
source myenv/bin/activate  # On macOS/Linux
myenv\Scripts\activate    # On Windows
```

### 3️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 4️⃣ **Run the Training Script**
```bash
python main.py
```

## 📊 Logging
All training logs are automatically saved in `output/logs/app.log`. If the logs directory does not exist, it is created dynamically.

## 📜 License
This project is open-source and available under the **MIT License**.



