import sys
import os
import pandas as pd 
import numpy as np 
import logging 
import torch
from src.utils.fetch_data import load_data
from src.utils.data_processing import feature_eng, chronological_split, normalization, create_sequences, data_loader
from models.model_LSTM import model_LSTM
from src.utils.train_model import train_model
from src.utils.test_model import evaluate, inverse_transform
from src.visualization.viz import plot_predictions

# Get the absolute path of the project root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from config import SEQ_LENGTH, PRED_LENGTH, BATCH_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, LEARNING_RATE, EPOCHS
from config import LOG_DIR


# --- Configure logging ---
# Define log file path
log_file = os.path.join(LOG_DIR, "app.log")
# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[
                            logging.FileHandler(log_file),  # Save logs to file
                            logging.StreamHandler()  # Print logs to console
                        ])

logger = logging.getLogger(__name__)

def main():
    # get data
    df = load_data()
    logger.info(f"✅ Data loaded")

    #create new features
    df = feature_eng(df)
    logger.info(f"✅ new features created => new data-shape {df.shape}")

    # split data into train, val, test
    df_train, df_val, df_test = chronological_split(df)
    logger.info(f"✅ Data split into train, validation, and test sets created. shapes :{df_train.shape}, {df_val.shape}, {df_test.shape}")

    # normalizing data
    df_train, df_val, df_test = normalization(df_train, df_val, df_test)
    logger.info(f"✅ Data normalized")

    # creating sequences
    X_train_seq, y_train_seq = create_sequences(df_train, SEQ_LENGTH, PRED_LENGTH)
    X_val_seq, y_val_seq     = create_sequences(df_val, SEQ_LENGTH, PRED_LENGTH)
    X_test_seq, y_test_seq   = create_sequences(df_test, SEQ_LENGTH, PRED_LENGTH)
    logger.info(f"✅ Sequences created: Training")

    # creating dataloaders
    train_loader = data_loader(X_train_seq, y_train_seq, BATCH_SIZE)
    val_loader   = data_loader(X_val_seq, y_val_seq, BATCH_SIZE)
    test_loader  = data_loader(X_test_seq, y_test_seq, BATCH_SIZE)
    logger.info(f"✅ DataLoaders created")

    # BASELINE model
    from sklearn.metrics import mean_squared_error as MSE
    from sklearn.metrics import mean_absolute_error as MAE

    mse_error = MSE(X_train_seq[:,-1,1], y_train_seq)
    rmse = np.sqrt(mse_error)
    mae_error = MAE(X_train_seq[:,-1,1], y_train_seq)

    logger.info(f"📌 BASELINE MODEL : MAE = {np.round(mae_error,3)}")
    logger.info(f"📌 BASELINE MODEL : RMSE = {np.round(rmse, 3)}")

    # -------- MODEL LSTM ---------------
    #clear cache
    torch.mps.empty_cache()

    # Set device (GPU if available, else CPU)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # instantiate model
    model = model_LSTM(input_size = X_train_seq.shape[2], hidden_size = HIDDEN_SIZE, batch_first = True, num_layers = NUM_LAYERS, dropout = DROPOUT)

    # send model to device
    model.to(device)

    # optimizer and loss
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = LEARNING_RATE)

    # ----------- Training -----------------
    logger.info(f"🚀🚀🚀 Training for {EPOCHS} epochs...🚀🚀🚀")
    model = train_model(model, train_loader, val_loader, optimizer, criterion, device, EPOCHS)
    logger.info("✅ Training complete.")

    # ----------- Evaluation ----------------
    logger.info("Evaluating model...")
    predictions, truths = evaluate(model, test_loader, device, df_test['T (degC)'], window_size=30)
    logger.info("✅ Evaluation complete.")

    # Reverse normalization 
    preds = inverse_transform(predictions, df_test['T (degC)'], window_size=30)
    preds = preds.reshape(-1,)

    index = X_test_seq.shape[0]
    trues = df_test['T (degC)'].iloc[-index:].values

    # Predicitons
    today_pred = np.squeeze(preds[-2])
    tomorrow_pred = np.squeeze(preds[-1])
 
    logger.info(f" 🌡️ T (degC) @ Now = {trues[-1]:.2f}")
    logger.info(f"🌡️ Predicted T (degC) for Now = {today_pred:.2f}")
    logger.info(f"🤔 Predicted T (degC) Next Hour = {tomorrow_pred:.2f}")

    # METRICS 
    #------------ MAE --------------
    mae_error = MAE(preds, trues)
    logger.info(f"📌 MAE = {mae_error:.2f}")

    #------------ RMSE -----------
    mse_error = MSE(preds, trues)
    logger.info(f"📌 RMSE = {np.sqrt(mse_error):.2f}")

    # ---------- Visualization --------
    plot_predictions(trues, preds, title="Test Set: Predictions vs True Temperatures")


if __name__ == "__main__":
    main()
  