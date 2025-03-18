import matplotlib.pyplot as plt
import os

def plot_predictions(true, pred, title="Predictions vs True Temperature"):
    """
    Plots true vs. predicted values for the first predicted time step.
    
    Parameters:
        true (np.array): Array of true values with shape (n_samples, n_predictions).
        pred (np.array): Array of predicted values with shape (n_samples, n_predictions).
        title (str): Title for the plot.
        
    Note:
        The y-axis label is set to "T (degC)".
    """

    plt.figure(figsize=(12, 6))
    plt.plot(true[-120:], label='Real Temperatures /degC', alpha=0.7)
    plt.plot(pred[-120:], label='Predicted Temperature /degC', linestyle='--')
    plt.title("Next Hour Temperature Prediction")
    plt.xlabel("Hours")
    plt.ylabel("Temperature /degC")
    plt.legend()
    plt.show()

    # Ensure the 'image' directory exists
    image_dir = "output/plots"
    os.makedirs(image_dir, exist_ok=True)  
    # Define the full path to save the image
    save_path = os.path.join(image_dir, "temperature_prediction.png")
    # Save the image
    plt.savefig(save_path, dpi=300)  # High resolution

    plt.close()  # Close plot to free memory