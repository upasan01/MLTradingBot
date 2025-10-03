import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- 1. Data Preparation and Feature Engineering ---

def create_mock_data(periods=1000):
    """Generates synthetic stock price data for demonstration."""
    np.random.seed(42)
    # Generate a random walk series to simulate prices
    prices = 100 + np.cumsum(np.random.normal(0, 1, periods))
    dates = pd.date_range(start='2020-01-01', periods=periods, freq='D')
    
    df = pd.DataFrame({'Close': prices}, index=dates)
    
    # Create features: Simple Moving Averages (SMAs)
    # Short Window (Fast Line)
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    # Long Window (Slow Line)
    df['SMA_30'] = df['Close'].rolling(window=30).mean()
    
    # Drop initial NaN values created by rolling window
    df.dropna(inplace=True)
    return df

def generate_target_variable(df):
    """
    Generates the Target (Y) variable based on the next day's price movement.
    0: Sell (Price goes down/stagnant)
    1: Buy (Price goes up)
    
    In a real bot, this defines what the model is trying to predict.
    """
    # Calculate the percentage change in price 1 day later
    df['Price_Next_Day'] = df['Close'].shift(-1)
    df['Price_Change'] = df['Price_Next_Day'] - df['Close']
    
    # 1 if price went up, 0 otherwise (our target for the ML model)
    df['Signal'] = np.where(df['Price_Change'] > 0, 1, 0)
    
    # Drop the last row as we can't calculate 'Price_Next_Day' for it
    df.dropna(inplace=True)
    return df

# --- 2. Machine Learning Model Training ---

def train_knn_model(df):
    """Trains a K-Nearest Neighbors classifier."""
    
    # Features (X): The two moving averages
    X = df[['SMA_10', 'SMA_30']]
    # Target (Y): The desired signal (1=Buy, 0=Sell)
    Y = df['Signal']

    # Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    # Initialize the KNN Classifier
    # KNN is a simple classification model to start with
    model = KNeighborsClassifier(n_neighbors=5)

    # Train the model
    model.fit(X_train, Y_train)

    # Evaluate performance (important step for any ML model)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(Y_test, predictions)
    
    print(f"--- Model Training Results ---")
    print(f"KNN Classifier Accuracy on Test Data: {accuracy:.2f}")
    print("------------------------------")

    return model

# --- 3. Prediction and Trading Strategy Execution ---

def generate_trading_signals(df, model):
    """Uses the trained model to generate trading signals on the entire dataset."""
    
    # Features to feed the model
    X = df[['SMA_10', 'SMA_30']]
    
    # Predict the signal (1=Buy, 0=Sell)
    df['Predicted_Signal'] = model.predict(X)
    
    return df

# --- Main Execution ---

if __name__ == "__main__":
    # 1. Prepare Data
    data = create_mock_data(periods=1500)
    data_with_target = generate_target_variable(data.copy())
    
    # 2. Train Model
    knn_model = train_knn_model(data_with_target)
    
    # 3. Generate Trading Signals
    trading_data = generate_trading_signals(data_with_target.copy(), knn_model)
    
    # 4. Display Results (Inspect the end of the data to see signals)
    print("\n--- Predicted Trading Signals (Last 5 Days) ---")
    print(trading_data[['Close', 'SMA_10', 'SMA_30', 'Signal', 'Predicted_Signal']].tail(5))
    
    # Interpretation:
    # 'Signal' is the *actual* historical outcome (what the model was aiming for)
    # 'Predicted_Signal' is the *model's decision* (1: BUY, 0: SELL/HOLD)

    print("\n[ML Trading Bot Complete: A 'Predicted_Signal' of 1 suggests a BUY for the next day.]")
