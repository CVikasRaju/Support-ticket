import pandas as pd
import os
from sklearn.model_selection import train_test_split

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_data(data_path: str = None) -> pd.DataFrame:
    """Load the customer support tickets dataset."""
    if data_path is None:
        data_path = os.path.join(ROOT_DIR, "data", "customer_support_tickets.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please download from Kaggle.")
    
    df = pd.read_csv(data_path)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare the dataset."""
    required_cols = ['Ticket Description', 'Ticket Type', 'Ticket Priority']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    df = df.dropna(subset=required_cols).copy()
    
    if 'Ticket Subject' in df.columns:
        df['text'] = df['Ticket Subject'].fillna('') + ' ' + df['Ticket Description'].fillna('')
    else:
        df['text'] = df['Ticket Description'].fillna('')
    
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split data into training and testing sets."""
    X = df['text']
    y_category = df['Ticket Type']
    y_priority = df['Ticket Priority']
    
    X_train, X_test, y_cat_train, y_cat_test, y_pri_train, y_pri_test = train_test_split(
        X, y_category, y_priority,
        test_size=test_size,
        random_state=random_state,
        stratify=y_category
    )
    
    return X_train, X_test, y_cat_train, y_cat_test, y_pri_train, y_pri_test


if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {len(df)} records")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nTicket Type distribution:\n{df['Ticket Type'].value_counts()}")
    print(f"\nTicket Priority distribution:\n{df['Ticket Priority'].value_counts()}")
    
    df = preprocess_data(df)
    print(f"\nAfter preprocessing: {len(df)} records")
    
    X_train, X_test, y_cat_train, y_cat_test, y_pri_train, y_pri_test = split_data(df)
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")