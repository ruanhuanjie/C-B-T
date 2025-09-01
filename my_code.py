import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
import shap

# Model hyper-parameters
MODEL_PARAMS ={
    'input_dim': 5,
    'embed_dim': 64,
    'hidden_dim': 64,
    'num_layers': 2,
    'nhead': 4,
    'num_transformer_layers': 2,
    'dropout': 0.01,
    'batch_size': 32,
    'num_epochs': 50,
    'learning_rate': 0.001,
    'sequence_length': 7,
    'lead_time': 1
}

def set_random_seed(seed=42):
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_random_seed(42)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

class TransformerLSTM(nn.Module):
    def __init__(self):
        super(TransformerLSTM, self).__init__()
        input_dim = MODEL_PARAMS['input_dim']
        embed_dim = MODEL_PARAMS['embed_dim']
        hidden_dim = MODEL_PARAMS['hidden_dim']
        num_layers = MODEL_PARAMS['num_layers']
        nhead = MODEL_PARAMS['nhead']
        num_transformer_layers = MODEL_PARAMS['num_transformer_layers']
        dropout = MODEL_PARAMS['dropout']

        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2)

        self.input_projection = nn.Linear(input_dim, embed_dim)

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=num_transformer_layers
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.relu = nn.ReLU()

    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_len, input_dim)
        x_permuted = x.permute(0, 2, 1)  # (batch_size, input_dim, seq_len)
        conv1_out = self.conv1(x_permuted)  # (batch_size, hidden_dim, seq_len)
        conv2_out = self.conv2(x_permuted)  # (batch_size, hidden_dim, seq_len)
        conv_out = (conv1_out + conv2_out).permute(0, 2, 1)  # (batch_size, seq_len, hidden_dim)

        x_proj = self.input_projection(x)  # (batch_size, seq_len, embed_dim)

        if mask is not None:
            transformer_out = self.transformer_encoder(x_proj, src_key_padding_mask=mask)
        else:
            transformer_out = self.transformer_encoder(x_proj)

        transformer_out = transformer_out + x_proj

        lstm_out, (hidden, cell) = self.lstm(transformer_out)

        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)

        combined = lstm_out + attn_output
        combined = self.layer_norm(combined)

        last_hidden = combined[:, -1, :]
        last_hidden = self.dropout(last_hidden)

        out = self.fc1(last_hidden)
        out = self.relu(out)
        out = self.dropout(out)
        output = self.fc2(out)

        return output

class DataProcessor:
    def __init__(self, sequence_length=MODEL_PARAMS['sequence_length'], random_state=42):
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.random_state = random_state

    def remove_outliers(self, data, threshold=3):
        """Remove outliers using z-score."""
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        return np.where(z_scores < threshold, data, np.nan)

    def handle_missing_values(self, data):
        """Interpolate missing values."""
        return pd.DataFrame(data).interpolate(method='cubic').values

    def add_time_features(self, df):
        """Add sine/cosine time features."""
        df['month'] = pd.to_datetime(df.index).month
        df['day'] = pd.to_datetime(df.index).day
        df['day_of_week'] = pd.to_datetime(df.index).dayofweek
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        return df

    def create_sequences(self, data):
        """Create sliding-window sequences and targets."""
        lead_time = MODEL_PARAMS.get('lead_time', 1)
        sequences = []
        targets = []
        for i in range(len(data) - self.sequence_length - lead_time + 1):
            seq = data[i:i + self.sequence_length]
            target = data[i + self.sequence_length + lead_time - 1, 0]
            sequences.append(seq)
            targets.append(target)
        if len(sequences) == 0:
            raise ValueError(
                f"Insufficient data for sequence_length={self.sequence_length} "
                f"and lead_time={lead_time}. "
                f"Please reduce sequence_length or lead_time, or provide more data."
            )
        return np.array(sequences), np.array(targets)

    def prepare_data(self, df):
        """Full data pipeline: clean, scale, split."""
        features = ["GWL","PRCP", "Discharge", "T","Cond"]
        data = df[features].values
        for i in range(data.shape[1]):
            data[:, i] = self.remove_outliers(data[:, i])
        data = self.handle_missing_values(data)
        if isinstance(df.index, pd.DatetimeIndex):
            df_with_time = self.add_time_features(df)
            time_features = ['month_sin', 'month_cos']
            time_data = df_with_time[time_features].values
            data = np.hstack([data, time_data])
        data_scaled = self.scaler.fit_transform(data)
        X, y = self.create_sequences(data_scaled)
        if X.shape[0] == 0 or y.shape[0] == 0:
            raise ValueError(
                f"Insufficient data for sequence_length={self.sequence_length}."
            )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=self.random_state,
            shuffle=True
        )
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
        y_train = torch.FloatTensor(y_train)
        y_test = torch.FloatTensor(y_test)
        return X_train, X_test, y_train, y_test

    def inverse_transform_gwl(self, scaled_data):
        """Invert scaling for groundwater level (first feature)."""
        dummy = np.zeros((scaled_data.shape[0], len(self.scaler.mean_)))
        dummy[:, 0] = scaled_data
        inverse_transformed = self.scaler.inverse_transform(dummy)
        return inverse_transformed[:, 0]

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """Train model with early stopping and LR scheduler."""
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    best_model_state = None
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.view(-1), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs.view(-1), batch_y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f'Early stopping triggered at epoch {epoch+1}')
            model.load_state_dict(best_model_state)
            break
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2, marker='o', markersize=3)
    plt.plot(val_losses, label='Validation Loss', linewidth=2, marker='o', markersize=3)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training & Validation Loss', pad=10, fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    return train_losses, val_losses

def calculate_metrics(actuals, predictions):
    """Compute RMSE, MAE, NSE, and KGE."""
    mse = np.mean((actuals - predictions) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actuals - predictions))
    nse = 1 - np.sum((actuals - predictions) ** 2) / np.sum((actuals - np.mean(actuals)) ** 2)
    r = np.corrcoef(actuals, predictions)[0, 1]
    beta = np.mean(predictions) / np.mean(actuals)
    gamma = (np.std(predictions) / np.mean(predictions)) / (np.std(actuals) / np.mean(actuals))
    kge = 1 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)
    return {
        'RMSE': rmse,
        'MAE': mae,
        'NSE': nse,
        'KGE': kge
    }

def evaluate_model(model, train_loader, test_loader, device, data_processor):
    """Evaluate model and plot results."""
    model.eval()
    def get_predictions(loader):
        predictions = []
        actuals = []
        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                predictions.extend(outputs.cpu().numpy().reshape(-1))
                actuals.extend(batch_y.numpy().reshape(-1))
        return np.array(predictions), np.array(actuals)
    train_pred, train_actual = get_predictions(train_loader)
    test_pred, test_actual = get_predictions(test_loader)
    train_pred = data_processor.inverse_transform_gwl(train_pred)
    train_actual = data_processor.inverse_transform_gwl(train_actual)
    test_pred = data_processor.inverse_transform_gwl(test_pred)
    test_actual = data_processor.inverse_transform_gwl(test_actual)
    train_metrics = calculate_metrics(train_actual, train_pred)
    test_metrics = calculate_metrics(test_actual, test_pred)
    fig = plt.figure(figsize=(15, 10))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(train_actual, label='Actual', alpha=0.7, linewidth=2)
    ax1.plot(train_pred, label='Predicted', alpha=0.7, linewidth=2)
    ax1.set_xlabel('Timestep', fontsize=12)
    ax1.set_ylabel('Groundwater Level (m)', fontsize=12)
    ax1.set_title('Training Set: Actual vs Predicted', pad=10, fontsize=14)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(test_actual, label='Actual', alpha=0.7, linewidth=2)
    ax2.plot(test_pred, label='Predicted', alpha=0.7, linewidth=2)
    ax2.set_xlabel('Timestep', fontsize=12)
    ax2.set_ylabel('Groundwater Level (m)', fontsize=12)
    ax2.set_title('Test Set: Actual vs Predicted', pad=10, fontsize=14)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax3 = plt.subplot(2, 2, 3)
    ax3.scatter(train_actual, train_pred, alpha=0.5, c='blue')
    ax3.plot([min(train_actual), max(train_actual)],
             [min(train_actual), max(train_actual)],
             'r--', label='1:1 line')
    ax3.set_xlabel('Actual Groundwater Level (m)', fontsize=12)
    ax3.set_ylabel('Predicted Groundwater Level (m)', fontsize=12)
    ax3.set_title('Training Set: Scatter Plot', pad=10, fontsize=14)
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax4 = plt.subplot(2, 2, 4)
    ax4.scatter(test_actual, test_pred, alpha=0.5, c='blue')
    ax4.plot([min(test_actual), max(test_actual)],
             [min(test_actual), max(test_actual)],
             'r--', label='1:1 line')
    ax4.set_xlabel('Actual Groundwater Level (m)', fontsize=12)
    ax4.set_ylabel('Predicted Groundwater Level (m)', fontsize=12)
    ax4.set_title('Test Set: Scatter Plot', pad=10, fontsize=14)
    ax4.legend(loc='best', fontsize=10)
    ax4.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    return {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics
    }

def shap_analysis(model, X_test, feature_names, device, data_processor):
    """SHAP analysis with KernelExplainer."""
    model.eval()

    # Prepare data
    n_samples = X_test.shape[0]  # use all samples
    # Ensure 2-D data
    if len(X_test.shape) == 3:
        n_features = X_test.shape[1] * X_test.shape[2]
        X_sample = X_test[:n_samples].reshape(n_samples, -1).cpu().numpy()
    else:
        X_sample = X_test[:n_samples].cpu().numpy()

    # Model wrapper
    def f(X):
        with torch.no_grad():
            if len(X.shape) == 2:
                X = X.reshape(X.shape[0], X_test.shape[1], -1)
            X_tensor = torch.FloatTensor(X).to(device)
            return model(X_tensor).cpu().numpy()

    # Background data
    background = shap.sample(X_sample, nsamples=50)

    try:
        # Explainer
        print("Initializing SHAP explainer...")
        explainer = shap.KernelExplainer(f, background)

        # Compute SHAP values
        print("Computing SHAP values (this may take a few minutes)...")
        shap_values = explainer.shap_values(X_sample, nsamples=100)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # Reset matplotlib style
        plt.style.use('default')

        # Summary plot
        plt.figure(figsize=(14, 8))
        shap.summary_plot(
            shap_values,
            X_sample,
            feature_names=feature_names,
            show=False,
            plot_size=(12, 6),
            max_display=20,
            color_bar_label='Feature value'
        )
        plt.gcf().texts = []  # remove default title
        ax = plt.gca()
        ax.set_title('SHAP Feature Importance', pad=20, fontsize=16, fontweight='bold')
        ax.set_xlabel('SHAP value (impact on model output)', fontsize=12)
        plt.tight_layout()
        plt.show()

        # Bar plot
        plt.figure(figsize=(14, 8))
        shap.summary_plot(
            shap_values,
            X_sample,
            feature_names=feature_names,
            plot_type="bar",
            show=False,
            plot_size=(12, 6),
            max_display=20,
            color='#2E86C1'
        )
        plt.gcf().texts = []
        ax = plt.gca()
        ax.set_title('SHAP Feature Importance (Bar)', pad=20, fontsize=16, fontweight='bold')
        ax.set_xlabel('mean(|SHAP value|)', fontsize=12)
        plt.tight_layout()
        plt.show()

        # Average SHAP per variable across timesteps
        n_timesteps = X_test.shape[1]
        n_features = 5  # GWL, PRCP, Discharge, T, Cond
        avg_shap_values = np.zeros((X_sample.shape[0], n_features))
        feature_groups = [
            "Ground Water Level",
            "Precipitation",
            "Discharge",
            "Water Temperature",
            "Conductivity"
        ]
        for i, feature in enumerate(feature_groups):
            feature_indices = [j for j, name in enumerate(feature_names) if feature in name]
            avg_shap_values[:, i] = np.mean(shap_values[:, feature_indices], axis=1)

        # Save paths
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        shap_summary_path = os.path.join(desktop_path, "shap_summary_plot.png")
        shap_bar_path = os.path.join(desktop_path, "shap_bar_plot.png")

        # Average summary plot
        plt.figure(figsize=(14, 8))
        shap.summary_plot(
            avg_shap_values,
            X_sample[:, :n_features],
            feature_names=feature_groups,
            show=False,
            plot_size=(12, 6),
            color_bar_label='Feature value'
        )
        plt.gcf().texts = []
        ax = plt.gca()
        ax.set_xlabel('SHAP value (impact on model output)', fontsize=12)
        plt.savefig(shap_summary_path, dpi=300)
        plt.tight_layout()
        plt.show()

        # Horizontal bar chart
        plt.figure(figsize=(10, 6))
        avg_abs_shap_values = np.mean(np.abs(avg_shap_values), axis=0)
        order = np.argsort(avg_abs_shap_values)[::-1]
        y_pos = np.arange(len(feature_groups))
        vals = avg_abs_shap_values[order]
        labs = [feature_groups[i] for i in order]

        plt.barh(y_pos, vals, color='#2E86C1')
        plt.yticks(y_pos, labs)
        plt.xlabel('mean(|SHAP value|)', fontsize=12)
        plt.gca().invert_yaxis()

        for spine in ['right', 'top']:
            plt.gca().spines[spine].set_visible(False)
        plt.gca().spines['left'].set_color('lightgray')
        plt.gca().spines['left'].set_linewidth(1)

        summary_text = " | ".join([
            f"{feature_groups[i]}: {avg_abs_shap_values[i]:.4f}" for i in range(len(feature_groups))
        ])
        plt.figtext(0.5, -0.1, summary_text, wrap=True, horizontalalignment='center', fontsize=10)

        plt.tight_layout()
        plt.savefig(shap_bar_path, dpi=300)
        plt.show()

        # Return sorted importance
        feature_importance = np.abs(shap_values).mean(0)
        importance_dict = dict(zip(feature_names, feature_importance))
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

        return sorted_importance

    except Exception as e:
        print(f"Error during SHAP analysis: {str(e)}")
        return None

def perform_shap_analysis(model, model_data):
    """Wrapper to run SHAP analysis."""
    if model is None or model_data is None:
        print("Error: model or data invalid, cannot perform SHAP analysis")
        return None

    X_test, data_processor, sequence_length, device = model_data

    print("\nStarting SHAP analysis...")
    print("Note: SHAP analysis may take a few minutes...")

    feature_names = []
    for t in range(sequence_length):
        for f in [
            "Ground Water Level",
            "Precipitation",
            "Discharge",
            "Water Temperature",
            "Conductivity"
        ]:
            feature_names.append(f"{f}(t-{sequence_length-t})")

    feature_importance = shap_analysis(model, X_test, feature_names, device, data_processor)

    if feature_importance is not None:
        print("\nSHAP feature importance ranking:")
        for feature, importance in feature_importance:
            print(f"{feature}: {importance:.4f}")

    return feature_importance

def load_and_train_model(csv_path, random_seed=42):
    """Main pipeline: load data, train, evaluate."""
    try:
        set_random_seed(random_seed)
        print("Loading data...")
        df = pd.read_csv(csv_path)
        required_columns = ["GWL","PRCP", "Discharge", "T","Cond"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"CSV missing columns: {', '.join(missing_columns)}")
        print("Preparing data...")
        data_processor = DataProcessor(
            sequence_length=MODEL_PARAMS['sequence_length'],
            random_state=random_seed
        )
        X_train, X_test, y_train, y_test = data_processor.prepare_data(df)
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=MODEL_PARAMS['batch_size'],
            shuffle=True,
            generator=torch.Generator().manual_seed(random_seed)
        )
        val_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=MODEL_PARAMS['batch_size'],
            shuffle=False
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        print("Initializing model...")
        torch.manual_seed(random_seed)
        model = TransformerLSTM().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=MODEL_PARAMS['learning_rate'])
        print("Training model...")
        train_losses, val_losses = train_model(
            model, train_loader, val_loader, criterion, optimizer,
            MODEL_PARAMS['num_epochs'], device
        )
        print("Evaluating model...")
        metrics = evaluate_model(model, train_loader, val_loader, device, data_processor)
        print("\nTraining set metrics:")
        print(f"RMSE: {metrics['train_metrics']['RMSE']:.4f}")
        print(f"MAE: {metrics['train_metrics']['MAE']:.4f}")
        print(f"NSE: {metrics['train_metrics']['NSE']:.4f}")
        print(f"KGE: {metrics['train_metrics']['KGE']:.4f}")
        print("\nTest set metrics:")
        print(f"RMSE: {metrics['test_metrics']['RMSE']:.4f}")
        print(f"MAE: {metrics['test_metrics']['MAE']:.4f}")
        print(f"NSE: {metrics['test_metrics']['NSE']:.4f}")
        print(f"KGE: {metrics['test_metrics']['KGE']:.4f}")
        print("\nTraining complete!")
        return model, metrics, (X_test, data_processor, MODEL_PARAMS['sequence_length'], device)
    except FileNotFoundError:
        print(f"Error: file {csv_path} not found")
        return None, None, None
    except pd.errors.EmptyDataError:
        print("Error: CSV file is empty")
        return None, None, None
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return None, None, None

if __name__ == "__main__":
    csv_file_path = "Homodata.csv"
    print("Step 1: Model training and evaluation")
    model, metrics, model_data = load_and_train_model(csv_file_path, random_seed=42)
    print("\nStep 2: SHAP analysis")
    feature_importance = perform_shap_analysis(model, model_data)