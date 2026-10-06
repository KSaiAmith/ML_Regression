import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore')

def load_data(filepath):
    """Loads data from CSV or Excel depending on the file extension."""
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    elif filepath.endswith(('.xls', '.xlsx')):
        return pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format for {filepath}. Use .csv or .xlsx")

def optimize_and_predict(train_path, test_path, out_path, max_degree, alphas):
    print(f"==================================================")
    print(f"--- Optimizing model for {train_path} ---")
    print(f"==================================================")
    
    train_df = load_data(train_path)
    test_df = load_data(test_path)
    
    X_train_full = train_df.drop(columns=['y'])
    y_train = train_df['y']
    features = list(X_train_full.columns)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    overall_best_mse = float('inf')
    overall_best_params = {}
    
    # Iterate through each degree to print its specific performance
    for degree in range(1, max_degree + 1):
        best_mse_for_deg = float('inf')
        best_params_for_deg = {}
        
        # Test all feature combinations and alphas for this specific degree
        for r in range(1, len(features) + 1):
            for subset in itertools.combinations(features, r):
                # Optimization guard: Skip extremely high dimensional spaces to prevent memory crashes
                if len(subset) > 3 and degree > 7:
                    continue
                    
                X_sub = X_train_full[list(subset)]
                
                for alpha in alphas:
                    model = make_pipeline(
                        PolynomialFeatures(degree=degree, include_bias=True),
                        StandardScaler(),
                        Ridge(alpha=alpha, random_state=42)
                    )

                    scores = cross_val_score(model, X_sub, y_train, cv=kf, scoring='neg_mean_squared_error')
                    mse = -scores.mean()
                    
                    if mse < best_mse_for_deg:
                        best_mse_for_deg = mse
                        best_params_for_deg = {
                            'features': list(subset), 
                            'degree': degree, 
                            'alpha': alpha
                        }
        
        # Now that we have the best parameters for this degree, 
        # let's train it once on the full dataset to get the R2 score
        best_features_for_deg = best_params_for_deg['features']
        deg_model = make_pipeline(
            PolynomialFeatures(degree=degree, include_bias=True),
            StandardScaler(),
            Ridge(alpha=best_params_for_deg['alpha'], random_state=42)
        )
        deg_model.fit(X_train_full[best_features_for_deg], y_train)
        train_predictions = deg_model.predict(X_train_full[best_features_for_deg])
        train_r2 = r2_score(y_train, train_predictions)
        
        # Print the results for this degree
        print(f"Degree {degree:2d} | Alpha: {best_params_for_deg['alpha']:<5.1f} | CV MSE: {best_mse_for_deg:.3f} | Train R2: {train_r2:.4f}")
        
        # Track the overall best across all degrees
        if best_mse_for_deg < overall_best_mse:
            overall_best_mse = best_mse_for_deg
            overall_best_params = best_params_for_deg

    print(f"\n>>> WINNING CONFIGURATION <<<")
    print(f"  - Best Features: {overall_best_params['features']}")
    print(f"  - Best Degree:   {overall_best_params['degree']}")
    print(f"  - Best Alpha:    {overall_best_params['alpha']}")
    print(f"  - Best CV MSE:   {overall_best_mse:.3f}\n")
    
    # Train the final overall optimal model to make test predictions
    X_train_opt = X_train_full[overall_best_params['features']]
    X_test_opt = test_df[overall_best_params['features']]
    
    final_model = make_pipeline(
        PolynomialFeatures(degree=overall_best_params['degree'], include_bias=True),
        StandardScaler(),
        Ridge(alpha=overall_best_params['alpha'], random_state=42)
    )
    final_model.fit(X_train_opt, y_train)
    
    predictions = final_model.predict(X_test_opt)
    pd.DataFrame({'y': predictions}).to_csv(out_path, index=False)
    print(f"Saved optimal predictions to {out_path}\n")

if __name__ == "__main__":
    
    alpha_grid = [0.1, 1.0, 3.0, 10.0, 30.0, 100.0]
    
    # Note: Remember to change the filenames to match your local files (e.g., IMT2024014)
    optimize_and_predict(
        train_path="./IMT2024004_train_var1.csv", 
        test_path="./IMT2024004_test_var1.csv",
        out_path="./IMT2024004_pred_var1.csv",
        max_degree=7, 
        alphas=alpha_grid
    )
    
    optimize_and_predict(
        train_path="./IMT2024004_train_var2.csv",
        test_path="./IMT2024004_test_var2.csv",
        out_path="./IMT2024004_pred_var2.csv",
        max_degree=14, 
        alphas=alpha_grid
    )