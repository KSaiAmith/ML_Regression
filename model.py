import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_validate
import warnings

warnings.filterwarnings('ignore')

def optimize_and_predict(train_path, test_path, out_path, plot_out_path, max_degree, alphas, l1_ratios):
    print(f"--- Optimizing model for {train_path} ---")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop(columns=['y'])
    y_train = train_df['y']
    X_test = test_df[X_train.columns]
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    overall_best_mse = float('inf')
    best_params = {}
    best_model_obj = None
    
    plot_degrees = []
    plot_mses = []
    
    model_grid = [('LinearRegression', LinearRegression())]
    for alpha in alphas:
        model_grid.append((f'Ridge(alpha={alpha})', Ridge(alpha=alpha, random_state=42, max_iter=5000)))
        model_grid.append((f'Lasso(alpha={alpha})', Lasso(alpha=alpha, random_state=42, max_iter=5000)))
        for l1_ratio in l1_ratios:
            model_grid.append((f'ElasticNet(alpha={alpha}, l1_ratio={l1_ratio})', 
                               ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=5000)))
    
    scoring = {'mse': 'neg_mean_squared_error', 'r2': 'r2'}

    for degree in range(1, max_degree + 1):
        best_mse_for_deg = float('inf')
        best_r2_for_deg = -float('inf')
        best_model_name_for_deg = ""
        best_model_obj_for_deg = None
        
        for model_name, model in model_grid:
            pipeline = make_pipeline(
                PolynomialFeatures(degree=degree, include_bias=False),
                StandardScaler(),
                model
            )

            scores = cross_validate(pipeline, X_train, y_train, cv=kf, scoring=scoring, n_jobs=-1)
            mse = -scores['test_mse'].mean()
            r2 = scores['test_r2'].mean()
            
            if mse < best_mse_for_deg:
                best_mse_for_deg = mse
                best_r2_for_deg = r2
                best_model_name_for_deg = model_name
                best_model_obj_for_deg = model
                
            if mse < overall_best_mse:
                overall_best_mse = mse
                best_params = {
                    'degree': degree, 
                    'model_name': model_name,
                    'r2': r2
                }
                best_model_obj = model
        
        plot_degrees.append(degree)
        plot_mses.append(best_mse_for_deg)

        print(f"Degree {degree:2d} | Best Model: {best_model_name_for_deg:<35} | MSE: {best_mse_for_deg:.3f} | R2: {best_r2_for_deg:.4f}")

    print(f"\nOptimal values:")
    print(f"  - Best Model:  {best_params['model_name']}")
    print(f"  - Best Degree: {best_params['degree']}")
    print(f"  - Best CV MSE: {overall_best_mse:.3f}")
    print(f"  - Best CV R2:  {best_params['r2']:.4f}\n")
    
    plt.figure(figsize=(8, 5))
    plt.plot(plot_degrees, plot_mses, marker='o', linestyle='-', color='b')
    plt.title(f'Cross-Validation MSE vs. Polynomial Degree\n({train_path})')
    plt.xlabel('Polynomial Degree')
    plt.yscale('log')
    plt.ylabel('Cross-Validation MSE (Log Scale)')
    plt.xticks(range(1, max_degree + 1))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axvline(x=best_params['degree'], color='r', linestyle='--', label=f"Best Degree: {best_params['degree']}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_out_path)
    plt.close()


    final_model = make_pipeline(
        PolynomialFeatures(degree=best_params['degree'], include_bias=False),
        StandardScaler(),
        best_model_obj
    )
    final_model.fit(X_train, y_train)
    
    predictions = final_model.predict(X_test)
    pd.DataFrame({'y': predictions}).to_csv(out_path, index=False)
    print(f"Saved optimal predictions to {out_path}\n")

if __name__ == "__main__":
    alpha_grid = [0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0]
    l1_ratio_grid = [0.2, 0.5, 0.8] 
    
    optimize_and_predict(
        train_path="IMT2024004_train_var1.csv", 
        test_path="IMT2024004_test_var1.csv",
        out_path="IMT2024004_pred_var1.csv",
        plot_out_path="IMT2024004_plot_var1.png", 
        max_degree=10, 
        alphas=alpha_grid,
        l1_ratios=l1_ratio_grid
    )
    
    optimize_and_predict(
        train_path="IMT2024004_train_var2.csv",
        test_path="IMT2024004_test_var2.csv",
        out_path="IMT2024004_pred_var2.csv",
        plot_out_path="IMT2024004_plot_var2.png", 
        max_degree=20, 
        alphas=alpha_grid,
        l1_ratios=l1_ratio_grid
    )
