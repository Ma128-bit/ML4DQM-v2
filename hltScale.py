import sys, os, json
import plotly.graph_objects as go
import pandas as pd
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import glob
import Utilities.database as db

# === MODEL FUNCTIONS ===
def exp_mod(x, a, b):
    return a * (np.exp(b * x) - 1)

def exp_mod_fixed_b(x, a, b):
    return a * (np.exp(b * x) - 1)

# === Data Loading ===
def load_data(job_id = "O0PnL6Vg66dM"):
    all_chunks = glob.glob(db.get_job_info(job_id)["S0"])
    monitoring_elements = pd.DataFrame()
    for chunk in all_chunks:
        df = pd.read_parquet(chunk, columns=["prescale_name", "recorded_lumi_per_lumisection", "entries"])
        monitoring_elements = pd.concat([monitoring_elements, df], ignore_index=True)
        del df
    return monitoring_elements

# === Plot3D ===
def plot_3d(df, x="prescale_name", y="recorded_lumi_per_lumisection", z="entries"):
    # Estrai dati
    x = df[x]
    y = df[y]
    z = df[z]
    
    # Crea un grafico a dispersione 3D
    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=5,
            color=z,                # colore in base a 'entries'
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(title='Entries')
        )
    )])
    
    # Layout
    fig.update_layout(
        scene=dict(
            xaxis_title="",
            yaxis_title="",
            zaxis_title=""
        ),
        title='3D Interactive Monitoring Plot',
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    # Mostra il grafico (interattivo in Jupyter)
    fig.write_html("hltScalePlots/monitoring_plot.html")


# === FIT REFERENCE Prescale ===
def fit_reference(df, target_name, p0=(1, 1), b_fit_max=10000, plot=True):
    df_ref = df[df["prescale_name"] == target_name]
    x_ref = df_ref["recorded_lumi_per_lumisection"].to_numpy()
    y_ref = df_ref["entries"].to_numpy()

    mask_ref = (x_ref > 0) & (y_ref > 0)
    x_fit_ref = x_ref[mask_ref]
    y_fit_ref = y_ref[mask_ref]

    popt_ref, _ = curve_fit(exp_mod, x_fit_ref, y_fit_ref, p0=p0, maxfev=b_fit_max)
    a_ref, b_fixed = popt_ref

    if plot:
        x_model = np.linspace(0, 0.5, 500)
        y_model = exp_mod(x_model, a_ref, b_fixed)

        plt.figure(figsize=(10, 6))
        plt.plot(x_ref, y_ref, 'o', markersize=5, label="Data")
        plt.plot(x_model, y_model, '-', color='red',
                 label=f"Fit: a·(exp(b·x)-1)\na={a_ref:.2f}, b={b_fixed:.2f}")
        plt.xlim(0, 0.5)
        plt.xlabel("Luminosity")
        plt.ylabel("Entries")
        plt.title(f"{target_name} (fit a & b)")
        plt.grid(True)
        plt.legend()
        plt.savefig("hltScalePlots/"+target_name+".png", dpi=300, bbox_inches='tight')
        plt.close()

    return a_ref, b_fixed

# === FIT with fixed b ===
def fit_all_prescales(df, b_fixed, exclude=None, min_points=5, plot=True):
    if exclude is None:
        exclude = []

    results = {}

    for p in np.unique(df["prescale_name"]):
        if p in exclude:
            continue

        df_p = df[df["prescale_name"] == p]
        if len(df_p[df_p["recorded_lumi_per_lumisection"] > 0.0]) < min_points:
            continue

        x = df_p["recorded_lumi_per_lumisection"].to_numpy()
        y = df_p["entries"].to_numpy()

        mask = (x > 0) & (y > 0)
        x_fit = x[mask]
        y_fit = y[mask]

        try:
            model = lambda x, a: exp_mod_fixed_b(x, a, b_fixed)
            popt, _ = curve_fit(model, x_fit, y_fit, p0=(1,), maxfev=10000)
            a = popt[0]
            results[p] = {"a": a, "b": b_fixed}

            if plot:
                x_model = np.linspace(0, 0.5, 500)
                y_model = exp_mod_fixed_b(x_model, a, b_fixed)

                plt.figure(figsize=(10, 6))
                plt.plot(x, y, 'o', markersize=5, label="Data")
                plt.plot(x_model, y_model, '-', color='red',
                         label=f"Fit: a·(exp({b_fixed:.2f}·x)-1)\na={a:.2f}")
                plt.xlim(0, 0.5)
                plt.xlabel("Luminosity")
                plt.ylabel("Entries")
                plt.title(f"{p} (fit a, b fisso)")
                plt.grid(True)
                plt.legend()
                plt.savefig("hltScalePlots/"+p if p!="" else "hltScalePlots/_"+".png", dpi=300, bbox_inches='tight')
                plt.close()

        except RuntimeError:
            print(f"⚠️ Fit failed for prescale: {p}")
            continue

    return results

# === PLOT FINALE DI TUTTI I FIT ===
def plot_all_fits(fit_results, b_fixed, x_range=(0, 0.5)):
    x_model = np.linspace(*x_range, 500)
    plt.figure(figsize=(12, 8))

    for p, params in fit_results.items():
        a = params["a"]
        y_model = exp_mod_fixed_b(x_model, a, b_fixed)
        plt.plot(x_model, y_model, label=f"{p}: a={a:.2f}")

    plt.xlim(*x_range)
    plt.xlabel("Luminosity")
    plt.ylabel("Entries (fit)")
    plt.title(f"Exponential Fits with b fixed = {b_fixed:.2f}")
    plt.grid(True)
    plt.legend(fontsize="small", ncol=2, loc="best")
    plt.savefig("hltScalePlots/FitComparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
if __name__ == "__main__":
    os.makedirs("hltScalePlots", exist_ok=True)
    
    df = load_data()
    print(df.head())
    plot_3d(df)
    
    # 1. Fit del prescale di riferimento (es. "2p0E34")
    a_ref, b_fixed = fit_reference(df, "2p0E34", plot=True)
    
    # 2. Fit di tutti gli altri prescale (con b fisso)
    fit_results = fit_all_prescales(df, b_fixed, exclude=["2p0E34"], plot=True)
    
    # 3. Plot finale comparativo di tutti i fit
    plot_all_fits(fit_results, b_fixed)

    hlt_scale = {"2p0E34": np.float64(1.0)}
    for k in fit_results.keys():
        hlt_scale[k] = fit_results[k]["a"]/a_ref

    # Salva come file JSON
    with open("hltsclae.json", "w") as f:
        json.dump(hlt_scale, f)