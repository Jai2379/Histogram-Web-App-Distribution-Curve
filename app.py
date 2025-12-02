#Student Name: Jaiaditya Asudani
#Student ID: 21182653
# Hisotgram andn Distribution Curve App

#import neccesary libs
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from math import ceil, log2
import io

#page config
st.set_page_config(page_title="Histogram & Distribution Fitting", layout="wide")
#session state initialization
if 'data' not in st.session_state:
    st.session_state.data = None
if 'data_weights' not in st.session_state:
    st.session_state.data_weights = None
if 'custom_bins' not in st.session_state:
    st.session_state.custom_bins = None
if 'histogram_params' not in st.session_state:
    st.session_state.histogram_params = {
        'title': 'Histogram',
        'xlabel': 'Value',
        'ylabel': 'Frequency'
    }

#distribution dictionaires
DISTRIBUTIONS = {
    'Normal': stats.norm,
    'Gamma': stats.gamma,
    'Weibull': stats.weibull_min,
    'Exponential': stats.expon,
    'Lognormal': stats.lognorm,
    'Beta': stats.beta,
    'Chi-Square': stats.chi2,
    'Cauchy': stats.cauchy,
    'Uniform': stats.uniform,
    'Rayleigh': stats.rayleigh,
    'Triangular': stats.triang
}

def store_loaded_data(data, weights=None, custom_bins=None, source_label="raw"):
    st.session_state.data = data
    st.session_state.data_weights = (
        None if weights is None else np.asarray(weights, dtype=float)
    )
    if custom_bins is None:
        st.session_state.custom_bins = None
    else:
        st.session_state.custom_bins = np.unique(np.array(custom_bins, dtype=float))
    st.session_state.data_source = source_label

def _prepare_frequency_arrays(prim_vals, freqs):
    try:
        prim_ar = np.asarray(prim_vals, dtype=float)
        freq_ar = np.asarray(freqs, dtype=float)
    except ValueError:
        return None, None, "Values and freqs must be numeric."
    
    if prim_ar.size == 0 or freq_ar.size == 0:
        return None, None, "No values or freqs detected."
    if prim_ar.size != freq_ar.size:
        return None, None, "Each value must have a matching frequency."
    
    if np.any(freq_ar < 0):
        return None, None, "freqs must be non-negative."
    
    positive_mask = freq_ar > 0
    if not np.any(positive_mask):
        return None, None, "Need at least one positive frequency."
    
    prim_ar = prim_ar[positive_mask]
    freq_ar = freq_ar[positive_mask]
    return prim_ar, freq_ar, None

def _determine_replication_counts(freq_ar):
    total_freq = np.sum(freq_ar)
    if total_freq <= 0:
        return None, "Sum of freqs must be positive."
    
    min_total_points = max(500, len(freq_ar) * 20)
    max_total_points = 20000
    desired_total = int(np.clip(total_freq, min_total_points, max_total_points))
    
    proportions = freq_ar / total_freq
    counts = np.maximum(1, np.round(proportions * desired_total).astype(int))
    if np.sum(counts) < 2:
        return None, "Need at least two synthesized samples."
    return counts, None

def expand_range_weighted_samples(min_vals, max_vals, freqs):
    try:
        min_arr = np.asarray(min_vals, dtype=float)
        max_arr = np.asarray(max_vals, dtype=float)
        freq_ar = np.asarray(freqs, dtype=float)
    except ValueError:
        return None, "Ranges and freqs must be numeric."
    if not (len(min_arr) and len(max_arr) and len(freq_ar)):
        return None, "No range rows detected."
    if not (len(min_arr) == len(max_arr) == len(freq_ar)):
        return None, "Each row must include min, max, and frequency."
    
    valid_mask = (~np.isnan(min_arr)) & (~np.isnan(max_arr)) & (~np.isnan(freq_ar))
    min_arr = min_arr[valid_mask]
    max_arr = max_arr[valid_mask]
    freq_ar = freq_ar[valid_mask]
    
    if min_arr.size == 0:
        return None, "Please fill in at least one complete row."
    if np.any(max_arr <= min_arr):
        return None, "Each max value must be greater than its min value."
    if np.any(freq_ar < 0):
        return None, "freqs must be non-negative."
    positive_mask = freq_ar > 0
    if not np.any(positive_mask):
        return None, "Need at least one positive frequency."
    min_arr = min_arr[positive_mask]
    max_arr = max_arr[positive_mask]
    freq_ar = freq_ar[positive_mask]
    counts, error = _determine_replication_counts(freq_ar)
    if error:
        return None, None, error
    synthesized = []
    weight_chunks = []
    for min_val, max_val, freq, count in zip(min_arr, max_arr, freq_ar, counts):
        if count <= 0:
            continue
        if count == 1:
            synthesized.append(np.array([(min_val + max_val) / 2]))
        else:
            synthesized.append(np.linspace(min_val, max_val, count, endpoint=True))
        weight_chunks.append(np.full(count, freq / max(count, 1)))
    if not synthesized:
        return None, None, "Unable to synthesize samples from provided ranges."
    data = np.concatenate(synthesized)
    weights = np.concatenate(weight_chunks)
    if data.size < 2:
        return None, None, "Need at least two synthesized samples for the histogram."
    return data, weights, None

def validate_csv_data(input_data):
    try:
        if isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            df = pd.read_csv(io.StringIO(input_data))
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return None, None, None, "CSV must contain at least one numeric column."
        def find_column_by_keywords(columns, keywords):
            for col in columns:
                lowered = col.lower()
                if any(keyword in lowered for keyword in keywords):
                    return col
            return None
        
        lower_col = find_column_by_keywords(numeric_cols, ['min', 'lower', 'start', 'from', 'lo'])
        upper_col = find_column_by_keywords(numeric_cols, ['max', 'upper', 'end', 'to', 'hi'])
        freq_col = find_column_by_keywords(numeric_cols, ['freq', 'frequency', 'count', 'prob', 'density'])
        
        #prefer explicit range columns if available
        if lower_col and upper_col and freq_col:
            subset = df[[lower_col, upper_col, freq_col]].dropna()
            if subset.empty:
                return None, None, None, "CSV range columns are empty."
            data, weights, error = expand_range_weighted_samples(
                subset.iloc[:, 0].values,
                subset.iloc[:, 1].values,
                subset.iloc[:, 2].values
            )
            if not error:
                custom_bins = np.sort(np.unique(np.concatenate([
                    subset.iloc[:, 0].values,
                    subset.iloc[:, 1].values
                ])))
                return data, weights, custom_bins, None
        elif len(numeric_cols) >= 3:
            subset = df[numeric_cols[:3]].dropna()
            if subset.empty:
                return None, None, None, "CSV missing complete range rows."
            data, weights, error = expand_range_weighted_samples(subset.iloc[:, 0].values,
                subset.iloc[:, 1].values,
                subset.iloc[:, 2].values)
            if not error:
                custom_bins = np.sort(np.unique(np.concatenate([
                    subset.iloc[:, 0].values,
                    subset.iloc[:, 1].values
                ])))
                return data, weights, custom_bins, None
        
        return None, None, None, "CSV must have range columns (min, max, frequency)"
        
    except Exception as e:
        return None, None, None, f"Error processing CSV: {str(e)}"
#sturges rule to calculate bins
def calculate_bins(data):
    n = len(data)
    if n == 0:
        return 1
    bins = int(ceil(log2(n))) + 1
    return max(1, bins)

def calculate_axis_limits(data):
    raw_min = np.min(data)
    raw_max = np.max(data)
    if raw_max == raw_min:
        padding = 1.0
    else:
        padding = 0.13 * (raw_max - raw_min)
    xmin = raw_min - padding
    xmax = raw_max + padding
    return xmin, xmax

def fit_distribution(dist_obj, dist_name, data):
    try:
        if dist_name == 'Beta':
            data_min = np.min(data)
            data_max = np.max(data)
            data_range = data_max - data_min
            if data_range == 0:
                return None, "Cannot fit Beta distribution to constant data"
            data_norm = (data - data_min) / data_range
            data_norm = np.clip(data_norm, 1e-6, 1 - 1e-6)
            params = dist_obj.fit(data_norm, floc=0, fscale=1)
            st.session_state.beta_normalization = (data_min, data_range)
            return params, None
        elif dist_name == 'Uniform':
            data_min = np.min(data)
            data_max = np.max(data)
            if data_max == data_min:
                return None, "Cannot fit Uniform distribution to constant data"
            params = (data_min, data_max - data_min)
            return params, None
        elif dist_name == 'Lognormal':
            if np.any(data <= 0):
                return None, "Lognormal requires positive data values"
            params = dist_obj.fit(data, floc=0)
            return params, None
        elif dist_name == 'Exponential':
            params = dist_obj.fit(data, floc=np.min(data))
            return params, None
        elif dist_name == 'Rayleigh':
            params = dist_obj.fit(data, floc=np.min(data))
            return params, None
        elif dist_name == 'Chi-Square':
            if np.any(data < 0):
                data = np.abs(data)
            params = dist_obj.fit(data)
            return params, None
        elif dist_name == 'Weibull':
            if np.any(data < 0):
                data = data - np.min(data) + 0.01
            params = dist_obj.fit(data, floc=0)
            return params, None
        else:
            params = dist_obj.fit(data)
            return params, None
    except Exception as e:
        return None, f"Fitting error: {str(e)}"

def get_parameter_names(dist_name):
    """Get parameter names for a distribution."""
    if dist_name == 'Uniform':
        return ['loc', 'scale']
    elif dist_name == 'Lognormal':
        return ['s (shape)', 'loc', 'scale']
    elif dist_name == 'Gamma':
        return ['a (shape)', 'loc', 'scale']
    elif dist_name == 'Weibull':
        return ['c (shape)', 'loc', 'scale']
    elif dist_name == 'Triangular':
        return ['c (shape)', 'loc', 'scale']
    elif dist_name in ['Normal', 'Exponential', 'Rayleigh', 'Cauchy']:
        return ['loc', 'scale']
    else:
        return ['shape', 'loc', 'scale']

def get_slider_range(param_name, fitted_value, data):
    if param_name in ['loc', 'location']:
        data_range = np.max(data) - np.min(data)
        return (np.min(data) - data_range, np.max(data) + data_range)
    elif param_name in ['scale']:
        base = max(abs(fitted_value), 0.1)
        return (max(0.01, base * 0.1), base * 10)
    elif param_name in ['shape', 's (shape)', 'a (shape)', 'c (shape)']:
        base = max(abs(fitted_value), 0.1)
        return (0.1, base * 5)
    else:
        base = abs(fitted_value) if fitted_value != 0 else 1.0
        return (base * 0.1, base * 10)

def plot_histogram(data, mode, selected_dist=None, fitted_params=None, manual_params=None, weights=None, custom_bins=None):
    params = st.session_state.histogram_params
    
    if custom_bins is not None and len(custom_bins) >= 2:
        bins = custom_bins
    else:
        bins = calculate_bins(data)
    
    xmin, xmax = calculate_axis_limits(data)
    x_vals = np.linspace(xmin, xmax, 400)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
#histogram coloring
    n, bin_edges, patches = ax.hist(
        data,
        bins=bins,
        density=False,
        weights=weights,
        color='#3498db',
        alpha=0.7,
        edgecolor='black',
        linewidth=0.5)
    bin_widths = np.diff(bin_edges)
    avg_bin_width = np.mean(bin_widths) if len(bin_widths) > 0 else 1.0
    total_weight = float(np.sum(weights)) if weights is not None else float(len(data))
    curve_scale = total_weight * (avg_bin_width if avg_bin_width > 0 else 1.0)
    
#overlay distribution curve if in auto fit or manual mode
    if mode in ['Auto Fit Distribution', 'Manual Fitting'] and selected_dist:
        dist_obj = DISTRIBUTIONS[selected_dist]
        
#special distribution handling systenm
        if selected_dist == 'Beta':
            if 'beta_normalization' in st.session_state:
                data_min, data_range = st.session_state.beta_normalization
                x_vals_norm = (x_vals - data_min) / (data_range + 1e-10)
                x_vals_norm = np.clip(x_vals_norm, 1e-6, 1 - 1e-6)
                if mode == 'Auto Fit Distribution' and fitted_params:
                    pdf_vals_norm = dist_obj.pdf(x_vals_norm, *fitted_params)
                    pdf_vals = (pdf_vals_norm / (data_range + 1e-10)) * curve_scale
                    ax.plot(x_vals, pdf_vals, color='#e74c3c', 
                           linewidth=2, label=f'{selected_dist} PDF')
                elif mode == 'Manual Fitting' and manual_params:
                    pdf_vals_norm = dist_obj.pdf(x_vals_norm, *manual_params)
                    pdf_vals = (pdf_vals_norm / (data_range + 1e-10)) * curve_scale
                    ax.plot(x_vals, pdf_vals, color='#e74c3c', 
                           linewidth=2, label=f'{selected_dist} PDF (Manual)')
        elif selected_dist == 'Uniform':
            if mode == 'Auto Fit Distribution' and fitted_params:
                if len(fitted_params) == 2:
                    loc, scale = fitted_params
                    mask = (x_vals >= loc) & (x_vals <= loc + scale)
                    pdf_vals = np.where(mask, 1.0 / (scale + 1e-10), 0.0) * curve_scale
                    ax.plot(x_vals, pdf_vals, color='#e74c3c', 
                           linewidth=2, label=f'{selected_dist} PDF')
            elif mode == 'Manual Fitting' and manual_params:
                if len(manual_params) == 2:
                    loc, scale = manual_params
                    mask = (x_vals >= loc) & (x_vals <= loc + scale)
                    pdf_vals = np.where(mask, 1.0 / (scale + 1e-10), 0.0) * curve_scale
                    ax.plot(x_vals, pdf_vals, color='#e74c3c', 
                           linewidth=2, label=f'{selected_dist} PDF (Manual)')
        else:
            if mode == 'Auto Fit Distribution' and fitted_params:
                try:
                    pdf_vals = dist_obj.pdf(x_vals, *fitted_params)
                    pdf_vals = np.where(np.isfinite(pdf_vals), pdf_vals, 0)
                    pdf_vals *= curve_scale
                    ax.plot(x_vals, pdf_vals, color='#e74c3c', 
                           linewidth=2, label=f'{selected_dist} PDF')
                except Exception as e:
                    st.error(f"Error plotting PDF: {str(e)}")
            elif mode == 'Manual Fitting' and manual_params:
                try:
                    pdf_vals = dist_obj.pdf(x_vals, *manual_params)
                    pdf_vals = np.where(np.isfinite(pdf_vals), pdf_vals, 0)
                    pdf_vals *= curve_scale
                    ax.plot(x_vals, pdf_vals, color='#e74c3c', 
                           linewidth=2, label=f'{selected_dist} PDF (Manual)')
                except Exception as e:
                    st.error(f"Error plotting PDF: {str(e)}")
#axis limitss
    ax.set_xlim(xmin, xmax)
    hist_peak = np.max(n) if len(n) else 0
    current_ylim = ax.get_ylim()[1]
    ymax = max(current_ylim * 1.1, hist_peak * 1.15, 1e-6)
    ax.set_ylim(0, ymax)
#set labels and title
    ax.set_xlabel(params['xlabel'], color='black', fontsize=12)
    ax.set_ylabel(params['ylabel'], color='black', fontsize=12)
    ax.set_title(params['title'], color='black', fontsize=14, fontweight='bold')
    
    ax.tick_params(colors='black')
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1)
    
#add legend if theres a curve
    if mode in ['Auto Fit Distribution', 'Manual Fitting'] and selected_dist:
        legend = ax.legend(loc='best', facecolor='white', edgecolor='black')
        for text in legend.get_texts():
            text.set_color('black')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    return fig
#main app ui
st.title("Jaiaditya's Histogram & Distribution Fitting Web App (NE 111)")
st.markdown("---")
#sidebar for data input
with st.sidebar:
    st.header("Data Input")
    input_method = st.radio(
        "Choose input method:",
        ["Manual Input", "CSV Upload"])
    if input_method == "Manual Input":
        st.subheader("Value Ranges + freqs")
        st.caption("Provide the lower/upper bounds for each bin along with its frequency.")
        table_columns = ["Min Value", "Max Value", "Frequency"]
        if 'range_freq_table' not in st.session_state:
            st.session_state.range_freq_table = pd.DataFrame(
                [{col: None for col in table_columns} for _ in range(5)])
        
        def _append_blank_row():
            blank_row = pd.DataFrame([{col: None for col in table_columns}])
            st.session_state.range_freq_table = pd.concat(
                [st.session_state.range_freq_table, blank_row],
                ignore_index=True)
        if st.button("Add Row", key="add_range_row"):
            _append_blank_row()
            st.rerun()
        with st.form("manual_range_form"):
            edited_table = st.data_editor(
                st.session_state.range_freq_table,
                num_rows="dynamic",
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Min Value": st.column_config.NumberColumn("Min Value", format="%.4f"),
                    "Max Value": st.column_config.NumberColumn("Max Value", format="%.4f"),
                    "Frequency": st.column_config.NumberColumn("Frequency", format="%.4f", help="Counts or relative freqs")
                },
                key="range_table_editor")
            submit_range = st.form_submit_button("Load Table Data")
        st.session_state.range_freq_table = edited_table
        
        if submit_range:
            cleaned = edited_table.dropna(subset=table_columns)
            if cleaned.empty:
                st.warning("Please fill out at least one complete row.")
            else:
                data, weights, error = expand_range_weighted_samples(
                    cleaned["Min Value"].values,
                    cleaned["Max Value"].values,
                    cleaned["Frequency"].values)
                if error:
                    st.error(error)
                else:
                    edges = np.sort(np.unique(np.concatenate([
                        cleaned["Min Value"].values,
                        cleaned["Max Value"].values
                    ])))
                    store_loaded_data(data, weights, edges, source_label="manual_range")
                    st.success(f"Loaded {len(data)} synthesized data points from ranges!")
                    st.write(f"Min: {np.min(data):.4f}, Max: {np.max(data):.4f}, Mean: {np.mean(data):.4f}")
    
    else:  #csv upload
        st.subheader("CSV Upload")
        uploaded_file = st.file_uploader(
            "Upload a CSV file",
            type=['csv'],
            help="provide the range columns (min, max, frequency)")
        st.caption("Accepted format: CSV with range_min, range_max, frequency columns.")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Preview (first 5 rows):")
                st.dataframe(df.head(), use_container_width=True)
                
                data, weights, custom_bins, error = validate_csv_data(df)
                if error:
                    st.error(error)
                else:
                    store_loaded_data(data, weights, custom_bins, source_label="csv")
                    st.success(f"Loaded {len(data)} data points!")
                    st.write(f"Min: {np.min(data):.4f}, Max: {np.max(data):.4f}, Mean: {np.mean(data):.4f}")
            except Exception as e:
                st.error(f"Error reading CSV: {str(e)}")
    
    #displays the current data status
    if st.session_state.data is not None:
        st.markdown("---")
        total_obs = float(np.sum(st.session_state.data_weights)) if st.session_state.data_weights is not None else len(st.session_state.data)
        st.success(f"**Data loaded:** {len(st.session_state.data)} synthesized points (total frequency ≈ {total_obs:.4f})")
        if st.button("Clear Data"):
            st.session_state.data = None
            st.session_state.data_weights = None
            st.session_state.custom_bins = None
            st.rerun()

#main display area 
if st.session_state.data is None:
    st.info("<<<<<  Please load data using the sidebar to get started.")
else:
    data = st.session_state.data
    data_weights = st.session_state.data_weights
    custom_bins = st.session_state.custom_bins
    
#hiistogram customization settings (name, labels, axis)
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.histogram_params['title'] = st.text_input(
            "Histogram Title",
            value=st.session_state.histogram_params['title'])
        st.session_state.histogram_params['xlabel'] = st.text_input(
            "X-axis Label",
            value=st.session_state.histogram_params['xlabel'])
    
    with col2:
        st.session_state.histogram_params['ylabel'] = st.text_input(
            "Y-axis Label",
            value=st.session_state.histogram_params['ylabel'])
    st.markdown("---")
#mode selection
    mode = st.selectbox(
        "Select Mode:",
        ["View Histogram", "Auto Fit Distribution", "Manual Fitting"],
        key="mode_selector")
    st.markdown("---")
    
#mode-specific ui and plotting
    if mode == "View Histogram":
        st.subheader("Histogram View")
        fig = plot_histogram(data, mode, weights=data_weights, custom_bins=custom_bins)
        st.pyplot(fig)
    
    elif mode == "Auto Fit Distribution":
        st.subheader("Auto Fit Distribution")
        selected_dist = st.selectbox(
            "Select Distribution:",
            list(DISTRIBUTIONS.keys()),
            key="auto_dist_selector")
        if st.button("Fit Distribution", key="fit_button"):
            dist_obj = DISTRIBUTIONS[selected_dist]
            fitted_params, error = fit_distribution(dist_obj, selected_dist, data)
            
            if error:
                st.error(f"Error fitting {selected_dist}: {error}")
            else:
                st.session_state.fitted_params = fitted_params
                st.session_state.fitted_dist = selected_dist
                st.success(f"Successfully fitted {selected_dist} distribution!")
        
        if 'fitted_params' in st.session_state and 'fitted_dist' in st.session_state:
            if st.session_state.fitted_dist == selected_dist:
#show fitted parameters
                st.write("**Fitted Parameters:**")
                param_names = get_parameter_names(selected_dist)
                if len(param_names) == len(st.session_state.fitted_params):
                    for name, value in zip(param_names, st.session_state.fitted_params):
                        st.write(f"- {name}: {value:.4f}")
                else:
                    for i, value in enumerate(st.session_state.fitted_params):
                        st.write(f"- Parameter {i+1}: {value:.4f}")
                
#plot with fitted curve
                fig = plot_histogram(
                    data,
                    mode,
                    selected_dist,
                    fitted_params=st.session_state.fitted_params,
                    weights=data_weights,
                    custom_bins=custom_bins)
                st.pyplot(fig)
            else:
                if 'fitted_params' in st.session_state:
                    del st.session_state.fitted_params
                if 'fitted_dist' in st.session_state:
                    del st.session_state.fitted_dist
        else:
            fig = plot_histogram(data, mode, weights=data_weights, custom_bins=custom_bins)
            st.pyplot(fig)
    elif mode == "Manual Fitting":
        st.subheader("Manual Fitting")
        selected_dist = st.selectbox(
            "Select Distribution:",
            list(DISTRIBUTIONS.keys()),
            key="manual_dist_selector")
        dist_obj = DISTRIBUTIONS[selected_dist]
        initial_params, error = fit_distribution(dist_obj, selected_dist, data)
        if error:
            st.error(f"Error getting initial parameters: {error}")
            st.info("Please try a different distribution.")
        else:
            param_names = get_parameter_names(selected_dist)
            if len(param_names) != len(initial_params):
                param_names = [f'param_{i+1}' for i in range(len(initial_params))]
            st.write("**Adjust Parameters:**")
            manual_params = []
            
            cols = st.columns(min(3, len(initial_params)))
            for i, (param_name, init_value) in enumerate(zip(param_names, initial_params)):
                col_idx = i % len(cols)
                with cols[col_idx]:
                    slider_min, slider_max = get_slider_range(param_name, init_value, data)
                    manual_value = st.slider(
                        param_name,
                        min_value=float(slider_min),
                        max_value=float(slider_max),
                        value=float(init_value),
                        step=(slider_max - slider_min) / 1000,
                        key=f"manual_slider_{selected_dist}_{i}")
                    manual_params.append(manual_value)
            fig = plot_histogram(
                data,
                mode,
                selected_dist,
                manual_params=manual_params,
                weights=data_weights,
                custom_bins=custom_bins)
            st.pyplot(fig)
            st.write("**Current Parameter Values:**")
            for name, value in zip(param_names, manual_params):
                st.write(f"- {name}: {value:.4f}")
st.markdown("---")
st.caption("NE111 Project - Jaiaditya Asudani's Distribution and Histogram Web App :)")
