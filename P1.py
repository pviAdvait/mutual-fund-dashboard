from mftool import Mftool
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.optimize import newton
import plotly.express as px
from prophet import Prophet

# ------------------- CACHED FUNCTIONS -------------------

@st.cache_resource
def load_mf_tool():
    return Mftool()

@st.cache_data
def get_scheme_names():
    mf = load_mf_tool()
    return {v: k for k, v in mf.get_scheme_codes().items()}

@st.cache_data
def get_aum_data():
    mf = load_mf_tool()
    return pd.DataFrame(mf.get_average_aum('July - September 2024', False))

@st.cache_data
def get_nav_data(scheme_code: str) -> pd.DataFrame:
    mf = load_mf_tool()
    nav_df = mf.get_scheme_historical_nav(scheme_code, as_Dataframe=True)
    nav_df = nav_df.reset_index().rename(columns={"index": "date"})
    nav_df["date"] = pd.to_datetime(nav_df["date"], dayfirst=True)
    nav_df["nav"] = pd.to_numeric(nav_df["nav"], errors='coerce')
    nav_df.dropna(inplace=True)
    return nav_df

@st.cache_data
def get_scheme_details(scheme_code):
    mf = load_mf_tool()
    return mf.get_scheme_details(scheme_code)

@st.cache_data
def get_available_schemes(amc):
    mf = load_mf_tool()
    return mf.get_available_schemes(amc)

# ------------------- INIT -------------------

st.set_page_config(page_title="MF Dashboard", layout="wide")

mf = load_mf_tool()
scheme_names = get_scheme_names()

st.title('📊 Mutual Fund Financial Dashboard')

option = st.sidebar.selectbox(
    "Choose an action",
    ["ℹ️ About", "Scheme Overview", "Historical NAV", "Compare NAVs",
     "Risk and Volatility Analysis", "Portfolio Simulator", 
     "SIP & Lumpsum Investment Simulator", "🧠 Smart Portfolio Recommender"]
)

# --- Helper Functions ---
def calculate_portfolio_metrics(portfolio_returns, benchmark_returns=None, risk_free_rate=0.06):
    ann_return = (1 + portfolio_returns.mean()) ** 252 - 1
    ann_volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe = (ann_return - risk_free_rate) / ann_volatility if ann_volatility else np.nan

    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino = (ann_return - risk_free_rate) / downside_std if downside_std else np.nan

    beta = alpha = treynor = r_squared = None
    if benchmark_returns is not None:
        aligned = pd.DataFrame({
            'portfolio': portfolio_returns,
            'benchmark': benchmark_returns
        }).dropna()

        cov_matrix = np.cov(aligned['portfolio'], aligned['benchmark'])
        covariance = cov_matrix[0][1]
        variance = np.var(aligned['benchmark'])
        beta = covariance / variance if variance else np.nan

        expected_return = risk_free_rate + beta * (aligned['benchmark'].mean() * 252 - risk_free_rate)
        alpha = ann_return - expected_return
        treynor = (ann_return - risk_free_rate) / beta if beta else np.nan

        corr_matrix = np.corrcoef(aligned['portfolio'], aligned['benchmark'])
        r_squared = corr_matrix[0, 1] ** 2

    cumulative = (1 + portfolio_returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()

    return {
        "Return": ann_return,
        "Volatility": ann_volatility,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Beta": beta,
        "Alpha": alpha,
        "Treynor Ratio": treynor,
        "R-squared": r_squared,
        "Max Drawdown": max_drawdown
    }

def calculate_risk_metrics(portfolio_returns, risk_free_rate=0.06):
    ann_return = (1 + portfolio_returns.mean()) ** 252 - 1
    ann_volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe = (ann_return - risk_free_rate) / ann_volatility if ann_volatility else np.nan

    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino = (ann_return - risk_free_rate) / downside_std if downside_std else np.nan

    cumulative = (1 + portfolio_returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()

    return {
        "Annualized Return": ann_return,
        "Annualized Volatility": ann_volatility,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_drawdown
    }

def xirr(cash_flows):
    def xnpv(rate):
        t0 = cash_flows[0][0]
        return sum(cf / (1 + rate) ** ((t - t0).days / 365.0) for t, cf in cash_flows)
    try:
        return newton(xnpv, 0.1)
    except Exception:
        return float('nan')


# ------------------- SCHEME OVERVIEW -------------------
if option == "ℹ️ About":
    st.title("📘 About the Mutual Fund Financial Dashboard")

    st.markdown("""
    Welcome to the **Mutual Fund Financial Dashboard** — a smart, interactive platform for analyzing, simulating, and recommending mutual fund portfolios tailored to your needs.

    This application is built using **Streamlit** and integrates data-driven intelligence with real-world advisory workflows. Whether you're an investor, wealth advisor, or data enthusiast, this tool helps you make sharper, more personalized financial decisions.

    ### 🔧 Core Features
    - **🧾 Scheme Overview:** Explore AMCs, schemes, asset sizes, and categories.
    - **📈 Historical NAV & Heatmap:** Visualize scheme performance and trends.
    - **📊 NAV Comparison:** Compare performance across multiple schemes.
    - **⚠️ Risk & Volatility Analysis:** Sharpe Ratio, CAGR, and Monte Carlo simulation.
    - **🧮 Portfolio Simulator:** Build, simulate, and forecast investment portfolios.
    - **💸 SIP & Lumpsum Simulator:** Model investment strategies with XIRR and CAGR.
    - **🧠 Smart Portfolio Recommender:** Personalized fund selection based on your risk profile, goals, and liquidity needs — using advisor-curated schemes.

    ### 🧠 Powered By
    - `Streamlit`, `pandas`, `numpy`, `plotly`, `prophet`, `seaborn`, `mftool`
    - Advisor-uploaded mutual fund universe for real-world compatibility

    ### 👥 Who Is It For?
    - Financial advisors & wealth managers
    - Individual investors & HNIs
    - Students & fintech developers

    ---

   _Crafted with ❤️ for modern investors and forward-thinking advisors._

    ---
    👨‍💻 Developed by: **Advait Iyer**
    """)




elif option == 'Scheme Overview':
    st.header("📋 Mutual Fund Scheme Overview")
    selected_amc = st.sidebar.text_input("Enter AMC Name (e.g., ICICI, HDFC, SBI)", value="")

    try:
        aum_df = get_aum_data()
        aum_df["Total AUM"] = aum_df[["AAUM Overseas", "AAUM Domestic"]].astype(float).sum(axis=1)

        if selected_amc.strip():
            filtered_aum_df = aum_df[aum_df["Fund Name"].str.contains(selected_amc, case=False)]
            if not filtered_aum_df.empty:
                st.subheader(f"💰 Average AUM for AMC: {selected_amc}")
                st.dataframe(filtered_aum_df[["Fund Name", "Total AUM"]])
            else:
                st.warning(f"No AUM data found for AMC: {selected_amc}")
        else:
            st.subheader("💼 Average AUM for All AMCs")
            st.dataframe(aum_df[["Fund Name", "Total AUM"]])
    except Exception as e:
        st.error(f"❌ Error fetching AUM data: {str(e)}")

    if selected_amc.strip():
        try:
            schemes = get_available_schemes(selected_amc)
            if schemes:
                scheme_df = pd.DataFrame(schemes.items(), columns=["Scheme Code", "Scheme Name"])
                st.subheader(f"📄 Schemes Offered by {selected_amc}")
                st.dataframe(scheme_df)

                selected_scheme_name = st.selectbox("Select a scheme to view details", ["-- Select --"] + list(schemes.values()))

                if selected_scheme_name != "-- Select --":
                    selected_scheme_code = next((code for code, name in schemes.items() if name == selected_scheme_name), None)
                    if selected_scheme_code:
                        details = get_scheme_details(selected_scheme_code)
                        if isinstance(details, dict) and len(details) > 0:
                            st.subheader(f"🔍 Scheme Details - {selected_scheme_name}")
                            st.write(pd.DataFrame([details]).T)
                        else:
                            st.warning("No valid details returned for the selected scheme.")
                    else:
                        st.error("Could not resolve scheme code from selected scheme name.")
            else:
                st.warning(f"No schemes found for AMC: {selected_amc}")
        except Exception as e:
            st.error(f"❌ Error fetching scheme list/details: {str(e)}")

# [The next section to update is "Historical NAV"]
elif option == 'Historical NAV':
    st.header('📈 Historical NAV & Heatmap')

    selected_scheme = st.sidebar.selectbox("Select a Scheme", ["-- Select --"] + list(scheme_names.keys()))

    if selected_scheme != "-- Select --":
        try:
            scheme_code = scheme_names[selected_scheme]
            nav_data = get_nav_data(scheme_code)  # ✅ Cached function

            if not nav_data.empty:
                # Line Chart
                st.subheader("📉 NAV Over Time")
                fig = px.line(nav_data, x="date", y="nav", title=f'NAV Trend - {selected_scheme}')
                st.plotly_chart(fig, use_container_width=True)

                # Heatmap of Monthly Daily Returns
                st.subheader("🔥 Monthly Daily Returns Heatmap")
                import seaborn as sns
                import matplotlib.pyplot as plt
                import matplotlib.dates as mdates

                nav_data["return_pct"] = nav_data["nav"].pct_change() * 100
                nav_data["day"] = nav_data["date"].dt.day
                nav_data["month_year"] = nav_data["date"].dt.to_period("M").dt.to_timestamp()

                pivot = nav_data.pivot_table(
                    index="month_year",
                    columns="day",
                    values="return_pct",
                    aggfunc="mean"
                )
                pivot = pivot.sort_index(ascending=False)

                fig2, ax = plt.subplots(figsize=(14, 10))
                sns.heatmap(
                    pivot,
                    cmap="RdYlGn",
                    center=0,
                    linewidths=0.05,
                    cbar_kws={'label': 'Avg Daily Return (%)'},
                    ax=ax
                )
                ax.set_title("Monthly Daily Returns Heatmap", fontsize=16)
                ax.set_xlabel("Day of Month")
                ax.set_ylabel("Month")
                ax.set_yticks(range(len(pivot.index)))
                ax.set_yticklabels([d.strftime('%b-%Y') for d in pivot.index])

                st.pyplot(fig2)

            else:
                st.warning("⚠️ No NAV data returned for the selected scheme.")
        except Exception as e:
            st.error(f"❌ Failed to fetch historical NAV: {str(e)}")
    else:
        st.info("Please select a mutual fund scheme to view its historical NAV.")


elif option == 'Compare NAVs':
    st.header("📊 Compare NAVs")

    selected_schemes = st.sidebar.multiselect("Select Schemes to Compare", options=list(scheme_names.keys()))
    if selected_schemes:
        comparison_df = pd.DataFrame()

        for scheme in selected_schemes:
            scheme_code = scheme_names[scheme]
            nav_data = get_nav_data(scheme_code)  # ✅ Cached

            nav_data = nav_data.set_index("date")
            nav_data = nav_data.sort_index()
            nav_data = nav_data[~nav_data.index.duplicated(keep='first')]
            comparison_df[scheme] = nav_data["nav"]

        comparison_df.dropna(inplace=True)

        if not comparison_df.empty:
            # NAV Comparison Chart
            st.subheader("📈 NAV Comparison Chart")
            fig = px.line(comparison_df, title="NAV Trend Comparison")
            st.plotly_chart(fig)

            # ------- Enhanced Fund Metrics Comparison --------
            st.subheader("📊 Fund Performance Metrics Comparison")

            metric_table = {}
            for scheme in selected_schemes:
                nav_series = comparison_df[scheme]
                returns = nav_series.pct_change().dropna()

                if returns.empty:
                    continue

                metrics = calculate_risk_metrics(returns)

                # CAGR calculation
                init_nav = nav_series.iloc[0]
                final_nav = nav_series.iloc[-1]
                duration_years = (nav_series.index[-1] - nav_series.index[0]).days / 365.25
                cagr = (final_nav / init_nav) ** (1 / duration_years) - 1 if duration_years > 0 else np.nan
                metrics["CAGR"] = cagr

                metric_table[scheme] = metrics

            if metric_table:
                metrics_df = pd.DataFrame(metric_table).T  # Funds as rows

                # Convert return-based metrics to percentage scale
                percentage_metrics = ["Annualized Return", "Annualized Volatility", "Max Drawdown", "CAGR"]
                for col in percentage_metrics:
                    if col in metrics_df.columns:
                        metrics_df[col] *= 100

                sorted_df = metrics_df.sort_values(by="CAGR", ascending=False)
                transposed_df = sorted_df.T  # Transpose: rows = metrics

                # Format settings
                format_dict = {
                    "Sharpe Ratio": "{:.2f}",
                    "Sortino Ratio": "{:.2f}"
                }
                for metric in transposed_df.index:
                    if metric not in format_dict:
                        format_dict[metric] = "{:.2f}%"  # Percentage display

                st.dataframe(transposed_df.style.format(format_dict))
            else:
                st.warning("⚠️ Unable to compute metrics due to insufficient return data.")
        else:
            st.warning("⚠️ No overlapping NAV data found for the selected schemes.")
    else:
        st.info("Select at least one scheme to compare.")


elif option == "Risk and Volatility Analysis":
    st.header("📊 Risk and Volatility Analysis")

    if scheme_names:
        scheme_options = ["Select a Scheme"] + list(scheme_names.keys())
        selected_scheme = st.sidebar.selectbox("Select a Scheme", scheme_options)

        if selected_scheme != "Select a Scheme":
            scheme_code = scheme_names.get(selected_scheme)

            try:
                nav_data = get_nav_data(scheme_code)  # ✅ Cached

                if not nav_data.empty:
                    nav_data.sort_values("date", inplace=True)
                    nav_data["returns"] = nav_data["nav"].pct_change()
                    nav_data.dropna(subset=["returns"], inplace=True)

                    annualized_volatility = nav_data["returns"].std() * np.sqrt(252)
                    annualized_return = (1 + nav_data["returns"].mean()) ** 252 - 1
                    risk_free_rate = 0.06
                    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility

                    st.write(f"### Metrics for {selected_scheme}")
                    st.metric("Annualized Volatility", f"{annualized_volatility:.2%}")
                    st.metric("Annualized Return", f"{annualized_return:.2%}")
                    st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

                    # CAGR Calculator
                    st.subheader("CAGR Calculator")
                    min_date = nav_data["date"].min().date()
                    max_date = nav_data["date"].max().date()
                    start_date = st.date_input("Start Date", min_value=min_date, max_value=max_date, value=min_date)
                    end_date = st.date_input("End Date", min_value=start_date, max_value=max_date, value=max_date)

                    if start_date < end_date:
                        start_nav = nav_data.loc[nav_data["date"] >= pd.to_datetime(start_date), "nav"].iloc[0]
                        end_nav = nav_data.loc[nav_data["date"] <= pd.to_datetime(end_date), "nav"].iloc[-1]
                        num_years = (end_date - start_date).days / 365.25
                        cagr = (end_nav / start_nav) ** (1 / num_years) - 1
                        st.metric("CAGR", f"{cagr:.2%}")
                    else:
                        st.warning("End date must be after start date.")

                    fig = px.scatter(nav_data, x="date", y="returns", title=f"Risk-Return Scatter for {selected_scheme}")
                    st.plotly_chart(fig)

                    # Monte Carlo Simulation
                    st.write("### Monte Carlo Simulation for Future NAV Projection")
                    num_simulations = st.slider("Number of Simulations", min_value=100, max_value=5000, value=1000)
                    num_days = st.slider("Projection Period (Days)", min_value=30, max_value=365, value=252)

                    last_nav = nav_data["nav"].iloc[-1]
                    daily_volatility = nav_data["returns"].std()
                    daily_mean_return = nav_data["returns"].mean()

                    simulation_results = []
                    for _ in range(num_simulations):
                        prices = [last_nav]
                        for _ in range(num_days):
                            simulated_return = np.random.normal(daily_mean_return, daily_volatility)
                            prices.append(prices[-1] * (1 + simulated_return))
                        simulation_results.append(prices)

                    simulation_df = pd.DataFrame(simulation_results).T
                    simulation_df.index.name = "Day"
                    simulation_df.columns = [f"Simulation {i+1}" for i in range(num_simulations)]

                    fig_simulation = px.line(
                        simulation_df,
                        title=f"Monte Carlo Simulation for {selected_scheme} NAV Projection",
                        labels={"value": "Projected NAV", "index": "Day"},
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig_simulation)

                    final_prices = simulation_df.iloc[-1]
                    st.write(f"### Simulation Summary for {selected_scheme}")
                    st.metric("Expected Final NAV", f"{final_prices.mean():.2f}")
                    st.metric("Minimum Final NAV", f"{final_prices.min():.2f}")
                    st.metric("Maximum Final NAV", f"{final_prices.max():.2f}")

                else:
                    st.warning("No historical NAV data available for the selected scheme.")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
        else:
            st.info("Please select a scheme to continue.")
    else:
        st.warning("Scheme data is empty or not loaded.")


elif option == "Portfolio Simulator":
    st.header("📊 Portfolio Simulator")

    selected_schemes = st.multiselect("Select Mutual Fund Schemes", options=list(scheme_names.keys()))
    if len(selected_schemes) < 2:
        st.warning("⚠️ Please select at least two funds for portfolio simulation.")
        st.stop()

    weights = {}
    total_weight = 0

    st.subheader("Weight Allocation (%)")
    for scheme in selected_schemes:
        weight = st.slider(f"{scheme}", 0, 100, 0)
        weights[scheme] = weight
        total_weight += weight

    if total_weight != 100:
        st.warning("⚠️ Total weights must sum up to 100%.")
        st.stop()

    # Date range & benchmark
    start_date = st.date_input("Start Date", value=datetime(2021, 1, 1))
    end_date = st.date_input("End Date", value=datetime.today())
    benchmark_scheme = st.selectbox("Benchmark (Optional)", options=["None"] + list(scheme_names.keys()))

    # Build NAV dataframe
    portfolio_nav_df = pd.DataFrame()
    for scheme in selected_schemes:
        code = scheme_names[scheme]
        nav_data = get_nav_data(code)
        nav_data = nav_data[(nav_data["date"] >= pd.to_datetime(start_date)) & (nav_data["date"] <= pd.to_datetime(end_date))]
        nav_data = nav_data[["date", "nav"]].rename(columns={"nav": scheme})
        nav_data.set_index("date", inplace=True)
        nav_data = nav_data.sort_index()

        portfolio_nav_df = nav_data if portfolio_nav_df.empty else portfolio_nav_df.join(nav_data, how="outer")

    portfolio_nav_df.dropna(inplace=True)
    if portfolio_nav_df.empty:
        st.error("❌ Not enough NAV data available for the selected schemes and date range.")
        st.stop()

    # Rebase & weight
    rebased_df = portfolio_nav_df / portfolio_nav_df.iloc[0] * 100
    for scheme in selected_schemes:
        rebased_df[scheme] *= (weights[scheme] / 100)

    rebased_df["Portfolio"] = rebased_df[selected_schemes].sum(axis=1)

    # Plot NAV
    st.subheader("📈 Portfolio NAV Over Time")
    st.plotly_chart(px.line(rebased_df, y="Portfolio", title="Portfolio NAV Time-Series"))

    # Correlation Matrix
    st.subheader("📊 Correlation Matrix of Selected Funds")
    returns_df = portfolio_nav_df[selected_schemes].pct_change().dropna()
    if not returns_df.empty:
        corr_matrix = returns_df.corr()
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            title="Correlation Matrix",
            labels={"color": "Correlation"}
        )
        st.plotly_chart(fig_corr)
        st.dataframe(corr_matrix.style.format("{:.2f}"))
    else:
        st.warning("⚠️ Not enough return data to compute correlation.")

    # Portfolio Returns
    portfolio_returns = rebased_df["Portfolio"].pct_change().dropna()
    benchmark_returns = None

    if benchmark_scheme != "None":
        benchmark_code = scheme_names[benchmark_scheme]
        b_nav_df = get_nav_data(benchmark_code).copy()
        b_nav_df = b_nav_df.set_index("date").sort_index()
        benchmark_nav = b_nav_df["nav"].reindex(portfolio_nav_df.index, method='ffill').bfill()
        benchmark_returns = benchmark_nav.pct_change().dropna()

        # Align both
        shared_index = portfolio_returns.index.intersection(benchmark_returns.index)
        portfolio_returns = portfolio_returns.loc[shared_index]
        benchmark_returns = benchmark_returns.loc[shared_index]

        if portfolio_returns.empty or benchmark_returns.empty:
            st.error("⚠️ Benchmark and portfolio returns have no overlapping data. Try adjusting dates.")
            st.stop()

    # --- Metrics ---
    st.subheader("📈 Portfolio Metrics")
    metrics = calculate_portfolio_metrics(portfolio_returns, benchmark_returns)

    cols = st.columns(5)
    cols[0].metric("Annualized Return", f"{metrics['Return']:.2%}")
    cols[1].metric("Volatility", f"{metrics['Volatility']:.2%}")
    cols[2].metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
    cols[3].metric("Sortino Ratio", f"{metrics['Sortino Ratio']:.2f}")
    cols[4].metric("Max Drawdown", f"{metrics['Max Drawdown']:.2%}")

    if benchmark_scheme != "None":
        cols2 = st.columns(4)
        cols2[0].metric("Portfolio Beta", f"{metrics['Beta']:.2f}")
        cols2[1].metric("Jensen’s Alpha", f"{metrics['Alpha']:.2%}")
        cols2[2].metric("Treynor Ratio", f"{metrics['Treynor Ratio']:.2f}")
        cols2[3].metric("R-squared", f"{metrics['R-squared']:.2%}")

    # Pie chart of weights
    st.subheader("📊 Weight Distribution")
    st.plotly_chart(px.pie(names=list(weights.keys()), values=list(weights.values()), title="Portfolio Allocation"))

    # Monte Carlo Simulation
    st.subheader("🎲 Monte Carlo Simulation")
    num_sim = st.slider("Number of Simulations", 100, 3000, 1000)
    num_days = st.slider("Projection Period (Days)", 30, 365, 252)

    last_value = rebased_df["Portfolio"].iloc[-1]
    daily_return = portfolio_returns.mean()
    daily_volatility = portfolio_returns.std()

    simulations = []
    for _ in range(num_sim):
        prices = [last_value]
        for _ in range(num_days):
            ret = np.random.normal(daily_return, daily_volatility)
            prices.append(prices[-1] * (1 + ret))
        simulations.append(prices)

    sim_df = pd.DataFrame(simulations).T
    sim_df.columns = [f"Sim {i+1}" for i in range(num_sim)]
    st.plotly_chart(px.line(sim_df, title="Monte Carlo Simulation"))

    final_vals = sim_df.iloc[-1]
    st.metric("Expected Final NAV", f"{final_vals.mean():.2f}")
    st.metric("Min Final NAV", f"{final_vals.min():.2f}")
    st.metric("Max Final NAV", f"{final_vals.max():.2f}")

            # Prophet Forecast
    st.subheader("🔮 ML Forecast with Prophet")

    # Prepare forecast data
    forecast_data = rebased_df["Portfolio"].asfreq("D").ffill().reset_index()
    forecast_data.columns = ["ds", "y"]  # Rename columns for Prophet

    # Fit Prophet model
    model = Prophet(daily_seasonality=True)
    model.fit(forecast_data)

    # Make future dataframe for prediction
    future = model.make_future_dataframe(periods=90)
    forecast = model.predict(future)

    # Plot the forecast
    st.plotly_chart(px.line(
        forecast, x="ds", y=["yhat", "yhat_upper", "yhat_lower"],
        labels={"ds": "Date", "value": "Projected NAV"},
        title="Prophet Forecast (90 Days)"
    ))

    # Renaming columns and formatting the table
    forecast = forecast.rename(columns={
        "ds": "Date",
        "yhat": "Forecasted NAV",
        "yhat_upper": "Upper Bound",
        "yhat_lower": "Lower Bound"
    })

    # Display the forecasted values (last 10 days)
    st.write("Forecasted Values (Last 10 Days)")
    st.dataframe(forecast[["Date", "Forecasted NAV", "Lower Bound", "Upper Bound"]].tail(10).style.format({
        "Forecasted NAV": "{:.2f}",
        "Lower Bound": "{:.2f}",
        "Upper Bound": "{:.2f}"
    }))



elif option == "SIP & Lumpsum Investment Simulator":
    st.header("💸 SIP & Lumpsum Investment Simulator")

    investment_type = st.radio("Select Investment Type", ["SIP", "Lumpsum"])
    allocation_mode = st.radio("Allocation Mode", ["Percentage (%)", "Amount (₹)"])
    duration_years = st.slider("Investment Duration (in years)", 1, 10, 5)

    selected_schemes = st.multiselect("Select Mutual Fund Schemes", options=list(scheme_names.keys()))

    if selected_schemes:
        allocations = {}
        st.subheader("🧮 Enter Allocation per Fund")

        total_alloc = 0
        for scheme in selected_schemes:
            if allocation_mode == "Percentage (%)":
                val = st.slider(f"{scheme}", 0, 100, 100 // len(selected_schemes))
            else:
                val = st.number_input(f"{scheme} amount", min_value=0, value=1000, step=500)
            allocations[scheme] = val
            total_alloc += val

        if (allocation_mode == "Percentage (%)" and total_alloc != 100) or (allocation_mode == "Amount (₹)" and total_alloc <= 0):
            st.warning("⚠️ Ensure total allocations are valid.")
        else:
            st.info("📥 Fetching NAV data...")
            end_date = datetime.today()
            requested_start = end_date - pd.DateOffset(years=duration_years)

            nav_data = {}
            actual_start_dates = []

            for scheme in selected_schemes:
                code = scheme_names[scheme]
                nav = get_nav_data(code)  # ✅ Cached
                nav = nav.set_index("date").sort_index()
                actual_start_dates.append(nav.index.min())
                nav_data[scheme] = nav["nav"]

            start_date = max(actual_start_dates)
            if start_date > requested_start:
                st.warning(f"⚠️ Adjusted start date due to limited data: {start_date.date()}")
            else:
                start_date = requested_start

            # Align NAV data
            nav_data_aligned = {}
            for scheme, series in nav_data.items():
                aligned = series[(series.index >= start_date) & (series.index <= end_date)]
                nav_data_aligned[scheme] = aligned

            nav_df = pd.concat(nav_data_aligned.values(), axis=1, join='inner')
            nav_df.columns = nav_data_aligned.keys()
            nav_df.dropna(inplace=True)

            if nav_df.empty:
                st.error("❌ No overlapping NAV data found. Try selecting different funds or shorter duration.")
                st.stop()

            # Normalize Allocations
            if allocation_mode == "Percentage (%)":
                normalized_alloc = {k: v / 100 for k, v in allocations.items()}
            else:
                total_amt = sum(allocations.values())
                normalized_alloc = {k: v / total_amt for k, v in allocations.items()}

            # Investment Calculation
            units = {scheme: 0 for scheme in selected_schemes}
            investment_tracker = []
            cash_flows = []

            if investment_type == "Lumpsum":
                invested_amt = 0
                for scheme in selected_schemes:
                    amt = normalized_alloc[scheme] * 100000
                    nav = nav_df.iloc[0][scheme]
                    units[scheme] = amt / nav
                    invested_amt += amt

                for dt in nav_df.index:
                    total_val = sum(units[s] * nav_df.loc[dt, s] for s in selected_schemes)
                    investment_tracker.append((dt, total_val))

                final_value = investment_tracker[-1][1]
                cagr = ((final_value / invested_amt) ** (1 / duration_years)) - 1

                st.metric("Amount Invested", f"₹{invested_amt:,.2f}")
                st.metric("Current Value", f"₹{final_value:,.2f}")
                st.metric("CAGR", f"{cagr:.2%}")

            else:
                freq = st.selectbox("SIP Frequency", ["Monthly", "Weekly", "Daily", "Quarterly"])
                freq_map = {"Daily": 'B', "Weekly": 'W-MON', "Monthly": 'MS', "Quarterly": 'QS'}
                sip_dates = pd.date_range(start=start_date, end=end_date, freq=freq_map[freq])
                sip_nav_df = nav_df.reindex(sip_dates, method='ffill').dropna()

                total_invested = 0
                for dt in sip_nav_df.index:
                    invested_this = 0
                    for scheme in selected_schemes:
                        amt = normalized_alloc[scheme] * 1000
                        nav = sip_nav_df.loc[dt, scheme]
                        units[scheme] += amt / nav
                        invested_this += amt
                    cash_flows.append((dt, -invested_this))
                    total_invested += invested_this

                for dt in nav_df.index:
                    total_val = sum(units[s] * nav_df.loc[dt, s] for s in selected_schemes)
                    investment_tracker.append((dt, total_val))

                final_value = investment_tracker[-1][1]
                cash_flows.append((investment_tracker[-1][0], final_value))
                returns = final_value - total_invested

                try:
                    xirr_val = newton(lambda r: sum(cf / (1 + r) ** ((t - cash_flows[0][0]).days / 365.0) for t, cf in cash_flows), 0.1)
                except:
                    xirr_val = float('nan')

                st.metric("Amount Invested", f"₹{total_invested:,.2f}")
                st.metric("Current Value", f"₹{final_value:,.2f}")
                st.metric("Returns", f"₹{returns:,.2f}")
                st.metric("XIRR", f"{xirr_val:.2%}" if not np.isnan(xirr_val) else "Calc Error")

            # Plot Portfolio Value
            df_tracker = pd.DataFrame(investment_tracker, columns=["Date", "Portfolio Value"]).set_index("Date")
            st.plotly_chart(px.line(df_tracker, title="📈 Portfolio Value Over Time"))

            st.subheader("📦 Units Held Per Fund")
            st.dataframe(pd.DataFrame({"Fund": list(units.keys()), "Units": list(units.values())}).style.format({"Units": "{:.2f}"}))

            st.subheader("📊 Individual Fund Performance")
            perf_data = []
            for scheme in selected_schemes:
                returns = nav_df[scheme].pct_change().dropna()
                metrics = calculate_risk_metrics(returns)
                final = nav_df[scheme].iloc[-1]
                init = nav_df[scheme].iloc[0]
                cagr = ((final / init) ** (1 / duration_years)) - 1
                perf_data.append({
                    "Scheme": scheme,
                    "CAGR": cagr,
                    **metrics
                })

            perf_df = pd.DataFrame(perf_data).set_index("Scheme")
            st.dataframe(perf_df.style.format("{:.2%}"))

elif option == "🧠 Smart Portfolio Recommender":


    st.title("🧠 Smart Portfolio Recommender")
    st.markdown("_Tailored mutual fund portfolios based on your goals, risk, and liquidity needs_")


    # ---------------------- Section 1: Risk Profiling Questionnaire ----------------------
    st.header("1. 📋 Investor Risk Profiling")

    with st.form("risk_form"):
        st.subheader("Investor Details")
        name = st.text_input("Name")
        phone = st.text_input("Phone")
        email = st.text_input("Email")
        dob = st.date_input("Date of Birth")
        pan = st.text_input("PAN")
        address = st.text_area("Address")

        st.subheader("Risk Assessment Questions")
        q1 = st.selectbox("1. How old are you?", ["Above 60 Years", "Between 45 - 60 Years", "Between 30 - 45 Years", "Less than 30 Years"])
        q2 = st.selectbox("2. How many years away is your nearest goal?", ["Less than 2 Years", "Between 2 - 3 years", "Between 3 - 5 Years", "Between 5 - 10 Years", "Above 10 Years"])
        q3 = st.selectbox("3. If your investment outlook is long-term (more than 5 years), how long will you hold on to a poorly performing portfolio before cashing in?", ["Not hold and cash in immediately", "Hold for 3 months", "Hold for 6 months", "Hold for 1 year", "Hold for more than 2 years"])
        q4 = st.selectbox("4. My current and future source of income are:", ["Very unstable", "Unstable", "Somewhat stable", "Stable", "Very Stable"])
        q5 = st.selectbox("5. Choose your investment preference/objective:", [
            "Principal protection",
            "Loss of 4% for gain of 10%",
            "Loss of 8% for gain of 20%",
            "Loss of 15% for gain of 30%",
            "Loss of 25% for gain of 50%"])
        q6 = st.selectbox("6. If portfolio goes down 20%, what would you do?", [
            "Sell completely",
            "Sell partially",
            "Wait a little longer",
            "Retain fully",
            "Buy more"])
        q7 = st.selectbox("7. Desired balance between return and tax efficiency:", [
            "Guaranteed returns before tax efficiency",
            "Stable returns, minimal tax efficiency",
            "Some variability in returns",
            "Moderate variability with tax efficiency",
            "Unstable but higher returns with tax efficiency"])
        q8 = st.selectbox("8. Familiarity with financial markets:", [
            "No idea",
            "Basic idea, no experience",
            "Fair knowledge and experience",
            "Thorough knowledge and philosophy"])
        q9 = st.selectbox("9. Describe your risk range:", [
            "1% to 15%",
            "-5% to 20%",
            "-10% to 25%",
            "-15% to 30%",
            "-18% to 35%",
            "-22% to 45%"])
        q10 = st.selectbox("10. Current net worth:", [
            "Less than 50 lakhs",
            "50 lakhs - 1 Crore",
            "1 Crore - 3 Crores",
            "3 Crores - 5 Crores",
            "More than 5 Crores"])

        submitted = st.form_submit_button("Submit Risk Profile")

    if submitted:
        score = 0
        score += [0, 1, 2, 3][["Above 60 Years", "Between 45 - 60 Years", "Between 30 - 45 Years", "Less than 30 Years"].index(q1)]
        score += [0, 1, 2, 3, 4][["Less than 2 Years", "Between 2 - 3 years", "Between 3 - 5 Years", "Between 5 - 10 Years", "Above 10 Years"].index(q2)]
        score += [0, 1, 2, 3, 4][["Not hold and cash in immediately", "Hold for 3 months", "Hold for 6 months", "Hold for 1 year", "Hold for more than 2 years"].index(q3)]
        score += [0, 1, 2, 3, 4][["Very unstable", "Unstable", "Somewhat stable", "Stable", "Very Stable"].index(q4)]
        score += [0, 1, 2, 3, 4][["Principal protection", "Loss of 4% for gain of 10%", "Loss of 8% for gain of 20%", "Loss of 15% for gain of 30%", "Loss of 25% for gain of 50%"].index(q5)]
        score += [0, 1, 2, 3, 4][["Sell completely", "Sell partially", "Wait a little longer", "Retain fully", "Buy more"].index(q6)]
        score += [0, 1, 2, 3, 4][["Guaranteed returns before tax efficiency", "Stable returns, minimal tax efficiency", "Some variability in returns", "Moderate variability with tax efficiency", "Unstable but higher returns with tax efficiency"].index(q7)]
        score += [0, 1, 2, 3][["No idea", "Basic idea, no experience", "Fair knowledge and experience", "Thorough knowledge and philosophy"].index(q8)]
        score += [0, 1, 2, 3, 4, 5][["1% to 15%", "-5% to 20%", "-10% to 25%", "-15% to 30%", "-18% to 35%", "-22% to 45%"].index(q9)]

        if score <= 12:
            risk_category = "Conservative"
        elif score <= 24:
            risk_category = "Moderate"
        else:
            risk_category = "Aggressive"

        st.success(f"Your Risk Profile is: **{risk_category}**")
        st.session_state["risk_category"] = risk_category

    # ---------------------- Section 2: Advisor Fund Universe Upload ----------------------
    st.header("2. 📁 Upload Advisor's Fund Universe")
    st.markdown("Upload a CSV file with columns: `Scheme Name`, `Scheme Code`, `Asset Class`, `Risk Category`, `Preferred Mode`")

    fund_file = st.file_uploader("Upload Fund Universe CSV", type=["csv"])

    if fund_file is not None:
        try:
            fund_df = pd.read_csv(fund_file)
            required_cols = ["Scheme Name", "Scheme Code", "Asset Class", "Risk Category", "Preferred Mode"]
            if all(col in fund_df.columns for col in required_cols):
                st.success("Fund universe loaded successfully.")
                st.dataframe(fund_df.head(10))
                st.session_state["fund_universe"] = fund_df
            else:
                st.error(f"Missing columns. Please include all: {', '.join(required_cols)}")
        except Exception as e:
            st.error(f"Failed to load file: {e}")

    # ---------------------- Section 3: Net Worth & Liquidity Inputs ----------------------
    st.header("3. 💼 Net Worth Distribution")
    st.markdown("Enter the approximate percentage of your net worth in each asset. Ensure the total equals 100%.")

    real_estate = st.slider("Real Estate (%)", 0, 100, 40)
    fd_pfs = st.slider("FDs / PPF / EPF (%)", 0, 100, 30)
    equity = st.slider("Equity (Stocks + Mutual Funds) (%)", 0, 100, 20)
    gold_silver = st.slider("Gold / Silver (%)", 0, 100, 5)
    cash = st.slider("Cash / Savings Account (%)", 0, 100, 5)

    total_net_worth = real_estate + fd_pfs + equity + gold_silver + cash
    if total_net_worth != 100:
        st.warning(f"⚠️ Total allocation must equal 100%. Currently: {total_net_worth}%")

    # ---------------------- Section 4: Liquidity Constraints ----------------------
    st.header("4. 💸 Liquidity Needs")

    need_liquidity = st.radio("Do you need any amount in the next 1–2 years?", ["No", "Yes"])
    short_term_amount = 0

    if need_liquidity == "Yes":
        short_term_amount = st.number_input("Enter amount needed in next 1–2 years (₹)", min_value=0)

    # ---------------------- Section 5: Target Allocation Based on Risk Profile ----------------------
    st.header("5. 📊 Recommended Asset Allocation")

    if "risk_category" in st.session_state:
        profile = st.session_state["risk_category"]

        if profile == "Conservative":
            base_alloc = {"Equity": 20, "Hybrid": 25, "Debt": 40, "Gold": 10, "Liquid": 5}
        elif profile == "Moderate":
            base_alloc = {"Equity": 40, "Hybrid": 25, "Debt": 20, "Gold": 10, "Liquid": 5}
        elif profile == "Aggressive":
            base_alloc = {"Equity": 60, "Hybrid": 20, "Debt": 10, "Gold": 5, "Liquid": 5}
        else:
            base_alloc = {"Equity": 35, "Hybrid": 25, "Debt": 25, "Gold": 10, "Liquid": 5}

        st.subheader(f"🎯 Risk Profile: {profile}")

        total_investment = st.number_input("Total amount available for investment (₹)", min_value=10000, step=10000)

        # Step 1: Allocate short-term need to Debt
        if short_term_amount > 0 and short_term_amount < total_investment:
            st.info(f"₹{short_term_amount:,.0f} will be reserved for short-term debt allocation.")
            remaining = total_investment - short_term_amount
        else:
            remaining = total_investment

        # Step 2: Apply base allocation on remaining amount
        alloc_table = []
        for asset, weight in base_alloc.items():
            amt = (weight / 100) * remaining
            if asset == "Debt":
                amt += short_term_amount
            alloc_table.append((asset, weight, f"₹{amt:,.0f}"))

        df_alloc = pd.DataFrame(alloc_table, columns=["Asset Class", "Target %", "Suggested Amount"])
        st.table(df_alloc)

    else:
        st.warning("⚠️ Please complete the Risk Profile form to generate recommendations.")


    # ---------------------- Section 6: Intelligent Fund Selection ----------------------
    st.header("6. 🧠 Recommended Mutual Fund Schemes (Intelligent Selector)")

    MAX_FUNDS = 10
    risk_priority = {"Conservative": 1, "Moderate": 2, "Aggressive": 3}

    if (
        "fund_universe" in st.session_state and 
        "risk_category" in st.session_state and 
        "df_alloc" in locals()
    ):
        mode = st.radio("Preferred Investment Mode", ["SIP", "Lumpsum", "Not Sure"])
        fund_df = st.session_state["fund_universe"].copy()
        profile = st.session_state["risk_category"]
        profile_level = risk_priority[profile]

        df_alloc = df_alloc.copy()
        allocation_data = df_alloc[["Asset Class", "Suggested Amount"]].copy()
        allocation_data["Suggested Amount"] = allocation_data["Suggested Amount"].replace('[₹,]', '', regex=True).astype(float)

        total_allocation = allocation_data["Suggested Amount"].sum()
        allocation_data["Weight %"] = allocation_data["Suggested Amount"] / total_allocation
        allocation_data["Fund Slots"] = (allocation_data["Weight %"] * MAX_FUNDS).round().astype(int)

        # Adjust total to not exceed MAX_FUNDS
        while allocation_data["Fund Slots"].sum() > MAX_FUNDS:
            idx = allocation_data["Fund Slots"].idxmax()
            allocation_data.at[idx, "Fund Slots"] -= 1
        while allocation_data["Fund Slots"].sum() < MAX_FUNDS:
            idx = allocation_data["Fund Slots"].idxmin()
            allocation_data.at[idx, "Fund Slots"] += 1

        scheme_rows = []

        for _, row in allocation_data.iterrows():
            asset = row["Asset Class"]
            amount = row["Suggested Amount"]
            n_funds = row["Fund Slots"]

            matching = fund_df[fund_df["Asset Class"].str.lower() == asset.lower()]

            # Primary filter: match mode and risk category
            if mode != "Not Sure":
                matching = matching[
                    (matching["Preferred Mode"].str.lower() == mode.lower()) |
                    (matching["Preferred Mode"].str.lower() == "both")
                ]

            matching = matching[
                matching["Risk Category"].map(lambda x: risk_priority.get(x, 0)) <= profile_level
            ]

            # Fallback #1: ignore mode
            if matching.empty:
                matching = fund_df[fund_df["Asset Class"].str.lower() == asset.lower()]
                matching = matching[
                    matching["Risk Category"].map(lambda x: risk_priority.get(x, 0)) <= profile_level
                ]

            # Fallback #2: ignore risk category
            if matching.empty:
                matching = fund_df[fund_df["Asset Class"].str.lower() == asset.lower()]

            if matching.empty:
                st.warning(f"No suitable funds found for {asset} with any filter.")
                continue

            # Sort by Risk Category (lower better), then Scheme Name
            matching["Risk Rank"] = matching["Risk Category"].map(lambda x: risk_priority.get(x, 0))
            matching = matching.sort_values(by=["Risk Rank", "Scheme Name"])

            selected = matching.head(n_funds)

            if selected.empty:
                st.warning(f"No funds selected for {asset}. Skipping allocation.")
                continue

            amt_per_fund = amount / len(selected)

            for _, fund in selected.iterrows():
                scheme_rows.append({
                    "Asset Class": asset,
                    "Scheme Name": fund["Scheme Name"],
                    "Scheme Code": fund["Scheme Code"],
                    "Amount": amt_per_fund
                })

        scheme_df = pd.DataFrame(scheme_rows)

        if not scheme_df.empty:
            st.subheader("📌 Final Portfolio Recommendation")
            scheme_df_display = scheme_df.copy()
            scheme_df_display["Amount"] = scheme_df_display["Amount"].apply(lambda x: f"₹{x:,.0f}")
            st.dataframe(scheme_df_display)

            # Pie Chart
            st.subheader("📊 Allocation Pie Chart")
            fig_pie = px.pie(
                scheme_df,
                names="Scheme Name",
                values="Amount",
                title="Portfolio Distribution by Scheme",
                hole=0.4
            )
            st.plotly_chart(fig_pie)

            # Download CSV Button
            csv = scheme_df_display.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Portfolio as CSV",
                data=csv,
                file_name='recommended_portfolio.csv',
                mime='text/csv'
            )

            st.success(f"✅ Portfolio contains {len(scheme_df)} schemes (Max allowed: {MAX_FUNDS})")
        else:
            st.warning("⚠️ No valid fund matches found for the selected profile and universe.")

    else:
        st.warning("⚠️ Please complete the risk profiling and asset allocation steps above to generate a portfolio recommendation.")

    # Footer
    st.markdown("---")
    st.markdown("Made with ❤️ by Om Apex • Powered by Smart Logic & Advisor Insights")
