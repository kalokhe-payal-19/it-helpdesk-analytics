import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    r2_score,
    silhouette_score
)
from mlxtend.frequent_patterns import apriori, association_rules


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="IT Helpdesk Analytics",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.dashboard-title {
    font-size: 38px;
    font-weight: 700;
    color: #17365D;
    margin-bottom: 5px;
}

.dashboard-subtitle {
    font-size: 17px;
    color: #5B6573;
    margin-bottom: 25px;
}

.section-title {
    font-size: 25px;
    font-weight: 650;
    color: #17365D;
    margin-top: 10px;
    margin-bottom: 15px;
}

.metric-card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

div[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #E2E8F0;
    padding: 15px;
    border-radius: 12px;
}

div[data-testid="stMetric"] label {
    color: #475569 !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #17365D !important;
}

div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    color: #475569 !important;
}

.stButton button {
    border-radius: 8px;
}

[data-testid="stSidebar"] {
    background-color: #17365D;
}

[data-testid="stSidebar"] * {
    color: white;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("helpdesk_tickets.csv")


@st.cache_data
def load_processed_data():
    return pd.read_csv("processed_helpdesk_tickets.csv")


df = load_data()
processed_df = load_processed_data()


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.markdown(
    """
    <div style="text-align:center;">
        <h1>🖥️</h1>
        <h2>IT Helpdesk</h2>
        <p>Analytics Dashboard</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Dashboard",
        "🧹 Data Preprocessing",
        "📊 OLAP Analysis",
        "🔗 Apriori",
        "🌳 Decision Tree",
        "🧠 Naive Bayes",
        "📈 Regression",
        "🎯 K-Means Clustering",
        "📋 Comparison"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Data Mining & Warehousing Mini Project\n\n"
    "Dataset: 6000 IT Helpdesk Tickets"
)


# ---------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="dashboard-title">IT Helpdesk Analytics</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="dashboard-subtitle">'
        'Data Warehousing and Data Mining based Helpdesk Ticket Management System'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Tickets", f"{len(df):,}")
    col2.metric("Departments", df["Department"].nunique())
    col3.metric("Issue Types", df["Issue_Type"].nunique())
    col4.metric("Technicians", df["Technician"].nunique())

    st.markdown("---")

    st.markdown(
        '<div class="section-title">Dataset Overview</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Tickets by Department")

        department_counts = (
            df["Department"]
            .value_counts()
            .sort_values(ascending=False)
        )

        st.bar_chart(department_counts)

    with c2:
        st.subheader("Tickets by Priority")

        priority_counts = (
            df["Priority"]
            .value_counts()
        )

        st.bar_chart(priority_counts)

    st.markdown("---")

    st.subheader("Issue Distribution")

    issue_counts = (
        df["Issue_Type"]
        .value_counts()
        .sort_values(ascending=False)
    )

    st.bar_chart(issue_counts)

    st.markdown("---")

    st.subheader("Project Pipeline")

    pipeline = [
        "Dataset Creation",
        "Data Preprocessing",
        "OLAP Analysis",
        "Apriori Association Rules",
        "Decision Tree",
        "Naive Bayes",
        "Linear Regression",
        "K-Means Clustering"
    ]

    for i, item in enumerate(pipeline, 1):
        st.write(f"**{i}.** {item}")


# ---------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------

elif page == "🧹 Data Preprocessing":

    st.markdown(
        '<div class="section-title">Data Preprocessing</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Missing values are handled, categorical attributes are encoded, "
        "and numerical attributes are standardized."
    )

    st.subheader("Original Dataset")

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Columns", len(df.columns))
    col3.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )

    st.dataframe(
        df.head(10),
        width="stretch"
    )

    st.subheader("Missing Values by Column")

    missing = df.isnull().sum()

    missing = missing[missing > 0]

    if len(missing) > 0:
        st.dataframe(
            missing.rename("Missing Values"),
            width="stretch"
        )
    else:
        st.success("No missing values found.")

    st.subheader("Processed Dataset")

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", f"{len(processed_df):,}")
    col2.metric("Columns", len(processed_df.columns))
    col3.metric(
        "Remaining Missing Values",
        int(processed_df.isnull().sum().sum())
    )

    st.dataframe(
        processed_df.head(10),
        width="stretch"
    )

    st.success(
        "Preprocessing completed successfully. "
        "The processed dataset contains 6000 records and no missing values."
    )


# ---------------------------------------------------------
# OLAP
# ---------------------------------------------------------

elif page == "📊 OLAP Analysis":

    st.markdown(
        '<div class="section-title">OLAP Analysis</div>',
        unsafe_allow_html=True
    )

    st.write(
        "OLAP operations are used to analyze helpdesk tickets "
        "across departments, issue types, priorities and resolution time."
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Department",
            "Issue Type",
            "Priority",
            "Department × Issue"
        ]
    )

    with tab1:

        department = (
            df.groupby("Department")
            .size()
            .sort_values(ascending=False)
        )

        st.subheader("Tickets by Department")

        st.bar_chart(department)

        st.dataframe(
            department.rename("Ticket Count"),
            width="stretch"
        )

    with tab2:

        issue = (
            df.groupby("Issue_Type")
            .size()
            .sort_values(ascending=False)
        )

        st.subheader("Tickets by Issue Type")

        st.bar_chart(issue)

        st.dataframe(
            issue.rename("Ticket Count"),
            width="stretch"
        )

    with tab3:

        priority = (
            df.groupby("Priority")
            .size()
            .sort_values(ascending=False)
        )

        st.subheader("Tickets by Priority")

        st.bar_chart(priority)

        st.dataframe(
            priority.rename("Ticket Count"),
            width="stretch"
        )

        st.subheader("Average Resolution Time by Department")

        resolution = (
            df.groupby("Department")["Resolution_Time"]
            .mean()
            .round(2)
        )

        st.dataframe(
            resolution.rename("Average Resolution Time"),
            width="stretch"
        )

    with tab4:

        pivot = pd.pivot_table(
            df,
            index="Department",
            columns="Issue_Type",
            values="Ticket_ID",
            aggfunc="count",
            fill_value=0
        )

        st.subheader("Department vs Issue Type")

        st.dataframe(
            pivot,
            width="stretch"
        )

        st.bar_chart(pivot)


# ---------------------------------------------------------
# APRIORI
# ---------------------------------------------------------

elif page == "🔗 Apriori":

    st.markdown(
        '<div class="section-title">Apriori Association Rule Mining</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Association rules identify relationships among "
        "Department, Issue Type, Device Type and Priority."
    )

    data = df[
        [
            "Department",
            "Issue_Type",
            "Device_Type",
            "Priority"
        ]
    ]

    basket = pd.get_dummies(data)

    frequent_itemsets = apriori(
        basket,
        min_support=0.05,
        use_colnames=True
    )

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=0.60
    )

    rules = rules.sort_values(
        by="lift",
        ascending=False
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Frequent Itemsets",
        len(frequent_itemsets)
    )

    col2.metric(
        "Association Rules",
        len(rules)
    )

    col3.metric(
        "Highest Lift",
        f"{rules['lift'].max():.3f}"
    )

    st.subheader("Top Association Rules")

    display_rules = rules.head(10).copy()

    display_rules["Antecedents"] = display_rules[
        "antecedents"
    ].apply(
        lambda x: ", ".join(sorted(x))
    )

    display_rules["Consequents"] = display_rules[
        "consequents"
    ].apply(
        lambda x: ", ".join(sorted(x))
    )

    display_rules = display_rules[
        [
            "Antecedents",
            "Consequents",
            "support",
            "confidence",
            "lift"
        ]
    ]

    display_rules.columns = [
        "Antecedents",
        "Consequents",
        "Support",
        "Confidence",
        "Lift"
    ]

    st.dataframe(
        display_rules.round(3),
        width="stretch"
    )

    st.subheader("Frequent Itemsets")

    itemsets_display = frequent_itemsets.copy()

    itemsets_display["Itemsets"] = itemsets_display[
        "itemsets"
    ].apply(
        lambda x: ", ".join(sorted(x))
    )

    itemsets_display = itemsets_display[
        ["Itemsets", "support"]
    ]

    itemsets_display.columns = [
        "Itemsets",
        "Support"
    ]

    st.dataframe(
        itemsets_display.head(20).round(3),
        width="stretch"
    )


# ---------------------------------------------------------
# DECISION TREE
# ---------------------------------------------------------

elif page == "🌳 Decision Tree":

    st.markdown(
        '<div class="section-title">Decision Tree Classification</div>',
        unsafe_allow_html=True
    )

    df_model = df.copy()

    df_model["Response_Time"] = (
        df_model["Response_Time"]
        .fillna(df_model["Response_Time"].mean())
    )

    df_model["Customer_Satisfaction"] = (
        df_model["Customer_Satisfaction"]
        .fillna(df_model["Customer_Satisfaction"].mean())
    )

    features = [
        "Department",
        "Issue_Type",
        "Device_Type",
        "Technician",
        "Response_Time",
        "Resolution_Time",
        "Status",
        "Reopened",
        "Customer_Satisfaction",
        "Number_of_Interactions"
    ]

    X = pd.get_dummies(df_model[features])
    y = df_model["Priority"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = DecisionTreeClassifier(
        random_state=42,
        max_depth=5
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    col1, col2, col3 = st.columns(3)

    col1.metric("Training Records", len(X_train))
    col2.metric("Testing Records", len(X_test))
    col3.metric("Accuracy", f"{accuracy * 100:.2f}%")

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    cm_df = pd.DataFrame(
        cm,
        index=model.classes_,
        columns=model.classes_
    )

    st.dataframe(
        cm_df,
        width="stretch"
    )

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(3),
        width="stretch"
    )

    st.subheader("Decision Tree Visualization")

    fig, ax = plt.subplots(figsize=(18, 9))

    plot_tree(
        model,
        feature_names=X.columns,
        class_names=model.classes_,
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=8,
        ax=ax
    )

    ax.set_title(
        "IT Helpdesk Priority - Decision Tree"
    )

    st.pyplot(fig)

    plt.close(fig)


# ---------------------------------------------------------
# NAIVE BAYES
# ---------------------------------------------------------

elif page == "🧠 Naive Bayes":

    st.markdown(
        '<div class="section-title">Naive Bayes Classification</div>',
        unsafe_allow_html=True
    )

    df_model = df.copy()

    df_model["Response_Time"] = (
        df_model["Response_Time"]
        .fillna(df_model["Response_Time"].mean())
    )

    df_model["Customer_Satisfaction"] = (
        df_model["Customer_Satisfaction"]
        .fillna(df_model["Customer_Satisfaction"].mean())
    )

    features = [
        "Department",
        "Issue_Type",
        "Device_Type",
        "Technician",
        "Response_Time",
        "Resolution_Time",
        "Status",
        "Reopened",
        "Customer_Satisfaction",
        "Number_of_Interactions"
    ]

    X = pd.get_dummies(df_model[features])
    y = df_model["Priority"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = GaussianNB()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    col1, col2, col3 = st.columns(3)

    col1.metric("Training Records", len(X_train))
    col2.metric("Testing Records", len(X_test))
    col3.metric("Accuracy", f"{accuracy * 100:.2f}%")

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    cm_df = pd.DataFrame(
        cm,
        index=model.classes_,
        columns=model.classes_
    )

    st.dataframe(
        cm_df,
        width="stretch"
    )

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(3),
        width="stretch"
    )

    st.success(
        "Naive Bayes classification completed successfully."
    )


# ---------------------------------------------------------
# REGRESSION
# ---------------------------------------------------------

elif page == "📈 Regression":

    st.markdown(
        '<div class="section-title">Linear Regression</div>',
        unsafe_allow_html=True
    )

    df_model = df.copy()

    df_model["Response_Time"] = (
        df_model["Response_Time"]
        .fillna(df_model["Response_Time"].mean())
    )

    df_model["Customer_Satisfaction"] = (
        df_model["Customer_Satisfaction"]
        .fillna(df_model["Customer_Satisfaction"].mean())
    )

    features = [
        "Department",
        "Issue_Type",
        "Device_Type",
        "Priority",
        "Response_Time",
        "Status",
        "Reopened",
        "Customer_Satisfaction",
        "Number_of_Interactions"
    ]

    X = pd.get_dummies(df_model[features])
    y = df_model["Resolution_Time"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model = LinearRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    col1, col2, col3 = st.columns(3)

    col1.metric("Training Records", len(X_train))
    col2.metric("Testing Records", len(X_test))
    col3.metric("R² Score", f"{r2:.4f}")

    st.metric(
        "Mean Squared Error",
        f"{mse:.2f}"
    )

    st.subheader("Actual vs Predicted Resolution Time")

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(
        y_test,
        y_pred,
        alpha=0.55
    )

    minimum = min(y_test.min(), y_pred.min())
    maximum = max(y_test.max(), y_pred.max())

    ax.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--"
    )

    ax.set_xlabel("Actual Resolution Time")
    ax.set_ylabel("Predicted Resolution Time")
    ax.set_title(
        "Actual vs Predicted Resolution Time"
    )

    ax.grid(alpha=0.25)

    st.pyplot(fig)

    plt.close(fig)

    st.success(
        "Linear regression completed successfully."
    )


# ---------------------------------------------------------
# K-MEANS
# ---------------------------------------------------------

elif page == "🎯 K-Means Clustering":

    st.markdown(
        '<div class="section-title">K-Means Clustering</div>',
        unsafe_allow_html=True
    )

    df_model = df.copy()

    df_model["Response_Time"] = (
        df_model["Response_Time"]
        .fillna(df_model["Response_Time"].mean())
    )

    df_model["Customer_Satisfaction"] = (
        df_model["Customer_Satisfaction"]
        .fillna(df_model["Customer_Satisfaction"].mean())
    )

    features = [
        "Response_Time",
        "Resolution_Time",
        "Customer_Satisfaction",
        "Number_of_Interactions"
    ]

    X = df_model[features]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    model = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    df_model["Cluster"] = model.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        df_model["Cluster"]
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Number of Clusters", "3")
    col2.metric(
        "Silhouette Score",
        f"{score:.3f}"
    )
    col3.metric(
        "Total Tickets",
        f"{len(df_model):,}"
    )

    st.subheader("Cluster Counts")

    counts = (
        df_model["Cluster"]
        .value_counts()
        .sort_index()
    )

    st.bar_chart(counts)

    st.dataframe(
        counts.rename("Ticket Count"),
        width="stretch"
    )

    st.subheader("Cluster Characteristics")

    characteristics = (
        df_model
        .groupby("Cluster")[features]
        .mean()
        .round(2)
    )

    st.dataframe(
        characteristics,
        width="stretch"
    )

    st.subheader("Cluster Visualization")

    fig, ax = plt.subplots(figsize=(9, 6))

    scatter = ax.scatter(
        df_model["Response_Time"],
        df_model["Resolution_Time"],
        c=df_model["Cluster"],
        alpha=0.6
    )

    ax.set_xlabel("Response Time")
    ax.set_ylabel("Resolution Time")
    ax.set_title(
        "IT Helpdesk Tickets - K-Means Clustering"
    )

    ax.grid(alpha=0.25)

    st.pyplot(fig)

    plt.close(fig)

    st.success(
        "K-Means clustering completed successfully."
    )
# ---------------------------------------------------------
# COMPARISON
# ---------------------------------------------------------

elif page == "📋 Comparison":

    st.markdown(
        '<div class="section-title">Algorithm Comparison</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Comparison of the data mining techniques used in the "
        "IT Helpdesk Analytics project."
    )

    # Decision Tree
    df_model = df.copy()

    df_model["Response_Time"] = (
        df_model["Response_Time"]
        .fillna(df_model["Response_Time"].mean())
    )

    df_model["Customer_Satisfaction"] = (
        df_model["Customer_Satisfaction"]
        .fillna(df_model["Customer_Satisfaction"].mean())
    )

    classification_features = [
        "Department",
        "Issue_Type",
        "Device_Type",
        "Technician",
        "Response_Time",
        "Resolution_Time",
        "Status",
        "Reopened",
        "Customer_Satisfaction",
        "Number_of_Interactions"
    ]

    X = pd.get_dummies(df_model[classification_features])
    y = df_model["Priority"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    tree = DecisionTreeClassifier(
        random_state=42,
        max_depth=5
    )

    tree.fit(X_train, y_train)

    tree_pred = tree.predict(X_test)

    tree_accuracy = accuracy_score(
        y_test,
        tree_pred
    )

    # Naive Bayes
    nb = GaussianNB()

    nb.fit(X_train, y_train)

    nb_pred = nb.predict(X_test)

    nb_accuracy = accuracy_score(
        y_test,
        nb_pred
    )

    # Linear Regression
    regression_features = [
        "Department",
        "Issue_Type",
        "Device_Type",
        "Priority",
        "Response_Time",
        "Status",
        "Reopened",
        "Customer_Satisfaction",
        "Number_of_Interactions"
    ]

    X_reg = pd.get_dummies(
        df_model[regression_features]
    )

    y_reg = df_model["Resolution_Time"]

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg,
        y_reg,
        test_size=0.20,
        random_state=42
    )

    regression = LinearRegression()

    regression.fit(
        X_train_reg,
        y_train_reg
    )

    regression_pred = regression.predict(
        X_test_reg
    )

    regression_mse = mean_squared_error(
        y_test_reg,
        regression_pred
    )

    regression_r2 = r2_score(
        y_test_reg,
        regression_pred
    )

    # K-Means
    clustering_features = [
        "Response_Time",
        "Resolution_Time",
        "Customer_Satisfaction",
        "Number_of_Interactions"
    ]

    X_cluster = df_model[clustering_features]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X_cluster
    )

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    clusters = kmeans.fit_predict(X_scaled)

    silhouette = silhouette_score(
        X_scaled,
        clusters
    )

    # Apriori
    apriori_data = df[
        [
            "Department",
            "Issue_Type",
            "Device_Type",
            "Priority"
        ]
    ]

    basket = pd.get_dummies(
        apriori_data
    )

    frequent_itemsets = apriori(
        basket,
        min_support=0.05,
        use_colnames=True
    )

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=0.60
    )

    highest_lift = rules["lift"].max()

    # Comparison table
    comparison = pd.DataFrame({
        "Algorithm": [
            "Apriori",
            "Decision Tree",
            "Naive Bayes",
            "Linear Regression",
            "K-Means"
        ],
        "Type": [
            "Association Mining",
            "Classification",
            "Classification",
            "Regression",
            "Clustering"
        ],
        "Primary Metric": [
            "Highest Lift",
            "Accuracy",
            "Accuracy",
            "R² Score",
            "Silhouette Score"
        ],
        "Value": [
            round(highest_lift, 3),
            round(tree_accuracy * 100, 2),
            round(nb_accuracy * 100, 2),
            round(regression_r2, 4),
            round(silhouette, 3)
        ]
    })

    st.subheader("Overall Comparison")

    st.dataframe(
        comparison,
        width="stretch",
        hide_index=True
    )

    st.markdown("---")

    st.subheader("Classification Comparison")

    classification_comparison = pd.DataFrame({
        "Algorithm": [
            "Decision Tree",
            "Naive Bayes"
        ],
        "Accuracy": [
            tree_accuracy * 100,
            nb_accuracy * 100
        ]
    })

    st.bar_chart(
        classification_comparison.set_index("Algorithm")
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Regression Performance")

        st.metric(
            "R² Score",
            f"{regression_r2:.4f}"
        )

        st.metric(
            "Mean Squared Error",
            f"{regression_mse:.2f}"
        )

    with col2:
        st.subheader("Clustering Performance")

        st.metric(
            "Silhouette Score",
            f"{silhouette:.3f}"
        )

        st.metric(
            "Number of Clusters",
            "3"
        )

    st.markdown("---")

    st.subheader("Apriori Performance")

    st.metric(
        "Highest Lift",
        f"{highest_lift:.3f}"
    )

    st.info(
        "Higher values are generally better for classification accuracy, "
        "R² score, lift and silhouette score. MSE is better when lower."
    )

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.caption(
    "IT Helpdesk Analytics • Python • DWM Mini Project"
)
