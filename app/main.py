import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Предсказание стоимости автомобилей", layout="wide")

RUB_SYMBOL = "₽"

# Взял из train датасета
DEFAULT_MEDIANS = {
    "mileage_kmpl": 19.33,
    "engine_cc": 1248.0,
    "max_power_bhp": 82.0,
    "torque_nm": 170.0,
    "torque_rpm_min": 2000.0,
    "torque_rpm_max": 3000.0,
    "seats": 5,
}

# Тоже из train датасета
KNOWN_CAR_MANUFACTURERS = [
    "Ambassador",
    "Ashok Leyland",
    "Audi",
    "BMW",
    "Chevrolet",
    "Daewoo",
    "Datsun",
    "Fiat",
    "Force Gurkha",
    "Force One",
    "Ford",
    "Honda",
    "Hyundai",
    "Isuzu",
    "Jaguar",
    "Jeep",
    "Kia",
    "Land Rover",
    "Lexus",
    "MG Hector",
    "Mahindra",
    "Maruti",
    "Mercedes-Benz",
    "Mitsubishi",
    "Nissan",
    "Opel",
    "Peugeot",
    "Renault",
    "Skoda",
    "Tata",
    "Toyota",
    "Volkswagen",
    "Volvo",
]

# Это сгенерировал ChatGPT
TORQUE_PATTERN = re.compile(
    r"""
    ^\s*
    (?P<torque>\d+(?:\.\d+)?)                        # main torque number
    (?:\s*\(\s*(?P<torque_alt>\d+(?:\.\d+)?)\s*\))?  # optional (alt torque)
    \s*
    (?P<unit>nm|kgm)?                                # optional unit right after number
    \s*
    (?:                                              # optional RPM block
        (?:@|at|/)                                   # "@", "at" or "/"
        \s*
        (?P<rpm1>\d[\d,]*(?:\.\d+)?)                 # first rpm, allows "4,500"
        \s*
        (?:rpm)?                                     # optional "rpm"
        \s*
        (?:                                          # optional second rpm
            (?:[-~]|\+/-)                            # "-", "~" or "+/-"
            \s*
            (?P<rpm2>\d[\d,]*(?:\.\d+)?)             # second rpm, also with comma
            \s*
            (?:rpm)?                                 # optional "rpm"
        )?
    )?
    .*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

CAT_FEATURES = [
    "fuel",
    "seller_type",
    "transmission",
    "owner",
    "seats",
    "manufacturer",
]
NUM_FEATURES = [
    "year",
    "km_driven",
    "mileage_kmpl",
    "engine_cc",
    "max_power_bhp",
    "torque_nm",
    "torque_rpm_min",
    "torque_rpm_max",
]


def parse_torque(df: pd.DataFrame) -> pd.DataFrame:
    parsed = df["torque"].astype(str).str.extract(TORQUE_PATTERN)

    for col in ["rpm1", "rpm2"]:
        parsed[col] = parsed[col].str.replace(",", "", regex=False)

    # Convert to numeric
    for col in ["torque", "torque_alt", "rpm1", "rpm2"]:
        parsed[col] = pd.to_numeric(parsed[col], errors="coerce")

    # Detect unit
    torque_text = df["torque"].astype(str).str.lower()
    is_kgm = torque_text.str.contains("kgm", na=False)
    is_nm = torque_text.str.contains("nm", na=False)

    # Prefer regex unit if present, otherwise infer
    unit = parsed["unit"].str.lower()
    unit = np.where(is_kgm, "kgm", np.where(is_nm, "nm", unit))

    # Base torque value: main → alt
    base_torque = parsed["torque"].fillna(parsed["torque_alt"])

    # Convert everything to Nm
    df["torque_nm"] = np.where(
        unit == "kgm",
        base_torque * 9.80665,  # kgm -> Nm
        base_torque,  # default: already Nm (or unknown)
    )

    # RPM min/max
    rpm1 = parsed["rpm1"]
    rpm2 = parsed["rpm2"]

    rpm_min = rpm1.copy()
    rpm_max = rpm1.copy()

    has_rpm2 = rpm2.notna()
    rpm_min[has_rpm2] = np.minimum(rpm1[has_rpm2], rpm2[has_rpm2])
    rpm_max[has_rpm2] = np.maximum(rpm1[has_rpm2], rpm2[has_rpm2])

    df["torque_rpm_min"] = rpm_min
    df["torque_rpm_max"] = rpm_max
    return df


def assign_manufacturer(x: str) -> str | None:
    for manufacturer in KNOWN_CAR_MANUFACTURERS:
        if str(x).startswith(manufacturer.strip()):
            return manufacturer
    return None


@st.cache_data
def parse_features(df: pd.DataFrame) -> pd.DataFrame:
    medians = DEFAULT_MEDIANS

    df_clean = df.copy()

    # Mileage
    df_clean["mileage_kmpl"] = df_clean["mileage"]
    mileage_suffix = (
        df_clean["mileage"].astype(str).str.replace(r"^[\d,\. ]+", "", regex=True)
    )
    # Убираем суффикс у kmpl
    df_clean.loc[mileage_suffix == "kmpl", "mileage_kmpl"] = (
        df_clean.loc[mileage_suffix == "kmpl", "mileage_kmpl"]
        .astype(str)
        .str.removesuffix(" kmpl")
        .astype(float)
    )
    # Конвертируем km/kg в kmpl
    df_clean.loc[mileage_suffix == "km/kg", "mileage_kmpl"] = (
        df_clean.loc[mileage_suffix == "km/kg", "mileage_kmpl"]
        .astype(str)
        .str.removesuffix(" km/kg")
        .astype(float)
        * 1.4
    )
    df_clean["mileage_kmpl"] = (
        df_clean["mileage_kmpl"]
        .infer_objects(copy=False)
        .fillna(medians["mileage_kmpl"])
    )

    # engine
    df_clean["engine_cc"] = df_clean["engine"].astype(str).str.removesuffix(" CC")
    df_clean["engine_cc"] = pd.to_numeric(df_clean["engine_cc"], errors="coerce")
    df_clean["engine_cc"] = df_clean["engine_cc"].fillna(medians["engine_cc"])

    # max_power
    df_clean["max_power_bhp"] = (
        df_clean["max_power"].astype(str).str.removesuffix(" bhp")
    )
    df_clean["max_power_bhp"] = pd.to_numeric(
        df_clean["max_power_bhp"], errors="coerce"
    )
    df_clean["max_power_bhp"] = df_clean["max_power_bhp"].fillna(
        medians["max_power_bhp"]
    )

    # torque
    df_clean = parse_torque(df_clean)
    df_clean["torque_nm"] = df_clean["torque_nm"].fillna(medians["torque_nm"])
    df_clean["torque_rpm_min"] = df_clean["torque_rpm_min"].fillna(
        medians["torque_rpm_min"]
    )
    df_clean["torque_rpm_max"] = df_clean["torque_rpm_max"].fillna(
        medians["torque_rpm_max"]
    )

    # seats
    df_clean["seats"] = df_clean["seats"].fillna(medians["seats"])

    # Remove raw columns
    for fieldname in ["mileage", "engine", "max_power", "torque"]:
        if fieldname in df_clean.columns:
            del df_clean[fieldname]

    # manufacturer
    df_clean["manufacturer"] = df_clean["name"].apply(assign_manufacturer)

    # Convert types
    df_clean["engine_cc"] = df_clean["engine_cc"].astype(int)
    df_clean["seats"] = df_clean["seats"].astype(int)
    df_clean["torque_rpm_min"] = df_clean["torque_rpm_min"].astype(int)
    df_clean["torque_rpm_max"] = df_clean["torque_rpm_max"].astype(int)

    return df_clean


@st.cache_resource
def load_model():
    model_path = Path(__file__).parent.parent / "models" / "model.pickle"
    if not model_path.exists():
        st.error(f"Model not found: {model_path}")
        return None
    return joblib.load(model_path)


def draw_graphs(df: pd.DataFrame, predictions: np.ndarray):
    df = df.copy()
    df["predicted_price"] = predictions

    fig = plt.figure(figsize=(20, 20))

    # Distribution
    ax1 = plt.subplot(2, 2, 1)
    ax1.hist(predictions, bins=30, edgecolor="black", alpha=0.7)
    ax1.set_title("Распределение предсказанных цен")
    ax1.set_xlabel("Цена")
    ax1.set_ylabel("Частота")
    ax1.grid(True, alpha=0.3)

    # Price by 15 most popular manufacturers
    ax2 = plt.subplot(2, 2, 2)
    top_manufacturers = df["manufacturer"].value_counts().head(15).index

    df_box = df[df["manufacturer"].isin(top_manufacturers)]
    data = [
        df_box.loc[df_box["manufacturer"] == m, "predicted_price"].values
        for m in top_manufacturers
    ]

    ax2.boxplot(
        data,
        vert=False,
        showfliers=False,
    )
    ax2.set_yticklabels(top_manufacturers)

    ax2.set_title("Предсказанная цена по производителям")
    ax2.set_xlabel("Цена")
    ax2.set_ylabel("Производитель")
    ax2.grid(True, alpha=0.3, axis="x")

    # Price by year
    ax3 = plt.subplot(2, 2, 3)
    ax3.scatter(predictions, df["year"], alpha=0.6)
    ax3.set_title("Предсказанная цена по годам")
    ax3.set_xlabel("Цена")
    ax3.set_ylabel("Год")
    ax3.grid(True, alpha=0.3)

    ax4 = plt.subplot(2, 2, 4)
    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        cmap="coolwarm",
        linewidth=0.5,
        ax=ax4,
    )

    plt.tight_layout()
    return fig


def main():
    st.title("Предскзываем стоимость автомобиля при помощи ElasticNet")

    try:
        model = load_model()
    except Exception as e:
        st.error("Не удалось загрузить модель")
        st.exception(e)
        raise e

    enet = model.best_estimator_.regressor_.named_steps["enet"]
    coefs = enet.coef_
    intercept = enet.intercept_

    feature_names = model.best_estimator_.regressor_.named_steps[
        "preprocess"
    ].get_feature_names_out()

    coef_df = (
        pd.DataFrame({"feature": feature_names, "weight": coefs})
        .assign(abs_weight=lambda d: d["weight"].abs())
        .sort_values("abs_weight", ascending=False)
        .drop(columns="abs_weight")
        .reset_index(drop=True)
    )

    st.markdown(f"""
        ## Информация о модели
        **Модель**: ElasticNet\n
        **Параметры модели:**
        - Alpha: {model.best_params_["regressor__enet__alpha"]}
        - L1: {model.best_params_["regressor__enet__l1_ratio"]}
        - Intercept: {intercept}
        \n\n
        **Коэффициенты модели:**
    """)

    st.dataframe(coef_df)

    st.markdown("""
        ## Использование модели
        Для того, чтобы получить предскзание стоимости, загрузите CSV файл. Он должен иметь поля:
        - name
        - year
        - km_driven
        - fuel
        - seller_type
        - transmission
        - owner
        - mileage
        - engine
        - max_power
        - torque
        - seats
    """)

    file = st.file_uploader("Загрузите CSV файл", type=["csv"])

    if not file:
        return

    try:
        df_raw = pd.read_csv(file)

        with st.spinner("Обработка данных..."):
            df_processed = parse_features(df_raw)
            X = df_processed[CAT_FEATURES + NUM_FEATURES]
            predictions = model.predict(X)

            df_results = df_processed.copy()
            df_results["predicted_price"] = predictions

            # Display results
            st.subheader("Результат")
            display_cols = [
                "predicted_price",
                "name",
                "manufacturer",
                "year",
                "selling_price",
                "km_driven",
                "fuel",
                "seller_type",
                "transmission",
                "owner",
                "seats",
                "mileage_kmpl",
                "engine_cc",
                "max_power_bhp",
                "torque_nm",
                "torque_rpm_min",
                "torque_rpm_max",
            ]
            available_cols = [col for col in display_cols if col in df_results.columns]
            st.dataframe(
                df_results[available_cols].style.format(
                    {
                        "predicted_price": "{:,.0f} " + RUB_SYMBOL,
                        "engine_cc": "{:,.0f}",
                        "max_power_bhp": "{:.2f}",
                        "mileage_kmpl": "{:.2f}",
                        "torque_nm": "{:.2f}",
                    }
                ),
            )

            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg", f"{predictions.mean():,.0f} {RUB_SYMBOL}")
            with col2:
                st.metric(
                    "Median",
                    f"{np.median(predictions):,.0f} {RUB_SYMBOL}",
                )
            with col3:
                st.metric("Min", f"{predictions.min():,.0f} {RUB_SYMBOL}")
            with col4:
                st.metric("Max", f"{predictions.max():,.0f} {RUB_SYMBOL}")

            # Visualizations
            st.subheader("Графики")
            fig = draw_graphs(df_processed, predictions)
            st.pyplot(fig)
    except Exception as e:
        st.error("При обработке возникла ошибка")
        st.exception(e)


if __name__ == "__main__":
    main()
