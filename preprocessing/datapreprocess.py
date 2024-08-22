import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split


def missing_check(df, pre=True):
    """
    参数：
    - missing_check(df)
    - df: DataFrame, 待检查的DataFrame

    返回:
    - DataFrame中缺失值和缺失值占比, 并打印到屏幕上
    """
    missing_df = pd.DataFrame(df.isnull().sum(), columns=["missing_count"])
    missing_df["missing_ratio"] = missing_df["missing_count"] / len(df)
    if pre:
        print(missing_df)
    return missing_df


def missing_fill(df, way_dict):
    """
    参数：
    - df: DataFrame, 待填充的DataFrame
    - way_dict: dict, 填充方式字典, key为列名, value为填充方式, 可选值有mean, median, - mode, ffill, bfill, 或者数字

    返回:
    - DataFrame:判断缺失值是否填充完毕, 并返回填充后的DataFrame
    """
    df_filled = df.copy()
    for key, value in way_dict.items():
        if value == "mean":
            df_filled[key] = df_filled[key].fillna(df_filled[key].mean())
        if value == "median":
            df_filled[key] = df_filled[key].fillna(df_filled[key].median())
        if value == "mode":
            df_filled[key] = df_filled[key].fillna(df_filled[key].mode()[0])
        if value == "ffill":
            df_filled[key] = df_filled[key].ffill()
        if value == "bfill":
            df_filled[key] = df_filled[key].bfill()
        if value.isdigit():
            df_filled[key] = df_filled[key].fillna(value)

    check = missing_check(df_filled, pre=False)
    if check.missing_ratio.sum() > 0:
        print("缺失值仍然存在！请检查！")
    else:
        print("缺失值检查通过！")
    return df_filled


def data_describe(df) -> None:
    """
    参数：
    - df: DataFrame, 待描述的DataFrame

    返回:
    - None, 打印df的描述性统计信息
    """
    # 基本信息
    print(f"基本信息:")
    print(df.info(), sep="\n")
    print("--------------------")

    # 描述性统计信息
    print(f"描述性统计信息：")
    print(df.describe())
    print("--------------------")

    # 提取描述性统计信息的转置（方便添加偏度和峰度）
    math_info = df.describe().T

    # 只处理数值型的列
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns

    # 计算偏度和峰度并加入到描述性统计信息中
    math_info["skew"] = df[numeric_cols].skew()
    math_info["kurt"] = df[numeric_cols].kurt()

    # 输出并返回数学统计信息
    print(f"数学统计信息：")
    print(math_info)


def replace_outliers(df, column, method="median"):
    """
    用于异常值替换的函数。支持中位数、均值、前后值插值和线性插值替换方法。

    参数:
    - df: DataFrame,要处理的数据表。
    - column: str,要处理的列名。
    - method: str,替换方法,可选 'median'（中位数替换）, 'mean'（均值替换）, 'ffill'（前向填充）, 'bfill'（后向填充）, 'linear'（线性插值）。

    返回:
    - df: DataFrame,已处理的DataFrame。
    """

    # 计算IQR来识别异常值的上下界
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 标记异常值为NaN
    df[column] = df[column].apply(
        lambda x: np.nan if x < lower_bound or x > upper_bound else x
    )

    if method == "median":
        # 使用中位数替换NaN值
        median_value = df[column].median()
        df[column].fillna(median_value, inplace=True)

    elif method == "mean":
        # 使用均值替换NaN值
        mean_value = df[column].mean()
        df[column].fillna(mean_value, inplace=True)

    elif method == "ffill":
        # 使用前一个有效值替换NaN值
        df[column].fillna(method="ffill", inplace=True)

    elif method == "bfill":
        # 使用后一个有效值替换NaN值
        df[column].fillna(method="bfill", inplace=True)

    elif method == "linear":
        # 使用线性插值替换NaN值
        x = np.arange(len(df))
        y = df[column].values
        mask = ~np.isnan(y)
        f = interp1d(x[mask], y[mask], kind="linear", fill_value="extrapolate")
        df[column] = f(x)

    else:
        raise ValueError(
            "属于replace_outliers函数的method参数只能是 'median', 'mean', 'ffill', 'bfill', 'linear'."
        )

    return df


def uniform_data(df: pd.DataFrame, col_operations: dict) -> pd.DataFrame:
    """
    对DataFrame进行数据一致化处理并进行无量纲化。

    参数:
    - df (pd.DataFrame): 需要处理的DataFrame。
    - col_operations (dict): 一个字典，键是列名，值是一个列表。
        列表的第一个元素是操作类型，可以是'min'、'max'、'mid'或'range'。
        如果操作类型为'range'，列表的后两个元素应为对应区间[a, b]。

    返回值:
    - pd.DataFrame: 处理后的DataFrame。
    """

    # 定义极小型操作
    def min_operation(col):
        return col.min(), col.max(), (col.max() - col)

    # 定义极大型操作
    def max_operation(col):
        return col.min(), col.max(), col

    # 定义中间型操作
    def mid_operation(col):
        min_val, max_val = col.min(), col.max()
        mid_val = (min_val + max_val) / 2
        return min_val, max_val, 1 - abs(col - mid_val) / (mid_val - min_val)

    # 定义区间型操作
    def range_operation(col, a, b):
        min_val, max_val = col.min(), col.max()
        c = max(a - min_val, max_val - b)

        def apply_range(x):
            if a <= x <= b:
                return 1
            elif x < a:
                return 1 - (a - x) / c
            else:
                return 1 - (x - b) / c

        return a, b, col.apply(apply_range)

    # 操作类型与对应函数的映射
    operations = {
        "min": min_operation,
        "max": max_operation,
        "mid": mid_operation,
        "range": range_operation,
    }

    processed_df = df.copy()

    # 对每一列按照指定的操作进行处理
    for col, ops in col_operations.items():
        operation = ops[0]
        if operation == "range":
            a, b = ops[1], ops[2]
            _, _, processed_df[col] = operations[operation](processed_df[col], a, b)
        else:
            _, _, processed_df[col] = operations[operation](processed_df[col])

    return processed_df


def standard_scale(df: pd.DataFrame, way: str = "standard") -> pd.DataFrame:
    """
    对DataFrame进行标准化处理。

    参数:
    - df (pd.DataFrame): 需要处理的DataFrame。
    - way (str): 标准化方法，可选'standard'（标准化）或'minmax'（最小最大值标准化）。

    返回值:
    - pd.DataFrame: 处理并标准化后的DataFrame。
    """
    if way == "standard":
        scaler = StandardScaler()
    elif way == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("属于standard_scale函数的way参数只能是'standard'或'minmax'.")
    scaled_df = pd.DataFrame(
        scaler.fit_transform(df),
        columns=df.select_dtypes(include=["float64", "int64"]).columns,
        index=df.index,
    )
    return scaled_df


def split(
    df: pd.DataFrame,
    target: str,
    test_size=0.3,
    random_state: int = 42,
) -> tuple:
    """
    对DataFrame进行训练集和测试集的划分。

    参数:
    - df (pd.DataFrame): 需要划分的DataFrame。
    - test_size (float): 测试集占比，默认0.3。
    - random_state (int): 随机种子，默认42。

    返回值:
    - tuple: 训练集和测试集。
    """
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(target, axis=1),
        df[target],
        test_size=test_size,
        random_state=random_state,
    )
    return X_train, X_test, y_train, y_test


def feature_selection(X_train: pd.DataFrame, y: pd.Series, way="all") -> dict:
    """
    特征选择函数，根据指定的方法选择特征，并打印各自的选择结果。

    参数:
    - X_train: DataFrame，训练特征数据。
    - y: Series，目标变量。
    - way: str，特征选择方法。可选 'RandomForest'（随机森林）、'ForwardSelection'（向前逐步回归）、'Lasso'（Lasso回归）、'all'（所有方法）。

    返回:
    - results: dict，包含各方法选择的特征。
    """
    results = {}

    if way in ["RandomForest", "all"]:
        print("RandomForest特征选择结果:")
        rf = RandomForestClassifier()
        rf.fit(X_train, y)
        importances = rf.feature_importances_
        feature_importances = pd.Series(importances, index=X_train.columns).sort_values(
            ascending=False
        )
        print(feature_importances)
        print("Accuracy:", rf.score(X_train, y))
        print("---------")
        results["RandomForest"] = [feature_importances.index.tolist(), rf]

    if way in ["ForwardSelection", "all"]:
        print("向前逐步回归特征选择结果:")
        X_train_std = standard_scale(X_train)
        sfs = SequentialFeatureSelector(
            LinearRegression(), n_features_to_select="auto", direction="forward"
        )
        sfs.fit(X_train_std, y)
        selected_features = X_train_std.columns[sfs.get_support()]

        # 使用选择的特征训练新的模型并评分
        X_train_selected = X_train_std[selected_features]
        rf_selected = RandomForestClassifier()
        rf_selected.fit(X_train_selected, y)
        accuracy = rf_selected.score(X_train_selected, y)

        print(selected_features)
        print("Accuracy:", accuracy)
        print("---------")
        results["ForwardSelection"] = [selected_features.tolist(), rf_selected]

    if way in ["Lasso", "all"]:
        print("Lasso回归特征选择结果:")
        X_train_std = standard_scale(X_train)
        lasso = LassoCV()
        lasso.fit(X_train_std, y)
        coefs = lasso.coef_
        selected_features = X_train_std.columns[abs(coefs) > 0]
        print(pd.Series(coefs, index=X_train_std.columns).sort_values(ascending=False))
        
        # 使用选择的特征训练新的模型并评分
        X_train_selected = X_train_std[selected_features]
        lasso.fit(X_train_selected, y)
        accuracy = lasso.score(X_train_selected, y)
        print("Accuracy:", accuracy)
        print("---------")
        results["Lasso"] = [selected_features.tolist(), lasso]

    return results
