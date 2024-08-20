import pandas as pd
import numpy as np
from scipy.interpolate import interp1d


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





def replace_outliers(df, column, method='median'):
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
    df[column] = df[column].apply(lambda x: np.nan if x < lower_bound or x > upper_bound else x)

    if method == 'median':
        # 使用中位数替换NaN值
        median_value = df[column].median()
        df[column].fillna(median_value, inplace=True)
    
    elif method == 'mean':
        # 使用均值替换NaN值
        mean_value = df[column].mean()
        df[column].fillna(mean_value, inplace=True)
    
    elif method == 'ffill':
        # 使用前一个有效值替换NaN值
        df[column].fillna(method='ffill', inplace=True)
    
    elif method == 'bfill':
        # 使用后一个有效值替换NaN值
        df[column].fillna(method='bfill', inplace=True)
    
    elif method == 'linear':
        # 使用线性插值替换NaN值
        x = np.arange(len(df))
        y = df[column].values
        mask = ~np.isnan(y)
        f = interp1d(x[mask], y[mask], kind='linear', fill_value="extrapolate")
        df[column] = f(x)
    
    else:
        raise ValueError("属于replace_outliers函数的method参数只能是 'median', 'mean', 'ffill', 'bfill', 'linear'.")
    
    return df

def 
