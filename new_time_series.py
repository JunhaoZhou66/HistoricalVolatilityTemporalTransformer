def part1():
    """
    Part 1: 预处理阶段
    功能包括：
    1. 加载你的K线 / 分钟价格数据
    2. 使用 local_TRANGE() 计算TR（True Range）
    3. 计算未来90分钟的 ATR(90)
    4. 删除最后90行（因为未来窗口不可计算）
    5. 通过 ACF 识别季节性 periodicity（seasonal lag）
    6. 在 K=1..8 中选择最佳 Fourier 阶数（最小MAE）
    7. 计算线性趋势 slope
    8. 按 70/20/10 切分
    9. 保存处理后的数据与参数（供 part2、part3 使用）
    """

    print("\n===== PART 1: ATR + ACF + Fourier Preprocessing =====\n")

    # -------------------------------------------------------
    # 1. 加载你的价格数据 
    # -------------------------------------------------------
    # 👉【需要你改】你自己的价格文件名
    df = pd.read_csv("AMD_20240102-090000_20251108-002700.csv")  

    # 👉【需要你改】必须有时间列；如果你的列不是 time，请修改
    df = df.sort_values("time")  # <-- 按时间排序

    # -------------------------------------------------------
    # 2. 使用 local_TRANGE 来计算 TR（高低差 + 跨bar差异）
    # -------------------------------------------------------
    df["TR"] = local_TRANGE(df, {}, ["TR"])

    # -------------------------------------------------------
    # 3. 计算 ATR(90) – 使用未来窗口
    # -------------------------------------------------------
    FUTURE = 90  # 未来窗口长度 = 90分钟
    df[f"atr_{FUTURE}"] = (
        df["TR"].rolling(FUTURE).mean().shift(-FUTURE)
    )

    # 删除最后90行（因为无法计算未来窗口）
    df = df.iloc[:-FUTURE].copy()
    print("ATR created, final rows:", len(df))

    # -------------------------------------------------------
    # 4. 使用 ACF 识别季节性周期（lag）
    # -------------------------------------------------------
    close_series = df["CLOSE"].values

    acf_vals = acf(close_series, nlags=500, fft=True)
    seasonal_lag = np.argmax(acf_vals[1:]) + 1

    print("\nDetected seasonality lag (via ACF):", seasonal_lag)

    # -------------------------------------------------------
    # 5. 在 K=1..8 测试 Fourier 组合，找最佳 K
    # -------------------------------------------------------
    def fourier_features(t, period, K):
        X = []
        for k in range(1, K + 1):
            X.append(np.sin(2 * np.pi * k * t / period))
            X.append(np.cos(2 * np.pi * k * t / period))
        return np.vstack(X).T   # shape=(len(t), 2K)

    t = np.arange(len(df))
    y = df["CLOSE"].values

    best_k = 1
    best_err = 1e18

    print("\nSelecting best Fourier K = 1..8")
    for K in range(1, 9):
        X = fourier_features(t, seasonal_lag, K)
        model = Ridge().fit(X, y)
        pred = model.predict(X)
        err = mean_absolute_error(y, pred)

        print(f"  K={K}, MAE={err:.4f}")

        if err < best_err:
            best_err = err
            best_k = K

    print("\nBest Fourier K =", best_k)

    # -------------------------------------------------------
    # 6. 线性趋势：提取 slope
    # -------------------------------------------------------
    idx = np.arange(len(df))
    slope, intercept, _, _, _ = stats.linregress(idx, y)
    print("Trend slope:", slope)

    # -------------------------------------------------------
    # 7. 70 / 20 / 10 split
    # -------------------------------------------------------
    N = len(df)
    train_end = int(N * 0.7)
    val_end = int(N * 0.9)

    train = df.iloc[:train_end]
    val   = df.iloc[train_end:val_end]
    test  = df.iloc[val_end:]

    print("\nSplit Summary:")
    print("Train:", len(train))
    print("Val  :", len(val))
    print("Test :", len(test))

    # -------------------------------------------------------
    # 8. 保存供 part2 / part3 使用
    # -------------------------------------------------------
    df.to_csv("processed_price_data.csv", index=False)

    params = {
        "seasonal_lag": int(seasonal_lag),
        "best_K": int(best_k),
        "slope": float(slope),
        "intercept": float(intercept)
    }
    json.dump(params, open("fourier_params.json", "w"), indent=4)

    print("\nSaved: processed_price_data.csv + fourier_params.json")
    print("===== PART 1 DONE =====\n")
