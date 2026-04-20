import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import BayesianRidge
from scipy.optimize import minimize
import logging
import warnings
import os

logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
logging.getLogger('prophet').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

# ==================== 彩色样式配置 ====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义协调的配色方案
COLORS = {
    'truth': '#1f77b4',  # 真实值：蓝色
    'prophet': '#ff7f0e',  # Prophet：橙色
    'svr': '#2ca02c',  # SVR：绿色
    'dbn': '#d62728',  # DBN：红色
    'fusion': '#9467bd',  # 融合模型：紫色
    'grid': '#cccccc',  # 网格线：浅灰色
    'fill': '#d3d3d3'  # 填充：浅灰色
}


def calculate_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true > 0
    if np.sum(mask) == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


class RollingTripleModelFusion:
    def __init__(self, data_path, test_size=0.3):
        self.data_path = data_path
        self.test_size = test_size
        self.target_col = 'total_cases_per_million'

    def load_data(self):
        print(f"📂 加载数据: {self.data_path}")
        if not os.path.exists(self.data_path):
            print(f"⚠️ 文件不存在: {self.data_path}")
            return None

        try:
            df = pd.read_excel(self.data_path, engine='openpyxl')
        except Exception as e:
            print(f"⚠️ 读取文件失败: {e}")
            return None

        if 'year_month' not in df.columns:
            print(f"⚠️ 数据中缺少 'year_month' 列")
            return None
        if self.target_col not in df.columns:
            print(f"⚠️ 数据中缺少 '{self.target_col}' 列")
            return None

        df = df.sort_values('year_month').reset_index(drop=True)
        df['ds'] = pd.to_datetime(df['year_month'] + '-01')
        df[self.target_col] = df[self.target_col].ffill()
        df = df.dropna(subset=[self.target_col])
        df['y_smooth'] = df[self.target_col].ewm(span=3, adjust=False).mean()
        df['y_log'] = np.log1p(df['y_smooth'])

        print(f"✅ 数据加载成功，共 {len(df)} 条记录")
        return df

    def _predict_prophet(self, train_df, next_date):
        if len(train_df) < 5:
            return train_df['y_log'].iloc[-1]
        try:
            df_prophet = train_df[['ds', 'y_log']].rename(columns={'y_log': 'y'}).copy()
            model = Prophet(growth='linear', changepoint_prior_scale=0.5,
                            yearly_seasonality=len(df_prophet) > 24,
                            weekly_seasonality=False, daily_seasonality=False)
            model.fit(df_prophet)
            future = pd.DataFrame({'ds': [next_date]})
            return model.predict(future)['yhat'].values[0]
        except:
            return train_df['y_log'].iloc[-1]

    def _predict_svr(self, train_vals):
        if len(train_vals) < 5:
            return train_vals[-1]
        diff_data = np.diff(train_vals)
        if len(diff_data) < 5:
            return train_vals[-1]
        X, y = [], []
        lb = min(3, len(diff_data) - 1)
        for i in range(lb, len(diff_data)):
            X.append(diff_data[i - lb:i])
            y.append(diff_data[i])
        X, y = np.array(X), np.array(y)
        sc_x = StandardScaler()
        sc_y = StandardScaler()
        try:
            X_sc = sc_x.fit_transform(X)
            y_sc = sc_y.fit_transform(y.reshape(-1, 1)).ravel()
            best_model = SVR(kernel='rbf', C=10.0, gamma='scale', epsilon=0.01)
            best_model.fit(X_sc, y_sc)
            last_seq = diff_data[-lb:].reshape(1, -1)
            pred_diff = sc_y.inverse_transform(best_model.predict(sc_x.transform(last_seq)).reshape(-1, 1))[0][0]
            return train_vals[-1] + pred_diff
        except:
            return train_vals[-1]

    def _predict_dbn(self, train_vals):
        min_data_len = 5
        if len(train_vals) < min_data_len:
            return train_vals[-1]
        diff_data = np.diff(train_vals)
        look_back = 3
        if len(diff_data) <= look_back:
            return train_vals[-1]
        X, y = [], []
        for i in range(look_back, len(diff_data)):
            X.append(diff_data[i - look_back:i])
            y.append(diff_data[i])
        X = np.array(X)
        y = np.array(y)
        model = BayesianRidge()
        try:
            model.fit(X, y)
            last_seq = diff_data[-look_back:].reshape(1, -1)
            pred_diff = model.predict(last_seq)
            return train_vals[-1] + pred_diff[0]
        except:
            return train_vals[-1]

    def train_stacking_model(self, train_df):
        print("🧠 正在进行三模型权重优化 (使用 SLSQP 科学优化算法)...")
        n_val = max(5, int(len(train_df) * 0.3))
        train_sub = train_df.iloc[:-n_val]
        val_sub = train_df.iloc[-n_val:]
        history = train_sub.copy()
        preds = {'prophet': [], 'svr': [], 'dbn': []}
        truths = val_sub[self.target_col].values

        for i in range(len(val_sub)):
            curr_ds = val_sub['ds'].iloc[i]
            p = np.expm1(self._predict_prophet(history, curr_ds))
            s = np.expm1(self._predict_svr(history['y_log'].values))
            d = np.expm1(self._predict_dbn(history['y_log'].values))
            preds['prophet'].append(p)
            preds['svr'].append(s)
            preds['dbn'].append(d)
            history = pd.concat([history, val_sub.iloc[[i]]], ignore_index=True)

        P = np.array(preds['prophet'])
        S = np.array(preds['svr'])
        D = np.array(preds['dbn'])
        Y = truths

        def loss_func(weights):
            w1, w2, w3 = weights
            y_pred = w1 * P + w2 * S + w3 * D
            bias = np.mean(Y - y_pred)
            rmse = np.sqrt(np.mean((Y - y_pred - bias) ** 2))
            return rmse + 20 * np.sum(weights ** 2)

        bounds = ((0.1, 0.8), (0.1, 0.8), (0.1, 0.8))
        constraints = {'type': 'eq', 'fun': lambda w: 1 - sum(w)}
        res = minimize(loss_func, [1 / 3, 1 / 3, 1 / 3], method='SLSQP', bounds=bounds, constraints=constraints)
        best_weights = res.x
        final_fusion = best_weights[0] * P + best_weights[1] * S + best_weights[2] * D
        base_bias = np.mean(Y - final_fusion)

        print(
            f"✅ 最佳参数锁定! 权重: Prophet={best_weights[0]:.3f} SVR={best_weights[1]:.3f} DBN={best_weights[2]:.3f}")

        class OptimizedFusion:
            def __init__(self, w, b):
                self.weights = w
                self.base_bias = b

            def predict(self, p, s, d):
                return self.weights[0] * p + self.weights[1] * s + self.weights[2] * d + self.base_bias

        return OptimizedFusion(best_weights, base_bias)

    def run_rolling_forecast(self):
        df = self.load_data()
        if df is None:
            return None
        n_test = max(2, int(len(df) * self.test_size))
        n_train = len(df) - n_test
        train_data = df.iloc[:n_train]
        test_data = df.iloc[n_train:]
        meta_model = self.train_stacking_model(train_data)
        history_df = train_data.copy()
        res = {'ds': [], 'truth': [], 'prophet': [], 'svr': [], 'dbn': [], 'fusion': []}
        error_buffer = []

        for i in range(len(test_data)):
            curr_date = test_data['ds'].iloc[i]
            p_real = np.expm1(self._predict_prophet(history_df, curr_date))
            s_real = np.expm1(self._predict_svr(history_df['y_log'].values))
            d_real = np.expm1(self._predict_dbn(history_df['y_log'].values))
            f_base = meta_model.predict(p_real, s_real, d_real)
            dynamic_correction = np.mean(error_buffer[-2:]) * 0.8 if len(error_buffer) > 0 else 0
            f_final = max(0, f_base + dynamic_correction)

            truth = test_data[self.target_col].iloc[i]
            res['ds'].append(curr_date)
            res['truth'].append(truth)
            res['prophet'].append(p_real)
            res['svr'].append(s_real)
            res['dbn'].append(d_real)
            res['fusion'].append(f_final)

            history_df = pd.concat([history_df, test_data.iloc[[i]]], ignore_index=True)
            error_buffer.append(truth - f_final)
            print(f"Step {i + 1}/{len(test_data)} | Fusion={f_final:.2f} | Err={truth - f_final:.2f}", end='\r')

        print()
        return pd.DataFrame(res)

    def evaluate_results(self, res_df):
        cols = ['prophet', 'svr', 'dbn', 'fusion']
        res_df[cols] = res_df[cols].clip(0)

        print("\n" + "=" * 70)
        print("📊 全部模型评估指标")
        print(f"{'模型':<10} {'R²':<8} {'RMSE':<10} {'MAE':<10} {'MAPE(%)':<10}")
        print("-" * 70)
        for c in cols:
            r2 = r2_score(res_df['truth'], res_df[c])
            rmse = np.sqrt(mean_squared_error(res_df['truth'], res_df[c]))
            mae = mean_absolute_error(res_df['truth'], res_df[c])
            mape = calculate_mape(res_df['truth'], res_df[c])
            print(f"{c.upper():<10} {r2:<8.4f} {rmse:<10.2f} {mae:<10.2f} {mape:.2f}")

        print("\n📄 Prophet 模型预测结果")
        print(res_df[['ds', 'truth', 'prophet']].round(2))
        self.plot_prophet(res_df)

        print("\n📄 SVR 模型预测结果")
        print(res_df[['ds', 'truth', 'svr']].round(2))
        self.plot_svr(res_df)

        print("\n📄 DBN 模型预测结果")
        print(res_df[['ds', 'truth', 'dbn']].round(2))
        self.plot_dbn(res_df)

        self.plot(res_df)

    # ==================== 彩色总对比图 ====================
    def plot(self, df):
        plt.figure(figsize=(14, 7), facecolor='white')

        # 真实值
        plt.plot(df['ds'], df['truth'], color=COLORS['truth'], label='真实值', lw=2.5, marker='o',
                 markersize=4, markeredgecolor='white', markeredgewidth=0.5)

        # 各模型预测
        plt.plot(df['ds'], df['prophet'], color=COLORS['prophet'], label='Prophet', lw=1.8, linestyle='--')
        plt.plot(df['ds'], df['svr'], color=COLORS['svr'], label='SVR', lw=1.8, linestyle='-.')
        plt.plot(df['ds'], df['dbn'], color=COLORS['dbn'], label='DBN', lw=1.8, linestyle=':')
        plt.plot(df['ds'], df['fusion'], color=COLORS['fusion'], label='融合模型', lw=2.5, marker='s',
                 markersize=5, markerfacecolor=COLORS['fusion'], markeredgecolor='white', markeredgewidth=0.5)

        # 填充区域
        plt.fill_between(df['ds'], df['truth'], df['fusion'], color=COLORS['fill'], alpha=0.4)

        plt.title('各模型预测对比图', fontsize=16, fontweight='bold', color='#333333')
        plt.xlabel('日期', fontsize=12, color='#333333')
        plt.ylabel('病例数', fontsize=12, color='#333333')
        plt.legend(frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=11)
        plt.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--', linewidth=0.8)
        plt.gca().set_facecolor('#fafafa')
        plt.tight_layout()
        plt.show()

    # ==================== Prophet 彩色独立图 ====================
    def plot_prophet(self, df):
        plt.figure(figsize=(12, 6), facecolor='white')
        plt.plot(df['ds'], df['truth'], color=COLORS['truth'], label='真实值', lw=2.5, marker='o',
                 markersize=4, markeredgecolor='white', markeredgewidth=0.5)
        plt.plot(df['ds'], df['prophet'], color=COLORS['prophet'], label='Prophet预测', lw=2, linestyle='--',
                 marker='s', markersize=4, markerfacecolor=COLORS['prophet'], markeredgecolor='white')
        plt.title('Prophet模型独立预测', fontsize=16, fontweight='bold', color='#333333')
        plt.xlabel('日期', fontsize=12, color='#333333')
        plt.ylabel('病例数', fontsize=12, color='#333333')
        plt.legend(frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=11)
        plt.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--', linewidth=0.8)
        plt.gca().set_facecolor('#fafafa')
        plt.tight_layout()
        plt.show()

    # ==================== SVR 彩色独立图 ====================
    def plot_svr(self, df):
        plt.figure(figsize=(12, 6), facecolor='white')
        plt.plot(df['ds'], df['truth'], color=COLORS['truth'], label='真实值', lw=2.5, marker='o',
                 markersize=4, markeredgecolor='white', markeredgewidth=0.5)
        plt.plot(df['ds'], df['svr'], color=COLORS['svr'], label='SVR预测', lw=2, linestyle='--',
                 marker='s', markersize=4, markerfacecolor=COLORS['svr'], markeredgecolor='white')
        plt.title('SVR模型独立预测', fontsize=16, fontweight='bold', color='#333333')
        plt.xlabel('日期', fontsize=12, color='#333333')
        plt.ylabel('病例数', fontsize=12, color='#333333')
        plt.legend(frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=11)
        plt.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--', linewidth=0.8)
        plt.gca().set_facecolor('#fafafa')
        plt.tight_layout()
        plt.show()

    # ==================== DBN 彩色独立图 ====================
    def plot_dbn(self, df):
        plt.figure(figsize=(12, 6), facecolor='white')
        plt.plot(df['ds'], df['truth'], color=COLORS['truth'], label='真实值', lw=2.5, marker='o',
                 markersize=4, markeredgecolor='white', markeredgewidth=0.5)
        plt.plot(df['ds'], df['dbn'], color=COLORS['dbn'], label='DBN预测', lw=2, linestyle='--',
                 marker='s', markersize=4, markerfacecolor=COLORS['dbn'], markeredgecolor='white')
        plt.title('DBN模型独立预测', fontsize=16, fontweight='bold', color='#333333')
        plt.xlabel('日期', fontsize=12, color='#333333')
        plt.ylabel('病例数', fontsize=12, color='#333333')
        plt.legend(frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=11)
        plt.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--', linewidth=0.8)
        plt.gca().set_facecolor('#fafafa')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    conflict_file = "Prophet.py"
    if os.path.exists(conflict_file):
        print("⚠️ 存在冲突文件 Prophet.py，请删除后运行")
        exit(1)

    model = RollingTripleModelFusion("data.xlsx")
    res = model.run_rolling_forecast()
    if res is not None:
        model.evaluate_results(res)