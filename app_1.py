# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import date
import io
import re

st.set_page_config(page_title="资产配置模拟器", layout="wide")

st.title("💹 资产配置模拟器 - Efficient Frontier")

# 用户输入部分
default_tickers = "BTC-USD,ETH-USD,BNB-USD,SOL-USD"
tickers_input = st.text_input("输入资产代码（用逗号分隔!)例如：BTC-USD,ETH-USD, 600519.ss） 上证股票后面加 .ss, 深证加 .sz !, 黄金符号是 GC=F", value=default_tickers)
start_date = st.date_input("数据开始日期", value=date(2022, 12, 31))
simulations = st.number_input("MCN模拟次数（建议 10000 - 100000）越大运算时间越长，数据约准，一般50000就够了", min_value=1000, max_value=200000, value=50000,
                              step=1000)
risk_free_rate = st.number_input("无风险收益率（默认 0.04）可以自己修改，可以理解为银行利率", value=0.04, step=0.01)
display_points = st.slider("显示点的数量（较少的点会使图表更清晰）", min_value=500, max_value=10000, value=2000, step=500)

run_button = st.button("🚀 开始模拟")

if run_button:
    try:
        # 处理中英文逗号
        tickers_input = re.sub(r'，', ',', tickers_input)
        tickers = [i.strip().upper() for i in tickers_input.split(",") if i.strip()]
        tnames = [i.split('-')[0] for i in tickers]  # 取简短名

        if len(tickers) == 0:
            st.error("❌ 请输入至少一个资产代码！")
            st.stop()

        # 数据下载部分
        st.info("⏳ 正在下载数据，请稍等...")
        download_progress = st.progress(0)
        df = pd.DataFrame()

        for i, ticker in enumerate(tickers):
            try:
                data = yf.download(ticker, start=start_date, end=date.today(), progress=False)
                if data.empty:
                    st.error(f"❌ 无法下载 {ticker} 的数据，请检查代码是否正确")
                    st.stop()
                df[ticker] = data['Close']
                download_progress.progress((i + 1) / len(tickers))
            except Exception as e:
                st.error(f"❌ 下载 {ticker} 数据时出错: {str(e)}")
                st.stop()

        download_progress.empty()
        st.success(f"✅ 成功下载 {len(tickers)} 种资产的数据！")

        # 数据处理
        data = np.log(df / df.shift(1)).dropna()
        interval_days = (date.today() - start_date).days

        returns_annual = data.mean() * 365  # 改为年化计算
        cov_annual = data.cov() * 365  # 改为年化计算
        number_of_assets = len(data.columns)

        # 蒙特卡洛模拟
        st.info(f"⏳ 正在进行蒙特卡洛模拟 ({simulations} 次)...")
        simulation_progress = st.progress(0)

        results = []
        optimal_front = pd.DataFrame(columns=['returns', 'volatility', 'sharpe', 'weights'])

        for i in range(simulations):
            # 生成随机权重
            weights = np.random.random(number_of_assets)
            weights /= np.sum(weights)

            # 计算组合指标
            returns = np.dot(weights, returns_annual)
            volatility = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
            sharpe = (returns - risk_free_rate) / (volatility + 1e-10)  # 避免除以0

            # 添加到结果
            results.append({
                'returns': returns,
                'volatility': volatility,
                'sharpe': sharpe,
                'weights': weights.tolist(),  # 转换为列表以便后续处理
                'assets': tnames
            })

            # 更新进度条
            if i % (simulations // 100) == 0:
                simulation_progress.progress(i / simulations)

        simulation_progress.progress(1.0)

        # 转换为DataFrame
        results_df = pd.DataFrame(results)

        st.success("✅ 模拟完成！绘制图表中...")

        # 优化后的绘图部分 ==============================================
        # 首先筛选出真正的有效前沿点
        # 按波动率排序
        results_df = results_df.sort_values('volatility')

        # 创建空的有效前沿数据框
        ef_points = []
        max_return = -float('inf')

        # 对每个波动率水平，只保留收益率最高的点
        for vol, group in results_df.groupby(np.round(results_df['volatility'], decimals=4)):
            max_return_point = group.loc[group['returns'].idxmax()]
            if max_return_point['returns'] >= max_return:  # 确保收益率单调递增
                max_return = max_return_point['returns']
                ef_points.append(max_return_point)

        # 创建真正的有效前沿数据框
        ef_df = pd.DataFrame(ef_points)

        # 优化显示的点，只保留最优点
        # 1. 按夏普比率排序并获取前50%的点
        top_sharpe = results_df.nlargest(int(len(results_df) * 0.5), 'sharpe')
        # 2. 从剩余点中随机选择一部分点以控制总数
        remaining_points = display_points - len(top_sharpe)

        if remaining_points > 0 and len(results_df) > len(top_sharpe):
            # 从剩余点中随机抽样
            other_points = results_df.drop(top_sharpe.index).sample(
                min(remaining_points, len(results_df) - len(top_sharpe)))
            display_df = pd.concat([top_sharpe, other_points])
        else:
            display_df = top_sharpe.sample(min(display_points, len(top_sharpe)))

        # 为悬停文本准备数据
        hover_data = []
        for idx, row in display_df.iterrows():
            weights_str = "<br>".join([f"{asset}: {weight:.2%}" for asset, weight in zip(tnames, row['weights'])])
            hover_data.append(
                f"夏普比率: {row['sharpe']:.4f}<br>" +
                f"收益率: {row['returns']:.4f}<br>" +
                f"波动率: {row['volatility']:.4f}<br>" +
                f"权重:<br>{weights_str}"
            )

        display_df['hover_text'] = hover_data

        # 基本散点图：使用热力图样式展示所有点的夏普比率
        fig = px.scatter(
            display_df,
            x='volatility',
            y='returns',
            color='sharpe',
            color_continuous_scale='Viridis',
            opacity=0.5,
            title='Efficient Frontier（有效前沿）',
            labels={
                'returns': '预期收益率 (年化)',
                'volatility': '波动率 (年化)',
                'sharpe': '夏普比率'
            },
            hover_data={
                'volatility': False,
                'returns': False,
                'sharpe': False,
                'hover_text': True
            },
            custom_data=['hover_text']
        )

        # 设置悬停模板
        fig.update_traces(
            hovertemplate='%{customdata[0]}',
            marker=dict(size=8)
        )

        # 添加有效前沿线
        fig.add_scatter(
            x=ef_df['volatility'],
            y=ef_df['returns'],
            mode='lines',
            line=dict(color='red', width=3),
            name='有效前沿'
        )

        # 最佳夏普点
        max_sharpe_point = results_df.loc[results_df['sharpe'].idxmax()]
        # 为最佳点准备悬停文本
        max_sharpe_hover = "<br>".join([
                                           f"最佳组合",
                                           f"夏普比率: {max_sharpe_point['sharpe']:.4f}",
                                           f"收益率: {max_sharpe_point['returns']:.4f}",
                                           f"波动率: {max_sharpe_point['volatility']:.4f}",
                                           f"权重:"
                                       ] + [f"{asset}: {weight:.2%}" for asset, weight in
                                            zip(tnames, max_sharpe_point['weights'])])

        fig.add_scatter(
            x=[max_sharpe_point['volatility']],
            y=[max_sharpe_point['returns']],
            mode='markers+text',
            marker=dict(size=12, color='red'),
            text=["最佳组合"],
            textposition="top center",
            name="最佳组合",
            hovertemplate=max_sharpe_hover
        )

        # 最小波动率点
        min_vol_point = results_df.loc[results_df['volatility'].idxmin()]
        # 为最小波动率点准备悬停文本
        min_vol_hover = "<br>".join([
                                        f"最小波动",
                                        f"夏普比率: {min_vol_point['sharpe']:.4f}",
                                        f"收益率: {min_vol_point['returns']:.4f}",
                                        f"波动率: {min_vol_point['volatility']:.4f}",
                                        f"权重:"
                                    ] + [f"{asset}: {weight:.2%}" for asset, weight in
                                         zip(tnames, min_vol_point['weights'])])

        fig.add_scatter(
            x=[min_vol_point['volatility']],
            y=[min_vol_point['returns']],
            mode='markers+text',
            marker=dict(size=12, color='green'),
            text=["最小波动"],
            textposition="bottom center",
            name="最小波动",
            hovertemplate=min_vol_hover
        )

        # 添加无风险利率线
        max_vol = results_df['volatility'].max()
        fig.add_shape(
            type="line",
            x0=0, y0=risk_free_rate, x1=max_vol, y1=risk_free_rate,
            line=dict(color="blue", width=1, dash="dot"),
            name="无风险利率"
        )

        # 添加无风险利率到最优点的切线
        # 计算切线斜率 = 夏普比率
        slope = max_sharpe_point['sharpe']
        x0 = 0
        y0 = risk_free_rate
        x1 = max_vol
        y1 = risk_free_rate + slope * max_vol

        fig.add_shape(
            type="line",
            x0=x0, y0=y0, x1=x1, y1=y1,
            line=dict(color="darkblue", width=1, dash="dash"),
            name="资本市场线"
        )

        # 图表布局优化
        fig.update_layout(
            plot_bgcolor='white',
            hovermode='closest',
            legend=dict(
                orientation="h",  # 水平排列图例
                yanchor="bottom",
                y=1.02,  # 放在图表上方
                xanchor="center",
                x=0.5  # 居中
            ),
            margin=dict(l=50, r=50, t=80, b=50),  # 调整边距，使图表居中
            coloraxis_colorbar=dict(
                title="夏普比率",
                x=0.95  # 将颜色条移到右侧
            )
        )

        # 设置轴范围，确保图表内容居中显示
        vol_min = results_df['volatility'].min()
        vol_max = results_df['volatility'].max()
        ret_min = results_df['returns'].min()
        ret_max = results_df['returns'].max()

        # 添加一些边距使图看起来更好
        padding_x = (vol_max - vol_min) * 0.05
        padding_y = (ret_max - ret_min) * 0.05

        fig.update_xaxes(
            title="波动率 (年化)",
            range=[max(0, vol_min - padding_x), vol_max + padding_x],
            showgrid=True,
            gridcolor='LightGray'
        )

        fig.update_yaxes(
            title="收益率 (年化)",
            range=[max(0, ret_min - padding_y), ret_max + padding_y],
            showgrid=True,
            gridcolor='LightGray'
        )

        st.plotly_chart(fig, use_container_width=True)
        # =============================================================

        # 显示最佳组合
        max_sharpe = results_df.loc[results_df['sharpe'].idxmax()]
        st.subheader("🏆 最佳夏普比率组合")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("夏普比率", f"{max_sharpe['sharpe']:.4f}")
        with col2:
            st.metric("预期年化收益率", f"{max_sharpe['returns']:.4f}")
        with col3:
            st.metric("年化波动率", f"{max_sharpe['volatility']:.4f}")

        st.subheader("资产权重分配")
        weights_df = pd.DataFrame({
            '资产': tnames,
            '权重': [f"{w:.2%}" for w in max_sharpe['weights']]
        })
        st.dataframe(weights_df, hide_index=True, use_container_width=True)

        # 饼图可视化权重
        fig_pie = px.pie(
            names=tnames,
            values=max_sharpe['weights'],
            title='最佳组合资产配置',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

        # 提供下载
        st.subheader("📥 下载模拟结果表格")
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="下载完整结果（CSV格式）",
            data=csv,
            file_name='efficient_frontier_results.csv',
            mime='text/csv',
            use_container_width=True
        )

    except Exception as e:
        st.error(f"❌ 出错了: {str(e)}")
        st.exception(e)  # 显示详细错误信息，便于调试