import pandas as pd
import matplotlib.pyplot as plt
import io

log_text = """
매매일자	주문번호	매매구분	종목코드	종목명	수량	체결가격
07/24	2720	매도	000660	SK하이닉스	36	270,514
07/24	2719	매도	082740	한화엔진	313	31,750
07/24	2611	매도	000660	SK하이닉스	1	275,500
07/24	2607	매도	082740	한화엔진	10	31,850
07/23	5344	매도	082740	한화엔진	1	30,900
07/23	5287	매도	082740	한화엔진	2	30,800
07/23	5048	매도	082740	한화엔진	1	30,650
07/23	4961	매도	082740	한화엔진	3	30,550
07/23	4731	매수	082740	한화엔진	2	30,250
07/23	4678	매수	082740	한화엔진	1	30,4500                                                              
07/23	4618	매수	000660	SK하이닉스	37	269,000 
07/23	4617	매수	082740	한화엔진	327	30,450
07/23	4538	매도	090430	아모레퍼시픽	37	134,603
07/23	4537	매도	034950	한국기업평가	104	100,812
07/23	4536	매도	024110	기업은행	495	20,100
07/23	4535	매도	014830	유니드	42	85,476
07/23	4533	매도	012450	한화에어로스페이스	10	905,000
07/23	4093	매도	000660	SK하이닉스	37	268,500
07/22	6674	매도	024110	기업은행	1	20,200
07/22	6536	매수	024110	기업은행	1	20,150                                                                                                  
07/22	6049	매도	024110	기업은행	1	20,200
07/22	5956	매수	024110	기업은행	1	20,150
07/22	5839	매도	024110	기업은행	1	20,200
07/22	5678	매도	024110	기업은행	1	20,200
07/22	5551	매도	024110	기업은행	1	20,100
07/22	5437	매수	024110	기업은행	2	20,050
07/22	5289	매도	024110	기업은행	1	20,150
07/22	5133	매수	024110	기업은행	4	20,100
07/22	4847	매도	024110	기업은행	3	20,250
07/22	4674	매도	024110	기업은행	1	20,150
07/22	4519	매수	024110	기업은행	1	20,100
07/22	4369	매수	024110	기업은행	1	20,150
07/22	4215	매수	012450	한화에어로스페이스	10	919,550
07/22	4213	매수	024110	기업은행	495	20,170
07/22	4210	매수	000660	SK하이닉스	37	269,000
07/16	8585	매수	090430	아모레퍼시픽	4	135,000
07/16	8448	매도	090430	아모레퍼시픽	16	134,700
07/16	8446	매도	014830	유니드	36	87,900
07/16	8271	매수	090430	아모레퍼시픽	6	134,700
07/16	8263	매수	014830	유니드	22	87,818
07/16	8024	매수	034950	한국기업평가	9	100,333
07/16	8023	매도	090430	아모레퍼시픽	17	134,106
07/16	7800	매도	034950	한국기업평가	11	100,100
07/16	7797	매수	090430	아모레퍼시픽	18	134,100
07/16	7795	매도	014830	유니드	7	88,200
07/16	7637	매도	034950	한국기업평가	72	100,457
07/16	7635	매수	090430	아모레퍼시픽	6	134,000
07/16	7634	매수	014830	유니드	30	88,240
07/16	7556	매수	034950	한국기업평가	34	100,100
07/16	7555	매도	090430	아모레퍼시픽	17	134,106
07/16	7554	매도	014830	유니드	15	88,300
07/16	7415	매수	034950	한국기업평가	21	100,114
07/16	7413	매도	014830	유니드	3	88,300
07/16	7266	매도	034950	한국기업평가	59	100,337
07/16	7264	매수	090430	아모레퍼시픽	19	134,100
07/16	7263	매도	014830	유니드	27	88,300
07/16	7162	매도	090430	아모레퍼시픽	5	134,890
07/16	7161	매수	014830	유니드	36	88,200
07/16	7100	매수	014830	유니드	1	88,300
07/16	7043	매수	034950	한국기업평가	36	100,356                                                                                                            
07/16	7042	매도	090430	아모레퍼시픽	2	135,250
07/16	7041	매도	014830	유니드	7	88,200
07/16	6973	매수	034950	한국기업평가	146	100,199
07/16	6971	매수	090430	아모레퍼시픽	41	135,500
07/16	6970	매수	014830	유니드	48	88,473
""".strip()

# 1️⃣ 데이터프레임 변환 (동일)
df = pd.read_csv(io.StringIO(log_text), sep='\t')
df.columns = [c.strip() for c in df.columns]

df['날짜'] = '2025-' + df['매매일자'].str.replace('/', '-')
df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')

df['수량'] = df['수량'].astype(str).str.replace(',', '').astype(float)
df['체결가격'] = df['체결가격'].astype(str).str.replace(',', '').astype(float)
df['매매금액'] = df['수량'] * df['체결가격']
df['잔고변화'] = df.apply(lambda x: -x['매매금액'] if x['매매구분'] == '매수' else x['매매금액'], axis=1)

df = df.sort_values('날짜')

# 2️⃣ 종목별 누적 수익 계산
cum_profit_by_stock = df.groupby(['날짜', '종목명'])['잔고변화'].sum().groupby('종목명').cumsum().reset_index()
cum_profit_by_stock['잔고변화'] = cum_profit_by_stock['잔고변화'].astype(float)

# 3️⃣ 폰트 및 Y축 표기 설정 (한글/지수 표기 해제)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 4️⃣ 종목별 누적 수익금 그래프 (실제 금액 표기)
plt.figure(figsize=(14, 7))
for stock in df['종목명'].unique():
    stock_data = cum_profit_by_stock[cum_profit_by_stock['종목명'] == stock]
    plt.plot(stock_data['날짜'], stock_data['잔고변화'], label=stock, marker='o')

plt.xlabel('일자')
plt.ylabel('종목별 누적 수익금 (원)')
plt.title('종목별 누적 수익금 그래프')
plt.legend()
plt.grid()
plt.ticklabel_format(style='plain', axis='y')   # <- 실제 금액(지수표기 해제)
plt.tight_layout()
plt.show()
