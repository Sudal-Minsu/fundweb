import matplotlib.pyplot as plt

# ──────────────── Step 1: Loop별 보유수량 데이터 입력 ────────────────

loops = list(range(1, 14))  # Loop 1 ~ Loop 13

# 각 종목별 보유수량 기록
qty_014830 = [48, 41, 42, 78, 51, 48, 33, 63, 47, 47, 69, 33, 46]
qty_090430 = [41, 39, 39, 34, 53, 39, 36, 42, 60, 43, 49, 33, 37]
qty_034950 = [146, 182, 137, 117, 123, 144, 178, 106, 95, 149, 89, 169, 149]

# ──────────────── Step 2: 그래프 그리기 ────────────────

plt.figure(figsize=(12, 6))
plt.plot(loops, qty_014830, marker='o', label='014830')
plt.plot(loops, qty_090430, marker='s', label='090430')
plt.plot(loops, qty_034950, marker='^', label='034950')

plt.title('종목별 Loop별 보유수량 변화', fontsize=16)
plt.xlabel('Loop 번호', fontsize=14)
plt.ylabel('보유 수량', fontsize=14)
plt.xticks(loops)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
