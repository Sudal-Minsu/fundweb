SET NAMES utf8mb4;
SET CHARACTER SET utf8mb4;

DROP TABLE IF EXISTS portfolio_weights;
CREATE TABLE portfolio_weights (
  id INT AUTO_INCREMENT PRIMARY KEY,
  ticker VARCHAR(20),
  weight FLOAT
);

DROP TABLE IF EXISTS portfolio_returns;
CREATE TABLE portfolio_returns (
  date DATE,
  cumulative_return FLOAT
);

DROP TABLE IF EXISTS portfolio_monthly_returns;
CREATE TABLE portfolio_monthly_returns (
  month VARCHAR(10),
  return_pct FLOAT
);

-- 기본 테스트 데이터
INSERT INTO portfolio_weights (ticker, weight)
VALUES ('005930', 50), ('000660', 50);

INSERT INTO portfolio_returns (date, cumulative_return)
VALUES ('2024-01-01', 0), ('2024-02-01', 3.2), ('2024-03-01', 6.1);

INSERT INTO portfolio_monthly_returns (month, return_pct)
VALUES ('1월', 1.2), ('2월', -0.3), ('3월', 2.0);