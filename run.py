from backtest.engine import BacktestEngine
from backtest.strategies.orb import ORBStrategy

engine = BacktestEngine(
    data_path      = "MNQ_5min.csv",
    instrument     = "MNQ",
    strategy       = ORBStrategy(params={
        "orb_minutes":   30,
        "rr":            1.5,
        "session_open":  "09:30",
        "session_close": "11:30",
        "direction":     "both",
        "retest":        False,
    }),
    contracts      = 1,
    start          = "2016-02-18",
    end            = "2026-02-18",
    output_dir     = "results/",
    initial_capital= 10_000.0,
)

results = engine.run()
