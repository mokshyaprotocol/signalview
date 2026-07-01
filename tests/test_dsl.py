import numpy as np
import pandas as pd
from perpsignal.dsl import evaluate

def test_roc_indicator():
    # Create a simple mock dataframe with closing prices
    df = pd.DataFrame(
        {"close": [100.0, 105.0, 110.0, 121.0]}, 
        index=pd.date_range("2026-01-01", periods=4, freq="D")
    )
    
    # Test a 2-period Rate of Change calculation: (110 / 100) - 1 = 0.10
    result = evaluate("roc(close, 2)", df, clip=False)
    assert np.isclose(result.iloc[2], 0.10)
    
    # Test a 1-period Rate of Change calculation: (121 / 110) - 1 = 0.10
    result_1 = evaluate("roc(close, 1)", df, clip=False)
    assert np.isclose(result_1.iloc[3], 0.10)