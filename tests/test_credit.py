import pandas as pd

from paid_media.credit import (
    CLICK_WEIGHT,
    VIEW_WEIGHT,
    apply_credit_weights,
    credit_weight,
    total_channel_credit,
)


def test_credit_weight_basic():
    assert credit_weight("click") == CLICK_WEIGHT
    assert credit_weight("view") == VIEW_WEIGHT
    assert credit_weight("impression") == VIEW_WEIGHT
    assert credit_weight("direct") == 1.0
    assert credit_weight("CLICK") == CLICK_WEIGHT  # case-insensitive


def test_apply_credit_weights_preserves_shape():
    paths = [
        [("sem", "click"), ("display", "view")],
        [("brand", "click")],
    ]
    out = apply_credit_weights(paths)
    assert len(out) == 2
    assert [len(p) for p in out] == [2, 1]
    assert out[0][0] == ("sem", CLICK_WEIGHT)
    assert out[0][1] == ("display", VIEW_WEIGHT)


def test_total_channel_credit_last_touch_credits_last():
    paths = [[("sem", "click"), ("brand", "click")]]
    weighted = apply_credit_weights(paths)
    credit = total_channel_credit(weighted, rule="last")
    assert credit["brand"] == 1.0 and credit.get("sem", 0.0) == 0.0


def test_total_channel_credit_linear_splits_evenly():
    paths = [[("sem", "click"), ("brand", "click")]]
    weighted = apply_credit_weights(paths)
    credit = total_channel_credit(weighted, rule="linear")
    assert abs(credit["sem"] - 0.5) < 1e-9
    assert abs(credit["brand"] - 0.5) < 1e-9


def test_view_haircut_reduces_display_share_on_mixed_path():
    # A path that is 2 clicks + 1 view should give less credit to the view touch.
    paths = [[("sem", "click"), ("display", "view"), ("brand", "click")]]
    weighted = apply_credit_weights(paths)
    credit = total_channel_credit(weighted, rule="linear")
    assert credit["display"] < credit["sem"]
    assert credit["display"] < credit["brand"]


def test_total_channel_credit_returns_series():
    weighted = apply_credit_weights([[("sem", "click")]])
    out = total_channel_credit(weighted, rule="last")
    assert isinstance(out, pd.Series)
