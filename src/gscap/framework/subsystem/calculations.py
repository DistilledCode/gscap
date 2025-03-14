from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from gscap.framework.subsystem import SubSystem


def prs_with_cost(
    ss: SubSystem,
    capital_series: pd.Series = None,
    fx_series: pd.Series = None,
) -> pd.Series:
    """
    Calculating SS positions
    KE, CME, 1d, ins, 8:	first time
    ZC, CME, 1d, ins, 8:	first time
    ZL, CME, 1d, ins, 8:	first time
    Calculating initial SS returns
    ! len(position_series.index)=6540; len(adjusted_price_series.index)=6540
    ! len(position_series.index)=6563; len(adjusted_price_series.index)=6563
    ! len(position_series.index)=6572; len(adjusted_price_series.index)=6572
    Calculating Instrument Weights
    Calculating IDM
    2.0926734000677243
    Recalculating SS positions
    KE, CME, 1d, ins, 8:	scaling position with IDM
    ZC, CME, 1d, ins, 8:	scaling position with IDM
    ZL, CME, 1d, ins, 8:	scaling position with IDM
    Recalculating SS returns
    ! len(position_series.index)=6681; len(adjusted_price_series.index)=6540
    ! len(position_series.index)=6681; len(adjusted_price_series.index)=6563
    ! len(position_series.index)=6681; len(adjusted_price_series.index)=6572
    """
    instrument = ss.instrument
    positions = ss.position.round()
    adjusted_price_series = instrument.close_price().adjusted.squeeze()
    price_delta = adjusted_price_series.diff()
    return_price_points = price_delta * positions.shift(1)
    lots_traded = positions.diff().abs()

    slippage_cost_currency = (
        lots_traded
        * instrument.meta.currency_tick_value
        * instrument.meta.cost_in_ticks
    )
    commission_cost_currency = lots_traded * instrument.meta.commission_cost
    total_cost_currency = slippage_cost_currency + commission_cost_currency

    rtrn_instrmnt_currency = return_price_points * instrument.meta.dollar_equivalent

    if not isinstance(type(capital_series), pd.Series):
        capital_series = 100_000 if capital_series is None else capital_series
        capital_series = pd.Series(capital_series, index=positions.index)
    if fx_series is None:
        fx_series_aligned = pd.Series(1.0, index=positions.index)
    else:
        fx_series_aligned = fx_series.reindex(positions.index)
    fx_series_aligned.ffill(inplace=True)

    return_base_currency = rtrn_instrmnt_currency * fx_series_aligned
    tcost_base_currency = total_cost_currency * fx_series_aligned

    gross_perc_return = return_base_currency / capital_series
    cost_in_perc = tcost_base_currency / capital_series

    net_perc_return = gross_perc_return - cost_in_perc

    # _annual_rolling_price_vol = adjusted_price_series.diff().ewm(
    #     span=gscap.DAYS_IN_MONTH
    # ).std() * np.sqrt(gscap.DAYS_IN_YEAR)

    ss.cost.slippage_currency = slippage_cost_currency * fx_series_aligned
    ss.cost.commission_currency = commission_cost_currency * fx_series_aligned
    ss.cost.total_currency = tcost_base_currency
    ss.cost.return_series = cost_in_perc
    ss.cost.return_series.name = ss.instrument.meta.symbol.lower()

    # ss.cost.risk_adj_per_lot = (tcost_base_currency / lots_traded) / (
    #     _annual_rolling_price_vol * instrument.meta.dollar_equivalent
    # )
    return net_perc_return
