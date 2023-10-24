##############################################################################
# Copyright (C) 2018, 2019, 2020 Dominic O'Kane
##############################################################################


import numpy as np
from enum import Enum


from ...utils.global_vars import gDaysInYear
from ...utils.error import FinError
from ...utils.global_types import DoubleTouchOptionTypes
from ...utils.helpers import label_to_string, check_argument_types
from ...utils.date import Date
from ...market.curves.discount_curve import DiscountCurve
from ...models.gbm_process_simulator import get_paths
from ...products.fx.fx_option import FXOption
from ...models.black_scholes import BlackScholes

from numba import njit

from ...utils.math import n_vect

###############################################################################

@njit()
def _sgn(i):
    if i%2>0:
        return -1.
    else:
        return 1.

@njit(error_model="numpy", fastmath=True, cache=True)
def _sum_knock_in_series(payment, fx_rate, U, L, v2, T, rd, rf, n):
    """ PLACEHOLDER """
    Z = np.log(U/L)
    alpha = - 0.5 * (2 * (rd - rf) / v2 - 1)
    beta = - 0.25 * np.power((2 * (rd - rf) / v2 - 1), 2) - 2 * rd / v2
    sum = 0
    for i in range(n):
        nth_summand = (2*np.pi*(i+1)*payment / np.power(Z, 2) 
                       * ((np.power(fx_rate / L, alpha) - _sgn(i) 
                          * np.power(fx_rate / U, alpha)) 
                          / (np.power(alpha, 2) + np.power(i*np.pi / 2)))
                        * np.sin(i*np.pi / Z * np.log(fx_rate / L)) 
                        * np.exp(-0.5*(np.power(i*np.pi / Z, 2) - beta) * v2 * T))
        sum += nth_summand

    return sum

###############################################################################

class FXDoubleOneTouchOption(FXOption):
    """ TAMAS to provide description """

    def __init__(self,
                 expiry_date: Date,
                 option_type: (DoubleTouchOptionTypes, list),
                 upper_barrier: (float, np.ndarray),
                 lower_barrier: (float, np.ndarray),
                 currency_pair: str,  # FORDOM
                 notional: float,
                 notional_currency: str):
        """ Create the double one touch option by defining its 
        expiry date, the barrier levels, the option type (knock-in
        or knock-out), the notional and the notional currency. """

        check_argument_types(self.__init__, locals())

        self._expiry_date = expiry_date

        self._option_type = option_type

        if np.any(upper_barrier <= lower_barrier):
            raise FinError("Upper barrier must be greater than lower barrier.")

        self._upper_barrier = upper_barrier
        self._lower_barrier = lower_barrier

        if len(currency_pair) != 6:
            raise FinError("Currency pair must be 6 characters.")

        self._currency_pair = currency_pair
        self._forName = self._currency_pair[0:3]
        self._domName = self._currency_pair[3:6]

        if notional_currency != self._domName and notional_currency != self._forName:
            raise FinError("Notional currency not in currency pair.")
        
        self._notional_currency = notional_currency

        self._notional = notional

###############################################################################

    def value(self,
              valuation_date: Date,
              spot_fx_rate: (float, np.ndarray),  # 1 unit of foreign in domestic
              dom_discount_curve: DiscountCurve,
              for_discount_curve: DiscountCurve,
              model,
              n: int):
        
        if isinstance(valuation_date, Date) is False:
            raise FinError("Valuation date is not a Date")

        if valuation_date > self._expiry_date:
            raise FinError("Valuation date after expiry date.")
        
        if dom_discount_curve._valuation_date != valuation_date:
            raise FinError(
                "Domestic Curve valuation date not same as valuation date")

        if for_discount_curve._valuation_date != valuation_date:
            raise FinError(
                "Foreign Curve valuation date not same as valuation date")
        
        T = (self._expiry_date - valuation_date) / gDaysInYear

        if np.any(spot_fx_rate <= 0.0):
            raise FinError("spot_fx_rate must be greater than zero.")
        
        if np.any(T < 0.0):
            raise FinError("Option time to maturity is less than zero.")
        
        T = np.maximum(T, 1e-10)

        domDF = dom_discount_curve._df(T)
        forDF = for_discount_curve._df(T)

        rd = -np.log(domDF) / T
        rf = -np.log(forDF) / T

        if type(model) != BlackScholes:
            raise FinError("Only the Black-Scholes model is supported for now.")

        U = self._upper_barrier
        L = self._lower_barrier
        sigma = model._volatility
        v2 = sigma * sigma

        if self._notional_currency == self._domName:
            payment = self._notional
        else:
            payment = self._notional * spot_fx_rate

        if self._option_type == DoubleTouchOptionTypes.KNOCK_OUT:
            v = _sum_knock_in_series(payment, spot_fx_rate, U, L, v2, T, rd, rf, n)

        elif self._option_type == DoubleTouchOptionTypes.KNOCK_IN:
            # long knock-out + long knock-in == sure payment at T
            # so long knock-in at t = short knock-out at t + dc*payment
            v_knock_out = _sum_knock_in_series(payment, spot_fx_rate, U, L, v2, T, rd, rf, n)
            v = payment*np.exp(rf - rd) - v_knock_out

        elif self._option_type == DoubleTouchOptionTypes.ASYMMETRICAL:
            print('PLACEHOLDER')


        


        
