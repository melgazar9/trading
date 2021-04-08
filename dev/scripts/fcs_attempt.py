# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 18:20:13 2020

@author: Matt
"""

fsapi_key = '1Hh1dE9IyDI4Q0YufLEI3974'
from fcs_api_py import Forex
import requests

forex = Forex(fsapi_key)
obj = forex.technical_indicator("EUR/JPY", "1d")


print(obj.response.oa_summary)
print(obj.response.indicators.ATR14.s)
print(obj.response.indicators.ATR14.v)
print(obj.response.indicators.WilliamsR.s)
print(obj.response.indicators.WilliamsR.v)


requests.get('https://fcsapi.com/api-v3/stock/indices?country=indonesia&access_key=' + fsapi_key)
