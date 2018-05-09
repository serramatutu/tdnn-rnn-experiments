import quandl

quandl.ApiConfig.api_key = "YOUR API KEY HERE"

data = quandl.get("EIA/PET_RWTC_D")

print(data)