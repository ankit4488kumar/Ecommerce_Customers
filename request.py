import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Avg. Session Length':8, 'Time on App':8, 'Time on Website':8,'Length of Membership':8})

print(r.json())