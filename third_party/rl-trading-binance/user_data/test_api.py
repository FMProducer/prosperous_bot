"""Test auth v2 - try Basic Auth and different endpoints"""
import urllib.request, urllib.error, base64, json

BASE = 'http://127.0.0.1:8080'
USERNAME = 'freqtrader'
PASSWORD = 'Pr0ducti0n'

def test(name, url, headers=None, method='GET', data=None):
    print(f'\n=== {name} ===')
    try:
        req = urllib.request.Request(url, headers=headers or {}, method=method, data=data)
        resp = urllib.request.urlopen(req, timeout=5)
        body = resp.read().decode()[:400]
        print(f'  OK ({resp.status}): {body}')
        return json.loads(body) if body.startswith('{') or body.startswith('[') else body
    except urllib.error.HTTPError as e:
        print(f'  FAIL ({e.code}): {e.read().decode()[:200]}')
    except Exception as e:
        print(f'  ERROR: {e}')

# Basic Auth header
basic = base64.b64encode(f'{USERNAME}:{PASSWORD}'.encode()).decode()
basic_headers = {'Authorization': f'Basic {basic}'}

# 1. Try Basic Auth on /status
test('Basic Auth - /status', f'{BASE}/api/v1/status', basic_headers)

# 2. Try Basic Auth on /profit
test('Basic Auth - /profit', f'{BASE}/api/v1/profit', basic_headers)

# 3. Try Basic Auth on /show_config
test('Basic Auth - /show_config', f'{BASE}/api/v1/show_config', basic_headers)

# 4. Try login WITH Basic Auth
test('Login+Basic', f'{BASE}/api/v1/token/login', 
     headers={**basic_headers, 'Content-Type': 'application/x-www-form-urlencoded'},
     data=b'username=freqtrader&password=Pr0ducti0n', method='POST')

# 5. Try /api/v1/version
test('Version (no auth)', f'{BASE}/api/v1/version')

# 6. List all routes via docs
test('Docs', f'{BASE}/docs')
test('Health', f'{BASE}/api/v1/health')

# 7. Try /api/v1/logs with Basic Auth
test('Logs+Basic', f'{BASE}/api/v1/logs?limit=5', basic_headers)

print('\nDone!')
