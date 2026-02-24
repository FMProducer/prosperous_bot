"""
Dashboard Proxy Server
Serves dashboard.html and proxies API requests to Freqtrade (127.0.0.1:8080).
Eliminates CORS issues.

Usage: python dashboard_server.py
Then open: http://localhost:8888
"""
import http.server
import urllib.request
import json
import os
import sys

FREQTRADE_API = 'http://127.0.0.1:8080'
PORT = 8888
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))


class ProxyHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DASHBOARD_DIR, **kwargs)

    def do_GET(self):
        if self.path == '/api/rl_stats':
            self._serve_rl_stats()
        elif self.path.startswith('/api/'):
            self._proxy('GET')
        else:
            # Serve static files (dashboard.html)
            if self.path == '/':
                self.path = '/dashboard.html'
            super().do_GET()

    def do_POST(self):
        if self.path.startswith('/api/'):
            self._proxy('POST')
        else:
            self.send_error(404)

    def _serve_rl_stats(self):
        file_path = os.path.join(DASHBOARD_DIR, 'rl_stats.json')
        try:
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    content = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Content-Length', str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_error(404, "Stats file not found")
        except Exception as e:
            self.send_error(500, str(e))

    def _proxy(self, method):
        target_url = FREQTRADE_API + self.path
        try:
            # Read request body for POST
            body = None
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > 0:
                body = self.rfile.read(content_length)

            # Build proxy request
            req = urllib.request.Request(target_url, data=body, method=method)

            # Forward relevant headers
            for header in ['Authorization', 'Content-Type']:
                val = self.headers.get(header)
                if val:
                    req.add_header(header, val)

            # Execute request
            with urllib.request.urlopen(req, timeout=15) as resp:
                resp_body = resp.read()
                self.send_response(resp.status)
                self.send_header('Content-Type', resp.getheader('Content-Type', 'application/json'))
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Content-Length', str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)

        except urllib.error.HTTPError as e:
            body = e.read()
            self.send_response(e.code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            err = json.dumps({'error': str(e)}).encode()
            self.send_response(502)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(err)

    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Authorization, Content-Type')
        self.end_headers()

    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    def log_message(self, format, *args):
        # Quiet logging - only show API proxied calls
        msg = format % args
        if '/api/' in msg:
            sys.stderr.write(f"[proxy] {msg}\n")


if __name__ == '__main__':
    print(f"🚀 Dashboard server starting on http://localhost:{PORT}")
    print(f"   Proxying API to {FREQTRADE_API}")
    print(f"   Press Ctrl+C to stop\n")
    server = http.server.HTTPServer(('127.0.0.1', PORT), ProxyHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n⏹ Server stopped.")
        server.server_close()
