"""Serve Stories Studio locally. Run: python start_website.py"""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    website = Path(__file__).resolve().parent / 'interface'
    handler = partial(SimpleHTTPRequestHandler, directory=str(website))
    try:
        server = ThreadingHTTPServer(('127.0.0.1', args.port), handler)
    except OSError as error:
        parser.exit(1, f'Cannot start on port {args.port}: {error}\n'
                    'If the website is already running, use its URL. Otherwise try --port 8001.\n')
    print(f'Stories Studio: http://127.0.0.1:{args.port}/', flush=True)
    print(f'Website files: {website}\nPress Ctrl+C to stop.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
