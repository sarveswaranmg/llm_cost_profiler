"""Web server.

FastAPI server exposing traces over REST and WebSocket for the React
dashboard. Requires the `server` extra: pip install "tokenlens[server]".

Import create_app from tokenlens.server.app (kept out of this namespace so
`import tokenlens.server` never hard-fails when FastAPI isn't installed).
"""
