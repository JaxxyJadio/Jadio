# TROUBLESHOOTING: Jadio-quick-llm

## Common Issues

### Model Fails to Load
- Check `config.yaml` for correct `model_path` and `torch_dtype`.
- Ensure all model files are present in the directory.
- Check GPU/CPU compatibility and available memory.
- Review logs for detailed error messages.

### API Returns 503 or 500 Errors
- Model may not be loaded or crashed. Restart the server.
- Check logs for stack traces and error details.
- Validate your `config.yaml` with the provided schema.

### CORS or Browser Errors
- Ensure `cors_origins` in `config.yaml` allows your client IP/domain.
- Try setting to `["*"]` for LAN testing.

### Authentication Fails
- If `auth_password` is set, ensure you send the correct `Authorization: Bearer ...` header.
- Remove or update the password in `config.yaml` for open LAN use.

### Agent Tools Not Found or Fail
- Place all tool scripts in the `tools/` directory.
- Use `/api/tools` to list available tools.
- Use `/api/reload-tools` to reload after adding new scripts.

## Debugging Tips
- Set `loglevel: DEBUG` in `config.yaml` for verbose logs.
- Use the `/api/status` endpoint to check model and server state.
- Check `sessions.json` for persisted chat history.

## Still Stuck?
- Review the README and inline code docs.
- File an issue or ask for help with logs and config details.
