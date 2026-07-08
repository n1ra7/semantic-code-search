# Multi-stage build. python:slim (Debian) is used rather than Alpine because the
# embedding runtime (onnxruntime, via fastembed) ships manylinux wheels that don't
# install cleanly on Alpine's musl libc. The final image stays small by copying only
# the installed environment and source.
FROM python:3.12-slim AS build
ENV PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY --from=build /install /usr/local
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps -e .

# Default: index the mounted workspace. Override the command to run the MCP server
# (`semantic-index-mcp`) or a search.
ENTRYPOINT ["semantic-index"]
CMD ["index", "/workspace"]
