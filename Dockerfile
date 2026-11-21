# The scorer builds this on linux/amd64 and runs it with no network. The
# digest is the image every commit was labeled in.
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

# sqlglot at the pinned commit: the selector reads the old source of every
# file a patch touches.
ADD https://github.com/tobymao/sqlglot/archive/eb11aec4be8cb2122afe713a4816662c3f0fb2fa.tar.gz /tmp/sqlglot.tar.gz
RUN mkdir -p /app/src \
    && tar xzf /tmp/sqlglot.tar.gz -C /app/src --strip-components=1 \
    && rm /tmp/sqlglot.tar.gz \
    && test -f /app/src/sqlglot/parser.py

COPY selector.py verify.py /app/
COPY maps /app/maps
RUN chmod -R a+rX /app && python -m compileall -q /app/selector.py /app/verify.py

CMD ["python", "-I", "/app/selector.py"]
