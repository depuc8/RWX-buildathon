# The scorer builds this on linux/amd64 and runs it with no network. The
# digest is the image every commit was labeled in.
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

# sqlglot at the pinned commit: the selector reads the old source of every
# file a patch touches, and runs parts of sqlglot on the patched code.
ADD https://github.com/tobymao/sqlglot/archive/eb11aec4be8cb2122afe713a4816662c3f0fb2fa.tar.gz /tmp/sqlglot.tar.gz
RUN mkdir -p /app/src \
    && tar xzf /tmp/sqlglot.tar.gz -C /app/src --strip-components=1 \
    && rm /tmp/sqlglot.tar.gz \
    && test -f /app/src/sqlglot/parser.py \
    && python -m compileall -q /app/src/sqlglot /app/src/tests/helpers.py

COPY selector.py verify.py tpcds_check.py /app/
COPY maps /app/maps
RUN chmod -R a+rX /app && python -m compileall -q /app/selector.py /app/verify.py /app/tpcds_check.py

# What importing the pinned commit gives (failures, and the values importing
# leaves behind), for the import checks to compare patched code with.
RUN cd /tmp && HOME=/tmp python -I /app/selector.py --import-baseline /app/maps/import_base.json \
    && chmod a+r /app/maps/import_base.json

CMD ["python", "-I", "/app/selector.py"]
