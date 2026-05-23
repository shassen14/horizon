# Multi-stage build for the Rust core. Produces both workspace binaries in one image;
# compose runs each via a different `command`. Build context is the repo root.
FROM rust:1.95-slim AS builder
WORKDIR /build
COPY . .
RUN cargo build --release --bin horizon-api --bin horizon-scheduler

FROM debian:bookworm-slim AS runtime
WORKDIR /app
# ca-certificates: TLS to Alpaca/Cloudflare via reqwest(rustls).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -r -u 10001 horizon
COPY --from=builder /build/target/release/horizon-api /usr/local/bin/horizon-api
COPY --from=builder /build/target/release/horizon-scheduler /usr/local/bin/horizon-scheduler
COPY config ./config
USER horizon
EXPOSE 8080
