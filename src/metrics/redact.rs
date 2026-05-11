//! Placeholder for a tracing Layer that redacts hex-encoded 32-byte sequences
//! matching the configured private key bytes. Belt-and-suspenders — we never
//! knowingly log the key, but this layer catches accidental logs.
//!
//! Implementer note: leave as no-op stub for v1. Add a real `tracing_subscriber::Layer`
//! impl when post-MVP hardening lands.

pub struct RedactionLayer;
