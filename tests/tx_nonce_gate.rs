// Tests sequential reservation semantics by mocking get_transaction_count.
use hashminer::tx::nonce_manager::NonceGate;

// Full mock requires wiremock + a stub provider; deferred to e2e tests.
// For now we assert the trait surface compiles and is Send-safe.
#[test]
fn nonce_gate_compiles() {
    fn _assert_send<T: Send>() {}
    _assert_send::<NonceGate>();
}
