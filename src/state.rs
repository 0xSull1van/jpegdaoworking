use std::sync::atomic::{AtomicU8, Ordering};

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MinerState {
    Healthy        = 0,
    RpcDegraded    = 1,
    ChallengeStale = 2,
    GpuFault       = 3,
    WalletLocked   = 4,
    Paused         = 5,
    Fatal          = 6,
}

#[derive(Debug)]
pub struct StateMachine {
    cur: AtomicU8,
}

impl Default for StateMachine {
    fn default() -> Self { Self::new() }
}

impl StateMachine {
    pub fn new() -> Self { Self { cur: AtomicU8::new(MinerState::Healthy as u8) } }
    pub fn get(&self) -> MinerState {
        match self.cur.load(Ordering::Acquire) {
            0 => MinerState::Healthy, 1 => MinerState::RpcDegraded,
            2 => MinerState::ChallengeStale, 3 => MinerState::GpuFault,
            4 => MinerState::WalletLocked, 5 => MinerState::Paused,
            _ => MinerState::Fatal,
        }
    }
    pub fn set(&self, s: MinerState) -> MinerState {
        let prev = self.cur.swap(s as u8, Ordering::AcqRel);
        // SAFETY: only ever store valid discriminants
        #[allow(clippy::undocumented_unsafe_blocks)]
        unsafe { std::mem::transmute(prev) }
    }
    pub fn is_grinding_ok(&self) -> bool {
        matches!(self.get(), MinerState::Healthy | MinerState::RpcDegraded | MinerState::ChallengeStale)
    }
    pub fn is_submitting_ok(&self) -> bool {
        matches!(self.get(), MinerState::Healthy | MinerState::RpcDegraded)
    }
}
