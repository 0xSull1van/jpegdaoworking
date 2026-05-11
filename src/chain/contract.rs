use alloy::sol;

sol! {
    #[sol(rpc)]
    contract Hash {
        // --- write ---
        function mine(uint256 nonce) external;

        // --- read ---
        function currentDifficulty() external view returns (uint256);
        function getChallenge(address miner) external view returns (bytes32);
        function epochBlocksLeft() external view returns (uint256);
        function currentReward() external view returns (uint256);
        function totalMints() external view returns (uint256);
        function mintsInBlock(uint256 blockNumber) external view returns (uint256);
        function genesisComplete() external view returns (bool);
        function miningState() external view returns (
            uint256 era,
            uint256 reward,
            uint256 difficulty,
            uint256 minted,
            uint256 remaining,
            uint256 epoch,
            uint256 epochBlocksLeft_
        );

        // --- events ---
        event Mined(address indexed miner, uint256 nonce, uint256 reward, uint256 era);
        event Halving(uint256 era, uint256 reward);
        event DifficultyAdjusted(uint256 old, uint256 next, uint256 takenBlocks);

        // --- reverts ---
        error InsufficientWork();
        error ProofAlreadyUsed();
        error BlockCapReached();
        error SupplyExhausted();
        error GenesisNotComplete();
    }
}

/// hash256 mainnet deployment.
pub const CONTRACT: alloy::primitives::Address =
    alloy::primitives::address!("AC7b5d06fa1e77D08aea40d46cB7C5923A87A0cc");
pub const CHAIN_ID: u64 = 1;
pub const EPOCH_BLOCKS: u64 = 100;
pub const MAX_MINTS_PER_BLOCK: u64 = 10;
