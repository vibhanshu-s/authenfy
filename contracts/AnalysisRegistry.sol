// SPDX-License-Identifier: MIT
pragma solidity ^0.8.37;

/// @notice Records analysis commitments, never factual authenticity.
/// @dev A receipt belongs to this registry's immutable publisher. The same file
/// may have multiple immutable receipts for different models or analyses.
contract AnalysisRegistry {
    address public immutable publisher;
    mapping(bytes32 => mapping(bytes32 => uint64)) public recordedAt;

    error Unauthorized();
    error EmptyCommitment();
    error AlreadyRecorded();

    event AnalysisRecorded(bytes32 indexed fileHash, bytes32 indexed receiptCommitment, uint64 timestamp);

    constructor() {
        publisher = msg.sender;
    }

    function record(bytes32 fileHash, bytes32 receiptCommitment) external {
        if (msg.sender != publisher) revert Unauthorized();
        if (fileHash == bytes32(0) || receiptCommitment == bytes32(0)) revert EmptyCommitment();
        if (recordedAt[fileHash][receiptCommitment] != 0) revert AlreadyRecorded();
        uint64 timestamp = uint64(block.timestamp);
        recordedAt[fileHash][receiptCommitment] = timestamp;
        emit AnalysisRecorded(fileHash, receiptCommitment, timestamp);
    }
}
