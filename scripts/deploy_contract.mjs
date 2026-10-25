import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import solc from 'solc';
import { JsonRpcProvider, Wallet, ContractFactory } from 'ethers';

const rpcUrl = process.env.AUTHENFY_RPC_URL;
const privateKey = process.env.AUTHENFY_DEPLOYER_PRIVATE_KEY;
if (!rpcUrl || !privateKey) {
  throw new Error('Set AUTHENFY_RPC_URL and AUTHENFY_DEPLOYER_PRIVATE_KEY in the deployment shell.');
}
assert(/^https?:\/\//i.test(rpcUrl), 'AUTHENFY_RPC_URL must use HTTP or HTTPS');
const source = await fs.readFile('contracts/AnalysisRegistry.sol', 'utf8');
const compiled = JSON.parse(solc.compile(JSON.stringify({
  language: 'Solidity',
  sources: { 'AnalysisRegistry.sol': { content: source } },
  settings: { evmVersion: 'shanghai', outputSelection: { '*': { '*': ['abi', 'evm.bytecode.object'] } } },
})));
const errors = (compiled.errors || []).filter((error) => error.severity === 'error');
assert.deepEqual(errors, [], errors.map((error) => error.formattedMessage).join('\n'));
const artifact = compiled.contracts['AnalysisRegistry.sol'].AnalysisRegistry;
const provider = new JsonRpcProvider(rpcUrl);
const signer = new Wallet(privateKey, provider);
const network = await provider.getNetwork();
console.log(JSON.stringify({ status: 'deploying', chain_id: Number(network.chainId), publisher: await signer.getAddress() }));
const contract = await new ContractFactory(artifact.abi, artifact.evm.bytecode.object, signer).deploy();
await contract.waitForDeployment();
const deploymentTx = contract.deploymentTransaction();
await deploymentTx.wait(1);
console.log(JSON.stringify({
  status: 'deployed',
  chain_id: Number(network.chainId),
  registry_address: await contract.getAddress(),
  publisher: await contract.publisher(),
  deployment_tx: deploymentTx.hash,
  compiler: solc.version(),
}));
await provider.destroy();
